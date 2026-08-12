"""molmo_flow ↔ BijouModel integration oracles (§8.13 step 5, CPU tier).

Runs the REAL composition end-to-end on the tiny converted checkpoint
(the converter-test fixture, backbone ref = the source dir, no hub
access): from_checkpoint assembly → MolmoAct2 collation → prefix encode
(cache + conditioning mask) → loss / predict. Pinned:

- the KI gradient contract both ways (§8.13 decision 8): insulated ⇒
  flow-loss gradients into EVERY trunk parameter exactly zero (while
  the expert's trainable set all receive grads); uninsulated ⇒ nonzero
  trunk gradients THROUGH the cached K/V — and the open seam composes
  with activation checkpointing (the cache-shim machinery);
- BijouModel.predict_chunk dispatch: molmo_flow defaults (recorded
  steps, Euler), their output tail (n_action_steps × action_dim raw
  units), noise determinism, target_time refused;
- loss arms: mean == sum/count across loss_components /
  loss_component_sums / loss_count_normalizers (the chunked-backward
  contract), and per-sample batch stats are deliberately unused
  (decision 6: perturbing them changes nothing).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_convert_molmoact2 import _ACTION_DIM, _HORIZON, _convert, source_dir

from bijou.decoders.molmo_flow import MolmoFlowDecoder, molmo_flow_loss
from bijou.interface import (
    CameraFrame,
    CollatedBatch,
    NormStats,
    PromptInputs,
    SamplingMethod,
)
from bijou.loading import from_checkpoint
from bijou.model import BijouModel

assert source_dir is not None  # re-exported pytest fixture (module-scoped)


@pytest.fixture(scope="module")
def tiny_model(
    source_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> BijouModel:
    out = _convert(
        source_dir,
        tmp_path_factory.mktemp("molmo-flow-int") / "converted",
        backbone_ref=str(source_dir),
    )
    model, _info = from_checkpoint(out)
    # The fixture expert is zero-init (adaLN-Zero: the field is exactly
    # 0 and upstream gradients are Wᵀδ = 0 — the gradient tests would
    # pass vacuously insulated and fail open). Perturb it into a
    # non-degenerate field, deterministically (the port tests' pattern).
    decoder = model.decoder
    assert isinstance(decoder, MolmoFlowDecoder)
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for block in decoder.iter_blocks():
            block.modulation.linear.bias.fill_(0.1)
        for parameter in (
            decoder.final_layer.linear.weight,
            decoder.final_layer.linear.bias,
            decoder.final_layer.modulation.linear.bias,
        ):
            parameter.add_(
                0.05 * torch.randn(parameter.shape, generator=generator),
            )
    return model


def _stub_collator(model: BijouModel):  # noqa: ANN202 — encoder-specific collator
    from test_molmoact2_encoder import _StubTokenizer

    from bijou.molmoact2.processing import BOS_ID, IM_END_ID, IM_PATCH_ID, IM_START_ID

    collator = model.encoder.inputs_collator()
    collator._tokenizer = _StubTokenizer()  # the tiny source has no real tokenizer
    collator._image_ids = (IM_START_ID, IM_END_ID, IM_PATCH_ID)
    collator._patch_id = IM_PATCH_ID
    collator._eos_id = BOS_ID
    collator._action_start_id = 151_932
    collator._action_end_id = 151_933
    return collator


def _batch(model: BijouModel, batch_size: int = 2) -> CollatedBatch:
    generator = torch.Generator().manual_seed(0)
    samples = []
    for index in range(batch_size):
        cameras = tuple(
            CameraFrame(
                name=f"cam{j}",
                kind="unknown",
                image=torch.rand((3, 48, 64), generator=generator),
            )
            for j in range(1 + index % 2)
        )
        samples.append(
            PromptInputs(
                instruction=f"Pick up cube {index}.",
                cameras=cameras,
                condition_text="",
                state=torch.rand((6,), generator=generator) * 2 - 1,
            ),
        )
    inputs = _stub_collator(model)(samples)
    ones = torch.ones(batch_size, _ACTION_DIM)
    stats = NormStats(mean=ones * 0.0, std=ones, q01=-ones, q99=ones)
    return CollatedBatch(
        encoder_inputs=inputs,
        state=torch.randn(batch_size, 6, generator=generator),
        actions=torch.randn(
            batch_size,
            _HORIZON,
            _ACTION_DIM,
            generator=generator,
        ),
        action_is_pad=torch.zeros(batch_size, _HORIZON, dtype=torch.bool),
        action_stats=stats,
        state_stats=stats,
        action_tokens=None,
        suffix_tokens=None,
        suffix_is_aux=None,
    )


def _trunk_grad_norm(model: BijouModel) -> float:
    total = 0.0
    for parameter in model.backbone.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.float().norm()) ** 2
    return total**0.5


def _run_backward(model: BijouModel, *, insulate: bool) -> tuple[float, int]:
    """One flow-loss backward with a LIVE trunk; returns (trunk grad
    norm, expert trainable params with grads)."""
    model.insulate_expert = insulate
    model.backbone.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    batch = _batch(model)
    memory = model.encode(batch.encoder_inputs, with_grad=True)
    decoder = model.decoder
    assert isinstance(decoder, MolmoFlowDecoder)
    loss = molmo_flow_loss(decoder, memory, batch, insulate=insulate)
    loss.backward()
    expert_grads = sum(
        1
        for p in decoder.parameters()
        if p.requires_grad and p.grad is not None and bool((p.grad != 0).any())
    )
    return _trunk_grad_norm(model), expert_grads


def test_insulation_gradient_contract(tiny_model: BijouModel) -> None:
    """decision 8, both ways: detached KV ⇒ trunk grads EXACTLY zero;
    open seam ⇒ nonzero trunk grads through the cached K/V. The expert
    trains under both."""
    insulated_norm, insulated_expert = _run_backward(tiny_model, insulate=True)
    open_norm, open_expert = _run_backward(tiny_model, insulate=False)
    tiny_model.backbone.requires_grad_(False)
    tiny_model.zero_grad(set_to_none=True)
    assert insulated_norm == 0.0
    assert open_norm > 0.0
    assert insulated_expert > 0 and open_expert > 0


def test_open_seam_composes_with_activation_checkpointing(
    tiny_model: BijouModel,
) -> None:
    """Uninsulated gradients arrive at the trunk THROUGH the cache —
    the checkpointed-block shim must carry them (the machinery's whole
    design battle; a silent drop would train the trunk on nothing)."""
    transformer = tiny_model.backbone.text.transformer
    transformer.gradient_checkpointing = True
    try:
        open_norm, _ = _run_backward(tiny_model, insulate=False)
    finally:
        transformer.gradient_checkpointing = False
        tiny_model.backbone.requires_grad_(False)
        tiny_model.zero_grad(set_to_none=True)
    assert open_norm > 0.0


def test_predict_chunk_dispatch_and_tail(tiny_model: BijouModel) -> None:
    """BijouModel → molmo_flow predict: recorded-steps Euler default,
    their tail (n_action_steps × action_dim, fp32 raw units inside the
    quantile box), deterministic under a seed, φ_s knob refused, and
    per-sample stats unused (decision 6)."""
    batch = _batch(tiny_model)
    first = tiny_model.predict_chunk(
        batch,
        generator=torch.Generator().manual_seed(3),
    )
    again = tiny_model.predict_chunk(
        batch,
        generator=torch.Generator().manual_seed(3),
    )
    other = tiny_model.predict_chunk(
        batch,
        generator=torch.Generator().manual_seed(4),
    )
    assert first.actions.shape == (2, _HORIZON, _ACTION_DIM)
    assert first.actions.dtype == torch.float32
    assert first.generations is None
    # The clamp tail: predictions cannot leave the quantile box (±3).
    assert float(first.actions.abs().max()) <= 3.0
    assert torch.equal(first.actions, again.actions)
    assert not torch.equal(first.actions, other.actions)
    with pytest.raises(ValueError, match="target_time"):
        tiny_model.predict_chunk(batch, target_time=0.0)
    # Per-sample stats are decoder-ignored: perturbing them is inert.
    import dataclasses

    perturbed = dataclasses.replace(
        batch,
        action_stats=NormStats(
            mean=batch.action_stats.mean + 100.0,
            std=batch.action_stats.std * 7.0,
            q01=batch.action_stats.q01 - 5.0,
            q99=batch.action_stats.q99 + 5.0,
        ),
    )
    same = tiny_model.predict_chunk(
        perturbed,
        generator=torch.Generator().manual_seed(3),
    )
    assert torch.equal(first.actions, same.actions)
    heun = tiny_model.predict_chunk(
        batch,
        generator=torch.Generator().manual_seed(3),
        method=SamplingMethod.HEUN,
        num_steps=2,
    )
    assert heun.actions.shape == first.actions.shape


def test_loss_arms_sum_form_contract(tiny_model: BijouModel) -> None:
    """loss_components' mean == loss_component_sums / normalizer, and the
    normalizer is B*T (the per-position valid-dim mean is the inner
    reduction) — the chunked-backward exactness contract."""
    batch = _batch(tiny_model)
    memory = tiny_model.encode(batch.encoder_inputs, with_grad=False)
    torch.manual_seed(11)  # the loss draws t/ε from the ambient stream
    total, action, aux_sum, aux_count = tiny_model.loss_components(memory, batch)
    torch.manual_seed(11)
    loss_sum, count, aux2, _ = tiny_model.loss_component_sums(memory, batch)
    normalizer, aux_norm = tiny_model.loss_count_normalizers(batch)
    assert aux_sum is None and aux2 is None and aux_norm is None
    assert int(normalizer) == 2 * _HORIZON
    assert int(count) == int(normalizer)
    assert float(total) == pytest.approx(float(loss_sum / count), rel=1e-6)
    assert float(action) == pytest.approx(float(total))


def test_chunk_length_mismatch_is_loud(tiny_model: BijouModel) -> None:
    import dataclasses

    batch = _batch(tiny_model)
    wrong = dataclasses.replace(
        batch,
        actions=batch.actions[:, : _HORIZON - 1],
        action_is_pad=batch.action_is_pad[:, : _HORIZON - 1],
    )
    memory = tiny_model.encode(wrong.encoder_inputs, with_grad=False)
    decoder = tiny_model.decoder
    assert isinstance(decoder, MolmoFlowDecoder)
    with pytest.raises(ValueError, match="chunk length"):
        molmo_flow_loss(decoder, memory, wrong)
