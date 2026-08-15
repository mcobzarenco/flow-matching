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
  (decision 6: perturbing them changes nothing);
- the eval seam (BijouPolicy, loading the CONVERTED VLA-format
  checkpoint through the family registry): the collator carries the
  checkpoint's merged q01/q99 STATE table (training's scheme — without
  it the per-sample mean/std path silently shifts every state bin),
  predict noise is frame-identity keyed (ambient RNG perturbation is
  inert), and draw ensembling is refused with the cache-tiling reason.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from test_convert_molmoact2 import _ACTION_DIM, _HORIZON, _convert, source_dir

from bijou.checkpoint import read_metadata
from bijou.eval.molmo_norm import MolmoNorm
from bijou.eval.policies import BijouPolicy
from bijou.loading import from_checkpoint
from bijou.model import BijouModel
from bijou.modelling.aux_text import AuxField
from bijou.modelling.decoders.molmo_flow import MolmoFlowDecoder, molmo_flow_loss
from bijou.modelling.interface import (
    CameraFrame,
    CollatedBatch,
    NormStats,
    PromptInputs,
    SamplingMethod,
)

assert source_dir is not None  # re-exported pytest fixture (module-scoped)


def legacy_bridge(converted: Path, legacy: Path) -> Path:
    """The converted (VLA-format) checkpoint re-expressed in the legacy
    layout — the inverse of ``bijou.convert_legacy``, for THIS module's
    old-world subject (BijouModel exercises the legacy reader until the
    old world is deleted; the bridge dies with it — BijouPolicy now
    loads the converted directory itself)."""
    metadata = read_metadata(converted)
    legacy.mkdir(parents=True)
    (legacy / "bijou_config.json").write_text(
        json.dumps(
            {
                "format": 3,
                "backbone": {
                    "id": metadata.backbone_id,
                    "depth": metadata.backbone_depth,
                },
                "prompt": metadata.components["prompt"]["config"],
                "decoder": metadata.components["flow_decoder"]["config"],
                "step": metadata.step,
                "train_args": metadata.train_args,
                "normalization": metadata.stats.state_dict(),
                "per_dataset_normalization": {},
            },
        ),
    )
    os.link(
        converted / "flow_decoder.safetensors",
        legacy / "expert.safetensors",
    )
    return legacy


@pytest.fixture(scope="module")
def tiny_worlds(
    source_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    """(converted VLA-format checkpoint, its legacy re-expression) —
    one conversion serves the module's new-world (BijouPolicy) and
    old-world (BijouModel) subjects."""
    root = tmp_path_factory.mktemp("molmo-flow-int")
    converted = _convert(
        source_dir,
        root / "converted",
        backbone_ref=str(source_dir),
    )
    return converted, legacy_bridge(converted, root / "legacy")


@pytest.fixture(scope="module")
def tiny_vla_checkpoint(tiny_worlds: tuple[Path, Path]) -> Path:
    return tiny_worlds[0]


@pytest.fixture(scope="module")
def tiny_checkpoint(tiny_worlds: tuple[Path, Path]) -> Path:
    return tiny_worlds[1]


@pytest.fixture(scope="module")
def tiny_model(tiny_checkpoint: Path) -> BijouModel:
    model, _info = from_checkpoint(tiny_checkpoint)
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


def _stub_ids(collator: Any) -> Any:
    """Patch the stub tokenizer + special ids onto an EXISTING inputs
    collator (the tiny source ships no real tokenizer)."""
    from test_molmoact2_encoder import _StubTokenizer

    from bijou.modelling.encoders.molmoact2_processing import (
        BOS_ID,
        IM_END_ID,
        IM_PATCH_ID,
        IM_START_ID,
    )

    collator._tokenizer = _StubTokenizer()
    collator._image_ids = (IM_START_ID, IM_END_ID, IM_PATCH_ID)
    collator._patch_id = IM_PATCH_ID
    collator._eos_id = BOS_ID
    collator._action_start_id = 151_932
    collator._action_end_id = 151_933
    return collator


def _stub_collator(model: BijouModel):  # noqa: ANN202 — encoder-specific collator
    return _stub_ids(model.encoder.inputs_collator())


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

    assert batch.action_stats.q01 is not None and batch.action_stats.q99 is not None
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
    total, action, aux_sum, _aux_count = tiny_model.loss_components(memory, batch)
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


@pytest.fixture(scope="module")
def tiny_policy(tiny_vla_checkpoint: Path) -> BijouPolicy:
    policy = BijouPolicy(tiny_vla_checkpoint, device=torch.device("cpu"), seed=7)
    _stub_ids(policy.collator.inputs)
    return policy


def _eval_item(index: int, generator: torch.Generator) -> dict[str, Any]:
    """A raw eval item as StatsAttachedDataset would hand over (stats
    tensors from the tiny fixture's normalization row — mean 1, std 2,
    q01 −3, q99 3 — plus the identity triple stable noise keys on)."""
    dim = torch.ones(_ACTION_DIM)
    return {
        "task": f"pick up cube {index}",
        "repo_id": "user/tiny",
        "episode_index": 0,
        "frame_index": index,
        "observation.images.cam0": torch.rand((3, 48, 64), generator=generator),
        "observation.state": torch.rand((6,), generator=generator) * 6 - 3,
        "action": torch.randn(_HORIZON, _ACTION_DIM, generator=generator),
        "action_is_pad": torch.zeros(_HORIZON, dtype=torch.bool),
        "action_mean": dim * 1.0,
        "action_std": dim * 2.0,
        "action_q01": dim * -3.0,
        "action_q99": dim * 3.0,
        "state_mean": dim * 1.0,
        "state_std": dim * 2.0,
        "state_q01": dim * -3.0,
        "state_q99": dim * 3.0,
    }


def test_eval_policy_carries_merged_state_table(tiny_policy: BijouPolicy) -> None:
    """The eval collator normalizes STATE with the checkpoint's merged
    q01/q99 table (training's scheme, §8.13 decision 6) — the
    per-sample mean/std path would silently shift every state bin
    (found in the 2026-08-12 pre-eval review; this is the class fix)."""
    table = tiny_policy.collator
    normalization = tiny_policy.info.normalization
    assert normalization.state_q01 is not None
    assert table.state_q01 is not None and table.state_q99 is not None
    assert torch.equal(table.state_q01, torch.tensor(normalization.state_q01))
    assert torch.equal(
        table.state_q99,
        torch.tensor(normalization.state_q99, dtype=torch.float32),
    )


def test_eval_policy_noise_is_keyed_not_ambient(tiny_policy: BijouPolicy) -> None:
    """Predict noise derives from the frame identity (stable keying),
    NOT the ambient RNG: perturbing the global stream between calls
    must be inert. Before the fix the decoder drew ``torch.randn``
    unseeded — irreproducible and batch-composition-dependent. Success
    also pins the noise geometry: a [chunk, action_dim]-shaped tensor
    (flow.py's convention) would fail the [horizon, max_action_dim]
    action embed."""
    items = [
        _eval_item(0, torch.Generator().manual_seed(21)),
        _eval_item(1, torch.Generator().manual_seed(22)),
    ]
    torch.manual_seed(101)
    first = tiny_policy.predict(items, [0, 1])
    torch.manual_seed(999)
    torch.rand(1024)  # a DIFFERENT ambient stream position
    second = tiny_policy.predict(items, [0, 1])
    for a, b in zip(first, second, strict=True):
        assert a.shape == (_HORIZON, _ACTION_DIM)
        assert torch.equal(a, b)


def test_eval_policy_refuses_draw_ensembling(tiny_vla_checkpoint: Path) -> None:
    """molmo_flow draws>1 needs prefix-memory tiling the molmo2 KV
    cache does not have — refused with THAT reason (the generic guard's
    deterministic-decode message would misdiagnose a stochastic
    decoder)."""
    with pytest.raises(SystemExit, match="molmo_flow"):
        BijouPolicy(
            tiny_vla_checkpoint,
            device=torch.device("cpu"),
            seed=7,
            sample_draws=2,
        )


def test_eval_policy_narrows_capabilities_loudly(
    tiny_vla_checkpoint: Path,
) -> None:
    """The loud-narrowing rule on a real flow-only family: every
    explicitly requested instrument this family cannot back exits
    naming it — never a silent skip (the module-level scar in
    eval/policies.py)."""

    def refused(**kwargs: Any) -> None:
        with pytest.raises(SystemExit, match="molmoact2_flow"):
            BijouPolicy(
                tiny_vla_checkpoint,
                device=torch.device("cpu"),
                seed=7,
                **kwargs,
            )

    refused(generate=(AuxField.SUBGOAL,))
    refused(ar_temperature=1.0)
    refused(sde_noise_level=0.0)
    refused(target_time=0.0)


def test_eval_policy_refuses_legacy_directory(tiny_checkpoint: Path) -> None:
    """BijouPolicy reads the VLA format ONLY — a legacy
    bijou_config.json directory is refused at load with the converter
    pointer, never silently read through the old path."""
    with pytest.raises(SystemExit, match="convert_legacy"):
        BijouPolicy(tiny_checkpoint, device=torch.device("cpu"), seed=7)


def _offset_item(
    index: int,
    *,
    state: torch.Tensor,
    q01: float,
    q99: float,
    mean: float,
) -> dict[str, Any]:
    """An eval item whose dataset stats put every joint in the box
    [q01, q99] (vs the tiny checkpoint table's [-3, 3]); images/actions
    are seeded off ``index`` so paired items collate identical prompts
    when their (translated) states match."""
    generator = torch.Generator().manual_seed(1000 + index)
    dim = torch.ones(_ACTION_DIM)
    return {
        "task": "pick up the cube",
        "repo_id": "user/offset",
        "episode_index": 0,
        "frame_index": index,
        "observation.images.cam0": torch.rand((3, 48, 64), generator=generator),
        "observation.state": state,
        "action": torch.randn(_HORIZON, _ACTION_DIM, generator=generator),
        "action_is_pad": torch.zeros(_HORIZON, dtype=torch.bool),
        "action_mean": dim * mean,
        "action_std": dim * 1.0,
        "action_q01": dim * q01,
        "action_q99": dim * q99,
        "state_mean": dim * mean,
        "state_std": dim * 1.0,
        "state_q01": dim * q01,
        "state_q99": dim * q99,
    }


def test_convention_map_mode_is_offset_equivariant(
    tiny_vla_checkpoint: Path,
    tiny_policy: BijouPolicy,
) -> None:
    """The convmap arm's end-to-end contract on the REAL composition: a
    dataset offset by +180 from the table decodes to EXACTLY the
    contract read's chunks + 180 — same translated state (so identical
    prompt bytes), same frame-identity noise, inverse map on the way
    out. Pins state rewrite, map fit (offset −180 per joint), and the
    chunk pull-back in one oracle."""
    convmap = BijouPolicy(
        tiny_vla_checkpoint,
        device=torch.device("cpu"),
        seed=7,
        molmo_norm=MolmoNorm.CONVENTION_MAP,
    )
    _stub_ids(convmap.collator.inputs)
    assert convmap.name.endswith("_convmap")
    generator = torch.Generator().manual_seed(33)
    inbox = torch.rand((6,), generator=generator) * 4 - 2  # within [-2, 2]
    shifted = _offset_item(0, state=inbox + 180.0, q01=178.0, q99=182.0, mean=180.5)
    reference = _offset_item(0, state=inbox, q01=-3.0, q99=3.0, mean=1.0)
    ours = convmap.predict([shifted], [0])
    contract = tiny_policy.predict([reference], [0])
    torch.testing.assert_close(ours[0], contract[0] + 180.0, atol=1e-4, rtol=0)


def test_per_dataset_mode_is_scale_equivariant(
    tiny_vla_checkpoint: Path,
    tiny_policy: BijouPolicy,
) -> None:
    """The pdnorm arm's contract: a dataset spanning [-6, 6] (2x the
    table box) quantile-equates through A(x) = x/2 — the translated
    state collates the same prompt as the contract read on x/2, and
    the decoded chunks come back exactly 2x the contract chunks."""
    pdnorm = BijouPolicy(
        tiny_vla_checkpoint,
        device=torch.device("cpu"),
        seed=7,
        molmo_norm=MolmoNorm.PER_DATASET,
    )
    _stub_ids(pdnorm.collator.inputs)
    assert pdnorm.name.endswith("_pdnorm")
    generator = torch.Generator().manual_seed(44)
    inbox = torch.rand((6,), generator=generator) * 4 - 2
    wide = _offset_item(1, state=inbox * 2, q01=-6.0, q99=6.0, mean=0.0)
    reference = _offset_item(1, state=inbox, q01=-3.0, q99=3.0, mean=1.0)
    ours = pdnorm.predict([wide], [1])
    contract = tiny_policy.predict([reference], [1])
    torch.testing.assert_close(ours[0], contract[0] * 2.0, atol=1e-4, rtol=0)
