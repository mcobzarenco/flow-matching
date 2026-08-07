"""#20 activation checkpointing keystone oracles — CPU, tiny molmo2
fixture. The flag is a memory lever only; these pin that it never
touches the math:

(i)   the K-configuration train step (live trunk, cache prefill + CE
      suffix + residual taps + flow) under checkpointing reproduces the
      plain step BITWISE — loss components and every parameter gradient
      across trunk, encoder, expert and CE rider — with a call spy
      pinning that the checkpointed path actually ran (equality must
      not be vacuous);
(ii)  the transformer-level prefill + cached-suffix forward/backward is
      bitwise under checkpointing: outputs, tap sink, the cache
      contents the suffix consumes, input-embedding gradients (which
      partly arrive THROUGH the cache — the escaped-K/V path) and every
      block parameter's gradient;
(iii) no-grad forwards with the flag on never enter
      ``torch.utils.checkpoint`` and stay bitwise the flag-off result
      (eval/generation untouched);
(iv)  the frozen-trunk F-arm step with the flag on is bitwise the
      flag-off step and never checkpoints — the gate is grad-enabled,
      and the F arm encodes its prefix under no_grad.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.utils.checkpoint
from test_molmo2_ar import text_config
from test_molmo2_residual import (
    build_flow_model,
    build_joint_model,
    flow_batch,
    fresh_model,
    perturb_zero_init_heads,
)

from bijou.decoders.flow import FlowDecoder
from bijou.model import BijouModel
from bijou.molmo2.cache import Molmo2KVCache
from bijou.molmo2.testing import write_tiny_text_checkpoint
from bijou.molmo2.text import Molmo2Transformer
from bijou.train import BijouTrainStep


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_tiny_text_checkpoint(
        tmp_path_factory.mktemp("molmo2-actckpt") / "tiny-molmo2",
    )


class CheckpointSpy:
    """Counts ``torch.utils.checkpoint.checkpoint`` calls while
    delegating to the real one — the non-vacuity probe: an equality
    oracle whose checkpointed arm silently took the plain path would
    pass on nothing."""

    def __init__(self) -> None:
        self.calls = 0
        self._real = torch.utils.checkpoint.checkpoint

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._real(*args, **kwargs)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[CheckpointSpy]:
    probe = CheckpointSpy()
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", probe)
    yield probe


def named_grads(module: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {name: p.grad for name, p in module.named_parameters()}


def assert_grads_bitwise(
    plain: dict[str, torch.Tensor | None],
    checkpointed: dict[str, torch.Tensor | None],
    *,
    context: str,
) -> None:
    assert plain.keys() == checkpointed.keys()
    for name, plain_grad in plain.items():
        ckpt_grad = checkpointed[name]
        assert (plain_grad is None) == (ckpt_grad is None), f"{context}: {name}"
        if plain_grad is not None and ckpt_grad is not None:
            assert torch.equal(plain_grad, ckpt_grad), (
                f"{context}: gradient {name} diverges under activation "
                "checkpointing — the flag must be memory-only"
            )


# -- (i) the K-configuration step, checkpointed == plain ----------------------


def joint_step_backward(
    tiny_checkpoint: Path,
    *,
    checkpointing: bool,
) -> tuple[BijouModel, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One full joint (K-arm) train step + backward; the builders are
    fully seeded, so both calls start from identical weights."""
    sample = flow_batch()
    model = build_joint_model(tiny_checkpoint, stop_grad=True)
    model.backbone.text.transformer.gradient_checkpointing = checkpointing
    step = BijouTrainStep(model, backbone_trained=True)
    torch.manual_seed(11)  # identical τ/ε draws across the two runs
    loss, flow_sum, ce_sum, ce_count = step(sample)
    loss.backward()
    assert ce_sum is not None and ce_count is not None
    return model, loss.detach(), flow_sum, ce_sum


def test_joint_step_checkpointed_is_bitwise_the_plain_step(
    tiny_checkpoint: Path,
    spy: CheckpointSpy,
) -> None:
    plain_model, plain_loss, plain_flow, plain_ce = joint_step_backward(
        tiny_checkpoint,
        checkpointing=False,
    )
    assert spy.calls == 0
    plain = named_grads(plain_model)

    ckpt_model, ckpt_loss, ckpt_flow, ckpt_ce = joint_step_backward(
        tiny_checkpoint,
        checkpointing=True,
    )
    # Exactly the live-trunk forwards: cache prefill + CE suffix, one
    # checkpoint call per block each (the flow expert never enters).
    n_blocks = len(ckpt_model.backbone.text.transformer.blocks)
    assert spy.calls == 2 * n_blocks

    assert torch.equal(plain_loss, ckpt_loss)
    assert torch.equal(plain_flow, ckpt_flow)
    assert torch.equal(plain_ce, ckpt_ce)
    assert_grads_bitwise(plain, named_grads(ckpt_model), context="joint step")
    # Non-vacuity of the trunk half: CE gradients must actually reach
    # the trunk through the retained (escaped-K/V) cache.
    assert any(
        grad is not None and bool(grad.any())
        for name, grad in plain.items()
        if name.startswith("backbone.")
    ), "no trunk gradient at all — the oracle would be comparing zeros"


# -- (ii) transformer-level prefill + cached suffix ---------------------------


def prefill_suffix_backward(
    transformer: Molmo2Transformer,
    *,
    checkpointing: bool,
) -> tuple[dict[str, torch.Tensor | None], list[torch.Tensor]]:
    """Prefill with a retained cache, continue a suffix against it, and
    backward a loss over prefill output + tap + suffix output — suffix
    gradients reach the prefill blocks only through the cache, the
    escaped-K/V path the shim must keep graph-connected."""
    transformer.gradient_checkpointing = checkpointing
    transformer.zero_grad(set_to_none=True)
    generator = torch.Generator().manual_seed(13)
    hidden = transformer.config.hidden_size
    prefix = torch.randn(2, 7, hidden, generator=generator).requires_grad_(True)
    suffix = torch.randn(2, 3, hidden, generator=generator).requires_grad_(True)

    cache = Molmo2KVCache(len(transformer.blocks))
    sink: dict[int, torch.Tensor] = {}
    tap = len(transformer.blocks) - 1
    prefill_out = transformer(
        inputs_embeds=prefix,
        cache=cache,
        residual_taps=(tap,),
        residual_sink=sink,
    )
    assert cache.seen_tokens == 7
    suffix_out = transformer(inputs_embeds=suffix, cache=cache)
    loss = (
        prefill_out.square().sum()
        + sink[tap].square().sum()
        + suffix_out.square().sum()
    )
    loss.backward()

    tensors = [loss.detach(), prefill_out.detach(), suffix_out.detach()]
    tensors.append(sink[tap].detach())
    assert prefix.grad is not None and suffix.grad is not None
    tensors.extend([prefix.grad, suffix.grad])
    for layer in cache.layers:
        assert layer.keys is not None and layer.values is not None
        assert layer.keys.shape[-2] == 10  # 7 prefill + 3 suffix, once each
        tensors.extend([layer.keys.detach(), layer.values.detach()])
    return named_grads(transformer), tensors


def test_prefill_and_cached_suffix_bitwise_under_checkpointing(
    spy: CheckpointSpy,
) -> None:
    torch.manual_seed(5)
    transformer = Molmo2Transformer(text_config().text, dtype=torch.float32)

    plain_grads, plain_tensors = prefill_suffix_backward(
        transformer,
        checkpointing=False,
    )
    assert spy.calls == 0
    ckpt_grads, ckpt_tensors = prefill_suffix_backward(
        transformer,
        checkpointing=True,
    )
    assert spy.calls == 2 * len(transformer.blocks)

    assert len(plain_tensors) == len(ckpt_tensors)
    for index, (plain_t, ckpt_t) in enumerate(
        zip(plain_tensors, ckpt_tensors, strict=True),
    ):
        assert torch.equal(plain_t, ckpt_t), f"tensor {index} diverges"
    assert_grads_bitwise(plain_grads, ckpt_grads, context="prefill+suffix")
    assert any(grad is not None and bool(grad.any()) for grad in plain_grads.values())


# -- (iii) no-grad paths never checkpoint -------------------------------------


def test_no_grad_forward_skips_checkpointing_bitwise(
    spy: CheckpointSpy,
) -> None:
    torch.manual_seed(5)
    transformer = Molmo2Transformer(text_config().text, dtype=torch.float32)
    generator = torch.Generator().manual_seed(17)
    embeds = torch.randn(2, 7, transformer.config.hidden_size, generator=generator)

    with torch.no_grad():
        plain_out = transformer(inputs_embeds=embeds)
        transformer.gradient_checkpointing = True
        flagged_out = transformer(inputs_embeds=embeds)
    assert spy.calls == 0, "a no-grad forward entered torch.utils.checkpoint"
    assert torch.equal(plain_out, flagged_out)


# -- (iv) the frozen-trunk F-arm step is untouched ----------------------------


def f_arm_step_backward(
    tiny_checkpoint: Path,
    *,
    checkpointing: bool,
) -> tuple[BijouModel, torch.Tensor]:
    sample = flow_batch()
    model = build_flow_model(tiny_checkpoint, fresh_model(tiny_checkpoint))
    decoder = model.decoder
    assert isinstance(decoder, FlowDecoder)
    perturb_zero_init_heads(decoder)
    model.backbone.text.transformer.gradient_checkpointing = checkpointing
    step = BijouTrainStep(model, backbone_trained=False)
    torch.manual_seed(11)
    loss, _, _, _ = step(sample)
    loss.backward()
    return model, loss.detach()


def test_f_arm_step_never_checkpoints_and_is_bitwise(
    tiny_checkpoint: Path,
    spy: CheckpointSpy,
) -> None:
    plain_model, plain_loss = f_arm_step_backward(
        tiny_checkpoint,
        checkpointing=False,
    )
    ckpt_model, ckpt_loss = f_arm_step_backward(
        tiny_checkpoint,
        checkpointing=True,
    )
    assert spy.calls == 0, (
        "the F arm encodes its prefix under no_grad — the flag must not engage there"
    )
    assert torch.equal(plain_loss, ckpt_loss)
    assert_grads_bitwise(
        named_grads(plain_model),
        named_grads(ckpt_model),
        context="F-arm step",
    )
