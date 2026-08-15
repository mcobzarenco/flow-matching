"""#20 activation checkpointing keystone oracles — CPU, tiny molmo2
fixture. The flag is a memory lever only; these pin that it never
touches the math:

(i)   the transformer-level prefill + cached-suffix forward/backward is
      bitwise under checkpointing: outputs, tap sink, the cache
      contents the suffix consumes, input-embedding gradients (which
      partly arrive THROUGH the cache — the escaped-K/V path) and every
      block parameter's gradient;
(ii)  no-grad forwards with the flag on never enter
      ``torch.utils.checkpoint`` and stay bitwise the flag-off result
      (eval/generation untouched).

(The retired flow-residual F/K-arm step oracles lived here too — tag
pre-decoder-simplify; the live model-level composition — molmo_flow's
open KI seam under checkpointing — is pinned in
test_molmo_flow_integration.py.)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.utils.checkpoint
from test_molmo2_ar import text_config
from torch.nn.attention import SDPBackend, sdpa_kernel

from bijou.modelling.molmo2.cache import Molmo2KVCache
from bijou.modelling.molmo2.text import Molmo2Transformer, _ambient_sdpa_backends


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


# -- (i) transformer-level prefill + cached suffix ----------------------------


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


# -- (ii) no-grad paths never checkpoint --------------------------------------


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


# -- (iii) #20 CUDA fix: backward recompute replays the forward's sdpa pin ----


def test_ambient_sdpa_backends_reconstruct_the_active_pin() -> None:
    """``sdpa_kernel`` pins flip the four global backend flags;
    ``_ambient_sdpa_backends`` must read back exactly the active pin
    (and the unpinned baseline once the context exits) — this capture
    is what the checkpointed region re-applies during recompute."""
    baseline = _ambient_sdpa_backends()
    with sdpa_kernel([SDPBackend.MATH]):
        assert _ambient_sdpa_backends() == [SDPBackend.MATH]
        with sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH],
        ):
            assert _ambient_sdpa_backends() == [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
            ]
        assert _ambient_sdpa_backends() == [SDPBackend.MATH]
    assert _ambient_sdpa_backends() == baseline


def pinned_prefill_suffix_backward_cuda(
    transformer: Molmo2Transformer,
    *,
    checkpointing: bool,
) -> tuple[dict[str, torch.Tensor | None], list[torch.Tensor]]:
    """The #20 crash shape: prefill + cached suffix FORWARD under the
    suffix path's non-cuDNN ``sdpa_kernel`` pin (landing on MATH) with
    bf16 autocast making fused kernels eligible, then backward OUTSIDE
    the pin — exactly ``train.py``'s ``share.backward()``, which runs
    far from the pin in ``ar_molmo2._continue_suffix``. Pre-fix, the
    unpinned recompute dispatched a fused bf16 kernel against the
    MATH-saved graph and raised ``CheckpointError`` (metadata
    mismatch); post-fix the recompute replays under the captured pin.
    """
    device = torch.device("cuda")
    transformer.gradient_checkpointing = checkpointing
    transformer.zero_grad(set_to_none=True)
    hidden = transformer.config.hidden_size
    torch.manual_seed(13)
    prefix = torch.randn(2, 7, hidden, device=device).requires_grad_(True)
    suffix = torch.randn(2, 3, hidden, device=device).requires_grad_(True)

    cache = Molmo2KVCache(len(transformer.blocks))
    with (
        sdpa_kernel([SDPBackend.MATH]),
        torch.autocast("cuda", torch.bfloat16),
    ):
        prefill_out = transformer(inputs_embeds=prefix, cache=cache)
        suffix_out = transformer(inputs_embeds=suffix, cache=cache)
    loss = prefill_out.float().square().sum() + suffix_out.float().square().sum()
    loss.backward()  # outside the pin — the regression under test

    tensors = [loss.detach(), prefill_out.detach(), suffix_out.detach()]
    assert prefix.grad is not None and suffix.grad is not None
    tensors.extend([prefix.grad, suffix.grad])
    for layer in cache.layers:
        assert layer.keys is not None and layer.values is not None
        tensors.extend([layer.keys.detach(), layer.values.detach()])
    return named_grads(transformer), tensors


@pytest.mark.gpu
def test_checkpointed_backward_replays_forward_sdpa_pin_on_cuda(
    spy: CheckpointSpy,
) -> None:
    torch.manual_seed(5)
    transformer = Molmo2Transformer(text_config().text, dtype=torch.float32).to(
        "cuda",
    )

    plain_grads, plain_tensors = pinned_prefill_suffix_backward_cuda(
        transformer,
        checkpointing=False,
    )
    assert spy.calls == 0
    ckpt_grads, ckpt_tensors = pinned_prefill_suffix_backward_cuda(
        transformer,
        checkpointing=True,
    )
    assert spy.calls == 2 * len(transformer.blocks)

    assert len(plain_tensors) == len(ckpt_tensors)
    for index, (plain_t, ckpt_t) in enumerate(
        zip(plain_tensors, ckpt_tensors, strict=True),
    ):
        assert torch.equal(plain_t, ckpt_t), f"tensor {index} diverges"
    assert_grads_bitwise(plain_grads, ckpt_grads, context="pinned prefill+suffix")
    assert any(grad is not None and bool(grad.any()) for grad in plain_grads.values())
