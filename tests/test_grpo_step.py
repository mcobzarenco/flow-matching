"""Token-GRPO step oracles (design memo 2026-08-13 §8 item 2), pure
CPU: the clipped-surrogate gradient step in bijou/train_grpo.py.

What these pin: (1) the item-2 mask oracle — the train-time grammar
mask (recomputed from ids alone) equals the rollout's recorded mask
bit-for-bit per token, greedy and sampled; (2) at an unchanged policy
the train-time forward reproduces the rollout logprobs to the §8
amended bound (1e-5 — reduction-shape noise, one-shot vs incremental
trunk forward), so every ratio is 1 to that noise, nothing clips, and
measured KL is ~0; (3) ratio ≡ 1 (old = new.detach()) reduces the
surrogate gradient to advantage-weighted CE's BIT-EXACTLY
(torch.minimum splits tie gradients evenly and the halves resum
exactly; exp(x − x.detach()) is exactly 1); (4) zero advantage zeroes
the gradient on every parameter identically — and the same path
produces nonzero gradients under a nonzero advantage; (5) the DAPO
clip bounds bind: beyond-bound ratios stop the gradient exactly when
the clipped branch is the min (A > 0) and keep it when the unclipped
branch is (A < 0); (6) PAD padding never contributes — old-logprob
values at non-decision positions are mask-multiplied out bit-exactly;
(7) the loud guards. Fixture family: tests/test_ar_backbone via
tests/test_token_rows."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
import torch
from test_ar_backbone import batch, encode_memory
from test_token_rows import greedy_rows, rngs, unpack

from bijou.eval.policies import TokenRow, token_rows_from_capture
from bijou.modelling.interface import ActionCaptureStep, ARSampling
from bijou.models.ar_suffix_ops import CollatedBatch, batch_action_quantiles
from bijou.train_grpo import (
    GRPOConfig,
    grammar_masks_from_ids,
    grpo_loss,
    grpo_objective_sums,
    sampled_token_logprobs,
)


def replay_batch(
    template: CollatedBatch[Any],
    rows: list[TokenRow],
    boa: int,
    pad: int,
    *,
    extra_pad: int = 0,
) -> tuple[CollatedBatch[Any], torch.Tensor]:
    """(batch with the rollout's sampled ids as action_tokens, padded
    old logprobs) — the replay collator convention item 3 will build:
    ``[BOA, a_1..a_T]`` PAD-padded to the batch width, old logprobs
    zero-padded to match."""
    width = max(len(row.ids) for row in rows) + extra_pad
    tokens = torch.tensor(
        [[boa, *row.ids.tolist(), *[pad] * (width - len(row.ids))] for row in rows],
        dtype=torch.long,
    )
    old = torch.zeros((len(rows), width), dtype=torch.float32)
    for i, row in enumerate(rows):
        old[i, : len(row.ids)] = torch.from_numpy(row.logprobs)
    return dataclasses.replace(template, action_tokens=tokens), old


def test_train_mask_equals_rollout_mask() -> None:
    """The item-2 oracle: grammar_masks_from_ids recomputes the
    decode's applied mask bit-for-bit from the recorded ids, per
    token, on greedy AND sampled rows — and its decision positions are
    exactly the recorded steps."""
    backbone, decoder, memory, _, greedy = greedy_rows()
    del memory
    loaded = decoder.codec
    capture: list[ActionCaptureStep] = []
    sample = batch(loaded)
    decoder.predict_chunk(
        backbone,
        encode_memory(backbone),
        sample,
        quantiles=batch_action_quantiles(sample),
        sampling=ARSampling(temperature=2.0, rngs=rngs(0)),
        action_capture=capture,
    )
    sampled = token_rows_from_capture(
        capture,
        block_base=decoder.config.block_base,
        temperature=2.0,
    )
    for rows in (greedy, sampled):
        width = max(len(row.ids) for row in rows)
        ids = torch.tensor(
            [
                [*row.ids.tolist(), *[loaded.pad] * (width - len(row.ids))]
                for row in rows
            ],
            dtype=torch.long,
        )
        allowed, decisions = grammar_masks_from_ids(decoder, ids)
        for i, row in enumerate(rows):
            count = len(row.ids)
            assert int(decisions[i].sum()) == count
            assert bool(decisions[i, :count].all())
            assert torch.equal(allowed[i, :count], unpack(row))


def test_fresh_policy_ratios_are_one_to_reduction_noise() -> None:
    """At the rollout's own checkpoint the train-time forward
    reproduces the recorded logprobs within the §8 amended bound, so
    ratios are 1 to reduction-shape noise: nothing clips, measured KL
    is ~0, and the surrogate mean is the mean advantage."""
    backbone, decoder, memory, _, rows = greedy_rows()
    codec = decoder.codec
    grpo, old = replay_batch(batch(codec), rows, codec.boa, codec.pad)
    snapshot = decoder.cache_snapshot(memory)
    new, decisions = sampled_token_logprobs(
        backbone,
        decoder,
        memory,
        grpo,
        temperature=1.0,
    )
    decoder.cache_restore(memory, snapshot)
    assert decisions.sum(dim=1).tolist() == [len(row.ids) for row in rows]
    drift = float(((new.detach() - old).abs() * decisions).max())
    assert drift < 1e-5, (
        f"train-time logprobs drifted {drift:.2e} from the rollout's "
        "recorded rows — beyond reduction-shape noise, the training "
        "forward is not the sampling distribution"
    )
    advantages = torch.tensor([1.7, -0.6])
    loss, stats = grpo_loss(
        backbone,
        decoder,
        memory,
        grpo,
        old_logprobs=old,
        advantages=advantages,
        config=GRPOConfig(),
    )
    decoder.cache_restore(memory, snapshot)
    assert stats.clip_fraction == 0.0
    assert abs(stats.max_ratio - 1.0) < 2e-5
    assert abs(stats.min_ratio - 1.0) < 2e-5
    assert stats.approx_kl < 1e-9
    counts = torch.tensor([float(len(row.ids)) for row in rows])
    expected = -float((advantages * counts).sum() / counts.sum())
    assert abs(float(loss.detach()) - expected) < 1e-4


def test_ratio_one_reduces_to_weighted_ce_exactly() -> None:
    """Old = new.detach(): the surrogate's value is the mean advantage
    and its gradient is BIT-EXACTLY advantage-weighted CE's — the
    clipped branch is interior (clamp passes gradient), torch.minimum
    halves the tie and the halves resum exactly, exp(0) multiplies by
    exactly 1."""
    generator = torch.Generator().manual_seed(3)
    new = torch.randn(4, 9, generator=generator, requires_grad=True)
    advantages = torch.randn(4, generator=generator)
    decisions = torch.rand(4, 9, generator=generator) > 0.3
    decisions[:, 0] = True
    objective_sum, count, _ = grpo_objective_sums(
        new,
        new.detach(),
        advantages,
        decisions,
        GRPOConfig(),
    )
    loss = -(objective_sum / count)
    (grad,) = torch.autograd.grad(loss, new)
    reference = new.detach().clone().requires_grad_(True)
    weighted_ce = -((advantages[:, None] * reference * decisions.float()).sum() / count)
    (reference_grad,) = torch.autograd.grad(weighted_ce, reference)
    assert torch.equal(grad, reference_grad)
    expected = -(advantages[:, None] * decisions.float()).sum() / count
    assert torch.equal(loss.detach(), expected)


def test_zero_advantage_is_zero_grad_on_every_parameter() -> None:
    """Zero advantage: the whole surrogate is 0·ratio, so backward
    writes an EXACT zero into every participating parameter — and the
    same path under a nonzero advantage moves real gradient (the
    zero is the objective's, not a dead graph's)."""
    backbone, decoder, memory, _, rows = greedy_rows()
    codec = decoder.codec
    grpo, old = replay_batch(batch(codec), rows, codec.boa, codec.pad)
    snapshot = decoder.cache_snapshot(memory)

    def gradients(advantages: torch.Tensor) -> list[torch.Tensor | None]:
        decoder.zero_grad()
        decoder.cache_restore(memory, snapshot)
        loss, _ = grpo_loss(
            backbone,
            decoder,
            memory,
            grpo,
            old_logprobs=old,
            advantages=advantages,
            config=GRPOConfig(),
        )
        loss.backward()
        return [
            parameter.grad
            for parameter in decoder.parameters()
            if parameter.requires_grad
        ]

    zero = gradients(torch.zeros(len(rows)))
    live = [grad for grad in zero if grad is not None]
    assert live, "no parameter participated in the GRPO graph"
    assert all(bool((grad == 0).all()) for grad in live)
    moved = gradients(torch.ones(len(rows)))
    assert any(grad is not None and bool((grad != 0).any()) for grad in moved), (
        "nonzero advantage moved no gradient — the zero above is a dead graph"
    )
    decoder.zero_grad()
    decoder.cache_restore(memory, snapshot)


def test_clip_bounds_bind() -> None:
    """DAPO clip-higher: past a bound the clipped branch caps the
    objective exactly when it is the min (A > 0 — gradient stops), and
    the unclipped branch keeps both value and gradient when it is
    (A < 0); the stats count exactly the beyond-bound tokens."""
    config = GRPOConfig(clip_low=0.8, clip_high=1.28)
    old = torch.zeros(4, 1)
    shift = torch.tensor([[0.5], [0.5], [-0.5], [-0.5]])  # ratio e^±0.5
    new = shift.clone().requires_grad_(True)
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
    decisions = torch.ones(4, 1, dtype=torch.bool)
    objective_sum, count, stats = grpo_objective_sums(
        new,
        old,
        advantages,
        decisions,
        config,
    )
    assert int(count) == 4
    assert stats.clip_fraction == 1.0
    ratio = torch.exp(shift)
    expected = torch.stack(
        [
            torch.tensor(config.clip_high),  # A>0, r>hi: capped
            -ratio[1, 0],  # A<0, r>hi: unclipped is the min
            ratio[2, 0],  # A>0, r<lo: unclipped is the min
            -torch.tensor(config.clip_low),  # A<0, r<lo: capped
        ],
    )
    assert torch.allclose(objective_sum, expected.sum())
    (grad,) = torch.autograd.grad(objective_sum, new)
    assert float(grad[0, 0]) == 0.0  # capped: gradient stopped
    assert float(grad[3, 0]) == 0.0
    assert torch.allclose(grad[1, 0], -ratio[1, 0])  # d(r·A)/dx = r·A
    assert torch.allclose(grad[2, 0], ratio[2, 0])


def test_padding_never_contributes() -> None:
    """Old-logprob values at non-decision positions are mask-multiplied
    out: rewriting them changes nothing, bit-for-bit."""
    backbone, decoder, memory, _, rows = greedy_rows()
    codec = decoder.codec
    grpo, old = replay_batch(batch(codec), rows, codec.boa, codec.pad, extra_pad=2)
    advantages = torch.tensor([0.9, -1.1])
    snapshot = decoder.cache_snapshot(memory)

    def loss_with(old_logprobs: torch.Tensor) -> float:
        decoder.cache_restore(memory, snapshot)
        loss, _ = grpo_loss(
            backbone,
            decoder,
            memory,
            grpo,
            old_logprobs=old_logprobs,
            advantages=advantages,
            config=GRPOConfig(),
        )
        return float(loss.detach())

    assert grpo.action_tokens is not None
    _, decisions = grammar_masks_from_ids(decoder, grpo.action_tokens[:, 1:])
    rewritten = old.masked_fill(~decisions, 1e6)
    assert not torch.equal(rewritten, old)
    assert loss_with(old) == loss_with(rewritten)
    decoder.cache_restore(memory, snapshot)


def test_guards_are_loud() -> None:
    backbone, decoder, memory, _, rows = greedy_rows()
    codec = decoder.codec
    template = batch(codec)
    with pytest.raises(ValueError, match="bracket 1"):
        GRPOConfig(clip_low=1.5)
    with pytest.raises(ValueError, match="positive"):
        GRPOConfig(temperature=0.0)
    with pytest.raises(SystemExit, match="aux"):
        sampled_token_logprobs(
            backbone,
            decoder,
            memory,
            dataclasses.replace(
                template,
                suffix_tokens=torch.zeros(2, 3, dtype=torch.long),
                suffix_is_aux=torch.zeros(2, 3, dtype=torch.bool),
            ),
            temperature=1.0,
        )
    with pytest.raises(SystemExit, match="action_tokens"):
        sampled_token_logprobs(
            backbone,
            decoder,
            memory,
            dataclasses.replace(template, action_tokens=None),
            temperature=1.0,
        )
    headless = dataclasses.replace(
        template,
        action_tokens=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="BOA"):
        sampled_token_logprobs(backbone, decoder, memory, headless, temperature=1.0)
    # Rows that do not consume the chunk exactly, and non-PAD (BOA,
    # symbol length 0) past consumption — single-row tensors, the
    # fixture's two rows need not share a length.
    short = torch.tensor([rows[0].ids[:-1].tolist()], dtype=torch.long)
    with pytest.raises(ValueError, match="consume the chunk"):
        grammar_masks_from_ids(decoder, short)
    trailing_boa = torch.tensor(
        [[*rows[0].ids.tolist(), codec.boa]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="past chunk consumption"):
        grammar_masks_from_ids(decoder, trailing_boa)
    new = torch.zeros(2, 3)
    decisions = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="shape mismatch"):
        grpo_objective_sums(
            new,
            torch.zeros(2, 4),
            torch.zeros(2),
            decisions,
            GRPOConfig(),
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        grpo_objective_sums(
            new,
            new,
            torch.zeros(3),
            decisions,
            GRPOConfig(),
        )
    with pytest.raises(ValueError, match="non-finite"):
        grpo_objective_sums(
            new,
            torch.full((2, 3), float("-inf")),
            torch.zeros(2),
            decisions,
            GRPOConfig(),
        )
    assert np.isfinite(rows[0].logprobs).all()  # fixture sanity
