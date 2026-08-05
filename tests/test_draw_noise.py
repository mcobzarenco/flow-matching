"""Noise-draw ensembling seed derivation (bijou.eval.policies.draw_noise)
and the ensemble collapse that --dump-draws taps."""

from __future__ import annotations

import torch

from bijou.eval.policies import (
    DRAW_SEED_STRIDE,
    collapse_draws,
    draw_noise,
    sample_noise,
)

SHAPE = (50, 6)


def test_draw_zero_matches_single_draw_convention() -> None:
    """Draw 0 must be byte-identical to the historical per-item noise —
    paired comparisons against every prior flow eval depend on it."""
    for index in (0, 7, 20_719_388):
        assert torch.equal(
            draw_noise(3, index, 0, SHAPE),
            sample_noise(3 + index, SHAPE),
        )


def test_draws_are_deterministic_and_distinct() -> None:
    first = draw_noise(0, 5, 1, SHAPE)
    again = draw_noise(0, 5, 1, SHAPE)
    other_draw = draw_noise(0, 5, 2, SHAPE)
    other_item = draw_noise(0, 6, 1, SHAPE)
    assert torch.equal(first, again)
    assert not torch.equal(first, other_draw)
    assert not torch.equal(first, other_item)


def test_draw_seeds_cannot_collide_with_frame_indices() -> None:
    """A draw>0 seed of one item must never equal the draw-0 seed of
    another item at realistic corpus sizes (~2x10^7 frames < the
    2**26 stride; curated-v0 has 20,719,389)."""
    max_index = 25_000_000
    assert torch.equal(
        draw_noise(0, 0, 1, SHAPE),
        draw_noise(0, DRAW_SEED_STRIDE, 0, SHAPE),
    )  # the collision DOES exist beyond the stride — documents the bound
    assert not torch.equal(
        draw_noise(0, 0, 1, SHAPE),
        draw_noise(0, max_index, 0, SHAPE),
    )


def test_stride_survives_torch_seed_truncation() -> None:
    """torch CPU Generator.manual_seed ignores bits >= 32 (measured
    2026-08-05): 10 draws of one item must be 10 DISTINCT tensors —
    the silent failure mode is every draw collapsing to draw 0 and the
    'ensemble' averaging N identical chunks."""
    draws = [draw_noise(0, 123, d, SHAPE) for d in range(10)]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not torch.equal(draws[i], draws[j]), (i, j)


def test_collapse_draws_mean_matches_and_keeps_every_draw() -> None:
    """The dumped per-draw stacks must (a) contain every draw verbatim and
    (b) average back to exactly the chunks the policy predicts — the dump
    is a pure tap, never a second computation of the ensemble."""
    stacked = torch.randn(3, 4, 5, 6)  # [draws, batch, chunk, dim]
    means, per_item = collapse_draws(stacked)
    assert len(means) == 4 and len(per_item) == 4
    for i in range(4):
        assert means[i].shape == (5, 6)
        assert per_item[i].shape == (3, 5, 6)
        for d in range(3):
            assert torch.equal(per_item[i][d], stacked[d, i])
        # Byte-identical to the pre-tap prediction path: one mean over the
        # full stack, taken before any per-item split.
        assert torch.equal(means[i], stacked.mean(dim=0)[i])
