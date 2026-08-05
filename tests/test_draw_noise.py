"""Noise-draw ensembling seed derivation (bijou.eval.policies.draw_noise)."""

from __future__ import annotations

import torch

from bijou.eval.policies import draw_noise, sample_noise

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
    another item at realistic corpus sizes (~2x10^7 frames)."""
    max_index = 50_000_000
    assert draw_noise(0, 0, 1, (1,)).ne(draw_noise(0, max_index, 0, (1,))).any()
