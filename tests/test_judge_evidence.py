"""Pure-CPU tests for length-adaptive evidence sizing."""

from __future__ import annotations

from itertools import pairwise

from bijou.judge.evidence import (
    MAX_TIMESTEPS,
    MIN_TIMESTEPS,
    adaptive_num_timesteps,
)


def test_adaptive_clips_and_scales() -> None:
    # 2s episode (shortest the mechanical filter admits is ~1.7s): floor.
    assert adaptive_num_timesteps(60, 30.0) == MIN_TIMESTEPS
    # 15s ~ corpus median: one per 1.5s = 10, the old fixed default.
    assert adaptive_num_timesteps(450, 30.0) == 10
    # 100s marathon: cap, not 67.
    assert adaptive_num_timesteps(3000, 30.0) == MAX_TIMESTEPS
    # fps-aware: the same wall-clock duration sizes identically at 10 fps.
    assert adaptive_num_timesteps(150, 10.0) == adaptive_num_timesteps(450, 30.0)


def test_adaptive_monotone_in_duration() -> None:
    counts = [adaptive_num_timesteps(frames, 30.0) for frames in range(50, 4000, 25)]
    assert all(a <= b for a, b in pairwise(counts))
    assert min(counts) == MIN_TIMESTEPS
    assert max(counts) == MAX_TIMESTEPS
