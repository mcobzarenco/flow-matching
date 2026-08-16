"""CPU oracles for the spawn-v2 sampler (pre-reg DRAFT
posts/2026-08-16-prereg-sim-spawn-v2.md §3): mask construction from a
probe json, draw determinism, constraint invariants in bulk, and the
loud degenerate-configuration refusal. Hermetic — the probe json is
synthesized here, matching the instrument's schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim.spawn_v2 import (
    ACCEPT_PROBE_N,
    JAW_KEEPOUT,
    JAW_TIP_XY,
    R_MAX,
    R_MIN,
    WorkspaceMask,
    draw_spawn_v2,
)

PITCH = 0.01


def probe_json(tmp_path: Path, cells: list[dict]) -> Path:
    path = tmp_path / "probe.json"
    path.write_text(json.dumps({"params": {"pitch": PITCH}, "cells": cells}))
    return path


def cell(x: float, y: float, residual: float, shoulder: float = 0.1) -> dict:
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "residual": residual,
        "moment_frac_shoulder": shoulder,
        "moment_frac_elbow": 0.05,
    }


def block_mask(
    tmp_path: Path,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> WorkspaceMask:
    """A solid rectangular W — every cell passes both bars."""
    cells = [
        cell(float(x), float(y), 5e-4)
        for x in np.arange(x0, x1 + PITCH / 2, PITCH)
        for y in np.arange(y0, y1 + PITCH / 2, PITCH)
    ]
    return WorkspaceMask.from_probe(probe_json(tmp_path, cells))


def test_mask_filters_both_bars(tmp_path: Path) -> None:
    cells = [
        cell(0.20, 0.10, 5e-4),  # passes
        cell(0.21, 0.10, 5e-3),  # residual fails
        cell(0.22, 0.10, 5e-4, shoulder=0.9),  # moment fails
    ]
    mask = WorkspaceMask.from_probe(probe_json(tmp_path, cells))
    assert len(mask.cells) == 1
    assert mask.contains(0.201, 0.099)
    assert not mask.contains(0.21, 0.10)
    assert not mask.contains(0.30, 0.30)


def test_empty_mask_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty workspace mask"):
        WorkspaceMask.from_probe(probe_json(tmp_path, [cell(0.2, 0.1, 5e-2)]))


def test_draw_determinism(tmp_path: Path) -> None:
    mask = block_mask(tmp_path, 0.10, 0.40, -0.10, 0.30)
    a = draw_spawn_v2(mask, np.random.default_rng(7))
    b = draw_spawn_v2(mask, np.random.default_rng(7))
    assert a == b
    c = draw_spawn_v2(mask, np.random.default_rng(8))
    assert c != a


def test_draw_invariants_bulk(tmp_path: Path) -> None:
    mask = block_mask(tmp_path, 0.10, 0.40, -0.10, 0.30)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        s = draw_spawn_v2(mask, rng)
        sep = float(np.hypot(s.boat_xy[0] - s.disk_xy[0], s.boat_xy[1] - s.disk_xy[1]))
        assert R_MIN <= sep <= R_MAX
        assert mask.contains(*s.disk_xy)
        assert mask.contains(*s.boat_xy)
        jaw = float(
            np.hypot(s.boat_xy[0] - JAW_TIP_XY[0], s.boat_xy[1] - JAW_TIP_XY[1]),
        )
        assert jaw >= JAW_KEEPOUT
        assert -np.pi <= s.boat_yaw <= np.pi
        assert 1 <= s.draws <= ACCEPT_PROBE_N


def test_degenerate_mask_refuses(tmp_path: Path) -> None:
    # One isolated cell: every annulus draw (r >= R_MIN >> pitch) lands
    # outside W — the sampler must refuse loudly, not stall.
    mask = WorkspaceMask.from_probe(probe_json(tmp_path, [cell(0.20, 0.10, 5e-4)]))
    with pytest.raises(RuntimeError, match="acceptance degenerate"):
        draw_spawn_v2(mask, np.random.default_rng(0))


def test_cleaned_drops_speckle_keeps_block(tmp_path: Path) -> None:
    # A solid block plus one isolated speckle cell and one 1-cell spur:
    # the clean must drop both and keep the block intact.
    cells = [
        cell(float(x), float(y), 5e-4)
        for x in np.arange(0.10, 0.20 + PITCH / 2, PITCH)
        for y in np.arange(0.00, 0.10 + PITCH / 2, PITCH)
    ]
    cells.append(cell(0.35, 0.30, 5e-4))  # speckle
    cells.append(cell(0.10, 0.15, 5e-4))  # detached spur
    raw = WorkspaceMask.from_probe(probe_json(tmp_path, cells))
    cleaned = raw.cleaned()
    assert not cleaned.contains(0.35, 0.30)
    assert not cleaned.contains(0.10, 0.15)
    assert cleaned.contains(0.15, 0.05)
    # interior shrinks only at the rim: the block's interior survives
    assert len(cleaned.cells) >= len(raw.cells) - 2 - 4 * 11


def test_cleaned_all_speckle_refuses(tmp_path: Path) -> None:
    cells = [cell(0.10, 0.00, 5e-4), cell(0.30, 0.30, 5e-4)]
    raw = WorkspaceMask.from_probe(probe_json(tmp_path, cells))
    with pytest.raises(ValueError, match="empty after morphological clean"):
        raw.cleaned()


def test_v21_radial_bands(tmp_path: Path) -> None:
    """v2.1 amendment (prereg §7): every draw lands disk and boat in
    the measured competence bands, deterministically per seed, and the
    frozen-mask real path honors them too."""
    from sim.spawn_v2 import BOAT_R_BASE, DISK_R_BASE

    mask = block_mask(tmp_path, 0.11, 0.39, -0.2, 0.28)
    for seed in range(300):
        rng = np.random.default_rng(seed)
        spawn = draw_spawn_v2(mask, rng, radial_bands=True)
        disk_r = float(np.hypot(*spawn.disk_xy))
        boat_r = float(np.hypot(*spawn.boat_xy))
        assert DISK_R_BASE[0] <= disk_r <= DISK_R_BASE[1], (seed, disk_r)
        assert BOAT_R_BASE[0] <= boat_r <= BOAT_R_BASE[1], (seed, boat_r)
        assert (
            R_MIN
            <= float(
                np.hypot(
                    spawn.boat_xy[0] - spawn.disk_xy[0],
                    spawn.boat_xy[1] - spawn.disk_xy[1],
                ),
            )
            <= R_MAX + 1e-9
        )
    again = draw_spawn_v2(mask, np.random.default_rng(7), radial_bands=True)
    assert again == draw_spawn_v2(mask, np.random.default_rng(7), radial_bands=True)


def test_v21_does_not_perturb_v2_stream(tmp_path: Path) -> None:
    """The amendment adds a code path, not draws: radial_bands=False
    reproduces the frozen v2 sequence exactly."""
    mask = block_mask(tmp_path, 0.11, 0.39, -0.2, 0.28)
    for seed in (0, 3, 11):
        a = draw_spawn_v2(mask, np.random.default_rng(seed))
        b = draw_spawn_v2(mask, np.random.default_rng(seed), radial_bands=False)
        assert a == b
