"""Oracles for fontaine/scripts/noise_ladder_seating_results.py (the
rung-2 seating paired read).

The script's main() runs against the banked/gitignored npzs; these
tests keep the gate semantics and the paired-read math under check.py
on planted synthetic worlds (anchors injected via the seating_read
parameters). The real top-10 anchor (gate iii vs 5.1847/1.3831) was
verified live against the banked npz before the script froze.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "fontaine"
    / "scripts"
    / "noise_ladder_seating_results.py"
)
spec = importlib.util.spec_from_file_location("noise_ladder_seating_results", SCRIPT)
assert spec is not None and spec.loader is not None
nls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nls)

FRAMES, CHUNK, DIMS = 12, 4, 3


def world(
    top10_offset: float,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Two aligned npz-shaped dicts: seating errs ~1.0/element, the
    top-10 arm shifted by top10_offset (negative = better)."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(FRAMES, CHUNK, DIMS))
    valid = np.ones((FRAMES, CHUNK), dtype=bool)
    core = np.array([True] * 10 + [False] * 2)
    repo = np.array([f"ds{i % 3}" for i in range(FRAMES)])
    identity = {
        "index": np.arange(FRAMES),
        "repo_id": repo,
        "episode_index": np.zeros(FRAMES, dtype=int),
        "frame_index": np.arange(FRAMES),
        "core": core,
        "truth": truth,
        "valid": valid,
    }
    seating = dict(identity)
    seating["pred:bijou@80000_seating"] = truth + 1.0
    top10 = dict(identity)
    top10["pred:bijou@80000_draws10_ticket"] = truth + 1.0 + top10_offset
    chunk, first = nls.core_pool(
        seating,
        "pred:bijou@80000_seating",
        nls.element_mask(truth, valid),
        core,
    )
    report = {
        "summaries": [
            {
                "policy": "bijou@80000",
                "chunk_mae": chunk,
                "first_mae": first,
            },
        ],
    }
    return top10, seating, report


def anchors(top10: dict[str, np.ndarray]) -> tuple[float, float]:
    mask = nls.element_mask(top10["truth"], top10["valid"])
    core = top10["core"].astype(bool)
    t = nls.core_pool(top10, "pred:bijou@80000_draws10_ticket", mask, core)
    s_chunk = round(t[0], 4)
    return (s_chunk, round(t[1], 4))


def run(
    top10: dict[str, np.ndarray],
    seating: dict[str, np.ndarray],
    report: dict[str, Any],
) -> dict[str, Any]:
    base = (
        round(report["summaries"][0]["chunk_mae"], 4),
        round(report["summaries"][0]["first_mae"], 4),
    )
    return nls.seating_read(
        top10,
        seating,
        report,
        banked_mean10=base,
        banked_top10=anchors(top10),
    )


def test_planted_top10_better_confirms_with_exact_delta() -> None:
    top10, seating, report = world(top10_offset=-0.25)
    out = run(top10, seating, report)
    read = out["read_paired"]
    assert read["delta_pooled"] == pytest.approx(-0.25, abs=1e-4)
    assert read["ci95"][1] < 0
    assert read["verdict"].startswith("CONFIRMED")
    assert out["rows"]["core"] == 10


def test_planted_null_world_not_confirmed() -> None:
    top10, seating, report = world(top10_offset=0.0)
    out = run(top10, seating, report)
    read = out["read_paired"]
    assert read["delta_pooled"] == pytest.approx(0.0, abs=1e-6)
    assert read["verdict"].startswith("NOT-CONFIRMED")


def test_identity_misalignment_aborts() -> None:
    top10, seating, report = world(top10_offset=-0.25)
    seating = dict(seating)
    seating["frame_index"] = seating["frame_index"] + 1
    with pytest.raises(SystemExit, match="identity column"):
        run(top10, seating, report)


def test_base_equality_mismatch_aborts() -> None:
    top10, seating, report = world(top10_offset=-0.25)
    report["summaries"][0]["chunk_mae"] += 0.01
    base = (
        round(report["summaries"][0]["chunk_mae"] - 0.01, 4),
        round(report["summaries"][0]["first_mae"], 4),
    )
    with pytest.raises(SystemExit, match="base-equality FAILED"):
        nls.seating_read(
            top10,
            seating,
            report,
            banked_mean10=base,
            banked_top10=anchors(top10),
        )


def test_top10_anchor_mismatch_aborts() -> None:
    top10, seating, report = world(top10_offset=-0.25)
    good = anchors(top10)
    with pytest.raises(SystemExit, match="top-10 anchor FAILED"):
        nls.seating_read(
            top10,
            seating,
            report,
            banked_mean10=(
                round(report["summaries"][0]["chunk_mae"], 4),
                round(report["summaries"][0]["first_mae"], 4),
            ),
            banked_top10=(good[0] + 0.01, good[1]),
        )


def test_frozen_banked_constants_unchanged() -> None:
    assert nls.BANKED_MEAN10_CHUNK == 5.3645
    assert nls.BANKED_MEAN10_FIRST == 1.4242
    assert nls.BANKED_TOP10_CHUNK == 5.1847
    assert nls.BANKED_TOP10_FIRST == 1.3831
