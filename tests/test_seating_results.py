"""Oracles for fontaine/scripts/noise_ladder_seating_results.py (the
rung-2 seating paired read).

The script's main() runs against the banked/gitignored npzs; these
tests keep the gate semantics and the paired-read math under check.py
on planted synthetic worlds (anchors injected via the seating_read
parameters). The real top-10 anchor (gate iii vs 5.1847/1.3831) was
verified live against the banked npz before the script froze. Gate
(i) is the AMENDED form (pre-reg Amendment 2): state-copy cells
exact, bijou pooled row within 5e-4, bijou cells within 5e-3.
"""

from __future__ import annotations

import copy
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
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, Any],
]:
    """Two aligned npz-shaped dicts plus seating/banked reports:
    seating errs ~1.0/element, the top-10 arm shifted by top10_offset
    (negative = better). The banked report matches the seating report
    exactly (drift 0); tests perturb copies."""
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
    per_dataset = {
        f"ds{i}": {
            "frames": int((repo == f"ds{i}").sum()),
            "chunk_mae": {"state-copy": 2.0 + i, "bijou@80000": 1.0 + 0.1 * i},
        }
        for i in range(3)
    }
    seating_report = {
        "summaries": [
            {
                "policy": "bijou@80000",
                "chunk_mae": chunk,
                "first_mae": first,
            },
        ],
        "per_dataset": per_dataset,
    }
    banked_report = copy.deepcopy(seating_report)
    return top10, seating, seating_report, banked_report


def anchors(top10: dict[str, np.ndarray]) -> tuple[float, float]:
    mask = nls.element_mask(top10["truth"], top10["valid"])
    core = top10["core"].astype(bool)
    t = nls.core_pool(top10, "pred:bijou@80000_draws10_ticket", mask, core)
    s_chunk = round(t[0], 4)
    return (s_chunk, round(t[1], 4))


def run(
    top10: dict[str, np.ndarray],
    seating: dict[str, np.ndarray],
    seating_report: dict[str, Any],
    banked_report: dict[str, Any],
) -> dict[str, Any]:
    return nls.seating_read(
        top10,
        seating,
        seating_report,
        banked_report,
        banked_top10=anchors(top10),
    )


def test_planted_top10_better_confirms_with_exact_delta() -> None:
    top10, seating, report, banked = world(top10_offset=-0.25)
    out = run(top10, seating, report, banked)
    read = out["read_paired"]
    assert read["delta_pooled"] == pytest.approx(-0.25, abs=1e-4)
    assert read["ci95"][1] < 0
    assert read["verdict"].startswith("CONFIRMED")
    assert out["rows"]["core"] == 10


def test_planted_null_world_not_confirmed() -> None:
    top10, seating, report, banked = world(top10_offset=0.0)
    out = run(top10, seating, report, banked)
    read = out["read_paired"]
    assert read["delta_pooled"] == pytest.approx(0.0, abs=1e-6)
    assert read["verdict"].startswith("NOT-CONFIRMED")


def test_identity_misalignment_aborts() -> None:
    top10, seating, report, banked = world(top10_offset=-0.25)
    seating = dict(seating)
    seating["frame_index"] = seating["frame_index"] + 1
    with pytest.raises(SystemExit, match="identity column"):
        run(top10, seating, report, banked)


def test_base_equality_pooled_drift_beyond_envelope_aborts() -> None:
    top10, seating, report, banked = world(top10_offset=-0.25)
    banked["summaries"][0]["chunk_mae"] += 0.01
    with pytest.raises(SystemExit, match="base-equality FAILED \\(pooled\\)"):
        run(top10, seating, report, banked)


def test_base_equality_within_envelope_passes() -> None:
    # Amendment-2 confirm branch: solver-drift-scale deltas pass —
    # pooled 2e-4 < 5e-4, cells 1e-3 < 5e-3, state-copy untouched.
    top10, seating, report, banked = world(top10_offset=-0.25)
    banked["summaries"][0]["chunk_mae"] += 2e-4
    banked["summaries"][0]["first_mae"] -= 2e-4
    banked["per_dataset"]["ds1"]["chunk_mae"]["bijou@80000"] += 1e-3
    out = run(top10, seating, report, banked)
    assert out["gates"]["base_equality_pooled_delta"][0] == pytest.approx(
        -2e-4,
        abs=1e-9,
    )
    assert out["gates"]["base_equality_cell_delta_max"] == pytest.approx(
        1e-3,
        abs=1e-9,
    )
    assert out["read_paired"]["verdict"].startswith("CONFIRMED")


def test_base_equality_state_copy_cell_drift_aborts() -> None:
    # state-copy is sampling-independent: ANY difference means the
    # rows/truth/pooling changed — never tolerated, however small.
    top10, seating, report, banked = world(top10_offset=-0.25)
    banked["per_dataset"]["ds2"]["chunk_mae"]["state-copy"] += 1e-9
    with pytest.raises(SystemExit, match="state-copy cell"):
        run(top10, seating, report, banked)


def test_base_equality_cell_drift_beyond_envelope_aborts() -> None:
    # A resample/keying fault: pooled row still inside the envelope,
    # but one small cell moves at draw-dispersion scale.
    top10, seating, report, banked = world(top10_offset=-0.25)
    banked["per_dataset"]["ds0"]["chunk_mae"]["bijou@80000"] += 0.05
    with pytest.raises(SystemExit, match="base-equality FAILED \\(cells\\)"):
        run(top10, seating, report, banked)


def test_top10_anchor_mismatch_aborts() -> None:
    top10, seating, report, banked = world(top10_offset=-0.25)
    good = anchors(top10)
    with pytest.raises(SystemExit, match="top-10 anchor FAILED"):
        nls.seating_read(
            top10,
            seating,
            report,
            banked,
            banked_top10=(good[0] + 0.01, good[1]),
        )


def test_frozen_banked_constants_unchanged() -> None:
    assert nls.BANKED_MEAN10_CHUNK == 5.3645
    assert nls.BANKED_MEAN10_FIRST == 1.4242
    assert nls.BANKED_TOP10_CHUNK == 5.1847
    assert nls.BANKED_TOP10_FIRST == 1.3831
    assert nls.BASE_TOL_POOLED == 5e-4
    assert nls.BASE_TOL_CELL == 5e-3
