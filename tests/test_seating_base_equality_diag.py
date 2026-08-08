"""Oracles for fontaine/scripts/seating_base_equality_diag.py (the
Amendment-2 drift-vs-keying adjudication, now also the seating
launcher's base-equality oracle)."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "fontaine"
    / "scripts"
    / "seating_base_equality_diag.py"
)
spec = importlib.util.spec_from_file_location("seating_base_equality_diag", SCRIPT)
assert spec is not None and spec.loader is not None
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def reports() -> tuple[dict[str, Any], dict[str, Any]]:
    banked: dict[str, Any] = {
        "summaries": [
            {
                "policy": diag.POLICY,
                "chunk_mae": 5.3645479,
                "first_mae": 1.4242034,
                "chunk_mse": 131.05445,
                "mae_p50": 3.9600286,
                "mae_p90": 10.431143,
                "per_motor_mae": [1.0, 2.0, 3.0],
            },
        ],
        "per_dataset": {
            f"ds{i}": {
                "frames": 4 * (i + 1),
                "chunk_mae": {"state-copy": 2.0 + i, diag.POLICY: 3.0 + i},
            }
            for i in range(120)
        },
    }
    seating = copy.deepcopy(banked)
    # solver-drift-scale perturbations: pooled ~1e-4, one cell 1e-3
    seating["summaries"][0]["chunk_mae"] -= 8.6e-5
    seating["summaries"][0]["first_mae"] -= 1.27e-4
    seating["per_dataset"]["ds3"]["chunk_mae"][diag.POLICY] += 1e-3
    return banked, seating


def test_drift_world_passes_with_verdict() -> None:
    banked, seating = reports()
    out = diag.diagnose(banked, seating)
    assert out["verdict"].startswith("BENIGN NUMERIC DRIFT")
    assert out["per_dataset"]["state_copy_abs_delta_max"] == 0.0
    assert out["per_dataset"]["abs_delta_max"] == pytest.approx(1e-3, abs=1e-9)
    assert out["pooled"]["first_mae"]["delta"] == pytest.approx(-1.27e-4, abs=1e-9)


def test_state_copy_drift_aborts() -> None:
    banked, seating = reports()
    seating["per_dataset"]["ds7"]["chunk_mae"]["state-copy"] += 1e-9
    with pytest.raises(SystemExit, match="state-copy cells differ"):
        diag.diagnose(banked, seating)


def test_pooled_drift_beyond_envelope_aborts() -> None:
    banked, seating = reports()
    seating["summaries"][0]["first_mae"] += 1e-3
    with pytest.raises(SystemExit, match="drift envelope"):
        diag.diagnose(banked, seating)


def test_resample_scale_cell_aborts() -> None:
    banked, seating = reports()
    seating["per_dataset"]["ds5"]["chunk_mae"][diag.POLICY] += 0.05
    with pytest.raises(SystemExit, match="resampled noise"):
        diag.diagnose(banked, seating)


def test_frame_count_mismatch_aborts() -> None:
    banked, seating = reports()
    seating["per_dataset"]["ds5"]["frames"] += 1
    with pytest.raises(SystemExit, match="frame count differs"):
        diag.diagnose(banked, seating)


def test_bounds_frozen() -> None:
    assert diag.TOL_POOLED == 5e-4
    assert diag.TOL_CELL == 5e-3
