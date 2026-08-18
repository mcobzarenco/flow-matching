"""Oracles for the released-row honest-wear re-expression
(fontaine/scripts/released_row_rewear.py): the re-expression identity
(re-wearing the worn table itself reproduces the raw row), the honest
re-wear on a hand-checkable crush, the anchor refusal, and the
same-panel midpoint-null identity anchor."""

import numpy as np
import pytest

from fontaine.scripts.disc1000_row_audit import chunk_mae, unnormalize_rows
from fontaine.scripts.released_row_rewear import (
    check_same_panel_null,
    honest_rewear,
)

PRED_KEY = "pred:bijou@released"


def _tiny_npz(
    pred_norm: np.ndarray,
    worn: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    # One repo, 2 frames x T=2 x D=1; truth sits OUTSIDE the worn box
    # so honest re-wear moves the row and the identity stays testable.
    truth = np.array([[[100.0], [101.0]], [[102.0], [103.0]]], np.float32)
    return {
        "truth": truth,
        PRED_KEY: unnormalize_rows(pred_norm, *worn),
        "pred:state-copy": truth.copy(),
        "core": np.array([True, True]),
        "valid": np.ones((2, 2), bool),
        "repo_id": np.array(["r", "r"]),
    }


def _run(pred_norm: np.ndarray, worn: tuple[np.ndarray, np.ndarray]) -> dict:
    npz = _tiny_npz(pred_norm, worn)
    mask = np.ones((2, 2), bool)
    anchors = {
        PRED_KEY: chunk_mae(npz[PRED_KEY], npz["truth"], mask),
        "state-copy": 0.0,
    }
    return honest_rewear(npz, anchors, worn[0], worn[1], PRED_KEY)


def test_rewear_identity_and_honest_row_on_crush() -> None:
    worn = (np.array([0.0], np.float32), np.array([10.0], np.float32))
    out = _run(np.zeros((2, 2, 1), np.float32), worn)  # mid-box constant
    # Identity: inverting through the worn table and re-wearing it
    # reproduces the raw predictions exactly (inside the clamp).
    assert out["round_trip_worst_abs_deg"] == pytest.approx(0.0, abs=1e-5)
    # Anchor reproduced: constant 5.0 vs truth 100..103 -> MAE 96.5.
    assert out["anchors_reproduced"][PRED_KEY] == pytest.approx(96.5)
    # Honest rows on [100..103]: norm 0 lands at the repo midpoint
    # 101.5 -> MAE 1.0 (|101.5 - t| = 1.5, 0.5, 0.5, 1.5).
    assert out["honest_wear_chunk_mae"] == pytest.approx(1.0, abs=0.05)
    # The constant mid-box prediction IS the midpoint null.
    assert out["null_repo_midpoint_chunk_mae"] == pytest.approx(
        out["honest_wear_chunk_mae"],
    )
    assert out["n_repo_rows"] == 1
    assert out["per_repo_worst10"][0]["repo"] == "r"


def test_non_constant_prediction_keeps_shape_under_rewear() -> None:
    worn = (np.array([0.0], np.float32), np.array([10.0], np.float32))
    # Normalized ramp -1, -0.5, 0.5, 1 -> honest rows map it across the
    # repo box; the re-worn row must beat the crushed own-table row.
    ramp = np.array([[[-1.0], [-0.5]], [[0.5], [1.0]]], np.float32)
    out = _run(ramp, worn)
    assert out["honest_wear_chunk_mae"] < out["anchors_reproduced"][PRED_KEY]


def test_refuses_anchor_mismatch() -> None:
    worn = (np.array([0.0], np.float32), np.array([10.0], np.float32))
    npz = _tiny_npz(np.zeros((2, 2, 1), np.float32), worn)
    with pytest.raises(SystemExit, match="anchor mismatch"):
        honest_rewear(
            npz,
            {PRED_KEY: 0.123, "state-copy": 0.0},
            worn[0],
            worn[1],
            PRED_KEY,
        )


def test_null_identity_anchor() -> None:
    audit = {"rewear": {"null_repo_midpoint_chunk_mae": 25.154476}}
    check_same_panel_null(25.154476, audit)  # exact: fine
    with pytest.raises(SystemExit, match="identity anchor failed"):
        check_same_panel_null(25.16, audit)
