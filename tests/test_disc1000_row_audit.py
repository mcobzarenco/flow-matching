"""Oracles for the disc-1000 panel-row wear audit
(fontaine/scripts/disc1000_row_audit.py): the summary MAE semantics,
the box floor, exact inversion/re-wear through q01q99 rows (ascending
AND descending pairs), per-repo row fitting, and the anchor refusal —
pinned on hand-checkable inputs."""

import numpy as np
import pytest

from fontaine.scripts.disc1000_row_audit import (
    audit,
    box_bounds,
    chunk_mae,
    fit_repo_rows,
    floor_error,
    normalize_rows,
    rewear,
    unnormalize_rows,
)


def test_chunk_mae_matches_summarize_semantics() -> None:
    # 2 frames, T=2, D=2; frame 1 has one invalid step. Total abs
    # error over masked steps x dims / (masked steps x dims) — NOT a
    # mean of per-frame means.
    truth = np.zeros((2, 2, 2), np.float32)
    pred = np.array(
        [[[1, 1], [1, 1]], [[4, 4], [9, 9]]],
        np.float32,
    )
    mask = np.array([[True, True], [True, False]])
    # masked elements: 1,1,1,1 (frame 0) + 4,4 (frame 1) = 12 over 6
    assert chunk_mae(pred, truth, mask) == pytest.approx(2.0)


def test_floor_is_distance_to_box_and_zero_inside() -> None:
    lo, hi = box_bounds(
        np.array([-1.0, 5.0], np.float32),  # second pair descending
        np.array([1.0, -5.0], np.float32),
    )
    assert lo.tolist() == [-1.0, -5.0] and hi.tolist() == [1.0, 5.0]
    truth = np.array([[0.5, -7.0], [3.0, 5.0]], np.float32)
    floor = floor_error(truth, lo, hi)
    assert floor.tolist() == [[0.0, 2.0], [2.0, 0.0]]


def test_inversion_exact_inside_box_descending_pair() -> None:
    # A descending pair must round-trip exactly like an ascending one.
    q01 = np.array([10.0, 50.0], np.float32)
    q99 = np.array([20.0, -50.0], np.float32)
    raw = np.array([[12.5, 25.0], [20.0, 50.0]], np.float32)
    norm = normalize_rows(raw, q01, q99)
    assert np.allclose(unnormalize_rows(norm, q01, q99), raw, atol=1e-5)
    # q01 maps to -1, q99 to +1 regardless of order.
    assert normalize_rows(q01[None], q01, q99)[0] == pytest.approx([-1.0, -1.0])
    assert normalize_rows(q99[None], q01, q99)[0] == pytest.approx([1.0, 1.0])


def test_unnormalize_clamps_first() -> None:
    q01 = np.array([0.0], np.float32)
    q99 = np.array([10.0], np.float32)
    out = unnormalize_rows(np.array([[3.0], [-3.0]], np.float32), q01, q99)
    assert out.tolist() == [[10.0], [0.0]]


def test_fit_repo_rows_and_rewear_by_repo() -> None:
    # Repo a: a 0..100 ramp on one joint; repo b the same shifted +1000.
    truth = np.zeros((2, 101, 1), np.float32)
    truth[0, :, 0] = np.linspace(0, 100, 101)
    truth[1, :, 0] = np.linspace(1000, 1100, 101)
    valid = np.ones((2, 101), bool)
    repos = np.array(["a", "b"])
    rows = fit_repo_rows(truth, valid, repos)
    assert rows["a"][0][0] == pytest.approx(1.0)
    assert rows["a"][1][0] == pytest.approx(99.0)
    assert rows["b"][0][0] == pytest.approx(1001.0)
    # Re-wearing norm=0 lands each repo at its own row midpoint.
    mid = rewear(np.zeros((2, 101, 1), np.float32), repos, rows)
    assert mid[0].mean() == pytest.approx(50.0)
    assert mid[1].mean() == pytest.approx(1050.0)


def _tiny_npz(
    pred_norm: np.ndarray,
    worn: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    # One repo, 2 frames x T=2 x D=1; truth sits OUTSIDE the worn box
    # so the floor is nonzero and honest re-wear can reach it.
    truth = np.array([[[100.0], [101.0]], [[102.0], [103.0]]], np.float32)
    pred = unnormalize_rows(pred_norm, *worn)
    return {
        "truth": truth,
        "pred:bijou@1000": pred,
        "pred:state-copy": truth.copy(),
        "core": np.array([True, True]),
        "valid": np.ones((2, 2), bool),
        "repo_id": np.array(["r", "r"]),
    }


def test_audit_decomposition_on_synthetic_crush() -> None:
    worn = (np.array([0.0], np.float32), np.array([10.0], np.float32))
    pred_norm = np.zeros((2, 2, 1), np.float32)  # mid-box everywhere
    npz = _tiny_npz(pred_norm, worn)
    truth = npz["truth"]
    mask = np.ones((2, 2), bool)
    anchors = {
        "bijou@1000": chunk_mae(npz["pred:bijou@1000"], truth, mask),
        "state-copy": 0.0,
    }
    out = audit(
        npz,
        anchors,
        worn[0],
        worn[1],
        np.array([5.0], np.float32),
        np.array([100.0], np.float32),  # released box covers truth
        np.array([104.0], np.float32),
    )
    # All truth outside the worn box; floor = truth - 10.
    assert out["box_audit"]["truth_any_joint_outside_frac"] == 1.0
    assert out["box_audit"]["floor_chunk_mae"] == pytest.approx(91.5)
    # Honest repo rows: q01/q99 of [100..103] ≈ [100.03, 102.97];
    # norm 0 re-wears to the midpoint 101.5 -> MAE 1.0 (per-element
    # |101.5 - t| = 1.5, 0.5, 0.5, 1.5).
    assert out["rewear"]["repo_rows_chunk_mae"] == pytest.approx(1.0, abs=0.05)
    # Released box [100, 104]: norm 0 -> 102, MAE 1.0 exactly.
    assert out["rewear"]["released_table_chunk_mae"] == pytest.approx(1.0)
    # The synthetic prediction IS the mid-box constant, so the
    # midpoint null coincides with the repo-rows read.
    assert out["rewear"]["null_repo_midpoint_chunk_mae"] == pytest.approx(
        out["rewear"]["repo_rows_chunk_mae"],
    )
    # Prediction sits at the worn mid-box: zero edge saturation.
    assert out["box_audit"]["pred_edge_saturation_per_joint"]["shoulder_pan"] == 0.0


def test_audit_refuses_anchor_mismatch() -> None:
    worn = (np.array([0.0], np.float32), np.array([10.0], np.float32))
    npz = _tiny_npz(np.zeros((2, 2, 1), np.float32), worn)
    with pytest.raises(SystemExit, match="anchor mismatch"):
        audit(
            npz,
            {"bijou@1000": 0.123, "state-copy": 0.0},
            worn[0],
            worn[1],
            np.array([5.0], np.float32),
            np.array([100.0], np.float32),
            np.array([104.0], np.float32),
        )
