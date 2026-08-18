"""Oracles for the pdnorm endpoint estimator-consistency cross-check
(fontaine/scripts/pdnorm_endpoint_truthfit_rewear.py): the PER-REPO
inversion identity (each repo round-trips through its OWN native row;
swapped rows refuse loudly), the truth-fit re-expression on a
hand-checkable case, degenerate-span handling, the native-row loader,
and the guard refusals (scheme, anchors, missing rows)."""

import json
from pathlib import Path

import numpy as np
import pytest

from fontaine.scripts.disc1000_row_audit import chunk_mae, unnormalize_rows
from fontaine.scripts.pdnorm_endpoint_truthfit_rewear import (
    invert_native,
    load_native_rows,
    require_per_dataset_scheme,
    truthfit_rewear,
)

PRED_KEY = "pred:bijou@3000"

# Two repos with DIFFERENT native rows: a global-table inversion
# cannot round-trip both.
ROW_A = (np.array([0.0], np.float32), np.array([10.0], np.float32))
ROW_B = (np.array([20.0], np.float32), np.array([40.0], np.float32))


def _two_repo_npz(
    norm_a: np.ndarray,
    norm_b: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    # Repo a: truth 100..103 (outside its native box); repo b: truth
    # 200..203. Predictions are worn through each repo's OWN row —
    # exactly what the per-dataset contract path serves.
    truth = np.array(
        [
            [[100.0], [101.0]],
            [[102.0], [103.0]],
            [[200.0], [201.0]],
            [[202.0], [203.0]],
        ],
        np.float32,
    )
    pred = np.concatenate(
        [unnormalize_rows(norm_a, *ROW_A), unnormalize_rows(norm_b, *ROW_B)],
    )
    npz = {
        "truth": truth,
        PRED_KEY: pred,
        "pred:state-copy": truth.copy(),
        "core": np.array([True, True, True, True]),
        "valid": np.ones((4, 2), bool),
        "repo_id": np.array(["a", "a", "b", "b"]),
    }
    return npz, {"a": ROW_A, "b": ROW_B}


def _run(npz: dict[str, np.ndarray], rows: dict) -> dict:
    mask = npz["core"][:, None] & npz["valid"]
    anchors = {
        PRED_KEY: chunk_mae(npz[PRED_KEY], npz["truth"], mask),
        "state-copy": 0.0,
    }
    return truthfit_rewear(npz, anchors, rows, PRED_KEY)


def test_per_repo_identity_and_truthfit_on_mid_box_constants() -> None:
    # Mid-box constants in both repos: norm 0 must invert back to 0
    # per repo and re-wear to each repo's truth midpoint (MAE 1.0).
    npz, rows = _two_repo_npz(
        np.zeros((2, 2, 1), np.float32),
        np.zeros((2, 2, 1), np.float32),
    )
    out = _run(npz, rows)
    assert out["round_trip_worst_abs_deg"] == pytest.approx(0.0, abs=1e-5)
    assert out["truthfit_wear_chunk_mae"] == pytest.approx(1.0, abs=0.05)
    # The constant mid-box prediction IS the midpoint null.
    assert out["null_repo_midpoint_chunk_mae"] == pytest.approx(
        out["truthfit_wear_chunk_mae"],
    )
    # Native rows: repo a decodes 5.0 vs 100..103 (MAE 96.5), repo b
    # 30.0 vs 200..203 (MAE 171.5) -> pooled 134.0; the seam delta is
    # the whole gap down to 1.0.
    assert out["native_wear_chunk_mae"] == pytest.approx(134.0)
    assert out["estimator_seam_delta_native_minus_truthfit"] == pytest.approx(
        133.0,
        abs=0.05,
    )
    assert out["n_repo_rows"] == 2
    assert out["degenerate_joints"] == []
    assert {r["repo"] for r in out["per_repo_worst10_by_abs_delta"]} == {"a", "b"}


def test_swapped_rows_fail_the_per_repo_identity() -> None:
    # The same predictions inverted through the OTHER repo's row (a
    # global/wrong-source table) must trip the identity oracle: repo
    # a's ramp decodes to 0..10, ROW_B's clamp pins it at 20.
    npz, _ = _two_repo_npz(
        np.array([[[-1.0], [-0.5]], [[0.5], [1.0]]], np.float32),
        np.zeros((2, 2, 1), np.float32),
    )
    mask = npz["core"][:, None] & npz["valid"]
    with pytest.raises(SystemExit, match="inversion identity failed for 'a'"):
        invert_native(npz[PRED_KEY], npz["repo_id"], {"a": ROW_B, "b": ROW_B}, mask)


def test_degenerate_span_pins_midpoint_and_bounds_the_constant() -> None:
    # Joint 0 healthy, joint 1 degenerate (span 0 at 7.0). A prediction
    # AT the constant inverts to norm 0 and is recorded; one off the
    # constant refuses.
    q01 = np.array([0.0, 7.0], np.float32)
    q99 = np.array([10.0, 7.0], np.float32)
    pred = np.array([[[5.0, 7.0], [2.5, 7.0]]], np.float32)
    repo_ids = np.array(["r"])
    mask = np.ones((1, 2), bool)
    norm, facts = invert_native(pred, repo_ids, {"r": (q01, q99)}, mask)
    assert norm[..., 1] == pytest.approx(0.0)
    assert norm[0, 0, 0] == pytest.approx(0.0)  # 5.0 is mid-box
    assert facts["degenerate_joints"] == [
        {"repo": "r", "joint": "shoulder_lift", "worst_abs_off_constant_deg": 0.0},
    ]
    off = pred.copy()
    off[..., 1] = 9.0
    with pytest.raises(SystemExit, match="degenerate"):
        invert_native(off, repo_ids, {"r": (q01, q99)}, mask)


def test_missing_native_row_refused() -> None:
    npz, rows = _two_repo_npz(
        np.zeros((2, 2, 1), np.float32),
        np.zeros((2, 2, 1), np.float32),
    )
    del rows["b"]
    with pytest.raises(SystemExit, match="lack native rows"):
        _run(npz, rows)


def test_anchor_mismatch_refused() -> None:
    npz, rows = _two_repo_npz(
        np.zeros((2, 2, 1), np.float32),
        np.zeros((2, 2, 1), np.float32),
    )
    with pytest.raises(SystemExit, match="anchor mismatch"):
        truthfit_rewear(
            npz,
            {PRED_KEY: 0.123, "state-copy": 0.0},
            rows,
            PRED_KEY,
        )


def test_scheme_guard_refuses_global_table_checkpoints() -> None:
    def metadata(tag: str) -> dict:
        return {
            "components": {"flow_decoder": {"config": {"normalization": tag}}},
        }

    require_per_dataset_scheme(metadata("q01q99_per_dataset"))
    with pytest.raises(SystemExit, match="wore a global table"):
        require_per_dataset_scheme(metadata("q01q99"))


def test_load_native_rows(tmp_path: Path) -> None:
    meta = tmp_path / "org" / "repo" / "meta"
    meta.mkdir(parents=True)
    (meta / "stats.json").write_text(
        json.dumps({"action": {"q01": [0.0, 7.0], "q99": [10.0, 7.0]}}),
    )
    rows = load_native_rows(tmp_path, ["org/repo"])
    np.testing.assert_array_equal(rows["org/repo"][0], [0.0, 7.0])
    np.testing.assert_array_equal(rows["org/repo"][1], [10.0, 7.0])
    with pytest.raises(SystemExit, match="no native stats"):
        load_native_rows(tmp_path, ["org/other"])
    (meta / "stats.json").write_text(json.dumps({"action": {"q01": [0.0]}}))
    with pytest.raises(SystemExit, match="lacks action q01/q99"):
        load_native_rows(tmp_path, ["org/repo"])
