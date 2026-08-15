"""Oracles for build_corrected_norm_stats.corrected_tag — the table
that feeds ``convert_molmoact2 --norm-stats-from`` for the corrected
retrain (quantile class bug, 2026-08-15)."""

import math

import pytest

from fontaine.scripts.build_corrected_norm_stats import corrected_tag


def donor_tag() -> dict:
    return {
        "setup_type": "single so100/so101 robotic arm in molmoact2",
        "control_mode": "absolute joint pose",
        "action_dim": 32,
        "action_stats": {
            "min": [-1.0, -2.0],
            "max": [1.0, 2.0],
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
            "count": [999],
            "q01": [-0.9, 35.5],  # dim-1 = the corrupt wrist_roll-style row
            "q99": [0.9, 94.4],
            "names": ["a.pos", "b.pos"],
            "mask": [True, True],
        },
        "state_stats": {
            "min": [-1.0, -2.0],
            "max": [1.0, 2.0],
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
            "count": [999],
            "q01": [-0.9, 30.0],
            "q99": [0.9, 90.0],
            "names": ["a.pos", "b.pos"],
            "mask": [True, True],
        },
    }


def dataset_stats() -> dict:
    return {
        "action": {
            "min": [-3.0, -160.0],
            "max": [3.0, 160.0],
            "mean": [0.1, -1.0],
            "std": [1.2, 80.0],
            "count": [54101],
            "q01": [-2.5, -157.2],
            "q99": [2.5, 157.2],
        },
        "observation.state": {
            "min": [-3.0, -160.0],
            "max": [3.0, 160.0],
            "mean": [0.1, -1.0],
            "std": [1.2, 80.0],
            "count": [54101],
            "q01": [-2.4, -157.0],
            "q99": [2.4, 157.0],
        },
    }


def test_replaces_stats_rows_exactly_and_preserves_metadata() -> None:
    tag = donor_tag()
    out, replaced = corrected_tag(tag, dataset_stats(), tag_name="t")
    # every stats row now carries the dataset's exact values
    assert out["action_stats"]["q01"] == [-2.5, -157.2]
    assert out["action_stats"]["q99"] == [2.5, 157.2]
    assert out["action_stats"]["count"] == [54101]
    assert out["state_stats"]["q01"] == [-2.4, -157.0]
    assert out["state_stats"]["std"] == [1.2, 80.0]
    # identity/metadata untouched
    assert out["action_stats"]["names"] == ["a.pos", "b.pos"]
    assert out["action_stats"]["mask"] == [True, True]
    assert out["setup_type"] == tag["setup_type"]
    assert out["control_mode"] == tag["control_mode"]
    # donor dict not mutated (deep copy)
    assert tag["action_stats"]["q01"] == [-0.9, 35.5]
    # 7 stats rows per block, both blocks
    assert len(replaced) == 14
    assert "action_stats.q01" in replaced and "state_stats.count" in replaced


def test_refuses_dim_mismatch() -> None:
    ds = dataset_stats()
    ds["action"]["q01"] = [-2.5]
    with pytest.raises(SystemExit, match="dim 1"):
        corrected_tag(donor_tag(), ds, tag_name="t")


def test_refuses_missing_row() -> None:
    ds = dataset_stats()
    del ds["action"]["q99"]
    with pytest.raises(SystemExit, match="lacks row 'q99'"):
        corrected_tag(donor_tag(), ds, tag_name="t")


def test_refuses_non_finite() -> None:
    ds = dataset_stats()
    ds["observation.state"]["std"] = [1.0, math.nan]
    with pytest.raises(SystemExit, match="non-finite"):
        corrected_tag(donor_tag(), ds, tag_name="t")


def test_refuses_inverted_quantiles() -> None:
    ds = dataset_stats()
    ds["action"]["q01"], ds["action"]["q99"] = ds["action"]["q99"], ds["action"]["q01"]
    with pytest.raises(SystemExit, match="q01 > q99"):
        corrected_tag(donor_tag(), ds, tag_name="t")
