"""Rollout pre-flight safety gates (bijou.rollout_safety).

Pure CPU: no robot, no model, no lerobot imports — the gates are
plain functions over stats/paths, tested here exactly as rollout
calls them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bijou.data import DatasetStats
from bijou.rollout_safety import (
    envelope_violations,
    require_clamp,
    resolve_camera_kinds,
    state_envelope,
)

MEAN = (0.0, -10.0, 20.0, 5.0, 0.0, 50.0)
STD = (10.0,) * 6
CAMERAS = ["top", "wrist"]


def make_stats(*, quantiles: bool = True) -> DatasetStats:
    return DatasetStats(
        action_mean=MEAN,
        action_std=STD,
        state_mean=MEAN,
        state_std=STD,
        action_q01=tuple(m - 20.0 for m in MEAN) if quantiles else None,
        action_q99=tuple(m + 20.0 for m in MEAN) if quantiles else None,
        state_q01=tuple(m - 20.0 for m in MEAN) if quantiles else None,
        state_q99=tuple(m + 20.0 for m in MEAN) if quantiles else None,
    )


def write_rig_dataset(
    tmp_path: Path,
    *,
    stamp_hash: str | None = "abc123",
    kinds_hash: str = "abc123",
    cameras: dict[str, str] | None = None,
) -> Path:
    dataset = tmp_path / "rig_dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    if stamp_hash is not None:
        (meta / "judge_annotations.json").write_text(
            json.dumps({"prompt_hash": stamp_hash, "model_filter": "judge-x"}),
        )
    (meta / "camera_kinds.json").write_text(
        json.dumps(
            {
                "prompt_hash": kinds_hash,
                "cameras": {
                    name: {"kind": kind}
                    for name, kind in (
                        cameras or {"front": "top", "wrist": "wrist"}
                    ).items()
                },
            },
        ),
    )
    return dataset


# --- clamp gate ---


def test_clamp_missing_refuses() -> None:
    with pytest.raises(SystemExit, match="max-relative-target"):
        require_clamp(None, unclamped=False)


def test_clamp_explicit_unclamped_proceeds(capsys: pytest.CaptureFixture) -> None:
    require_clamp(None, unclamped=True)
    assert "UNCLAMPED" in capsys.readouterr().out


def test_clamp_valid_value_proceeds() -> None:
    require_clamp(20.0, unclamped=False)


@pytest.mark.parametrize("value", [0.0, -5.0, math.inf, math.nan])
def test_clamp_nonpositive_or_nonfinite_refuses(value: float) -> None:
    with pytest.raises(SystemExit, match="positive"):
        require_clamp(value, unclamped=False)


def test_clamp_with_unclamped_is_contradictory() -> None:
    with pytest.raises(SystemExit, match="contradictory"):
        require_clamp(20.0, unclamped=True)


# --- first-observation envelope ---


def test_envelope_from_quantiles_widens_half_band() -> None:
    # q band is mean±20 (width 40) → pad max(20, 15) = 20 → mean±40.
    lo, hi = state_envelope(make_stats(), expected_dim=6)
    assert lo == tuple(m - 40.0 for m in MEAN)
    assert hi == tuple(m + 40.0 for m in MEAN)


def test_envelope_fallback_uses_mean_std() -> None:
    # No quantiles: band mean±3σ = ±30 (width 60) → pad 30 → mean±60.
    lo, hi = state_envelope(make_stats(quantiles=False), expected_dim=6)
    assert lo == tuple(m - 60.0 for m in MEAN)
    assert hi == tuple(m + 60.0 for m in MEAN)


def test_envelope_floor_binds_on_narrow_band() -> None:
    narrow = DatasetStats(
        action_mean=MEAN,
        action_std=STD,
        state_mean=MEAN,
        state_std=(0.1,) * 6,
        action_q01=None,
        action_q99=None,
        state_q01=None,
        state_q99=None,
    )
    lo, hi = state_envelope(narrow, expected_dim=6)
    # band mean±0.3 → pad floor 15 binds.
    assert lo == pytest.approx(tuple(m - 15.3 for m in MEAN))
    assert hi == pytest.approx(tuple(m + 15.3 for m in MEAN))


def test_envelope_dim_mismatch_refuses() -> None:
    with pytest.raises(SystemExit, match="dimension"):
        state_envelope(make_stats(), expected_dim=7)


def test_violations_inside_and_boundary_pass() -> None:
    envelope = state_envelope(make_stats(), expected_dim=6)
    assert envelope_violations(MEAN, envelope) == []
    # Bounds are inclusive.
    assert envelope_violations(envelope[0], envelope) == []
    assert envelope_violations(envelope[1], envelope) == []


def test_violations_catch_ticks_scale_state() -> None:
    # Raw servo ticks (~2048) instead of degrees: every joint flags.
    envelope = state_envelope(make_stats(), expected_dim=6)
    assert envelope_violations((2048.0,) * 6, envelope) == list(range(6))


def test_violations_catch_single_joint_and_nan() -> None:
    envelope = state_envelope(make_stats(), expected_dim=6)
    state = list(MEAN)
    state[2] = MEAN[2] + 40.1
    assert envelope_violations(state, envelope) == [2]
    state[2] = math.nan
    assert envelope_violations(state, envelope) == [2]


# --- camera kinds (--camera keys ARE kinds; asserted, cross-checked) ---


def test_kinds_off_vocabulary_key_is_refused() -> None:
    with pytest.raises(SystemExit, match="vocabulary"):
        resolve_camera_kinds(["overhead", "wrist"], None)


def test_kinds_duplicate_key_is_refused() -> None:
    with pytest.raises(SystemExit, match="more than once"):
        resolve_camera_kinds(["wrist", "wrist"], None)


def test_kinds_asserted_without_dataset_are_silent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    kinds = resolve_camera_kinds(CAMERAS, None)
    assert kinds == {"top": "top", "wrist": "wrist"}
    assert capsys.readouterr().out == ""


def test_kinds_matching_judged_kinds_are_silent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The documented wild case: the rig's scene camera is NAMED front
    # but JUDGED top — kind-keyed cameras match the judgment by VALUE,
    # so the operator's top=/wrist= agrees with front→top/wrist→wrist.
    dataset = write_rig_dataset(tmp_path)
    kinds = resolve_camera_kinds(CAMERAS, dataset)
    assert kinds == {"top": "top", "wrist": "wrist"}
    assert capsys.readouterr().out == ""


def test_kinds_mismatch_warns_and_keeps_asserted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Asserted kinds always win; the warning names BOTH the asserted
    kinds the dataset never judged and the judged kinds no camera
    covers (the operator decides, informed)."""
    dataset = write_rig_dataset(tmp_path)
    kinds = resolve_camera_kinds(["front", "wrist"], dataset)
    assert kinds == {"front": "front", "wrist": "wrist"}
    warning = capsys.readouterr().out
    assert "WARNING" in warning
    assert "['front']" in warning  # asserted but never judged
    assert "['top']" in warning  # judged but uncovered


def test_kinds_hash_mismatch_counts_as_unstamped(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A kinds file whose hash mismatches the stamp was not what
    # training rendered — same as no stamp: warn against 'unknown'.
    dataset = write_rig_dataset(tmp_path, kinds_hash="other")
    kinds = resolve_camera_kinds(CAMERAS, dataset)
    assert kinds == {"top": "top", "wrist": "wrist"}
    assert "no usable stamped kinds" in capsys.readouterr().out


def test_kinds_unstamped_dataset_warns_only_for_non_unknown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = write_rig_dataset(tmp_path, stamp_hash=None)
    kinds = resolve_camera_kinds(CAMERAS, dataset)
    assert kinds == {"top": "top", "wrist": "wrist"}
    assert "no usable stamped kinds" in capsys.readouterr().out
    # Asserting 'unknown' mirrors what unstamped training rendered —
    # nothing to warn about.
    kinds = resolve_camera_kinds(["unknown"], dataset)
    assert kinds == {"unknown": "unknown"}
    assert capsys.readouterr().out == ""


def test_rollout_noise_keying_is_index_not_stable() -> None:
    """Live-rig observations carry no dataset identity (repo_id/episode/
    frame), so stable noise keying cannot apply — bijou.rollout must pin
    noise_key="index" (fresh noise per replan via the replan counter).
    Regression: the #18.2 stable default flip crashed every flow rollout
    at the first predict (KeyError: 'repo_id'), caught by --check."""
    import inspect

    from bijou import rollout
    from bijou.eval.policies import noise_for_item

    rig_item = {"state": [0.0] * 6}  # no repo_id/episode_index/frame_index
    noise = noise_for_item("index", 0, rig_item, 3, 0, (50, 6))
    assert noise.shape == (50, 6)
    with pytest.raises(KeyError):
        noise_for_item("stable", 0, rig_item, 3, 0, (50, 6))
    assert 'noise_key="index"' in inspect.getsource(rollout)


def test_home_trajectory_eases_and_lands_exactly() -> None:
    """Return-home glide: ends EXACTLY at home, cosine-eased (boundary
    steps smaller than mid-glide steps), tick count = seconds x fps."""
    from bijou.rollout_safety import home_trajectory

    current, home = [0.0, 10.0, -20.0], [30.0, 10.0, 40.0]
    rows = home_trajectory(current, home, seconds=1.5, fps=30.0)
    assert len(rows) == 45
    assert rows[-1] == pytest.approx(home)
    # Monotone approach on the moving joints.
    gaps = [abs(row[0] - home[0]) for row in rows]
    assert gaps == sorted(gaps, reverse=True)
    # Ease-in/out: first and last steps far smaller than the largest.
    steps = [abs(b[0] - a[0]) for a, b in zip([current, *rows[:-1]], rows, strict=True)]
    assert steps[0] < max(steps) / 3
    assert steps[-1] < max(steps) / 3
    # The stationary joint never wobbles.
    assert all(row[1] == pytest.approx(10.0) for row in rows)
    # Degenerate short glide still lands home.
    assert home_trajectory([0.0], [1.0], seconds=0.01, fps=30.0)[-1] == [1.0]
