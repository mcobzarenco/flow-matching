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
    parse_camera_kind_overrides,
    require_clamp,
    resolve_camera_kinds,
    state_envelope,
)

MEAN = (0.0, -10.0, 20.0, 5.0, 0.0, 50.0)
STD = (10.0,) * 6
CAMERAS = ["front", "wrist"]


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


# --- camera kinds ---


def test_overrides_parse_and_validate() -> None:
    assert parse_camera_kind_overrides(["front=top"], CAMERAS) == {"front": "top"}
    with pytest.raises(SystemExit, match="NAME=KIND"):
        parse_camera_kind_overrides(["front"], CAMERAS)
    with pytest.raises(SystemExit, match="not a --camera name"):
        parse_camera_kind_overrides(["overhead=top"], CAMERAS)
    with pytest.raises(SystemExit, match="vocabulary"):
        parse_camera_kind_overrides(["front=ceiling"], CAMERAS)


def test_kinds_mirror_stamped_dataset(tmp_path: Path) -> None:
    # The documented wild case: a camera NAMED front judged kind top —
    # rollout must use the judged kind, not the name.
    dataset = write_rig_dataset(tmp_path)
    kinds = resolve_camera_kinds(CAMERAS, {}, dataset)
    assert kinds == {"front": "top", "wrist": "wrist"}


def test_kinds_camera_missing_from_file_renders_unknown(tmp_path: Path) -> None:
    dataset = write_rig_dataset(tmp_path, cameras={"front": "top"})
    kinds = resolve_camera_kinds(CAMERAS, {}, dataset)
    assert kinds == {"front": "top", "wrist": "unknown"}


def test_kinds_hash_mismatch_mirrors_training_unknown(tmp_path: Path) -> None:
    # Training rendered unknown when the kinds file's hash mismatched
    # the stamp; the rollout mirror must not fall back to names.
    dataset = write_rig_dataset(tmp_path, kinds_hash="other")
    kinds = resolve_camera_kinds(CAMERAS, {}, dataset)
    assert kinds == {"front": "unknown", "wrist": "unknown"}


def test_kinds_unstamped_dataset_mirrors_training_unknown(tmp_path: Path) -> None:
    dataset = write_rig_dataset(tmp_path, stamp_hash=None)
    kinds = resolve_camera_kinds(CAMERAS, {}, dataset)
    assert kinds == {"front": "unknown", "wrist": "unknown"}


def test_kinds_name_heuristic_without_dataset() -> None:
    kinds = resolve_camera_kinds(["front", "cam9"], {}, None)
    assert kinds == {"front": "front", "cam9": "unknown"}


def test_kinds_override_wins_over_dataset(tmp_path: Path) -> None:
    dataset = write_rig_dataset(tmp_path)
    kinds = resolve_camera_kinds(CAMERAS, {"front": "side"}, dataset)
    assert kinds == {"front": "side", "wrist": "wrist"}


def test_kinds_fallback_notice_skipped_for_overridden_camera(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A camera the kinds file does not know normally prints the loud
    'rendering as unknown' notice — but not when an explicit
    --camera-kind covers it: the override wins, so the notice would
    describe a fallback that never happens."""
    dataset = write_rig_dataset(tmp_path, cameras={"front": "top"})
    kinds = resolve_camera_kinds(["front", "top"], {"top": "top"}, dataset)
    assert kinds == {"front": "top", "top": "top"}
    captured = capsys.readouterr()
    assert "rendering as 'unknown'" not in captured.out
    # The un-overridden miss still prints (loud fallback stays loud).
    kinds = resolve_camera_kinds(["front", "top"], {}, dataset)
    assert kinds == {"front": "top", "top": "unknown"}
    assert "rendering as 'unknown'" in capsys.readouterr().out


def test_kinds_name_heuristic_warning_skipped_for_overridden_camera(
    capsys: pytest.CaptureFixture[str],
) -> None:
    kinds = resolve_camera_kinds(["front", "cam9"], {"cam9": "side"}, None)
    assert kinds == {"front": "front", "cam9": "side"}
    assert "not in the semantic" not in capsys.readouterr().out
    kinds = resolve_camera_kinds(["front", "cam9"], {}, None)
    assert kinds == {"front": "front", "cam9": "unknown"}
    assert "not in the semantic" in capsys.readouterr().out


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
