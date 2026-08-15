"""CPU oracles for the stage-B demo collector (sim/collect_demos.py,
pre-reg §2/§6) — no GL, no physics: episodes come from synthetic
sources, so what is pinned is the WRITER contract the SFT trains on:
eval-seed refusal, success-only keeping, the LeRobot round-trip
(action/state bit-equal, schema names, task string), and resume.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim.collect_demos import (
    MOTOR_NAMES,
    TASK,
    DemoEpisode,
    DemoFrame,
    collect,
)


def synthetic_episode(seed: int, *, success: bool, ticks: int = 4) -> DemoEpisode:
    rng = np.random.default_rng(seed)
    frames = [
        DemoFrame(
            top=rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            wrist=rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            state=rng.uniform(-90, 90, 6).astype(np.float32),
            action=rng.uniform(-90, 90, 6).astype(np.float32),
        )
        for _ in range(ticks)
    ]
    return DemoEpisode(
        seed=seed,
        success=success,
        frames=frames,
        ticks=ticks,
        final_disk_cm=1.0 if success else 9.0,
    )


def test_eval_seed_refusal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eval holdout"):
        collect(
            tmp_path / "d",
            lambda seed: synthetic_episode(seed, success=True),
            target_kept=1,
            seed_start=99,
            log=lambda _: None,
        )


def test_success_only_and_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "demos"
    episodes: dict[int, DemoEpisode] = {}

    def source(seed: int) -> DemoEpisode:
        episodes[seed] = synthetic_episode(seed, success=seed != 1001)
        return episodes[seed]

    summary = collect(
        root,
        source,
        target_kept=2,
        seed_start=1000,
        log=lambda _: None,
    )
    # 1000 kept, 1001 missed, 1002 kept.
    assert summary["kept_seeds"] == [1000, 1002]
    assert summary["attempted"] == 3
    assert json.loads((root / "meta" / "demo_provenance.json").read_text())[
        "state_units"
    ].startswith("rig (identity")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("fontaine/grasp_sft_demos_v0", root=root)
    assert dataset.meta.total_episodes == 2
    assert dataset.meta.fps == 30
    assert dataset.meta.features["action"]["names"] == MOTOR_NAMES
    assert dataset.meta.features["observation.state"]["names"] == MOTOR_NAMES
    assert set(dataset.meta.video_keys) == {
        "observation.images.front",
        "observation.images.wrist",
    }
    # Bit-equal state/action against what the source produced, in order.
    row = dataset.hf_dataset[0]
    want = episodes[1000].frames[0]
    np.testing.assert_array_equal(np.asarray(row["action"]), want.action)
    np.testing.assert_array_equal(
        np.asarray(row["observation.state"]),
        want.state,
    )
    assert dataset.meta.tasks.index.tolist() == [TASK]


def test_resume_appends(tmp_path: Path) -> None:
    root = tmp_path / "demos"
    collect(
        root,
        lambda seed: synthetic_episode(seed, success=True),
        target_kept=1,
        seed_start=1000,
        log=lambda _: None,
    )
    summary = collect(
        root,
        lambda seed: synthetic_episode(seed, success=True),
        target_kept=2,
        seed_start=1000,
        log=lambda _: None,
    )
    assert summary["kept_seeds"] == [1000, 1001]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("fontaine/grasp_sft_demos_v0", root=root)
    assert dataset.meta.total_episodes == 2
    indices = {int(i) for i in dataset.hf_dataset["episode_index"]}
    assert sorted(indices) == [0, 1]


def test_refuses_foreign_directory(tmp_path: Path) -> None:
    root = tmp_path / "demos"
    root.mkdir()
    (root / "unrelated.txt").write_text("x")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        collect(
            root,
            lambda seed: synthetic_episode(seed, success=True),
            target_kept=1,
            seed_start=1000,
            log=lambda _: None,
        )


def test_rewrite_quantile_stats_fixes_bimodal_aggregation(tmp_path: Path) -> None:
    """The 2026-08-15 class bug: lerobot merges per-episode quantiles as
    a weighted MEAN of quantiles, so a channel bimodal ACROSS episodes
    (the π-flipped wrist_roll branch) gets a clamp box that excludes an
    entire mode. The rewrite must land the exact all-frame quantiles."""
    import pandas as pd

    from sim.collect_demos import rewrite_quantile_stats

    root = tmp_path / "demos"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    # Two "episodes": one mode at +90, one at -150 — per-episode q01
    # averaging would land mid-air between the modes.
    pos = np.full((80, 6), 90.0)
    neg = np.full((20, 6), -150.0)
    values = np.concatenate([pos, neg])
    frame = pd.DataFrame(
        {
            "action": list(values.astype(np.float32)),
            "observation.state": list(values.astype(np.float32)),
            "episode_index": [0] * 80 + [1] * 20,
        },
    )
    frame.to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    corrupt = {
        key: {"q01": [42.0] * 6, "q99": [66.0] * 6, "mean": [42.0] * 6}
        for key in ("action", "observation.state")
    }
    (root / "meta" / "stats.json").write_text(json.dumps(corrupt))

    fixed = rewrite_quantile_stats(root)

    stats = json.loads((root / "meta" / "stats.json").read_text())
    for key in ("action", "observation.state"):
        q01 = np.array(stats[key]["q01"])
        q99 = np.array(stats[key]["q99"])
        expected_q01 = np.quantile(values, 0.01, axis=0)
        expected_q99 = np.quantile(values, 0.99, axis=0)
        np.testing.assert_allclose(q01, expected_q01)
        np.testing.assert_allclose(q99, expected_q99)
        # The -150 mode is INSIDE the box again.
        assert (q01 <= -150.0 + 1e-9).all()
        # Non-quantile rows untouched.
        assert stats[key]["mean"] == [42.0] * 6
    assert set(fixed) == {
        "action/q01",
        "action/q99",
        "observation.state/q01",
        "observation.state/q99",
    }
