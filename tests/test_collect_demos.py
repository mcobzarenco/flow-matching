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
