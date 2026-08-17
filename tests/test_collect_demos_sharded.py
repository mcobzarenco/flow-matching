"""CPU oracles for the sharded demo-generation path
(sim/collect_demos_sharded.py + sim/merge_demo_shards.py, queue item
``demo-gen-sharded-a100``): shard planning invariants, the driver's
resume-manifest refusal, and the headline oracle — a 2-shard merge is
bit-identical to a single run over the same seeds (frame tables,
decoded video pixels, exact quantile stats).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sim.collect_demos import DemoEpisode, DemoFrame, collect
from sim.collect_demos_sharded import _manifest_path, plan_shards, run, shard_env
from sim.merge_demo_shards import merge


def synthetic_episode(seed: int, *, success: bool, ticks: int = 4) -> DemoEpisode:
    """Seed-keyed synthetic episode (same construction as
    test_collect_demos: content depends only on the seed, so shard
    and single-run sources produce identical episodes)."""
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


def test_plan_disjoint_ranges_and_target_split() -> None:
    specs = plan_shards(
        shards=3,
        target_kept=10,
        seed_start=2000,
        seeds_per_shard=500,
    )
    assert [s.seed_start for s in specs] == [2000, 2500, 3000]
    assert all(s.max_seeds == 500 for s in specs)
    # Ranges are contiguous and disjoint.
    for a, b in itertools.pairwise(specs):
        assert a.seed_start + a.max_seeds == b.seed_start
    # 10 = 4 + 3 + 3 — remainder spread over the first shards.
    assert [s.target_kept for s in specs] == [4, 3, 3]
    assert sum(s.target_kept for s in specs) == 10


def test_plan_refuses_eval_holdout() -> None:
    with pytest.raises(ValueError, match="eval holdout"):
        plan_shards(shards=2, target_kept=4, seed_start=50, seeds_per_shard=100)


def test_shard_env_round_robins_gpus() -> None:
    specs = plan_shards(
        shards=4,
        target_kept=4,
        seed_start=2000,
        seeds_per_shard=100,
    )
    gpus = [0, 1, 2]
    for spec in specs:
        env = shard_env(spec, gpus)
        assert env["MUJOCO_GL"] == "egl"
        assert env["MUJOCO_EGL_DEVICE_ID"] == str(gpus[spec.index % 3])
        assert env["CUDA_VISIBLE_DEVICES"] == env["MUJOCO_EGL_DEVICE_ID"]


def _driver_args(root: Path, **overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "out": root,
        "repo_id": "fontaine/test_demos",
        "shards": 2,
        "target_kept": 4,
        "seed_start": 2000,
        "seeds_per_shard": 100,
        "spawn_version": "v2",
        "tint_band": "mix70",
        "bracket_appearance": "v1",
        "wrist_pose": "v1",
        "max_wall_hours": 1.0,
        "max_ticks": 600,
        "gpus": [0],
        "poll_s": 1.0,
        "dry_run": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_driver_refuses_changed_plan(tmp_path: Path) -> None:
    """A rerun with different sharding args must refuse, not silently
    re-partition seed ranges over half-filled shards."""
    root = tmp_path / "gen"
    root.mkdir()
    args = _driver_args(root)
    manifest = {
        "repo_id": args.repo_id,
        "spawn_version": args.spawn_version,
        "tint_band": args.tint_band,
        "bracket_appearance": args.bracket_appearance,
        "wrist_pose": args.wrist_pose,
        "target_kept": args.target_kept,
        "shards": [
            spec.__dict__
            for spec in plan_shards(
                shards=2,
                target_kept=4,
                seed_start=2000,
                seeds_per_shard=100,
            )
        ],
    }
    _manifest_path(root).write_text(json.dumps(manifest))
    assert run(args) == 0  # same plan: accepted (dry run)
    with pytest.raises(SystemExit, match="re-partition"):
        run(_driver_args(root, shards=4))


def test_two_shard_merge_bit_identical_to_single_run(tmp_path: Path) -> None:
    """THE oracle: shard 0 over seeds [1000, 1005) + shard 1 over
    [1005, 1010), merged, must equal one run over [1000, 1010) —
    same synthetic episodes (seed-keyed, success on all but 1002 and
    1007), compared on every parquet column, decoded video pixels,
    stats, and provenance."""

    def source(seed: int) -> DemoEpisode:
        return synthetic_episode(seed, success=seed not in (1002, 1007), ticks=3)

    single = tmp_path / "single"
    collect(
        single,
        source,
        target_kept=8,
        seed_start=1000,
        max_seeds=10,
        log=lambda _: None,
    )

    root = tmp_path / "gen"
    for k in range(2):
        collect(
            root / "shards" / f"shard_{k:02d}",
            source,
            target_kept=4,
            seed_start=1000 + 5 * k,
            max_seeds=5,
            extra_provenance={"spawn_version": "v1", "tint_band": "rig_gray"},
            log=lambda _: None,
        )
    merged_root = tmp_path / "merged"
    provenance = merge(root, merged_root)

    assert provenance["kept"] == 8
    assert provenance["kept_seeds"] == [
        1000,
        1001,
        1003,
        1004,
        1005,
        1006,
        1008,
        1009,
    ]
    single_prov = json.loads(
        (single / "meta" / "demo_provenance.json").read_text(),
    )
    assert provenance["kept_seeds"] == single_prov["kept_seeds"]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    want = LeRobotDataset("fontaine/grasp_sft_demos_v0", root=single)
    got = LeRobotDataset("fontaine/grasp_sft_demos_v0", root=merged_root)
    assert got.meta.total_episodes == want.meta.total_episodes == 8
    assert got.meta.total_frames == want.meta.total_frames == 24

    # Every parquet column, bit-equal and in the same order.
    want_table = want.hf_dataset.with_format("numpy")
    got_table = got.hf_dataset.with_format("numpy")
    assert set(want_table.column_names) == set(got_table.column_names)
    for column in want_table.column_names:
        np.testing.assert_array_equal(
            np.asarray(got_table[column]),
            np.asarray(want_table[column]),
            err_msg=f"column {column!r} differs after merge",
        )

    # Decoded video pixels, frame by frame.
    for index in range(len(want)):
        for key in ("observation.images.front", "observation.images.wrist"):
            np.testing.assert_array_equal(
                np.asarray(got[index][key]),
                np.asarray(want[index][key]),
                err_msg=f"decoded {key} differs at frame {index}",
            )

    # Exact quantile rows equal the single run's (both rewritten from
    # raw frames — the count-weighted-mean class bug stays dead).
    want_stats = json.loads((single / "meta" / "stats.json").read_text())
    got_stats = json.loads((merged_root / "meta" / "stats.json").read_text())
    for feature in ("action", "observation.state"):
        for key, value in want_stats[feature].items():
            if key.startswith("q") and key[1:].isdigit():
                np.testing.assert_allclose(
                    np.asarray(got_stats[feature][key]),
                    np.asarray(value),
                    err_msg=f"{feature}/{key} differs after merge",
                )


def test_merge_refuses_protocol_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "gen"
    for k, tint in enumerate(("rig_gray", "wide")):
        collect(
            root / "shards" / f"shard_{k:02d}",
            lambda seed: synthetic_episode(seed, success=True, ticks=2),
            target_kept=1,
            seed_start=1000 + 5 * k,
            max_seeds=5,
            extra_provenance={"spawn_version": "v2", "tint_band": tint},
            log=lambda _: None,
        )
    with pytest.raises(SystemExit, match="disagree on provenance"):
        merge(root, tmp_path / "merged")
