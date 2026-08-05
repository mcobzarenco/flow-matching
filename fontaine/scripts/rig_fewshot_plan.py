"""Build the frozen rig few-shot benchmark plan (ideas #16, pre-reg
posts/2026-08-05-prereg-rig-fewshot-benchmark.md).

k4l2 panel over the 12-episode benchmark holdout of the two owner rig
repos, in ORIGINAL repo coordinates (eval loads the source repos; the
derived training subsets never contain these episodes). The holdout is
the codebase's native split — ``holdout_episodes(repo_id, n, 0.212,
split_seed=16)`` — NOT a bespoke draw, so ``bijou.eval.leakage`` can
recompute the radioactive set from the plan header alone (the
pre-reg's SeedSequence(16) uniform draw could not feed the checker;
mechanism amendment posted on the pre-reg). Fraction 0.212 is chosen
so per-repo banker's rounding lands exactly on the pre-registered
counts: round(.212*50)=11 of so101_pick_place_v2 + round(.212*7)=1 of
so101_pick_place_clean = 12 held out, 45 train.

Frame draws go through ``bijou.eval.plan.build_plan`` itself (episode
tables reconstructed from the data parquet — column reads only, no
video decode) and are then FILTERED to the holdout episodes: the
per-episode draw is a pure function of (plan_seed, repo_id, episode),
so filtering the full-corpus plan is exactly the holdout plan, with
zero reimplementation of the draw.

Scoring invocation this plan freezes (validate_plan checks these):
    uv run python -m bijou.eval --data <the two rig repo dirs> \
        --sample-plan plans/rig_fewshot_v0_k4l2.json \
        --episodes holdout --holdout-episodes 0.212 --split-seed 16 \
        --dump-predictions ...

Run from the repo root: uv run python fontaine/scripts/rig_fewshot_plan.py
Refuses to overwrite an existing plan (frozen panel) without --force.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bijou.data import holdout_episodes, repo_id_of
from bijou.eval.plan import EpisodeTable, SamplePlan, build_plan

SOURCE_DIRS = (
    Path("~/datasets/mcobzarenco/so101_pick_place_v2").expanduser(),
    Path("~/datasets/mcobzarenco/so101_pick_place_clean").expanduser(),
)
PLAN_PATH = Path("plans/rig_fewshot_v0_k4l2.json")
PLAN_SEED = 1  # house k4l2 practice (community panel uses 1)
FRAMES_PER_EPISODE = 4
LABELED_PER_EPISODE = 2
HOLDOUT_FRACTION = 0.212
SPLIT_SEED = 16
# Pre-registered: 12 held out / 45 train over 57 episodes.
EXPECTED_HOLDOUT = {"so101_pick_place_v2": 11, "so101_pick_place_clean": 1}


def episode_table_from_parquet(dataset_dir: Path, offset: int) -> EpisodeTable:
    """Reconstruct one dataset's EpisodeTable from its data parquet —
    same layout scan as bijou.eval.plan.episode_tables, minus the
    LeRobot dataset load (no torch, no video)."""
    import pyarrow as pa

    files = sorted(dataset_dir.glob("data/*/*.parquet"))
    table = pa.concat_tables(
        pq.read_table(path, columns=["episode_index", "annotation.progress"])
        for path in files
    )
    episode_column = np.asarray(table.column("episode_index"))
    episode_ids, starts = np.unique(episode_column, return_index=True)
    assert bool(np.all(np.diff(starts) > 0)), (
        f"{dataset_dir}: episode rows not grouped in ascending order"
    )
    progress = np.asarray(table.column("annotation.progress"), dtype=np.float32)
    return EpisodeTable(
        offset=offset,
        length=len(episode_column),
        episode_ids=episode_ids,
        starts=starts,
        labeled=np.isfinite(progress),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if PLAN_PATH.exists() and not args.force:
        sys.exit(f"{PLAN_PATH} exists — a plan is a frozen panel (--force to rebuild)")

    tables: dict[str, EpisodeTable] = {}
    held_out: dict[str, tuple[int, ...]] = {}
    offset = 0
    for dataset_dir in SOURCE_DIRS:
        repo_id = repo_id_of(dataset_dir)
        table = episode_table_from_parquet(dataset_dir, offset)
        offset += table.length
        tables[repo_id] = table
        held = holdout_episodes(
            repo_id,
            len(table.episode_ids),
            HOLDOUT_FRACTION,
            SPLIT_SEED,
        )
        expected = EXPECTED_HOLDOUT[dataset_dir.name]
        assert len(held) == expected, (
            f"{repo_id}: {len(held)} holdout episodes, pre-reg fixes {expected}"
        )
        held_out[repo_id] = held
        print(f"{repo_id}: {len(table.episode_ids)} episodes, holdout {list(held)}")

    full = build_plan(
        tables,
        plan_seed=PLAN_SEED,
        frames_per_episode=FRAMES_PER_EPISODE,
        labeled_per_episode=LABELED_PER_EPISODE,
        episodes="holdout",
        holdout_episodes=HOLDOUT_FRACTION,
        split_seed=SPLIT_SEED,
        fps=None,
        camera_counts=None,
    )
    keep = {(repo_id, episode) for repo_id, eps in held_out.items() for episode in eps}
    plan = SamplePlan(
        plan_seed=full.plan_seed,
        frames_per_episode=full.frames_per_episode,
        labeled_per_episode=full.labeled_per_episode,
        episodes=full.episodes,
        holdout_episodes=full.holdout_episodes,
        split_seed=full.split_seed,
        fps=full.fps,
        camera_counts=full.camera_counts,
        created_at=full.created_at,
        core=[f for f in full.core if (f.repo_id, f.episode_index) in keep],
        labeled=[f for f in full.labeled if (f.repo_id, f.episode_index) in keep],
    )
    total_holdout = sum(len(eps) for eps in held_out.values())
    assert total_holdout == 12, f"holdout total {total_holdout} != pre-registered 12"
    assert len(plan.core) == total_holdout * FRAMES_PER_EPISODE, (
        f"core {len(plan.core)} != {total_holdout}x{FRAMES_PER_EPISODE} "
        "(an episode shorter than k would explain this — inspect before freezing)"
    )
    plan.save(PLAN_PATH)
    print(
        f"wrote {PLAN_PATH}: {len(plan.core)} core + {len(plan.labeled)} labeled "
        f"frames over {total_holdout} episodes",
    )


if __name__ == "__main__":
    main()
