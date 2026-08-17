"""Merge sharded demo collections into one LeRobot v3 dataset (queue
item ``demo-gen-sharded-a100``; companion to ``collect_demos_sharded``).

The heavy lifting is lerobot's own ``aggregate_datasets`` (episode /
frame / task reindexing, parquet + video concatenation). On top of it,
this tool restores the two collector invariants aggregation does not
know about:

1. **Provenance union** — ``meta/demo_provenance.json``: kept-seed
   lists concatenated in shard order (shard ranges are disjoint and
   ascending, so the merged order equals the single-run order),
   attempts summed, per-shard summaries kept verbatim, the shared
   protocol fields (spawn_version, tint band, expert head, ...)
   carried after an equality check across shards.
2. **Exact quantile stats** — ``rewrite_quantile_stats`` over the
   merged frames (the 2026-08-15 class bug: lerobot merges per-episode
   quantiles as a count-weighted mean, catastrophic on cross-episode
   bimodal channels; aggregation would inherit the same error).

Oracle (tests/test_collect_demos_sharded.py): a 2-shard collection
merged by this tool is bit-identical to a single run over the same
seeds — frame tables, decoded video pixels, stats.

Usage:
  uv run python -m sim.merge_demo_shards \
      --root ~/datasets/fontaine/grasp_demos_v1 \
      --out ~/datasets/fontaine/grasp_demos_v1/merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .collect_demos import REPO_ID, rewrite_quantile_stats

#: Provenance fields that must agree across shards (one protocol, one
#: expert) and are carried into the merged record verbatim.
SHARED_PROVENANCE_KEYS = (
    "state_units",
    "expert_head",
    "prereg",
    "substrate",
    "success_definition",
    "spawn_version",
    "tint_band",
    "spawn_v2_prereg",
    "bracket_appearance",
    "wrist_pose",
    "retreat_tail",
)


def find_shards(root: Path) -> list[Path]:
    shards = sorted((root / "shards").glob("shard_*"))
    if not shards:
        raise SystemExit(f"ABORT: no shards under {root}/shards")
    missing = [s.name for s in shards if not (s / "meta" / "info.json").exists()]
    if missing:
        raise SystemExit(f"ABORT: shards without meta/info.json: {missing}")
    return shards


def merge_provenance(shards: list[Path]) -> dict[str, Any]:
    summaries = [
        json.loads((s / "meta" / "demo_provenance.json").read_text()) for s in shards
    ]
    merged: dict[str, Any] = {
        "kept": sum(s["kept"] for s in summaries),
        "attempted": sum(s["attempted"] for s in summaries),
        "kept_seeds": [seed for s in summaries for seed in s["kept_seeds"]],
        "stop_reason": [s["stop_reason"] for s in summaries],
        "wall_s": max(s["wall_s"] for s in summaries),
        "shards": [
            {"shard": shard.name, **summary}
            for shard, summary in zip(shards, summaries, strict=True)
        ],
    }
    for key in SHARED_PROVENANCE_KEYS:
        values = {json.dumps(s.get(key)) for s in summaries}
        if len(values) > 1:
            raise SystemExit(
                f"ABORT: shards disagree on provenance {key!r}: {values} — "
                "these are not shards of one protocol",
            )
        merged[key] = summaries[0].get(key)
    return merged


def merge(root: Path, out: Path, repo_id: str = REPO_ID) -> dict[str, Any]:
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"ABORT: merge target {out} exists and is non-empty")
    shards = find_shards(root)
    provenance = merge_provenance(shards)
    aggregate_datasets(
        repo_ids=[repo_id] * len(shards),
        aggr_repo_id=repo_id,
        roots=shards,
        aggr_root=out,
    )
    fixed = rewrite_quantile_stats(out)
    (out / "meta" / "demo_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
    )
    merged_meta = LeRobotDatasetMetadata(repo_id, root=out)
    if merged_meta.total_episodes != provenance["kept"]:
        raise SystemExit(
            f"ABORT: merged dataset has {merged_meta.total_episodes} episodes "
            f"but shard provenance says {provenance['kept']} kept",
        )
    print(
        f"[merge] {len(shards)} shards -> {out}: "
        f"{merged_meta.total_episodes} episodes, "
        f"{merged_meta.total_frames} frames, quantiles rewritten "
        f"({len(fixed)} rows), provenance united",
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()
    merge(args.root.expanduser(), args.out.expanduser(), args.repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
