"""Merge seed-sharded rollout_sim JSONs into one leg JSON.

Sharding a 100-seed leg across GPUs is exact — every stochastic stream
in the rollout derives from the (env seed, replan, draw) triple
(sim/rollout_sim.py sim_item docstring: deterministic per key, invariant
to batch composition), so N processes over disjoint seed ranges produce
bitwise the episodes one process would. This script re-assembles the
single-leg payload the reads stack expects
(fontaine/scripts/grasp_sft_joint_probe_reads.py).

Guards: shard configs must agree on every key except seed/num_seeds;
seed ranges must tile [seed, seed+num_seeds) with no gap or overlap.

Usage:
  uv run python fontaine/scripts/merge_rollout_shards.py \
      --out outputs/sim/grasp_sft/v1_endpoint/flow_unseen.json \
      shard0.json shard1.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VARYING = {"seed", "num_seeds"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(p.read_text()) for p in args.shards]
    base = {k: v for k, v in payloads[0]["config"].items() if k not in VARYING}
    for path, payload in zip(args.shards, payloads, strict=True):
        got = {k: v for k, v in payload["config"].items() if k not in VARYING}
        if got != base:
            diff = {k for k in base if got.get(k) != base[k]}
            print(f"FATAL: {path} config mismatch on {sorted(diff)}", file=sys.stderr)
            return 2

    episodes = sorted(
        (e for p in payloads for e in p["episodes"]),
        key=lambda e: (e["seed"], e.get("draw", 0)),
    )
    seeds = [e["seed"] for e in episodes if e.get("draw", 0) == 0]
    lo, hi = min(seeds), max(seeds)
    if sorted(seeds) != list(range(lo, hi + 1)):
        missing = sorted(set(range(lo, hi + 1)) - set(seeds))
        dupes = sorted({s for s in seeds if seeds.count(s) > 1})
        print(
            f"FATAL: seed tiling broken (missing {missing}, dupes {dupes})",
            file=sys.stderr,
        )
        return 2

    merged = {
        "config": {**base, "seed": lo, "num_seeds": hi - lo + 1},
        "merged_from": [str(p) for p in args.shards],
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=1))
    print(f"wrote {args.out}: {len(episodes)} episodes, seeds {lo}-{hi}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
