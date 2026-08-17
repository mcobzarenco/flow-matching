"""Reconstruct the grasp_sft_v1 endpoint sim100 merged-leg JSONs from
the surviving shard logs (incident 2026-08-17: `rm -rf
~/flow-matching/outputs` on the box — run before the demo-gen v2 driver
to clear stale state — deleted the merged sim100 JSONs and rollout
videos before the rsync-local step of the endpoint boundary ran; the
shard stdout logs lived in ~ and survived).

Each shard log ends in the per-seed summary table rollout_sim prints:

  seed | draw | init cm | min cm | final cm | progress cm | success
     9 |    0 |    11.0 |    4.5 |      5.4 |         6.5 | -
    12 |    0 |     7.3 |    3.9 |      3.9 |         3.3 | tick 167

Reconstructed per episode: seed, draw, initial_cm, min_cm, final_cm,
success_tick, progress_final_cm (= initial - final; the table's
"progress cm" column is initial - min, a different metric). Values
carry the table's 0.1 cm print precision. Traces (distance_cm, grip,
spawn_xy) and the best-seed videos are NOT recoverable.

The output carries config.reconstructed_from_logs = true and a minimal
config block (only what the reads stack touches: serve_head); the
merge-guard fields are gone with the shard JSONs.

Usage:
  uv run python fontaine/scripts/reconstruct_sim100_from_logs.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path("/home/ubuntu/flow-matching")
LOG_DIR = REPO / "outputs/sim/grasp_sft/v1_endpoint/logs"
OUT_DIR = REPO / "outputs/sim/grasp_sft/v1_endpoint"

ROW = re.compile(
    r"^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.-]+)\s*\|\s*(.+?)\s*$",
)

LEGS = {
    "flow_unseen": {"shards": ("s0", "s25", "s50", "s75"), "serve_head": None},
    "token_unseen": {"shards": ("s0", "s25", "s50", "s75"), "serve_head": "ar"},
}


def parse_shard(path: Path) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    in_table = False
    for line in path.read_text().splitlines():
        if line.startswith("seed | draw |"):
            in_table = True
            continue
        if in_table:
            m = ROW.match(line)
            if not m:
                if episodes:
                    in_table = False
                continue
            seed, draw = int(m.group(1)), int(m.group(2))
            init, mn, fin = (float(m.group(i)) for i in (3, 4, 5))
            succ = m.group(7)
            tick = None
            tm = re.match(r"tick (\d+)", succ)
            if tm:
                tick = int(tm.group(1))
            episodes.append(
                {
                    "seed": seed,
                    "draw": draw,
                    "initial_cm": init,
                    "min_cm": mn,
                    "final_cm": fin,
                    "success_tick": tick,
                    "progress_final_cm": round(init - fin, 1),
                },
            )
    return episodes


def main() -> int:
    for leg, spec in LEGS.items():
        episodes: list[dict[str, object]] = []
        for shard in spec["shards"]:
            path = LOG_DIR / f"eval__sft_v1_{leg.split('_')[0]}_unseen_{shard}.log"
            rows = parse_shard(path)
            if len(rows) != 25:
                print(f"FATAL: {path.name} parsed {len(rows)} rows, want 25")
                return 2
            episodes.extend(rows)
        episodes.sort(key=lambda e: (e["seed"], e["draw"]))
        seeds = [e["seed"] for e in episodes]
        if seeds != list(range(100)):
            print(f"FATAL: {leg} seed tiling broken")
            return 2
        payload = {
            "config": {
                "reconstructed_from_logs": True,
                "note": (
                    "per-seed summary reconstructed from shard stdout logs "
                    "after the 2026-08-17 box outputs/ wipe; 0.1 cm print "
                    "precision; traces + videos unrecoverable"
                ),
                "serve_head": spec["serve_head"],
                "seed": 0,
                "num_seeds": 100,
            },
            "episodes": episodes,
        }
        out = OUT_DIR / f"{leg}.json"
        out.write_text(json.dumps(payload, indent=1))
        succ = sum(1 for e in episodes if e["success_tick"] is not None)
        moved = sum(1 for e in episodes if e["progress_final_cm"] > 0.5)
        print(f"{leg}: {succ}/100 success, moved>0.5cm {moved} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
