"""Reads for the owner-ordered step2000 probe (2026-08-15 10:10Z):
train-seed vs unseen-seed success of the killed stage-C checkpoint.

Consumes the rollout JSONs as they land (tolerant of the train arm
still running) plus the banked stage-B collect state for the
kept-seed split:

  outputs/sim/grasp_sft/step2000_probe/unseen.json   (seeds 0-99)
  outputs/sim/grasp_sft/step2000_probe/train.json    (seeds 1000-1099)
  reports/curve__grasp_sft_stageb_collect.json       (kept_seeds)

The train band splits into KEPT (episodes actually in the training
set) and NON-KEPT (expert-failed spawns — near-distribution but never
trained on). Success = success_tick non-null; also reports moved
(progress_final_cm > 0.5) and mean progress, mirroring the sim100
read conventions. Context anchors (record-only): ftrig4k ~1/100,
stage-1 W0 2/100 on the same unseen protocol.

Usage:
  uv run python fontaine/scripts/grasp_sft_step2000_probe_reads.py \
      [--out reports/analysis__grasp_sft_step2000_probe.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path("/home/ubuntu/flow-matching")
PROBE = REPO / "outputs/sim/grasp_sft/step2000_probe"
COLLECT = REPO / "reports/curve__grasp_sft_stageb_collect.json"


def arm_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    succ = [e for e in episodes if e.get("success_tick") is not None]
    moved = [e for e in episodes if e.get("progress_final_cm", 0.0) > 0.5]
    prog = [e.get("progress_final_cm", 0.0) for e in episodes]
    strikes = sum(e.get("reset_strikes", 0) for e in episodes)
    return {
        "n": n,
        "successes": len(succ),
        "success_seeds": sorted(e["seed"] for e in succ),
        "moved_gt_half_cm": len(moved),
        "mean_progress_cm": sum(prog) / n if n else None,
        "reset_strikes": strikes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "reports/analysis__grasp_sft_step2000_probe.json",
    )
    args = ap.parse_args()

    kept = set(json.loads(COLLECT.read_text())["kept_seeds"])
    out: dict[str, Any] = {
        "read": "grasp_sft_step2000_probe",
        "order": "owner 2026-08-15 10:10:20Z (kill + train-vs-unseen probe)",
        "checkpoint": "molmoact2_grasp_sft_stagec_ar_step2000 (killed @2040)",
        "protocol": "euler-10, sequential, 30 s episodes, videos banked",
        "anchor_primary": "released base (SFT init) 9/100 — release_officialmap_a_100ep_30s, owner-agreed primary comparator 2026-08-15 12:0xZ",
        "anchors_record_only": "ftrig4k ~1/100, stage-1 W0 2/100 (unseen protocol)",
    }

    unseen_path = PROBE / "unseen.json"
    if unseen_path.exists():
        eps = json.loads(unseen_path.read_text())["episodes"]
        out["unseen_0_99"] = arm_stats(eps)
    else:
        out["unseen_0_99"] = None

    train_path = PROBE / "train.json"
    if train_path.exists():
        eps = json.loads(train_path.read_text())["episodes"]
        in_train = [e for e in eps if e["seed"] in kept]
        held = [e for e in eps if e["seed"] not in kept]
        out["train_band_kept"] = arm_stats(in_train)
        out["train_band_nonkept_expert_failed"] = arm_stats(held)
    else:
        out["train_band_kept"] = None
        out["train_band_nonkept_expert_failed"] = None

    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    main()
