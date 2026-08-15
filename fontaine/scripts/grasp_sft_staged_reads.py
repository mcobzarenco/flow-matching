"""Stage-D decision reads for the grasp-SFT bootstrap
(posts/2026-08-14-prereg-grasp-sft-bootstrap.md §2/§6, frozen).

Consumes the per-arm JSON ``sim/rollout_sim.py --out-json`` wrote for
the SFT endpoint(s) on the FROZEN 100 eval seeds (0-99) and emits one
analysis JSON carrying the registered decision surface:

- PRIMARY (frozen §2): successes on the AR-primary arm —
  ``>= 20/100 -> GRPO_GO`` (fresh pre-reg per Decision 11),
  ``5-19 -> ITERATE_BC_ONCE`` (one more B/C round, more demos),
  ``< 5 -> F_TRANSFER`` (observation-side gap; the wrist screen's
  F-instrument read becomes the binding diagnosis).
- Gates: reset strikes == 0 (sim100 convention); seed list must be
  exactly 0-99 (the frozen holdout — any other set is not this read).
- Record-only context: mean/median progress_final_cm, moved count,
  per-seed success ticks; the banked context anchors (ftrig4k +0.08 /
  47 moved / ~1 success; stage-1 W0 +0.054 / 44 / 2) ride the payload
  for the results post, not as gates.

Usage:
  uv run python fontaine/scripts/grasp_sft_staged_reads.py \
      --ar-json outputs/sim/grasp_sft/stageD/ar.json \
      [--flow-json outputs/sim/grasp_sft/stageD/flow.json] \
      --out reports/analysis__grasp_sft_stageD_sim100.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FROZEN_SEEDS = list(range(100))
BAND_GO = 20  # >= 20/100 -> GRPO GO
BAND_ITERATE = 5  # 5-19 -> one B/C iteration; < 5 -> F-transfer
CONTEXT_ANCHORS = {
    "ftrig4k_v3_sim100": {"mean_progress_cm": 0.08, "moved": 47, "successes": 1},
    "wrist_screen_stage1_W0": {"mean_progress_cm": 0.054, "moved": 44, "successes": 2},
}


def read_arm(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    episodes = payload["episodes"]
    seeds = sorted(int(e["seed"]) for e in episodes)
    if seeds != FROZEN_SEEDS:
        raise SystemExit(
            f"{path}: seeds are not the frozen 0-99 eval holdout "
            f"(got {len(seeds)} seeds, first/last {seeds[0]}/{seeds[-1]}) — "
            "this is not the registered stage-D read",
        )
    strikes = sum(int(e.get("reset_strikes", 0)) for e in episodes)
    prog = np.array([float(e["progress_final_cm"]) for e in episodes])
    # sim100 convention: success IS a recorded success_tick (no separate flag)
    succ = [e for e in episodes if e.get("success_tick") is not None]
    return {
        "config": payload.get("config"),
        "successes": len(succ),
        "success_seeds": [int(e["seed"]) for e in succ],
        "success_ticks": [int(e["success_tick"]) for e in succ],
        "reset_strikes_total": strikes,
        "mean_progress_final_cm": round(float(prog.mean()), 3),
        "median_progress_final_cm": round(float(np.median(prog)), 3),
        "moved_count": int((np.abs(prog) >= 0.5).sum()),
    }


def verdict(successes: int) -> str:
    if successes >= BAND_GO:
        return "GRPO_GO"
    if successes >= BAND_ITERATE:
        return "ITERATE_BC_ONCE"
    return "F_TRANSFER"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar-json", type=Path, required=True)
    parser.add_argument("--flow-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ar = read_arm(args.ar_json)
    arms: dict[str, dict[str, object]] = {"ar_primary": ar}
    if args.flow_json is not None:
        arms["flow"] = read_arm(args.flow_json)

    gates = {
        "reset_strikes_zero": all(a["reset_strikes_total"] == 0 for a in arms.values()),
    }
    payload = {
        "read": "grasp_sft_stageD_sim100",
        "prereg": "posts/2026-08-14-prereg-grasp-sft-bootstrap.md §2/§6",
        "primary_arm": "ar_primary",
        "bands": {
            "grpo_go": f">={BAND_GO}",
            "iterate": f"{BAND_ITERATE}-19",
            "f_transfer": f"<{BAND_ITERATE}",
        },
        "successes": ar["successes"],
        "verdict": verdict(int(ar["successes"])),  # type: ignore[arg-type]
        "gates": gates,
        "arms": arms,
        "context_anchors_record_only": CONTEXT_ANCHORS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[stageD] AR primary: {ar['successes']}/100 successes -> {payload['verdict']} "
        f"(strikes gate {'OK' if gates['reset_strikes_zero'] else 'FAIL'}); "
        f"banked {args.out}",
    )


if __name__ == "__main__":
    main()
