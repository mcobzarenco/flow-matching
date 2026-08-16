"""Reads for the route-C joint endpoint probes (registered amendment
2026-08-16, posts/2026-08-16-amendment-grasp-sft-route-c-joint.md §4):
both parents' decision surfaces on the ONE joint checkpoint.

Consumes the launcher's JSONs as they land (tolerant of legs still
queued — a missing file reads as null, never a crash):

  outputs/sim/grasp_sft/joint_probes/flow_unseen.json   (euler-10, 0-99)
  outputs/sim/grasp_sft/joint_probes/flow_train.json    (euler-10, 1000-1099)
  outputs/sim/grasp_sft/joint_probes/token_unseen.json  (_arhead greedy, 0-99)
  outputs/sim/grasp_sft/joint_probes/token_base.json    (_arhead greedy on the
                                                         corrected-base conversion)
  reports/curve__grasp_sft_stageb_collect.json          (kept-seed split)

Verdicts (frozen upstream, restated here so the JSON is
self-describing): flow unseen vs A §5 — >28 table-fix positive / 25-31
(~28±3) data-limited / <25 seam investigation first; token unseen vs
B §3 — >=20 R2 competent-base holds / 5-19 owner decision / <5
token-SFT did not transfer. The train band splits KEPT vs NON-KEPT
exactly as the step2000 probe did (memorization read). Success =
success_tick non-null; moved = progress_final_cm > 0.5 cm.

Usage:
  uv run python fontaine/scripts/grasp_sft_joint_probe_reads.py \
      [--probe-dir outputs/sim/grasp_sft/joint_probes] \
      [--collect reports/curve__grasp_sft_stageb_collect.json] \
      [--out reports/analysis__grasp_sft_joint_probes.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path("/home/ubuntu/flow-matching")


def arm_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """The step2000-probe row conventions, verbatim."""
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


def load_arm(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not path.exists():
        return None, []
    payload = json.loads(path.read_text())
    episodes = payload["episodes"]
    stats = arm_stats(episodes)
    # Provenance guard: a token json must come from the _arhead decode
    # (config.serve_head == "ar"), a flow json must not — a mixed-up
    # path would silently bank the wrong head's number.
    stats["serve_head"] = payload.get("config", {}).get("serve_head")
    return stats, episodes


def flow_verdict(successes: int | None) -> str | None:
    """A §5 on the unseen flow read (floor 28, band ±3, seam bar 25).
    The frozen text's two clauses OVERLAP on 29–31 ('>28 → positive'
    and '~28±3 → data-limited' both apply) — the instrument surfaces
    the overlap instead of silently resolving it; the boundary post
    owns the call there."""
    if successes is None:
        return None
    if successes > 31:
        return "TABLE_FIX_POSITIVE (>28+3: corrected lineage becomes the SFT artifact)"
    if successes > 28:
        return (
            "TABLE_FIX_POSITIVE_WITHIN_BAND (A §5 clauses overlap on 29-31: "
            ">28 says positive, ~28±3 says data-limited — boundary post "
            "states both, owner owns the pricing)"
        )
    if successes >= 25:
        return "DATA_LIMITED (~28±3: next lever is demos, not normalization)"
    return "SEAM_INVESTIGATION (<25: stack/objective seam before any pricing)"


def token_verdict(successes: int | None) -> str | None:
    """B §3 on the unseen token read (R2 activation bar 20)."""
    if successes is None:
        return None
    if successes >= 20:
        return "R2_PREMISE_HOLDS (>=20: token-GRPO's competent base exists)"
    if successes >= 5:
        return "OWNER_DECISION (5-19: head lags its flow sibling materially)"
    return (
        "NO_TRANSFER (<5: token-SFT did not transfer — the discrepancy is the finding)"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--probe-dir",
        type=Path,
        default=REPO / "outputs/sim/grasp_sft/joint_probes",
    )
    ap.add_argument(
        "--collect",
        type=Path,
        default=REPO / "reports/curve__grasp_sft_stageb_collect.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "reports/analysis__grasp_sft_joint_probes.json",
    )
    args = ap.parse_args()

    kept = set(json.loads(args.collect.read_text())["kept_seeds"])
    out: dict[str, Any] = {
        "read": "grasp_sft_joint_probes",
        "registration": "posts/2026-08-16-amendment-grasp-sft-route-c-joint.md §4",
        "checkpoint": "fontaine_grasp_sft_joint_corrected/step_002000 "
        "(molmoact2_joint, insulated, λ=1, corrected table, --offload-optim)",
        "protocol": "flow: euler-10; token: --serve-head ar greedy (_arhead); "
        "sequential, 30 s episodes, videos banked",
        "anchors": {
            "flow_base_unseen": 9,
            "flow_corrupt_table_step2000_unseen": 28,
            "token_r2_activation_bar": 20,
            "record_only": "ftrig4k ~1/100, W0 2/100 (unseen flow protocol); "
            "R1-B token floor 2/20 (different protocol)",
        },
    }

    flow_unseen, _ = load_arm(args.probe_dir / "flow_unseen.json")
    out["flow_unseen_0_99"] = flow_unseen
    out["flow_verdict_A5"] = flow_verdict(
        flow_unseen["successes"] if flow_unseen else None,
    )

    flow_train, train_eps = load_arm(args.probe_dir / "flow_train.json")
    out["flow_train_1000_1099"] = flow_train
    if train_eps:
        out["flow_train_kept"] = arm_stats(
            [e for e in train_eps if e["seed"] in kept],
        )
        out["flow_train_nonkept"] = arm_stats(
            [e for e in train_eps if e["seed"] not in kept],
        )

    token_unseen, _ = load_arm(args.probe_dir / "token_unseen.json")
    out["token_unseen_0_99"] = token_unseen
    out["token_verdict_B3"] = token_verdict(
        token_unseen["successes"] if token_unseen else None,
    )

    token_base, _ = load_arm(args.probe_dir / "token_base.json")
    out["token_base_unseen_0_99"] = token_base
    if token_base and token_unseen:
        out["token_sft_delta"] = token_unseen["successes"] - token_base["successes"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
