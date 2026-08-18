"""Paired per-seed read between two sim100 flow_unseen JSONs.

Registered consumer: the pdnorm endpoint's paired read vs the
disc-1000 demosonly baseline — the calibration note recorded
pre-launch in posts/2026-08-xx-prereg-grasp-sft-v2-joint-pdnorm.md
(the baseline's 11/100 sits inside the draft's own 11-19 ambiguous
band, so this read rides ALONGSIDE the frozen absolute bands; it is
recorded, not gating). Also runnable retro on any two banked
flow_unseen JSONs that share a seed list.

Reads, all paired on the shared seeds:
- success-count delta (A - B) with bootstrap CI95 (seed 0, 10k
  resamples — the sim100_reads.py bootstrap);
- discordant-seed counts (McNemar table: A-only / B-only successes)
  with the exact two-sided McNemar p (binomial, no asymptotics);
- progress_final delta (cm) with bootstrap CI95 and per-seed
  win/tie/loss split.

Usage:
  uv run python -m fontaine.scripts.sim100_paired_read \
      --a outputs/sim/grasp_sft/joint_probes/flow_unseen.json \
      --b outputs/sim/grasp_sft/disc1000_baseline/flow_unseen.json \
      --label-a probe_joint2000 --label-b disc1000_demosonly \
      --out reports/analysis__sim100_paired_probe_vs_disc1000.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from fontaine.scripts.sim100_reads import bootstrap_ci

PREREG = "posts/2026-08-xx-prereg-grasp-sft-v2-joint-pdnorm.md"


def load_paired(path_a: Path, path_b: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load two flow_unseen JSONs and align their episodes by seed.

    Paired design: the seed SETS must match exactly (order may
    differ); duplicate seeds within an arm are refused.
    """
    arms = []
    for path in (path_a, path_b):
        payload = json.loads(path.read_text())
        episodes = sorted(payload["episodes"], key=lambda e: e["seed"])
        seeds = np.array([e["seed"] for e in episodes])
        assert len(set(seeds.tolist())) == len(seeds), f"{path}: duplicate seeds"
        arms.append(
            {
                "config": payload["config"],
                "seeds": seeds,
                "progress_final": np.array([e["progress_final_cm"] for e in episodes]),
                "success": np.array(
                    [e["success_tick"] is not None for e in episodes],
                ),
                "strikes": int(sum(e["reset_strikes"] for e in episodes)),
            },
        )
    a, b = arms
    assert (a["seeds"] == b["seeds"]).all(), (
        f"seed sets differ: {path_a} vs {path_b} — not a paired read"
    )
    return a, b


def mcnemar_table(succ_a: np.ndarray, succ_b: np.ndarray) -> dict[str, int]:
    return {
        "both_succeed": int((succ_a & succ_b).sum()),
        "a_only": int((succ_a & ~succ_b).sum()),
        "b_only": int((~succ_a & succ_b).sum()),
        "both_fail": int((~succ_a & ~succ_b).sum()),
    }


def mcnemar_exact_p(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar p: discordant seeds are Binomial(n, 1/2)
    under H0 (no success-rate difference); doubled one-sided tail,
    capped at 1."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = max(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def paired_read(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    n = len(a["seeds"])
    succ_diff = a["success"].astype(float) - b["success"].astype(float)
    count_low, count_high = bootstrap_ci(succ_diff)  # mean scale -> count
    table = mcnemar_table(a["success"], b["success"])
    prog_delta = a["progress_final"] - b["progress_final"]
    prog_low, prog_high = bootstrap_ci(prog_delta)
    return {
        "n_seeds": n,
        "success": {
            "count_a": int(a["success"].sum()),
            "count_b": int(b["success"].sum()),
            "count_delta": int(a["success"].sum() - b["success"].sum()),
            "count_delta_ci95": [round(count_low * n, 4), round(count_high * n, 4)],
            "ci_excludes_zero": bool(count_low > 0 or count_high < 0),
        },
        "discordant": {
            **table,
            "mcnemar_exact_p_two_sided": mcnemar_exact_p(
                table["a_only"],
                table["b_only"],
            ),
        },
        "progress": {
            "mean_delta_cm": round(float(prog_delta.mean()), 4),
            "ci95": [round(prog_low, 4), round(prog_high, 4)],
            "ci_excludes_zero": bool(prog_low > 0 or prog_high < 0),
            "win_rate": round(float((prog_delta > 0).mean()), 4),
            "tie_rate": round(float((prog_delta == 0).mean()), 4),
        },
        "reset_strikes": {"a": a["strikes"], "b": b["strikes"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True, help="candidate arm JSON")
    parser.add_argument("--b", type=Path, required=True, help="baseline arm JSON")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    a, b = load_paired(args.a, args.b)
    label_a = args.label_a or args.a.parent.name
    label_b = args.label_b or args.b.parent.name
    result = {
        "prereg": PREREG,
        "role": "recorded non-gating read (rides alongside the frozen absolute bands)",
        "arms": {
            "a": {"label": label_a, "path": str(args.a), "config": a["config"]},
            "b": {"label": label_b, "path": str(args.b), "config": b["config"]},
        },
        "read": paired_read(a, b),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"paired read: a={label_a} b={label_b}")
    print(json.dumps(result["read"], indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
