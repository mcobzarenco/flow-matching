"""Frozen reads for the 100-seed sim policy eval
(posts/2026-08-11-prereg-sim-policy-eval-100seeds.md).

Consumes the per-arm JSONs `sim/rollout_sim.py --out-json` wrote,
enforces the registered gates, and banks one analysis JSON:

- per-arm summary: mean/median progress_final (PRIMARY), progress_min,
  success rate, median success tick, latency;
- paired per-seed deltas (bootstrap CI95, seed 0, 10k resamples) for
  every registered pair;
- the validation ordering read: rungs ranked by mean progress_final vs
  the banked panel MAE ordering — the five pairs with panel gap >= 0.1
  must rank correctly, (er55k, er60k) is record-only; Spearman rho and
  the max panel gap among misranked pairs (SIMPLER-style violation
  weight) are reported;
- gates: reset strikes == 0 everywhere, hold-arm floor |mean| < 0.5 cm.

Usage:
  uv run python fontaine/scripts/sim100_reads.py \
      --in-dir outputs/sim/eval100 --out reports/analysis__sim100_seed_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Banked panel fast-path core MAE (k4l2), the validation anchor — from
# analysis__er{15,35,55,60}k_panel_vs_banked_k4l2.json (reports.md).
PANEL_MAE = {
    "er15k": 7.5283,
    "er35k": 6.2892,
    "er55k": 5.8269,
    "er60k": 5.7782,
}
# Pairs with panel gap >= 0.1 MAE must rank correctly in sim (the
# registered expectation); the (er55k, er60k) gap is 0.0487 —
# record-only either way.
GATED_PAIR_MIN_GAP = 0.1

ARMS = ("er60k", "hold", "er15k", "er35k", "er55k")
HOLD_FLOOR_CM = 0.5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


def load_arm(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    episodes = payload["episodes"]
    return {
        "config": payload["config"],
        "seeds": np.array([e["seed"] for e in episodes]),
        "progress_final": np.array([e["progress_final_cm"] for e in episodes]),
        "progress_min": np.array([e["progress_cm"] for e in episodes]),
        "success": np.array([e["success_tick"] is not None for e in episodes]),
        "success_ticks": [
            e["success_tick"] for e in episodes if e["success_tick"] is not None
        ],
        "strikes": np.array([e["reset_strikes"] for e in episodes]),
        "latency_ms": np.array(
            [v for e in episodes for v in e["latency_ms"]],
            dtype=float,
        ),
    }


def bootstrap_ci(deltas: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(deltas, size=(BOOTSTRAP_RESAMPLES, len(deltas)), replace=True)
    means = draws.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def spearman_rho(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ca, cb = ra - ra.mean(), rb - rb.mean()
    return float((ca * cb).sum() / np.sqrt((ca**2).sum() * (cb**2).sum()))


def ordering_read(mean_progress: dict[str, float]) -> dict[str, Any]:
    """The validation read: sim mean progress vs banked panel MAE.

    Correct ranking for a pair = the panel-better rung (lower MAE) has
    strictly higher sim progress. Gated pairs are those with panel gap
    >= GATED_PAIR_MIN_GAP.
    """
    rungs = sorted(PANEL_MAE)
    pairs = []
    violations = []
    for i, a in enumerate(rungs):
        for b in rungs[i + 1 :]:
            panel_gap = abs(PANEL_MAE[a] - PANEL_MAE[b])
            panel_better = a if PANEL_MAE[a] < PANEL_MAE[b] else b
            other = b if panel_better == a else a
            correct = mean_progress[panel_better] > mean_progress[other]
            gated = panel_gap >= GATED_PAIR_MIN_GAP
            pairs.append(
                {
                    "pair": [a, b],
                    "panel_gap_mae": round(panel_gap, 4),
                    "panel_better": panel_better,
                    "sim_correct": bool(correct),
                    "gated": gated,
                },
            )
            if gated and not correct:
                violations.append(round(panel_gap, 4))
    return {
        "pairs": pairs,
        "gated_pairs_correct": sum(p["gated"] and p["sim_correct"] for p in pairs),
        "gated_pairs_total": sum(p["gated"] for p in pairs),
        "expectation_met": not violations,
        "max_violation_panel_gap": max(violations) if violations else 0.0,
        "spearman_rho_progress_vs_neg_panel": spearman_rho(
            [mean_progress[r] for r in rungs],
            [-PANEL_MAE[r] for r in rungs],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    arms = {name: load_arm(args.in_dir / f"{name}.json") for name in ARMS}

    # Gate: identical seed lists (paired design) + zero reset strikes.
    seeds = arms["er60k"]["seeds"]
    for name, arm in arms.items():
        assert (arm["seeds"] == seeds).all(), f"{name}: seed list mismatch"
    total_strikes = int(sum(arm["strikes"].sum() for arm in arms.values()))

    summary = {}
    for name, arm in arms.items():
        summary[name] = {
            "mean_progress_final_cm": round(float(arm["progress_final"].mean()), 4),
            "median_progress_final_cm": round(
                float(np.median(arm["progress_final"])),
                4,
            ),
            "mean_progress_min_cm": round(float(arm["progress_min"].mean()), 4),
            "success_rate": round(float(arm["success"].mean()), 4),
            "median_success_tick": (
                float(np.median(arm["success_ticks"])) if arm["success_ticks"] else None
            ),
            "median_latency_ms": (
                round(float(np.median(arm["latency_ms"])), 1)
                if len(arm["latency_ms"])
                else None
            ),
            "panel_mae": PANEL_MAE.get(name),
        }

    paired = {}
    for a, b in [
        ("er60k", "hold"),
        ("er60k", "er15k"),
        ("er60k", "er35k"),
        ("er60k", "er55k"),
        ("er55k", "er35k"),
        ("er35k", "er15k"),
    ]:
        deltas = arms[a]["progress_final"] - arms[b]["progress_final"]
        low, high = bootstrap_ci(deltas)
        paired[f"{a}_minus_{b}"] = {
            "mean_delta_cm": round(float(deltas.mean()), 4),
            "ci95": [round(low, 4), round(high, 4)],
            "ci_excludes_zero": bool(low > 0 or high < 0),
            "win_rate": round(float((deltas > 0).mean()), 4),
        }

    mean_progress = {r: summary[r]["mean_progress_final_cm"] for r in PANEL_MAE}
    hold_mean = summary["hold"]["mean_progress_final_cm"]
    gates = {
        "reset_strikes_total": total_strikes,
        "reset_strikes_gate": total_strikes == 0,
        "hold_floor_cm": hold_mean,
        "hold_floor_gate": abs(hold_mean) < HOLD_FLOOR_CM,
    }

    result = {
        "prereg": "posts/2026-08-11-prereg-sim-policy-eval-100seeds.md",
        "n_seeds": len(seeds),
        "configs": {name: arm["config"] for name, arm in arms.items()},
        "gates": gates,
        "summary": summary,
        "paired": paired,
        "ordering": ordering_read(mean_progress),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("gates", "summary")}, indent=1))
    print(f"ordering: {json.dumps(result['ordering'], indent=1)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
