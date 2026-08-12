"""Registered reads for the 20-seed v3 behavioral spot-check
(posts/2026-08-12-prereg-sim-spot20-v3.md): per-arm paired per-seed
delta progress vs the banked sim100 v0 rows, engagement split,
integrity tripwires.

Usage:
  uv run python fontaine/scripts/spot20_reads.py \
      --spot-dir outputs/sim/spot20 \
      --baseline-dir outputs/sim/eval100 \
      --arms er60k snap30k teacher80k \
      --out reports/analysis__spot20_v3_reads.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

N_BOOT = 10_000
ENGAGE_CM = 1.0


def rows_by_seed(path: Path, seeds: list[int]) -> dict[int, dict]:
    data = json.loads(path.read_text())
    rows = {int(e["seed"]): e for e in data["episodes"]}
    missing = [s for s in seeds if s not in rows]
    if missing:
        raise SystemExit(f"{path}: missing seeds {missing}")
    return {s: rows[s] for s in seeds}


def progress_final(row: dict) -> float:
    return row["initial_cm"] - row["final_cm"]


def progress_best(row: dict) -> float:
    return row["initial_cm"] - row["min_cm"]


def bootstrap_ci(deltas: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = rng.choice(deltas, size=(N_BOOT, len(deltas)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot-dir", type=Path, default=Path("outputs/sim/spot20"))
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("outputs/sim/eval100"),
    )
    parser.add_argument("--arms", nargs="+", default=["er60k", "snap30k", "teacher80k"])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    rng = np.random.default_rng(0)
    results: dict[str, object] = {}
    for arm in args.arms:
        v3 = rows_by_seed(args.spot_dir / f"{arm}.json", seeds)
        v0 = rows_by_seed(args.baseline_dir / f"{arm}.json", seeds)
        # Integrity tripwires (registered): physics identity.
        strikes = [s for s in seeds if v3[s]["reset_strikes"] != 0]
        spawn_mismatch = [
            s for s in seeds if tuple(v3[s]["spawn_xy"]) != tuple(v0[s]["spawn_xy"])
        ]
        deltas = np.array(
            [progress_final(v3[s]) - progress_final(v0[s]) for s in seeds],
        )
        lo, hi = bootstrap_ci(deltas, rng)
        engage_v3 = sum(progress_best(v3[s]) > ENGAGE_CM for s in seeds)
        engage_v0 = sum(progress_best(v0[s]) > ENGAGE_CM for s in seeds)
        latency = float(
            np.mean([np.mean(v3[s]["latency_ms"]) for s in seeds]),
        )
        results[arm] = {
            "tripwires": {
                "reset_strikes_nonzero_seeds": strikes,
                "spawn_xy_mismatch_seeds": spawn_mismatch,
                "void": bool(strikes or spawn_mismatch),
            },
            "paired_delta_progress_final_cm": {
                "mean": round(float(deltas.mean()), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "positive_seeds": int((deltas > 0).sum()),
                "negative_seeds": int((deltas < 0).sum()),
                "per_seed": [round(float(d), 3) for d in deltas],
            },
            "means": {
                "v3_progress_final_cm": round(
                    float(np.mean([progress_final(v3[s]) for s in seeds])),
                    4,
                ),
                "v0_progress_final_cm": round(
                    float(np.mean([progress_final(v0[s]) for s in seeds])),
                    4,
                ),
            },
            "engagement_gt1cm_of_20": {"v3": engage_v3, "v0": engage_v0},
            "mean_replan_latency_ms": round(latency, 1),
        }
        d = results[arm]["paired_delta_progress_final_cm"]  # type: ignore[index]
        print(
            f"{arm}: dP {d['mean']:+.3f} cm CI [{d['ci95'][0]:+.3f}, "
            f"{d['ci95'][1]:+.3f}] signs +{d['positive_seeds']}/-"
            f"{d['negative_seeds']} | engage {engage_v0}->{engage_v3} | "
            f"void={results[arm]['tripwires']['void']}",  # type: ignore[index]
        )

    payload = {
        "config": {
            "prereg": "posts/2026-08-12-prereg-sim-spot20-v3.md",
            "seeds": seeds,
            "engage_cm": ENGAGE_CM,
            "n_boot": N_BOOT,
            "commit": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
        },
        "arms": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
