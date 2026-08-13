"""Amendment-1 paired read: arm B (grammar-masked) vs arm A
(reference greedy), same 100 seeds — posts/2026-08-13-prereg-
molmoact2-ar100.md. Prints the frozen reads + a paired bootstrap CI95
on the per-seed progress_final_cm delta (B − A)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
A = REPO_ROOT / "outputs/sim/molmoact2_ar100/rows.json"
B = REPO_ROOT / "outputs/sim/molmoact2_ar100b_masked/rows.json"


def load(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text())
    rows = {r["seed"]: r for r in data["episodes"]}
    if len(rows) != 100:
        sys.exit(f"{path}: {len(rows)} seeds, expected 100 — stop")
    return rows


def facts(rows: dict[int, dict]) -> dict:
    final = np.array([rows[s]["progress_final_cm"] for s in sorted(rows)])
    best = np.array(
        [rows[s]["initial_cm"] - rows[s]["min_cm"] for s in sorted(rows)],
    )
    return {
        "successes": sorted(
            s for s, r in rows.items() if r.get("success_tick") is not None
        ),
        "strikes": int(sum(r["reset_strikes"] for r in rows.values())),
        "mean_final": round(float(final.mean()), 3),
        "median_final": round(float(np.median(final)), 3),
        "engaged_best_1cm": int((best > 1.0).sum()),
        "knock_le_-1": int((final <= -1.0).sum()),
    }


def main() -> None:
    a, b = load(A), load(B)
    fa, fb = facts(a), facts(b)
    print("arm A (reference):", json.dumps(fa))
    print("arm B (masked):   ", json.dumps(fb))
    seeds = sorted(a)
    delta = np.array(
        [b[s]["progress_final_cm"] - a[s]["progress_final_cm"] for s in seeds],
    )
    rng = np.random.default_rng(0)
    boots = np.array(
        [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(10_000)],
    )
    low, high = np.percentile(boots, [2.5, 97.5])
    print(
        f"paired delta (B - A) progress_final_cm: mean {delta.mean():+.3f} "
        f"CI95 [{low:+.3f}, {high:+.3f}] "
        f"({'EXCLUDES' if low > 0 or high < 0 else 'includes'} 0)",
    )
    moved = int((delta != 0).sum())
    print(f"seeds with any behavioral difference: {moved}/100")
    bj = json.loads(B.read_text())["config"]["molmoact2_discrete"]
    print(
        f"arm B fallbacks: {bj['zero_fallbacks']}/{bj['predicts']} "
        "(must be 0 — masked decodes by construction)",
    )


if __name__ == "__main__":
    main()
