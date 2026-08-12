"""Frozen reads for the GRPO signal probe (pre-reg 2026-08-12).

Implements posts/2026-08-12-prereg-grpo-signal-probe.md "Reads":
  1. primary: per-seed within-group std of progress_final_cm over the
     stochastic draws; statistic = median over seeds; bar >= 0.25 cm.
     ddof was not frozen: primary uses ddof=0 (the population std GRPO
     itself normalizes advantages by); ddof=1 is recorded alongside.
  2. non-degeneracy (record-only): fraction of groups with std >= 0.05.
  3. competence cost: mean over seeds of (group mean - anchor), 10k
     bootstrap CI95 paired by seed. Anchors: cells 1-2 -> the
     anchor_er60k_greedy pass; cells 3-4 -> in-cell draw 0; cell 5 ->
     the anchor_teacher80k_euler10 pass.
  4. guards (record-only): knock-aways (progress_final_cm <= -1),
     final_upright < 0.9, reset strikes (validity: must be 0),
     successes; best-point (initial_cm - min_cm) group std alongside.

Partial cells are reported as PARTIAL and excluded from decision-rule
lines (pre-reg: partial cells are discarded, never pooled).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = list(range(15))
BAR_CM = 0.25
NONDEG_CM = 0.05

# pass dir -> (label, expected stochastic draws, anchor spec)
# anchor spec: ("pass", dir) | ("draw0",) — draw 0 excluded from groups.
CELLS = {
    "cell1_er60k_t10": (
        "cell 1  er60k    AR sampled t=1.0",
        8,
        ("pass", "anchor_er60k_greedy"),
    ),
    "cell2_er60k_t16": (
        "cell 2  er60k    AR sampled t=1.6",
        8,
        ("pass", "anchor_er60k_greedy"),
    ),
    "cell3_teacher80k_heun30": (
        "cell 3  teacher80k ODE heun-30 fresh noise",
        8,
        ("draw0",),
    ),
    "cell4_ftrig4k_euler1": ("cell 4  ftrig4k  ODE euler-1 fresh noise", 8, ("draw0",)),
    "cell5_teacher80k_sde05": (
        "cell 5  teacher80k SDE euler-10 a=0.5",
        8,
        ("pass", "anchor_teacher80k_euler10"),
    ),
    "cell5b_teacher80k_sde03": (
        "cell 5b teacher80k SDE euler-10 a=0.3 (hedge)",
        8,
        ("pass", "anchor_teacher80k_euler10"),
    ),
}


def load_rows(out_dir: Path, name: str) -> list[dict]:
    p = out_dir / name / "rows.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open() if line.strip()]


def anchor_by_seed(rows: list[dict]) -> dict[int, float]:
    return {r["seed"]: r["progress_final_cm"] for r in rows}


def read_cell(
    rows: list[dict],
    k_expected: int,
    anchor: dict[int, float] | None,
    *,
    in_cell_draw0: bool,
) -> dict:
    by_seed: dict[int, list[dict]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r)

    anc = dict(anchor) if anchor is not None else {}
    groups: dict[int, list[dict]] = {}
    for seed, rs in by_seed.items():
        if in_cell_draw0:
            d0 = [r for r in rs if r["draw"] == 0]
            if d0:
                anc[seed] = d0[0]["progress_final_cm"]
            rs = [r for r in rs if r["draw"] != 0]
        if len(rs) == k_expected:
            groups[seed] = rs

    complete = sorted(groups)
    out: dict = {
        "n_rows": len(rows),
        "complete_groups": len(complete),
        "partial": len(complete) < len(SEEDS),
    }
    if not complete:
        return out

    fin = {s: np.array([r["progress_final_cm"] for r in groups[s]]) for s in complete}
    best = {
        s: np.array([r["initial_cm"] - r["min_cm"] for r in groups[s]])
        for s in complete
    }
    stds0 = np.array([fin[s].std(ddof=0) for s in complete])
    stds1 = np.array([fin[s].std(ddof=1) for s in complete])
    out.update(
        median_std=float(np.median(stds0)),
        median_std_ddof1=float(np.median(stds1)),
        nondeg_frac=float((stds0 >= NONDEG_CM).mean()),
        best_median_std=float(np.median([best[s].std(ddof=0) for s in complete])),
        clears_bar=bool(np.median(stds0) >= BAR_CM),
        per_seed_std={int(s): float(fin[s].std(ddof=0)) for s in complete},
    )

    flat = [r for s in complete for r in groups[s]]
    out["guards"] = {
        "knock_aways": sum(r["progress_final_cm"] <= -1.0 for r in flat),
        "tipped": sum(r["final_upright"] < 0.9 for r in flat),
        "reset_strikes": sum(r["reset_strikes"] for r in flat),
        "successes": sum(r["success_tick"] is not None for r in flat),
        "episodes": len(flat),
    }

    paired = [s for s in complete if s in anc]
    if paired:
        deltas = np.array([fin[s].mean() - anc[s] for s in paired])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(deltas), size=(10_000, len(deltas)))
        boot = deltas[idx].mean(axis=1)
        out["competence_cost"] = {
            "n_seeds": len(paired),
            "mean": float(deltas.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/sim/grpo_signal_probe")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    anchors = {
        name: anchor_by_seed(load_rows(out_dir, name))
        for name in ("anchor_er60k_greedy", "anchor_teacher80k_euler10")
    }
    results: dict[str, dict] = {}
    for name, (label, k, anchor_spec) in CELLS.items():
        rows = load_rows(out_dir, name)
        if not rows:
            continue
        in_cell = anchor_spec[0] == "draw0"
        anchor = None if in_cell else anchors.get(anchor_spec[1])
        results[name] = {
            "label": label,
            **read_cell(rows, k, anchor, in_cell_draw0=in_cell),
        }

    if args.json:
        print(
            json.dumps(
                {"anchors": {k: len(v) for k, v in anchors.items()}, "cells": results},
                indent=1,
            ),
        )
        return

    for name, a in anchors.items():
        print(f"[anchor] {name}: {len(a)}/15 seeds")
    for r in results.values():
        state = "PARTIAL" if r.get("partial") else "COMPLETE"
        line = f"[{state}] {r['label']}: rows {r['n_rows']}, groups {r.get('complete_groups', 0)}/15"
        if "median_std" in r:
            line += (
                f"\n  primary median group std {r['median_std']:.3f} cm (ddof1 {r['median_std_ddof1']:.3f})"
                f" vs bar {BAR_CM} -> {'CLEARS' if r['clears_bar'] else 'below'}"
                f"\n  non-degeneracy {r['nondeg_frac']:.2f} >= {NONDEG_CM} cm; best-point median std {r['best_median_std']:.3f}"
            )
            g = r["guards"]
            line += (
                f"\n  guards: knock {g['knock_aways']}, tipped {g['tipped']}, strikes {g['reset_strikes']},"
                f" successes {g['successes']} / {g['episodes']} eps"
            )
            if "competence_cost" in r:
                c = r["competence_cost"]
                line += (
                    f"\n  competence cost {c['mean']:+.3f} cm, CI95 [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
                    f" (paired, n={c['n_seeds']})"
                )
        print(line)


if __name__ == "__main__":
    main()
