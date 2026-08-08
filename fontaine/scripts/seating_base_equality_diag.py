"""Seating base-equality diagnosis (pre-reg
2026-08-08-prereg-noise-ladder-perdataset.md, Amendment 2) — the
drift-vs-keying adjudication owed after the seating arm's 4dp
base-equality abort (re-run 5.3645/1.4241 vs banked 5.3645/1.4242).

The banked 2026-08-05 mean-of-10 run retained no per-frame npz (that
gap is the seating arm's reason to exist), so the diagnosis runs at
the finest banked granularity: the 878-cell per-dataset table plus
per-motor and quantile rows of the two report jsons.

Discriminator: different noise draws move small per-dataset cells at
draw-level dispersion (~0.05-0.5; single-draw ticket spread on this
panel is 5.71-9.37). Numeric drift from the batched-solver merge
(owner 2ee2be5, merged 85cdc0a 2026-08-07: sequential per-draw solver
calls at batch 32 -> one tiled call at batch 320) moves them at
~1e-3. The two hypotheses are two orders of magnitude apart at
4-frame cells; the state-copy columns (sampling-independent) must be
exactly equal under either.

Run:  uv run python fontaine/scripts/seating_base_equality_diag.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]

RUN_STEM = "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
BANKED_JSON = f"{RUN_STEM}__panel_curated_v0_k4l2_draws10_heun30.json"
SEATING_JSON = f"{RUN_STEM}__panel_curated_v0_k4l2_draws10_seating_heun30.json"
POLICY = "bijou@80000_draws10"

# Amendment-2 bounds (see the pre-reg post for the calibration):
TOL_POOLED = 5e-4  # 4x the observed batched-solver drift
TOL_CELL = 5e-3  # noise-resample exclusion bound at 4-frame cells


def _fail(message: str) -> None:
    raise SystemExit(f"ABORT: {message}")


def _bijou_row(report: dict[str, Any], label: str) -> dict[str, Any]:
    rows = [s for s in report["summaries"] if str(s["policy"]) == POLICY]
    if len(rows) != 1:
        _fail(f"{label} has {len(rows)} {POLICY!r} summaries, want 1")
    return rows[0]


def diagnose(
    banked: dict[str, Any],
    seating: dict[str, Any],
) -> dict[str, Any]:
    rb, rs = _bijou_row(banked, "banked json"), _bijou_row(seating, "seating json")

    pooled = {
        key: {
            "banked": rb[key],
            "seating": rs[key],
            "delta": rs[key] - rb[key],
        }
        for key in ("chunk_mae", "first_mae", "chunk_mse", "mae_p50", "mae_p90")
    }
    per_motor = [
        s - b for b, s in zip(rb["per_motor_mae"], rs["per_motor_mae"], strict=True)
    ]

    pdb, pds = banked["per_dataset"], seating["per_dataset"]
    if set(pdb) != set(pds):
        _fail("per-dataset key sets differ between the two jsons")

    state_copy_max = 0.0
    cells: list[tuple[float, int, str]] = []
    for key in pdb:
        eb, es = pdb[key], pds[key]
        if eb["frames"] != es["frames"]:
            _fail(f"frame count differs in cell {key!r}")
        sc = abs(es["chunk_mae"]["state-copy"] - eb["chunk_mae"]["state-copy"])
        state_copy_max = max(state_copy_max, sc)
        cells.append(
            (es["chunk_mae"][POLICY] - eb["chunk_mae"][POLICY], eb["frames"], key),
        )
    absd = sorted((abs(d) for d, _, _ in cells), reverse=True)
    n = len(absd)
    small = [abs(d) for d, frames, _ in cells if frames <= 30]

    # -- the adjudication clauses (Amendment 2 gate (i), b/c/d)
    if state_copy_max != 0.0:
        _fail(
            f"state-copy cells differ (max |delta| {state_copy_max:.3e}) — "
            "rows/truth/pooling NOT reproduced; this is not solver drift",
        )
    for key in ("chunk_mae", "first_mae"):
        if abs(pooled[key]["delta"]) > TOL_POOLED:
            _fail(
                f"pooled {key} delta {pooled[key]['delta']:+.3e} exceeds "
                f"the batched-solver drift envelope {TOL_POOLED:g}",
            )
    worst = max(absd)
    if worst > TOL_CELL:
        _fail(
            f"per-dataset cell delta {worst:.3e} exceeds {TOL_CELL:g} — "
            "consistent with resampled noise, NOT numeric drift; the "
            "index-keying reproduction claim fails",
        )

    return {
        "inputs": {"banked_json": BANKED_JSON, "seating_json": SEATING_JSON},
        "verdict": (
            "BENIGN NUMERIC DRIFT — noise reproduction confirmed, "
            "resampling excluded; mechanism = batched-solver merge "
            "2ee2be5/85cdc0a (sequential batch-32 -> tiled batch-320)"
        ),
        "pooled": {
            k: {kk: round(vv, 7) for kk, vv in v.items()} for k, v in pooled.items()
        },
        "per_motor_delta": [round(d, 7) for d in per_motor],
        "per_dataset": {
            "cells": n,
            "abs_delta_max": round(absd[0], 7),
            "abs_delta_p99": round(absd[n // 100], 7),
            "abs_delta_median": round(absd[n // 2], 7),
            "cells_le_30_frames": len(small),
            "small_cell_abs_delta_max": round(max(small), 7),
            "state_copy_abs_delta_max": state_copy_max,
        },
        "bounds": {"tol_pooled": TOL_POOLED, "tol_cell": TOL_CELL},
        "context": {
            "draw_level_dispersion": "single-draw ticket spread 5.71-9.37 "
            "(goldenticket stage 1) — resampled noise moves 4-frame cells "
            "~0.05-0.5, two orders above the observed bound",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "reports/analysis__seating_base_equality_diag.json",
    )
    args = parser.parse_args()
    out = diagnose(
        json.loads((REPO / BANKED_JSON).read_text()),
        json.loads((REPO / SEATING_JSON).read_text()),
    )
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"verdict: {out['verdict']}")
    print(
        f"pooled deltas: chunk {out['pooled']['chunk_mae']['delta']:+.3e} "
        f"first {out['pooled']['first_mae']['delta']:+.3e}",
    )
    print(
        f"per-dataset |delta|: max {out['per_dataset']['abs_delta_max']:.3e} "
        f"median {out['per_dataset']['abs_delta_median']:.3e} "
        f"(state-copy max {out['per_dataset']['state_copy_abs_delta_max']})",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
