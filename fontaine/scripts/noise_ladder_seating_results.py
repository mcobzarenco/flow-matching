"""Noise-ladder rung-2 SEATING read (pre-reg
2026-08-08-prereg-noise-ladder-perdataset.md, "Folded-in arm") — the
offline adjudication that runs after eval_flow80k_noiseladder_seating.sh
exits rc=0.

The frozen read: paired per-frame chunk MAE, mean-of-top-10-tickets
(the BANKED stage-3 npz — never re-run) minus mean-of-random-10 (the
seating re-run, dumps retained), pooled over core rows with a
bootstrap CI95. CI entirely below 0 => the top-10 ensemble takes the
mean-of-10 board row; otherwise the row stands and R3 stays a
record. The pre-reg wrote "paired CI95" without a clustering clause,
so the verdict CI is the repo's paired-panel default (frame-level
bootstrap, BOOT_N/BOOT_SEED — the box-batch/selfsubgoal convention);
a dataset-clustered CI is recorded alongside, record-only.

Gates BEFORE the read (abort loud, never re-tolerance):
  i.   base-equality — the seating re-run's REPORT reproduces the
       banked mean-of-10 row 5.3645/1.4242 at 4dp (the launcher's
       oracle, re-enforced here); the npz-side core_pool recompute
       is recorded alongside, not gated (report pooling is the
       banked row's own convention).
  ii.  identity alignment — index/repo_id/episode/frame/core columns
       bit-equal between the two npzs.
  iii. anchor equality — the banked top-10 npz still core_pools to
       the stage-3 record 5.1847/1.3831 at 4dp (verified live
       against the real npz before this script was frozen).

Run:  uv run python fontaine/scripts/noise_ladder_seating_results.py
Test: tests/test_seating_results.py (planted worlds; check.py tier).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(_HERE))

from box_batch_results import bootstrap_ci
from draws_fairness import element_mask, frame_mae
from goldenticket_stage2_results import load_npz, policy_key
from goldenticket_stage3_results import core_pool
from noise_ladder_rung2_results import clustered_ci

RUN_STEM = "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000"
TOP10_NPZ = f"{RUN_STEM}__panel_curated_v0_k4l2_top10tickets_heun30.npz"
SEATING_NPZ = f"{RUN_STEM}__panel_curated_v0_k4l2_draws10_seating_heun30.npz"
SEATING_JSON = f"{RUN_STEM}__panel_curated_v0_k4l2_draws10_seating_heun30.json"

# Banked anchors (frozen in the pre-reg / stage-3 record):
BANKED_MEAN10_CHUNK = 5.3645  # 2026-08-05 random mean-of-10 board row
BANKED_MEAN10_FIRST = 1.4242
BANKED_TOP10_CHUNK = 5.1847  # goldenticket stage-3 R3
BANKED_TOP10_FIRST = 1.3831
TIE_BAND = 0.02  # R3's tie band, kept for context only


def _fail(message: str) -> None:
    raise SystemExit(f"ABORT: {message}")


def seating_read(
    top10: dict[str, np.ndarray],
    seating: dict[str, np.ndarray],
    seating_report: dict[str, Any],
    banked_mean10: tuple[float, float] = (BANKED_MEAN10_CHUNK, BANKED_MEAN10_FIRST),
    banked_top10: tuple[float, float] = (BANKED_TOP10_CHUNK, BANKED_TOP10_FIRST),
) -> dict[str, Any]:
    top10_key = policy_key(top10, "top-10 npz")
    seating_key = policy_key(seating, "seating npz")

    # -- gate ii: identity alignment (the paired read's licence)
    for column in ("index", "repo_id", "episode_index", "frame_index", "core"):
        if not np.array_equal(top10[column], seating[column]):
            _fail(f"identity column {column!r} differs between the two npzs")

    core = top10["core"].astype(bool)

    # -- gate i: base-equality on the seating re-run (report + npz)
    rows = [
        s for s in seating_report["summaries"] if str(s["policy"]).startswith("bijou@")
    ]
    if len(rows) != 1:
        _fail(f"seating report has {len(rows)} bijou@ summaries, want 1")
    rep_chunk = round(float(rows[0]["chunk_mae"]), 4)
    rep_first = round(float(rows[0]["first_mae"]), 4)
    if (rep_chunk, rep_first) != banked_mean10:
        _fail(
            f"base-equality FAILED (report): {rep_chunk}/{rep_first} != "
            f"banked {banked_mean10[0]}/{banked_mean10[1]}",
        )
    # npz-side recompute of the same row, stage-3 core_pool
    # convention — RECORDED alongside the report gate (a convention
    # drift between report pooling and core_pool must not fake a
    # base-equality failure; the report gate is the pre-reg's oracle).
    mask_all = element_mask(seating["truth"], seating["valid"])
    npz_chunk, npz_first = core_pool(seating, seating_key, mask_all, core)

    # -- gate iii: the banked top-10 npz still pools to the stage-3 record
    t_chunk, t_first = core_pool(top10, top10_key, mask_all, core)
    if (round(t_chunk, 4), round(t_first, 4)) != banked_top10:
        _fail(
            f"top-10 anchor FAILED: {t_chunk:.4f}/{t_first:.4f} != "
            f"banked {banked_top10[0]}/{banked_top10[1]}",
        )

    # -- the frozen read: paired per-frame delta on core rows
    mask = mask_all
    f_top10 = frame_mae(top10[top10_key], top10["truth"], mask)
    f_seating = frame_mae(seating[seating_key], seating["truth"], mask)
    delta = (f_top10 - f_seating)[core]
    pooled = float(delta.mean())
    lo, hi = bootstrap_ci(delta)
    confirmed = bool(hi < 0.0)

    repo = np.asarray([str(r) for r in top10["repo_id"]])
    clo, chi = clustered_ci(delta, repo[core])

    first_mask = mask[:, 0, :]
    err_t = (np.abs(top10[top10_key] - top10["truth"]) * mask)[:, 0, :]
    err_s = (np.abs(seating[seating_key] - seating["truth"]) * mask)[:, 0, :]
    nvalid = np.maximum(first_mask.sum(axis=1), 1)
    d_first = (err_t.sum(axis=1) - err_s.sum(axis=1)) / nvalid
    d_first = d_first[core]
    flo, fhi = bootstrap_ci(d_first)

    return {
        "inputs": {
            "top10_npz": TOP10_NPZ,
            "seating_npz": SEATING_NPZ,
            "top10_policy": top10_key,
            "seating_policy": seating_key,
        },
        "rows": {"panel": int(core.shape[0]), "core": int(core.sum())},
        "gates": {
            "base_equality_report": [rep_chunk, rep_first],
            "base_equality_npz_record": [round(npz_chunk, 4), round(npz_first, 4)],
            "top10_anchor": [round(t_chunk, 4), round(t_first, 4)],
        },
        "read_paired": {
            "delta_pooled": round(pooled, 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "pass_rule": "CI95 entirely below 0",
            "verdict": "CONFIRMED — board row moves to the top-10 ensemble"
            if confirmed
            else "NOT-CONFIRMED — mean-of-10 row stands, R3 stays a record",
            "ci95_clustered_record_only": [round(clo, 5), round(chi, 5)],
            "r3_unpaired_context": {
                "delta": round(BANKED_TOP10_CHUNK - BANKED_MEAN10_CHUNK, 4),
                "tie_band": TIE_BAND,
            },
        },
        "first_mirror_record_only": {
            "delta_pooled": round(float(d_first.mean()), 5),
            "ci95": [round(flo, 5), round(fhi, 5)],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "reports/analysis__noise_ladder_seating.json",
    )
    args = parser.parse_args()
    out = seating_read(
        load_npz(TOP10_NPZ),
        load_npz(SEATING_NPZ),
        json.loads((REPO / SEATING_JSON).read_text()),
    )
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    read = out["read_paired"]
    print(
        f"seating read: paired Δ {read['delta_pooled']} "
        f"CI95 {read['ci95']} -> {read['verdict']}",
    )
    print(
        f"  clustered CI (record) {read['ci95_clustered_record_only']}; "
        f"first mirror {out['first_mirror_record_only']}",
    )
    print(f"gates: {out['gates']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
