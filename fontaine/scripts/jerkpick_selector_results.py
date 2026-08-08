"""Jerk-pick selector read — RECORD-ONLY, exploratory (queue item
idea19-jerkpick-selector-read; lit: papers/noise-space-steering-3.md).

SDN (2606.14084) selects among sampled action decodes with a two-stage
rule whose ablation says the cheap half — pick the minimum-jerk
candidate — carries most of the real-robot gain (+16.7 of +18.3 pp).
Jerk-pick is a pure function of a --dump-draws stack: no forwards, no
labels, no judge. This read places it on the selection ladder for a
banked stack:

    average single draw  ->  mean-of-N ensemble  ->  jerk-pick
                         ->  oracle best-of-N

All pooled core-row chunk MAE through the draws_fairness conventions
(element_mask/pooled_mae verbatim), first_mae mirrors alongside.
Diagnostics: jerk-pick vs oracle argmin agreement rate, pooled
Spearman(jerk score, per-draw frame MAE), and the fraction of the
oracle gap (single - bestN) the jerk-pick recovers. NO decision rule:
like selection_ceiling_results.py this bounds/locates a selector family
before anyone builds one; escalation needs its own pre-reg.

Jerk score (SDN's S(A), discrete third difference):
    S = sqrt( mean_t || a[t+3] - 3 a[t+2] + 3 a[t+1] - a[t] ||^2 )
over triplets whose four steps are all valid; frames with no valid
triplet fall back to draw 0 (recorded, excluded from agreement stats).

Guards (abort, never silent): draws stack present, width == the
sample_draws metadata, >= 2 draws; identity/valid/truth columns
present.

Oracle mode (--oracle):
  (a) anchor reproduction: the banked flow drawsprobe draws-10 stack's
      per-draw pooled MAEs reproduce analysis__sigma_draw_direct.json's
      10 values exactly (4 dp);
  (b) jerk formula exactness: a cubic-free (quadratic) trajectory has
      zero third difference; a planted spike has the hand-computed
      value;
  (c) planted selector geometry: when the smoothest draw is also the
      most accurate, jerk-pick == oracle and the ladder collapses
      accordingly; when smoothness is planted anti-accurate, jerk-pick
      lands on the worst draw (the falsifiable direction);
  (d) refusals: width/metadata mismatch, < 2 draws, no valid triplet
      fallback recorded.

Pure CPU, read-only on inputs, deterministic.

  uv run python fontaine/scripts/jerkpick_selector_results.py \\
      [--npz <..._draws.npz>] [--label <name>] \\
      [--out reports/analysis__jerkpick_selector.json]
  uv run python fontaine/scripts/jerkpick_selector_results.py --oracle
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
sys.path.insert(0, str(_HERE))
from draws_fairness import element_mask, pooled_mae, spearman

DEFAULT_NPZ = (
    "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__"
    "panel_curated_v0_k4l2_drawsprobe_s7_draws10_heun30.npz"
)
SIGMA_DIRECT_JSON = "reports/analysis__sigma_draw_direct.json"


def _fail(message: str) -> None:
    raise SystemExit(f"ABORT: {message}")


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(REPO / path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def jerk_scores(draws: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[F, N] RMS third-difference per draw + [F] bool 'has a valid
    triplet'. Triplets whose four steps are not all valid are dropped."""
    d3 = (
        draws[:, :, 3:]
        - 3 * draws[:, :, 2:-1]
        + 3 * draws[:, :, 1:-2]
        - draws[:, :, :-3]
    )
    v = valid.astype(bool)
    tri = v[:, 3:] & v[:, 2:-1] & v[:, 1:-2] & v[:, :-3]  # [F, L-3]
    n_tri = tri.sum(axis=1)  # [F]
    sq = (d3**2).sum(axis=3) * tri[:, None, :]  # [F, N, L-3]
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.sqrt(sq.sum(axis=2) / np.maximum(n_tri, 1)[:, None])
    return s, n_tri > 0


def analyze(d: dict[str, np.ndarray], label: str) -> dict[str, Any]:
    if "draws" not in d:
        _fail(f"{label}: no draws stack in npz")
    stack = d["draws"].astype(np.float64)
    n_draws = int(np.asarray(d["sample_draws"]).ravel()[0])
    if stack.shape[1] != n_draws:
        _fail(f"{label}: stack width {stack.shape[1]} != sample_draws {n_draws}")
    if n_draws < 2:
        _fail(f"{label}: need >= 2 draws for a selector read")
    truth = d["truth"].astype(np.float64)
    valid = d["valid"].astype(bool)
    core = d["core"].astype(bool)
    mask = element_mask(truth, valid)

    err = np.abs(stack - truth[:, None])  # [F, N, L, A]
    w = mask[:, None].astype(np.float64)
    per_draw = (err * w).sum(axis=(2, 3)) / np.maximum(mask.sum(axis=(1, 2)), 1)[
        :,
        None,
    ]

    jerk, has_tri = jerk_scores(stack, valid)
    pick = np.where(has_tri, jerk.argmin(axis=1), 0)
    best = per_draw.argmin(axis=1)
    rows = np.arange(len(pick))
    composite_pick = stack[rows, pick]
    composite_best = stack[rows, best]
    ensemble = stack.mean(axis=1)

    def pool(pred: np.ndarray) -> tuple[float, float]:
        chunk = pooled_mae(pred[core], truth[core], mask[core])
        e = np.abs(pred - truth) * mask
        first = e[core][:, 0, :].sum(-1) / np.maximum(mask[core][:, 0, :].sum(-1), 1)
        return round(float(chunk), 4), round(float(first.mean()), 4)

    # Average single draw: pooled score of each draw, averaged.
    singles = [
        pooled_mae(stack[:, j][core], truth[core], mask[core]) for j in range(n_draws)
    ]
    single_chunk = round(float(np.mean(singles)), 4)
    pick_chunk, pick_first = pool(composite_pick)
    best_chunk, best_first = pool(composite_best)
    ens_chunk, ens_first = pool(ensemble)

    keep = core & has_tri
    agreement = float((pick[keep] == best[keep]).mean())
    sp = spearman(jerk[keep].ravel(), per_draw[keep].ravel())
    gap = single_chunk - best_chunk
    recovered = round((single_chunk - pick_chunk) / gap, 4) if gap > 0 else None

    return {
        "label": label,
        "n_frames": len(core),
        "core": int(core.sum()),
        "n_draws": n_draws,
        "no_triplet_fallback_rows": int((~has_tri).sum()),
        "ladder_chunk": {
            "single_draw_avg": single_chunk,
            "ensemble_mean_of_n": ens_chunk,
            "jerk_pick": pick_chunk,
            "oracle_best_of_n": best_chunk,
        },
        "first_mae": {
            "ensemble": ens_first,
            "jerk_pick": pick_first,
            "oracle": best_first,
        },
        "diagnostics": {
            "oracle_agreement": round(agreement, 4),
            "expected_agreement_null": round(1.0 / n_draws, 4),
            "spearman_jerk_vs_mae": round(sp, 4),
            "oracle_gap_recovered": recovered,
        },
        "record_only": True,
    }


# ----------------------------------------------------------------- oracle


def _fixture(
    n: int = 12,
    n_draws: int = 6,
    *,
    smooth_is_best: bool = True,
) -> dict[str, np.ndarray]:
    steps, dims = 12, 3
    t = np.arange(steps, dtype=np.float64)
    truth = np.stack(
        [np.stack([t * (i + 1) * 0.01] * dims, axis=1) for i in range(n)],
    )
    draws = np.zeros((n, n_draws, steps, dims))
    for j in range(n_draws):
        # Draw j oscillates with amplitude ~ j (jerk increases with j);
        # error magnitude increases (or decreases) with j.
        wave = 0.05 * j * np.sin(np.pi * t / 2.0)[None, :, None]
        rank = j if smooth_is_best else (n_draws - 1 - j)
        off = 0.1 * (1 + rank)
        draws[:, j] = truth + off + wave
    return {
        "truth": truth.astype(np.float64),
        "valid": np.ones((n, steps), dtype=bool),
        "core": np.ones(n, dtype=bool),
        "draws": draws,
        "sample_draws": np.array(n_draws),
        "index": np.arange(n),
    }


def run_oracles() -> None:
    print("oracle (a): banked per-draw pooled MAEs reproduce sigma_draw_direct")
    d = load_npz(DEFAULT_NPZ)
    ref = json.loads((REPO / SIGMA_DIRECT_JSON).read_text())
    want = [
        round(float(v), 4)
        for v in ref["pooled_level_crosscheck"]["per_draw_pooled_mae"]
    ]
    truth, valid, core = d["truth"], d["valid"].astype(bool), d["core"].astype(bool)
    mask = element_mask(truth, valid)
    got = [
        round(pooled_mae(d["draws"][:, j][core], truth[core], mask[core]), 4)
        for j in range(d["draws"].shape[1])
    ]
    assert got == want, f"{got} != {want}"
    print(f"  reproduced {got[:3]}... ({len(got)} draws)")

    print("oracle (b): jerk formula exactness")
    steps = 8
    t = np.arange(steps, dtype=np.float64)
    quad = (0.5 * t**2 + 2 * t + 1)[None, None, :, None]  # third diff == 0
    s, ok = jerk_scores(quad, np.ones((1, steps), dtype=bool))
    assert ok[0] and abs(s[0, 0]) < 1e-12, s
    spike = np.zeros((1, 1, steps, 1))
    spike[0, 0, 4, 0] = 1.0  # third diffs touching index 4: -1, 3, -3, 1
    s, _ = jerk_scores(spike, np.ones((1, steps), dtype=bool))
    want_s = float(np.sqrt((1 + 9 + 9 + 1) / (steps - 3)))
    assert abs(s[0, 0] - want_s) < 1e-12, (s[0, 0], want_s)
    print(f"  quadratic -> 0; planted spike -> {s[0, 0]:.6f} == {want_s:.6f}")

    print("oracle (c): planted selector geometry")
    out = analyze(_fixture(smooth_is_best=True), "smooth-is-best")
    lad, diag = out["ladder_chunk"], out["diagnostics"]
    assert diag["oracle_agreement"] == 1.0, diag
    assert lad["jerk_pick"] == lad["oracle_best_of_n"], lad
    assert lad["single_draw_avg"] > lad["jerk_pick"], lad
    assert diag["spearman_jerk_vs_mae"] > 0.9, diag
    print(f"  aligned: agreement 1.0, ladder {lad}")
    out = analyze(_fixture(smooth_is_best=False), "smooth-is-worst")
    lad, diag = out["ladder_chunk"], out["diagnostics"]
    assert diag["oracle_agreement"] == 0.0, diag
    assert lad["jerk_pick"] > lad["single_draw_avg"], lad
    assert diag["spearman_jerk_vs_mae"] < -0.9, diag
    print("  anti-aligned: jerk-pick lands on the worst draw (falsifier direction)")

    print("oracle (d): refusals + fallback")
    fx = _fixture()
    fx["sample_draws"] = np.array(99)
    try:
        analyze(fx, "bad")
        raise AssertionError("width mismatch did not abort")
    except SystemExit as e:
        print(f"  width mismatch fired: {e}")
    fx = _fixture(n_draws=2)
    fx["draws"] = fx["draws"][:, :1]
    fx["sample_draws"] = np.array(1)
    try:
        analyze(fx, "bad")
        raise AssertionError("draws=1 did not abort")
    except SystemExit as e:
        print(f"  draws=1 fired: {e}")
    fx = _fixture()
    fx["valid"][0] = False  # frame 0: no valid triplet
    out = analyze(fx, "fallback")
    assert out["no_triplet_fallback_rows"] == 1, out
    print("  no-triplet fallback recorded")

    print("ORACLES GREEN")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--npz", default=DEFAULT_NPZ)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "reports/analysis__jerkpick_selector.json",
    )
    args = parser.parse_args()
    if args.oracle:
        run_oracles()
        return
    label = args.label or Path(args.npz).stem
    out = analyze(load_npz(args.npz), label)
    existing: dict[str, Any] = {}
    if args.out.exists():
        existing = json.loads(args.out.read_text())
    existing[label] = out
    args.out.write_text(json.dumps(existing, indent=1) + "\n")
    print(f"[{label}] ladder: {out['ladder_chunk']}")
    print(f"  first: {out['first_mae']}")
    print(f"  diagnostics: {out['diagnostics']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
