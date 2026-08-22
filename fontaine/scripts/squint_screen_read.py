"""Squint twin screen — pre-registered reads over the eval client's
banked JSONs (queue item squint-gate2-harness; pre-reg
posts/2026-08-22-prereg-squint-twin-screen.md, frozen expectation
grid + finalization amendment).

Subcommands:
  band       pilot JSON -> 20-80% band verdict (the banked bench rule;
             substitution ladder Reach/Stack logged, never auto-run)
  merge      pilot rows + remainder rows -> one n=100 cell (paired
             seeds preserved; pilot seeds 0-19 are a reusable prefix)
  gate1      adapted stronger arm's task JSONs -> best-task >=20/100
  read       adapted pair (+ optional unadapted riders) -> McNemar
             primary + KM/KS co-primary + frozen-grid verdict + the
             CDF panel chart (no scalar summaries without the panel)
  self-test  oracle: hand-computed McNemar/KS/censoring/band cases

Analysis constants are the standing house machinery: McNemar exact
two-sided p + seed-0 10k bootstrap imported from sim100_paired_read /
sim100_reads. The co-primary implements the eval-design tier-3 sketch
(posts/2026-08-21-vla-eval-design-v0.md) on the twin's honest per-step
predicates: per-predicate time-to-event CDFs (every episode runs the
full 50-step horizon, so non-events are right-censored at horizon and
the KM estimate reduces to the ECDF of first_true_step), KS distance +
signed AUC delta, seed-clustered bootstrap CI, and a paired label-swap
permutation p. Pre-reg direction: onerig above democlean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fontaine.scripts.sim100_paired_read import mcnemar_exact_p, mcnemar_table
from fontaine.scripts.sim100_reads import bootstrap_ci

PREREG = "posts/2026-08-22-prereg-squint-twin-screen.md"
HORIZON = 50
# Predicate keys are DERIVED from the banked rows (banked order
# preserved), not pinned here: the ladders are per-task — the env's own
# evaluate() keys (place has no reached_object; its approach milestone
# is is_item_above_bin — the 13:40Z 08-22 leg C KeyError). Both arms of
# a read must bank the identical set.


def banked_pred_keys(rows_a: list[dict], rows_b: list[dict]) -> tuple[str, ...]:
    keys = tuple(rows_a[0]["predicates"].keys())
    for rows in (rows_a, rows_b):
        for r in rows:
            if tuple(r["predicates"].keys()) != keys:
                raise SystemExit(
                    f"predicate keys differ across rows: {keys} vs "
                    f"{tuple(r['predicates'].keys())} — arms/tasks mixed?",
                )
    return keys


BAND_LOW, BAND_HIGH = 0.20, 0.80  # inclusive (the banked bench rule)
GATE1_MIN = 20  # successes /100, best task, adapted stronger arm
PERM_DRAWS = 10_000
BOOT_DRAWS = 10_000
RNG_SEED = 0  # the house bootstrap seed


# ---------------------------------------------------------------- loading


def load_cell(path: Path) -> dict[str, Any]:
    """Load one eval-client JSON; rows sorted by seed, seeds unique."""
    payload = json.loads(Path(path).read_text())
    rows = sorted(payload["rows"], key=lambda r: r["seed"])
    seeds = [r["seed"] for r in rows]
    assert len(set(seeds)) == len(seeds), f"{path}: duplicate seeds"
    payload["rows"] = rows
    return payload


def event_times(rows: list[dict], pred: str) -> np.ndarray:
    """First-true step per episode; HORIZON where never true (the
    right-censoring time — episodes always run the full horizon)."""
    return np.array(
        [
            HORIZON
            if r["first_true_step"][pred] is None
            else int(r["first_true_step"][pred])
            for r in rows
        ],
        dtype=np.int64,
    )


# ---------------------------------------------------------------- co-primary


def ecdf(times: np.ndarray) -> np.ndarray:
    """F(t) = P(event <= t) for t = 0..HORIZON-1. With all censoring at
    the fixed horizon, the Kaplan-Meier estimate equals this ECDF."""
    t_grid = np.arange(HORIZON)
    return (times[None, :] <= t_grid[:, None]).mean(axis=1)


def ks_and_auc(ta: np.ndarray, tb: np.ndarray) -> tuple[float, float]:
    diff = ecdf(ta) - ecdf(tb)
    return float(np.abs(diff).max()), float(diff.mean())


def km_ks_read(rows_a: list[dict], rows_b: list[dict]) -> dict[str, Any]:
    """Per-predicate KS + signed AUC delta (A - B), macro-KS across
    predicates, paired label-swap permutation p on macro-KS, and a
    seed-clustered bootstrap CI95 on the macro AUC delta."""
    n = len(rows_a)
    pred_keys = banked_pred_keys(rows_a, rows_b)
    ta = {p: event_times(rows_a, p) for p in pred_keys}
    tb = {p: event_times(rows_b, p) for p in pred_keys}

    per_pred: dict[str, Any] = {}
    for p in pred_keys:
        ks, auc = ks_and_auc(ta[p], tb[p])
        per_pred[p] = {
            "events_a": int((ta[p] < HORIZON).sum()),
            "events_b": int((tb[p] < HORIZON).sum()),
            "ks": round(ks, 4),
            "auc_delta": round(auc, 4),
            "cdf_a": ecdf(ta[p]).round(4).tolist(),
            "cdf_b": ecdf(tb[p]).round(4).tolist(),
        }
    macro_ks = float(np.mean([per_pred[p]["ks"] for p in pred_keys]))
    macro_auc = float(np.mean([per_pred[p]["auc_delta"] for p in pred_keys]))

    rng = np.random.default_rng(RNG_SEED)
    swaps = rng.integers(0, 2, size=(PERM_DRAWS, n)).astype(bool)
    perm_ge = 0
    for s in swaps:
        stat = 0.0
        for p in pred_keys:
            pa = np.where(s, tb[p], ta[p])
            pb = np.where(s, ta[p], tb[p])
            stat += ks_and_auc(pa, pb)[0]
        if stat / len(pred_keys) >= macro_ks - 1e-12:
            perm_ge += 1
    perm_p = (1 + perm_ge) / (1 + PERM_DRAWS)

    idx = rng.integers(0, n, size=(BOOT_DRAWS, n))
    boot = np.empty(BOOT_DRAWS)
    for i, sel in enumerate(idx):
        boot[i] = np.mean(
            [ks_and_auc(ta[p][sel], tb[p][sel])[1] for p in pred_keys],
        )
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "n_seeds": n,
        "horizon": HORIZON,
        "per_predicate": per_pred,
        "macro_ks": round(macro_ks, 4),
        "macro_auc_delta": round(macro_auc, 4),
        "macro_auc_delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "perm_p_macro_ks": round(perm_p, 5),
        "ks_agrees_prereg_direction": bool(macro_auc > 0 and perm_p < 0.05),
    }


# ---------------------------------------------------------------- primary


def paired_success_read(rows_a: list[dict], rows_b: list[dict]) -> dict[str, Any]:
    sa = np.array([r["success"] for r in rows_a], dtype=bool)
    sb = np.array([r["success"] for r in rows_b], dtype=bool)
    n = len(sa)
    lo, hi = bootstrap_ci(sa.astype(float) - sb.astype(float))
    table = mcnemar_table(sa, sb)
    return {
        "n_seeds": n,
        "count_a": int(sa.sum()),
        "count_b": int(sb.sum()),
        "count_delta": int(sa.sum() - sb.sum()),
        "count_delta_ci95": [round(lo * n, 4), round(hi * n, 4)],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "discordant": {
            **table,
            "mcnemar_exact_p_two_sided": mcnemar_exact_p(
                table["a_only"],
                table["b_only"],
            ),
        },
    }


def frozen_grid_verdict(primary: dict[str, Any], co: dict[str, Any]) -> str:
    """The pre-reg expectation grid, verbatim (primary task, adapted
    pair; A = onerig, B = democlean; pre-reg direction A > B)."""
    delta = primary["count_delta"]
    ci_excl = primary["ci_excludes_zero"]
    if delta > 0 and (ci_excl or co["ks_agrees_prereg_direction"]):
        return "ordering_preserved"
    if delta < 0 and ci_excl:
        return "substrate_divergence"
    return "underpowered_or_insensitive_at_n100"


# ---------------------------------------------------------------- subcommands


def cmd_band(args: argparse.Namespace) -> int:
    cell = load_cell(args.json)
    n = len(cell["rows"])
    k = sum(r["success"] for r in cell["rows"])
    rate = k / n
    if rate < BAND_LOW:
        verdict = "BELOW_BAND"
        ladder = "substitution ladder: Reach (easier) — new demos + adaptation leg, next pre-registered session"
    elif rate > BAND_HIGH:
        verdict = "ABOVE_BAND"
        ladder = "substitution ladder: Stack (harder) — new demos + adaptation leg, next pre-registered session"
    else:
        verdict = "IN_BAND"
        ladder = "none"
    print(
        json.dumps(
            {
                "task": cell["task"],
                "arm": cell["arm"],
                "n": n,
                "successes": k,
                "rate": round(rate, 4),
                "band": [BAND_LOW, BAND_HIGH],
                "ladder": ladder,
            },
        ),
    )
    print(verdict)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    parts = [load_cell(p) for p in args.parts]
    head = parts[0]
    frozen = (
        "replans",
        "sample_steps",
        "method",
        "sim_backend",
        "horizon",
        "exec_steps",
        "subsample",
    )
    for p in parts[1:]:
        for key in ("arm", "task", "checkpoint", "step", "stats_repo_id"):
            assert p[key] == head[key], f"{key} differs across parts"
        for key in frozen:
            assert p["config"][key] == head["config"][key], (
                f"config.{key} differs across parts"
            )
    rows = sorted(
        (r for p in parts for r in p["rows"]),
        key=lambda r: r["seed"],
    )
    seeds = [r["seed"] for r in rows]
    assert len(set(seeds)) == len(seeds), "overlapping seeds across parts"
    merged = {
        **head,
        "config": {
            **head["config"],
            "num_seeds": len(rows),
            "seed0": min(seeds),
            "merged_from": [str(p) for p in args.parts],
        },
        "successes": sum(r["success"] for r in rows),
        "rows": rows,
    }
    args.out.write_text(json.dumps(merged, indent=2))
    print(f"merged {len(rows)} seeds ({merged['successes']} successes) -> {args.out}")
    return 0


def cmd_gate1(args: argparse.Namespace) -> int:
    best_task, best_k, cells = None, -1, {}
    for path in args.jsons:
        cell = load_cell(path)
        k, n = sum(r["success"] for r in cell["rows"]), len(cell["rows"])
        assert n == 100, f"{path}: Gate-1 runs at treatment n=100, got {n}"
        cells[cell["task"]] = k
        if k > best_k:
            best_task, best_k = cell["task"], k
    verdict = "PASS" if best_k >= GATE1_MIN else "FAIL_F_INSTRUMENT"
    print(
        json.dumps(
            {
                "cells": cells,
                "best_task": best_task,
                "best": best_k,
                "gate": GATE1_MIN,
                "verdict": verdict,
            },
        ),
    )
    print(verdict)
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    a, b = load_cell(args.a), load_cell(args.b)
    assert a["task"] == b["task"], "arms evaluate different tasks"
    seeds_a = [r["seed"] for r in a["rows"]]
    seeds_b = [r["seed"] for r in b["rows"]]
    assert seeds_a == seeds_b, "seed sets differ — not a paired read"

    primary = paired_success_read(a["rows"], b["rows"])
    co = km_ks_read(a["rows"], b["rows"])
    report: dict[str, Any] = {
        "prereg": PREREG,
        "task": a["task"],
        "arms": {
            "a": {k: a[k] for k in ("arm", "checkpoint", "step", "stats_repo_id")},
            "b": {k: b[k] for k in ("arm", "checkpoint", "step", "stats_repo_id")},
        },
        "primary_mcnemar": primary,
        "co_primary_km_ks": co,
        "verdict": frozen_grid_verdict(primary, co),
    }
    for name, path in (("rider_a", args.rider_a), ("rider_b", args.rider_b)):
        if path is not None:
            rider = load_cell(path)
            report[f"record_only_{name}"] = {
                "arm": rider["arm"],
                "successes": sum(r["success"] for r in rider["rows"]),
                "n": len(rider["rows"]),
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    if args.charts_dir is not None:
        chart = render_cdf_panel(a, b, co, args.charts_dir)
        report_note = f" + panel {chart}"
    else:
        report_note = " (NO PANEL — scalar summaries alone are not publishable)"
    print(json.dumps({k: report[k] for k in ("task", "verdict")}, indent=1))
    print(f"wrote {args.out}{report_note}")
    return 0


# ---------------------------------------------------------------- chart

# House eval-report scheme (sim100_charts.py; adjacent-pair OKLab
# deltaE validated 2026-08-11 on this surface). Identity is never
# color-alone: adapted arms solid + direct-labeled, riders dashed.
PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
COL_A, COL_B = "#648fff", "#ffb000"  # onerig blue, democlean amber


def render_cdf_panel(
    a: dict[str, Any],
    b: dict[str, Any],
    co: dict[str, Any],
    out_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = np.arange(HORIZON)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), facecolor=PAGE)
    for ax, pred in zip(axes.flat, list(co["per_predicate"]), strict=True):
        pp = co["per_predicate"][pred]
        ax.set_facecolor(PAGE)
        ax.step(t, pp["cdf_a"], where="post", color=COL_A, lw=2)
        ax.step(t, pp["cdf_b"], where="post", color=COL_B, lw=2)
        ax.text(
            HORIZON - 1,
            pp["cdf_a"][-1],
            f" {a['arm']}",
            color=TEXT,
            fontsize=8,
            va="bottom",
            ha="right",
        )
        ax.text(
            HORIZON - 1,
            pp["cdf_b"][-1],
            f" {b['arm']}",
            color=TEXT,
            fontsize=8,
            va="top",
            ha="right",
        )
        ax.set_title(
            f"{pred}   KS {pp['ks']:.2f}  ΔAUC {pp['auc_delta']:+.2f}",
            color=TEXT,
            fontsize=10,
            loc="left",
        )
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, HORIZON - 1)
        ax.tick_params(colors=META, labelsize=8)
        ax.grid(color=GRID, lw=0.5, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    axes[1][0].set_xlabel("env step (10 Hz twin)", color=META, fontsize=9)
    axes[0][0].set_ylabel("P(reached by step)", color=META, fontsize=9)
    fig.suptitle(
        f"{a['task']}: time-to-predicate CDFs, adapted pair "
        f"(n={co['n_seeds']} paired seeds; macro-KS {co['macro_ks']:.2f}, "
        f"perm p {co['perm_p_macro_ks']:.4f})",
        color=TEXT,
        fontsize=11,
    )
    handles = [
        Line2D([], [], color=COL_A, lw=2, label=a["arm"]),
        Line2D([], [], color=COL_B, lw=2, label=b["arm"]),
    ]
    fig.legend(
        handles=handles,
        loc="lower right",
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"km_panel_{a['task']}.png"
    fig.savefig(out, dpi=150, facecolor=PAGE)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- oracle


def _fake_cell(arm: str, task: str, rows: list[dict]) -> dict[str, Any]:
    return {
        "arm": arm,
        "task": task,
        "checkpoint": "fake",
        "step": 0,
        "stats_repo_id": "fake",
        "instruction": "fake",
        "config": {
            "num_seeds": len(rows),
            "seed0": 0,
            "replans": 5,
            "sample_steps": 10,
            "method": "euler",
            "sim_backend": "physx_cpu",
            "horizon": HORIZON,
            "exec_steps": 10,
            "subsample": 3,
        },
        "successes": sum(r["success"] for r in rows),
        "rows": rows,
    }


def _fake_row(seed: int, first: dict[str, int | None]) -> dict[str, Any]:
    return {
        "seed": seed,
        "success": first["success"] is not None,
        "first_true_step": dict(first),
        # keys mirror first_true_step (banked_pred_keys reads them);
        # per-step traces are not exercised by the oracle cases
        "predicates": {k: [] for k in first},
        "qpos_trace": [],
        "target_trace": [],
        "predict_ms_mean": 0.0,
    }


def cmd_self_test(_args: argparse.Namespace) -> int:
    # 1. McNemar exact p, hand-computed: a_only=8, b_only=2 ->
    #    2 * P(X >= 8 | n=10, p=.5) = 2 * 56/1024 = 0.109375.
    assert abs(mcnemar_exact_p(8, 2) - 0.109375) < 1e-12
    assert mcnemar_exact_p(0, 0) == 1.0

    # 2. ECDF + censoring: events at 0,1,2 + one never (n=4).
    times = np.array([0, 1, 2, HORIZON])
    f = ecdf(times)
    assert f[0] == 0.25 and f[1] == 0.5 and f[2] == 0.75 and f[-1] == 0.75

    # 3. KS/AUC extremes: A all at step 0, B never -> KS=1, AUC=+1.
    ks, auc = ks_and_auc(np.zeros(4, dtype=int), np.full(4, HORIZON))
    assert ks == 1.0 and auc == 1.0
    # identical arms -> 0, 0
    ks0, auc0 = ks_and_auc(times, times)
    assert ks0 == 0.0 and auc0 == 0.0

    # 4. Permutation p: maximally separated 8-seed pair -> only the
    #    identity/full-swap patterns reach KS 1 (p ~= 2/256); identical
    #    arms -> p ~= 1.
    lift_keys = ("reached_object", "is_item_grasped", "item_lifted", "success")
    fast = [_fake_row(s, dict.fromkeys(lift_keys, 0)) for s in range(8)]
    slow = [_fake_row(s, dict.fromkeys(lift_keys)) for s in range(8)]
    co = km_ks_read(fast, slow)
    assert co["macro_ks"] == 1.0 and co["macro_auc_delta"] == 1.0
    assert co["perm_p_macro_ks"] < 0.05
    assert co["ks_agrees_prereg_direction"]
    co_same = km_ks_read(fast, fast)
    assert co_same["perm_p_macro_ks"] > 0.5
    assert not co_same["ks_agrees_prereg_direction"]

    # 5. Band edges (inclusive 20-80).
    for rate, want in (
        (0.20, "IN_BAND"),
        (0.19, "BELOW_BAND"),
        (0.80, "IN_BAND"),
        (0.81, "ABOVE_BAND"),
    ):
        if rate < BAND_LOW:
            got = "BELOW_BAND"
        elif rate > BAND_HIGH:
            got = "ABOVE_BAND"
        else:
            got = "IN_BAND"
        assert got == want, (rate, got, want)

    # 6. Frozen grid: constructed ordering-preserved / divergence /
    #    straddle cases.
    strong = {"count_delta": 15, "ci_excludes_zero": True}
    weak = {"count_delta": 3, "ci_excludes_zero": False}
    neg = {"count_delta": -15, "ci_excludes_zero": True}
    ks_yes = {"ks_agrees_prereg_direction": True}
    ks_no = {"ks_agrees_prereg_direction": False}
    assert frozen_grid_verdict(strong, ks_no) == "ordering_preserved"
    assert frozen_grid_verdict(weak, ks_yes) == "ordering_preserved"
    assert frozen_grid_verdict(weak, ks_no) == "underpowered_or_insensitive_at_n100"
    assert frozen_grid_verdict(neg, ks_no) == "substrate_divergence"

    # 7. Paired success read on a hand case: A succeeds seeds 0-5,
    #    B seeds 4-7 (n=10): delta +2, a_only=4, b_only=2.
    ra = [
        _fake_row(
            s,
            {
                "success": 10 if s < 6 else None,
                **{p: 0 for p in lift_keys if p != "success"},
            },
        )
        for s in range(10)
    ]
    rb = [
        _fake_row(
            s,
            {
                "success": 10 if 4 <= s < 8 else None,
                **{p: 0 for p in lift_keys if p != "success"},
            },
        )
        for s in range(10)
    ]
    pr = paired_success_read(ra, rb)
    assert pr["count_delta"] == 2
    assert pr["discordant"]["a_only"] == 4 and pr["discordant"]["b_only"] == 2

    print(
        "self-test OK: McNemar, ECDF/censoring, KS/AUC, permutation, "
        "band edges, frozen grid, paired read",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("band")
    p.add_argument("--json", type=Path, required=True)
    p.set_defaults(fn=cmd_band)

    p = sub.add_parser("merge")
    p.add_argument("--parts", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("gate1")
    p.add_argument("--jsons", type=Path, nargs="+", required=True)
    p.set_defaults(fn=cmd_gate1)

    p = sub.add_parser("read")
    p.add_argument("--a", type=Path, required=True, help="adapted onerig")
    p.add_argument("--b", type=Path, required=True, help="adapted democlean")
    p.add_argument("--rider-a", type=Path, default=None)
    p.add_argument("--rider-b", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--charts-dir", type=Path, default=None)
    p.set_defaults(fn=cmd_read)

    p = sub.add_parser("self-test")
    p.set_defaults(fn=cmd_self_test)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
