"""House dark-mode charts for the grasp-SFT MAE-drift saga page
(posts/2026-08-17-sft-drift-saga.md).

Reads ONLY banked artifacts (regenerable, no live hosts):
  reports/curve__sft_drift_saga.json  — written by --extract from
      outputs/train/rigonly_artifacts/{rigonly,mixedv2,demosonly}
      train_log.jsonl copies (rsynced off the box 08-17 before any
      cleanup) + reports/curve__grasp_sft_v1_wandb.json (run-2 pooled)

Figures -> fontaine/blog/src/img/grasp_sft_drift/:
  1. drift_grid.png     — 2x2 small multiples, eval+train chunk MAE
     per run, drift-onset band from step 500, kill markers
  2. drift_indexed.png  — eval MAE delta from each run's step-500
     value vs steps-since-500 (the magnitude-honesty overlay)
  3. twin_rulers.png    — demosonly exemplar: training loss (top)
     vs chunk MAE (bottom), two rulers, aligned x — NOT dual-axis
  4. head_asymmetry.png — sim100 successes, flow vs token head at
     step500/endpoint + probe anchor

Discriminator post-processing (sft-drift-discriminator-run kit): once
the owner-gated 1-GPU run exists, point --discriminator at its
babysit-rsynced train_log.jsonl to get
  5. disc_overlay.png — the 1-GPU curve on the indexed-drift
     instrument (delta eval/train MAE vs own step-500) against the
     banked 8x drift band + run-2's healthy pooled curve
  6. reports/analysis__sft_drift_discriminator.json + a printed
     verdict block (the launcher header's read rule, operationalized
     BEFORE the run: falls/holds through 1000 -> distributed path
     CONVICTED; same-drift -> distributed EXONERATED, remaining
     suspects augment/eff-96/recompute-stats/init/corpus-scale)

Usage:
  uv run python fontaine/scripts/sft_drift_saga_charts.py --extract
  uv run python fontaine/scripts/sft_drift_saga_charts.py
  uv run python fontaine/scripts/sft_drift_saga_charts.py \
      --discriminator outputs/train/disc_artifacts/train_log.jsonl
  # dry run against the rigonly log (writes *_fixture outputs only):
  uv run python fontaine/scripts/sft_drift_saga_charts.py \
      --discriminator outputs/train/rigonly_artifacts/train_log.jsonl \
      --fixture
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# House eval-report scheme (grasp_sft_chain_charts.py lineage): dark
# page, identity never color-alone (direct labels + line styles).
PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
MAGENTA = "#dc267f"  # eval slice / the drifting emphasis
BLUE = "#648fff"  # train slice / secondary series
GOLD = "#ffb000"  # rigonly (the run under verdict on this page)
GRAY = "#9aa0a8"
DISC = "#e8eaed"  # the 1-GPU discriminator run: near-white, lightness-
# separated from every context hue under all CVD types; identity also
# carried by weight (bold vs faint context) + markers + direct labels

REPO = Path("/home/ubuntu/flow-matching")
REPORTS = REPO / "reports"
BANK = REPORTS / "curve__sft_drift_saga.json"
ART = REPO / "outputs/train/rigonly_artifacts"
IMG_OUT = REPO / "fontaine/blog/src/img/grasp_sft_drift"


def _style(ax: plt.Axes, title: str | None = None) -> None:
    ax.set_facecolor(PAGE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.tick_params(colors=META, labelsize=8)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9.5, loc="left", pad=6)


def _fig(w: float, h: float) -> plt.Figure:
    return plt.figure(figsize=(w, h), facecolor=PAGE)


def curve(path: Path) -> dict:
    steps, ev, tr, loss_s, loss_v = [], [], [], [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        if "eval_chunk_mae" in d:
            steps.append(d["step"])
            ev.append(d["eval_chunk_mae"])
            tr.append(d["train_mae"])
        elif "loss" in d and "step" in d:
            loss_s.append(d["step"])
            loss_v.append(d["loss"])
    return {
        "steps": steps,
        "eval": ev,
        "train": tr,
        "loss_steps": loss_s,
        "loss": loss_v,
    }


def extract() -> dict:
    wandb = json.loads((REPORTS / "curve__grasp_sft_v1_wandb.json").read_text())
    r2s, r2e, r2t = [], [], []
    for row in wandb["rows"]:
        if row.get("eval/chunk_mae") is not None:
            r2s.append(row["_step"])
            r2e.append(row["eval/chunk_mae"])
            r2t.append(row["train/mae"])

    bank = {
        "note": "grasp-SFT drift saga curves, banked 2026-08-17 from "
        "box train_log.jsonl copies rsynced before cleanup",
        "runs": {
            "run2_pooled": {
                "label": "run-2 (mix, pooled table)",
                "killed_at": None,
                "endpoint": 3000,
                "steps": r2s,
                "eval": r2e,
                "train": r2t,
            },
            "mixedv2": {
                "label": "mixed v2 (mix, fresh merged table)",
                "killed_at": 1150,
                **curve(ART / "mixedv2_train_log.jsonl"),
            },
            "demosonly": {
                "label": "demos-only (demos-native table)",
                "killed_at": 1350,
                **curve(ART / "demosonly_train_log.jsonl"),
            },
            "rigonly": {
                "label": "rig-only (rig-native table)",
                "killed_at": None,
                "endpoint": 1000,
                **curve(ART / "train_log.jsonl"),
            },
        },
        "head_asymmetry": {
            "note": "run-2 sim100 successes/100, unseen seeds 0-99",
            "flow": {"step500": 4, "step3000": 5},
            "token_fixed": {"step500": 16, "step3000": 14},
            "probe_flow_44": 44,
        },
        "run1b": {
            "label": "run-1b (mix, rig-lineage remap table)",
            "v2_slice_eval": [16.0, 18.4],
            "note": "kill signature summary only; per-step curve not banked",
        },
    }
    BANK.write_text(json.dumps(bank, indent=1))
    print(f"banked -> {BANK}")
    return bank


def chart_grid(bank: dict) -> None:
    runs = bank["runs"]
    order = ["run2_pooled", "mixedv2", "demosonly", "rigonly"]
    subtitles = {
        "run2_pooled": "3000 steps, complete — endpoint flow 5/100",
        "mixedv2": "owner-killed @~1150 — both slices rising",
        "demosonly": "owner-killed @~1350 — drift REPRODUCED",
        "rigonly": "complete @1000 — the verdict run (rig data only)",
    }
    fig = _fig(9.6, 6.4)
    axes = fig.subplots(2, 2, sharex=False)
    for ax, key in zip(axes.flat, order, strict=True):
        r = runs[key]
        hue = GOLD if key == "rigonly" else MAGENTA
        _style(ax, f"{r['label']}\n{subtitles[key]}")
        ax.axvspan(500, max(r["steps"]) + 60, color=MAGENTA, alpha=0.045)
        ax.axvline(500, color=GRID, linewidth=0.8, linestyle=":")
        ax.plot(
            r["steps"],
            r["eval"],
            color=hue,
            linewidth=2,
            marker="o",
            markersize=4.5,
        )
        ax.plot(
            r["steps"],
            r["train"],
            color=BLUE,
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=3.5,
        )
        ev_up = r["eval"][-1] >= r["train"][-1]
        ax.annotate(
            "eval",
            (r["steps"][-1], r["eval"][-1]),
            textcoords="offset points",
            xytext=(6, 3 if ev_up else -10),
            color=hue,
            fontsize=8,
        )
        ax.annotate(
            "train",
            (r["steps"][-1], r["train"][-1]),
            textcoords="offset points",
            xytext=(6, -10 if ev_up else 3),
            color=BLUE,
            fontsize=8,
        )
        if r.get("killed_at"):
            ax.axvline(
                r["killed_at"],
                color=TEXT,
                linewidth=0.9,
                linestyle="-.",
                alpha=0.7,
            )
            ax.annotate(
                "killed",
                (r["killed_at"], ax.get_ylim()[0]),
                textcoords="offset points",
                xytext=(-4, 6),
                color=META,
                fontsize=7.5,
                ha="right",
            )
        ax.margins(x=0.18, y=0.12)
        ax.set_xlabel("step", color=META, fontsize=8)
    axes[0][0].set_ylabel("chunk MAE (deg)", color=META, fontsize=8)
    axes[1][0].set_ylabel("chunk MAE (deg)", color=META, fontsize=8)
    fig.suptitle(
        "Four runs, one shape: held-out chunk MAE turns up after step 500\n"
        "(absolute levels NOT comparable across panels — each run has its own normalization table)",
        color=TEXT,
        fontsize=10.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(IMG_OUT / "drift_grid.png", dpi=160, facecolor=PAGE)
    plt.close(fig)


def chart_indexed(bank: dict) -> None:
    runs = bank["runs"]
    series = [
        ("demosonly", MAGENTA, "-"),
        ("mixedv2", BLUE, "-"),
        ("run2_pooled", GRAY, "--"),
        ("rigonly", GOLD, "-"),
    ]
    fig = _fig(8.4, 4.6)
    ax = fig.subplots()
    _style(ax, None)
    ax.axhline(0, color=GRID, linewidth=1.2)
    for key, hue, ls in series:
        r = runs[key]
        steps = np.array(r["steps"], dtype=float)
        ev = np.array(r["eval"], dtype=float)
        m = steps >= 500
        base = ev[steps == 500][0]
        x = steps[m] - 500
        y = ev[m] - base
        keep = x <= 1000
        ax.plot(
            x[keep],
            y[keep],
            color=hue,
            linewidth=2.2,
            linestyle=ls,
            marker="o",
            markersize=4.5,
        )
        dy = {"demosonly": 4, "mixedv2": 4, "run2_pooled": -10, "rigonly": 8}[key]
        ax.annotate(
            f"{r['label']}  {y[keep][-1]:+.2f}",
            (x[keep][-1], y[keep][-1]),
            textcoords="offset points",
            xytext=(8, dy),
            color=hue,
            fontsize=8.5,
        )
    ax.set_xlabel("steps since step 500", color=META, fontsize=9)
    ax.set_ylabel("Δ eval chunk MAE vs own step-500 (deg)", color=META, fontsize=9)
    ax.set_xlim(-30, 1420)
    ax.set_title(
        "Same shape, very different sizes: eval-MAE rise indexed to each run's step-500 value\n"
        "(rig-only's +0.69 is the ambiguity — the shape matches, the magnitude doesn't)",
        color=TEXT,
        fontsize=10,
        loc="left",
        pad=8,
    )
    fig.tight_layout()
    fig.savefig(IMG_OUT / "drift_indexed.png", dpi=160, facecolor=PAGE)
    plt.close(fig)


def chart_twin_rulers(bank: dict) -> None:
    r = bank["runs"]["demosonly"]
    fig = _fig(8.4, 5.4)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    _style(
        ax1,
        "the ruler the optimizer sees: combined training loss (falling the whole run)",
    )
    ls = np.array(r["loss_steps"], dtype=float)
    lv = np.array(r["loss"], dtype=float)
    ax1.plot(ls, lv, color=BLUE, linewidth=1.4)
    ax1.set_ylabel("loss (normalized space)", color=META, fontsize=8.5)
    _style(
        ax2,
        "the ruler we care about: chunk MAE in raw degrees (rising from step 500)",
    )
    ax2.axvspan(500, 1360, color=MAGENTA, alpha=0.045)
    ax2.plot(
        r["steps"],
        r["eval"],
        color=MAGENTA,
        linewidth=2.2,
        marker="o",
        markersize=4.5,
    )
    ax2.plot(
        r["steps"],
        r["train"],
        color=BLUE,
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=3.5,
    )
    ax2.annotate(
        "eval",
        (r["steps"][-1], r["eval"][-1]),
        textcoords="offset points",
        xytext=(6, 2),
        color=MAGENTA,
        fontsize=8,
    )
    ax2.annotate(
        "train",
        (r["steps"][-1], r["train"][-1]),
        textcoords="offset points",
        xytext=(6, -8),
        color=BLUE,
        fontsize=8,
    )
    ax2.set_ylabel("chunk MAE (deg)", color=META, fontsize=8.5)
    ax2.set_xlabel("step (demos-only run, killed @~1350)", color=META, fontsize=9)
    ax2.margins(x=0.06)
    fig.suptitle(
        "Two rulers: the loss can fall while the policy gets worse in degrees",
        color=TEXT,
        fontsize=10.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(IMG_OUT / "twin_rulers.png", dpi=160, facecolor=PAGE)
    plt.close(fig)


def chart_heads(bank: dict) -> None:
    h = bank["head_asymmetry"]
    fig = _fig(7.2, 4.2)
    ax = fig.subplots()
    _style(ax, None)
    ax.grid(axis="x", visible=False)
    x = np.array([0, 1])
    w = 0.32
    flow = [h["flow"]["step500"], h["flow"]["step3000"]]
    tok = [h["token_fixed"]["step500"], h["token_fixed"]["step3000"]]
    b1 = ax.bar(x - w / 2 - 0.01, flow, width=w, color=MAGENTA, zorder=3)
    b2 = ax.bar(x + w / 2 + 0.01, tok, width=w, color=BLUE, zorder=3)
    for bars in (b1, b2):
        for rect in bars:
            rect.set_linewidth(0)
            ax.annotate(
                f"{int(rect.get_height())}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                color=TEXT,
                fontsize=10,
            )
    ax.axhline(h["probe_flow_44"], color=GOLD, linewidth=1.6, linestyle="--")
    ax.annotate(
        "probe checkpoint, flow head: 44/100 (313 demos, table fits)",
        (1.42, h["probe_flow_44"]),
        color=GOLD,
        fontsize=8.5,
        ha="right",
        va="bottom",
    )
    ax.set_xticks(x, ["step 500", "step 3000 (endpoint)"])
    ax.tick_params(axis="x", colors=TEXT, labelsize=9.5)
    ax.set_ylabel("sim100 successes (seeds 0–99)", color=META, fontsize=9)
    ax.set_ylim(0, 50)
    ax.legend(
        [b1, b2],
        ["flow head (euler-10)", "token head (greedy, decode fix)"],
        loc="upper left",
        frameon=False,
        labelcolor=TEXT,
        fontsize=8.5,
    )
    ax.set_title(
        "run-2: the damage is flow-head-specific — token ~flat 16→14, flow collapsed 4→5",
        color=TEXT,
        fontsize=10,
        loc="left",
        pad=8,
    )
    fig.tight_layout()
    fig.savefig(IMG_OUT / "head_asymmetry.png", dpi=160, facecolor=PAGE)
    plt.close(fig)


# --- discriminator kit (sft-drift-discriminator-run post-processing) ---
# Verdict operationalization, FROZEN before the run exists (this is the
# numeric form of the launcher header's read rule; the formal pre-reg
# post cut on the owner GO references these bounds verbatim):
#   instrument  delta(s) = eval_chunk_mae(s) - eval_chunk_mae(500),
#               primary read at s = 1000 (the endpoint probe)
#   HEALTHY     delta(1000) <= +0.30  -> distributed path CONVICTED
#               (+0.30 sits above healthy-curve probe wiggle — run-2's
#               pooled curve moved -0.92 over the same window — and far
#               below every drifting run)
#   SAME-DRIFT  delta(1000) >= 0.5 x demosonly's delta over the same
#               offset window (+2.03 => bound ~+1.01) -> distributed
#               EXONERATED; remaining suspects: image-augment, eff-96,
#               recompute-stats-at-launch, init checkpoint, corpus scale
#   else        AMBIGUOUS (the rigonly class: its +0.69 lands here,
#               matching its posted ambiguous-leaning-drift verdict —
#               that agreement is the fixture check)
# Train-slice delta and monotone-rise flags are reported as
# corroboration, not gates.
HEALTHY_BOUND = 0.30
DRIFT_FRACTION = 0.5


def _delta_at(run: dict, offset: int) -> float | None:
    steps = list(run["steps"])
    if 500 not in steps or 500 + offset not in steps:
        return None
    ev = run["eval"]
    return ev[steps.index(500 + offset)] - ev[steps.index(500)]


def disc_verdict(bank: dict, disc: dict, label: str, *, fixture: bool) -> dict:
    steps = disc["steps"]
    if 500 not in steps:
        raise SystemExit(f"no step-500 eval probe in the log ({steps=})")
    probes = [s for s in steps if s >= 500]
    last = probes[-1]
    provisional = last < 1000
    read_at = last - 500
    d_eval = _delta_at(disc, read_at)
    i500 = steps.index(500)
    d_train = disc["train"][steps.index(last)] - disc["train"][i500]
    demos_ref = _delta_at(bank["runs"]["demosonly"], read_at)
    drift_bound = DRIFT_FRACTION * demos_ref
    if d_eval <= HEALTHY_BOUND:
        verdict = "HEALTHY"
        text = (
            "falls/holds through the read point -> the DISTRIBUTED PATH IS "
            "CONVICTED (torchrun + zero1 + chunk-grad-allreduce is the delta "
            "that separates every drifting run from every healthy one)"
        )
    elif d_eval >= drift_bound:
        verdict = "SAME-DRIFT"
        text = (
            "drifts like the 8x runs -> distributed machinery EXONERATED; "
            "remaining suspects: image-augment, eff-96 batch geometry, "
            "recompute-stats-at-launch, init checkpoint, corpus scale"
        )
    else:
        verdict = "AMBIGUOUS"
        text = (
            "between the bounds (the rigonly class) -> no conviction either "
            "way; escalation is an owner call (extend past 1000 or cut the "
            "next single-delta run)"
        )
    ev_tail = [disc["eval"][steps.index(s)] for s in probes]
    tr_tail = [disc["train"][steps.index(s)] for s in probes]
    refs = {
        k: _delta_at(bank["runs"][k], read_at)
        for k in ("demosonly", "mixedv2", "run2_pooled", "rigonly")
    }
    return {
        "run": label,
        "fixture": fixture,
        "read_step": last,
        "provisional": provisional,
        "delta_eval_vs_500": round(d_eval, 4),
        "delta_train_vs_500": round(d_train, 4),
        "eval_monotone_rise_from_500": all(
            b > a for a, b in itertools.pairwise(ev_tail)
        ),
        "train_monotone_rise_from_500": all(
            b > a for a, b in itertools.pairwise(tr_tail)
        ),
        "bounds": {
            "healthy_max": HEALTHY_BOUND,
            "drift_min": round(drift_bound, 4),
            "demosonly_ref_same_window": round(demos_ref, 4),
        },
        "references_same_window": {
            k: round(d, 4) if d is not None else None for k, d in refs.items()
        },
        "verdict": verdict,
        "verdict_text": text,
    }


def chart_disc_overlay(
    bank: dict,
    disc: dict,
    label: str,
    out_png: Path,
    *,
    fixture: bool,
) -> None:
    runs = bank["runs"]
    context = [
        ("demosonly", MAGENTA, "-"),
        ("mixedv2", BLUE, "-"),
        ("run2_pooled", GRAY, "--"),
        ("rigonly", GOLD, "-"),
    ]
    fig = _fig(9.6, 4.6)
    ax_e, ax_t = fig.subplots(1, 2, sharex=True)
    for ax, slice_key, ylab in (
        (ax_e, "eval", "Δ eval chunk MAE vs own step-500 (deg)"),
        (ax_t, "train", "Δ train chunk MAE vs own step-500 (deg)"),
    ):
        _style(ax, None)
        ax.axhline(0, color=GRID, linewidth=1.2)
        band = []
        for key, hue, ls in context:
            r = runs[key]
            steps = np.array(r["steps"], dtype=float)
            v = np.array(r[slice_key], dtype=float)
            m = (steps >= 500) & (steps <= 1250)
            base = v[steps == 500][0]
            x, y = steps[m] - 500, v[m] - base
            if key in ("demosonly", "mixedv2"):
                band.append((x, y))
            ax.plot(x, y, color=hue, linewidth=1.4, linestyle=ls, alpha=0.5)
            ax.annotate(
                r["label"].split(" (")[0],
                (x[-1], y[-1]),
                textcoords="offset points",
                xytext=(5, -2),
                color=hue,
                fontsize=7.5,
                alpha=0.9,
            )
        common = sorted(set(band[0][0]) & set(band[1][0]))
        lo = [min(b[1][list(b[0]).index(s)] for b in band) for s in common]
        hi = [max(b[1][list(b[0]).index(s)] for b in band) for s in common]
        ax.fill_between(common, lo, hi, color=MAGENTA, alpha=0.08, linewidth=0)
        steps = np.array(disc["steps"], dtype=float)
        v = np.array(disc[slice_key], dtype=float)
        m = steps >= 500
        x, y = steps[m] - 500, v[m] - v[steps == 500][0]
        ax.plot(
            x,
            y,
            color=DISC,
            linewidth=2.6,
            marker="D",
            markersize=5.5,
            zorder=5,
        )
        ax.annotate(
            f"{label}  {y[-1]:+.2f}",
            (x[-1], y[-1]),
            textcoords="offset points",
            xytext=(-2, 9),
            ha="right",
            color=DISC,
            fontsize=9,
            fontweight="bold",
        )
        ax.set_xlabel("steps since step 500", color=META, fontsize=9)
        ax.set_ylabel(ylab, color=META, fontsize=9)
        ax.margins(x=0.24)
    demos_ref = _delta_at(runs["demosonly"], 500)
    for bound, name in (
        (HEALTHY_BOUND, "healthy bound"),
        (DRIFT_FRACTION * demos_ref, "same-drift bound"),
    ):
        ax_e.axhline(bound, color=TEXT, linewidth=0.8, linestyle=":", alpha=0.55)
        ax_e.annotate(
            f"{name} {bound:+.2f}",
            (0, bound),
            textcoords="offset points",
            xytext=(2, 3),
            color=META,
            fontsize=7.5,
        )
    stamp = "  —  FIXTURE DRY-RUN (rigonly log as stand-in)" if fixture else ""
    fig.suptitle(
        f"1-GPU discriminator vs the banked curves — indexed to each run's "
        f"own step-500{stamp}\n"
        "(shaded band: the two unambiguous drifting 8× runs; bounds frozen "
        "before the run — see analysis JSON)",
        color=TEXT,
        fontsize=10,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png, dpi=160, facecolor=PAGE)
    plt.close(fig)


def run_discriminator(bank: dict, log_path: Path, *, fixture: bool) -> None:
    disc = curve(log_path)
    label = "1-GPU disc (fixture: rigonly)" if fixture else "1-GPU disc"
    suffix = "_fixture" if fixture else ""
    out_dir = REPORTS / "disc_fixture" if fixture else IMG_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"disc_overlay{suffix}.png"
    chart_disc_overlay(bank, disc, label, out_png, fixture=fixture)
    verdict = disc_verdict(bank, disc, label, fixture=fixture)
    verdict["log"] = str(log_path)
    out_json = REPORTS / f"analysis__sft_drift_discriminator{suffix}.json"
    out_json.write_text(json.dumps(verdict, indent=1))
    print(f"overlay -> {out_png}\nverdict -> {out_json}")
    print(
        f"\nVERDICT [{verdict['verdict']}"
        f"{' PROVISIONAL' if verdict['provisional'] else ''}] "
        f"read@{verdict['read_step']}: "
        f"Δeval {verdict['delta_eval_vs_500']:+.2f} "
        f"(healthy ≤ +{HEALTHY_BOUND:.2f}, drift ≥ "
        f"+{verdict['bounds']['drift_min']:.2f}), "
        f"Δtrain {verdict['delta_train_vs_500']:+.2f}\n"
        f"{verdict['verdict_text']}",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extract", action="store_true")
    p.add_argument(
        "--discriminator",
        type=Path,
        default=None,
        metavar="TRAIN_LOG_JSONL",
        help="1-GPU discriminator run's rsynced train_log.jsonl",
    )
    p.add_argument(
        "--fixture",
        action="store_true",
        help="dry-run: label as fixture, write *_fixture outputs only",
    )
    args = p.parse_args()
    bank = extract() if args.extract else json.loads(BANK.read_text())
    if args.discriminator is not None:
        run_discriminator(bank, args.discriminator, fixture=args.fixture)
        return
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    chart_grid(bank)
    chart_indexed(bank)
    chart_twin_rulers(bank)
    chart_heads(bank)
    print(f"4 figures -> {IMG_OUT}")


if __name__ == "__main__":
    main()
