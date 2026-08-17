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

Usage:
  uv run python fontaine/scripts/sft_drift_saga_charts.py --extract
  uv run python fontaine/scripts/sft_drift_saga_charts.py
"""

from __future__ import annotations

import argparse
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


def extract() -> dict:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extract", action="store_true")
    args = p.parse_args()
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    bank = extract() if args.extract else json.loads(BANK.read_text())
    chart_grid(bank)
    chart_indexed(bank)
    chart_twin_rulers(bank)
    chart_heads(bank)
    print(f"4 figures -> {IMG_OUT}")


if __name__ == "__main__":
    main()
