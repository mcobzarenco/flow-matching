"""Charts for the consolidated golden-ticket visual report (owner
steering 2026-08-08 08:42Z): five SVGs rendered from the BANKED stage
analyses — no re-computation of any claimed number, every value read
from the frozen jsons (`analysis__goldenticket_stage{1,2,3}.json`).

Output: fontaine/blog/src/img/goldenticket/*.svg (committed).

Palette: the dataviz reference palette's pre-validated categorical
slots (blue #2a78d6, orange #eb6834) + neutral text tokens — no custom
hues (no node on the box, so the palette validator can't run; using
the reference instance unchanged is the sanctioned fallback).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/goldenticket"

BLUE = "#2a78d6"  # top-10 ticket set / ensemble
ORANGE = "#eb6834"  # winner ticket 33
GRAY = "#9b9a95"  # other tickets / reference marks
DARKGRAY = "#52514e"  # baseline series, secondary text
INK = "#0b0b0b"
SURFACE = "#fcfcfb"
GRID = "#e5e4e0"


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=DARKGRAY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(width: float = 8.2, height: float = 4.2) -> tuple:
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def chart_r1(stage1: dict) -> None:
    scores = np.array(stage1["per_ticket_chunk_mae"], dtype=float)
    top10 = set(stage1["top10_tickets"])
    winner = int(stage1["winner"]["ticket"])
    r1 = stage1["r1"]
    order = np.argsort(scores)

    fig, ax = new_fig(8.6, 4.4)
    mean, sigma = r1["mean"], r1["sigma_probe_null"]
    ax.axhspan(
        mean - 2 * sigma,
        mean + 2 * sigma,
        color=GRAY,
        alpha=0.25,
        linewidth=0,
    )
    ax.axhline(mean, color=DARKGRAY, linewidth=1, linestyle=":")
    ax.axhline(r1["expected_null_min"], color=DARKGRAY, linewidth=1, linestyle="--")
    for rank, idx in enumerate(order):
        t = int(idx)
        color = ORANGE if t == winner else (BLUE if t in top10 else GRAY)
        size = 46 if t == winner else (30 if t in top10 else 16)
        ax.scatter(rank, scores[t], s=size, color=color, zorder=3)
    ax.annotate(
        f"ticket 33 — {scores[winner]:.3f}",
        xy=(0, scores[winner]),
        xytext=(3.5, scores[winner] - 0.06),
        fontsize=10,
        color=INK,
        va="top",
    )
    ax.annotate(
        f"null band: mean ± 2σ (σ = {sigma})",
        xy=(63, mean + 2 * sigma),
        fontsize=9,
        color=DARKGRAY,
        ha="right",
        va="bottom",
    )
    ax.annotate(
        f"expected null minimum {r1['expected_null_min']:.3f}",
        xy=(63, r1["expected_null_min"]),
        fontsize=9,
        color=DARKGRAY,
        ha="right",
        va="top",
    )
    ax.set_xlabel("64 tickets, sorted by probe score", color=DARKGRAY, fontsize=10)
    ax.set_ylabel("pooled probe chunk MAE", color=DARKGRAY, fontsize=10)
    ax.set_title(
        "R1 — the ticket distribution is ~12x wider than the i.i.d.-noise null "
        f"(sd {r1['sd']:.3f} vs line {r1['sd_line']})",
        color=INK,
        fontsize=11,
        loc="left",
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="winner (33)"),
        plt.Line2D([], [], marker="o", ls="", color=BLUE, label="top-10 set"),
        plt.Line2D([], [], marker="o", ls="", color=GRAY, label="other tickets"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    save(fig, "r1_tickets.svg")


def chart_deltas(stage2: dict, stage3: dict) -> None:
    r2, r3 = stage2["r2"], stage3["r3"]
    rows = [
        (
            "R2 — winner vs stable-key\n(held-out complement rows, paired)",
            r2["delta_pooled"],
            r2["ci95"],
            r2["line"],
            "adopt line −0.05",
        ),
        (
            "R3 — mean-of-top-10 vs banked\nmean-of-10 (pooled, record-only)",
            r3["delta_pooled"],
            None,
            -r3["tie_band"],
            "tie band ±0.02",
        ),
    ]
    fig, ax = new_fig(8.2, 3.2)
    ax.axvline(0, color=DARKGRAY, linewidth=1)
    for i, (_label, delta, ci, line, line_label) in enumerate(rows):
        y = len(rows) - 1 - i
        if ci is not None:
            ax.plot(ci, [y, y], color=BLUE, linewidth=2.5, solid_capstyle="round")
            ax.plot(
                [line, line],
                [y - 0.28, y + 0.28],
                color=DARKGRAY,
                linewidth=1.2,
                linestyle="--",
            )
            ax.annotate(
                line_label,
                xy=(line, y - 0.32),
                fontsize=8.5,
                color=DARKGRAY,
                ha="center",
                va="top",
            )
        else:
            ax.axvspan(
                line,
                -line,
                ymin=(y + 0.5 - 0.28) / len(rows),
                ymax=(y + 0.5 + 0.28) / len(rows),
                color=GRAY,
                alpha=0.35,
                linewidth=0,
            )
            ax.annotate(
                line_label,
                xy=(line - 0.015, y - 0.32),
                fontsize=8.5,
                color=DARKGRAY,
                ha="right",
                va="top",
            )
        ax.scatter([delta], [y], s=60, color=BLUE, zorder=3)
        ax.annotate(
            f"{delta:+.3f}",
            xy=(delta, y),
            xytext=(delta, y + 0.18),
            fontsize=11,
            color=INK,
            ha="center",
            fontweight="bold",
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.5, color=INK)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel(
        "Δ chunk MAE (negative = searched noise wins)",
        color=DARKGRAY,
        fontsize=10,
    )
    ax.set_title(
        "R2 + R3 — one searched vector is real; the searched ensemble beats "
        "the random ensemble",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.set_ylim(-0.6, len(rows) - 0.2)
    save(fig, "r2_r3_deltas.svg")


def chart_r4a(stage1: dict) -> None:
    per_dataset = stage1["per_dataset"]
    top10 = set(stage1["top10_tickets"])
    winner = int(stage1["winner"]["ticket"])
    counts = np.zeros(64, dtype=int)
    for rec in per_dataset.values():
        counts[int(rec["argmin_ticket"])] += 1
    order = np.argsort(-counts)
    n = len(per_dataset)
    contain = sum(counts[t] for t in top10) / n

    fig, ax = new_fig(8.6, 4.0)
    for rank, idx in enumerate(order):
        t = int(idx)
        color = ORANGE if t == winner else (BLUE if t in top10 else GRAY)
        ax.bar(rank, counts[t], width=0.82, color=color, linewidth=0)
    ax.annotate(
        f"winner 33 is the per-dataset argmin in {counts[winner]}/{n} "
        f"datasets ({counts[winner] / n:.1%})",
        xy=(np.where(order == winner)[0][0], counts[winner]),
        xytext=(8, counts[winner] + 6),
        fontsize=10,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": DARKGRAY, "linewidth": 0.8},
    )
    ax.annotate(
        f"top-10 set contains the argmin {contain:.1%} of the time (null 15.6%)",
        xy=(0.98, 0.82),
        xycoords="axes fraction",
        fontsize=10,
        color=INK,
        ha="right",
    )
    ax.set_xlabel(
        f"64 tickets, sorted by how many of the {n} probe datasets each wins",
        color=DARKGRAY,
        fontsize=10,
    )
    ax.set_ylabel("datasets won", color=DARKGRAY, fontsize=10)
    ax.set_title(
        "R4a — every ticket wins somewhere: the shared winner is task-local, "
        "not universal",
        color=INK,
        fontsize=11,
        loc="left",
    )
    handles = [
        plt.Line2D([], [], marker="s", ls="", color=ORANGE, label="winner (33)"),
        plt.Line2D([], [], marker="s", ls="", color=BLUE, label="top-10 set"),
        plt.Line2D([], [], marker="s", ls="", color=GRAY, label="other tickets"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.75),
    )
    save(fig, "r4a_argmin.svg")


def chart_r4b(stage3: dict) -> None:
    quartiles = stage3["r4b"]["quartiles"]
    names = ["q1_tight", "q2", "q3", "q4_dispersed"]
    labels = [
        "Q1\ntightest draws",
        "Q2",
        "Q3",
        "Q4\nmost dispersed",
    ]
    gains = [quartiles[n]["winner_gain"] for n in names]
    disps = [quartiles[n]["dispersion"] for n in names]

    fig, ax = new_fig(7.2, 4.0)
    ax.axhline(0, color=DARKGRAY, linewidth=1)
    ax.bar(range(4), gains, width=0.62, color=BLUE, linewidth=0)
    for i, (g, d) in enumerate(zip(gains, disps, strict=True)):
        ax.annotate(
            f"{g:+.2f}",
            xy=(i, g),
            xytext=(i, g - 0.07),
            fontsize=11,
            color=INK,
            ha="center",
            va="top",
            fontweight="bold",
        )
        ax.annotate(
            f"σ={d:.1f}",
            xy=(i, 0),
            xytext=(i, 0.05),
            fontsize=9,
            color=DARKGRAY,
            ha="center",
        )
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9.5, color=INK)
    ax.set_ylabel(
        "winner − stable-key, per-frame chunk MAE",
        color=DARKGRAY,
        fontsize=10,
    )
    ax.set_title(
        "R4b — the ticket buys most where the decoder's draws disagree most",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.set_ylim(min(gains) - 0.35, 0.28)
    save(fig, "r4b_quartiles.svg")


def chart_horizon(stage3: dict) -> None:
    h = stage3["r4c_core_horizon"]
    steps = np.arange(1, len(h["winner"]) + 1)
    series = [
        ("stable-key (default noise)", h["stablekey"], DARKGRAY, ":"),
        ("winner ticket 33", h["winner"], ORANGE, "-"),
        ("mean-of-top-10 ensemble", h["mean_of_top10"], BLUE, "-"),
    ]
    fig, ax = new_fig(8.6, 4.4)
    for label, values, color, ls in series:
        ax.plot(steps, values, color=color, linewidth=2, linestyle=ls, label=label)
        ax.annotate(
            f"{values[-1]:.2f}",
            xy=(steps[-1], values[-1]),
            xytext=(steps[-1] + 0.6, values[-1]),
            fontsize=9,
            color=color,
            va="center",
        )
    ax.set_xlabel("step in 50-step action chunk", color=DARKGRAY, fontsize=10)
    ax.set_ylabel("per-step MAE (core rows)", color=DARKGRAY, fontsize=10)
    ax.set_title(
        "R4c — the gains are horizon-wide, not a first-step artifact",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_xlim(1, 55)
    save(fig, "horizon.svg")


def main() -> None:
    stage1 = json.loads(
        (ROOT / "reports/analysis__goldenticket_stage1.json").read_text(),
    )
    stage2 = json.loads(
        (ROOT / "reports/analysis__goldenticket_stage2.json").read_text(),
    )
    stage3 = json.loads(
        (ROOT / "reports/analysis__goldenticket_stage3.json").read_text(),
    )
    chart_r1(stage1)
    chart_deltas(stage2, stage3)
    chart_r4a(stage1)
    chart_r4b(stage3)
    chart_horizon(stage3)


if __name__ == "__main__":
    main()
