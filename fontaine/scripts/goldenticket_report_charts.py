"""Charts for the consolidated golden-ticket visual report (owner
steering 2026-08-08 08:42Z; seating refresh 23:2xZ): six SVGs rendered
from the BANKED stage analyses — no re-computation of any claimed
number, every value read from the frozen jsons
(`analysis__goldenticket_stage{1,2,3}.json`,
`analysis__noise_ladder_seating.json`).

Output: fontaine/blog/src/img/goldenticket/*.svg (committed).

Palette: the eval reports' DARK theme (dark_background + the IBM
colorblind-safe pair #648fff/#ffb000 on page #121417; standing owner
rule 2026-08-08 16:32Z — the set predates the rule and is retro-fitted
here because the seating refresh touches it). No node on the box, so
the palette validator can't run; the standing pre-validated pair
unchanged is the sanctioned fallback (same as
`noiseladder_rung2_charts.py`).
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

PAGE = "#121417"
BLUE = "#648fff"  # top-10 ticket set / seated ensemble
AMBER = "#ffb000"  # winner ticket 33
MUTED = "#6a7178"  # other tickets / reference marks
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"

# Decision-line constants (pre-registered bars, not measurements):
AR_BAR = 5.8026  # AR-100k greedy chunk MAE — the pre-reg comparison bar
STAR_BAR = 5.0  # the leaderboard ☆ bar


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=META, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def new_fig(width: float = 8.2, height: float = 4.2) -> tuple:
    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    fig.patch.set_facecolor(PAGE)
    style_axes(ax)
    return fig, ax


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor=PAGE)
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
        color=META,
        alpha=0.18,
        linewidth=0,
    )
    ax.axhline(mean, color=META, linewidth=1, linestyle=":")
    ax.axhline(r1["expected_null_min"], color=META, linewidth=1, linestyle="--")
    for rank, idx in enumerate(order):
        t = int(idx)
        color = AMBER if t == winner else (BLUE if t in top10 else MUTED)
        size = 46 if t == winner else (30 if t in top10 else 16)
        ax.scatter(rank, scores[t], s=size, color=color, zorder=3)
    ax.annotate(
        f"ticket 33 — {scores[winner]:.3f}",
        xy=(0, scores[winner]),
        xytext=(3.5, scores[winner] - 0.06),
        fontsize=10,
        color=TEXT,
        va="top",
    )
    ax.annotate(
        f"null band: mean ± 2σ (σ = {sigma})",
        xy=(63, mean + 2 * sigma),
        fontsize=9,
        color=META,
        ha="right",
        va="bottom",
    )
    ax.annotate(
        f"expected null minimum {r1['expected_null_min']:.3f}",
        xy=(63, r1["expected_null_min"]),
        fontsize=9,
        color=META,
        ha="right",
        va="top",
    )
    ax.set_xlabel("64 tickets, sorted by probe score", color=META, fontsize=10)
    ax.set_ylabel("pooled probe chunk MAE", color=META, fontsize=10)
    ax.set_title(
        "R1 — the ticket distribution is ~12x wider than the i.i.d.-noise null "
        f"(sd {r1['sd']:.3f} vs line {r1['sd_line']})",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=AMBER, label="winner (33)"),
        plt.Line2D([], [], marker="o", ls="", color=BLUE, label="top-10 set"),
        plt.Line2D([], [], marker="o", ls="", color=MUTED, label="other tickets"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    save(fig, "r1_tickets.svg")


def chart_deltas(stage2: dict, seating: dict) -> None:
    r2 = stage2["r2"]
    paired = seating["read_paired"]
    rows = [
        (
            "R2 — winner vs stable-key\n(held-out complement rows, paired)",
            r2["delta_pooled"],
            r2["ci95"],
            r2["line"],
            "adopt line −0.05",
        ),
        (
            "R3 seated — mean-of-top-10 vs\nmean-of-random-10 (paired re-run)",
            paired["delta_pooled"],
            paired["ci95"],
            0.0,
            "seat rule: CI95 < 0",
        ),
    ]
    clustered = paired["ci95_clustered_record_only"]
    fig, ax = new_fig(8.2, 3.2)
    ax.axvline(0, color=META, linewidth=1)
    for i, (_label, delta, ci, line, line_label) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot(ci, [y, y], color=BLUE, linewidth=2.5, solid_capstyle="round")
        if line != 0.0:
            ax.plot(
                [line, line],
                [y - 0.28, y + 0.28],
                color=META,
                linewidth=1.2,
                linestyle="--",
            )
        ax.annotate(
            line_label,
            xy=(line, y - 0.32),
            fontsize=8.5,
            color=META,
            ha="center" if line != 0.0 else "right",
            va="top",
        )
        ax.scatter([delta], [y], s=60, color=BLUE, zorder=3)
        ax.annotate(
            f"{delta:+.3f}",
            xy=(delta, y),
            xytext=(delta, y + 0.18),
            fontsize=11,
            color=TEXT,
            ha="center",
            fontweight="bold",
        )
    # dataset-clustered CI on the seated row, thin under-whisker (record-only)
    ax.plot(clustered, [-0.15, -0.15], color=BLUE, linewidth=1, alpha=0.6)
    ax.annotate(
        "dataset-clustered CI (record-only)",
        xy=(clustered[0] - 0.012, -0.15),
        fontsize=8,
        color=META,
        ha="right",
        va="center",
    )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.5, color=TEXT)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel(
        "Δ chunk MAE (negative = searched noise wins)",
        color=META,
        fontsize=10,
    )
    ax.set_title(
        "R2 + R3 — one searched vector is real; the searched ensemble is "
        "confirmed and seated",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    ax.set_ylim(-0.6, len(rows) - 0.2)
    save(fig, "r2_r3_deltas.svg")


def chart_board(seating: dict) -> None:
    random10 = seating["gates"]["base_equality_npz_record"][0]
    top10 = seating["gates"]["top10_anchor"][0]
    rows = [
        ("AR-100k greedy (the AR bar)", AR_BAR, MUTED),
        ("flow heun-30, mean-of-10 random draws", random10, BLUE),
        ("flow heun-30, mean-of-top-10 TICKETS", top10, AMBER),
    ]
    fig, ax = new_fig(8.2, 3.1)
    ax.axvline(STAR_BAR, color=META, linewidth=1.2, linestyle="--")
    for i, (label, value, color) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot(
            [STAR_BAR, value],
            [y, y],
            color=color,
            linewidth=1.6,
            alpha=0.7,
        )
        ax.scatter([value], [y], s=90, color=color, zorder=3)
        ax.annotate(
            f"{value:.4f}",
            xy=(value, y),
            xytext=(value + 0.035, y),
            fontsize=11,
            color=TEXT,
            va="center",
            fontweight="bold",
        )
        ax.annotate(
            label,
            xy=(STAR_BAR + 0.02, y + 0.2),
            fontsize=9.5,
            color=META,
            va="bottom",
        )
    ax.annotate(
        "☆ bar 5.0",
        xy=(STAR_BAR + 0.02, -0.62),
        fontsize=9.5,
        color=META,
        ha="left",
        va="center",
    )
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlim(STAR_BAR - 0.1, AR_BAR + 0.3)
    ax.set_ylim(-0.8, len(rows) - 0.1)
    ax.set_xlabel("panel chunk MAE (lower is better)", color=META, fontsize=10)
    ax.set_title(
        "Where the board stands — the seated ticket ensemble closes half the "
        "remaining gap to the ☆ bar (0.37 → 0.18)",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    save(fig, "seating_board.svg")


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
        color = AMBER if t == winner else (BLUE if t in top10 else MUTED)
        ax.bar(rank, counts[t], width=0.82, color=color, linewidth=0)
    ax.annotate(
        f"winner 33 is the per-dataset argmin in {counts[winner]}/{n} "
        f"datasets ({counts[winner] / n:.1%})",
        xy=(np.where(order == winner)[0][0], counts[winner]),
        xytext=(8, counts[winner] + 6),
        fontsize=10,
        color=TEXT,
        arrowprops={"arrowstyle": "-", "color": META, "linewidth": 0.8},
    )
    ax.annotate(
        f"top-10 set contains the argmin {contain:.1%} of the time (null 15.6%)",
        xy=(0.98, 0.82),
        xycoords="axes fraction",
        fontsize=10,
        color=TEXT,
        ha="right",
    )
    ax.set_xlabel(
        f"64 tickets, sorted by how many of the {n} probe datasets each wins",
        color=META,
        fontsize=10,
    )
    ax.set_ylabel("datasets won", color=META, fontsize=10)
    ax.set_title(
        "R4a — every ticket wins somewhere: the shared winner is task-local, "
        "not universal",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    handles = [
        plt.Line2D([], [], marker="s", ls="", color=AMBER, label="winner (33)"),
        plt.Line2D([], [], marker="s", ls="", color=BLUE, label="top-10 set"),
        plt.Line2D([], [], marker="s", ls="", color=MUTED, label="other tickets"),
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
    ax.axhline(0, color=META, linewidth=1)
    ax.bar(range(4), gains, width=0.62, color=BLUE, linewidth=0)
    for i, (g, d) in enumerate(zip(gains, disps, strict=True)):
        ax.annotate(
            f"{g:+.2f}",
            xy=(i, g),
            xytext=(i, g - 0.07),
            fontsize=11,
            color=TEXT,
            ha="center",
            va="top",
            fontweight="bold",
        )
        ax.annotate(
            f"σ={d:.1f}",
            xy=(i, 0),
            xytext=(i, 0.05),
            fontsize=9,
            color=META,
            ha="center",
        )
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9.5, color=TEXT)
    ax.set_ylabel(
        "winner − stable-key, per-frame chunk MAE",
        color=META,
        fontsize=10,
    )
    ax.set_title(
        "R4b — the ticket buys most where the decoder's draws disagree most",
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    ax.set_ylim(min(gains) - 0.35, 0.28)
    save(fig, "r4b_quartiles.svg")


def chart_horizon(stage3: dict) -> None:
    h = stage3["r4c_core_horizon"]
    steps = np.arange(1, len(h["winner"]) + 1)
    series = [
        ("stable-key (default noise)", h["stablekey"], META, ":"),
        ("winner ticket 33", h["winner"], AMBER, "-"),
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
    ax.set_xlabel("step in 50-step action chunk", color=META, fontsize=10)
    ax.set_ylabel("per-step MAE (core rows)", color=META, fontsize=10)
    ax.set_title(
        "R4c — the gains are horizon-wide, not a first-step artifact",
        color=TEXT,
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
    seating = json.loads(
        (ROOT / "reports/analysis__noise_ladder_seating.json").read_text(),
    )
    with plt.style.context("dark_background"):
        chart_r1(stage1)
        chart_deltas(stage2, seating)
        chart_board(seating)
        chart_r4a(stage1)
        chart_r4b(stage3)
        chart_horizon(stage3)


if __name__ == "__main__":
    main()
