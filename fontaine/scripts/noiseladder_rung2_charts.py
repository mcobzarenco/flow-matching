"""Charts for the noise-ladder rung-2 stage-2 results post — rendered
from the BANKED adjudication json (`analysis__noise_ladder_rung2.json`),
no re-computation of any claimed number.

Output: fontaine/blog/src/img/noiseladder/*.svg (committed).

Palette: the eval reports' DARK theme (dark_background + the IBM
colorblind-safe pair #648fff/#ffb000 on page #121417; standing owner
rule 2026-08-08 16:32Z). No node on the box, so the palette validator
can't run; the standing pre-validated pair unchanged is the
sanctioned fallback. Polarity chart uses the pair as poles (blue =
routing wins, amber = routing loses) with a neutral zero line.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/noiseladder"
ANALYSIS = ROOT / "reports/analysis__noise_ladder_rung2.json"

PAGE = "#121417"
BLUE = "#648fff"  # routing wins / routed series
AMBER = "#ffb000"  # routing loses / ticket-33 series
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"


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


def per_dataset_chart(d: dict) -> None:
    table = d["read3_win_table"]["per_dataset"]
    deltas = sorted(row["delta_route"] for row in table.values())
    read1 = d["read1_primary"]
    wins = d["read3_win_table"]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=110)
        fig.patch.set_facecolor(PAGE)
        style_axes(ax)
        x = np.arange(len(deltas))
        colors = [BLUE if v < 0 else (GRID if v == 0 else AMBER) for v in deltas]
        ax.bar(x, deltas, width=0.82, color=colors, linewidth=0)
        ax.axhline(0, color=META, linewidth=1.0)
        lo, hi = read1["ci95_clustered"]
        ax.axhspan(lo, hi, color=AMBER, alpha=0.14, linewidth=0)
        ax.axhline(
            read1["delta_route_pooled"],
            color=AMBER,
            linewidth=1.2,
            linestyle=(0, (4, 3)),
        )
        ax.text(
            1,
            read1["delta_route_pooled"] + 0.06,
            f"pooled Δ_route +{read1['delta_route_pooled']:.3f}"
            f"  (CI95 [{lo:+.3f}, {hi:+.3f}] — entirely above 0)",
            color=AMBER,
            fontsize=9.5,
        )
        ax.text(
            0.01,
            0.97,
            f"routing wins {wins['wins']}",
            transform=ax.transAxes,
            color=BLUE,
            fontsize=10,
            va="top",
        )
        ax.text(
            0.01,
            0.90,
            f"routing loses {wins['losses']}  (ties {wins['ties']}; "
            f"sign p = {wins['p_sign_two_sided']:.3f})",
            transform=ax.transAxes,
            color=AMBER,
            fontsize=10,
            va="top",
        )
        ax.set_xlabel("qualifying dataset (sorted by Δ_route)", color=META, fontsize=10)
        ax.set_ylabel(
            "Δ_route = routed − ticket 33 (chunk MAE)",
            color=META,
            fontsize=10,
        )
        ax.set_title(
            "Per-dataset routing, out-of-sample: the map loses\n"
            "Δ_route per qualifying dataset, complement core rows",
            color=TEXT,
            fontsize=11,
            loc="left",
            pad=12,
        )
        ax.set_xlim(-1, len(deltas))
        fig.tight_layout()
        fig.savefig(OUT / "rung2_per_dataset_delta.svg", facecolor=PAGE)
        plt.close(fig)


def horizon_chart(d: dict) -> None:
    mirrors = d["read4_mirrors"]
    routed = np.asarray(mirrors["complement_horizon_routed"])
    t33 = np.asarray(mirrors["complement_horizon_ticket33"])
    steps = np.arange(1, len(routed) + 1)

    with plt.style.context("dark_background"):
        fig, (ax, axd) = plt.subplots(
            2,
            1,
            figsize=(9.2, 5.6),
            dpi=110,
            sharex=True,
            height_ratios=[2.2, 1.0],
        )
        fig.patch.set_facecolor(PAGE)
        for a in (ax, axd):
            style_axes(a)
        ax.plot(
            steps,
            routed,
            color=BLUE,
            linewidth=2.0,
            label="routed (per-dataset map)",
        )
        ax.plot(
            steps,
            t33,
            color=AMBER,
            linewidth=2.0,
            label="ticket 33 (global winner)",
        )
        ax.text(
            steps[-1] + 0.4,
            routed[-1],
            "routed",
            color=BLUE,
            fontsize=9.5,
            va="center",
        )
        ax.text(
            steps[-1] + 0.4,
            t33[-1],
            "ticket 33",
            color=AMBER,
            fontsize=9.5,
            va="center",
        )
        ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT)
        ax.set_ylabel("MAE at horizon step", color=META, fontsize=10)
        ax.set_title(
            "Horizon mirror on the qualifying complement: routed vs ticket 33"
            " — the gap opens against routing late",
            color=TEXT,
            fontsize=11,
            loc="left",
            pad=12,
        )
        ax.set_xlim(1, len(steps) + 4)

        delta = routed - t33
        axd.axhline(0, color=META, linewidth=1.0)
        axd.plot(steps, delta, color=TEXT, linewidth=1.8)
        axd.fill_between(
            steps,
            0,
            delta,
            where=delta > 0,
            color=AMBER,
            alpha=0.25,
            linewidth=0,
        )
        axd.fill_between(
            steps,
            0,
            delta,
            where=delta < 0,
            color=BLUE,
            alpha=0.25,
            linewidth=0,
        )
        axd.set_ylabel("routed − t33", color=META, fontsize=10)
        axd.set_xlabel("horizon step within the action chunk", color=META, fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "rung2_horizon_mirror.svg", facecolor=PAGE)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(ANALYSIS.read_text())
    per_dataset_chart(d)
    horizon_chart(d)
    print(f"wrote {OUT}/rung2_per_dataset_delta.svg + rung2_horizon_mirror.svg")


if __name__ == "__main__":
    main()
