"""Subgoal-swap results chart — the presence/channel/CONTENT triangle
readout (analysis__subgoal_swap_ar100k_k4l2.json, frozen reads).

Two panels: (left) the paired-delta CI dot plot — Δ_swap beside the
banked Δ_oracle bound and the swap−oracle contrast, zero line = the
baseline; (right) the horizon fingerprint — per-step delta-vs-baseline
curves for the oracle and swap arms (the −0.464-shaped late-horizon
dive is the content-read's signature; a format effect is flat).

Output: fontaine/blog/src/img/subgoal_swap/swap_reads.svg. House
eval-report dark theme (page #121417, blue = banked oracle context,
amber = the new swap reads; standing owner rule: dark-mode friendly).
Identity is carried by row labels / direct labels, never color alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/subgoal_swap"
SRC = ROOT / "reports/analysis__subgoal_swap_ar100k_k4l2.json"

PAGE = "#121417"
BLUE = "#648fff"  # banked oracle-arm context
AMBER = "#ffb000"  # the new swap reads
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"


def main() -> None:
    data = json.loads(SRC.read_text())
    ob = data["oracle_banked_context"]
    ds = data["delta_swap"]
    sv = data["swap_vs_oracle"]
    rows = [
        (
            "Δ_oracle — TRUE label in the slot\n(banked bound, labeled)",
            ob["delta"],
            ob["ci95"],
            BLUE,
        ),
        (
            "Δ_swap — WRONG label in the slot\n(labeled subset)",
            ds["labeled_subset"]["delta_frame_mean"],
            ds["labeled_subset"]["ci95"],
            AMBER,
        ),
        (
            "Δ_swap (all core frames, primary)",
            ds["core"]["delta_frame_mean"],
            ds["core"]["ci95"],
            AMBER,
        ),
        (
            "swap − oracle (paired, labeled)",
            sv["labeled_subset"]["delta_frame_mean"],
            sv["labeled_subset"]["ci95"],
            AMBER,
        ),
    ]

    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.6, 3.8),
        dpi=110,
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.32},
    )
    fig.patch.set_facecolor(PAGE)

    # ---- left: paired-delta CI dot plot ----
    ax.set_facecolor(PAGE)
    ax.axvline(0, color=META, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    for index, (_, value, (lo, hi), color) in enumerate(rows):
        ax.plot([lo, hi], [index, index], color=color, linewidth=2, zorder=3)
        ax.plot([value], [index], marker="o", markersize=9, color=color, zorder=4)
        ax.annotate(
            f"{value:+.3f}",
            xy=(value, index - 0.32),
            color=TEXT,
            fontsize=9.5,
            ha="center",
            zorder=4,
        )
    ax.annotate(
        "baseline (no slot) = 0",
        xy=(0, -0.62),
        color=META,
        fontsize=8.5,
        ha="center",
        annotation_clip=False,
    )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=TEXT, fontsize=9)
    ax.set_ylim(-0.8, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(
        "paired per-frame chunk-MAE delta (negative = better than baseline)",
        color=META,
        fontsize=9,
    )
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ---- right: horizon fingerprint ----
    ax2.set_facecolor(PAGE)
    base = np.array(data["curves"]["baseline"])
    steps = np.arange(1, len(base) + 1)
    ax2.axhline(0, color=META, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    for name, color, label in (
        ("oracle", BLUE, "oracle − baseline"),
        ("swap", AMBER, "swap − baseline"),
    ):
        curve = np.array(data["curves"][name]) - base
        ax2.plot(steps, curve, color=color, linewidth=2, zorder=3)
        ax2.annotate(
            label,
            xy=(steps[-1], curve[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
            annotation_clip=False,
        )
    ax2.set_xlabel("step in 50-step horizon", color=META, fontsize=9)
    ax2.set_ylabel("delta vs baseline (MAE)", color=META, fontsize=9)
    ax2.tick_params(colors=META, labelsize=9)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)

    fig.suptitle(
        "Subgoal-swap: what the slot does with WRONG words",
        color=TEXT,
        fontsize=11.5,
        x=0.005,
        y=1.02,
        ha="left",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "swap_reads.svg", facecolor=PAGE, bbox_inches="tight")
    fig.savefig(OUT / "swap_reads.png", facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {OUT / 'swap_reads.svg'} (+.png)")


if __name__ == "__main__":
    main()
