"""Boundary-incompatibility dt-curve chart — seam disagreement D vs
observation gap dt for the banked panel stacks
(analysis__boundary_incompat_panels.json).

Output: fontaine/blog/src/img/boundary_incompat/dt_curve.svg. House
eval-report dark theme (page #121417; standing owner rule: dark-mode
friendly). molmo2 AR-40k is omitted from the plot (within 0.12 of the
60k curve at every bin — stated in the caption, numbers in the JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/boundary_incompat"
SRC = ROOT / "reports/analysis__boundary_incompat_panels.json"

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
SERIES = [
    # (stack label in json, display label, color)
    ("flow80k_stablekey", "flow 80k — fresh noise per frame (stable-key)", "#dc267f"),
    ("molmo2_ar60k_greedy", "molmo2 AR 60k — greedy (deterministic)", "#648fff"),
    ("flow80k_draws10mean", "flow 80k — mean of 10 draws", "#785ef0"),
    ("flow80k_ticket33", "flow 80k — one shared noise ticket", "#ffb000"),
]


def main() -> None:
    data = json.loads(SRC.read_text())["stacks"]

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=110)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)

    for key, label, color in SERIES:
        bins = data[key]["model"]["dt_bins"]
        x = [(b["dt"][0] + b["dt"][1]) / 2 for b in bins]
        y = [b["D_mean"] for b in bins]
        lo = [b["D_mean"] - b["ci95"][0] for b in bins]
        hi = [b["ci95"][1] - b["D_mean"] for b in bins]
        ax.errorbar(
            x,
            y,
            yerr=[lo, hi],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            capsize=2,
            zorder=4,
            label=label,
        )
        ax.annotate(
            f"{y[-1]:.1f}",
            xy=(x[-1] + 1.2, y[-1]),
            color=color,
            fontsize=9,
            va="center",
            annotation_clip=False,
        )

    # Scale reference: scene motion (state-copy D, identical across
    # stacks) and typical per-step motion of the truth trajectory.
    sc_bins = data["flow80k_stablekey"]["state_copy_reference"]["dt_bins"]
    ax.plot(
        [(b["dt"][0] + b["dt"][1]) / 2 for b in sc_bins],
        [b["D_mean"] for b in sc_bins],
        color=META,
        linestyle=(0, (4, 3)),
        linewidth=1.4,
        zorder=2,
        label="state-copy — scene motion over dt",
    )
    w_truth = data["flow80k_stablekey"]["model"]["within_chunk_step_truth"]
    ax.axhline(w_truth, color=META, linestyle=(0, (1, 3)), linewidth=1.2, zorder=2)
    ax.annotate(
        f"typical per-step motion (truth) = {w_truth:.2f}",
        xy=(1, w_truth + 0.25),
        color=META,
        fontsize=8.5,
        va="bottom",
    )

    ax.set_xlabel(
        "observation gap dt between the two frames (ticks)",
        color=TEXT,
        fontsize=10,
    )
    ax.set_ylabel(
        "seam disagreement D — MAE between the two chunks\n"
        "on their overlap (norm. action units)",
        color=TEXT,
        fontsize=10,
    )
    ax.set_title(
        "Chunks disagree at the seam — and fresh noise nearly triples the dt→0 floor",
        color=TEXT,
        fontsize=11.5,
        pad=12,
    )
    ax.set_xlim(0, 56)
    ax.set_ylim(0, 11.2)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    leg = ax.legend(
        loc="lower right",
        fontsize=8.5,
        frameon=True,
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
    )
    leg.set_zorder(5)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "dt_curve.svg", format="svg", facecolor=PAGE)
    print(f"wrote {OUT / 'dt_curve.svg'}")


if __name__ == "__main__":
    main()
