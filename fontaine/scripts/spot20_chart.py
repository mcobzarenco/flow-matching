"""House dark-mode chart for the spot20-v3 close: per-seed paired
delta progress (v3 - v0) strips per arm, CI + mean annotated.

Usage:
  uv run python fontaine/scripts/spot20_chart.py \
      --analysis reports/analysis__spot20_v3_reads.json \
      --out reports/chart__spot20_v3_deltas.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
ARMS = (  # fixed order, bottom to top; house categorical hues
    ("teacher80k", "teacher80k (heun-30)", "#dc267f"),
    ("snap30k", "snap30k (snapflow, euler-1)", "#785ef0"),
    ("er60k", "er60k (reference trunk)", "#648fff"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())

    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=150)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.axvline(0.0, color=META, linewidth=1.2, zorder=1)
    rng = np.random.default_rng(0)
    notes = []
    for row, (key, _label, color) in enumerate(ARMS):
        block = data["arms"][key]["paired_delta_progress_final_cm"]
        deltas = np.asarray(block["per_seed"])
        jitter = rng.uniform(-0.22, 0.22, len(deltas))
        ax.scatter(
            deltas,
            np.full(len(deltas), row) + jitter,
            s=22,
            color=color,
            alpha=0.6,
            linewidths=0,
            zorder=2,
        )
        lo, hi = block["ci95"]
        ax.plot(
            [lo, hi],
            [row - 0.33, row - 0.33],
            color=color,
            linewidth=2.0,
            zorder=3,
        )
        ax.plot(
            [block["mean"], block["mean"]],
            [row - 0.42, row - 0.24],
            color=color,
            linewidth=2.5,
            zorder=4,
        )
        star = " — CI excludes zero" if block["excludes_zero"] else ""
        notes.append(
            (row, f"Δ {block['mean']:+.2f} cm [{lo:+.2f}, {hi:+.2f}]{star}"),
        )
    right = ax.get_xlim()[1]
    for row, note in notes:
        ax.text(
            right,
            row + 0.34,
            note,
            ha="right",
            va="bottom",
            color=META,
            fontsize=8.5,
        )
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([label for _, label, _ in ARMS], color=TEXT, fontsize=9)
    ax.set_ylim(-0.7, len(ARMS) - 0.2)
    ax.set_xlabel(
        "paired per-seed Δ progress_final (v3 − v0 visuals, cm; >0 = toward the disk)",
        fontsize=9,
        color=TEXT,
    )
    ax.set_title(
        "Same physics, new eyes: 20-seed paired behavioral deltas under v3 visuals",
        fontsize=10.5,
        loc="left",
        color=TEXT,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
