"""House dark-mode chart for the encoder OOD probe: per-camera strip
plots of the distance-to-real reads (centroid primary | knn5
secondary), three groups each, AUROC + gap ratio annotated.

Reads ONLY the banked analysis json (regenerable, no live hosts).

Usage:
  uv run python fontaine/scripts/sim_encoder_ood_chart.py \
      --analysis reports/analysis__sim_encoder_ood_probe.json \
      --out fontaine/blog/src/img/sim_ood/ood_distances.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# House eval-report scheme (sim100_charts.py): dark page, IBM CVD-safe
# categorical hues validated on this surface 2026-08-11; identity is
# never color-alone — groups are named on the y axis.
PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
GROUPS = (  # fixed order, bottom to top
    ("sim", "sim (er60k rollouts)", "#dc267f"),
    ("real_clean", "real clean (anchor)", "#ffb000"),
    ("real_heldout", "real v2 held-out", "#648fff"),
)
SCALE = 1e5  # distances ride a ~1e-5 residual; plot in units of 1e-5


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def strip_panel(
    ax: plt.Axes,
    reads: dict,
    *,
    title: str,
    ratio: float,
    auroc: float,
) -> None:
    rng = np.random.default_rng(0)
    for row, (key, _label, color) in enumerate(GROUPS):
        values = np.asarray(reads["distances"][key]) * SCALE
        jitter = rng.uniform(-0.22, 0.22, len(values))
        ax.scatter(
            values,
            np.full(len(values), row) + jitter,
            s=14,
            color=color,
            alpha=0.45,
            linewidths=0,
            zorder=2,
        )
        ax.plot(
            [values.mean(), values.mean()],
            [row - 0.32, row + 0.32],
            color=color,
            linewidth=2.5,
            zorder=3,
        )
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels([label for _, label, _ in GROUPS], color=TEXT, fontsize=9)
    ax.set_ylim(-0.6, len(GROUPS) - 0.4)
    ax.set_title(title, fontsize=10, loc="left")
    ax.text(
        0.98,
        0.04,
        f"sim gap {ratio:.2f}x · AUROC {auroc:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=META,
        fontsize=9,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.analysis.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.0), dpi=150, sharey=True)
    fig.patch.set_facecolor(PAGE)
    for row, camera in enumerate(("top", "wrist")):
        reads = data["cameras"][camera]
        panels = (
            (reads, "centroid cosine (registered primary)"),
            (reads["knn5_secondary"], "5-NN cosine (secondary)"),
        )
        for col, (block, metric) in enumerate(panels):
            ax = axes[row][col]
            style_axis(ax)
            strip_panel(
                ax,
                block,
                title=f"{camera} camera — {metric}",
                ratio=block["gap_ratio_sim_vs_real"],
                auroc=block["auroc_sim_vs_real"],
            )
            if row == 1:
                ax.set_xlabel("cosine distance to real reference  (x1e-5)", fontsize=9)
    fig.suptitle(
        "Sim frames at the policy's eyes: er_60k vision-trunk distance to the real-rig manifold",
        color=TEXT,
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
