"""Results chart for the top-cam gap decomposition probe
(sim-top-gap-foreground-decomposition; eval-report dark scheme).
Two panels: full-frame + shadow-band-crop 5-NN AUROC per arm against
the 0.5 null and the clean-repo anchors, and the paired per-frame
dknn5 vs the fresh v3 baseline with bootstrap CI95 whiskers — the
read that swaps the rendered foreground for real pixels (real-fg)
collapsing the gap while removing the foreground (fg->plate /
plate-only) blows it up is the decomposition's whole story.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
BLUE, ACCENT, RED = "#4589ff", "#ffb000", "#fa4d56"

FULL_ARMS = (
    ("v3\n(baseline)", "v3", META),
    ("v4\n(+shadow)", "v4", BLUE),
    ("fg→plate", "fg_to_plate", RED),
    ("plate only", "plate_only", RED),
    ("real fg\n(real pixels)", "real_fg", ACCENT),
)
CROP_ARMS = (("crop v3", "v3", META), ("crop v4", "v4", BLUE))
PAIRED = (
    ("v4", BLUE),
    ("fg_to_plate", RED),
    ("plate_only", RED),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis",
        type=Path,
        default=Path("reports/analysis__sim_top_gap_decomposition.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/top_gap_decomp/chart__top_gap_decomposition.png"),
    )
    args = ap.parse_args()
    results = json.loads(args.analysis.read_text())["results"]
    full, crop = results["full_frame"], results["crop"]

    fig, (ax, dax) = plt.subplots(1, 2, figsize=(12.0, 4.8), width_ratios=[1.25, 1])
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, dax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)

    labels = [label for label, _, _ in FULL_ARMS] + [label for label, _, _ in CROP_ARMS]
    values = [full["arms"][key]["auroc_vs_real"] for _, key, _ in FULL_ARMS] + [
        crop["arms"][key]["auroc_vs_real"] for _, key, _ in CROP_ARMS
    ]
    colors = [color for _, _, color in FULL_ARMS] + [c for _, _, c in CROP_ARMS]
    x = np.array([0, 1, 2, 3, 4, 5.6, 6.6])
    ax.bar(x, values, 0.62, color=colors, zorder=3)
    ax.axhline(0.5, color=GRID, lw=1.0, zorder=2)
    ax.text(
        4.55,
        0.515,
        "0.5 =\nindistinguishable",
        color=META,
        fontsize=8,
        ha="center",
        va="bottom",
        zorder=4,
    )
    clean = full["clean_anchor"]["auroc_vs_real"]
    ax.axhline(clean, color=ACCENT, lw=1.0, ls=":", zorder=2)
    ax.text(
        4.88,
        clean - 0.02,
        f"clean-repo\nanchor {clean:.3f}",
        color=ACCENT,
        fontsize=8,
        ha="center",
        va="top",
        zorder=4,
    )
    for xi, val in zip(x, values, strict=True):
        ax.text(xi, val + 0.008, f"{val:.3f}", color=TEXT, fontsize=9, ha="center")
    ax.set_xticks(x, labels, color=TEXT, fontsize=8.5)
    ax.axvline(5.0, color=GRID, lw=0.8, ls=":")
    ax.text(6.1, 1.05, "shadow-band crop", color=META, fontsize=8, ha="center")
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("top 5-NN AUROC (arm vs held-out real)", color=TEXT, fontsize=9.5)
    ax.set_title(
        "5-NN AUROC per arm (20 seeds x 5 draws)",
        color=TEXT,
        fontsize=10.5,
        pad=10,
    )

    dx = np.arange(len(PAIRED))
    for xi, (key, color) in enumerate(PAIRED):
        block = full["paired_vs_v3"][key]
        mean, (lo, hi) = block["mean_delta"], block["ci95"]
        dax.errorbar(
            xi,
            mean * 1e6,
            yerr=[[(mean - lo) * 1e6], [(hi - mean) * 1e6]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=5,
            markersize=8,
            zorder=3,
        )
        dax.text(
            xi,
            mean * 1e6 + 0.22,
            f"{block['n_closer']}/100 closer",
            color=TEXT,
            fontsize=8.5,
            ha="center",
        )
    dax.axhline(0.0, color=GRID, lw=1.0, zorder=2)
    dax.set_xticks(
        dx,
        ["v4 (+shadow)", "fg→plate", "plate only"],
        color=TEXT,
        fontsize=9,
    )
    dax.set_xlim(-0.6, 2.6)
    dax.set_ylabel("paired dknn5 vs v3 baseline (x1e-6)", color=TEXT, fontsize=9.5)
    dax.set_title(
        "Paired per-frame dknn5 vs v3, bootstrap CI95",
        color=TEXT,
        fontsize=10.5,
        pad=10,
    )

    fig.suptitle(
        "Top-cam gap decomposition: the residue lives in the rendered "
        "foreground pixels, not the composite arithmetic",
        color=TEXT,
        fontsize=11.5,
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
