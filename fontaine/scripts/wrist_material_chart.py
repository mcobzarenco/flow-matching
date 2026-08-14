"""Results chart for the wrist-view material read (queue
`sim-wrist-view-material-read`; eval-report dark scheme — wrist Carbon
yellow-30, top teal, the shipped pair). Two panels: the registered
paired dknn5 reads (wrist PRIMARY vs the top record-only rider) with
bootstrap CI95 whiskers against the zero rule, and the per-camera
AUROC context ladder with the pinned anchors.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
WRIST, TOP, ANCHOR = "#ffb000", "#08bdba", "#9aa0a8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis",
        type=Path,
        default=Path("reports/analysis__sim_wrist_material_read.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/wrist_material/chart__wrist_material_read.png"),
    )
    args = ap.parse_args()
    results = json.loads(args.analysis.read_text())["results"]

    fig, (dax, ax) = plt.subplots(1, 2, figsize=(13.2, 4.6), width_ratios=[1, 1.4])
    fig.patch.set_facecolor(PAGE)
    for panel in (dax, ax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)
        panel.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
        panel.set_axisbelow(True)

    # left: the registered paired reads, CI95 whiskers, zero rule
    reads = [
        (
            "PRIMARY (registered)\nwrist: stack - v3",
            results["wrist"]["paired_stack_vs_v3"],
            WRIST,
        ),
        (
            "record-only rider\ntop: stack - v3",
            results["top"]["paired_stack_vs_v3"],
            TOP,
        ),
    ]
    spans = []
    for y, (label, read, color) in enumerate(reads):
        lo, hi = (v * 1e7 for v in read["ci95"])
        spans.extend((lo, hi))
        dax.plot([lo, hi], [y, y], color=color, lw=2.4)
        dax.plot([read["mean_delta"] * 1e7], [y], "o", color=color, ms=8)
        dax.text(
            (lo + hi) / 2,
            y + 0.18,
            f"{read['n_closer']}/100 slots closer",
            color=META,
            fontsize=8.5,
            ha="center",
        )
        dax.text(
            (lo + hi) / 2,
            y - 0.34,
            label,
            color=TEXT,
            fontsize=9.5,
            ha="center",
            va="center",
        )
    dax.axvline(0.0, color=TEXT, linewidth=1.0)
    lo, hi = min(spans), max(spans)
    pad = 0.15 * max(hi - lo, 1.0)
    dax.set_xlim(lo - pad, max(hi + pad, 0.6))
    dax.text(0.05, 1.55, "0 = no change", color=META, fontsize=8.5, ha="left")
    dax.set_ylim(-0.6, 1.9)
    dax.set_yticks([])
    dax.set_xlabel(
        "paired Δknn5 vs real reference (×1e-7), CI95 10k resamples",
        color=META,
    )
    dax.set_title(
        "Two-flag stack, paired per camera — wrist CI straddles zero",
        color=TEXT,
        fontsize=11,
        loc="left",
    )

    # right: per-camera AUROC context ladder with pinned anchors
    rows = [
        (
            "wrist (settled resets)\nv3 -> stack",
            results["wrist"]["arms"]["v3_wrist"]["auroc_vs_real"],
            results["wrist"]["arms"]["v3_stack_wrist"]["auroc_vs_real"],
            WRIST,
        ),
        (
            "top (same slots)\nv3 -> stack",
            results["top"]["arms"]["v3_top"]["auroc_vs_real"],
            results["top"]["arms"]["v3_stack_top"]["auroc_vs_real"],
            TOP,
        ),
    ]
    for y, (label, before, after, color) in enumerate(rows):
        ax.annotate(
            "",
            xy=(after, y),
            xytext=(before, y),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.2},
        )
        ax.plot([before], [y], "o", color=META, ms=7, zorder=3)
        ax.plot([after], [y], "o", color=color, ms=8, zorder=3)
        ax.text(
            (before + after) / 2,
            y + 0.17,
            f"{before:.3f} → {after:.3f}",
            color=color,
            fontsize=9.5,
            fontweight="bold",
            ha="center",
        )
        ax.text(0.175, y - 0.02, label, color=TEXT, fontsize=9.5, va="center")
    context = results["context"]
    wrist_clean = results["wrist"]["clean_anchor"]["auroc_vs_real"]
    for x, name, height in (
        (0.5, "chance 0.5", 1.42),
        (
            context["reset_wrist_baselines_100x1"][0],
            "banked resets\n0.544 / 0.548",
            1.62,
        ),
        (
            context["rollout_wrist_baseline_knn5_auroc"],
            "wrist ROLLOUT\nbaseline 0.828",
            1.42,
        ),
        (wrist_clean, f"wrist clean {wrist_clean:.3f}", 1.62),
    ):
        ax.axvline(x, color=ANCHOR, linewidth=0.8, linestyle=":", alpha=0.8)
        ax.text(x, height, name, color=META, fontsize=8, ha="center", va="bottom")
    ax.set_xlim(0.15, 0.95)
    ax.set_ylim(-0.5, 2.15)
    ax.set_yticks([])
    ax.set_xlabel("knn5 AUROC vs held-out real (lower = closer to real)", color=META)
    ax.set_title(
        "Where the wrist read sits — reset poses are already near-chance",
        color=TEXT,
        fontsize=11,
        loc="left",
    )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
