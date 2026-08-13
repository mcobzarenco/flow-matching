"""Results chart for the foreground content split
(sim-foreground-appearance-pass leg (a); eval-report dark scheme,
class hues CVD-validated on the dark surface). Two panels: 5-NN AUROC
per arm against the 0.5 null and the banked anchors, and the paired
per-frame dknn5 with bootstrap CI95 whiskers (removal arms vs v3,
keep-only arms vs plate-only) — the clutter stand-ins' visible pixels
(~5% of frame) carrying −0.137 AUROC on removal is the story.
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
CLASS_COLOR = {
    "arm": "#4589ff",
    "benchy": "#08bdba",
    "clutter": "#ffb000",
    "disk": "#fa4d56",
}
CLASSES = ("arm", "benchy", "clutter", "disk")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis",
        type=Path,
        default=Path("reports/analysis__sim_fg_content_split.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/fg_split/chart__sim_fg_content_split.png"),
    )
    args = ap.parse_args()
    payload = json.loads(args.analysis.read_text())
    results = payload["results"]
    arms = results["arms"]
    anchors = payload["config"]["anchors"]

    fig, (ax, dax) = plt.subplots(1, 2, figsize=(12.6, 4.8), width_ratios=[1.3, 1])
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, dax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)

    order = (
        [("v3\n(baseline)", "v3", META), ("plate\nonly", "plate_only", META)]
        + [(f"no\n{c}", f"no_{c}", CLASS_COLOR[c]) for c in CLASSES]
        + [(f"only\n{c}", f"only_{c}", CLASS_COLOR[c]) for c in CLASSES]
    )
    x = np.concatenate([[0, 1], 2.6 + np.arange(4), 7.2 + np.arange(4)])
    values = [arms[key]["auroc_vs_real"] for _, key, _ in order]
    ax.bar(x, values, 0.62, color=[c for _, _, c in order], zorder=3)
    ax.axhline(0.5, color=GRID, lw=1.0, zorder=2)
    ax.text(
        1.8,
        0.487,
        "0.5 null",
        color=META,
        fontsize=7.5,
        ha="center",
        va="top",
        zorder=4,
    )
    clean = results["clean_anchor"]["auroc_vs_real"]
    ax.axhline(clean, color=META, lw=1.0, ls=":", zorder=2)
    ax.text(
        6.4,
        clean - 0.007,
        f"clean {clean:.3f}",
        color=META,
        fontsize=7.5,
        ha="center",
        va="top",
        zorder=4,
    )
    real_fg = anchors["decomposition_real_fg"]
    ax.axhline(real_fg, color=META, lw=1.0, ls="--", zorder=2)
    ax.text(
        6.4,
        real_fg + 0.007,
        f"real-fg {real_fg:.3f}",
        color=META,
        fontsize=7.5,
        ha="center",
        va="bottom",
        zorder=4,
    )
    for xi, val in zip(x, values, strict=True):
        ax.text(xi, val + 0.008, f"{val:.3f}", color=TEXT, fontsize=8.4, ha="center")
    v3 = arms["v3"]["auroc_vs_real"]
    for xi, (_, key, _) in zip(x, order, strict=True):
        if key.startswith("no_"):
            delta = arms[key]["auroc_vs_real"] - v3
            ax.text(
                xi,
                0.03,
                f"{delta:+.3f}",
                color=TEXT,
                fontsize=8,
                ha="center",
                va="bottom",
            )
    ax.annotate(
        "target class\n(registered rule)",
        xy=(4.6, values[4] + 0.05),
        xytext=(4.6, 0.93),
        color=CLASS_COLOR["clutter"],
        fontsize=8.5,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": CLASS_COLOR["clutter"], "lw": 0.8},
    )
    ax.set_xticks(x, [label for label, _, _ in order], color=TEXT, fontsize=8.2)
    for divider in (1.8, 6.4):
        ax.axvline(divider, color=GRID, lw=0.8, ls=":")
    ax.text(
        4.1,
        1.06,
        "class removed → plate (Δ vs v3 below)",
        color=META,
        fontsize=8,
        ha="center",
    )
    ax.text(8.7, 1.06, "class alone on the plate", color=META, fontsize=8, ha="center")
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("top 5-NN AUROC (arm vs held-out real)", color=TEXT, fontsize=9.5)
    ax.set_title(
        "5-NN AUROC per arm (20 seeds x 5 draws, shared physics/plate/noise)",
        color=TEXT,
        fontsize=10.5,
        pad=10,
    )

    for xi, cls in enumerate(CLASSES):
        for offset, block, marker in (
            (0.0, results["paired_vs_v3"][f"no_{cls}"], "o"),
            (5.0, results["paired_vs_plate_only"][f"only_{cls}"], "s"),
        ):
            mean, (lo, hi) = block["mean_delta"], block["ci95"]
            dax.errorbar(
                xi + offset,
                mean * 1e6,
                yerr=[[(mean - lo) * 1e6], [(hi - mean) * 1e6]],
                fmt=marker,
                color=CLASS_COLOR[cls],
                ecolor=CLASS_COLOR[cls],
                capsize=5,
                markersize=7,
                zorder=3,
            )
    winner = results["paired_vs_v3"]["no_clutter"]
    dax.text(
        2,
        winner["mean_delta"] * 1e6 - 0.35,
        f"{winner['n_closer']}/100 closer",
        color=TEXT,
        fontsize=8.5,
        ha="center",
        va="top",
    )
    dax.axhline(0.0, color=GRID, lw=1.0, zorder=2)
    dax.axvline(4.5, color=GRID, lw=0.8, ls=":")
    dax.text(1.5, 2.05, "no_<class> vs v3", color=META, fontsize=8, ha="center")
    dax.text(
        7.0,
        2.05,
        "only_<class> vs plate-only",
        color=META,
        fontsize=8,
        ha="center",
    )
    dax.set_xticks(
        list(range(4)) + list(range(5, 9)),
        [f"no\n{c}" for c in CLASSES] + [f"only\n{c}" for c in CLASSES],
        color=TEXT,
        fontsize=8.2,
    )
    dax.set_xlim(-0.7, 8.7)
    dax.set_ylabel("paired dknn5 (x1e-6), bootstrap CI95", color=TEXT, fontsize=9.5)
    dax.set_title(
        "Paired per-frame dknn5 (negative = closer to real)",
        color=TEXT,
        fontsize=10.5,
        pad=10,
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[c], label=c) for c in CLASSES
    ]
    dax.legend(
        handles=handles,
        loc="lower right",
        fontsize=8,
        frameon=False,
        labelcolor=TEXT,
        ncols=2,
    )

    fig.suptitle(
        "Foreground content split: the clutter stand-ins (~5% of pixels) "
        "carry the removable share of the top-cam gap",
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
