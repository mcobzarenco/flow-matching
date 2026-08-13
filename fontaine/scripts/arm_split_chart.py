"""Results chart for the arm sub-part split (queue
`sim-arm-appearance-leg`; eval-report dark scheme — the four base
class hues are the fg-split CVD-validated set, plus Carbon purple-40
for the fifth class; every class is direct-labeled so identity is
never color-alone). Two panels: 5-NN AUROC per arm against the leg-(a)
anchors, and the paired per-frame dknn5 with bootstrap CI95 whiskers —
links carrying 88% of the arm's keep-only delta (on 6.1% of pixels)
while mount removal is the only one that moves v3 TOWARD real is the
story.
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
    "gripper": "#4589ff",
    "links": "#ffb000",
    "mount": "#fa4d56",
    "follower": "#08bdba",
    "leader": "#be95ff",
}
PARTS = ("gripper", "links", "mount")
INSTANCES = ("follower", "leader")
CLASSES = PARTS + INSTANCES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis",
        type=Path,
        default=Path("reports/analysis__sim_arm_split.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/arm_split/chart__sim_arm_split.png"),
    )
    args = ap.parse_args()
    payload = json.loads(args.analysis.read_text())
    results = payload["results"]
    arms = results["arms"]
    anchors = payload["config"]["anchors"]
    fractions = results["subclass_pixel_fraction_mean"]

    fig, (ax, dax) = plt.subplots(1, 2, figsize=(13.6, 5.0), width_ratios=[1.45, 1])
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, dax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)

    order = (
        [
            ("v3\n(baseline)", "v3", META),
            ("plate\nonly", "plate_only", META),
            ("only\narm", "only_arm", META),
            ("no\narm", "no_arm", META),
        ]
        + [(f"no\n{c}", f"no_{c}", CLASS_COLOR[c]) for c in CLASSES]
        + [(f"only\n{c}", f"only_{c}", CLASS_COLOR[c]) for c in CLASSES]
    )
    x = np.concatenate([[0, 1, 2, 3], 4.6 + np.arange(5), 10.2 + np.arange(5)])
    values = [arms[key]["auroc_vs_real"] for _, key, _ in order]
    ax.bar(x, values, 0.62, color=[c for _, _, c in order], zorder=3)
    ax.axhline(0.5, color=GRID, lw=1.0, zorder=2)
    ax.text(4.0, 0.487, "0.5 null", color=META, fontsize=7.5, ha="center", va="top")
    real_fg = anchors["decomposition_real_fg"]
    ax.axhline(real_fg, color=META, lw=1.0, ls="--", zorder=2)
    ax.text(
        12.0,
        real_fg + 0.007,
        f"real-fg {real_fg:.3f}",
        color=META,
        fontsize=7.5,
        ha="center",
        va="bottom",
    )
    for xi, val in zip(x, values, strict=True):
        ax.text(xi, val + 0.008, f"{val:.3f}", color=TEXT, fontsize=7.8, ha="center")
    v3 = arms["v3"]["auroc_vs_real"]
    for xi, (_, key, _) in zip(x, order, strict=True):
        if key.startswith("no_"):
            delta = arms[key]["auroc_vs_real"] - v3
            ax.text(
                xi,
                0.03,
                f"{delta:+.3f}",
                color=TEXT,
                fontsize=7.6,
                ha="center",
                va="bottom",
            )
    ax.annotate(
        "named target\n(registered rule, 88%)",
        xy=(11.2, values[10] + 0.045),
        xytext=(11.2, 0.99),
        color=CLASS_COLOR["links"],
        fontsize=8.5,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": CLASS_COLOR["links"], "lw": 0.8},
    )
    ax.set_xticks(x, [label for label, _, _ in order], color=TEXT, fontsize=7.8)
    for divider in (3.8, 9.4):
        ax.axvline(divider, color=GRID, lw=0.8, ls=":")
    ax.text(1.5, 1.06, "leg-(a) bridges", color=META, fontsize=8, ha="center")
    ax.text(
        6.6,
        1.06,
        "sub-class removed → rest stays (Δ vs v3 below)",
        color=META,
        fontsize=8,
        ha="center",
    )
    ax.text(
        12.2,
        1.06,
        "sub-class alone on the plate",
        color=META,
        fontsize=8,
        ha="center",
    )
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
            (6.0, results["paired_vs_plate_only"][f"only_{cls}"], "s"),
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
    arm_delta = results["only_arm_vs_plate_only"]["mean_delta"] * 1e6
    dax.axhline(arm_delta, color=META, lw=1.0, ls="--", zorder=2)
    dax.text(
        8.4,
        arm_delta - 0.12,
        f"whole arm class {arm_delta:+.2f}",
        color=META,
        fontsize=7.5,
        ha="center",
        va="top",
    )
    dax.axhline(0.0, color=GRID, lw=1.0, zorder=2)
    dax.axvline(5.0, color=GRID, lw=0.8, ls=":")
    labels = [f"no\n{c}\n{fractions[c]:.1%} px" for c in CLASSES] + [
        f"only\n{c}\n{fractions[c]:.1%} px" for c in CLASSES
    ]
    dax.set_xticks(
        list(range(5)) + list(range(6, 11)),
        labels,
        color=TEXT,
        fontsize=7.8,
    )
    # share labels under the only_<part> markers, pixel fractions on top
    for xi, cls in enumerate(CLASSES):
        if cls in PARTS:
            share = results["decision"]["per_part"][cls]["share_of_only_arm_delta"]
            block = results["paired_vs_plate_only"][f"only_{cls}"]
            dax.text(
                xi + 6.0,
                block["mean_delta"] * 1e6 - 0.35,
                f"{share:.0%}",
                color=CLASS_COLOR[cls],
                fontsize=8.4,
                ha="center",
                va="top",
            )
    dax.set_ylim(-6.4, 3.6)
    dax.set_ylabel("paired Δknn5 ×1e-6 (CI95, 10k bootstrap)", color=TEXT, fontsize=9.5)
    dax.set_title(
        "paired per-frame deltas: ○ no_X vs v3 · □ only_X vs plate-only",
        color=TEXT,
        fontsize=10.5,
        pad=10,
    )
    dax.text(
        2.0,
        -5.9,
        "below 0 = closer to real",
        color=META,
        fontsize=7.8,
        ha="center",
    )

    fig.suptitle(
        "Arm sub-part split — which rendered arm pixels carry the sim signature "
        "(er_60k trunk, top cam)",
        color=TEXT,
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=PAGE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
