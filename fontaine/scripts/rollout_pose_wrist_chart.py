"""Results chart for the rollout-pose wrist read (queue
`sim-rollout-pose-wrist-read`; eval-report dark scheme — the shipped
Carbon pair). Reads the frozen analysis JSON only. Two panels: the
wrist knn5 AUROC ladder across pose regimes (reset band vs
manipulation poses, banked rollout anchor, real-real calibration
floor); and the paired material-stack reads against the zero rule —
reset top (the banked rider replication), reset wrist (neutral),
manipulation wrist (the regression).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
FIX, REFUTED, ANCHOR = "#08bdba", "#ffb000", "#9aa0a8"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/rollout_pose_wrist/chart__rollout_pose_wrist.png"),
    )
    args = ap.parse_args()

    results = json.loads(
        (args.reports / "analysis__sim_rollout_pose_wrist_read.json").read_text(),
    )["results"]
    reset = results["reset_anchor"]
    manip = results["manip"]
    calibration = results["abort_gates"]["calibration_auroc"]

    fig, (ax, dax) = plt.subplots(1, 2, figsize=(13.6, 5.0), width_ratios=[1.15, 1])
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, dax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)
        panel.set_axisbelow(True)

    # left: wrist AUROC by pose regime
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6)
    rows = [
        ("real-real calibration (eps 20-25, drift floor)", calibration, ANCHOR),
        (
            "RESET pose, v3 + fitted lens",
            reset["wrist"]["arms"]["fit"]["auroc_vs_real"],
            FIX,
        ),
        (
            "RESET pose, + material stack",
            reset["wrist"]["arms"]["fit_stack"]["auroc_vs_real"],
            FIX,
        ),
        (
            "MANIPULATION pose, v3 + fitted lens",
            manip["arms"]["fit"]["auroc_vs_real"],
            REFUTED,
        ),
        (
            "MANIPULATION pose, + material stack",
            manip["arms"]["fit_stack"]["auroc_vs_real"],
            REFUTED,
        ),
    ]
    for y, (label, auroc, color) in enumerate(reversed(rows)):
        ax.barh(y, auroc, height=0.5, color=color, alpha=0.85)
        ax.text(auroc + 0.008, y, f"{auroc:.3f}", color=TEXT, fontsize=9, va="center")
        ax.text(
            0.02,
            y + 0.38,
            label,
            color=TEXT,
            fontsize=8.5,
            ha="left",
            va="center",
        )
    banked = results["context"]["rollout_wrist_banked_auroc_old_visuals"]
    ax.axvline(banked, color=META, linewidth=0.8, linestyle="--", alpha=0.8)
    ax.text(
        banked - 0.01,
        2.5,
        "banked rollout-frame 0.828\n(old visuals, sim poses)",
        color=META,
        fontsize=8,
        ha="right",
        va="center",
    )
    ax.axvline(0.5, color=META, linewidth=0.8, alpha=0.5)
    ax.text(0.5, -0.62, "chance", color=META, fontsize=8, ha="center")
    ax.set_xlim(0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("er_60k knn5 AUROC vs held-out real (higher = more fake)", color=META)
    ax.set_title(
        "Wrist camera honesty by pose regime — the gap lives at manipulation poses",
        color=TEXT,
        fontsize=11,
        loc="left",
    )

    # right: paired material-stack deltas vs the zero rule
    dax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.6)
    reads = [
        (
            "reset TOP\n(banked rider\nreplication)",
            reset["top"]["paired_stack_vs_default"],
            FIX,
        ),
        (
            "reset WRIST\n(neutral,\nreplicates 08-14)",
            reset["wrist"]["paired_stack_vs_default"],
            ANCHOR,
        ),
        (
            "MANIP wrist\n(regression:\nCI > 0)",
            manip["paired_stack_vs_default"],
            REFUTED,
        ),
    ]
    for x, (_label, read, color) in enumerate(reads):
        lo, hi = (v * 1e7 for v in read["ci95"])
        mean = read["mean_delta"] * 1e7
        dax.vlines(x, lo, hi, color=color, linewidth=2.5)
        for cap in (lo, hi):
            dax.hlines(cap, x - 0.06, x + 0.06, color=color, linewidth=2.5)
        dax.plot(
            [x],
            [mean],
            marker="o",
            markersize=8,
            color=color,
            markeredgecolor=PAGE,
            markeredgewidth=2,
        )
        dax.text(
            x + 0.12,
            mean,
            f"{read['mean_delta']:+.2e}\n{read['n_closer']}/100 closer",
            color=TEXT,
            fontsize=8.5,
            va="center",
        )
        dax.text(x, dax.get_ylim()[0], "", color=TEXT)
    dax.axhline(0.0, color=META, linewidth=0.9, alpha=0.8)
    dax.set_xticks(range(len(reads)))
    dax.set_xticklabels([r[0] for r in reads], color=TEXT, fontsize=8.5)
    dax.set_xlim(-0.5, len(reads) - 0.2)
    dax.set_ylabel("paired Δknn5, stack vs default (×1e-7)", color=META)
    dax.set_title(
        "Material stack, paired CI95 — helps top, breaks even at reset wrist,\n"
        "REGRESSES the wrist where the arm fills the frame",
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
