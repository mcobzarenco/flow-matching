"""Results chart for the GRPO signal probe (pre-reg 2026-08-12):
per-cell per-seed within-group std of progress_final_cm (the primary
read) against the frozen 0.25 cm signal bar, plus the paired
competence-cost CIs against the decision rule's -1.0 cm boundary
(eval-report dark scheme, IBM CVD-safe hues).

Reads read_grpo_signal_probe.py --json output from stdin or --reads.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
AR, FLOW, SDE = "#648fff", "#ffb000", "#dc267f"
BAR_CM, COST_FLOOR = 0.25, -1.0

FAMILY = {
    "cell1_er60k_t10": AR,
    "cell2_er60k_t16": AR,
    "cell3_teacher80k_heun30": FLOW,
    "cell4_ftrig4k_euler1": FLOW,
    "cell5_teacher80k_sde05": SDE,
    "cell5b_teacher80k_sde03": SDE,
}
SHORT = {
    "cell1_er60k_t10": "1: AR t=1.0\ner60k",
    "cell2_er60k_t16": "2: AR t=1.6\ner60k",
    "cell3_teacher80k_heun30": "3: ODE heun-30\nteacher80k",
    "cell4_ftrig4k_euler1": "4: ODE euler-1\nftrig4k",
    "cell5_teacher80k_sde05": "5: SDE a=0.5\nteacher80k",
    "cell5b_teacher80k_sde03": "5b: SDE a=0.3\nteacher80k",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", type=Path, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sim/grpo_signal_probe/chart__grpo_signal_probe.png"),
    )
    args = ap.parse_args()
    if args.reads:
        blob = json.loads(args.reads.read_text())
    else:
        blob = json.loads(
            subprocess.run(
                [
                    sys.executable,
                    "fontaine/scripts/read_grpo_signal_probe.py",
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout,
        )
    cells = {
        name: r
        for name, r in blob["cells"].items()
        if not r.get("partial") and "median_std" in r
    }
    if not cells:
        raise SystemExit("no complete cells to chart")

    fig, (ax, cost_ax) = plt.subplots(
        2,
        1,
        figsize=(9.5, 6.2),
        height_ratios=[2.6, 1.3],
    )
    fig.patch.set_facecolor(PAGE)
    for panel in (ax, cost_ax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)
        panel.set_axisbelow(True)
        panel.yaxis.grid(True, color=GRID, linewidth=0.8)

    xs = list(range(len(cells)))
    rng = np.random.default_rng(0)
    for x, (name, r) in zip(xs, cells.items(), strict=True):
        color = FAMILY[name]
        stds = np.array(list(r["per_seed_std"].values()))
        jitter = rng.uniform(-0.16, 0.16, size=len(stds))
        ax.scatter(
            x + jitter,
            stds,
            s=26,
            color=color,
            zorder=3,
            edgecolors=PAGE,
            linewidths=0.8,
            alpha=0.9,
        )
        ax.hlines(
            r["median_std"],
            x - 0.28,
            x + 0.28,
            color=TEXT,
            linewidth=2.2,
            zorder=4,
        )
        ax.annotate(
            f"{r['median_std']:.2f}",
            (x + 0.32, r["median_std"]),
            color=TEXT,
            fontsize=9,
            va="center",
        )
    ax.axhline(BAR_CM, color=META, linewidth=1.2, linestyle="--")
    ax.annotate(
        "signal bar 0.25 cm",
        (xs[-1] + 0.45, BAR_CM),
        color=META,
        fontsize=8.5,
        va="center",
        ha="right",
    )
    ax.set_xlim(-0.6, len(cells) - 0.4 + 0.8)
    ax.set_yscale("log")
    ax.set_ylabel("per-seed group std of progress_final (cm)", color=TEXT, fontsize=9.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[n] for n in cells], color=TEXT, fontsize=8.5)
    ax.set_title(
        "GRPO signal probe — within-group spread per cell "
        "(15 seeds × K=8 draws, 30 s episodes, v3 frames)",
        color=TEXT,
        fontsize=11,
        pad=10,
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=AR, label="AR sampled"),
            Line2D([], [], marker="o", ls="", color=FLOW, label="flow ODE fresh noise"),
            Line2D([], [], marker="o", ls="", color=SDE, label="flow SDE"),
            Line2D([], [], color=TEXT, linewidth=2.2, label="cell median"),
        ],
        loc="lower right",
        frameon=False,
        labelcolor=META,
        fontsize=8.5,
    )

    for x, (name, r) in zip(xs, cells.items(), strict=True):
        if "competence_cost" not in r:
            continue
        c = r["competence_cost"]
        lo, hi = c["ci95"]
        cost_ax.errorbar(
            x,
            c["mean"],
            yerr=[[c["mean"] - lo], [hi - c["mean"]]],
            fmt="o",
            color=FAMILY[name],
            ecolor=FAMILY[name],
            capsize=4,
            markersize=7,
            markeredgecolor=PAGE,
            zorder=3,
        )
    cost_ax.axhline(0, color=META, linewidth=1, linestyle=":")
    cost_ax.axhline(COST_FLOOR, color=SDE, linewidth=1.2, linestyle="--", alpha=0.8)
    cost_ax.annotate(
        "decision-rule floor −1.0 cm",
        (xs[-1] + 0.45, COST_FLOOR),
        color=META,
        fontsize=8.5,
        va="bottom",
        ha="right",
    )
    cost_ax.set_ylabel("competence cost vs anchor (cm)", color=TEXT, fontsize=9.5)
    cost_ax.set_xticks(xs)
    cost_ax.set_xticklabels([SHORT[n] for n in cells], color=TEXT, fontsize=8.5)
    cost_ax.set_xlim(ax.get_xlim())

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
