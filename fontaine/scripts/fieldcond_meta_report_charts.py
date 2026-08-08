"""Charts for the field-conditioning + subgoal meta-report (queue
item `fieldcond-subgoal-meta-report`) — the sections whose inputs are
already banked (§1 aux attribution, §2 selfsubgoal horizon reads).
Every value is read from the frozen analysis jsons; nothing is
recomputed. §3's fields grouped bar waits on the 60k fields panel and
is NOT rendered here.

Output: fontaine/blog/src/img/fieldcond/*.svg (committed).

Palette: the eval reports' DARK theme (page #121417; standing owner
rule 2026-08-08 16:32Z) with the IBM colorblind-safe categorical
steps for the 4-series horizon chart (#648fff blue, #ffb000 amber,
#dc267f magenta + neutral gray reference). No node on the box, so the
palette validator can't run; the IBM set unchanged is the sanctioned
fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fontaine/blog/src/img/fieldcond"
SELFSUBGOAL = ROOT / "reports/analysis__selfsubgoal_ar100k_k4l2.json"
BOXBATCH = ROOT / "reports/analysis__box_batch_40k_k4l2.json"

PAGE = "#121417"
BLUE = "#648fff"  # oracle-truth subgoal arm
AMBER = "#ffb000"  # self-generated arm
MAGENTA = "#dc267f"  # narrated arm
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"


def style_axes(ax: Axes) -> None:
    ax.set_facecolor(PAGE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=META, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def aux_arms_chart(d: dict) -> None:
    pooled = d["pooled"]
    primary = d["primary_B_minus_As0"]
    names = ["A-s0", "A-s1", "A-s2", "B"]
    values = [pooled[n]["chunk_mae"] for n in names]
    labels = ["aux ON\nseed 0", "aux ON\nseed 1", "aux ON\nseed 2", "aux OFF"]
    colors = [BLUE, BLUE, BLUE, AMBER]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=110)
        fig.patch.set_facecolor(PAGE)
        style_axes(ax)
        x = np.arange(len(names))
        ax.bar(x, values, width=0.62, color=colors, linewidth=0)
        for xi, v in zip(x.tolist(), values, strict=True):
            ax.text(
                xi,
                v + 0.02,
                f"{v:.3f}",
                ha="center",
                color=TEXT,
                fontsize=9.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=META, fontsize=9.5)
        lo, hi = primary["ci95"]
        ax.set_ylim(7.4, 8.55)
        ax.annotate(
            f"aux-off costs +{primary['mean']:.3f}\n(CI95 [{lo:.3f}, {hi:.3f}])",
            xy=(3, values[3]),
            xytext=(1.9, 8.42),
            color=AMBER,
            fontsize=10,
            arrowprops={"arrowstyle": "->", "color": META, "linewidth": 1.0},
        )
        ax.set_ylabel("panel chunk MAE (40k box batch)", color=META, fontsize=10)
        ax.set_title(
            "The aux text head is load-bearing\nthree aux-on seeds vs aux-off, same data + steps",
            color=TEXT,
            fontsize=11,
            loc="left",
            pad=12,
        )
        fig.tight_layout()
        fig.savefig(OUT / "aux_arms.svg", facecolor=PAGE)
        plt.close(fig)


def horizon_curves_chart(d: dict) -> None:
    steps = np.arange(1, 51)
    baseline = np.asarray(d["baseline_curve"])
    series = [
        ("baseline (no subgoal)", baseline, META),
        ("oracle-truth subgoal", np.asarray(d["arms"]["oracle"]["curve"]), BLUE),
        ("self-generated subgoal", np.asarray(d["arms"]["self"]["curve"]), AMBER),
        ("narrated (suffix channel)", np.asarray(d["arms"]["narr"]["curve"]), MAGENTA),
    ]

    with plt.style.context("dark_background"):
        fig, (ax, axd) = plt.subplots(
            2,
            1,
            figsize=(9.2, 6.2),
            dpi=110,
            sharex=True,
            height_ratios=[2.0, 1.2],
        )
        fig.patch.set_facecolor(PAGE)
        for a in (ax, axd):
            style_axes(a)
        for label, curve, color in series:
            ax.plot(steps, curve, color=color, linewidth=2.0, label=label)
        ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT)
        ax.set_ylabel("MAE at horizon step", color=META, fontsize=10)
        ax.set_title(
            "What the subgoal slot buys, by horizon\nAR-100k panel, 17,204 core frames — the oracle gap opens late",
            color=TEXT,
            fontsize=11,
            loc="left",
            pad=12,
        )

        for label, curve, color in series[1:]:
            axd.plot(
                steps,
                curve - baseline,
                color=color,
                linewidth=1.8,
                label=label,
            )
        axd.axhline(0, color=META, linewidth=1.0)
        oracle = d["arms"]["oracle"]
        axd.text(
            2,
            oracle["core"]["delta_pooled"] - 0.17,
            f"oracle pooled Δ {oracle['core']['delta_pooled']:+.3f}"
            f"  (last-10 {oracle['horizon_delta']['last10']:+.3f}"
            f" vs first-10 {oracle['horizon_delta']['first10']:+.3f})",
            color=BLUE,
            fontsize=9.5,
        )
        axd.set_ylabel("arm − baseline", color=META, fontsize=10)
        axd.set_xlabel("horizon step within the action chunk", color=META, fontsize=10)
        axd.set_xlim(1, 50)
        fig.tight_layout()
        fig.savefig(OUT / "selfsubgoal_horizon.svg", facecolor=PAGE)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    aux_arms_chart(json.loads(BOXBATCH.read_text()))
    horizon_curves_chart(json.loads(SELFSUBGOAL.read_text()))
    print(f"wrote {OUT}/aux_arms.svg + selfsubgoal_horizon.svg")


if __name__ == "__main__":
    main()
