"""Stack-parity confirmation chart for the discriminator verdict post.

One panel, one axis, one instrument: the PRE-MERGE eval surface
(9094e60, MolmoNorm.CHECKPOINT units) — the surface every 8x
comparator's in-train probe reported in. Series:

- discriminator saves re-scored by the parity probe (reports/
  stack_parity/step_000500.json, step_001000.json) — the verdict
  evidence;
- the demosonly 8x comparator's frozen in-train anchors (same units,
  from the pre-reg) — the drift signature;
- the discriminator's own in-train probe curve (post-merge stack) —
  drawn thin/dashed to show the two instruments nearly coincide at
  the shared saves (x1.034 @500, x1.024 @1000).

Palette/style: the blog's established dark eval-report scheme
(sft_drift_saga_charts.py) — IBM CVD-safe hues, identity carried by
direct labels + line styles, single axis, recessive grid.

Usage: uv run python fontaine/scripts/stack_parity_chart.py
       -> fontaine/blog/src/img/grasp_sft_drift/stack_parity.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PARITY = ROOT / "reports/stack_parity"
IMG_OUT = ROOT / "fontaine/blog/src/img/grasp_sft_drift"

PAGE = "#121417"
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"
MAGENTA = (
    "#dc267f"  # the drifting comparator (drifting emphasis, as on every saga page)
)
BLUE = "#648fff"  # secondary series: our post-merge in-train probe
DISC = "#e8eaed"  # the discriminator run (near-white, as on disc_overlay)

# Frozen comparator anchors, old units (pre-reg + stack_parity_probe.sh header).
COMP_STEPS = [250, 500, 750, 1000]
COMP_EVAL = [3.4623, 3.2397, 4.22, 5.27]
# Discriminator in-train probe (post-merge stack), from the verdict read.
DISC_STEPS = [250, 500, 750, 1000]
DISC_EVAL = [12.5087, 7.5654, 6.5906, 5.8989]


def parity_points() -> tuple[list[int], list[float]]:
    steps, maes = [], []
    for name in ("step_000500", "step_001000"):
        d = json.loads((PARITY / f"{name}.json").read_text())
        (policy,) = [s for s in d["summaries"] if s["policy"].startswith("bijou@")]
        steps.append(int(name.split("_")[1]))
        maes.append(policy["chunk_mae"])
    return steps, maes


def main() -> None:
    p_steps, p_mae = parity_points()
    fig = plt.figure(figsize=(7.4, 4.4), facecolor=PAGE)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PAGE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.tick_params(colors=META, labelsize=8)

    ax.axvspan(500, 1060, color=MAGENTA, alpha=0.045)
    ax.axvline(500, color=GRID, linewidth=0.8, linestyle=":")

    ax.plot(
        DISC_STEPS,
        DISC_EVAL,
        color=BLUE,
        linewidth=1.2,
        linestyle="--",
        marker=".",
        markersize=5,
        alpha=0.8,
        label="discriminator, in-train probe (post-merge stack)",
    )
    ax.plot(
        COMP_STEPS,
        COMP_EVAL,
        color=MAGENTA,
        linewidth=2.0,
        marker="o",
        markersize=6,
        label="demosonly 8×, in-train probe (drifting comparator)",
    )
    ax.plot(
        p_steps,
        p_mae,
        color=DISC,
        linewidth=2.0,
        marker="D",
        markersize=8,
        label="discriminator saves, parity probe (pre-merge stack)",
    )

    ax.annotate(
        "Δ(1000−500) = −1.55\nsame instrument, still falling",
        xy=(1000, p_mae[1]),
        xytext=(660, 8.6),
        color=DISC,
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": META, "linewidth": 0.7},
    )
    ax.annotate(
        "Δ(1000−500) = +2.03\n(drift_min +1.02, healthy ≤ +0.30)",
        xy=(1000, 5.27),
        xytext=(630, 3.0),
        color=MAGENTA,
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": META, "linewidth": 0.7},
    )
    ax.annotate(
        "instruments nearly coincide at the shared saves\n(×1.034 @500, ×1.024 @1000)",
        xy=(500, 7.31),
        xytext=(255, 10.6),
        color=META,
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": META, "linewidth": 0.7},
    )

    ax.set_xlabel("step", color=META, fontsize=8)
    ax.set_ylabel("holdout chunk MAE (deg), pre-merge units", color=META, fontsize=8)
    ax.set_xlim(200, 1060)
    ax.set_title(
        "Stack-parity probe: the HEALTHY verdict holds on the comparator's own instrument",
        color=TEXT,
        fontsize=9.5,
        loc="left",
        pad=6,
    )
    ax.legend(
        loc="upper right",
        fontsize=7.5,
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
        framealpha=0.9,
    )
    fig.tight_layout()
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    out = IMG_OUT / "stack_parity.png"
    fig.savefig(out, dpi=160, facecolor=PAGE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
