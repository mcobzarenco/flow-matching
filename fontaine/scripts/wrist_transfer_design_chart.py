"""Schematic chart for the wrist-transfer screen design memo (queue
`wrist-transfer-screen-design`; eval-report dark scheme). This is a
DESIGN figure, not a results chart: the left panel carries only the
two banked honesty anchors (0.523 reset / 0.877 manipulation, from
analysis__sim_rollout_pose_wrist_read.json) on a real axis — treatment
arm x-positions are drawn as an ordered ladder in the measurable
region and labeled as illustrative until the stage-0 placement read;
the two outcome sketches are the memo's F-flat and F-live verdicts,
with no y units. The right panel is the planned stage ladder (real
planned GPU-h) against its <=14 gate, stage 3 conditional.

Palette is the validated eval-report triple (#08bdba/#ffb000/#6f747c
on #121417) — CVD-checked 2026-08-14, see renderer_decision_chart.py;
no new colors introduced (no local node runtime to re-run the
validator this session).
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
LIVE, DEGRADE, ANCHOR = "#08bdba", "#ffb000", "#6f747c"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/sim/wrist_transfer_design/chart__wrist_transfer_screen_design.png",
        ),
    )
    args = ap.parse_args()

    pose = json.loads(
        (args.reports / "analysis__sim_rollout_pose_wrist_read.json").read_text(),
    )["results"]
    w_reset = pose["abort_gates"]["wrist_auroc"]
    w_manip = pose["primary1"]["manip_wrist_auroc"]

    fig, (ax, bax) = plt.subplots(1, 2, figsize=(14.6, 5.2), width_ratios=[1.35, 1])
    fig.patch.set_facecolor(PAGE)
    fig.subplots_adjust(wspace=0.42)
    for panel in (ax, bax):
        panel.set_facecolor(PAGE)
        for side in panel.spines.values():
            side.set_color(GRID)
        panel.tick_params(colors=META, labelsize=9)

    # ---- left: the screen's logic on the banked honesty axis ----
    ax.set_xlim(0.44, 1.02)
    ax.set_ylim(-0.15, 1.05)
    ax.set_yticks([])
    ax.set_xlabel(
        "wrist honesty — knn5 AUROC vs held-out real, manipulation reference"
        " (lower = reads more real)",
        color=META,
        fontsize=9,
    )
    ax.set_ylabel(
        "Δ closed-loop behavior (unmeasured — schematic)",
        color=META,
        fontsize=9,
    )

    # payload span (extrapolation target) vs measurable degradation side
    ax.axvspan(w_reset, w_manip, color=LIVE, alpha=0.10, lw=0)
    ax.axvspan(w_manip, 1.02, color=DEGRADE, alpha=0.08, lw=0)
    for x, name in (
        (w_reset, f"reset anchor {w_reset:.3f}"),
        (w_manip, f"W0 baseline {w_manip:.3f}"),
    ):
        ax.axvline(x, color=ANCHOR, lw=1.4, ls=(0, (4, 3)))
        ax.text(x, 1.02, name, color=TEXT, fontsize=8.5, ha="center", va="bottom")
    ax.text(
        (w_reset + w_manip) / 2,
        0.88,
        "renderer payload span\n(slope applied here — extrapolation)",
        color=LIVE,
        fontsize=8.5,
        ha="center",
        va="center",
    )
    ax.text(
        (w_manip + 1.02) / 2,
        0.88,
        "measurable side\n(degradation ladder)",
        color=DEGRADE,
        fontsize=8.5,
        ha="center",
        va="center",
    )

    # ordered arm ladder on the measurable side — x illustrative until stage 0
    arm_x = {"W4": 0.895, "W3": 0.935, "W2": 0.965, "W1": 0.995}
    for name, x in arm_x.items():
        ax.plot([x], [0.02], marker="o", ms=8, mfc=DEGRADE, mec=PAGE, mew=2, zorder=5)
        ax.text(x, -0.055, name, color=TEXT, fontsize=9, ha="center")
    ax.plot([w_manip], [0.02], marker="o", ms=8, mfc=LIVE, mec=PAGE, mew=2, zorder=5)
    ax.text(w_manip - 0.008, -0.055, "W0", color=TEXT, fontsize=9, ha="center")
    ax.text(
        0.73,
        -0.125,
        "arm order fixed, x-positions illustrative until the stage-0 placement read",
        color=META,
        fontsize=7.5,
        ha="center",
    )

    # outcome sketches: F-flat vs F-live (and its dashed extrapolation)
    xs = np.linspace(w_manip, 1.0, 60)
    ax.plot(xs, np.full_like(xs, 0.30), color=META, lw=2)
    ax.text(
        0.998,
        0.255,
        "F-flat / F-null:\ndishonesty tolerated",
        color=META,
        fontsize=8.5,
        ha="right",
        va="top",
    )
    slope = 3.2
    ax.plot(xs, 0.28 + slope * (xs - w_manip) ** 1.3, color=LIVE, lw=2)
    xe = np.linspace(w_reset + 0.04, w_manip, 40)
    ax.plot(
        xe,
        0.28 - slope * (w_manip - xe) ** 1.3,
        color=LIVE,
        lw=1.6,
        ls=(0, (4, 3)),
        alpha=0.8,
    )
    ax.text(
        0.60,
        0.52,
        "F-live: link is real —\nslope prices the tier-2 payload",
        color=LIVE,
        fontsize=8.5,
        ha="center",
    )
    ax.set_title(
        "The screen's logic: behavior change per unit of wrist honesty",
        color=TEXT,
        fontsize=11,
        pad=24,
    )

    # ---- right: staged budget vs the gate ----
    stages = [
        ("0  hook + oracles + placement", 0.1, False),
        ("1  P1 × {W0,W1,W3} + T1", 3.3, False),
        ("2  simft + P2 × {W0,W1,W3}", 4.8, False),
        ("3  W2/W4 ladder (conditional)", 3.8, True),
    ]
    y = np.arange(len(stages))[::-1]
    left = 0.0
    for (_label, hours, cond), yi in zip(stages, y, strict=True):
        bax.barh(
            yi,
            hours,
            left=left,
            height=0.52,
            color=DEGRADE if cond else LIVE,
            edgecolor=PAGE,
            linewidth=2,
            hatch="//" if cond else None,
            alpha=0.85 if cond else 1.0,
        )
        bax.text(
            left + hours + 0.2,
            yi,
            f"{hours:.1f}",
            color=TEXT,
            fontsize=8.5,
            ha="left",
            va="center",
        )
        left += hours
    bax.set_yticks(y)
    bax.set_yticklabels([s[0] for s in stages], color=TEXT, fontsize=8.5)
    bax.axvline(14.0, color=ANCHOR, lw=1.6, ls=(0, (4, 3)))
    bax.text(
        13.8,
        0.55,
        "gate ≤ 14 GPU-h",
        color=TEXT,
        fontsize=8.5,
        ha="right",
        va="bottom",
        rotation=90,
    )
    bax.axvline(left, color=META, lw=1.0)
    bax.text(
        left - 0.2,
        3.42,
        f"worst-case {left:.1f}",
        color=META,
        fontsize=8.5,
        ha="right",
    )
    bax.set_xlim(0, 15.2)
    bax.set_ylim(-0.6, 3.7)
    bax.set_xlabel(
        "cumulative GPU-h (planned; bars start at the prior stage's end)",
        color=META,
        fontsize=9,
    )
    bax.xaxis.grid(True, color=GRID, lw=0.6)
    bax.set_axisbelow(True)
    bax.set_title(
        "Stage ladder vs its gate — boundaries are hard stops",
        color=TEXT,
        fontsize=11,
        pad=10,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
