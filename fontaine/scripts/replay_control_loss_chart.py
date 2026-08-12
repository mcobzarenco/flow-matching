"""Chart for the replay control-loss results post (queue item
sim-sysid-replay-control-loss, closed 13:5xZ 08-12).

Two panels from outputs/sim/replay_control_loss.json, eval-report dark
theme (page #121417): (left) SIMPLER control loss per servo-param
candidate, translation + rotation stacked (blue/amber, 2px surface
gaps), the real-command floor as a dashed reference line and SIMPLER's
best Table II anchor for scale; (right) per-joint EE-space error
contribution (joint MAE x lever arm at the median pose) for the pinned
fit vs the floor — the panel that explains why the joint-MAE win does
not carry to EE space. Identity via direct labels + position, never
color alone.

Usage: uv run python fontaine/scripts/replay_control_loss_chart.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "outputs/sim/replay_control_loss.json"
OUT = ROOT / "fontaine/blog/src/img/replay_control_loss.png"

PAGE = "#121417"
BLUE = "#648fff"  # translation term / pinned candidate
AMBER = "#ffb000"  # rotation term
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"


def main() -> int:
    report = json.loads(REPORT.read_text())
    floor = report["real_command_floor"]["pooled_all"]
    candidates = [
        ("menagerie\n(kp 998)", report["candidates"]["menagerie"]["pooled_all"]),
        ("upstream\n(kp 17.8)", report["candidates"]["upstream"]["pooled_all"]),
        (
            "pinned fit\n(kp 108)",
            report["candidates"]["pinned_so101_sim"]["pooled_all"],
        ),
    ]
    sensitivity = report["ee_sensitivity_mm_per_deg"]
    pinned_joints = report["candidates"]["pinned_so101_sim"]["pooled_all"][
        "per_joint_mae_deg"
    ]
    floor_joints = floor["per_joint_mae_deg"]
    arm = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=PAGE)
    for ax in (ax1, ax2):
        ax.set_facecolor(PAGE)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=META, labelsize=9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)

    # Left: stacked control loss per candidate, floor + anchor lines.
    xs = np.arange(len(candidates))
    trans = [c["trans_m"] for _, c in candidates]
    rot = [c["rot_rad"] for _, c in candidates]
    ax1.bar(xs, trans, 0.55, color=BLUE, edgecolor=PAGE, linewidth=2)
    ax1.bar(
        xs,
        rot,
        0.55,
        bottom=trans,
        color=AMBER,
        edgecolor=PAGE,
        linewidth=2,
    )
    for x, (_label, c) in zip(xs, candidates, strict=True):
        ax1.text(
            x,
            c["control_loss"] + 0.004,
            f"{c['control_loss']:.3f}",
            ha="center",
            color=TEXT,
            fontsize=10,
        )
    ax1.axhline(
        floor["control_loss"],
        color=META,
        linestyle="--",
        linewidth=1.2,
    )
    ax1.text(
        2.42,
        floor["control_loss"] - 0.0035,
        f"real-command floor {floor['control_loss']:.3f}",
        ha="right",
        color=META,
        fontsize=8.5,
    )
    ax1.axhline(0.131, color=META, linestyle=":", linewidth=1.0)
    ax1.text(
        2.42,
        0.1315,
        "SIMPLER best anchor 0.131 (scale caveat)",
        ha="right",
        color=META,
        fontsize=8.5,
    )
    ax1.set_xticks(xs, [label for label, _ in candidates], color=TEXT, fontsize=9.5)
    ax1.set_ylabel("control loss  L = ‖Δx‖ (m) + Δθ/2 (rad)", color=TEXT, fontsize=9.5)
    ax1.set_ylim(0, 0.14)
    ax1.set_title(
        "SIMPLER replay control loss — 26 reference episodes",
        color=TEXT,
        fontsize=11,
        pad=10,
    )
    ax1.legend(
        ["floor", "anchor", "translation", "rotation"],
        loc="upper left",
        frameon=False,
        labelcolor=TEXT,
        fontsize=8.5,
    )

    # Right: per-joint EE contribution, pinned vs floor.
    ys = np.arange(len(arm))
    pinned_mm = [pinned_joints[j] * sensitivity[j] for j in arm]
    floor_mm = [floor_joints[j] * sensitivity[j] for j in arm]
    ax2.barh(
        ys + 0.19,
        pinned_mm,
        0.34,
        color=BLUE,
        edgecolor=PAGE,
        linewidth=2,
        label="pinned fit",
    )
    ax2.barh(
        ys - 0.19,
        floor_mm,
        0.34,
        color=PAGE,
        edgecolor=META,
        linewidth=1.2,
        label="real-command floor",
    )
    for y, v in zip(ys + 0.19, pinned_mm, strict=True):
        ax2.text(v + 0.25, y, f"{v:.1f}", va="center", color=TEXT, fontsize=9)
    for y, v in zip(ys - 0.19, floor_mm, strict=True):
        ax2.text(v + 0.25, y, f"{v:.1f}", va="center", color=META, fontsize=9)
    ax2.set_yticks(
        ys,
        [f"{j}\n{sensitivity[j]:.1f} mm/°" for j in arm],
        color=TEXT,
        fontsize=9,
    )
    ax2.invert_yaxis()
    ax2.xaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax2.yaxis.grid(False)
    ax2.set_xlabel(
        "EE error contribution  (joint MAE x lever arm, mm)",
        color=TEXT,
        fontsize=9.5,
    )
    ax2.set_title(
        "the elbow IS the EE loss (pinned fit vs floor)",
        color=TEXT,
        fontsize=11,
        pad=10,
    )
    ax2.legend(loc="lower right", frameon=False, labelcolor=TEXT, fontsize=8.5)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
