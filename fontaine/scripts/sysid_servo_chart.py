"""Charts for the servo-sysid results post (sim-servo-sysid).

Two-panel figure, house dark style (eval-report scheme):
  left  - per-joint replay MAE on the held-out episodes, grouped bars:
          vendored menagerie gains vs TheRobotStudio upstream vs the
          fitted (pinned) params; gripper bars dimmed (record-only,
          contact-coupled in real).
  right - open-loop replay overlay on held-out v2 episode 4,
          shoulder_lift: commanded target + recorded rig trajectory vs
          the menagerie-gains sim and the fitted sim.

Reads outputs/sim/sysid_servo.json (banked by sim.sysid_servo) and
regenerates the two overlay trajectories with a deterministic local
replay (no live hosts). Writes fontaine/blog/src/img/sim/sysid_servo.png.

Usage: MUJOCO_GL=egl uv run python fontaine/scripts/sysid_servo_chart.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sim.sysid_servo import (
    OUTPUT_DIR,
    load_episode,
    replay_trajectory,
    set_params,
)

OUT = REPO / "fontaine" / "blog" / "src" / "img" / "sim" / "sysid_servo.png"

PAGE = "#121417"
BLUE = "#648fff"  # vendored menagerie gains
GOLD = "#ffb000"  # TheRobotStudio upstream gains
MAGENTA = "#dc267f"  # fitted (pinned) params
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"

OVERLAY_EPISODE = ("so101_pick_place_v2", 4)  # held-out
OVERLAY_JOINT = 1  # shoulder_lift
JOINT_LABELS = (
    "shoulder\npan",
    "shoulder\nlift",
    "elbow\nflex",
    "wrist\nflex",
    "wrist\nroll",
    "gripper\n(record-only)",
)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PAGE)
    ax.tick_params(colors=META, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main() -> int:
    report = json.loads((OUTPUT_DIR / "sysid_servo.json").read_text())
    candidates = report["candidates"]
    bars = (
        ("menagerie (kp 998)", "menagerie", BLUE),
        ("upstream (kp 17.8)", "upstream", GOLD),
        ("fitted (kp 108, pinned)", "pinned_so101_sim", MAGENTA),
    )

    fig, (ax_bar, ax_traj) = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.8),
        width_ratios=[1, 1.15],
    )
    fig.patch.set_facecolor(PAGE)

    # Left: per-joint held-out MAE, grouped bars.
    joints = list(candidates["menagerie"]["val"]["per_joint_mae_deg"].keys())
    x = np.arange(len(joints))
    width = 0.26
    for offset, (label, key, color) in zip((-width, 0.0, width), bars, strict=True):
        values = [candidates[key]["val"]["per_joint_mae_deg"][j] for j in joints]
        alphas = [1.0] * 5 + [0.45]  # gripper record-only
        bar = ax_bar.bar(
            x + offset,
            values,
            width - 0.03,
            color=color,
            zorder=3,
            label=label,
        )
        for patch, alpha in zip(bar.patches, alphas, strict=True):
            patch.set_alpha(alpha)
    for i, joint in enumerate(joints[:5]):
        fitted_v = candidates["pinned_so101_sim"]["val"]["per_joint_mae_deg"][joint]
        ax_bar.annotate(
            f"{fitted_v:.1f}",
            (i + width, fitted_v),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            color=TEXT,
            fontsize=8,
        )
    men = candidates["menagerie"]["val"]["arm_mae_deg"]
    fit_v = candidates["pinned_so101_sim"]["val"]["arm_mae_deg"]
    ax_bar.set_xticks(x, JOINT_LABELS, color=META, fontsize=8)
    ax_bar.set_ylabel("replay MAE, held-out episodes (deg)", color=META, fontsize=9)
    ax_bar.set_title(
        f"per-joint open-loop replay error - arm mean {men:.2f}° → {fit_v:.2f}°",
        color=TEXT,
        fontsize=10,
        pad=10,
    )
    ax_bar.legend(
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=8,
        loc="upper right",
    )
    style_axis(ax_bar)

    # Right: replay overlay on a held-out episode, one joint.
    from sim.so101_sim import SO101Sim

    sim = SO101Sim(width=64, height=48)
    actions, states = load_episode(*OVERLAY_EPISODE)
    trajectories = {}
    for _, key, _ in (bars[0], bars[2]):
        set_params(sim, candidates[key]["params"])
        trajectories[key] = replay_trajectory(sim, actions, states)
    t = np.arange(1, len(states)) / 30.0
    j = OVERLAY_JOINT
    ax_traj.plot(
        t,
        actions[:-1, j],
        color=META,
        linewidth=1.0,
        linestyle=":",
        label="commanded target",
        zorder=2,
    )
    ax_traj.plot(
        t,
        states[1:, j],
        color=TEXT,
        linewidth=2.2,
        label="recorded rig",
        zorder=4,
    )
    ax_traj.plot(
        t,
        trajectories["menagerie"][:, j],
        color=BLUE,
        linewidth=1.6,
        linestyle="--",
        label="sim, menagerie gains",
        zorder=3,
    )
    ax_traj.plot(
        t,
        trajectories["pinned_so101_sim"][:, j],
        color=MAGENTA,
        linewidth=1.6,
        label="sim, fitted params",
        zorder=5,
    )
    ax_traj.set_xlabel("episode time (s)", color=META, fontsize=9)
    ax_traj.set_ylabel("shoulder_lift (deg)", color=META, fontsize=9)
    ax_traj.set_title(
        "open-loop replay - held-out v2 episode 4, shoulder_lift",
        color=TEXT,
        fontsize=10,
        pad=10,
    )
    ax_traj.legend(
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=8,
        loc="lower right",
    )
    style_axis(ax_traj)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
