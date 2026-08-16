"""Smoother-demos v1.1 charts (owner 16:53Z 2026-08-16: 'action
traces are very jumpy').

Three figures for the blog report, eval-report dark theme:
  1. smooth_step_sizes.png — per-tick max |Δcommand| across the 5 arm
     channels over one episode, v1 one-shot (blue) vs v1.1 slew
     (magenta), log y: the 100–300°/tick phase-boundary spikes vs the
     10°/tick ceiling.
  2. smooth_wrist_trace.png — the jumpiest channel's (wrist_roll)
     commanded trace, same seed, both configs overlaid.
  3. smooth_yield.png — kept% and parked%-of-placed for the five
     measured configs (n=120 each, notes/smooth_*.json).

Traces re-run live (seed 1005 places under both configs; physics-only,
no GL frames beyond the reset).

Usage: uv run python fontaine/scripts/smooth_traces_chart.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "fontaine/blog/src/img/smooth"

PAGE = "#121417"
BLUE = "#648fff"  # v1 one-shot expert (shipped)
MAGENTA = "#dc267f"  # v1.1 slew 10/12
TEXT, META, GRID = "#d8dade", "#9aa0a8", "#3a3f46"

TRACE_SEED = 1005


def capture_trace(arm_slew: float | None, jaw_slew: float | None) -> np.ndarray:
    import mujoco

    from sim.scripted_expert import ScriptedExpert
    from sim.so101_sim import PHYSICS_STEPS_PER_TICK, SO101Sim

    sim = SO101Sim(spawn_version="v2.1", tint_band="mix70")
    ScriptedExpert.SLEW_ARM_DEG = arm_slew
    ScriptedExpert.SLEW_JAW_DEG = jaw_slew
    from sim.so101_sim import HOME_DEGREES

    sim.reset(TRACE_SEED)
    expert = ScriptedExpert(sim)
    cmds = []

    def step() -> None:
        action = expert.action(sim)
        cmds.append(np.asarray(action, dtype=np.float64))
        target = np.clip(np.deg2rad(action), sim._ctrl_low, sim._ctrl_high)
        sim.data.ctrl[sim._actuator_ids] = target
        mujoco.mj_step(sim.model, sim.data, nstep=PHYSICS_STEPS_PER_TICK)

    for _ in range(600):
        step()
        if sim.success():
            break
    # ride the retreat tail like the collector: the home swing (the
    # jumpiest stretch of a v1 episode) lives here, not the main loop
    home_rad = np.deg2rad(HOME_DEGREES[:5])
    arm_qpos = sim._joint_qpos[:5]
    for _ in range(300):
        step()
        at_home = float(
            np.max(np.abs(sim.data.qpos[arm_qpos] - home_rad)),
        ) < np.deg2rad(10.0)
        if at_home and float(np.abs(sim.data.qvel).max()) < 0.05:
            break
    return np.stack(cmds)


def style_ax(ax: Axes) -> None:
    ax.set_facecolor(PAGE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=META, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.5)
    ax.xaxis.label.set_color(META)
    ax.yaxis.label.set_color(META)
    ax.title.set_color(TEXT)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = capture_trace(None, None)
    v11 = capture_trace(10.0, 12.0)

    # 1. per-tick max arm step, log scale
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=PAGE)
    style_ax(ax)
    for trace, color, label in (
        (base, BLUE, "v1 one-shot"),
        (v11, MAGENTA, "v1.1 slew 10°/tick"),
    ):
        step = np.abs(np.diff(trace[:, :5], axis=0)).max(axis=1)
        ax.plot(np.maximum(step, 1e-2), color=color, linewidth=1.6, label=label)
    ax.axhline(10.0, color=MAGENTA, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("tick (30 Hz)")
    ax.set_ylabel("max |Δcommand| across arm joints (°/tick, log)")
    ax.set_title(
        f"Commanded step size per tick — seed {TRACE_SEED} "
        "(phase-boundary jumps vs the rate-bounded ceiling)",
        fontsize=11,
    )
    ax.legend(facecolor=PAGE, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "smooth_step_sizes.png", dpi=140, facecolor=PAGE)
    plt.close(fig)

    # 2. the jumpiest channel's commanded trace
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    ch = int(np.abs(np.diff(base[:, :5], axis=0)).max(axis=0).argmax())
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=PAGE)
    style_ax(ax)
    ax.plot(base[:, ch], color=BLUE, linewidth=1.6, label="v1 one-shot")
    ax.plot(v11[:, ch], color=MAGENTA, linewidth=1.6, label="v1.1 slew 10°/tick")
    lo = max(0, min(len(base), len(v11)) - 80)
    hi = max(len(base), len(v11)) + 5
    ax.set_xlim(lo, hi)
    ax.set_xlabel("tick (30 Hz)")
    ax.set_ylabel(f"{names[ch]} commanded target (°)")
    ax.set_title(
        f"The jumpiest channel through the retreat — seed {TRACE_SEED} "
        "(recorded action = this trace)",
        fontsize=11,
    )
    ax.legend(facecolor=PAGE, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "smooth_wrist_trace.png", dpi=140, facecolor=PAGE)
    plt.close(fig)

    # 3. yield panel from the banked measurement JSONs
    configs = [
        ("v1\n(one-shot, tail 150)", "smooth_base"),
        ("slew 6\ntail 150", "smooth_slew6"),
        ("slew 10\ntail 150", "smooth_slew10"),
        ("slew 10\ntail 300\n(v1.1)", "smooth_slew10_tail300"),
        ("one-shot\ntail 300", "smooth_base_tail300"),
    ]
    kept, parked = [], []
    for _, stem in configs:
        s = json.loads((ROOT / f"fontaine/notes/{stem}.json").read_text())["summary"]
        kept.append(s["kept_pct"])
        parked.append(s["parked_pct_of_placed"])
    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=PAGE)
    style_ax(ax)
    ax.bar(x - 0.19, kept, 0.34, color=BLUE, label="kept % of attempts")
    ax.bar(x + 0.19, parked, 0.34, color=MAGENTA, label="parked % of placed")
    for xi, (k, p) in enumerate(zip(kept, parked, strict=True)):
        ax.text(xi - 0.19, k + 1.2, f"{k:.0f}", ha="center", color=TEXT, fontsize=9)
        ax.text(xi + 0.19, p + 1.2, f"{p:.0f}", ha="center", color=TEXT, fontsize=9)
    ax.axhline(48.3, color=META, linewidth=0.8, linestyle="--", alpha=0.8)
    ax.text(4.45, 49.3, "48.3 kept anchor", color=META, fontsize=8, ha="right")
    ax.set_xticks(x, [c[0] for c in configs])
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title(
        "Yield vs smoothing config — 120 seeds each, spawn v2.1 + mix70",
        fontsize=11,
    )
    ax.legend(
        facecolor=PAGE,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=9,
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "smooth_yield.png", dpi=140, facecolor=PAGE)
    plt.close(fig)
    print("charts ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
