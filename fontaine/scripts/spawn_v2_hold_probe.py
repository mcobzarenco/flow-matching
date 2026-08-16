"""Static-hold diagnostic for the spawn-v2 far-radius grasp failure.

The expert probe measured a success cliff in boat r_base (48% below
0.26 m -> ~1% above 0.34) with failures looping jam-flip<->recover:
the measured pads never reach the solve target (>3.5 cm for 50 ticks).
Two mechanisms fit: (a) the servo cannot HOLD the far solution pose
(true saturation — the reachability instrument's static-moment read
was wrong about the live actuator), or (b) it cannot TRAVEL there
(slew path through saturating poses / per-tick IK oscillation).

This probe separates them: for a grid of far boat targets it solves
the expert's own grasp IK, TELEPORTS the arm onto the solution
(bypassing travel), then lets physics run with the servo holding that
target and measures the steady-state pad error and per-joint actuator
force fractions. Small steady error => (b) travel; large => (a) hold.

Usage:
  uv run python fontaine/scripts/spawn_v2_hold_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    from sim.scripted_expert import JAW_OPEN_RAD, ExpertPlanner, ScriptedExpert
    from sim.so101_sim import SO101Sim

    sim = SO101Sim(render_style="v0")
    planner = ExpertPlanner(sim)
    expert = ScriptedExpert(sim)  # for _grasp_seed / constants
    force_limits = np.abs(
        sim.model.actuator_forcerange[sim._actuator_ids[:5], 1],
    )

    print("r_base  bearing  ik_resid_mm  hold_pad_err_mm  worst_joint  force_frac")
    for r in (0.20, 0.24, 0.28, 0.32, 0.36):
        for bearing_deg in (-30.0, 0.0, 30.0):
            b = np.radians(bearing_deg)
            target = np.array(
                [r * np.cos(b), r * np.sin(b), ScriptedExpert.GRASP_Z],
            )
            hull_yaw = b + np.pi / 2  # hull perpendicular to the bearing
            arm, resid = planner.solve_grasp(
                target,
                hull_yaw,
                expert._grasp_seed(),
            )
            # Teleport onto the solution and hold it for 300 steps.
            sim.reset(3000)
            sim.data.qpos[planner.arm_qpos] = arm
            sim.data.qpos[planner.jaw_qpos] = JAW_OPEN_RAD
            sim.data.qvel[:] = 0.0
            sim.data.ctrl[sim._actuator_ids[:5]] = arm
            sim.data.ctrl[sim._actuator_ids[5]] = JAW_OPEN_RAD
            import mujoco

            for _ in range(300):
                mujoco.mj_step(sim.model, sim.data)
            pads = planner.grasp_point(
                sim.data.qpos[planner.arm_qpos],
                JAW_OPEN_RAD,
            )
            err = float(np.linalg.norm(pads - target))
            frac = np.abs(sim.data.actuator_force[sim._actuator_ids[:5]]) / force_limits
            worst = int(np.argmax(frac))
            names = ("pan", "lift", "elbow", "wristf", "wristr")
            print(
                f"{r:.2f}  {bearing_deg:+5.0f}   {resid * 1000:8.2f}     "
                f"{err * 1000:8.1f}       {names[worst]:6s}    "
                f"{float(frac[worst]):.2f}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
