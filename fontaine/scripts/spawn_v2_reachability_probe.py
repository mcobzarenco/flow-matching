"""Spawn-v2 reachability probe — the measuring instrument behind the
workspace mask W (pre-reg DRAFT posts/2026-08-16-prereg-sim-spawn-v2.md
§3). CPU-only, no GL, no sim mutation: kinematics run on the scripted
expert's scratch-data planner.

For every cell of a Cartesian grid over the table's working quadrant,
solve the stage-A grasp pose (jaw-pad-midpoint IK at GRASP_Z, wrist
locked to the P4 pitch, roll aligned to a radially-facing hull — the
pan-arc carry convention) seeded from the pickup keyframe with the pan
pre-swung to the cell's bearing, then measure at the solved pose:

- ``residual``: pad-midpoint position error (m) — the IK feasibility
  field (the pre-reg's < 1 mm candidate bar);
- ``moment_frac``: static gravity torque at shoulder_lift / elbow_flex
  as a fraction of the sysid'd forcerange 3.478 (``qfrc_bias`` at rest
  — the stage-A torque wall, measured per pose rather than assumed).

The probe emits raw fields; the W mask CONSTANT is frozen at pre-reg
finalization by picking the residual/torque bounds over these fields.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from sim.scripted_expert import (
    JAW_OPEN_RAD,
    PICKUP_QPOS,
    ExpertPlanner,
    ScriptedExpert,
)
from sim.so101_sim import SERVO_SYSID, SO101Sim

GRID_X = (0.05, 0.40)
GRID_Y = (-0.20, 0.40)
PITCH = 0.01  # m


def solve_grasp_tight(
    planner: ExpertPlanner,
    target: np.ndarray,
    boat_yaw: float,
    seed_rad: np.ndarray,
    *,
    rounds: int = 3,
) -> tuple[np.ndarray, float]:
    """The stage-A grasp solve with an instrument-grade tolerance —
    ``ExpertPlanner.solve_grasp`` stops at solve_ik's 2 mm SITE
    tolerance, so whether a cell's pad residual lands under 1 mm is
    stopping luck, not reachability (the v0 ring-banding). Same
    alternation (roll align + wrist-locked pad IK), tol 0.2 mm and a
    doubled iteration budget, LOCAL to the probe: stage-A behavior is
    frozen, the instrument only measures harder."""
    arm = seed_rad.copy()
    arm[3] = seed_rad[3]
    for _ in range(rounds):
        arm = planner.align_wrist_roll(arm, boat_yaw, rounds=1)
        for _ in range(2):
            mid = planner.grasp_point(arm, JAW_OPEN_RAD)
            site = planner.scratch.site_xpos[planner.site_id].copy()
            arm, _ = planner.solve_ik(
                target - (mid - site),
                arm,
                free_dofs=4,
                tol=2e-4,
                iters=120,
            )
    residual = float(
        np.linalg.norm(target - planner.grasp_point(arm, JAW_OPEN_RAD)),
    )
    return arm, residual


def probe_cell(
    planner: ExpertPlanner,
    scratch: mujoco.MjData,
    model: mujoco.MjModel,
    x: float,
    y: float,
    pan_anchor: np.ndarray,
) -> dict[str, float]:
    bearing = float(np.arctan2(y - pan_anchor[1], x - pan_anchor[0]))
    seed = PICKUP_QPOS[:5].copy()
    seed[0] = np.clip(bearing, planner.arm_low[0], planner.arm_high[0])
    # Radially-facing hull: jaw axis lands along the bearing (the yaw
    # class the pan-arc traverse produces at every carry endpoint).
    target = np.array([x, y, ScriptedExpert.GRASP_Z])
    arm, residual = solve_grasp_tight(planner, target, bearing, seed)
    scratch.qpos[:] = 0.0
    scratch.qvel[:] = 0.0
    scratch.qpos[planner.arm_qpos] = arm
    mujoco.mj_forward(model, scratch)
    bias = scratch.qfrc_bias[planner.arm_dofs]
    frac = np.abs(bias) / SERVO_SYSID["forcerange"]
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "residual": round(residual, 5),
        "moment_frac_shoulder": round(float(frac[1]), 4),
        "moment_frac_elbow": round(float(frac[2]), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/analysis__spawn_v2_reachability_v0.json"),
    )
    args = parser.parse_args()
    sim = SO101Sim()
    planner = ExpertPlanner(sim)
    scratch = mujoco.MjData(sim.model)
    mujoco.mj_forward(sim.model, scratch)
    pan_anchor = scratch.xanchor[sim.model.joint("shoulder_pan").id].copy()
    xs = np.arange(GRID_X[0], GRID_X[1] + PITCH / 2, PITCH)
    ys = np.arange(GRID_Y[0], GRID_Y[1] + PITCH / 2, PITCH)
    cells = [
        probe_cell(planner, scratch, sim.model, float(x), float(y), pan_anchor)
        for x in xs
        for y in ys
    ]
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    reachable_1mm = sum(c["residual"] < 1e-3 for c in cells)
    out = {
        "instrument": "spawn_v2_reachability_probe v0",
        "head": head,
        "params": {
            "grid_x": GRID_X,
            "grid_y": GRID_Y,
            "pitch": PITCH,
            "grasp_z": ScriptedExpert.GRASP_Z,
            "forcerange": SERVO_SYSID["forcerange"],
            "pan_anchor": [round(float(v), 4) for v in pan_anchor],
            "yaw_convention": "radial hull (pan-arc carry endpoint class)",
        },
        "summary": {
            "cells": len(cells),
            "reachable_residual_lt_1mm": reachable_1mm,
        },
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(
        f"[probe] {len(cells)} cells, {reachable_1mm} inside the 1 mm "
        f"residual bar -> {args.out}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
