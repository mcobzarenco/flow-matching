"""Side-spawn feasibility probe (owner ask 2026-08-16 12:18:57Z,
queue item ``side-spawn-feasibility-probe``).

Three measured phases, all CPU (unrendered expert):

A. rest: does ``reset(boat_start="side")`` produce a stable side-lying
   boat? (settled upright distribution over seeds)
B. stock: what does the CURRENT expert do against the upright>0.9
   success oracle on side spawns? (expected ~0)
C. right: prototype righting maneuver — non-prehensile push-roll: the
   closed jaw TIP descends on the DECK side of the side-lying hull,
   then sweeps horizontally toward the KEEL side, aiming to tip the
   boat over its keel-side contact edge so it lands keel-down; a fresh
   ScriptedExpert then runs the normal grasp-place. Reports righting
   rate and end-to-end success rate — the numbers that decide whether
   side spawns become a v1.1 dataset slice. (Measured outcome: the
   boat slides, it does not roll — see ``right_boat``.)

The maneuver is privileged (reads the deck normal from sim state),
matching the scripted expert's charter: demos, not policy.

Usage: uv run python -m sim.probe_side_spawn [--n 120] [--phase all]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .scripted_expert import (
    DEMO_SEED_BASE,
    JAW_CLOSED_RAD,
    JAW_OPEN_RAD,
    PICKUP_QPOS,
    ExpertPlanner,
    ScriptedExpert,
    _quat_yaw,
)
from .so101_sim import HOME_DEGREES, PHYSICS_STEPS_PER_TICK, SO101Sim

PUSH_Z = 0.029  # TIP height during the sweep — right at the raised
# hull edge (side-lying top ~0.031); lower pushes slide even harder.
START_OFFSET = 0.045  # sweep start on the DECK side of the hull
END_OFFSET = 0.025  # sweep end: this far PAST center on the keel side
SWEEP_STEP = 0.001  # m per tick along the sweep line (slow: quasistatic)
HOVER_Z = 0.07  # tip-space hover clearing the side-lying hull


def _tick(sim: SO101Sim, arm_rad: np.ndarray, jaw_rad: float) -> None:
    """One 30 Hz control tick of absolute joint targets, unrendered."""
    action = np.concatenate([arm_rad, [jaw_rad]])
    target = np.clip(action, sim._ctrl_low, sim._ctrl_high)
    sim.data.ctrl[sim._actuator_ids] = target
    mujoco.mj_step(sim.model, sim.data, nstep=PHYSICS_STEPS_PER_TICK)


def _solve_push(
    planner: ExpertPlanner,
    target: np.ndarray,
    hull_yaw: float,
    seed_rad: np.ndarray,
) -> np.ndarray:
    """TIP-space IK: the ``gripperframe`` site sits at the jaw-tip
    cluster, the only part of the gripper that can work below ~0.077 —
    pad-midpoint space has a PHYSICAL floor there (shoulder saturates
    with the jaw boxes/tip doing the touching; measured, and true even
    for the stock expert's own descend). Jaw axis is aligned PARALLEL
    to the hull so the closed jaw presents its broad face along the
    push direction (alignment target shifted by -pi/2)."""
    arm = seed_rad.copy()
    for _ in range(2):
        arm = planner.align_wrist_roll(arm, hull_yaw - np.pi / 2, rounds=1)
        arm, _ = planner.solve_ik(target, arm, free_dofs=4)
    return arm


def right_boat(sim: SO101Sim, planner: ExpertPlanner, budget: int = 260) -> dict:
    """Run the push-roll on the live sim; return measured outcome.

    Deck normal (body z in world) projected to XY gives the push line:
    start on the deck side, sweep the jaw TIP through the raised hull
    edge toward the keel side — the roll that would land the keel down.

    MEASURED RESULT (2026-08-16, n=4 dev seeds per variant): the boat
    SLIDES instead of rolling — 6-7 cm of plow with peak upright <=
    0.12 — across every execution variant tried: closed-jaw pad-space
    sweep (jaw tip struck the table: pads have a hard floor at
    ~0.077), open-jaw pad-space sweep (same floor), pad-space keel
    press at 2 alignments (press bottomed out on the table), tip-space
    sweep at z 0.022/0.024/0.029 and 1-1.5 mm/tick. Quasistatic
    lateral pushing cannot beat table friction + the rounded hull:
    the transferable tipping moment tops out below the restoring
    moment. Kept as the honest 0% baseline the report cites; a viable
    righting design needs a different mechanism (see the probe
    report), not more tuning of this one."""
    boat_pos, upright0 = sim.benchy_pose()
    deck = sim.data.xmat[sim._benchy_body].reshape(3, 3)[:, 2]
    deck_xy = deck[:2] / max(float(np.hypot(*deck[:2])), 1e-9)
    hull_yaw = _quat_yaw(
        sim.data.qpos[sim._benchy_qpos + 3 : sim._benchy_qpos + 7],
    )
    start = np.array(
        [*(boat_pos[:2] + deck_xy * START_OFFSET), PUSH_Z],
    )
    end = np.array(
        [*(boat_pos[:2] - deck_xy * END_OFFSET), PUSH_Z],
    )
    ticks = 0

    def run_to(target: np.ndarray, *, tol: float, tmax: int) -> None:
        nonlocal ticks
        droop = np.zeros(3)
        for step in range(tmax):
            tip = sim.data.site_xpos[planner.site_id].copy()
            if float(np.linalg.norm(tip - target)) < tol:
                return
            arm_speed = float(np.abs(sim.data.qvel[planner.arm_dofs]).max())
            if step > 15 and arm_speed < 0.08:
                droop = np.clip(droop + (target - tip), -0.04, 0.04)
            arm = _solve_push(planner, target + droop, hull_yaw, PICKUP_QPOS[:5])
            _tick(sim, arm, JAW_CLOSED_RAD)
            ticks += 1

    # Hover the tip above the start point, descend, sweep.
    run_to(np.array([*start[:2], HOVER_Z]), tol=0.015, tmax=50)
    run_to(start, tol=0.008, tmax=50)
    line = end - start
    length = float(np.linalg.norm(line))
    direction = line / length
    steps = int(np.ceil(length / SWEEP_STEP))
    droop = np.zeros(3)
    for step in range(min(steps, budget)):
        target = start + direction * min((step + 1) * SWEEP_STEP, length)
        tip = sim.data.site_xpos[planner.site_id].copy()
        arm_speed = float(np.abs(sim.data.qvel[planner.arm_dofs]).max())
        if arm_speed < 0.08:
            droop = np.clip(droop + (target - tip), -0.03, 0.03)
        arm = _solve_push(planner, target + droop, hull_yaw, PICKUP_QPOS[:5])
        _tick(sim, arm, JAW_CLOSED_RAD)
        ticks += 1
        _, upright = sim.benchy_pose()
        if upright > 0.92:
            break
    # Retreat up-and-back so the follow-on grasp starts clean.
    parked = sim.data.qpos[planner.arm_qpos].copy()
    parked[1] -= np.deg2rad(35.0)
    for _ in range(20):
        _tick(sim, parked, JAW_OPEN_RAD)
        ticks += 1
    home = np.deg2rad(HOME_DEGREES[:5])
    for _ in range(25):
        _tick(sim, home, JAW_OPEN_RAD)
        ticks += 1
    pos, upright = sim.benchy_pose()
    return {
        "righted": bool(upright > 0.9),
        "upright_before": float(upright0),
        "upright_after": float(upright),
        "boat_z": float(pos[2]),
        "ticks": ticks,
    }


def episode_with_righting(sim: SO101Sim, seed: int, max_ticks: int = 900) -> dict:
    """Side spawn -> push-roll -> stock grasp-place, one outcome row."""
    sim.reset(seed, boat_start="side")
    planner = ExpertPlanner(sim)
    righting = right_boat(sim, planner)
    expert = ScriptedExpert(sim)
    tick = 0
    for tick in range(max_ticks - righting["ticks"]):  # noqa: B007
        arm_jaw = np.deg2rad(expert.action(sim))
        _tick(sim, arm_jaw[:5], float(arm_jaw[5]))
        if sim.success():
            break
    _pos, upright = sim.benchy_pose()
    return {
        "seed": seed,
        "success": bool(sim.success()),
        **righting,
        "total_ticks": righting["ticks"] + tick + 1,
        "final_disk_cm": sim.benchy_disk_distance() * 100,
        "final_upright": float(upright),
        "phase_trace": [*expert.state.trace, expert.state.phase],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed-base", type=int, default=DEMO_SEED_BASE)
    parser.add_argument(
        "--phase",
        choices=["rest", "stock", "right", "all"],
        default="all",
    )
    parser.add_argument("--out", default="/tmp/probe_side_spawn.json")
    args = parser.parse_args()
    seeds = range(args.seed_base, args.seed_base + args.n)
    sim = SO101Sim(spawn_version="v2.1")
    report: dict[str, object] = {"n": args.n, "seed_base": args.seed_base}

    if args.phase in ("rest", "all"):
        uprights = []
        for seed in seeds:
            sim.reset(seed, boat_start="side")
            _pos, upright = sim.benchy_pose()
            uprights.append(upright)
        u = np.array(uprights)
        report["rest"] = {
            "on_side": float(np.mean(np.abs(u) < 0.5)),
            "self_righted": float(np.mean(u > 0.9)),
            "capsized": float(np.mean(u < -0.9)),
            "median_upright": float(np.median(u)),
        }
        print(f"[rest] {report['rest']}")

    if args.phase in ("stock", "all"):
        from .scripted_expert import run_expert_episode

        rows = [
            run_expert_episode(sim, seed, reset_kwargs={"boat_start": "side"})
            for seed in seeds
        ]
        u = np.array([float(r["upright"]) for r in rows])  # type: ignore[arg-type]
        report["stock"] = {
            "success": float(np.mean([bool(r["success"]) for r in rows])),
            "ended_upright": float(np.mean(u > 0.9)),
            "median_final_upright": float(np.median(u)),
        }
        print(f"[stock] {report['stock']}")

    if args.phase in ("right", "all"):
        rows = [episode_with_righting(sim, seed) for seed in seeds]
        righted = float(np.mean([r["righted"] for r in rows]))
        report["right"] = {
            "righted": righted,
            "end_to_end_success": float(np.mean([r["success"] for r in rows])),
            "median_righting_ticks": float(
                np.median([r["ticks"] for r in rows]),
            ),
            "median_total_ticks": float(
                np.median([r["total_ticks"] for r in rows]),
            ),
        }
        print(f"[right] {report['right']}")
        report["right_rows"] = rows

    with Path(args.out).open("w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
