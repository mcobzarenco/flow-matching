"""Instrumented kept%/parked% read for expert-trajectory smoothing
(owner 16:53Z 2026-08-16: 'action traces are very jumpy — smoother
overall?'; extends the 15:22Z retreat-slew work order).

Replicates the collector's episode loop (main clock -> post-success
retreat tail -> success re-verify, sim/collect_demos.py
expert_episode_source) WITHOUT recording frames, and attributes every
placed-but-demoted episode to a concrete cause instead of guessing:

  still-bar-arm   post-tail success() failed ONLY its stillness bar,
                  and an ARM dof is what's moving (the 150-tick tail
                  expired mid-ramp) — a measurement artifact, the
                  boat is placed
  still-bar-other stillness bar failed on a non-arm dof (boat wobble)
  boat-moved      boat left the disk radius during the tail
  boat-tipped     upright dropped below the 0.9 bar
  boat-height     base z left the [0.004, 0.03] window

Also reports the commanded-trace smoothness (per-tick |delta| of the
recorded action channels) so 'smoother' is a measured claim, and
parked% (arm at home + quiet at tail exit) which is what the recorded
tail actually teaches the policy.

Usage:
  uv run python fontaine/scripts/smooth_expert_measure.py \
      --arm-slew 6 --jaw-slew 8 --seeds 1000:1120 \
      --out fontaine/notes/smooth_slew6.json
  --arm-slew none disables the output stage (legacy one-shot baseline).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

# Pin BLAS to one thread BEFORE numpy loads anywhere (parent env is
# inherited by spawn workers): under worker oversubscription OpenBLAS
# varies its reduction partitioning at runtime, the DLS IK solutions
# jitter bitwise, and the contact-rich sim amplifies that into
# DIFFERENT EPISODE OUTCOMES on marginal seeds run-to-run (measured
# 08-18: identical invocations placed 6/23 vs 3/23 with disjoint seed
# sets; pinned runs are bitwise identical).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

#: Per-worker state — filled by the pool initializer (a dict so no
#: ``global`` rebinding is needed).
_W: dict = {}


def _slew_arg(v: str) -> float | None:
    return None if v.lower() in ("none", "off") else float(v)


def _init_worker(cfg: dict) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sim.scripted_expert import ScriptedExpert
    from sim.so101_sim import SO101Sim

    _W["cfg"] = cfg
    ScriptedExpert.SLEW_ARM_DEG = cfg["arm_slew"]
    ScriptedExpert.SLEW_JAW_DEG = cfg["jaw_slew"]
    ScriptedExpert.APPROACH_SLEW_DEG = cfg["approach_slew"]
    ScriptedExpert.RETREAT_GLIDE = cfg["retreat_glide"]
    _W["sim"] = SO101Sim(
        spawn_version=cfg["spawn_version"],
        tint_band=cfg["tint_band"],
    )


def run_episode(seed: int) -> dict:
    import mujoco

    from sim.scripted_expert import ScriptedExpert
    from sim.so101_sim import HOME_DEGREES, PHYSICS_STEPS_PER_TICK

    sim = _W["sim"]
    sim.reset(seed)
    expert = ScriptedExpert(sim)
    cmds: list[np.ndarray] = []

    def step() -> None:
        action = expert.action(sim)
        cmds.append(np.asarray(action, dtype=np.float64))
        target = np.clip(np.deg2rad(action), sim._ctrl_low, sim._ctrl_high)
        sim.data.ctrl[sim._actuator_ids] = target
        mujoco.mj_step(sim.model, sim.data, nstep=PHYSICS_STEPS_PER_TICK)

    tick = 0
    for tick in range(_W["cfg"]["max_ticks"]):  # noqa: B007
        step()
        if sim.success():
            break
    placed = bool(sim.success())
    ticks_main = tick + 1

    home_rad = np.deg2rad(HOME_DEGREES[:5])
    arm_qpos = sim._joint_qpos[:5]
    arm_dofs = expert.planner.arm_dofs
    boat_at_place, _ = sim.benchy_pose()
    boat_at_place = boat_at_place.copy()
    tail_used = 0
    tail_exit = "none"
    if placed:
        tail_exit = "budget"
        for _ in range(_W["cfg"]["tail_ticks"]):
            step()
            tail_used += 1
            at_home = float(
                np.max(np.abs(sim.data.qpos[arm_qpos] - home_rad)),
            ) < np.deg2rad(10.0)
            quiet = float(np.abs(sim.data.qvel).max()) < 0.05
            if at_home and quiet:
                tail_exit = "parked"
                break

    kept = placed and bool(sim.success())

    # Post-tail success components, re-derived for attribution.
    pos, upright = sim.benchy_pose()
    on_disk = sim.benchy_disk_distance() < sim.disk_radius
    at_height = 0.004 < pos[2] < 0.03
    qvel = np.abs(sim.data.qvel)
    max_qvel_arm = float(qvel[arm_dofs].max())
    rest = np.ones(len(sim.data.qvel), dtype=bool)
    rest[arm_dofs] = False
    max_qvel_rest = float(qvel[rest].max()) if rest.any() else 0.0
    still = float(qvel.max()) < 0.5
    at_home_deg = float(
        np.rad2deg(np.max(np.abs(sim.data.qpos[arm_qpos] - home_rad))),
    )

    cause = ""
    if placed and not kept:
        if on_disk and at_height and upright > 0.9 and not still:
            cause = (
                "still-bar-arm" if max_qvel_arm >= max_qvel_rest else "still-bar-other"
            )
        elif not on_disk:
            cause = "boat-moved"
        elif upright <= 0.9:
            cause = "boat-tipped"
        else:
            cause = "boat-height"

    deltas = np.abs(np.diff(np.stack(cmds), axis=0))
    return {
        "seed": seed,
        "placed": placed,
        "kept": kept,
        "cause": cause,
        "ticks_main": ticks_main,
        "tail_used": tail_used,
        "tail_exit": tail_exit,
        "parked": tail_exit == "parked",
        "at_home_deg": round(at_home_deg, 2),
        "boat_tail_move_cm": round(
            float(np.hypot(*(pos[:2] - boat_at_place[:2]))) * 100,
            2,
        ),
        "final_disk_cm": round(sim.benchy_disk_distance() * 100, 2),
        "upright": round(float(upright), 3),
        "max_qvel_arm": round(max_qvel_arm, 3),
        "max_qvel_rest": round(max_qvel_rest, 3),
        "arm_step_mean_deg": round(float(deltas[:, :5].mean()), 3),
        "arm_step_p99_deg": round(float(np.quantile(deltas[:, :5], 0.99)), 2),
        "arm_step_max_deg": round(float(deltas[:, :5].max()), 2),
        "jaw_step_max_deg": round(float(deltas[:, 5].max()), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-slew", type=_slew_arg, default="6")
    parser.add_argument("--jaw-slew", type=_slew_arg, default="8")
    parser.add_argument(
        "--seeds",
        default="1000:1120",
        help="start:stop range, or a comma list (targeted smokes)",
    )
    parser.add_argument("--tail-ticks", type=int, default=150)
    parser.add_argument("--max-ticks", type=int, default=600)
    parser.add_argument("--spawn-version", default="v2.1")
    parser.add_argument("--tint-band", default="mix70")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    # v1.2 knobs (owner 2026-08-16 19:42Z): approach-leg eased cap and
    # the retreat home glide. Defaults mirror the class defaults
    # (approach ease OFF — measured NO-GO, see ScriptedExpert); the
    # v1.1 reference arm is --approach-slew none --retreat-glide off.
    parser.add_argument("--approach-slew", type=_slew_arg, default="none")
    parser.add_argument(
        "--retreat-glide",
        choices=("on", "off"),
        default="on",
    )
    args = parser.parse_args()

    if ":" in args.seeds:
        start, stop = (int(x) for x in args.seeds.split(":"))
        seeds = list(range(start, stop))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]
    cfg = {
        "arm_slew": args.arm_slew,
        "jaw_slew": args.jaw_slew,
        "tail_ticks": args.tail_ticks,
        "max_ticks": args.max_ticks,
        "spawn_version": args.spawn_version,
        "tint_band": args.tint_band,
        "approach_slew": args.approach_slew,
        "retreat_glide": args.retreat_glide == "on",
    }

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        rows = pool.map(run_episode, seeds)

    n = len(rows)
    placed = [r for r in rows if r["placed"]]
    kept = [r for r in rows if r["kept"]]
    parked = [r for r in placed if r["parked"]]
    causes: dict[str, int] = {}
    for r in placed:
        if r["cause"]:
            causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    summary = {
        "config": cfg,
        "n": n,
        "placed_pct": round(100 * len(placed) / n, 1),
        "kept_pct": round(100 * len(kept) / n, 1),
        "parked_pct_of_placed": round(100 * len(parked) / max(len(placed), 1), 1),
        "demoted_causes": causes,
        "tail_used_mean": round(
            float(np.mean([r["tail_used"] for r in placed])) if placed else 0.0,
            1,
        ),
        "tail_budget_hits": sum(1 for r in placed if r["tail_exit"] == "budget"),
        "arm_step_mean_deg": round(
            float(np.mean([r["arm_step_mean_deg"] for r in rows])),
            3,
        ),
        "arm_step_max_deg": round(max(r["arm_step_max_deg"] for r in rows), 2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
