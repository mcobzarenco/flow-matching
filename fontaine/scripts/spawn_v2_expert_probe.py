"""Spawn-v2 expert failure-mode probe (A-prime diagnostic, unrendered).

The first rendered smoke on the A100 box measured the scripted expert
at ~14% under spawn-v2 (vs ~62% on the v1 band) — the pre-reg's
registered A-prime risk (§4: the traverse was designed around a fixed disk
bearing). This probe runs the expert WITHOUT rendering (observe()
monkeypatched to state-only; the expert consumes privileged reads, not
pixels) so hundreds of seeds sweep in minutes on CPU, and records the
geometry + phase trace of every episode for failure classification.

Usage (on the A100 box or anywhere):
  uv run python fontaine/scripts/spawn_v2_expert_probe.py \
      --seed-start 2000 --count 600 --procs 120 \
      --out reports/analysis__spawn_v2_expert_probe.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Spawn-context workers re-import this module without the repo cwd on
# sys.path — pin it so `sim` resolves.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _run_seed(job: tuple[int, float, int, str]) -> dict[str, Any]:
    seed, droop_clip, max_ticks, spawn_version = job
    # Constructed per call under maxtasksperchild -- but model compile
    # amortizes via the module-level cache below when tasks batch.
    global _SIM  # noqa: PLW0603 — per-process sim cache
    try:
        sim = _SIM
    except NameError:
        from sim.so101_sim import SimObservation, SO101Sim

        sim = SO101Sim(render_style="v0", spawn_version=spawn_version)

        def observe_stateonly() -> SimObservation:
            state = np.rad2deg(sim.data.qpos[sim._joint_qpos])
            return SimObservation(
                top=np.zeros((1, 1, 3), dtype=np.uint8),
                wrist=np.zeros((1, 1, 3), dtype=np.uint8),
                state=state,
            )

        sim.observe = observe_stateonly  # type: ignore[method-assign]
        _SIM = sim

    from sim.scripted_expert import ScriptedExpert

    ScriptedExpert.DROOP_CLIP = droop_clip
    sim.reset(seed)
    spawn = sim.reset_spawn_v2
    expert = ScriptedExpert(sim)
    ticks = 0
    for ticks in range(max_ticks):  # noqa: B007
        action = expert.action(sim)
        sim.step(action)
        if sim.success():
            break
    pos, upright = sim.benchy_pose()
    disk = np.array(spawn.disk_xy)
    boat = np.array(spawn.boat_xy)
    return {
        "seed": seed,
        "success": bool(sim.success()),
        "ticks": ticks + 1,
        "final_disk_cm": sim.benchy_disk_distance() * 100,
        "disk_xy": [round(float(v), 4) for v in disk],
        "boat_xy": [round(float(v), 4) for v in boat],
        "disk_r_base": round(float(np.hypot(*disk)), 4),
        "disk_bearing_deg": round(float(np.degrees(np.arctan2(disk[1], disk[0]))), 1),
        "boat_r_base": round(float(np.hypot(*boat)), 4),
        "boat_disk_r": round(float(np.hypot(*(boat - disk))), 4),
        "final_boat_z": round(float(pos[2]), 4),
        "final_upright": round(float(upright), 3),
        "trace": expert.state.trace[-14:],
        "final_phase": expert.state.phase,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--procs", type=int, default=mp.cpu_count() // 2)
    parser.add_argument("--droop-clip", type=float, default=0.04)
    parser.add_argument("--max-ticks", type=int, default=600)
    parser.add_argument("--spawn-version", choices=("v2", "v2.1"), default="v2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    jobs = [
        (seed, args.droop_clip, args.max_ticks, args.spawn_version)
        for seed in range(args.seed_start, args.seed_start + args.count)
    ]
    with mp.get_context("spawn").Pool(args.procs) as pool:
        records = pool.map(_run_seed, jobs, chunksize=4)

    ok = [r for r in records if r["success"]]
    print(f"success {len(ok)}/{len(records)} = {len(ok) / len(records):.1%}")
    fails = [r for r in records if not r["success"]]
    # Failure geometry: bin by disk radius from base and by final miss
    # distance; print the modal final phases.
    for lo, hi in ((0.10, 0.18), (0.18, 0.26), (0.26, 0.34), (0.34, 0.42)):
        band = [r for r in records if lo <= r["disk_r_base"] < hi]
        if band:
            rate = sum(r["success"] for r in band) / len(band)
            print(f"  disk r_base [{lo:.2f},{hi:.2f}): {rate:.1%} of {len(band)}")
    for lo, hi in ((0.10, 0.18), (0.18, 0.26), (0.26, 0.34), (0.34, 0.42)):
        band = [r for r in records if lo <= r["boat_r_base"] < hi]
        if band:
            rate = sum(r["success"] for r in band) / len(band)
            print(f"  boat r_base [{lo:.2f},{hi:.2f}): {rate:.1%} of {len(band)}")
    phases: dict[str, int] = {}
    for r in fails:
        phases[r["final_phase"]] = phases.get(r["final_phase"], 0) + 1
    print(
        "  failure final phases:",
        dict(sorted(phases.items(), key=lambda kv: -kv[1])),
    )
    near = sum(1 for r in fails if r["final_disk_cm"] < 6)
    print(f"  fails ending <6 cm from disk: {near}/{len(fails)}")
    lifted = sum(1 for r in fails if r["final_boat_z"] > 0.03)
    print(f"  fails ending with boat lifted (z>3cm): {lifted}/{len(fails)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"records": records}, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
