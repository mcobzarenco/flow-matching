"""Stage-A gate read for the grasp-rich SFT bootstrap
(posts/2026-08-14-prereg-grasp-sft-bootstrap.md §6, frozen 01:4xZ 08-15).

Registered read: the scripted expert on the 20 HELD demo seeds
1020-1039 (disjoint from the 1000-1015 tuning smoke by construction),
rendered on the production sim100 substrate (``SO101Sim()`` defaults —
v3, deployed equidistant wrist), one top|wrist video banked per seed.
Gate: successes >= 14/20 (>= 70%) -> stage B GO; below -> F-physics
(Squint twin tier escalation per pre-reg §4).

The expert itself is pixel-free (privileged state); rendering is for
the record and to run the read on exactly the observation pipeline
stage-B demos will be collected under.

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/grasp_sft_stagea_gate.py \
      [--seed-base 1020] [--num-seeds 20] [--gate-bar 14] \
      [--out reports/analysis__grasp_sft_stageA_gate.json] \
      [--video-dir outputs/sim/grasp_sft/stageA_gate]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from sim.rollout_sim import VideoWriter
from sim.scripted_expert import run_expert_episode
from sim.so101_sim import SO101Sim

GATE_SEED_BASE = 1020  # frozen §6: held seeds 1020-1039
GATE_NUM_SEEDS = 20
GATE_BAR = 14  # >= 70% of 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-base", type=int, default=GATE_SEED_BASE)
    parser.add_argument("--num-seeds", type=int, default=GATE_NUM_SEEDS)
    parser.add_argument("--gate-bar", type=int, default=GATE_BAR)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/analysis__grasp_sft_stageA_gate.json"),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("outputs/sim/grasp_sft/stageA_gate"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.video_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    sim = SO101Sim()
    rows: list[dict[str, object]] = []
    t_start = time.time()
    for seed in range(args.seed_base, args.seed_base + args.num_seeds):
        writer = VideoWriter(args.video_dir / f"expert_seed{seed}.mp4")
        t0 = time.time()
        row = run_expert_episode(
            sim,
            seed,
            render=True,
            on_obs=lambda obs, w=writer: w.append(
                np.hstack([obs.top, obs.wrist]),
            ),
        )
        writer.close()
        row["wall_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print(
            f"seed {seed}: {'SUCCESS' if row['success'] else 'MISS   '} "
            f"ticks {row['ticks']:>3} final {row['final_disk_cm']:.1f} cm "
            f"upright {row['upright']:.2f} ({row['wall_s']}s) "
            f"phases {'>'.join(row['phase_trace'])}",
            flush=True,
        )

    successes = sum(bool(r["success"]) for r in rows)
    gate_pass = successes >= args.gate_bar
    payload = {
        "read": "grasp_sft_stageA_gate",
        "prereg": "posts/2026-08-14-prereg-grasp-sft-bootstrap.md §6",
        "head": head,
        "substrate": "SO101Sim() defaults (v3, deployed equidistant wrist)",
        "seeds": [args.seed_base, args.seed_base + args.num_seeds - 1],
        "gate": f">={args.gate_bar}/{args.num_seeds}",
        "successes": successes,
        "gate_pass": gate_pass,
        "verdict": "STAGE-B GO" if gate_pass else "F-physics (Squint twin tier)",
        "wall_s_total": round(time.time() - t_start, 1),
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"\nGATE {'PASS' if gate_pass else 'FAIL'}: {successes}/"
        f"{args.num_seeds} (bar {args.gate_bar}) -> {payload['verdict']}"
        f"\nbanked {args.out}",
    )
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
