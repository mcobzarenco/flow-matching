"""Sample videos for the expert-approach smoothing work (owner eyeball
deliverable, queue item expert-approach-quasistatic-redesign).

Renders full collector-style episodes (main clock + post-success
retreat tail) as side-by-side top|wrist mp4s under a chosen approach
config — the visual counterpart of smooth_expert_measure.py's numbers.
CPU-only by design (CUDA stays hidden) so it can run beside a live
training job; a handful of seeds is cheap.

Usage:
  CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl uv run python \
      fontaine/scripts/smooth_expert_videos.py \
      --approach-slew 5 --seeds 1001,1002,1009 \
      --out-dir outputs/sim/smooth_expert/v14
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _slew_arg(v: str) -> float | None:
    return None if v.lower() in ("none", "off") else float(v)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach-slew", type=_slew_arg, default="none")
    parser.add_argument("--arm-slew", type=_slew_arg, default="10")
    parser.add_argument("--seeds", required=True, help="comma list")
    parser.add_argument("--tail-ticks", type=int, default=300)
    parser.add_argument("--max-ticks", type=int, default=600)
    parser.add_argument("--spawn-version", default="v2.1")
    parser.add_argument("--tint-band", default="mix70")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--suffix", default="", help="filename tag")
    args = parser.parse_args()

    from sim.rollout_sim import VideoWriter
    from sim.scripted_expert import ScriptedExpert
    from sim.so101_sim import SO101Sim

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ScriptedExpert.SLEW_ARM_DEG = args.arm_slew
    ScriptedExpert.APPROACH_SLEW_DEG = args.approach_slew
    sim = SO101Sim(spawn_version=args.spawn_version, tint_band=args.tint_band)

    for seed in (int(s) for s in args.seeds.split(",")):
        tag = args.suffix or ("ease" if args.approach_slew is not None else "baseline")
        writer = VideoWriter(args.out_dir / f"seed{seed}_{tag}.mp4")
        obs = sim.reset(seed)
        writer.append(np.hstack([obs.top, obs.wrist]))
        expert = ScriptedExpert(sim)
        placed_at = None
        for tick in range(args.max_ticks + args.tail_ticks):
            obs = sim.step(expert.action(sim))
            writer.append(np.hstack([obs.top, obs.wrist]))
            if placed_at is None:
                if sim.success():
                    placed_at = tick
                elif tick + 1 >= args.max_ticks:
                    break
            elif tick - placed_at >= args.tail_ticks:
                break
        writer.close()
        print(
            f"seed {seed} [{tag}]: "
            f"{'placed@' + str(placed_at) if placed_at is not None else 'MISS'} "
            f"final {sim.benchy_disk_distance() * 100:.1f} cm",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
