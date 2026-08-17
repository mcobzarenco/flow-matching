"""Render expert sample episodes as side-by-side top|wrist videos.

The in-channel eyeball artifact (owner pattern 2026-08-16: "seen
rather than described") — the exact collector episode loop
(sim/collect_demos.py expert_episode_source: main clock -> post-success
retreat tail -> re-verify) through the production visual config, with
the frames written to one H.264 stream instead of a dataset. Live
dataset shards stay locked (memory: sample by re-render from a kept
seed); this script IS that re-render path, now reusable.

Usage:
  uv run python fontaine/scripts/render_expert_samples.py \
      --seeds 1002,1005 --out-dir /tmp/expert_samples \
      [--spawn-version v2.1] [--tint-band mix70] [--prefix v12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True, help="comma-separated")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spawn-version", default="v2.1")
    parser.add_argument("--tint-band", default="mix70")
    parser.add_argument("--bracket-appearance", default="v1")
    parser.add_argument("--wrist-pose", default="v1")
    parser.add_argument("--prefix", default="sample")
    args = parser.parse_args()

    from sim.collect_demos import expert_episode_source
    from sim.rollout_sim import VideoWriter
    from sim.so101_sim import SO101Sim

    sim = SO101Sim(
        spawn_version=args.spawn_version,
        tint_band=args.tint_band,
        bracket_appearance=args.bracket_appearance,
        wrist_pose=args.wrist_pose,
    )
    source = expert_episode_source(sim)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in args.seeds.split(",")):
        episode = source(seed)
        path = args.out_dir / f"{args.prefix}_seed{seed}.mp4"
        writer = VideoWriter(path)
        for frame in episode.frames:
            writer.append(np.concatenate([frame.top, frame.wrist], axis=1))
        writer.close()
        print(
            f"seed {seed}: success={episode.success} ticks={episode.ticks} "
            f"final_disk_cm={episode.final_disk_cm:.1f} -> {path}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
