"""Closed-loop bijou rollout in the SO-101 sim (the sim twin of
bijou/rollout.py).

Same inference path as the physical rollouts — BijouPolicy + the real
``observation_to_item`` — with SO101Sim in place of SOFollower. The sim's
top camera is fed under the name "front" so the prompt's sorted camera
slots match training (front, wrist); the viewpoint mismatch is part of
the measured domain gap, not a bug.

Writes a full-resolution side-by-side (front|wrist) H.264 video to
outputs/sim/ and prints per-replan telemetry + the success predicate.

Usage:
  MUJOCO_GL=egl uv run python -m sim.rollout_sim \
      --checkpoint outputs/train/bijou_ft_rig_from_adarms100k_ddp2/step_005000
"""

import argparse
import time
from pathlib import Path

import av
import numpy as np
import torch

from bijou.decoders.flow import SamplingMethod
from bijou.eval.policies import BijouPolicy
from bijou.rollout import SO_MOTORS, observation_to_item

from . import OUTPUT_DIR
from .so101_sim import CONTROL_HZ, DISK_CENTER, SimObservation, SO101Sim

STATS_REPO_ID = "mcobzarenco/so101_pick_place_v2"
TASK = "Pick up the toy boat and place it on the wooden disk."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="env + noise seed")
    parser.add_argument("--replans", type=int, default=15)
    parser.add_argument("--execute-horizon", type=int, default=30)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument(
        "--expert-dtype",
        default="bfloat16",
        choices=["float32", "bfloat16"],
    )
    parser.add_argument("--video", type=Path, default=OUTPUT_DIR / "rollout.mp4")
    return parser.parse_args()


def to_observation(obs: SimObservation) -> dict[str, object]:
    """SimObservation -> the robot-observation dict observation_to_item
    consumes: <motor>.pos floats + named HWC uint8 camera frames."""
    observation: dict[str, object] = {
        f"{motor}.pos": float(obs.state[index]) for index, motor in enumerate(SO_MOTORS)
    }
    observation["front"] = (
        obs.top
    )  # top view rides the front slot (see module docstring)
    observation["wrist"] = obs.wrist
    return observation


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = BijouPolicy(
        args.checkpoint,
        device=device,
        seed=args.seed,
        sample_steps=args.sample_steps,
        method=SamplingMethod.HEUN,
        expert_dtype=getattr(torch, args.expert_dtype),
    )
    stats = policy.info.per_dataset_normalization[STATS_REPO_ID]
    chunk_size = policy.info.chunk_size
    horizon = min(args.execute_horizon, chunk_size)
    print(f"policy: {policy.name} (chunk {chunk_size}, heun-{args.sample_steps})")

    sim = SO101Sim()
    obs = sim.reset(args.seed)
    frames: list[np.ndarray] = []
    success_tick: int | None = None

    for replan in range(args.replans):
        item = observation_to_item(to_observation(obs), TASK, stats, chunk_size)
        start = time.perf_counter()
        chunk = policy.predict([item], [replan])[0]
        latency = time.perf_counter() - start
        pos, _ = sim.benchy_pose()
        distance = float(np.hypot(pos[0] - DISK_CENTER[0], pos[1] - DISK_CENTER[1]))
        print(
            f"replan {replan}: {latency * 1000:.0f} ms | state "
            f"{np.round(obs.state, 1)} | chunk[0] "
            f"{np.round(chunk[0].numpy(), 1)} | benchy->disk {distance * 100:.1f} cm",
            flush=True,
        )
        for step in range(horizon):
            obs = sim.step(chunk[step].numpy())
            frames.append(np.concatenate([obs.top, obs.wrist], axis=1))
            if sim.success():
                success_tick = replan * horizon + step
                break
        if success_tick is not None:
            break

    print(
        f"success: {success_tick is not None}"
        + (f" (tick {success_tick})" if success_tick else ""),
    )
    args.video.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(args.video), mode="w")
    stream = container.add_stream("h264", rate=CONTROL_HZ)
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = "yuv420p"
    for frame in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"wrote {args.video} ({len(frames)} frames @ {CONTROL_HZ} fps, full res)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
