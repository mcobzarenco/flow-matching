"""Closed-loop rollout of a bijou checkpoint on a physical SO-100/SO-101 arm.

Uses lerobot's robot layer (SOFollower + cameras) for hardware I/O and the
SAME inference path as ``bijou.eval`` (``BijouPolicy``: prompt collation,
prefix encode, flow sampling, per-rig unnormalization) — what you scored
offline is what drives the arm.

The control loop predicts a 50-action chunk from the current observation,
executes ``--execute-horizon`` of it at the control rate, then replans from
a fresh observation. Normalization stats for the rig come from the
checkpoint's per-dataset table (``--stats-repo-id``, present when the
checkpoint was fine-tuned on this rig's data) or directly from a local
dataset directory (``--stats-dataset``).

Camera naming matters: prompt slots are positional over SORTED camera keys,
so the ``--camera`` names must sort the same way as the training dataset's
camera keys (e.g. a dataset recorded with front/wrist must roll out with
--camera front=... --camera wrist=...).

Usage::

    uv run python -m bijou.rollout \
        --checkpoint outputs/train/bijou_ft_marius_2k/step_002000 \
        --stats-repo-id mcobzarenco/so101_pick_place_clean \
        --port /dev/ttyACM0 --robot-id my_follower \
        --camera front=/dev/video0 --camera wrist=/dev/video2 \
        --task "Pick up the cube and place it in the box" \
        --max-relative-target 20 --duration 60

    # Verify everything except the robot (loads policy, prints the plan):
    ... --check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

from .annotations import CAMERA_KINDS, ConditionField
from .aux_text import AuxDecodeMode
from .data import DatasetStats
from .eval.policies import BijouPolicy
from .model import SamplingMethod

# Canonical SO-100/101 joint order (bus order; dataset motor names are these
# with the ".pos" suffix).
SO_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument(
        "--robot-id",
        default=None,
        help="lerobot robot id (selects the calibration file)",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="NAME=INDEX_OR_PATH",
        help="repeatable; NAMEs must sort like the training dataset's camera "
        "keys (positional prompt slots)",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--task", required=True, help="language instruction")
    parser.add_argument(
        "--stats-repo-id",
        default=None,
        help="rig stats from the checkpoint's per-dataset table (repo id)",
    )
    parser.add_argument(
        "--stats-dataset",
        type=Path,
        default=None,
        help="rig stats from a local LeRobot dataset directory (meta/stats.json)",
    )
    parser.add_argument("--fps", type=int, default=30, help="control rate")
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=40,
        help="actions executed per predicted chunk before replanning "
        "(< chunk size leaves reaction headroom)",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="seconds")
    parser.add_argument("--sample-steps", type=int, default=5)
    parser.add_argument(
        "--sample-method",
        choices=[m.value for m in SamplingMethod],
        default=SamplingMethod.HEUN.value,
    )
    parser.add_argument(
        "--aux-mode",
        choices=[m.value for m in AuxDecodeMode],
        default=AuxDecodeMode.ACT.value,
        help="ar_backbone decode mode: 'act' = actions only (fast path), "
        "'free' = generate the aux text (subgoal etc.) before each chunk "
        "(aux-trained checkpoints only; costs ~30-45 extra suffix "
        "forwards per replan)",
    )
    parser.add_argument(
        "--outcome",
        choices=["success", "partial", "failure"],
        default="success",
        help="outcome conditioning at deployment (condition-trained "
        "checkpoints render it; others ignore it): ask for the behavior "
        "you want",
    )
    parser.add_argument(
        "--smoothness",
        choices=["high", "medium", "low"],
        default="high",
        help="smoothness conditioning at deployment (as --outcome)",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="safety clamp on per-tick joint motion (lerobot SOFollower "
        "feature; strongly recommended for first runs, e.g. 20)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--expert-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="bfloat16 halves expert memory for small inference GPUs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="noise seed (default: stochastic)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="load the policy and stats, print the plan, exit without "
        "touching the robot",
    )
    return parser.parse_args()


def rig_stats(args: argparse.Namespace, policy: BijouPolicy) -> DatasetStats:
    """Per-rig MEAN_STD stats, from the checkpoint table (already floored at
    training time) or a local dataset directory (floored on load, matching
    training's construction path)."""
    if args.stats_repo_id is not None:
        table = policy.info.per_dataset_normalization
        if args.stats_repo_id not in table:
            raise SystemExit(
                f"{args.stats_repo_id!r} not in the checkpoint's stats table "
                f"({len(table)} entries); pass --stats-dataset instead",
            )
        return table[args.stats_repo_id]
    if args.stats_dataset is not None:
        stats = json.loads((args.stats_dataset / "meta" / "stats.json").read_text())
        return DatasetStats.from_lerobot_stats(stats)
    raise SystemExit("pass --stats-repo-id or --stats-dataset")


def camera_kinds_from_names(names: Iterable[str]) -> dict[str, str]:
    """Per-camera semantic kinds from the operator's own camera names: a
    name inside the judge vocabulary IS its kind; anything else renders
    "unknown" (trained in-distribution via kind dropout) with a LOUD
    warning — name cameras by viewpoint to give the model the signal."""
    kinds: dict[str, str] = {}
    for name in names:
        if name in CAMERA_KINDS:
            kinds[name] = name
        else:
            print(
                f"WARNING: camera name {name!r} is not in the semantic "
                f"kind vocabulary {sorted(CAMERA_KINDS)} — its prompt tag "
                "renders as 'unknown'",
                flush=True,
            )
            kinds[name] = "unknown"
    return kinds


def observation_to_item(
    observation: dict[str, Any],
    task: str,
    stats: DatasetStats,
    chunk_size: int,
    camera_kinds: dict[str, str],
    condition_values: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Robot observation -> the item shape BijouPolicy consumes (mirrors a
    StatsAttachedDataset item, including the stats tensors). Ground-truth
    fields are zero stubs: the collator requires them and the policy reads
    only their shapes.

    Produced item shapes (matching the Collator's contract):
    observation.state [state_dim]; action [chunk, action_dim];
    action_is_pad [chunk]; observation.images.* [3, height, width]
    (float, [0, 1], from the camera's HWC uint8 frame). Stats resolved
    from an old checkpoint's tables carry no quantile keys — the flow
    policy never reads them."""
    state = torch.tensor([float(observation[f"{m}.pos"]) for m in SO_MOTORS])
    item: dict[str, Any] = {
        "task": task,
        # Kinds travel with the item, like the stats (the collator reads
        # item["camera_kinds"] — see rollout.camera_kinds_from_names).
        "camera_kinds": camera_kinds,
        # Deployment conditioning values (rendered only for fields the
        # checkpoint trained — the collator's condition_fields gate).
        **{
            f"condition_{field}": value
            for field, value in (condition_values or {}).items()
        },
        "observation.state": state,
        "action": torch.zeros(chunk_size, len(SO_MOTORS)),
        "action_is_pad": torch.zeros(chunk_size, dtype=torch.bool),
        **stats.item_tensors(),
    }
    for key, value in observation.items():
        if key.endswith(".pos"):
            continue
        frame = torch.from_numpy(value)  # HWC uint8
        item[f"observation.images.{key}"] = frame.permute(2, 0, 1).float() / 255.0
    return item


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    rig_kinds = camera_kinds_from_names(spec.partition("=")[0] for spec in args.camera)
    condition_values = {
        ConditionField.OUTCOME.value: args.outcome,
        ConditionField.SMOOTHNESS.value: args.smoothness,
    }

    policy = BijouPolicy(
        args.checkpoint,
        device=device,
        seed=args.seed if args.seed is not None else int(time.time()),
        sample_steps=args.sample_steps,
        method=SamplingMethod(args.sample_method),
        expert_dtype=getattr(torch, args.expert_dtype),
        aux_mode=AuxDecodeMode(args.aux_mode),
    )
    stats = rig_stats(args, policy)
    chunk_size = policy.info.chunk_size
    horizon = min(args.execute_horizon, chunk_size)

    cameras = {}
    for spec in args.camera:
        name, _, source = spec.partition("=")
        if not source:
            raise SystemExit(f"--camera expects NAME=INDEX_OR_PATH, got {spec!r}")
        index_or_path: int | Path = int(source) if source.isdigit() else Path(source)
        cameras[name] = OpenCVCameraConfig(
            index_or_path=index_or_path,
            fps=args.fps,
            width=args.camera_width,
            height=args.camera_height,
        )

    print(
        f"policy: {policy.name} (chunk {chunk_size}, "
        f"{args.sample_method}-{args.sample_steps}, {args.expert_dtype} expert)",
    )
    print(f"task: {args.task!r}")
    print(f"cameras (prompt order): {sorted(cameras)}")
    print(f"state stats mean: {[round(x, 1) for x in stats.state_mean]}")
    print(
        f"loop: {args.fps} Hz, execute {horizon}/{chunk_size} per replan, "
        f"{args.duration:.0f}s, max_relative_target={args.max_relative_target}",
    )
    if args.check:
        print("check mode: not connecting to the robot")
        return 0

    robot = SOFollower(
        SOFollowerRobotConfig(
            port=args.port,
            id=args.robot_id,
            cameras=cameras,
            max_relative_target=args.max_relative_target,
        ),
    )
    robot.connect()
    print("robot connected; ctrl-c to stop", flush=True)

    tick = 1.0 / args.fps
    deadline = time.perf_counter() + args.duration
    replans = 0
    try:
        while time.perf_counter() < deadline:
            observation = robot.get_observation()
            item = observation_to_item(
                observation,
                args.task,
                stats,
                chunk_size,
                rig_kinds,
                condition_values,
            )
            start = time.perf_counter()
            chunk = policy.predict([item], [replans])[0]
            latency = time.perf_counter() - start
            replans += 1
            print(
                f"replan {replans}: {latency * 1000:.0f} ms | state "
                f"{[round(float(x), 1) for x in item['observation.state']]}",
                flush=True,
            )
            next_tick = time.perf_counter()
            for step in range(horizon):
                if time.perf_counter() >= deadline:
                    break
                action = {
                    f"{motor}.pos": float(chunk[step, j])
                    for j, motor in enumerate(SO_MOTORS)
                }
                robot.send_action(action)
                next_tick += tick
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        print("\nstopping (keyboard interrupt)")
    finally:
        robot.disconnect()
        print("robot disconnected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
