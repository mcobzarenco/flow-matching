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
--camera front=... --camera wrist=...). Camera *kinds* (the semantic prompt
tags) mirror training: the rig dataset's stamped ``meta/camera_kinds.json``
when ``--stats-dataset`` is given, ``--camera-kind name=kind`` to override.

Safety gates before the arm moves (``bijou.rollout_safety``):
``--max-relative-target`` is mandatory (``--unclamped`` opts out,
explicitly), and the first observation must lie inside the rig stats'
per-joint envelope (``--skip-envelope-check`` opts out).

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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

from .annotations import ConditionField
from .aux_text import AuxField, AuxGeneration
from .data import DatasetStats
from .eval.policies import BijouPolicy
from .model import SamplingMethod
from .rollout_async import AsyncExecutor, AsyncPlanner, sustainable
from .rollout_safety import (
    envelope_violations,
    parse_camera_kind_overrides,
    require_clamp,
    resolve_camera_kinds,
    state_envelope,
)

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
    parser.add_argument(
        "--control-fps",
        type=int,
        default=30,
        help="control-loop rate (actions/s). The chunks are 30 Hz data: "
        "lower values time-dilate execution — a deliberate trade for "
        "async sustainability on slow GPUs, not a free parameter",
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=30,
        help="camera capture rate — independent of --control-fps (cameras often "
        "support only their native rates; observations are snapshots of "
        "the latest frame, so capture may run faster than control)",
    )
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=40,
        help="actions executed per predicted chunk before replanning "
        "(< chunk size leaves reaction headroom)",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="seconds")
    parser.add_argument(
        "--async-inference",
        action="store_true",
        help="overlap planning with execution: infer the next chunk "
        "while the current one's tail executes, switch at the horizon "
        "boundary with a skip-ahead — removes the per-replan freeze "
        "(~625 ms on the 8 GiB laptop) at identical replan cadence",
    )
    parser.add_argument(
        "--trigger-margin-ticks",
        type=int,
        default=3,
        help="async: extra control ticks of slack on top of measured "
        "p95 inference latency when scheduling the next plan",
    )
    parser.add_argument(
        "--switch-blend",
        type=int,
        default=0,
        help="async: crossfade this many ticks between the outgoing and "
        "incoming chunk at a switch (0 = hard switch; "
        "max_relative_target stays the hard safety clamp)",
    )
    parser.add_argument("--sample-steps", type=int, default=5)
    parser.add_argument(
        "--sample-method",
        choices=[m.value for m in SamplingMethod],
        default=SamplingMethod.HEUN.value,
    )
    parser.add_argument(
        "--target-time",
        choices=["t", "zero"],
        default="t",
        help="flow target-time conditioning s (SnapFlow φ_s checkpoints "
        "only): 't' = standard s=t forwards; 'zero' = the 1-NFE endpoint "
        "decode (pair with --sample-steps 1 --sample-method euler); "
        "loud, never inferred from step count — mirrors bijou.eval",
    )
    parser.add_argument(
        "--generate",
        nargs="*",
        choices=[f.value for f in AuxField],
        default=None,
        help="ar_backbone request set: fields to elicit before each chunk "
        "(template order; 'actions' implicit and terminal; aux-trained "
        "checkpoints only; ~1 extra suffix forward per requested field "
        "plus its value tokens per replan). Omit for the fast path",
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
        "--subgoal",
        default=None,
        help="subgoal conditioning at deployment (free text, e.g. from a "
        "planner): rendered only by subgoal-condition-trained "
        "checkpoints; omit to run the (well-trained) unconditioned "
        "context",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="safety clamp on per-tick joint motion in degrees (lerobot "
        "SOFollower feature) — MANDATORY: it is the only limiter between "
        "a bad chunk and full-speed arbitrary servo motion; start with "
        "e.g. 20. --unclamped is the explicit opt-out",
    )
    parser.add_argument(
        "--unclamped",
        action="store_true",
        help="explicitly run without --max-relative-target (nothing limits "
        "per-tick joint motion — not for first runs)",
    )
    parser.add_argument(
        "--camera-kind",
        action="append",
        default=[],
        metavar="NAME=KIND",
        help="repeatable; override a camera's semantic kind in the prompt "
        "(the training-time tag). Default resolution mirrors training: "
        "the rig dataset's stamped meta/camera_kinds.json when "
        "--stats-dataset is given, else the name-is-kind heuristic",
    )
    parser.add_argument(
        "--skip-envelope-check",
        action="store_true",
        help="proceed even when the first observation falls outside the "
        "stats envelope (deliberately unusual start pose only — the check "
        "exists to catch wrong stats and ticks-vs-degrees mismatches)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--offload-ple",
        action="store_true",
        help="park the backbone's per-layer-embedding token table in "
        "host RAM (4.7 GB of the full-depth ar_backbone's 9.6 GB bf16 "
        "weights, lookup-only at inference) — fits ≤8 GiB GPUs at "
        "negligible latency cost",
    )
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
        # item["camera_kinds"] — see rollout_safety.resolve_camera_kinds).
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


def print_generation(
    generations: list[AuxGeneration] | None,
    requested: tuple[AuxField, ...],
) -> None:
    """One stdout line per replan with the decode's aux field values
    (parsed form; the raw bytes follow when a lenient parse failed —
    they are the ground truth)."""
    if generations is None or not requested:
        return
    generation = generations[0]
    parts: list[str] = []
    unparsed = False
    for aux_field in requested:
        value: str | bool | float | None = getattr(generation, aux_field.value)
        if value is None:
            unparsed = True
        parts.append(
            f"{aux_field.value}: {value if value is not None else '<unparsed>'}",
        )
    print(f"  aux | {' | '.join(parts)}", flush=True)
    if unparsed:
        raw = generation.text.strip().replace("\n", " \\n ")
        print(f"  aux (raw) | {raw}", flush=True)


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    # Safety gates fail before the (slow) policy load; the clamp gate
    # runs in --check mode too, so checking the exact command you will
    # run catches a missing clamp early.
    require_clamp(args.max_relative_target, unclamped=args.unclamped)
    camera_names = [spec.partition("=")[0] for spec in args.camera]
    rig_kinds = resolve_camera_kinds(
        camera_names,
        parse_camera_kind_overrides(args.camera_kind, camera_names),
        args.stats_dataset,
    )
    condition_values = {
        ConditionField.OUTCOME.value: args.outcome,
        ConditionField.SMOOTHNESS.value: args.smoothness,
    }
    if args.subgoal is not None:
        condition_values[ConditionField.SUBGOAL.value] = args.subgoal

    policy = BijouPolicy(
        args.checkpoint,
        device=device,
        seed=args.seed if args.seed is not None else int(time.time()),
        sample_steps=args.sample_steps,
        method=SamplingMethod(args.sample_method),
        # Stable noise keying is an eval instrument — it keys draws by
        # dataset identity (repo_id/episode/frame), which a live rig
        # observation does not have. Index keying with the replan counter
        # as the index IS the deployment semantics: fresh noise per
        # replan, reproducible under a fixed --seed.
        noise_key="index",
        target_time=0.0 if args.target_time == "zero" else None,
        expert_dtype=getattr(torch, args.expert_dtype),
        generate=tuple(AuxField(f) for f in (args.generate or ())),
        include_subgoal_condition=args.subgoal is not None,
        offload_ple=args.offload_ple,
    )
    stats = rig_stats(args, policy)
    envelope = state_envelope(stats, expected_dim=len(SO_MOTORS))
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
            fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
        )

    decode_tag = f"{args.sample_method}-{args.sample_steps}"
    if args.target_time == "zero":
        decode_tag += "-s0"  # 1-NFE endpoint decode (SnapFlow shortcut)
    print(
        f"policy: {policy.name} (chunk {chunk_size}, "
        f"{decode_tag}, {args.expert_dtype} expert)",
    )
    print(f"task: {args.task!r}")
    print(f"cameras (prompt order): {sorted(cameras)}")
    print(f"camera kinds (prompt tags): {rig_kinds}")
    print(f"state stats mean: {[round(x, 1) for x in stats.state_mean]}")
    print(
        "first-obs envelope: "
        + ", ".join(
            f"{motor} [{lo:.1f}, {hi:.1f}]"
            for motor, lo, hi in zip(SO_MOTORS, *envelope, strict=True)
        ),
    )
    print(
        f"loop: {args.control_fps} Hz, execute {horizon}/{chunk_size} per replan, "
        f"{args.duration:.0f}s, max_relative_target={args.max_relative_target}",
    )
    if args.check:
        # Exercise the FULL inference path on a synthetic observation
        # (prompt collation with conditioning + [generate|…], prefix
        # encode, decode, aux parse) — everything except the robot. A
        # checkpoint/flag combination that would die mid-rollout dies
        # here instead.
        print("check mode: one synthetic-observation predict, no robot")
        synthetic: dict[str, Any] = {f"{m}.pos": 0.0 for m in SO_MOTORS}
        for name in cameras:
            synthetic[name] = np.zeros(
                (args.camera_height, args.camera_width, 3),
                dtype=np.uint8,
            )
        item = observation_to_item(
            synthetic,
            args.task,
            stats,
            chunk_size,
            rig_kinds,
            condition_values,
        )
        start = time.perf_counter()
        chunks, generations = policy.predict_with_text([item], [0])
        latency = time.perf_counter() - start
        print(
            f"predict ok: chunk {tuple(chunks[0].shape)} in {latency * 1000:.0f} ms",
            flush=True,
        )
        print_generation(generations, policy.generate)
        if args.async_inference:
            # Async dry-run: measure the warm path and report the slack
            # arithmetic the trigger will run on — a starvation-prone
            # configuration is visible here, before the arm moves.
            planner = AsyncPlanner(
                lambda one, index: policy.predict_with_text([one], [index]),
            )
            planner.warmup(item, replan_index=0)
            latency_ticks = planner.latency_ticks(args.control_fps)
            trigger_at = horizon - latency_ticks - args.trigger_margin_ticks
            fine = sustainable(chunk_size, latency_ticks, args.trigger_margin_ticks)
            print(
                f"async: warm latency {latency_ticks} ticks @ {args.control_fps} Hz "
                f"(+{args.trigger_margin_ticks} margin) → trigger at action "
                f"{max(trigger_at, 0)}/{horizon} — "
                + (
                    "SUSTAINABLE (2·latency + margin ≤ chunk)"
                    if fine
                    else (
                        "UNSUSTAINABLE: each plan arrives one latency "
                        f"stale, leaving {max(chunk_size - latency_ticks, 0)} "
                        f"fresh rows per {latency_ticks}-tick cycle — the "
                        "loop WILL starve. Use sync mode, lower --control-fps, or "
                        "shrink latency"
                    )
                ),
                flush=True,
            )
            planner.shutdown()
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

    # First-observation envelope gate: no action is sent until the arm's
    # reported state is plausible under the rig stats it will be
    # normalized with.
    first_state = [float(robot.get_observation()[f"{m}.pos"]) for m in SO_MOTORS]
    out = envelope_violations(first_state, envelope)
    lo, hi = envelope
    for j, motor in enumerate(SO_MOTORS):
        print(
            f"  first-obs | {motor}: {first_state[j]:.1f} in "
            f"[{lo[j]:.1f}, {hi[j]:.1f}] " + ("OUT" if j in out else "ok"),
            flush=True,
        )
    if out:
        joints = ", ".join(SO_MOTORS[j] for j in out)
        if args.skip_envelope_check:
            print(
                f"WARNING: first observation outside the stats envelope "
                f"({joints}) — proceeding under --skip-envelope-check",
                flush=True,
            )
        else:
            robot.disconnect()
            raise SystemExit(
                f"first observation outside the stats envelope ({joints}): "
                "wrong --stats-repo-id/--stats-dataset, a ticks-vs-degrees "
                "mismatch, or an uncalibrated arm. Fix the stats source, or "
                "pass --skip-envelope-check for a deliberately unusual "
                "start pose.",
            )

    tick = 1.0 / args.control_fps
    deadline = time.perf_counter() + args.duration
    try:
        if args.async_inference:
            run_async_loop(
                args,
                robot,
                policy,
                stats,
                rig_kinds,
                condition_values,
                chunk_size=chunk_size,
                horizon=horizon,
                tick=tick,
                deadline=deadline,
            )
        else:
            run_sync_loop(
                args,
                robot,
                policy,
                stats,
                rig_kinds,
                condition_values,
                chunk_size=chunk_size,
                horizon=horizon,
                tick=tick,
                deadline=deadline,
            )
    except KeyboardInterrupt:
        print("\nstopping (keyboard interrupt)")
    finally:
        robot.disconnect()
        print("robot disconnected")
    return 0


def run_sync_loop(
    args: argparse.Namespace,
    robot: SOFollower,
    policy: BijouPolicy,
    stats: DatasetStats,
    rig_kinds: dict[str, str],
    condition_values: dict[str, str],
    *,
    chunk_size: int,
    horizon: int,
    tick: float,
    deadline: float,
) -> None:
    """The original serial loop: observe → predict (arm frozen) →
    execute horizon actions → repeat. Byte-identical behavior to the
    pre-async rollout."""
    replans = 0
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
        chunks, generations = policy.predict_with_text([item], [replans])
        chunk = chunks[0]
        latency = time.perf_counter() - start
        replans += 1
        print(
            f"replan {replans}: {latency * 1000:.0f} ms | state "
            f"{[round(float(x), 1) for x in item['observation.state']]}",
            flush=True,
        )
        print_generation(generations, policy.generate)
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


def run_async_loop(
    args: argparse.Namespace,
    robot: SOFollower,
    policy: BijouPolicy,
    stats: DatasetStats,
    rig_kinds: dict[str, str],
    condition_values: dict[str, str],
    *,
    chunk_size: int,
    horizon: int,
    tick: float,
    deadline: float,
) -> None:
    """Overlapped loop: one action per tick, planning in the background,
    switches at the horizon boundary with skip-ahead (design:
    rollout_async module docstring)."""

    def snapshot() -> dict[str, Any]:
        return observation_to_item(
            robot.get_observation(),
            args.task,
            stats,
            chunk_size,
            rig_kinds,
            condition_values,
        )

    planner = AsyncPlanner(
        lambda one, index: policy.predict_with_text([one], [index]),
    )
    executor = AsyncExecutor(
        chunk_size=chunk_size,
        execute_horizon=horizon,
        fps=args.control_fps,
        margin_ticks=args.trigger_margin_ticks,
        blend_ticks=args.switch_blend,
    )
    try:
        # Cold start: synchronous first plan (arm idle, staleness-free)
        # + kernel warmup, before the first tick.
        first = planner.warmup(snapshot(), replan_index=0)
        latency_ticks = planner.latency_ticks(args.control_fps)
        if not sustainable(chunk_size, latency_ticks, args.trigger_margin_ticks):
            raise SystemExit(
                f"async refused: measured latency {latency_ticks} ticks at "
                f"{args.control_fps} Hz needs 2·{latency_ticks}+"
                f"{args.trigger_margin_ticks} ≤ chunk {chunk_size} — each "
                "plan would arrive one latency stale and the loop would "
                "starve into hold-lunge cycles (field-tested 2026-08-05). "
                "Run sync mode, lower --control-fps, or shrink latency "
                "(power profile, batch-free GPU).",
            )
        executor.start(first)
        print(
            f"async: warm latency {latency_ticks} ticks "
            f"(+{args.trigger_margin_ticks} margin) — sustainable",
            flush=True,
        )
        print_generation(first.generations, policy.generate)
        replans = 1
        held_before = 0
        next_tick = time.perf_counter()
        while time.perf_counter() < deadline:
            if executor.wants_plan(
                planner.latency_ticks(args.control_fps),
                in_flight=planner.in_flight,
            ):
                planner.submit(snapshot(), replans)
                executor.note_submit()
                replans += 1
            executor.offer(planner.poll())
            adopted = executor.maybe_switch(time.perf_counter())
            if adopted is not None:
                held = executor.held_ticks - held_before
                held_before = executor.held_ticks
                print(
                    f"switch {executor.switches}: replan "
                    f"{adopted.replan_index} → entering its chunk at row "
                    f"{executor.last_skip_ahead}/{chunk_size} (its "
                    f"observation is {executor.last_skip_ahead} ticks old)"
                    + (f" after {held} HELD ticks" if held > 0 else "")
                    + (
                        f" (wall-vs-executed skew {executor.last_staleness_skew} ticks)"
                        if executor.last_staleness_skew > 2
                        else ""
                    ),
                    flush=True,
                )
                print_generation(adopted.generations, policy.generate)
            action_row, starved = executor.next_action()
            if starved and executor.held_ticks == held_before + 1:
                print(
                    "STARVED: chunk exhausted before the next plan "
                    "arrived — holding position (inference too slow for "
                    "horizon + tail at this fps)",
                    flush=True,
                )
            action = {
                f"{motor}.pos": float(action_row[j])
                for j, motor in enumerate(SO_MOTORS)
            }
            robot.send_action(action)
            next_tick += tick
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        print(
            f"async summary: {executor.switches} switches, "
            f"{executor.held_ticks} held ticks, last skip-ahead "
            f"{executor.last_skip_ahead}",
            flush=True,
        )
    finally:
        planner.shutdown()


if __name__ == "__main__":
    sys.exit(main())
