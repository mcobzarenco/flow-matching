"""Closed-loop bijou rollout(s) in the SO-101 sim (the sim twin of
bijou/rollout.py).

Same inference path as the physical rollouts — BijouPolicy + the real
``observation_to_item`` — with SO101Sim in place of SOFollower. The sim
models a TOP and a WRIST camera and names them exactly that (kind tags
derive from the names via the shared rollout helper, no privileged
mapping; that the rig teleop dataset recorded its top view under a
"front" key is a data-side mislabel to fix in the data, not here).

Runs one episode per seed (policy loaded once), writes a full-resolution
side-by-side (top|wrist) H.264 video per seed to
outputs/sim/rollout_seed<NNN>.mp4, and prints a per-seed summary table:
initial/min/final benchy->disk distance and success.

Usage:
  MUJOCO_GL=egl uv run python -m sim.rollout_sim \
      --checkpoint outputs/train/bijou_ft_rig_from_adarms100k_ddp2/step_005000 \
      --seed 0 --num-seeds 20
"""

import argparse
import dataclasses
import json
import math
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import av
import numpy as np
import torch

from bijou.data import DatasetStats
from bijou.decoders.flow import SamplingMethod
from bijou.eval.policies import BijouPolicy
from bijou.rollout import SO_MOTORS, observation_to_item
from bijou.rollout_safety import camera_kinds_from_names

from . import OUTPUT_DIR
from .so101_sim import CONTROL_HZ, SimObservation, SO101Sim

# The sim's cameras, named for what they ARE (both names sit inside the
# semantic kind vocabulary, so the tags follow for free).
SIM_CAMERAS = ("top", "wrist")

STATS_REPO_ID = "mcobzarenco/so101_pick_place_v2"
TASK = "Pick up the toy boat and place it on the wooden disk."


class RolloutSim(Protocol):
    """What the episode loop needs from an environment — SO101Sim in
    production; the parallel-harness oracle substitutes a pure-python
    fake to pin scheduler equivalence without a GL context."""

    reset_spawn_xy: tuple[float, float]
    reset_strike_contacts: int

    def reset(self, seed: int) -> SimObservation: ...
    def step(self, action_degrees: np.ndarray) -> SimObservation: ...
    def benchy_disk_distance(self) -> float: ...
    def success(self) -> bool: ...
    def benchy_pose(self) -> tuple[np.ndarray, float]: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--hold",
        action="store_true",
        help="no policy: command the settled reset state every tick "
        "(the sim analog of state-copy — metric floor + reset-artifact "
        "anchor)",
    )
    parser.add_argument("--seed", type=int, default=0, help="first env + noise seed")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="episodes: seed .. seed+N-1",
    )
    parser.add_argument("--replans", type=int, default=None)
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=None,
        help="episode TIME budget; the replan count derives from the "
        "resolved chunk horizon at 30 Hz, so 30 seconds means 30 "
        "seconds for any checkpoint's chunk length (a fixed --replans "
        "count quietly scales the budget with chunk size: 15 replans "
        "of 1-second molmoact2 chunks was 15 s, not the intended 30)",
    )
    parser.add_argument("--execute-horizon", type=int, default=30)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument(
        "--draws",
        type=int,
        default=1,
        help="stochastic rollouts per seed (GRPO signal probe): draw 0 "
        "is the deterministic banked-identity row and the only one that "
        "records video; draws >= 1 re-key every policy noise/sampling "
        "stream per draw (flow fresh-noise groups need no other flag; "
        "AR groups also want --ar-temperature)",
    )
    parser.add_argument(
        "--ar-temperature",
        type=float,
        default=None,
        help="sample the AR head at this temperature instead of the "
        "greedy deployment decode (BijouPolicy knob; the row/report "
        "name carries _t<T>)",
    )
    parser.add_argument(
        "--sde-noise-level",
        type=float,
        default=None,
        help="decode flow actions with the Euler–Maruyama SDE at this "
        "noise scale a instead of the deterministic ODE (GRPO probe "
        "cell 5; requires --method euler; the row/report name carries "
        "_sde<a>; per-step noise is keyed per (seed, replan, draw) so "
        "rows stay batch-invariant and reproducible)",
    )
    parser.add_argument(
        "--method",
        default="heun",
        choices=["euler", "heun"],
        help="ODE solver: heun for full-flow checkpoints, euler for "
        "1-NFE SnapFlow students (euler-1 IS their training target)",
    )
    parser.add_argument(
        "--expert-dtype",
        default="bfloat16",
        choices=["float32", "bfloat16"],
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="write a config header + per-seed rows (incl. per-tick "
        "distance series) for the reads instrument",
    )
    args = parser.parse_args()
    if (args.checkpoint is None) == (not args.hold):
        parser.error("exactly one of --checkpoint / --hold is required")
    if args.replans is not None and args.episode_seconds is not None:
        parser.error(
            "--replans and --episode-seconds state the same budget in "
            "two units — pick one",
        )
    if args.episode_seconds is not None and args.episode_seconds <= 0:
        parser.error(f"--episode-seconds must be > 0, got {args.episode_seconds}")
    if args.draws < 1:
        parser.error(f"--draws must be >= 1, got {args.draws}")
    if args.draws > 1 and args.hold:
        parser.error("--draws > 1 is meaningless for --hold (deterministic)")
    if args.sde_noise_level is not None:
        if args.hold:
            parser.error("--sde-noise-level decodes a policy — meaningless with --hold")
        if args.ar_temperature is not None:
            parser.error(
                "--sde-noise-level and --ar-temperature sample "
                "different decoder families — pick one",
            )
        if args.method != "euler":
            parser.error("the SDE decode is Euler-only — pass --method euler")
    return args


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    seed: int
    initial_cm: float
    min_cm: float
    final_cm: float
    success_tick: int | None
    spawn_xy: tuple[float, float]
    reset_strikes: int
    final_z_mm: float
    final_upright: float
    ticks: int
    latency_ms: list[float] = field(default_factory=list)
    distance_cm: list[float] = field(default_factory=list)
    # Stochastic-group draw index (GRPO signal probe): draw 0 is THE
    # deterministic row (identity triple untouched, bit-comparable with
    # every banked run); draws >= 1 re-key the policy's noise/sampling
    # streams. Defaulted so pre-probe rows and JSONs load unchanged.
    draw: int = 0

    @property
    def progress_cm(self) -> float:
        """Distance recovered from spawn to the episode's closest point."""
        return self.initial_cm - self.min_cm

    @property
    def progress_final_cm(self) -> float:
        """PRIMARY metric: distance recovered from spawn to episode end."""
        return self.initial_cm - self.final_cm


def resolve_replans(
    replans: int | None,
    episode_seconds: float | None,
    horizon: int,
) -> int:
    """The replan count for one episode, resolved AFTER the horizon is
    (min(execute_horizon, chunk_size) needs the loaded policy). Default
    stays the historical 15; --episode-seconds converts a TIME budget at
    CONTROL_HZ so short-chunk policies get the same seconds as
    long-chunk ones."""
    if episode_seconds is not None:
        return max(1, math.ceil(episode_seconds * CONTROL_HZ / horizon))
    return 15 if replans is None else replans


def to_observation(obs: SimObservation) -> dict[str, object]:
    """SimObservation -> the robot-observation dict observation_to_item
    consumes: <motor>.pos floats + named HWC uint8 camera frames."""
    observation: dict[str, object] = {
        f"{motor}.pos": float(obs.state[index]) for index, motor in enumerate(SO_MOTORS)
    }
    top, wrist = SIM_CAMERAS
    observation[top] = obs.top
    observation[wrist] = obs.wrist
    return observation


class VideoWriter:
    """Streaming H.264 writer: frames are encoded as they arrive (a full
    900-tick episode buffered at 640x960 is ~1.6 GB RSS — untenable when
    N env workers each hold one)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._container: Any = None
        self._stream: Any = None

    def append(self, frame: np.ndarray) -> None:
        if self._container is None:
            self._container = av.open(str(self._path), mode="w")
            self._stream = self._container.add_stream("h264", rate=CONTROL_HZ)
            self._stream.width = frame.shape[1]
            self._stream.height = frame.shape[0]
            self._stream.pix_fmt = "yuv420p"
        for packet in self._stream.encode(
            av.VideoFrame.from_ndarray(frame, format="rgb24"),
        ):
            self._container.mux(packet)

    def close(self) -> None:
        if self._container is None:
            return
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()


def run_episode_loop(
    sim: RolloutSim,
    seed: int,
    next_chunk: Callable[[SimObservation, int], np.ndarray],
    *,
    replans: int,
    horizon: int,
    video_path: Path | None,
    latencies: list[float],
) -> EpisodeResult:
    """One episode's control loop, with the chunk source abstracted:
    ``next_chunk(obs, replan)`` is the policy call in the sequential
    driver and the predict round-trip in the parallel one (it appends its
    own timing to ``latencies``). The parallel-vs-sequential determinism
    oracle rides on both drivers sharing this exact loop."""
    obs = sim.reset(seed)
    writer = VideoWriter(video_path) if video_path is not None else None
    initial = sim.benchy_disk_distance()
    closest = initial
    success_tick: int | None = None
    distances: list[float] = [initial * 100]
    ticks = 0

    for replan in range(replans):
        chunk = next_chunk(obs, replan)
        print(
            f"  seed {seed} replan {replan}: "
            f"{latencies[-1] if latencies else 0:.0f} ms | "
            f"benchy->disk {sim.benchy_disk_distance() * 100:.1f} cm",
            flush=True,
        )
        for step in range(horizon):
            obs = sim.step(chunk[step])
            if writer is not None:
                writer.append(np.concatenate([obs.top, obs.wrist], axis=1))
            ticks += 1
            distance = sim.benchy_disk_distance()
            distances.append(distance * 100)
            closest = min(closest, distance)
            if sim.success():
                success_tick = replan * horizon + step
                break
        if success_tick is not None:
            break

    if writer is not None:
        writer.close()
    pos, upright = sim.benchy_pose()
    return EpisodeResult(
        seed=seed,
        initial_cm=initial * 100,
        min_cm=closest * 100,
        final_cm=sim.benchy_disk_distance() * 100,
        success_tick=success_tick,
        spawn_xy=sim.reset_spawn_xy,
        reset_strikes=sim.reset_strike_contacts,
        final_z_mm=float(pos[2]) * 1000,
        final_upright=upright,
        ticks=ticks,
        latency_ms=[round(v, 1) for v in latencies],
        distance_cm=[round(v, 3) for v in distances],
    )


def hold_chunk_fn(horizon: int) -> Callable[[SimObservation, int], np.ndarray]:
    """Hold arm: command the settled reset state for every tick (the sim
    analog of state-copy). The chunk is tiled lazily from the first
    observation the loop hands over — which is the reset observation."""
    hold_action: np.ndarray | None = None

    def next_chunk(obs: SimObservation, replan: int) -> np.ndarray:
        nonlocal hold_action
        if hold_action is None:
            hold_action = np.tile(obs.state.copy(), (horizon, 1))
        return hold_action

    return next_chunk


def sim_item(
    obs: SimObservation,
    seed: int,
    replan: int,
    *,
    stats: DatasetStats,
    chunk_size: int,
    draw: int = 0,
) -> dict[str, object]:
    """Observation -> policy item, with the identity triple for stable-key
    noise checkpoints (the SnapFlow lineage): deterministic per
    (env seed, replan), invariant to batch composition — the property the
    parallel driver's determinism claim leans on."""
    item = observation_to_item(
        to_observation(obs),
        TASK,
        stats=stats,
        chunk_size=chunk_size,
        camera_kinds=camera_kinds_from_names(SIM_CAMERAS),
    )
    # Draw keying (GRPO signal probe): every policy-side stochastic
    # stream — stable-key flow noise AND the AR sample RNG — derives
    # from this triple, so a draw-suffixed repo_id makes the whole
    # episode's streams draw-distinct with zero policy-side changes,
    # while draw 0 keeps the banked identity BIT-EXACTLY.
    item["repo_id"] = "sim/eval100" if draw == 0 else f"sim/eval100/draw{draw:02d}"
    item["episode_index"] = seed
    item["frame_index"] = replan
    return item


def run_episode(
    policy: BijouPolicy | None,
    sim: SO101Sim,
    seed: int,
    *,
    replans: int,
    horizon: int,
    video_path: Path | None,
    draw: int = 0,
) -> EpisodeResult:
    latencies: list[float] = []
    next_chunk: Callable[[SimObservation, int], np.ndarray]

    if policy is not None:
        chunk_size = policy.info.chunk_size
        # Converted checkpoints (molmoact2 lineage) carry no per-dataset
        # table — their items must wear the checkpoint's MERGED stats
        # (the exact normalization the model trained with; its state
        # binning already rides BijouPolicy's molmo_flow state table).
        stats = policy.info.per_dataset_normalization.get(
            STATS_REPO_ID,
            policy.info.normalization,
        )

        def policy_chunk(obs: SimObservation, replan: int) -> np.ndarray:
            item = sim_item(
                obs,
                seed,
                replan,
                stats=stats,
                chunk_size=chunk_size,
                draw=draw,
            )
            start = time.perf_counter()
            chunk = policy.predict([item], [replan])[0].numpy()
            latencies.append((time.perf_counter() - start) * 1000)
            return chunk

        next_chunk = policy_chunk
    else:
        next_chunk = hold_chunk_fn(horizon)

    row = run_episode_loop(
        sim,
        seed,
        next_chunk,
        replans=replans,
        horizon=horizon,
        video_path=video_path,
        latencies=latencies,
    )
    return dataclasses.replace(row, draw=draw) if draw else row


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.hold:
        policy = None
        horizon = args.execute_horizon
        print(f"policy: hold (settled reset state, horizon {horizon})")
    else:
        policy = BijouPolicy(
            args.checkpoint,
            device=device,
            seed=args.seed,
            sample_steps=args.sample_steps,
            method=SamplingMethod[args.method.upper()],
            ar_temperature=args.ar_temperature,
            sde_noise_level=args.sde_noise_level,
            expert_dtype=getattr(torch, args.expert_dtype),
        )
        horizon = min(args.execute_horizon, policy.info.chunk_size)
        print(
            f"policy: {policy.name} "
            f"({args.method}-{args.sample_steps}, horizon {horizon})",
        )
    replans = resolve_replans(args.replans, args.episode_seconds, horizon)
    print(
        f"episode budget: {replans} replans x {horizon} ticks = "
        f"{replans * horizon / CONTROL_HZ:.1f} s at {CONTROL_HZ} Hz",
    )

    sim = SO101Sim()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[EpisodeResult] = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        for draw in range(args.draws):
            video_path = (
                args.out_dir / f"rollout_seed{seed:03d}.mp4" if draw == 0 else None
            )
            results.append(
                run_episode(
                    policy,
                    sim,
                    seed,
                    replans=replans,
                    horizon=horizon,
                    video_path=video_path,
                    draw=draw,
                ),
            )

    print("\nseed | draw | init cm | min cm | final cm | progress cm | success")
    for r in sorted(results, key=lambda r: -r.progress_cm):
        success = f"tick {r.success_tick}" if r.success_tick is not None else "-"
        print(
            f"{r.seed:4d} | {r.draw:4d} | {r.initial_cm:7.1f} | {r.min_cm:6.1f} | "
            f"{r.final_cm:8.1f} | {r.progress_cm:11.1f} | {success}",
        )
    best = max(results, key=lambda r: r.progress_cm)
    print(
        f"\nbest seed by progress: {best.seed} (video rollout_seed{best.seed:03d}.mp4)",
    )

    if args.out_json is not None:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        payload = {
            "config": {
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "hold": args.hold,
                "seed": args.seed,
                "num_seeds": args.num_seeds,
                "replans": replans,
                "episode_seconds": args.episode_seconds,
                "execute_horizon": horizon,
                "sample_steps": args.sample_steps,
                "method": args.method,
                "draws": args.draws,
                "ar_temperature": args.ar_temperature,
                "sde_noise_level": args.sde_noise_level,
                "expert_dtype": args.expert_dtype,
                "control_hz": CONTROL_HZ,
                "task": TASK,
                "stats_repo_id": STATS_REPO_ID,
                "commit": commit,
            },
            "episodes": [
                {**asdict(r), "progress_final_cm": r.progress_final_cm} for r in results
            ],
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
