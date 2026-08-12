"""Parallel closed-loop sim rollouts: N env workers, ONE batched policy.

The sequential driver (sim.rollout_sim) is render-bound — the H100 idles
through most of every heun-10 decode at batch 1. This driver splits the
work the way the queue item registered it: each worker process owns a
full ``SO101Sim`` (physics + its own GL context; MuJoCo's EGL display is
per-process global state, so envs must not share a process), while the
parent holds the single checkpoint copy and serves batched
``policy.predict`` calls.

Scheduling is deterministic LOCKSTEP ROUNDS: each round, the parent
collects exactly one message stream per still-active worker in
worker-index order until that worker either requests a predict or
finishes its seed slice, then answers all requests with one batched
forward. Batch membership is therefore a pure function of (seed
partition, worker count, policy outputs) — never of wall-clock timing.
Seeds are partitioned round-robin (worker ``w`` gets ``seeds[w::N]``)
and every row carries the same identity triple as the sequential driver
(``repo_id="sim/eval100"``, ``episode_index=seed``,
``frame_index=replan``), so stable-key flow noise is untouched by
batching or by which worker runs a seed.

Determinism contract (pre-registered, posts/
2026-08-12-prereg-sim-parallel-rollouts.md): the CPU-tier oracle in
tests/test_sim_parallel_rollouts.py pins harness equivalence — same
per-seed (obs -> chunk -> step) sequence and rows as the sequential
loop, which both drivers share via ``run_episode_loop``. Whether the
BATCHED forward is bit-identical to batch-1 decode is an empirical GPU
question (GEMM reduction order can move with batch shape); the
registered smoke (fontaine/scripts/sim_parallel_oracle.py) answers it
before any registered eval uses this path.

Usage (MUJOCO_GL=egl must be set — spawn workers inherit it):
  MUJOCO_GL=egl uv run python -m sim.rollout_sim_parallel \
      --checkpoint outputs/train/er_60k/step_060000 \
      --seed 0 --num-seeds 100 --workers 8 --out-json out.json
"""

import argparse
import json
import multiprocessing as mp
import subprocess
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np

from . import OUTPUT_DIR
from .rollout_sim import (
    STATS_REPO_ID,
    TASK,
    EpisodeResult,
    RolloutSim,
    hold_chunk_fn,
    run_episode_loop,
    sim_item,
)
from .so101_sim import CONTROL_HZ, SimObservation, SO101Sim

# Worker -> parent messages are picklable tagged tuples: "predict"
# carries (worker_id, seed, replan, top, wrist, state); "row" carries
# (worker_id, EpisodeResult); "done" and "error" carry the worker_id
# (plus the traceback text for "error"). The parent answers each
# "predict" with the [horizon, 6] action chunk, nothing else.


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    worker_id: int
    seeds: tuple[int, ...]
    replans: int
    horizon: int
    hold: bool
    out_dir: Path | None
    post_backend: str
    flip_camera_mount: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--hold",
        action="store_true",
        help="no policy: command the settled reset state every tick "
        "(runs fully worker-local, zero predict rounds)",
    )
    parser.add_argument("--seed", type=int, default=0, help="first env + noise seed")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="episodes: seed .. seed+N-1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="env worker processes (each owns a SO101Sim + GL context; "
        "capped at num-seeds)",
    )
    parser.add_argument("--replans", type=int, default=15)
    parser.add_argument("--execute-horizon", type=int, default=30)
    parser.add_argument("--sample-steps", type=int, default=10)
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
    parser.add_argument(
        "--post-backend",
        default="auto",
        choices=["auto", "numpy", "torch"],
        help="SO101Sim compositor per worker (auto/torch means one CUDA "
        "context per worker, ~0.5-1 GiB VRAM each; numpy frames differ "
        "from torch by the pinned <=2/255 compositor tolerance)",
    )
    parser.add_argument(
        "--no-mount-flip",
        action="store_true",
        help="run the PRE-flip wrist-bracket physics (mirrored Menagerie "
        "mount) — paired flip-effect reads only; flipped is the "
        "registered geometry",
    )
    parser.add_argument(
        "--convmap-seam-stats",
        type=Path,
        default=None,
        help="OFF-CONTRACT release-in-sim arm (sim.convmap): checkpoint "
        "dir whose normalization table states the sim seam's units (the "
        "ftrig rig-recomputed table); fits the discrete convention map "
        "seam -> checkpoint table and wraps the policy with it (state "
        "in through A, chunks back through A⁻¹). The policy name and "
        "the rows carry _convmap — never pooled with contract reads",
    )
    parser.add_argument(
        "--convmap-override",
        action="append",
        default=[],
        metavar="JOINT=OFFSET",
        help="explicit per-joint convention offset (degrees, sign +1) "
        "overriding the gated fit — only after the tripwire script "
        "shows the fit failing coverage and the override passing the "
        "first-action check; recorded verbatim in the rows JSON",
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
    if args.convmap_seam_stats is not None and args.hold:
        parser.error("--convmap-seam-stats wraps a policy — meaningless with --hold")
    if args.convmap_override and args.convmap_seam_stats is None:
        parser.error("--convmap-override requires --convmap-seam-stats")
    return args


def run_worker_episodes(
    sim: RolloutSim,
    config: WorkerConfig,
    send: Callable[[tuple[Any, ...]], None],
    recv: Callable[[], np.ndarray],
) -> None:
    """One worker's seed slice, strictly in order. ``send``/``recv`` are
    the predict round-trip transport — a Pipe in production, plain queues
    in the CPU-tier oracle. Timing recorded per replan is the round-trip
    wait (batch forward + lockstep barrier), not a solo forward."""
    for seed in config.seeds:
        latencies: list[float] = []
        next_chunk: Callable[[SimObservation, int], np.ndarray]

        if config.hold:
            next_chunk = hold_chunk_fn(config.horizon)
        else:

            def remote_chunk(
                obs: SimObservation,
                replan: int,
                _seed: int = seed,
                _latencies: list[float] = latencies,
            ) -> np.ndarray:
                start = time.perf_counter()
                send(
                    (
                        "predict",
                        config.worker_id,
                        _seed,
                        replan,
                        obs.top,
                        obs.wrist,
                        obs.state,
                    ),
                )
                chunk = recv()
                _latencies.append((time.perf_counter() - start) * 1000)
                return chunk

            next_chunk = remote_chunk

        video_path = (
            config.out_dir / f"rollout_seed{seed:03d}.mp4"
            if config.out_dir is not None
            else None
        )
        row = run_episode_loop(
            sim,
            seed,
            next_chunk,
            replans=config.replans,
            horizon=config.horizon,
            video_path=video_path,
            latencies=latencies,
        )
        send(("row", config.worker_id, row))
    send(("done", config.worker_id))


def _worker_main(config: WorkerConfig, conn: Connection) -> None:
    try:
        sim = SO101Sim(
            post_backend=config.post_backend,
            flip_camera_mount=config.flip_camera_mount,
        )
        run_worker_episodes(sim, config, conn.send, conn.recv)
    except Exception:  # noqa: BLE001 — shipped whole to the parent, which raises
        conn.send(("error", config.worker_id, traceback.format_exc()))


class WorkerDiedError(RuntimeError):
    pass


def serve(
    conns: Sequence[Any],
    predict_batch: Callable[[list[tuple[Any, ...]]], list[np.ndarray]],
    on_row: Callable[[EpisodeResult], None],
) -> list[int]:
    """The lockstep-rounds scheduler. ``conns`` is indexed by worker_id
    (anything with .send/.recv). Returns the per-round batch sizes — a
    deterministic trace the oracle asserts on. Raises WorkerDiedError
    with the worker's traceback if one errors out."""
    active = list(range(len(conns)))
    batch_sizes: list[int] = []
    while active:
        requests: list[tuple[Any, ...]] = []
        still_active: list[int] = []
        for worker_id in active:
            while True:
                try:
                    message = conns[worker_id].recv()
                except EOFError as error:
                    raise WorkerDiedError(
                        f"worker {worker_id} closed its pipe without a "
                        "done/error message — check the worker's stderr",
                    ) from error
                tag = message[0]
                if tag == "row":
                    on_row(message[2])
                elif tag == "done":
                    break
                elif tag == "predict":
                    requests.append(message)
                    still_active.append(worker_id)
                    break
                elif tag == "error":
                    raise WorkerDiedError(f"worker {message[1]} died:\n{message[2]}")
                else:
                    raise WorkerDiedError(f"unknown worker message tag {tag!r}")
        active = still_active
        if requests:
            chunks = predict_batch(requests)
            for message, chunk in zip(requests, chunks, strict=True):
                conns[message[1]].send(chunk)
            batch_sizes.append(len(requests))
    return batch_sizes


def main() -> int:
    # Policy-side imports are parent-only; workers re-import this module
    # under spawn and must not pay for (or touch) the policy stack there.
    import torch

    from bijou.decoders.flow import SamplingMethod
    from bijou.eval.molmo_norm import MolmoNorm
    from bijou.eval.policies import BijouPolicy

    from .convmap import seam_convention_map

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
            expert_dtype=getattr(torch, args.expert_dtype),
            molmo_norm=(
                MolmoNorm.CONVENTION_MAP
                if args.convmap_seam_stats is not None
                else MolmoNorm.CHECKPOINT
            ),
        )
        horizon = min(args.execute_horizon, policy.info.chunk_size)
        print(
            f"policy: {policy.name} "
            f"({args.method}-{args.sample_steps}, horizon {horizon})",
        )

    seeds = list(range(args.seed, args.seed + args.num_seeds))
    workers = max(1, min(args.workers, len(seeds)))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    processes: list[Any] = []
    conns: list[Connection] = []
    for worker_id in range(workers):
        parent_conn, child_conn = context.Pipe()
        config = WorkerConfig(
            worker_id=worker_id,
            seeds=tuple(seeds[worker_id::workers]),
            replans=args.replans,
            horizon=horizon,
            hold=args.hold,
            out_dir=args.out_dir,
            post_backend=args.post_backend,
            flip_camera_mount=not args.no_mount_flip,
        )
        process = context.Process(
            target=_worker_main,
            args=(config, child_conn),
            daemon=True,
        )
        process.start()
        child_conn.close()
        processes.append(process)
        conns.append(parent_conn)
    print(f"spawned {workers} env workers for {len(seeds)} seeds", flush=True)

    results: list[EpisodeResult] = []
    predict_ms: list[float] = []
    seam = None

    if policy is None:

        def predict_batch(requests: list[tuple[Any, ...]]) -> list[np.ndarray]:
            raise WorkerDiedError("hold arm workers must not request predicts")
    else:
        chunk_size = policy.info.chunk_size
        if args.convmap_seam_stats is not None:
            # Off-contract seam (sim.convmap): items wear the SEAM's
            # stats (the units the sim actually speaks), and the policy's
            # per-repo map cache is seeded with the resolved fit — so an
            # override rides the exact rewrite path the gated fit would,
            # and the policy never re-fits behind our back.
            seam = seam_convention_map(
                args.convmap_seam_stats,
                policy.info.normalization,
                args.convmap_override,
            )
            policy._molmo_norm_maps["sim/eval100"] = seam.item_maps
            stats = seam.seam_stats
            print(
                f"convmap seam {args.convmap_seam_stats.name}: "
                f"scale {seam.map.scale.tolist()} "
                f"offset {seam.map.offset.tolist()} "
                f"(gated fit offset {seam.fit.map.offset.tolist()}, "
                f"overrides {seam.overrides or 'none'})",
                flush=True,
            )
        else:
            # Converted checkpoints (molmoact2 lineage) carry no
            # per-dataset table — their items must wear the checkpoint's
            # MERGED stats (same fallback as the sequential driver).
            stats = policy.info.per_dataset_normalization.get(
                STATS_REPO_ID,
                policy.info.normalization,
            )

        def predict_batch(requests: list[tuple[Any, ...]]) -> list[np.ndarray]:
            items = []
            indices = []
            for _, _, seed, replan, top, wrist, state in requests:
                obs = SimObservation(top=top, wrist=wrist, state=state)
                items.append(
                    sim_item(obs, seed, replan, stats=stats, chunk_size=chunk_size),
                )
                indices.append(replan)
            start = time.perf_counter()
            chunks = policy.predict(items, indices)
            predict_ms.append((time.perf_counter() - start) * 1000)
            return [chunk.numpy() for chunk in chunks]

    started = time.perf_counter()
    try:
        batch_sizes = serve(conns, predict_batch, results.append)
    finally:
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
    wall_s = time.perf_counter() - started
    results.sort(key=lambda r: r.seed)

    print("\nseed | init cm | min cm | final cm | progress cm | success")
    for r in sorted(results, key=lambda r: -r.progress_cm):
        success = f"tick {r.success_tick}" if r.success_tick is not None else "-"
        print(
            f"{r.seed:4d} | {r.initial_cm:7.1f} | {r.min_cm:6.1f} | "
            f"{r.final_cm:8.1f} | {r.progress_cm:11.1f} | {success}",
        )
    mean_batch = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0
    print(
        f"\n{len(results)} episodes in {wall_s / 60:.1f} min | "
        f"{len(batch_sizes)} predict rounds, mean batch {mean_batch:.1f}",
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
                "replans": args.replans,
                "execute_horizon": horizon,
                "sample_steps": args.sample_steps,
                "method": args.method,
                "expert_dtype": args.expert_dtype,
                "control_hz": CONTROL_HZ,
                "task": TASK,
                "stats_repo_id": STATS_REPO_ID,
                "mount_flip": not args.no_mount_flip,
                "commit": commit,
                # Off-contract provenance: the resolved seam map (fit +
                # overrides) — None on contract reads. Rows under a
                # non-None convmap must never pool with contract rows.
                "convmap": (
                    None
                    if seam is None or policy is None
                    else {
                        "seam_stats": str(args.convmap_seam_stats),
                        "scale": seam.map.scale.tolist(),
                        "offset": seam.map.offset.tolist(),
                        "fit_offset": seam.fit.map.offset.tolist(),
                        "overrides": seam.overrides,
                        "policy_name": policy.name,
                    }
                ),
            },
            "parallel": {
                "workers": workers,
                "scheduler": "lockstep-v1",
                "post_backend": args.post_backend,
                "wall_s": round(wall_s, 1),
                "batch_sizes": batch_sizes,
                "predict_ms": [round(v, 1) for v in predict_ms],
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
