"""Sharded demo-generation driver (owner work order 2026-08-16
12:18:57Z: "Make the sharding changes to be ready to generate the demo
dataset"; queue item ``demo-gen-sharded-a100``).

Demo collection is render-bound and embarrassingly seed-parallel
(measured: the unrendered expert runs 200 seeds in ~3 min vs 313 kept
in 4 h rendered single-process), so the driver fans
``sim.collect_demos`` out over N subprocesses with DISJOINT contiguous
seed ranges — shard k owns ``[seed_start + k*stride, +stride)`` — each
writing its own LeRobot dataset under ``<out>/shards/shard_kk`` with
its own resume state. EGL contexts round-robin over the visible GPUs
(``MUJOCO_EGL_DEVICE_ID`` + ``CUDA_VISIBLE_DEVICES`` per shard, both
pinned to the same physical device so the render and the torch lens
remap share it).

Resume: rerunning the same command resumes every shard from its
``collect_state.json`` (the single-process mechanism, unchanged). The
driver banks its plan in ``driver_manifest.json`` and REFUSES a rerun
whose sharding arguments disagree with the banked plan — a changed
shard count would silently re-partition seed ranges over half-filled
shards.

After all shards finish, merge with ``sim.merge_demo_shards`` (2-shard
smoke merge is oracle-pinned bit-identical to a single run over the
same seeds, tests/test_collect_demos_sharded.py).

Usage (the v1 5k dataset, owner-approved 12:21:03Z):
  MUJOCO_GL=egl uv run python -m sim.collect_demos_sharded \
      --out ~/datasets/fontaine/grasp_demos_v1 \
      --repo-id fontaine/grasp_demos_v1 \
      --shards 32 --target-kept 5000 --seed-start 2000 \
      --spawn-version v2 --tint-band mix70
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .collect_demos import REPO_ID
from .scripted_expert import DEMO_SEED_BASE


@dataclass(frozen=True)
class ShardSpec:
    index: int
    out: str  # relative to the driver root
    seed_start: int
    max_seeds: int
    target_kept: int


def plan_shards(
    *,
    shards: int,
    target_kept: int,
    seed_start: int,
    seeds_per_shard: int,
) -> list[ShardSpec]:
    """Disjoint contiguous seed ranges; the kept target split evenly
    with the remainder spread over the first shards."""
    if shards < 1:
        raise ValueError(f"shards must be >= 1, got {shards}")
    if seed_start < DEMO_SEED_BASE:
        raise ValueError(
            f"seed_start {seed_start} is inside the frozen eval holdout — "
            f"demo seeds begin at {DEMO_SEED_BASE}",
        )
    base, extra = divmod(target_kept, shards)
    return [
        ShardSpec(
            index=k,
            out=f"shards/shard_{k:02d}",
            seed_start=seed_start + k * seeds_per_shard,
            max_seeds=seeds_per_shard,
            target_kept=base + (1 if k < extra else 0),
        )
        for k in range(shards)
    ]


def _manifest_path(root: Path) -> Path:
    return root / "driver_manifest.json"


def shard_command(spec: ShardSpec, root: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sim.collect_demos",
        "--out",
        str(root / spec.out),
        "--repo-id",
        args.repo_id,
        "--target-kept",
        str(spec.target_kept),
        "--seed-start",
        str(spec.seed_start),
        "--max-seeds",
        str(spec.max_seeds),
        "--max-wall-hours",
        str(args.max_wall_hours),
        "--max-ticks",
        str(args.max_ticks),
        "--spawn-version",
        args.spawn_version,
        "--tint-band",
        args.tint_band,
    ]


def shard_env(spec: ShardSpec, gpus: list[int]) -> dict[str, str]:
    """Render and lens-remap pinned to one physical GPU per shard,
    round-robined: EGL device ids enumerate physical devices, so the
    EGL id stays absolute while CUDA_VISIBLE_DEVICES makes the same
    card the shard's only torch device."""
    gpu = gpus[spec.index % len(gpus)]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _detect_gpus() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [int(line) for line in out.split()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [0]


def _shard_progress(root: Path, spec: ShardSpec) -> tuple[int, int]:
    """(kept, attempted) from the shard's banked resume state."""
    state = root / spec.out / "collect_state.json"
    if not state.exists():
        return 0, 0
    data = json.loads(state.read_text())
    return len(data["kept_seeds"]), int(data["attempted"])


def run(args: argparse.Namespace) -> int:
    root = args.out.expanduser()
    specs = plan_shards(
        shards=args.shards,
        target_kept=args.target_kept,
        seed_start=args.seed_start,
        seeds_per_shard=args.seeds_per_shard,
    )
    gpus = args.gpus or _detect_gpus()
    manifest = {
        "repo_id": args.repo_id,
        "spawn_version": args.spawn_version,
        "tint_band": args.tint_band,
        "target_kept": args.target_kept,
        "shards": [asdict(s) for s in specs],
    }
    if _manifest_path(root).exists():
        banked = json.loads(_manifest_path(root).read_text())
        if banked != manifest:
            raise SystemExit(
                f"ABORT: {root} was planned with different sharding arguments "
                "(driver_manifest.json disagrees) — resuming would "
                "re-partition seed ranges over half-filled shards",
            )
        print(f"[driver] RESUME: manifest matches, {len(specs)} shards")
    if args.dry_run:
        for spec in specs:
            gpu = gpus[spec.index % len(gpus)]
            print(
                f"[driver] shard {spec.index:02d}: seeds "
                f"[{spec.seed_start}, {spec.seed_start + spec.max_seeds}) "
                f"target {spec.target_kept} gpu {gpu}",
            )
        return 0
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    _manifest_path(root).write_text(json.dumps(manifest, indent=2) + "\n")

    procs: list[tuple[ShardSpec, subprocess.Popen[bytes], Path]] = []
    for spec in specs:
        log_path = root / "logs" / f"shard_{spec.index:02d}.log"
        log_file = log_path.open("ab")
        proc = subprocess.Popen(
            shard_command(spec, root, args),
            env=shard_env(spec, gpus),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        log_file.close()  # the child holds its own fd
        procs.append((spec, proc, log_path))
        print(f"[driver] shard {spec.index:02d} pid {proc.pid} -> {log_path}")

    t0 = time.time()
    while any(proc.poll() is None for _, proc, _ in procs):
        time.sleep(args.poll_s)
        kept = attempted = 0
        for spec, _, _ in procs:
            k, a = _shard_progress(root, spec)
            kept += k
            attempted += a
        live = sum(proc.poll() is None for _, proc, _ in procs)
        rate = kept / max(time.time() - t0, 1) * 60
        eta_min = (args.target_kept - kept) / rate if rate > 0 else float("inf")
        print(
            f"[driver] {kept}/{args.target_kept} kept ({attempted} attempted), "
            f"{live}/{len(procs)} shards live, {rate:.1f} kept/min, "
            f"ETA {eta_min:.0f} min",
            flush=True,
        )

    failed = [
        (spec, proc.returncode) for spec, proc, _ in procs if proc.returncode != 0
    ]
    kept = sum(_shard_progress(root, spec)[0] for spec, _, _ in procs)
    print(f"[driver] DONE: {kept}/{args.target_kept} kept, {len(failed)} failed")
    for spec, code in failed:
        print(f"[driver] shard {spec.index:02d} EXIT {code} — see logs/")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--target-kept", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds-per-shard", type=int, default=2000)
    parser.add_argument("--spawn-version", choices=("v1", "v2"), default="v2")
    parser.add_argument(
        "--tint-band",
        choices=("rig_gray", "wide", "mix70"),
        default="mix70",
    )
    parser.add_argument("--max-wall-hours", type=float, default=12.0)
    parser.add_argument("--max-ticks", type=int, default=600)
    parser.add_argument("--gpus", type=lambda s: [int(x) for x in s.split(",")])
    parser.add_argument("--poll-s", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
