"""Registered determinism oracle for the parallel sim-rollout path
(pre-reg: fontaine/blog/src/posts/2026-08-12-prereg-sim-parallel-rollouts.md).

Runs the SAME seeds through the sequential driver (sim.rollout_sim) and
the parallel one (sim.rollout_sim_parallel), then compares per-seed rows
field by field, excluding only latency_ms (wall-clock by construction).

GREEN  = every compared field bit-identical -> the parallel path may
         substitute for the sequential one in registered evals.
FAIL   = any mismatch -> the parallel path may NOT be used for a
         registered eval until a tolerance is registered; the printed
         per-field diffs are the input to that decision.

The CPU-tier twin (tests/test_sim_parallel_rollouts.py) pins harness
equivalence with fakes; what THIS run adds is the real stack — EGL
rendering per worker and, crucially, whether the batched forward decodes
bit-identically to batch-1 (GEMM reduction order can move with batch
shape, and heun-10 feeds any last-bit drift back through the physics).

Usage (GPU box, ~15-20 min at the defaults):
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_parallel_oracle.py \
      --checkpoint <ckpt-dir> [--num-seeds 6] [--workers 2]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "outputs" / "sim" / "parallel_oracle"

# Wall-clock by construction (sequential: solo predict; parallel:
# round-trip incl. the lockstep barrier). Everything else must match.
EXCLUDED_FIELDS = {"latency_ms"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--replans", type=int, default=30)
    parser.add_argument("--execute-horizon", type=int, default=30)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--method", default="heun", choices=["euler", "heun"])
    parser.add_argument(
        "--flow-decoder-dtype",
        default="bfloat16",
        choices=["float32", "bfloat16"],
    )
    parser.add_argument(
        "--post-backend",
        default="auto",
        choices=["auto", "numpy", "torch"],
        help="parallel workers' compositor; the sequential driver always "
        "runs SO101Sim defaults (auto), so keep this at auto for an "
        "apples-to-apples read",
    )
    return parser.parse_args()


def run_driver(module: str, args: argparse.Namespace, extra: list[str]) -> float:
    out_dir = OUT_DIR / module.rsplit(".", 1)[-1]
    command = [
        sys.executable,
        "-m",
        module,
        "--checkpoint",
        str(args.checkpoint),
        "--seed",
        str(args.seed),
        "--num-seeds",
        str(args.num_seeds),
        "--replans",
        str(args.replans),
        "--execute-horizon",
        str(args.execute_horizon),
        "--sample-steps",
        str(args.sample_steps),
        "--method",
        args.method,
        "--flow-decoder-dtype",
        args.flow_decoder_dtype,
        "--out-dir",
        str(out_dir),
        "--out-json",
        str(out_dir / "rows.json"),
        *extra,
    ]
    print(f"\n$ {' '.join(command)}", flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=REPO, check=True)
    return time.monotonic() - started


def load_rows(module: str) -> dict[int, dict[str, object]]:
    payload = json.loads(
        (OUT_DIR / module.rsplit(".", 1)[-1] / "rows.json").read_text(),
    )
    return {row["seed"]: row for row in payload["episodes"]}


def max_abs_diff(a: object, b: object) -> float | None:
    """Max absolute difference for scalar/list numeric fields (None for
    non-numeric or shape mismatches — those are just 'different')."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        diffs = [max_abs_diff(x, y) for x, y in zip(a, b, strict=True)]
        if any(d is None for d in diffs):
            return None
        return max((d for d in diffs if d is not None), default=0.0)
    return None


def main() -> int:
    args = parse_args()
    sequential_s = run_driver("sim.rollout_sim", args, [])
    parallel_s = run_driver(
        "sim.rollout_sim_parallel",
        args,
        ["--workers", str(args.workers), "--post-backend", args.post_backend],
    )

    sequential = load_rows("sim.rollout_sim")
    parallel = load_rows("sim.rollout_sim_parallel")
    print(
        f"\nsequential {sequential_s / 60:.1f} min | parallel "
        f"{parallel_s / 60:.1f} min ({args.workers} workers, "
        f"{sequential_s / max(parallel_s, 1e-9):.2f}x)",
    )

    if sorted(sequential) != sorted(parallel):
        print(f"seed sets differ: {sorted(sequential)} vs {sorted(parallel)}")
        print("SIM-PARALLEL ORACLE: FAIL")
        return 1

    mismatches = 0
    for seed in sorted(sequential):
        fields = set(sequential[seed]) | set(parallel[seed])
        for name in sorted(fields - EXCLUDED_FIELDS):
            a = sequential[seed].get(name)
            b = parallel[seed].get(name)
            if a == b:
                continue
            mismatches += 1
            diff = max_abs_diff(a, b)
            detail = f"max|diff| {diff:.6g}" if diff is not None else f"{a!r} vs {b!r}"
            print(f"  seed {seed} {name}: MISMATCH ({detail})")

    if mismatches:
        print(
            f"\n{mismatches} field mismatches — the parallel path is NOT "
            "row-identical at these settings; register a tolerance (or fix "
            "the divergence) before any registered eval uses it.",
        )
        print("SIM-PARALLEL ORACLE: FAIL")
        return 1
    print(
        f"\nall {len(sequential)} seeds row-identical "
        f"(every field except {sorted(EXCLUDED_FIELDS)})",
    )
    print("SIM-PARALLEL ORACLE: GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
