"""Leaderboard decode-cost micro-benchmark (owner steering 2026-08-07
10:04Z follow-up; queue item leaderboard-decode-cost-microbench).

RECORD-ONLY: this measures WALL-CLOCK per frame for every leaderboard
decode config under one shared harness — it replaces the ≈
(mtime-bounded) `eval ms/frame` leaderboard entries with same-config
numbers and adds the batch=1 single-stream latency (the
deployment-facing number, #16 hook). It makes NO model-quality claims
and cannot reorder the leaderboard's MAE column.

Pre-reg: fontaine/blog/src/posts/2026-08-07-prereg-leaderboard-decode-microbench.md

Method (the draws_rate_gate.py pattern, offline): each config runs
``bijou.eval`` on the SAME seeded frame sample (``--num-samples N
--seed 0`` over the community holdout selection — frame choice is
data-side and checkpoint-independent, so every config scores identical
frames within a mode). The wrapper polls the run's log and timestamps
each new ``scored N/M frames`` progress line with its own monotonic
clock; the quoted rate is (last line − first line), which excludes
model load, dataset scan and the first progress interval (CUDA
warmup). ``bijou/eval/cli.py`` prints a progress line every 5 batches,
so the registered frame counts guarantee ≥ 2 lines per run:

  - batched mode: batch 32 / workers 20 (the banked panel-eval
    config), N = 320 → measured window 160 frames.
  - single mode: batch 1 / workers 4, N = 50 → measured window ≥ 45
    frames; per-frame wall-clock IS single-stream latency here (one
    frame in flight at a time; AR decodes stay token-serial inside it).

Registered configs (leaderboard rows, decode flags byte-matched to the
banked stems; noise keying is pinned ``stable`` for every flow config —
keying changes RNG derivation, not compute, and is recorded):

  ar_greedy             AR-100k, deployment fast path
  ar_draws10_t1         AR-100k, --ar-temperature 1.0 --sample-draws 10
  teacher_heun30_draws1 flow teacher @80k, --sample-method heun --sample-steps 30
  teacher_heun30_draws10  ... --sample-draws 10
  student_1nfe_draws1   SnapFlow student @30k, --sample-method euler
                        --sample-steps 1 --target-time zero
  student_1nfe_draws5   ... --sample-draws 5
  student_1nfe_draws10  ... --sample-draws 10

Guards die loud: unknown config name, missing checkpoint, GPU busy at
start, eval rc != 0, < 2 progress lines, non-increasing frame counts,
per-run watchdog timeout. ``--selftest`` runs the rate-arithmetic
oracle fixtures (pure function, exact expectations) and exits.

Usage::

    uv run python fontaine/scripts/leaderboard_decode_microbench.py --selftest
    uv run python fontaine/scripts/leaderboard_decode_microbench.py            # full pass
    uv run python fontaine/scripts/leaderboard_decode_microbench.py \
        --configs ar_greedy --modes single --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/ubuntu/flow-matching")
PYTHON = str(REPO / ".venv/bin/python")
LOG_DIR = Path("/home/ubuntu")
OUT_DEFAULT = "reports/analysis__leaderboard_decode_microbench.json"

DATA_FLAGS = [
    "--data",
    "/home/ubuntu/datasets/mcobzarenco/community_curated_v0",
    "--episodes",
    "holdout",
    "--holdout-episodes",
    "0.1",
    "--split-seed",
    "0",
    "--fps",
    "30",
    "--camera-counts",
    "1",
    "2",
]

CKPT_AR100K = (
    "/home/ubuntu/checkpoints/bijou-checkpoints/bijou_arb_rcond_100k_ddp4/step_100000"
)
CKPT_TEACHER = "/home/ubuntu/checkpoints/bijou-checkpoints/bijou_flow_artrunk_h1024_40k_ddp2/step_080000"
CKPT_STUDENT = "outputs/train/fontaine_flow_snapdistill_h1024_30k_1xh100/step_030000"

_FLOW_STABLE = ["--noise-key", "stable"]
_TEACHER = ["--sample-method", "heun", "--sample-steps", "30", *_FLOW_STABLE]
_STUDENT = [
    "--sample-method",
    "euler",
    "--sample-steps",
    "1",
    "--target-time",
    "zero",
    *_FLOW_STABLE,
]

CONFIGS: dict[str, dict] = {
    "ar_greedy": {"checkpoint": CKPT_AR100K, "flags": []},
    "ar_draws10_t1": {
        "checkpoint": CKPT_AR100K,
        "flags": ["--ar-temperature", "1.0", "--sample-draws", "10"],
    },
    "teacher_heun30_draws1": {
        "checkpoint": CKPT_TEACHER,
        "flags": [*_TEACHER, "--sample-draws", "1"],
    },
    "teacher_heun30_draws10": {
        "checkpoint": CKPT_TEACHER,
        "flags": [*_TEACHER, "--sample-draws", "10"],
    },
    "student_1nfe_draws1": {
        "checkpoint": CKPT_STUDENT,
        "flags": [*_STUDENT, "--sample-draws", "1"],
    },
    "student_1nfe_draws5": {
        "checkpoint": CKPT_STUDENT,
        "flags": [*_STUDENT, "--sample-draws", "5"],
    },
    "student_1nfe_draws10": {
        "checkpoint": CKPT_STUDENT,
        "flags": [*_STUDENT, "--sample-draws", "10"],
    },
}

MODES: dict[str, dict] = {
    "batched": {"batch_size": 32, "num_workers": 20, "num_samples": 320},
    "single": {"batch_size": 1, "num_workers": 4, "num_samples": 50},
}

PROGRESS_RE = re.compile(r"scored (\d+)/(\d+) frames")
POLL_SECONDS = 0.5
WATCHDOG_SECONDS = 30 * 60  # per run; AR draws10 batched projects ~10 min


def rate_from_events(events: list[tuple[float, int]]) -> dict:
    """ms/frame from (monotonic_seconds, frames_done) progress events.

    Pure arithmetic (the oracle targets this): rate is measured strictly
    between the FIRST and LAST progress line, so startup and the first
    progress interval are excluded. Dies loud on < 2 events or
    non-increasing frame counts.
    """
    if len(events) < 2:
        raise ValueError(f"need >= 2 progress events, got {len(events)}")
    frames = [f for _, f in events]
    if any(b <= a for a, b in itertools.pairwise(frames)):
        raise ValueError(f"non-increasing frame counts: {frames}")
    (t0, f0), (t1, f1) = events[0], events[-1]
    if t1 <= t0:
        raise ValueError(f"non-increasing timestamps: {t0} -> {t1}")
    return {
        "ms_per_frame": (t1 - t0) / (f1 - f0) * 1000.0,
        "window_frames": f1 - f0,
        "window_seconds": t1 - t0,
        "events": len(events),
    }


def selftest() -> None:
    # Exact arithmetic: 160 frames in 80 s -> 500 ms/frame.
    r = rate_from_events([(10.0, 160), (90.0, 320)])
    assert r["ms_per_frame"] == 500.0 and r["window_frames"] == 160, r
    # Multi-event: only first/last matter -> 100 frames in 10 s = 100 ms/frame.
    r = rate_from_events([(0.0, 5), (2.0, 50), (10.0, 105)])
    assert r["ms_per_frame"] == 100.0 and r["events"] == 3, r
    # Guard: single event.
    for bad in (
        [(0.0, 5)],
        [(0.0, 5), (1.0, 5)],
        [(0.0, 10), (1.0, 5)],
        [(5.0, 10), (5.0, 20)],
    ):
        try:
            rate_from_events(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"guard failed to fire on {bad}")
    # Log-line parse fixture.
    assert PROGRESS_RE.search("  scored 160/320 frames").group(1) == "160"
    assert PROGRESS_RE.search("  scored 45/50 frames (rank 0 shard)").group(1) == "45"
    assert PROGRESS_RE.search("no progress here") is None
    print("selftest PASS: rate arithmetic exact, guards fire, parser matches")


def gpu_quiet() -> None:
    mem = int(
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()[0],
    )
    if mem > 1024:
        sys.exit(f"GPU busy ({mem} MiB used) — abort (draws10_t1 still running?)")


def run_one(config: str, mode: str, *, dry_run: bool = False) -> dict | None:
    cfg, m = CONFIGS[config], MODES[mode]
    ckpt = Path(cfg["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    if not ckpt.is_dir():
        sys.exit(f"checkpoint missing: {ckpt} — abort")
    name = f"bench__decode__{config}__{mode}"
    log = LOG_DIR / f"{name}.log"
    cmd = [
        PYTHON,
        "-m",
        "bijou.eval",
        *DATA_FLAGS,
        "--checkpoint",
        str(ckpt),
        "--num-samples",
        str(m["num_samples"]),
        "--seed",
        "0",
        "--batch-size",
        str(m["batch_size"]),
        "--num-workers",
        str(m["num_workers"]),
        "--report-samples",
        "0",
        "--output-json",
        f"reports/{name}.json",
        *cfg["flags"],
    ]
    if dry_run:
        print(f"[dry-run] {name}\n  {' '.join(cmd)}")
        return None
    print(
        f"=== {name}: N={m['num_samples']} b{m['batch_size']} "
        f"w{m['num_workers']} -> {log}",
    )
    events: list[tuple[float, int]] = []
    seen = 0
    start = time.monotonic()
    with log.open("w") as lf:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                if time.monotonic() - start > WATCHDOG_SECONDS:
                    proc.terminate()
                    sys.exit(f"{name}: watchdog {WATCHDOG_SECONDS}s exceeded — abort")
                text = log.read_text()
                for match in list(PROGRESS_RE.finditer(text))[seen:]:
                    events.append((time.monotonic(), int(match.group(1))))
                    seen += 1
                time.sleep(POLL_SECONDS)
        finally:
            if proc.poll() is None:
                proc.terminate()
    # Progress lines that landed between the last poll and process exit.
    t_exit = time.monotonic()
    events.extend(
        (t_exit, int(match.group(1)))
        for match in list(PROGRESS_RE.finditer(log.read_text()))[seen:]
    )
    if proc.returncode != 0:
        sys.exit(f"{name}: eval rc={proc.returncode} (log: {log}) — abort")
    try:
        rate = rate_from_events(events)
    except ValueError as err:
        sys.exit(f"{name}: rate extraction failed ({err}; log: {log}) — abort")
    print(
        f"    {rate['ms_per_frame']:.1f} ms/frame over {rate['window_frames']} "
        f"frames ({rate['events']} progress lines)",
    )
    return {
        "config": config,
        "mode": mode,
        **rate,
        "num_samples": m["num_samples"],
        "batch_size": m["batch_size"],
        "num_workers": m["num_workers"],
        "checkpoint": str(ckpt),
        "flags": cfg["flags"],
        "log": str(log),
        "total_wall_seconds": time.monotonic() - start,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--modes", nargs="*", default=list(MODES), choices=list(MODES))
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.dry_run:
        gpu_quiet()
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    results = []
    for mode in args.modes:  # batched pass first, then latency pass
        for config in args.configs:  # cheap-to-expensive order not needed
            row = run_one(config, mode, dry_run=args.dry_run)
            if row is not None:
                results.append(row)
    if args.dry_run:
        return
    out = REPO / args.out
    out.write_text(
        json.dumps(
            {
                "instrument": "leaderboard_decode_microbench",
                "prereg": "fontaine/blog/src/posts/"
                "2026-08-07-prereg-leaderboard-decode-microbench.md",
                "record_only": True,
                "gpu": gpu_name,
                "poll_seconds": POLL_SECONDS,
                "results": results,
            },
            indent=1,
        )
        + "\n",
    )
    print(f"=== MICROBENCH DONE: {len(results)} rows -> {out} ===")


if __name__ == "__main__":
    main()
