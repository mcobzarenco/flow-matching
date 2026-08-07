"""Cost gate for sampled-draws panel evals (ideas #19), mechanized.

Pre-reg: fontaine/blog/src/posts/2026-08-06-prereg-ar-sampled-draws.md
— "measure the rate over the first ~200 frames; if a full-panel draws10
run projects > 24 GPU-h, BOTH arms drop to the frozen q4 subset … the
switch is recorded, not silent." This script IS that measurement, so
the launch-time decision is pre-registered arithmetic, not judgment.

It polls the eval's log for the runner's progress lines
(``scored N/M frames``; under torchrun DDP the counts are the RANK-0
SHARD, ~total/world — ``bijou/eval/cli.py``). The wall-clock projection
from shard progress is world-invariant: ranks advance in near-lockstep
over an evenly sharded plan, so shard fraction ≈ global fraction. The
rate baseline is the FIRST observed progress line, not process start —
model load / dataset scan never dilutes the measured rate.

Exit codes (the launcher branches on these; anything else is a bug):

- 0 PASS — projected total ≤ the gate; let the full-panel run finish.
- 2 FALLBACK — projected total > the gate; the launcher kills the run
  and relaunches on the frozen q4 subset.
- 1 INDETERMINATE — no progress line ever appeared inside the timeout
  (or the log never existed). The launcher leaves the run alive; the
  babysit registry's gpu_hours_max gate is the backstop. A timeout
  WITH partial progress still decides (slow runs must not dodge the
  gate by being too slow to reach the measurement window).

Usage (from the box launcher, right after backgrounding the eval):

    .venv/bin/python fontaine/scripts/draws_rate_gate.py \
        --log ~/logs/eval__<run>__step_<step>__panel_k4l2_draws10_t1.log \
        --ngpu 4 --gate-gpu-hours 24 --min-frames 200
"""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Callable
from pathlib import Path

PROGRESS_RE = re.compile(r"scored (\d+)/(\d+) frames")

PASS, INDETERMINATE, FALLBACK = 0, 1, 2


def parse_progress(text: str) -> tuple[int, int] | None:
    """Last ``scored N/M frames`` occurrence in ``text`` as ``(n, m)``
    (the ``(rank 0 shard)`` suffix is irrelevant to the match), or
    ``None`` if no progress line has appeared yet."""
    matches = PROGRESS_RE.findall(text)
    if not matches:
        return None
    n, m = matches[-1]
    return int(n), int(m)


def project_gpu_hours(
    first: tuple[float, int],
    last: tuple[float, int],
    total: int,
    ngpu: int,
) -> float | None:
    """Projected GPU-hours for the WHOLE run from two ``(seconds,
    frames)`` samples of one rank's shard progress: shard rate extends
    to the shard total, wall-clock multiplies by ``ngpu``. ``None``
    until the samples show forward progress."""
    elapsed = last[0] - first[0]
    frames = last[1] - first[1]
    if elapsed <= 0 or frames <= 0:
        return None
    return elapsed / frames * total / 3600.0 * ngpu


def decide(projected_gpu_hours: float, gate_gpu_hours: float) -> int:
    """PASS or FALLBACK per the pre-reg's strict inequality: only a
    projection strictly over the gate triggers the q4 fallback."""
    return FALLBACK if projected_gpu_hours > gate_gpu_hours else PASS


def run_gate(
    log: Path,
    *,
    ngpu: int,
    gate_gpu_hours: float,
    min_frames: int,
    poll_seconds: float,
    timeout_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Poll ``log`` until the shard advanced ``ceil(min_frames / ngpu)``
    frames past the first observed progress line (≈ ``min_frames``
    global frames), then decide. On timeout, decide from whatever
    progress exists; INDETERMINATE only if none ever appeared."""
    min_shard_frames = -(-min_frames // ngpu)
    deadline = clock() + timeout_seconds
    first: tuple[float, int] | None = None
    last: tuple[float, int] | None = None
    total = 0
    while clock() < deadline:
        progress = parse_progress(log.read_text()) if log.exists() else None
        if progress is not None:
            now = clock()
            n, total = progress
            if first is None:
                first = (now, n)
                print(f"gate: baseline {n}/{total} shard frames", flush=True)
            last = (now, n)
            if n - first[1] >= min_shard_frames:
                break
        sleep(poll_seconds)
    else:
        if first is None or last is None:
            print(
                f"gate: INDETERMINATE — no progress line in {log} within "
                f"{timeout_seconds:.0f}s; leaving the run to the babysit "
                "registry's gpu_hours_max gate",
                flush=True,
            )
            return INDETERMINATE
        print(
            f"gate: timeout with partial progress "
            f"({last[1] - first[1]} shard frames) — deciding anyway",
            flush=True,
        )
    assert first is not None and last is not None  # loop exits ensure it
    projected = project_gpu_hours(first, last, total, ngpu)
    if projected is None:
        print("gate: INDETERMINATE — no forward progress measured", flush=True)
        return INDETERMINATE
    verdict = decide(projected, gate_gpu_hours)
    rate = (last[1] - first[1]) / (last[0] - first[0]) * 60.0
    print(
        f"gate: {'FALLBACK' if verdict == FALLBACK else 'PASS'} — "
        f"{last[1] - first[1]} shard frames at {rate:.1f} shard-f/min over "
        f"{last[0] - first[0]:.0f}s -> projected {projected:.1f} GPU-h for "
        f"{total} shard frames x {ngpu} GPU (gate {gate_gpu_hours:.1f})",
        flush=True,
    )
    return verdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ngpu", type=int, default=4)
    parser.add_argument("--gate-gpu-hours", type=float, default=24.0)
    parser.add_argument("--min-frames", type=int, default=200)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=10800.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_gate(
        args.log,
        ngpu=args.ngpu,
        gate_gpu_hours=args.gate_gpu_hours,
        min_frames=args.min_frames,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
