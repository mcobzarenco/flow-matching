"""Cost gate for the attach-screen training arms (#4), mechanized.

Pre-reg (2026-08-07-prereg-molmo2-attach-screen.md, Gates): "first ~200
steps of each arm project the batch total (the draws_rate_gate.py
mechanization pattern); projected total > 70 GPU-h ⇒ both arms
downshift to 5k matched steps, the switch echoed loudly and the result
labeled 5k-screen." This script IS that measurement, so the launch-time
branch is pre-registered arithmetic, not judgment.

It polls the arm's ``train_log.jsonl`` (rank-0 rows, one per
``--log-every`` steps, each carrying a measured ``s_per_step``) until a
row at or past ``--min-step`` appears, then projects

    total = median(s_per_step) * arm_steps * ngpu / 3600 + extra

where ``--extra-gpu-hours`` carries the REST of the pre-registered
batch (the other arm's measured-or-estimated train hours + the frozen
eval obligations — each launcher pins its own arithmetic in a comment).
The median over post-warmup rows is deliberate: save/eval boundaries
put multi-minute outliers into single rows, and a mean would let one
checkpoint write flip a 70 GPU-h decision.

Exit codes (the draws_rate_gate contract; the launcher branches):

- 0 PASS — projected batch total ≤ the gate; the 10k arm proceeds.
- 2 FALLBACK — projected total > the gate; the launcher kills the arm
  and relaunches BOTH arms' schedule at 5k matched steps (5k-screen).
- 1 INDETERMINATE — no usable rows inside the timeout; the run is left
  alive and the babysit registry is the backstop. A timeout WITH
  post-warmup rows still decides (a crawling run must not dodge the
  gate by never reaching the measurement window).

Usage (from the arm launcher, right after backgrounding the train):

    .venv/bin/python fontaine/scripts/attach_rate_gate.py \
        --jsonl outputs/train/<run>/train_log.jsonl \
        --arm-steps 10000 --ngpu 4 \
        --gate-gpu-hours 70 --extra-gpu-hours 36
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

PASS, INDETERMINATE, FALLBACK = 0, 1, 2


def parse_rows(text: str) -> list[tuple[int, float]]:
    """``(step, s_per_step)`` per parseable jsonl row, in file order.
    Rows missing either key (or unparseable — a torn tail write) are
    skipped, not errors."""
    rows: list[tuple[int, float]] = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
            rows.append((int(record["step"]), float(record["s_per_step"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return rows


def measured_rate(
    rows: list[tuple[int, float]],
    *,
    skip_steps: int,
) -> float | None:
    """Median s_per_step over rows past the warmup skip (compile +
    dataloader spin-up dilute the first rows' rate), or None if no row
    cleared it yet."""
    samples = [rate for step, rate in rows if step > skip_steps]
    if not samples:
        return None
    return statistics.median(samples)


def project_gpu_hours(
    rate_s_per_step: float,
    arm_steps: int,
    ngpu: int,
    extra_gpu_hours: float,
) -> float:
    """The pre-registered batch total: this arm at the measured rate
    plus the rest of the batch (other arm + frozen eval obligations)."""
    return rate_s_per_step * arm_steps * ngpu / 3600.0 + extra_gpu_hours


def decide(projected_gpu_hours: float, gate_gpu_hours: float) -> int:
    """Strict inequality, as frozen: only a projection strictly over
    the gate triggers the 5k downshift."""
    return FALLBACK if projected_gpu_hours > gate_gpu_hours else PASS


def run_gate(
    jsonl: Path,
    *,
    arm_steps: int,
    ngpu: int,
    gate_gpu_hours: float,
    extra_gpu_hours: float,
    min_step: int,
    skip_steps: int,
    poll_seconds: float,
    timeout_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Poll ``jsonl`` until a row at/past ``min_step`` exists, then
    decide. On timeout, decide from whatever post-warmup rows exist;
    INDETERMINATE only if none ever appeared."""
    deadline = clock() + timeout_seconds
    rows: list[tuple[int, float]] = []
    while clock() < deadline:
        rows = parse_rows(jsonl.read_text()) if jsonl.exists() else []
        if rows and rows[-1][0] >= min_step:
            break
        sleep(poll_seconds)
    else:
        if measured_rate(rows, skip_steps=skip_steps) is None:
            print(
                f"gate: INDETERMINATE — no post-warmup row in {jsonl} within "
                f"{timeout_seconds:.0f}s; run left alive, babysit registry "
                "is the backstop",
                flush=True,
            )
            return INDETERMINATE
        print(
            f"gate: timeout at step {rows[-1][0]} (< {min_step}) with "
            "post-warmup rows — deciding anyway",
            flush=True,
        )
    rate = measured_rate(rows, skip_steps=skip_steps)
    if rate is None:
        # Reached min_step before any post-warmup row exists — only
        # possible when skip_steps >= min_step, a launcher config error.
        print(
            f"gate: INDETERMINATE — rows reach step {rows[-1][0]} but none "
            f"past --skip-steps {skip_steps}; check the launcher's flags",
            flush=True,
        )
        return INDETERMINATE
    projected = project_gpu_hours(rate, arm_steps, ngpu, extra_gpu_hours)
    verdict = decide(projected, gate_gpu_hours)
    print(
        f"gate: {'FALLBACK' if verdict == FALLBACK else 'PASS'} — median "
        f"{rate:.3f} s/step x {arm_steps} steps x {ngpu} GPU "
        f"+ {extra_gpu_hours:.1f} extra -> projected batch "
        f"{projected:.1f} GPU-h (gate {gate_gpu_hours:.1f})",
        flush=True,
    )
    return verdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--arm-steps", type=int, required=True)
    parser.add_argument("--ngpu", type=int, default=4)
    parser.add_argument("--gate-gpu-hours", type=float, default=70.0)
    parser.add_argument(
        "--extra-gpu-hours",
        type=float,
        required=True,
        help="the rest of the pre-registered batch: other arm's "
        "measured/estimated train GPU-h + frozen eval obligations "
        "(the launcher pins the arithmetic)",
    )
    parser.add_argument("--min-step", type=int, default=200)
    parser.add_argument(
        "--skip-steps",
        type=int,
        default=60,
        help="rows at/below this step are warmup (compile, dataloader "
        "spin-up) and excluded from the rate",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_gate(
        args.jsonl,
        arm_steps=args.arm_steps,
        ngpu=args.ngpu,
        gate_gpu_hours=args.gate_gpu_hours,
        extra_gpu_hours=args.extra_gpu_hours,
        min_step=args.min_step,
        skip_steps=args.skip_steps,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
