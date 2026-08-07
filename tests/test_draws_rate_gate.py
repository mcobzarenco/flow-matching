"""Oracles for the mechanized draws cost gate (ideas #19).

The pre-reg (2026-08-06-prereg-ar-sampled-draws.md) froze the gate
BEFORE launch: measure the rate over the first ~200 frames; a
full-panel draws10 projecting > 24 GPU-h drops to the q4 subset. These
tests pin the measurement's arithmetic and the polling loop's decision
protocol so the launch-time branch is pre-registered code, not
judgment: progress-line parsing (DDP rank-0-shard suffix included),
the shard→GPU-hours projection, the strict-inequality threshold, and
the loop's three exits (PASS / FALLBACK / INDETERMINATE — including
the timeout-with-partial-progress case, where a run too slow to reach
the measurement window must still be decided, not left running).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fontaine.scripts.draws_rate_gate import (
    FALLBACK,
    INDETERMINATE,
    PASS,
    decide,
    parse_progress,
    project_gpu_hours,
    run_gate,
)


def test_parse_progress_ddp_suffix_and_last_match() -> None:
    text = (
        "loading checkpoint...\n"
        "  scored 20/6450 frames (rank 0 shard)\n"
        "  scored 60/6450 frames (rank 0 shard)\n"
    )
    assert parse_progress(text) == (60, 6450)


def test_parse_progress_single_process_form() -> None:
    assert parse_progress("  scored 320/25800 frames\n") == (320, 25800)


def test_parse_progress_absent() -> None:
    assert parse_progress("dataset scan: 878 datasets\n") is None


def test_projection_shard_to_gpu_hours() -> None:
    # 100 shard frames in 300 s, shard total 6450, 4 GPUs:
    # 300/100 * 6450 s wall = 5.375 h -> 21.5 GPU-h.
    projected = project_gpu_hours((0.0, 40), (300.0, 140), 6450, 4)
    assert projected is not None
    assert abs(projected - 21.5) < 1e-9


def test_projection_requires_forward_progress() -> None:
    assert project_gpu_hours((0.0, 40), (300.0, 40), 6450, 4) is None
    assert project_gpu_hours((0.0, 40), (0.0, 90), 6450, 4) is None


def test_decide_strict_inequality_at_the_gate() -> None:
    assert decide(24.0, 24.0) == PASS  # pre-reg says "> 24 GPU-h"
    assert decide(24.0 + 1e-9, 24.0) == FALLBACK
    assert decide(21.5, 24.0) == PASS


class FakeEval:
    """Deterministic clock + log writer: each sleep() advances the
    clock by the requested amount and appends the next pending line
    (once exhausted, the log just stops growing)."""

    def __init__(self, log: Path, lines: list[str]) -> None:
        self.log = log
        self.now = 0.0
        self.pending = list(lines)

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.pending:
            with self.log.open("a") as handle:
                handle.write(self.pending.pop(0))


def gate(log: Path, fake: FakeEval, timeout: float = 10_000.0) -> int:
    return run_gate(
        log,
        ngpu=4,
        gate_gpu_hours=24.0,
        min_frames=200,  # ceil(200/4) = 50 shard frames past baseline
        poll_seconds=30.0,
        timeout_seconds=timeout,
        clock=fake.clock,
        sleep=fake.sleep,
    )


def test_run_gate_pass(tmp_path: Path) -> None:
    # 20 shard frames per 30 s poll: ~10.75 GPU-h projected — PASS.
    log = tmp_path / "eval.log"
    lines = [f"  scored {n}/6450 frames (rank 0 shard)\n" for n in range(20, 201, 20)]
    assert gate(log, FakeEval(log, lines)) == PASS


def test_run_gate_fallback(tmp_path: Path) -> None:
    # 5 shard frames per 30 s poll: 6450/10 f/min wall -> 43 GPU-h — FALLBACK.
    log = tmp_path / "eval.log"
    lines = [f"  scored {n}/6450 frames (rank 0 shard)\n" for n in range(20, 101, 5)]
    assert gate(log, FakeEval(log, lines)) == FALLBACK


def test_run_gate_timeout_with_partial_progress_still_decides(
    tmp_path: Path,
) -> None:
    # Only 10 shard frames ever appear — the measurement window is
    # never reached, but the run must not dodge the gate by crawling:
    # 60 s / 10 frames -> 43 GPU-h — FALLBACK at timeout.
    log = tmp_path / "eval.log"
    lines = [
        "  scored 20/6450 frames (rank 0 shard)\n",
        "  scored 25/6450 frames (rank 0 shard)\n",
        "  scored 30/6450 frames (rank 0 shard)\n",
    ]
    assert gate(log, FakeEval(log, lines), timeout=100.0) == FALLBACK


def test_run_gate_indeterminate_when_no_progress_ever(tmp_path: Path) -> None:
    log = tmp_path / "eval.log"
    assert gate(log, FakeEval(log, []), timeout=100.0) == INDETERMINATE
