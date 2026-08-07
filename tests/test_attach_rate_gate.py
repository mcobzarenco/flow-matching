"""Oracles for the mechanized attach-screen cost gate (#4 launch prep).

The pre-reg (2026-08-07-prereg-molmo2-attach-screen.md) froze the gate
before launch: first ~200 steps of each arm project the batch total;
strictly > 70 GPU-h ⇒ both arms downshift to 5k matched steps. These
tests pin the measurement's arithmetic and the polling loop's decision
protocol: jsonl-row parsing (torn tails skipped), the warmup skip, the
MEDIAN rate (a save-boundary outlier row must not flip the decision),
the projection including the batch's --extra-gpu-hours term, the
strict-inequality threshold, and the loop's three exits (PASS /
FALLBACK / INDETERMINATE, including timeout-with-partial-rows, where a
crawling run is decided, not left to dodge the gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fontaine.scripts.attach_rate_gate import (
    FALLBACK,
    INDETERMINATE,
    PASS,
    decide,
    measured_rate,
    parse_rows,
    project_gpu_hours,
    run_gate,
)


def row(step: int, s_per_step: float) -> str:
    return f'{{"step": {step}, "loss": 3.5, "s_per_step": {s_per_step}}}'


def test_parse_rows_skips_torn_and_alien_lines() -> None:
    text = "\n".join(
        [
            row(20, 2.1),
            '{"step": 40, "loss": 3.5, "s_per_st',  # torn tail write
            '{"event": "save", "path": "step_002500"}',  # no rate key
            row(60, 2.2),
        ],
    )
    assert parse_rows(text) == [(20, 2.1), (60, 2.2)]


def test_measured_rate_is_post_warmup_median() -> None:
    rows = [(20, 9.0), (40, 9.0), (80, 2.0), (100, 2.2), (120, 60.0)]
    # Warmup rows (<= 60) excluded; the 60 s/step save-boundary outlier
    # must not drag the rate — median of [2.0, 2.2, 60.0] = 2.2.
    assert measured_rate(rows, skip_steps=60) == 2.2
    assert measured_rate(rows[:2], skip_steps=60) is None


def test_projection_includes_the_batch_extra_term() -> None:
    # 2.5 s/step x 10k steps x 4 GPU = 27.78 GPU-h + 36 extra = 63.78.
    projected = project_gpu_hours(2.5, 10_000, 4, 36.0)
    assert abs(projected - (2.5 * 10_000 * 4 / 3600 + 36.0)) < 1e-9


def test_decision_is_strict_inequality() -> None:
    assert decide(70.0, 70.0) == PASS
    assert decide(70.0001, 70.0) == FALLBACK


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def gate(jsonl: Path, clock: FakeClock, **overrides: float) -> int:
    kwargs: dict[str, float | int] = {
        "arm_steps": 10_000,
        "ngpu": 4,
        "gate_gpu_hours": 70.0,
        "extra_gpu_hours": 36.0,
        "min_step": 200,
        "skip_steps": 60,
        "poll_seconds": 30.0,
        "timeout_seconds": 600.0,
    }
    kwargs.update(overrides)
    return run_gate(jsonl, clock=clock, sleep=clock.sleep, **kwargs)  # type: ignore[arg-type]


def test_loop_pass_at_measurement_window(tmp_path: Path) -> None:
    jsonl = tmp_path / "train_log.jsonl"
    jsonl.write_text("\n".join(row(s, 2.5) for s in range(20, 221, 20)))
    # 2.5 s/step -> 27.8 + 36 extra = 63.8 <= 70.
    assert gate(jsonl, FakeClock()) == PASS


def test_loop_fallback_over_the_gate(tmp_path: Path) -> None:
    jsonl = tmp_path / "train_log.jsonl"
    jsonl.write_text("\n".join(row(s, 3.2) for s in range(20, 221, 20)))
    # 3.2 s/step -> 35.6 + 36 extra = 71.6 > 70.
    assert gate(jsonl, FakeClock()) == FALLBACK


def test_loop_timeout_with_partial_rows_still_decides(tmp_path: Path) -> None:
    jsonl = tmp_path / "train_log.jsonl"
    jsonl.write_text("\n".join(row(s, 3.2) for s in range(20, 121, 20)))
    # Never reaches min_step 200 — but post-warmup rows exist, so the
    # crawling run is decided (FALLBACK), not left running.
    assert gate(jsonl, FakeClock()) == FALLBACK


def test_loop_indeterminate_without_rows(tmp_path: Path) -> None:
    assert gate(tmp_path / "never_written.jsonl", FakeClock()) == INDETERMINATE
