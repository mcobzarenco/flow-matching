"""Overlapped (async) planning for the rollout control loop.

Sync rollout freezes the arm for one inference latency per replan
(~625 ms on the 8 GiB laptop via --offload-ple, ~32% of wall time at
horizon 40 / 30 Hz). The chunk structure already contains the slack to
hide it: only ``execute_horizon`` of ``chunk_size`` actions are ever
executed, so planning can run while the current chunk's tail executes.

Two pieces, split for testability:

- :class:`AsyncPlanner` — one background inference at a time on a
  single-worker thread (torch releases the GIL inside kernels; the
  control loop's per-tick work is microseconds). Latency is tracked
  from observed predictions and drives the trigger. Thread exceptions
  re-raise at ``poll()`` — a dead planner must kill the rollout loudly,
  not starve it silently.
- :class:`AsyncExecutor` — the tick-level state machine (no I/O): when
  to trigger a plan, when to switch chunks (at the horizon boundary,
  never on arrival — the replan cadence stays identical to sync mode,
  only the freeze disappears), the skip-ahead index that re-aligns a
  fresh chunk with the arm's actual progress, starvation holds, and
  the optional switch crossfade.

Skip-ahead counts EXECUTED motion ticks since the plan's observation,
not wall time: held ticks advance the clock but not the arm, and a
wall-clock skip after holds adopts far-future rows the arm never
traveled toward — field-tested as a max_relative_target
clamp storm ("chaotic fast motion"): 42-tick latency at 30 Hz produced
switches at rows 45-47/50 with 38-43 held ticks between.

Sustainability: a plan is one latency stale on arrival, so each chunk
yields ``chunk_size - latency_ticks`` fresh rows per latency —
freeze-free 30 Hz needs ``2*latency + margin <= chunk_size`` (~0.75 s
at chunk 50). Above that NO schedule works at full fps; async refuses
to start (sync mode or a lower --fps are the remedies).

Deployment shift note: a chunk is consumed from index
``round(latency x fps)`` — the model trained on chunks executed from
index 0, but the skipped prefix corresponds to motion that already
happened; this is the standard async-VLA staleness trade (SmolVLA's
async stack; pi0's real-time chunking inpaints the overlap instead —
the escalation path if switch seams look rough on the arm).
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from .aux_text import AuxGeneration

PredictFn = Callable[
    [dict[str, Any], int],
    tuple[list[Tensor], list[AuxGeneration] | None],
]


@dataclass(frozen=True, slots=True)
class PlannedChunk:
    """One inference result: the chunk, its narration, and WHEN the
    observation it was planned from was taken (``planned_at``, the
    submit time under the planner's clock — drives the skip-ahead)."""

    actions: Tensor
    generations: list[AuxGeneration] | None
    planned_at: float
    replan_index: int


class AsyncPlanner:
    """At most one inference in flight; latency measured per prediction.

    ``clock`` is injectable for tests; the worker thread reads the same
    clock so latencies stay consistent under a fake one.
    """

    def __init__(
        self,
        predict: PredictFn,
        *,
        clock: Callable[[], float] = time.perf_counter,
        latency_window: int = 16,
    ) -> None:
        self._predict = predict
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: Future[tuple[PlannedChunk, float]] | None = None
        self._latencies: deque[float] = deque(maxlen=latency_window)

    @property
    def in_flight(self) -> bool:
        return self._pending is not None

    def warmup(
        self,
        item: dict[str, Any],
        replan_index: int,
        *,
        extra_calls: int = 3,
    ) -> PlannedChunk:
        """Synchronous predict on the CALLING thread: pays first-call
        kernel compilation (measured 2.5-5 s on the laptop) before the
        control loop starts, then keeps predicting until the latency
        settles and seeds the window with the LAST measurement only —
        call 2 still reads ~2x steady state (autotune tail; measured
        1.33 s vs 0.63 s steady), and an over-seeded window means early
        triggers, which means staler chunks at every switch."""
        start = self._clock()
        chunks, generations = self._predict(item, replan_index)
        first = PlannedChunk(
            actions=chunks[0],
            generations=generations,
            planned_at=start,
            replan_index=replan_index,
        )
        # Warm calls on the same item (contents are irrelevant to
        # timing; results discarded).
        latency = 0.0
        for _ in range(extra_calls):
            start = self._clock()
            self._predict(item, replan_index)
            latency = self._clock() - start
        self._latencies.append(latency)
        return first

    def submit(self, item: dict[str, Any], replan_index: int) -> None:
        if self._pending is not None:
            raise RuntimeError("a plan is already in flight (submit gated on idle)")
        start = self._clock()

        def run() -> tuple[PlannedChunk, float]:
            chunks, generations = self._predict(item, replan_index)
            end = self._clock()
            return (
                PlannedChunk(
                    actions=chunks[0],
                    generations=generations,
                    planned_at=start,
                    replan_index=replan_index,
                ),
                end - start,
            )

        self._pending = self._executor.submit(run)

    def poll(self) -> PlannedChunk | None:
        """The finished plan, or None while still computing. Re-raises
        worker exceptions (a dead planner must not read as starvation)."""
        if self._pending is None or not self._pending.done():
            return None
        planned, latency = self._pending.result()
        self._pending = None
        self._latencies.append(latency)
        return planned

    def latency_ticks(self, fps: float) -> int:
        """p95 of observed latencies, in control ticks (conservative:
        the trigger would rather plan one tick early than starve)."""
        ordered = sorted(self._latencies)
        p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
        return math.ceil(p95 * fps)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class AsyncExecutor:
    """Tick-level chunk bookkeeping. All times are passed in (pure
    against the clock); all decisions are unit-tested against scripted
    timelines in tests/test_rollout_async.py."""

    def __init__(
        self,
        *,
        chunk_size: int,
        execute_horizon: int,
        fps: float,
        margin_ticks: int,
        blend_ticks: int,
    ) -> None:
        self.chunk_size = chunk_size
        self.execute_horizon = execute_horizon
        self.fps = fps
        self.margin_ticks = margin_ticks
        self.blend_ticks = blend_ticks
        self.chunk: Tensor | None = None
        self.index = 0
        self.pending: PlannedChunk | None = None
        self.switches = 0
        self.held_ticks = 0
        self.last_skip_ahead = 0
        self.last_staleness_skew = 0
        self._blend_rows: deque[Tensor] = deque()
        # Motion ticks executed since the in-flight plan's observation
        # (holds excluded) — the skip-ahead basis; None = nothing
        # submitted yet.
        self._executed_since_submit: int | None = None

    def start(self, planned: PlannedChunk) -> None:
        """Cold start: the episode's first chunk executes from index 0
        (the arm was idle during its inference — no skip-ahead)."""
        self.chunk = planned.actions
        self.index = 0

    def wants_plan(self, latency_ticks: int, *, in_flight: bool) -> bool:
        """Trigger when the remaining pre-horizon actions just cover the
        measured latency plus margin."""
        if self.chunk is None or in_flight or self.pending is not None:
            return False
        remaining = self.execute_horizon - self.index
        return remaining <= latency_ticks + self.margin_ticks

    def note_submit(self) -> None:
        """Start counting executed motion against the just-submitted
        plan's observation (call alongside planner.submit)."""
        self._executed_since_submit = 0

    def offer(self, planned: PlannedChunk | None) -> None:
        """Park a finished plan until the horizon boundary — early
        arrivals wait so the replan cadence matches sync mode."""
        if planned is not None:
            self.pending = planned

    def maybe_switch(self, now: float) -> PlannedChunk | None:
        """At/after the horizon boundary, adopt the pending chunk at the
        EXECUTED-ticks skip (the arm's actual progress since the plan's
        observation — holds advance wall time but not the arm; wall-
        clock skipping after holds lunges at far-future rows). The
        wall-vs-executed skew is kept for logging: persistent skew =
        the loop is not holding its fps."""
        if self.pending is None or self.index < self.execute_horizon:
            return None
        planned = self.pending
        self.pending = None
        executed = self._executed_since_submit or 0
        self._executed_since_submit = None
        wall = round((now - planned.planned_at) * self.fps)
        self.last_skip_ahead = max(0, min(executed, self.chunk_size - 1))
        self.last_staleness_skew = wall - executed
        if self.blend_ticks > 0 and self.chunk is not None:
            self._stage_blend(planned.actions)
        self.chunk = planned.actions
        self.index = self.last_skip_ahead
        self.switches += 1
        return planned

    def _stage_blend(self, incoming: Tensor) -> None:
        """Linear crossfade rows for the switch seam, consumed before
        regular rows. Uses whatever old-chunk tail actually remains."""
        assert self.chunk is not None
        self._blend_rows.clear()
        for j in range(self.blend_ticks):
            old_row = self.index + j
            new_row = self.last_skip_ahead + j
            if old_row >= self.chunk_size or new_row >= self.chunk_size:
                break
            weight = (j + 1) / (self.blend_ticks + 1)
            self._blend_rows.append(
                (1.0 - weight) * self.chunk[old_row] + weight * incoming[new_row],
            )

    def next_action(self) -> tuple[Tensor, bool]:
        """(action row, starved). Starved = the chunk is fully exhausted
        and no replacement arrived: hold the last row (the arm keeps its
        position) and count the tick — callers print loudly. Held ticks
        do NOT count as executed motion (the skip-ahead basis)."""
        assert self.chunk is not None  # start() precedes the loop
        if self._blend_rows:
            self.index += 1
            if self._executed_since_submit is not None:
                self._executed_since_submit += 1
            return self._blend_rows.popleft(), False
        if self.index >= self.chunk_size:
            self.held_ticks += 1
            return self.chunk[self.chunk_size - 1], True
        row = self.chunk[self.index]
        self.index += 1
        if self._executed_since_submit is not None:
            self._executed_since_submit += 1
        return row, False


def sustainable(chunk_size: int, latency_ticks: int, margin_ticks: int) -> bool:
    """Freeze-free pipelining bound: a plan is one latency stale on
    arrival, so each chunk yields chunk_size - latency fresh rows per
    latency of execution."""
    return 2 * latency_ticks + margin_ticks <= chunk_size
