"""Async rollout scheduler contract (bijou.rollout_async).

Pure CPU: the executor is tested against scripted timelines (it takes
every timestamp as an argument); the planner against a fake predict
with real (millisecond) thread latencies. No robot, no model, no
lerobot imports.
"""

from __future__ import annotations

import time

import pytest
import torch
from torch import Tensor

from bijou.rollout_async import AsyncExecutor, AsyncPlanner, PlannedChunk

CHUNK, DIM, FPS = 50, 6, 30.0


def make_chunk(fill: float) -> Tensor:
    return torch.full((CHUNK, DIM), fill)


def planned(fill: float, planned_at: float, replan_index: int = 1) -> PlannedChunk:
    return PlannedChunk(
        actions=make_chunk(fill),
        generations=None,
        planned_at=planned_at,
        replan_index=replan_index,
    )


def make_executor(blend_ticks: int = 0) -> AsyncExecutor:
    return AsyncExecutor(
        chunk_size=CHUNK,
        execute_horizon=40,
        fps=FPS,
        margin_ticks=3,
        blend_ticks=blend_ticks,
    )


def test_trigger_fires_exactly_at_latency_plus_margin() -> None:
    executor = make_executor()
    executor.start(planned(0.0, planned_at=0.0, replan_index=0))
    latency_ticks = 19  # ~625 ms at 30 Hz
    # Consume until remaining > latency + margin is no longer true:
    # trigger at index where 40 - index <= 19 + 3 = 22, i.e. index 18.
    fired_at: int | None = None
    for _ in range(40):
        if executor.wants_plan(latency_ticks, in_flight=False):
            fired_at = executor.index
            break
        executor.next_action()
    assert fired_at == 18
    # In-flight or parked plans gate re-triggering.
    assert not executor.wants_plan(latency_ticks, in_flight=True)
    executor.offer(planned(1.0, planned_at=0.6))
    assert not executor.wants_plan(latency_ticks, in_flight=False)


def test_early_arrival_waits_for_the_horizon_boundary() -> None:
    executor = make_executor()
    executor.start(planned(0.0, planned_at=0.0, replan_index=0))
    executor.offer(planned(1.0, planned_at=1.0))
    while executor.index < 40:
        assert executor.maybe_switch(now=1.3) is None  # cadence held
        executor.next_action()
    adopted = executor.maybe_switch(now=1.5)
    assert adopted is not None
    # Skip-ahead: (1.5 - 1.0) s * 30 Hz = 15.
    assert executor.last_skip_ahead == 15
    assert executor.index == 15
    row, starved = executor.next_action()
    assert not starved
    assert torch.equal(row, make_chunk(1.0)[15])


def test_skip_ahead_clamps_to_chunk_end() -> None:
    executor = make_executor()
    executor.start(planned(0.0, planned_at=0.0, replan_index=0))
    for _ in range(40):
        executor.next_action()
    executor.offer(planned(1.0, planned_at=0.0))
    executor.maybe_switch(now=10.0)  # absurd staleness
    assert executor.last_skip_ahead == CHUNK - 1


def test_starvation_holds_last_row_and_counts() -> None:
    executor = make_executor()
    executor.start(planned(2.0, planned_at=0.0, replan_index=0))
    for _ in range(CHUNK):
        _, starved = executor.next_action()
        assert not starved
    row, starved = executor.next_action()
    assert starved
    assert torch.equal(row, make_chunk(2.0)[CHUNK - 1])
    executor.next_action()
    assert executor.held_ticks == 2
    # Recovery: a plan arriving mid-starvation switches immediately
    # (index is already past the horizon).
    executor.offer(planned(3.0, planned_at=2.0))
    assert executor.maybe_switch(now=2.5) is not None
    row, starved = executor.next_action()
    assert not starved
    assert torch.equal(row, make_chunk(3.0)[15])


def test_switch_blend_crossfades_and_advances() -> None:
    executor = make_executor(blend_ticks=2)
    executor.start(planned(0.0, planned_at=0.0, replan_index=0))
    for _ in range(40):
        executor.next_action()  # index at the horizon boundary
    executor.offer(planned(1.0, planned_at=1.0))
    executor.maybe_switch(now=1.0)  # zero staleness: skip-ahead 0
    row1, _ = executor.next_action()
    row2, _ = executor.next_action()
    row3, _ = executor.next_action()
    # Old rows are all-0.0, new all-1.0: weights 1/3 then 2/3, then pure new.
    assert row1[0] == pytest.approx(1 / 3)
    assert row2[0] == pytest.approx(2 / 3)
    assert row3[0] == pytest.approx(1.0)
    assert executor.index == 3


def test_planner_roundtrip_latency_and_errors() -> None:
    def predict(
        item: dict[str, object],
        index: int,
    ) -> tuple[list[Tensor], None]:
        time.sleep(0.005)
        if item.get("boom") is not None:
            raise ValueError("planner must die loudly")
        return [make_chunk(float(index))], None

    planner = AsyncPlanner(predict)
    first = planner.warmup({}, replan_index=0)
    assert torch.equal(first.actions, make_chunk(0.0))
    assert planner.latency_ticks(FPS) >= 1  # 5 ms measured, ceil to a tick

    planner.submit({}, replan_index=1)
    assert planner.in_flight
    with pytest.raises(RuntimeError, match="already in flight"):
        planner.submit({}, replan_index=2)
    deadline = time.perf_counter() + 2.0
    result = None
    while result is None and time.perf_counter() < deadline:
        result = planner.poll()
    assert result is not None
    assert result.replan_index == 1
    assert not planner.in_flight

    planner.submit({"boom": True}, replan_index=3)
    deadline = time.perf_counter() + 2.0
    with pytest.raises(ValueError, match="loudly"):
        while time.perf_counter() < deadline:
            planner.poll()
            time.sleep(0.001)
    planner.shutdown()
