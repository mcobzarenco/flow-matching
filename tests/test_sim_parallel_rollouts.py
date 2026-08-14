"""CPU-tier oracle for the parallel sim-rollout harness: the lockstep
scheduler + worker loop must produce EXACTLY the sequential loop's rows.

Both drivers share ``run_episode_loop``, so what this file pins is the
new surface: seed partitioning, the predict round-trip transport, batch
assembly order, early-success drop-out, and row collection. The fakes
are built so plumbing bugs are visible in the rows — the fake sim's
dynamics depend on the commanded actions, and the fake policy's chunk
depends on the observation bytes, so routing the wrong chunk (or the
wrong observation) to a worker changes every subsequent row field.

The batched-forward-vs-batch-1 GPU question is deliberately out of
scope here (GEMM reduction order moves with batch shape); that is the
registered smoke in fontaine/scripts/sim_parallel_oracle.py.
"""

import dataclasses
import queue
import threading
from collections.abc import Callable
from dataclasses import asdict
from hashlib import blake2b
from typing import Any

import numpy as np
import pytest

from sim.rollout_sim import hold_chunk_fn, run_episode_loop
from sim.rollout_sim_parallel import (
    WorkerConfig,
    WorkerDiedError,
    run_worker_episodes,
    serve,
)
from sim.so101_sim import SimObservation

REPLANS = 4
HORIZON = 10


class FakeSim:
    """Deterministic pure-python stand-in for SO101Sim. Distance shrinks
    as a function of the commanded actions; seeds start at different
    distances so some episodes succeed early (exercising the lockstep
    drop-out path) and others never do."""

    def __init__(self) -> None:
        self._seed = 0
        self._tick = 0
        self._state = np.zeros(6)
        self._distance = 1.0
        self.reset_spawn_xy = (0.0, 0.0)
        self.reset_strike_contacts = 0

    def reset(self, seed: int) -> SimObservation:
        self._seed = seed
        self._tick = 0
        self._state = np.linspace(-1.0, 1.0, 6) * (1 + seed % 7)
        self._distance = 0.06 + 0.02 * (seed % 4)
        self.reset_spawn_xy = (0.1 + 0.001 * seed, -0.05 + 0.002 * seed)
        self.reset_strike_contacts = seed % 2
        return self._observe()

    def _observe(self) -> SimObservation:
        def frame(salt: int) -> np.ndarray:
            base = self._seed * 7919 + self._tick * 104729 + salt
            values = (base + np.arange(4 * 4 * 3)) % 256
            return values.reshape(4, 4, 3).astype(np.uint8)

        return SimObservation(top=frame(0), wrist=frame(31), state=self._state.copy())

    def step(self, action_degrees: np.ndarray) -> SimObservation:
        self._tick += 1
        action = np.asarray(action_degrees, dtype=float)
        self._state = self._state + 0.01 * action
        self._distance = max(
            0.0,
            self._distance - 0.002 * (1.0 + np.tanh(float(action.sum()))),
        )
        return self._observe()

    def benchy_disk_distance(self) -> float:
        return self._distance

    def success(self) -> bool:
        return self._distance < 0.045

    def benchy_pose(self) -> tuple[np.ndarray, float]:
        position = np.array([0.2, 0.1, 0.005 + 0.0001 * self._seed])
        return position, 0.9 + 0.001 * (self._seed % 5)

    def benchy_grip_contacts(self) -> tuple[bool, bool]:
        # deterministic off the same state the distance rides, so the
        # grip trace exercises all codes across the oracle's seeds
        return self._distance < 0.055, self._distance < 0.05


def fake_chunk(
    seed: int,
    replan: int,
    obs: SimObservation,
    draw: int = 0,
) -> np.ndarray:
    """Deterministic per-item 'policy': keyed off the identity pair AND
    the observation bytes AND the draw (the real policy re-keys its
    noise per draw through the repo_id suffix), and batch-composition
    invariant by construction — the same property the real stable-key
    noise has."""
    digest = blake2b(
        b"|".join(
            [
                str(seed).encode(),
                str(replan).encode(),
                str(draw).encode(),
                obs.top.tobytes(),
                obs.wrist.tobytes(),
                obs.state.tobytes(),
            ],
        ),
        digest_size=8,
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    return rng.normal(size=(HORIZON, 6))


def fake_predict_batch(requests: list[tuple[Any, ...]]) -> list[np.ndarray]:
    chunks = []
    for _, _, seed, replan, draw, top, wrist, state in requests:
        obs = SimObservation(top=top, wrist=wrist, state=state)
        chunks.append(fake_chunk(seed, replan, obs, draw))
    return chunks


def sequential_rows(
    seeds: list[int],
    *,
    hold: bool = False,
    draws: int = 1,
) -> list[Any]:
    """Seed-major, draw-minor — the sequential driver's loop order."""
    rows = []
    for seed in seeds:
        for draw in range(draws):
            sim = FakeSim()
            latencies: list[float] = []
            next_chunk: Callable[[SimObservation, int], np.ndarray]
            if hold:
                next_chunk = hold_chunk_fn(HORIZON)
            else:

                def scripted_chunk(
                    obs: SimObservation,
                    replan: int,
                    _seed: int = seed,
                    _draw: int = draw,
                ) -> np.ndarray:
                    return fake_chunk(_seed, replan, obs, _draw)

                next_chunk = scripted_chunk

            row = run_episode_loop(
                sim,
                seed,
                next_chunk,
                replans=REPLANS,
                horizon=HORIZON,
                video_path=None,
                latencies=latencies,
            )
            if draw:
                row = dataclasses.replace(row, draw=draw)
            rows.append(row)
    return rows


class QueueConn:
    """A Pipe stand-in over two queues (worker-side send/recv callables
    on one end, .send/.recv for the scheduler on the other)."""

    def __init__(self) -> None:
        self.to_parent: queue.Queue[Any] = queue.Queue()
        self.to_worker: queue.Queue[Any] = queue.Queue()

    # scheduler side
    def recv(self) -> Any:
        return self.to_parent.get(timeout=30)

    def send(self, chunk: Any) -> None:
        self.to_worker.put(chunk)


def parallel_rows(
    seeds: list[int],
    workers: int,
    *,
    hold: bool = False,
    draws: int = 1,
    predict_batch: Any = fake_predict_batch,
) -> tuple[list[Any], list[int]]:
    """Drive the PRODUCTION scheduler + worker loop in-process: worker
    threads run run_worker_episodes over QueueConn transports while the
    main thread runs serve()."""
    units = [(seed, draw) for seed in seeds for draw in range(draws)]
    conns = [QueueConn() for _ in range(workers)]
    threads = []
    for worker_id, conn in enumerate(conns):
        config = WorkerConfig(
            worker_id=worker_id,
            units=tuple(units[worker_id::workers]),
            replans=REPLANS,
            horizon=HORIZON,
            hold=hold,
            out_dir=None,
            post_backend="numpy",
        )
        thread = threading.Thread(
            target=run_worker_episodes,
            args=(
                FakeSim(),
                config,
                conn.to_parent.put,
                lambda _c=conn: _c.to_worker.get(timeout=30),
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    rows: list[Any] = []
    batch_sizes = serve(conns, predict_batch, rows.append)
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    rows.sort(key=lambda r: (r.seed, r.draw))
    return rows, batch_sizes


def comparable(row: Any) -> dict[str, Any]:
    """Row minus latency_ms — the one field that measures wall clock
    (sequential: solo predict; parallel: round-trip incl. the barrier)."""
    fields = asdict(row)
    fields.pop("latency_ms")
    return fields


def test_parallel_rows_match_sequential() -> None:
    seeds = list(range(10))
    expected = sequential_rows(seeds)
    actual, _ = parallel_rows(seeds, workers=3)
    assert [comparable(r) for r in actual] == [comparable(r) for r in expected]


def test_parallel_rows_match_sequential_single_worker() -> None:
    seeds = [3, 4, 5]
    expected = sequential_rows(seeds)
    actual, _ = parallel_rows(seeds, workers=1)
    assert [comparable(r) for r in actual] == [comparable(r) for r in expected]


def test_parallel_draws_match_sequential() -> None:
    """(seed, draw) work units: the parallel driver must produce exactly
    the sequential seed-major/draw-minor rows, with each unit's chunks
    routed under its OWN draw (the fake policy keys on draw, so a
    draw-routing bug changes every downstream row field)."""
    seeds = list(range(5))
    expected = sequential_rows(seeds, draws=3)
    actual, _ = parallel_rows(seeds, workers=3, draws=3)
    assert [comparable(r) for r in actual] == [comparable(r) for r in expected]
    assert [r.draw for r in actual] == [d for _ in seeds for d in range(3)]


def test_hold_arm_runs_worker_local() -> None:
    seeds = list(range(6))
    expected = sequential_rows(seeds, hold=True)

    def refuse(requests: list[tuple[Any, ...]]) -> list[np.ndarray]:
        raise AssertionError("hold arm must not request predicts")

    actual, batch_sizes = parallel_rows(
        seeds,
        workers=2,
        hold=True,
        predict_batch=refuse,
    )
    assert [comparable(r) for r in actual] == [comparable(r) for r in expected]
    assert batch_sizes == []


def test_early_success_drops_out_of_the_batch() -> None:
    """Per-seed predict counts must equal the sequential episode's replan
    count (success stops the requests), and the round trace must be the
    lockstep one: round r's batch = workers still running an episode."""
    seeds = list(range(8))
    workers = 4
    requested: list[tuple[int, int]] = []

    def tracing_predict(requests: list[tuple[Any, ...]]) -> list[np.ndarray]:
        requested.extend((m[2], m[3]) for m in requests)
        return fake_predict_batch(requests)

    expected = sequential_rows(seeds)
    _, batch_sizes = parallel_rows(
        seeds,
        workers=workers,
        predict_batch=tracing_predict,
    )

    expected_counts = {
        row.seed: (
            REPLANS if row.success_tick is None else row.success_tick // HORIZON + 1
        )
        for row in expected
    }
    actual_counts: dict[int, int] = {}
    for seed, _replan in requested:
        actual_counts[seed] = actual_counts.get(seed, 0) + 1
    assert actual_counts == expected_counts
    assert sum(batch_sizes) == sum(expected_counts.values())
    assert batch_sizes[0] == workers
    # The trace is deterministic: a second run reproduces it exactly.
    _, batch_sizes_again = parallel_rows(seeds, workers=workers)
    assert batch_sizes_again == batch_sizes


def test_worker_error_propagates() -> None:
    conn = QueueConn()
    conn.to_parent.put(("error", 0, "synthetic traceback"))
    with pytest.raises(WorkerDiedError, match="synthetic traceback"):
        serve([conn], fake_predict_batch, lambda row: None)
