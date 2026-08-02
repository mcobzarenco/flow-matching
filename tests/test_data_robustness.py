"""StatsAttachedDataset's fetch robustness: substitution + circuit breaker.

Pure CPU/synthetic: a stub stands in for LeRobotDataset. The contract
under test (grown from two killed multi-hour runs and the 2026-08-02
curation field report on timestamp-desynced episodes): a RARE
unfetchable item is substituted deterministically and loudly, bounded
retries keep fully-broken datasets fatal, and the rate-based circuit
breaker keeps SCATTERED systematic breakage from degrading a long run
behind scrolling substitution prints.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from bijou.data import DatasetStats, EpisodeAnnotations, StatsAttachedDataset

DIM = 6


class StubDataset:
    """Duck-typed LeRobotDataset: items for good indices, a
    tolerance-style ValueError for bad ones."""

    repo_id = "user/stub"

    def __init__(self, length: int, bad: set[int]) -> None:
        self.length = length
        self.bad = bad

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index in self.bad:
            raise ValueError(
                "One or several query timestamps unexpectedly violate the "
                "tolerance (stub)",
            )
        return {"index": index, "episode_index": torch.tensor(0)}


def stats() -> DatasetStats:
    return DatasetStats(
        action_mean=(0.0,) * DIM,
        action_std=(1.0,) * DIM,
        state_mean=(0.0,) * DIM,
        state_std=(1.0,) * DIM,
        action_q01=(-1.0,) * DIM,
        action_q99=(1.0,) * DIM,
        state_q01=(-1.0,) * DIM,
        state_q99=(1.0,) * DIM,
    )


def wrap(stub: StubDataset) -> StatsAttachedDataset:
    # cast: the stub duck-types the (untyped-at-runtime) LeRobotDataset
    # surface StatsAttachedDataset touches (repo_id, __len__, __getitem__).
    return StatsAttachedDataset(
        cast(LeRobotDataset, stub),
        stats(),
        {"front": "top"},
        {0: EpisodeAnnotations(("rewrite",), "success", "high")},
    )


def test_rare_bad_item_is_substituted_loudly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = wrap(StubDataset(20000, bad={5}))
    item = dataset[5]
    # Substituted with the stride-hop index, decorated like any item.
    assert item["index"] == (5 + 9973) % 20000
    assert item["repo_id"] == "user/stub"
    assert item["camera_kinds"] == {"front": "top"}
    assert item["condition_outcome"] == "success"
    assert dataset.failed_fetches == 1
    assert "unfetchable" in capsys.readouterr().err


def test_fully_broken_dataset_stays_fatal() -> None:
    dataset = wrap(StubDataset(50, bad=set(range(50))))
    with pytest.raises(ValueError, match="tolerance"):
        dataset[0]


def test_scattered_breakage_trips_the_circuit_breaker() -> None:
    """Every first try fails, every substitution succeeds — bounded
    retries never exhaust, so without the breaker this would degrade
    forever behind prints. The rate check aborts instead."""
    bad = {i for i in range(20000) if i % 7 == 0}
    assert (9973 % 7) != 0  # substitution hop lands on a good index
    dataset = wrap(StubDataset(20000, bad=bad))
    with pytest.raises(RuntimeError, match="systematic"):
        for _ in range(200):
            dataset[0]
    assert dataset.total_fetches >= dataset._BREAKER_MIN_FETCHES
