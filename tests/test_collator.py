"""Tests for the shared Collator core and the stats plumbing around it.

Pure CPU/synthetic: a fake InputsCollator stands in for the encoder
strategy, so no processor/checkpoint is needed. What's covered is the
trunk-agnostic half — camera policy, instruction override, NormStats
stacking (quantiles present / absent / mixed), and DatasetStats'
quantile lifecycle (required on the data path, Optional from old
checkpoint tables).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import pytest
import torch

from bijou.data import DatasetStats
from bijou.interface import CollatedBatch, Collator, PromptInputs

CHUNK, DIM = 4, 6


@dataclass(frozen=True, slots=True)
class FakeInputs:
    """Minimal BatchInputs: records what the strategy saw."""

    samples: tuple[PromptInputs, ...]

    def pin_memory(self) -> Self:
        return self

    def to(self, device: Any, *, non_blocking: bool = False) -> Self:
        return self

    def tensors(self) -> dict[str, torch.Tensor]:
        return {}


def fake_inputs_collator(samples: list[PromptInputs]) -> FakeInputs:
    return FakeInputs(samples=tuple(samples))


def stats(*, with_quantiles: bool) -> DatasetStats:
    return DatasetStats(
        action_mean=(0.0,) * DIM,
        action_std=(1.0,) * DIM,
        state_mean=(0.0,) * DIM,
        state_std=(1.0,) * DIM,
        action_q01=(-2.0,) * DIM if with_quantiles else None,
        action_q99=(2.0,) * DIM if with_quantiles else None,
        state_q01=(-2.0,) * DIM if with_quantiles else None,
        state_q99=(2.0,) * DIM if with_quantiles else None,
    )


def item(*, with_quantiles: bool, cameras: tuple[str, ...] = ("front",)) -> dict:
    payload: dict[str, Any] = {
        "task": "pick up the cube",
        "observation.state": torch.zeros(DIM),
        "action": torch.zeros(CHUNK, DIM),
        "action_is_pad": torch.zeros(CHUNK, dtype=torch.bool),
        **stats(with_quantiles=with_quantiles).item_tensors(),
    }
    for name in cameras:
        payload[f"observation.images.{name}"] = torch.rand(3, 8, 8)
    return payload


def collator(**overrides: Any) -> Collator[FakeInputs]:
    kwargs: dict[str, Any] = {
        "inputs": fake_inputs_collator,
        "instruction": None,
        "camera_filter": None,
        "max_cameras": None,
    }
    kwargs.update(overrides)
    return Collator(**kwargs)


def test_collates_stats_with_quantiles() -> None:
    batch = collator()([item(with_quantiles=True), item(with_quantiles=True)])
    assert isinstance(batch, CollatedBatch)
    assert batch.action_stats.mean.shape == (2, DIM)
    assert batch.action_stats.q01 is not None
    assert batch.action_stats.q99 is not None
    assert batch.state_stats.q01 is not None
    torch.testing.assert_close(
        batch.action_stats.q99,
        torch.full((2, DIM), 2.0),
    )


def test_collates_stats_without_quantiles() -> None:
    """Items built from an old checkpoint's stats table carry no quantile
    keys; the batch says so explicitly (None, not sentinels)."""
    batch = collator()([item(with_quantiles=False)])
    assert batch.action_stats.q01 is None
    assert batch.action_stats.q99 is None
    assert batch.state_stats.q01 is None


def test_mixed_quantile_batch_fails_loudly() -> None:
    with pytest.raises(ValueError, match="mixes items"):
        collator()([item(with_quantiles=True), item(with_quantiles=False)])


def test_camera_policy_and_instruction_override() -> None:
    sample_item = item(with_quantiles=True, cameras=("wrist", "front"))
    batch = collator(instruction="do the thing", max_cameras=1)([sample_item])
    (prompt,) = batch.encoder_inputs.samples
    assert prompt.instruction == "do the thing"
    # Sorted camera keys, then truncated by max_cameras: front wins.
    assert tuple(camera.name for camera in prompt.cameras) == ("front",)


def test_camera_filter_matches_bare_names() -> None:
    sample_item = item(with_quantiles=True, cameras=("wrist", "front"))
    batch = collator(camera_filter=("wrist",))([sample_item])
    (prompt,) = batch.encoder_inputs.samples
    assert tuple(camera.name for camera in prompt.cameras) == ("wrist",)


def test_dataset_stats_quantile_lifecycle() -> None:
    """state_dict round-trips quantiles when present and parses legacy
    tables (no quantile keys) to None; half-present quantiles are a
    construction error."""
    full = stats(with_quantiles=True)
    assert DatasetStats.from_state_dict(full.state_dict()) == full

    legacy_payload = {
        "action": {"mean": [0.0] * DIM, "std": [1.0] * DIM},
        "observation.state": {"mean": [0.0] * DIM, "std": [1.0] * DIM},
    }
    legacy = DatasetStats.from_state_dict(legacy_payload)
    assert legacy.action_q01 is None
    assert "q01" not in legacy.state_dict()["action"]

    with pytest.raises(ValueError, match="both present or both absent"):
        DatasetStats(
            action_mean=(0.0,) * DIM,
            action_std=(1.0,) * DIM,
            state_mean=(0.0,) * DIM,
            state_std=(1.0,) * DIM,
            action_q01=(-1.0,) * DIM,
            action_q99=None,
            state_q01=None,
            state_q99=None,
        )


def test_from_lerobot_stats_requires_quantiles() -> None:
    payload = {
        "action": {"mean": [0.0] * DIM, "std": [1.0] * DIM},
        "observation.state": {"mean": [0.0] * DIM, "std": [1.0] * DIM},
    }
    with pytest.raises(SystemExit, match="backfill"):
        DatasetStats.from_lerobot_stats(payload)
