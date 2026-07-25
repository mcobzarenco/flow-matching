"""Chunk-prediction policies for open-loop evaluation.

Every policy consumes raw LeRobot items (with per-dataset stats attached by
``StatsAttachedDataset``) in batches and returns per-item action chunks in
raw action units, [chunk, action_dim] each. Batching is up to the policy —
the runner hands over the same items to every policy, so comparisons are
paired by construction.

Determinism: policies that sample flow-matching noise derive it per item
from ``seed + global_index`` on CPU, so predictions are independent of batch
composition, evaluation order and device.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from ..data import PrefixCollator, encode_prefix
from ..gemma4.loading import resolve_checkpoint_dir
from ..loading import CheckpointInfo, from_checkpoint
from ..model import BijouModel, SamplingMethod


class ChunkPolicy(Protocol):
    """A named policy that predicts action chunks for a batch of items."""

    name: str

    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        """Raw-unit chunk predictions, one [chunk, action_dim] per item."""
        ...


class StateCopyPolicy:
    """The trivial baseline: hold the current joint positions for the whole
    chunk. Near-optimal on quasi-static segments; collapses on motion —
    beating it in aggregate is the minimum bar for a learned policy."""

    name = "state-copy"

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        return [
            item["observation.state"][None, :].expand(self.chunk_size, -1).clone()
            for item in items
        ]


class NormalizedStateCopyPolicy:
    """State-copy through the per-dataset stats: normalize the state with the
    STATE stats, unnormalize with the ACTION stats —
    ``â = μ_a + σ_a · (s − μ_s)/σ_s``, held for the whole chunk.

    On SO-100 teleop data, action (leader/commanded) and state (follower/
    measured) differ by quasi-constant offsets: gravity droop on loaded
    joints, leader-follower calibration deltas, gripper clamp force. Those
    offsets live in μ_a − μ_s, so this baseline hits offset-but-constant
    traces that raw state-copy misses — with zero learning. It is the
    correct trivial policy under per-dataset normalization, and the honest
    reference for how much of a learned model's edge is stats arithmetic."""

    name = "state-copy-norm"

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        predictions: list[Tensor] = []
        for item in items:
            normalized = (item["observation.state"] - item["state_mean"]) / item[
                "state_std"
            ]
            action = normalized * item["action_std"] + item["action_mean"]
            predictions.append(action[None, :].expand(self.chunk_size, -1).clone())
        return predictions


def sample_noise(seed: int, shape: tuple[int, ...]) -> Tensor:
    """Seeded on CPU so values are identical regardless of device."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


class BijouPolicy:
    """A bijou training checkpoint. Normalization is per dataset with the
    stats attached to each item (identical to training; works on held-out
    datasets because stats travel with the data, not the checkpoint)."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: torch.device,
        seed: int,
        sample_steps: int = 10,
        method: SamplingMethod = SamplingMethod.HEUN,
    ) -> None:
        self.name = f"bijou@{checkpoint.name.removeprefix('step_').lstrip('0') or '0'}"
        self.device = device
        self.seed = seed
        self.sample_steps = sample_steps
        self.method = method
        self.model: BijouModel
        self.info: CheckpointInfo
        self.model, self.info = from_checkpoint(checkpoint, device=device)
        self.collator = PrefixCollator(
            checkpoint=str(resolve_checkpoint_dir(self.info.backbone)),
            instruction=None,
            max_soft_tokens=self.info.max_soft_tokens,
        )

    @torch.no_grad()
    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        batch = self.collator(items).to(self.device)
        prefix = encode_prefix(self.model, batch)
        state = (batch.state - batch.state_mean) / batch.state_std
        chunk = batch.actions.shape[1]
        noise = torch.stack(
            [
                sample_noise(self.seed + index, (chunk, batch.actions.shape[2]))
                for index in indices
            ]
        ).to(self.device)
        sampled = self.model.sample_actions(
            prefix,
            state,
            num_steps=self.sample_steps,
            method=self.method,
            noise=noise,
        )
        raw = (
            sampled.float() * batch.action_std[:, None, :]
            + batch.action_mean[:, None, :]
        )
        return [prediction.cpu() for prediction in raw]
