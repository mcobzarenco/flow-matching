"""State-masking rewrite (--mask-state, the state-reliance probe).

The mask substitutes each item's dataset state MEAN for its true state,
so the collator's normalization produces an exactly-zero soft state
token — zero information at in-distribution magnitude. Pure CPU/
synthetic through the shared Collator (the same fake-strategy pattern
as test_collator.py); the BijouPolicy plumbing on a real checkpoint is
covered by the probe's live oracles (baseline rows byte-identical
between masked and intact runs of the same subset).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import torch

from bijou.data import DatasetStats
from bijou.eval.policies import mask_state_items
from bijou.interface import Collator, PromptInputs

CHUNK, DIM = 4, 6


@dataclass(frozen=True, slots=True)
class FakeInputs:
    samples: tuple[PromptInputs, ...]

    def pin_memory(self) -> Self:
        return self

    def to(self, device: Any, *, non_blocking: bool = False) -> Self:
        return self

    def tensors(self) -> dict[str, torch.Tensor]:
        return {}


def collator() -> Collator[FakeInputs]:
    return Collator(
        inputs=lambda samples: FakeInputs(samples=tuple(samples)),
        instruction=None,
        camera_filter=None,
        max_cameras=None,
        action_codec=None,
        aux=None,
        camera_kind_dropout=0.0,
        instruction_augment=0.0,
        condition_fields=(),
        condition_dropout=0.0,
        generate_bracket=False,
        generate_override=None,
        subgoal_condition_dropout=0.0,
    )


def item(state: torch.Tensor) -> dict[str, Any]:
    # Nonzero, non-unit stats so mean-substitution and zero-state are
    # distinguishable outcomes.
    stats = DatasetStats(
        action_mean=(0.5,) * DIM,
        action_std=(2.0,) * DIM,
        state_mean=(1.25,) * DIM,
        state_std=(0.5,) * DIM,
        action_q01=None,
        action_q99=None,
        state_q01=None,
        state_q99=None,
    )
    return {
        "task": "pick up the cube",
        "repo_id": "user/rig",
        "observation.state": state,
        "action": torch.rand(CHUNK, DIM),
        "action_is_pad": torch.zeros(CHUNK, dtype=torch.bool),
        "observation.images.front": torch.rand(3, 8, 8),
        **stats.item_tensors(),
    }


def test_masked_items_collate_to_exactly_zero_state_token() -> None:
    original = item(torch.full((DIM,), 3.0))
    batch = collator()(mask_state_items([original]))
    prompt_state = batch.encoder_inputs.samples[0].state
    # Bitwise zero, not just close: (mean - mean) / std ≡ 0.
    assert torch.equal(prompt_state, torch.zeros(DIM))
    # The raw batch state carries the mean (decoders read only its shape).
    assert torch.equal(batch.state[0], original["state_mean"])


def test_mask_is_identity_when_state_equals_mean() -> None:
    at_mean = item(torch.full((DIM,), 1.25))
    intact = collator()([at_mean])
    masked = collator()(mask_state_items([at_mean]))
    assert torch.equal(
        intact.encoder_inputs.samples[0].state,
        masked.encoder_inputs.samples[0].state,
    )
    assert torch.equal(intact.state, masked.state)


def test_mask_rebuilds_items_without_mutating_originals() -> None:
    original = item(torch.full((DIM,), 3.0))
    [masked] = mask_state_items([original])
    assert masked is not original
    assert torch.equal(original["observation.state"], torch.full((DIM,), 3.0))
    # Truth actions and stats ride through untouched (same objects):
    # scoring and the baseline policies see intact data.
    assert masked["action"] is original["action"]
    assert masked["state_mean"] is original["state_mean"]
    # The substituted tensor is a clone, never an alias of the stats.
    masked["observation.state"] += 1.0
    assert torch.equal(original["state_mean"], torch.full((DIM,), 1.25))
