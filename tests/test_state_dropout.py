"""Train-time state dropout (--state-dropout, ideas #9).

With probability p per sample the collator rewrites the item through
``mask_state_item`` — the SAME primitive as the eval-side reliance
probe (--mask-state), so the regularizer trains exactly the condition
the probe measures: normalized soft state token exactly zero, raw
batch state at the dataset mean, actions/targets untouched. p=0 must
be byte-inert AND consume no RNG (existing runs' dropout/augment
streams stay reproducible). Pure CPU/synthetic through the shared
Collator (the test_mask_state.py fake-strategy pattern).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Self

import pytest
import torch

from bijou.data import DatasetStats
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


def collator(state_dropout: float = 0.0) -> Collator[FakeInputs]:
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
        state_dropout=state_dropout,
    )


def item(state_value: float = 3.0) -> dict[str, Any]:
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
        "observation.state": torch.full((DIM,), state_value),
        "action": torch.rand(CHUNK, DIM),
        "action_is_pad": torch.zeros(CHUNK, dtype=torch.bool),
        "observation.images.front": torch.rand(3, 8, 8),
        **stats.item_tensors(),
    }


def seeded(c: Collator[FakeInputs], seed: int) -> torch.Generator:
    generator = torch.Generator().manual_seed(seed)
    c._generator = generator
    return generator


def dropped_rows(p: float, seed: int, n: int) -> list[bool]:
    # The collator's per-item decision stream, reproduced exactly.
    generator = torch.Generator().manual_seed(seed)
    return [float(torch.rand((), generator=generator)) < p for _ in range(n)]


def test_dropped_rows_mask_kept_rows_intact_bitwise() -> None:
    items = [item(float(i)) for i in range(16)]
    dropout = collator(state_dropout=0.5)
    seeded(dropout, 7)
    batch = dropout(items)
    reference = collator()([dict(row) for row in items])
    decisions = dropped_rows(0.5, 7, 16)
    assert True in decisions and False in decisions  # both branches hit
    for row, dropped in enumerate(decisions):
        prompt_state = batch.encoder_inputs.samples[row].state
        if dropped:
            # The probe's exact condition: normalized token ≡ 0, raw
            # state at the dataset mean.
            assert torch.equal(prompt_state, torch.zeros(DIM))
            assert torch.equal(batch.state[row], items[row]["state_mean"])
        else:
            assert torch.equal(
                prompt_state,
                reference.encoder_inputs.samples[row].state,
            )
            assert torch.equal(batch.state[row], items[row]["observation.state"])
        # Targets never masked, either branch.
        assert torch.equal(batch.actions[row], items[row]["action"])


def test_dropout_never_mutates_source_items() -> None:
    items = [item(3.0) for _ in range(8)]
    originals = [row["observation.state"].clone() for row in items]
    dropout = collator(state_dropout=0.9)
    seeded(dropout, 0)
    dropout(items)
    for row, original in zip(items, originals, strict=True):
        assert torch.equal(row["observation.state"], original)


def test_p_zero_is_inert_and_draws_no_rng() -> None:
    items = [item(3.0) for _ in range(4)]
    inert = collator(state_dropout=0.0)
    generator = seeded(inert, 3)
    before = generator.get_state()
    batch = inert(items)
    # No RNG consumed: every pre-existing dropout/augment stream in a
    # p=0 run is byte-identical to the pre-flag code.
    assert torch.equal(before, generator.get_state())
    assert torch.equal(batch.state[0], items[0]["observation.state"])


def test_state_dropout_outside_range_rejected() -> None:
    with pytest.raises(ValueError, match="state dropout"):
        collator(state_dropout=1.0)
    with pytest.raises(ValueError, match="state dropout"):
        collator(state_dropout=-0.1)


def test_probe_clone_pattern_zeroes_state_dropout() -> None:
    # The train.py probe-collator convention: dataclasses.replace with
    # state_dropout=0.0 must yield an inert clone of a dropout collator.
    train_side = collator(state_dropout=0.8)
    probe_side = dataclasses.replace(train_side, state_dropout=0.0)
    items = [item(3.0) for _ in range(4)]
    seeded(probe_side, 11)
    batch = probe_side(items)
    for row in range(4):
        assert torch.equal(batch.state[row], items[row]["observation.state"])
