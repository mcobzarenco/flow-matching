"""Oracles for the per-dataset flow-normalization scheme
(--per-dataset-flow-norm, section tag "q01q99_per_dataset").

The 2026-08-17 flow-regression isolation's recipe enabler: a mixture's
pooled/foreign q01/q99 table can crush (wrist_flex 0.24x weight) or
clip (wrist_roll ±66° vs ±157°) a single rig's channels; under this
scheme each item normalizes — and each served chunk denormalizes —
under its OWN dataset's row. The queue-registered oracle: a
two-dataset synthetic fixture where pooled vs per-dataset
normalization produce measurably different normalized targets, with
the round trip exact.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Self

import pytest
import torch

from bijou.data import DatasetStats
from bijou.modelling.decoders.molmo_flow import (
    item_quantile_rows,
    normalize_targets,
    unnormalize_chunk,
)
from bijou.modelling.interface import Collator, PromptInputs
from bijou.testing import tiny_molmoact2_flow_section

CHUNK, DIM = 4, 6

# Two rigs with deliberately mismatched action boxes: a narrow one (the
# crushed-channel shape) and a wide one (the clipped-channel shape).
NARROW = ((-1.0,) * DIM, (1.0,) * DIM)
WIDE = ((-100.0,) * DIM, (100.0,) * DIM)
# The mixture's pooled box is the wide envelope — the narrow rig's
# targets occupy 1% of it (the wrist_flex 0.24x-weight failure shape).
POOLED = ((-100.0,) * DIM, (100.0,) * DIM)


@dataclass(frozen=True, slots=True)
class FakeInputs:
    samples: tuple[PromptInputs, ...]

    def pin_memory(self) -> Self:
        return self

    def to(self, device: Any, *, non_blocking: bool = False) -> Self:
        return self

    def tensors(self) -> dict[str, torch.Tensor]:
        return {}


def _stats(box: tuple[tuple[float, ...], tuple[float, ...]]) -> DatasetStats:
    q01, q99 = box
    return DatasetStats(
        action_mean=(0.0,) * DIM,
        action_std=(1.0,) * DIM,
        state_mean=(0.0,) * DIM,
        state_std=(1.0,) * DIM,
        action_q01=q01,
        action_q99=q99,
        state_q01=q01,
        state_q99=q99,
    )


def _item(
    box: tuple[tuple[float, ...], tuple[float, ...]],
    *,
    with_quantiles: bool = True,
) -> dict[str, Any]:
    stats = _stats(box)
    tensors = stats.item_tensors()
    if not with_quantiles:
        tensors = {
            k: v for k, v in tensors.items() if "q01" not in k and "q99" not in k
        }
    return {
        "task": "pick up the cube",
        "repo_id": "user/rig",
        "observation.state": torch.zeros(DIM),
        "action": torch.zeros(CHUNK, DIM),
        "action_is_pad": torch.zeros(CHUNK, dtype=torch.bool),
        "observation.images.front": torch.rand(3, 8, 8),
        **tensors,
    }


def _collator(**overrides: Any) -> Collator[FakeInputs]:
    kwargs: dict[str, Any] = {
        "inputs": lambda samples: FakeInputs(samples=tuple(samples)),
        "instruction": None,
        "camera_filter": None,
        "max_cameras": None,
        "action_codec": None,
        "aux": None,
        "camera_kind_dropout": 0.0,
        "instruction_augment": 0.0,
        "condition_fields": (),
        "condition_dropout": 0.0,
        "generate_bracket": False,
        "generate_override": None,
        "subgoal_condition_dropout": 0.0,
    }
    kwargs.update(overrides)
    return Collator(**kwargs)


def test_pooled_vs_per_dataset_targets_differ_and_round_trip() -> None:
    """THE registered oracle: the narrow rig's targets under the pooled
    box collapse to 1% of the normalized range; under its own row they
    span it. Round trips under the OWN row are exact inside the box."""
    raw = torch.tensor([[-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]]).expand(CHUNK, DIM)
    own_q01 = torch.tensor(NARROW[0])
    own_q99 = torch.tensor(NARROW[1])
    pooled_q01 = torch.tensor(POOLED[0])
    pooled_q99 = torch.tensor(POOLED[1])

    under_own = normalize_targets(raw, own_q01, own_q99)
    under_pooled = normalize_targets(raw, pooled_q01, pooled_q99)
    # Measurably different: the pooled map crushes the narrow rig's
    # full-range motion to |x| <= 0.01 while the own row keeps it
    # full-range.
    assert under_own.abs().max() == pytest.approx(1.0)
    assert under_pooled.abs().max() <= 0.011
    assert (under_own - under_pooled).abs().max() > 0.9

    # Round trip under the own row is exact inside the box.
    assert torch.allclose(unnormalize_chunk(under_own, own_q01, own_q99), raw)


def test_collator_carries_raw_item_rows_alongside_merged_override() -> None:
    """The joint-family serve shape: action_stats stays pinned to the
    merged CE table (the b779ba4 contract) while item_action_stats
    carries each item's RAW row for the flow leg — the override never
    leaks into the carrier."""
    merged_q01 = torch.full((DIM,), -3.0)
    merged_q99 = torch.full((DIM,), 5.0)
    batch = _collator(
        action_q01=merged_q01,
        action_q99=merged_q99,
        carry_item_action_stats=True,
    )([_item(NARROW), _item(WIDE)])
    assert batch.action_stats.q01 is not None
    assert torch.equal(batch.action_stats.q01, merged_q01.expand(2, -1))
    carried = batch.item_action_stats
    assert carried is not None
    assert carried.q01 is not None and carried.q99 is not None
    assert torch.equal(carried.q01[0], torch.tensor(NARROW[0]))
    assert torch.equal(carried.q01[1], torch.tensor(WIDE[0]))
    assert torch.equal(carried.q99[0], torch.tensor(NARROW[1]))
    assert torch.equal(carried.q99[1], torch.tensor(WIDE[1]))

    # item_quantile_rows hands the decoder broadcast-ready [B, 1, D]
    # rows; normalize→unnormalize under them is exact per item.
    rows_q01, rows_q99 = item_quantile_rows(batch, DIM)
    assert rows_q01.shape == (2, 1, DIM)
    raw = torch.stack(
        [torch.full((CHUNK, DIM), 0.5), torch.full((CHUNK, DIM), 50.0)],
    )
    normalized = normalize_targets(raw, rows_q01, rows_q99)
    # Same normalized coordinate (each at half its own box) — the
    # per-item map is the whole point.
    assert torch.allclose(normalized[0], normalized[1])
    assert torch.allclose(unnormalize_chunk(normalized, rows_q01, rows_q99), raw)


def test_collator_without_carry_leaves_field_none() -> None:
    batch = _collator()([_item(NARROW), _item(WIDE)])
    assert batch.item_action_stats is None


def test_carry_refuses_quantile_less_items() -> None:
    with pytest.raises(ValueError, match="cannot fall back"):
        _collator(carry_item_action_stats=True)(
            [
                _item(NARROW, with_quantiles=False),
                _item(WIDE, with_quantiles=False),
            ],
        )


def test_item_quantile_rows_refusals() -> None:
    """Every wiring gap is loud: a batch without the carrier, and a
    width mismatch against the configured decoder geometry."""
    plain = _collator()([_item(NARROW)])
    with pytest.raises(ValueError, match="carry_item_action_stats"):
        item_quantile_rows(plain, DIM)
    carried = _collator(carry_item_action_stats=True)([_item(NARROW)])
    with pytest.raises(ValueError, match="-wide"):
        item_quantile_rows(carried, DIM + 1)


def test_batch_to_moves_carried_rows() -> None:
    """The carrier rides the batch's device hooks (DevicePrefetcher
    walks all_tensors; .to must not strand it on CPU)."""
    batch = _collator(carry_item_action_stats=True)([_item(NARROW)])
    assert batch.item_action_stats is not None
    tensor_count = len(batch.all_tensors())
    moved = batch.to("cpu")
    assert moved.item_action_stats is not None
    assert len(moved.all_tensors()) == tensor_count
    moved_q01 = moved.item_action_stats.q01
    original_q01 = batch.item_action_stats.q01
    assert moved_q01 is not None and original_q01 is not None
    assert torch.equal(moved_q01, original_q01)


def test_section_tag_builds_decoder_mode() -> None:
    """The scheme is ONE recorded fact: the section tag. Build reads it
    ("q01q99" = baked merged table, "q01q99_per_dataset" = per-item
    rows), round-trips it through to_dict/from_dict, and refuses a tag
    this code does not know."""
    from bijou.sections import MolmoFlowDecoderConfig, build_molmo_flow_decoder

    section = tiny_molmoact2_flow_section()
    dim = section.action_dim
    stats = _stats(((-2.0,) * DIM, (2.0,) * DIM))
    stats = dataclasses.replace(
        stats,
        action_mean=(0.0,) * dim,
        action_std=(1.0,) * dim,
        state_mean=(0.0,) * dim,
        state_std=(1.0,) * dim,
        action_q01=(-2.0,) * dim,
        action_q99=(2.0,) * dim,
        state_q01=(-2.0,) * dim,
        state_q99=(2.0,) * dim,
    )
    merged = build_molmo_flow_decoder(section, stats, device="cpu")
    assert merged.per_dataset_norm is False

    tagged = dataclasses.replace(section, normalization="q01q99_per_dataset")
    round_tripped = MolmoFlowDecoderConfig.from_dict(tagged.to_dict())
    assert round_tripped.normalization == "q01q99_per_dataset"
    per_dataset = build_molmo_flow_decoder(round_tripped, stats, device="cpu")
    assert per_dataset.per_dataset_norm is True

    with pytest.raises(SystemExit, match="normalization scheme"):
        build_molmo_flow_decoder(
            dataclasses.replace(section, normalization="zscore"),
            stats,
            device="cpu",
        )
