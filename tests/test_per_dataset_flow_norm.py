"""Oracles for the per-dataset flow-normalization scheme
(--per-dataset-flow-norm, section tag "q01q99_per_dataset").

The 2026-08-17 flow-regression isolation's recipe enabler: a mixture's
pooled/foreign q01/q99 table can crush (wrist_flex 0.24x weight) or
clip (wrist_roll ±66° vs ±157°) a single rig's channels; under this
scheme each item normalizes — and each served chunk denormalizes —
under its OWN dataset's row. Family-owned since main ebaa8e0 (the
decoder is a pure normalized-space program): the branch lives in
:func:`flow_normalize_targets`/:func:`flow_denormalize_chunk`, reading
each item's honest ``batch.action_stats`` row. The queue-registered
oracle: a two-dataset synthetic fixture where pooled vs per-dataset
normalization produce measurably different normalized targets, with
the round trip exact.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Self, cast

import pytest
import torch

from bijou.checkpoint import VLAMetadata
from bijou.data import DatasetStats
from bijou.fast.molmoact2 import QuantileStats
from bijou.modelling.interface import Collator, PromptInputs
from bijou.models.molmoact2_flow import (
    flow_denormalize_chunk,
    flow_normalize_targets,
    item_flow_quantiles,
    per_dataset_flow_scheme,
)
from bijou.testing import tiny_molmoact2_flow_section

CHUNK, DIM = 4, 6

# Two rigs with deliberately mismatched action boxes: a narrow one (the
# crushed-channel shape) and a wide one (the clipped-channel shape).
NARROW = ((-1.0,) * DIM, (1.0,) * DIM)
WIDE = ((-100.0,) * DIM, (100.0,) * DIM)
# The mixture's pooled box is the wide envelope — the narrow rig's
# targets occupy 1% of it (the wrist_flex 0.24x-weight failure shape).
POOLED = QuantileStats(
    q01=torch.full((DIM,), -100.0),
    q99=torch.full((DIM,), 100.0),
)


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
    action: torch.Tensor | None = None,
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
        "action": torch.zeros(CHUNK, DIM) if action is None else action,
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
    span it. The chunk round trip under the own rows is exact inside
    the box."""
    raw = torch.tensor([[-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]]).expand(CHUNK, DIM)
    batch = _collator()([_item(NARROW, action=raw.clone())])

    under_own = flow_normalize_targets(batch, POOLED, per_dataset=True)
    under_pooled = flow_normalize_targets(batch, POOLED, per_dataset=False)
    # Measurably different: the pooled map crushes the narrow rig's
    # full-range motion to |x| <= 0.01 while the own row keeps it
    # full-range.
    assert under_own.abs().max() == pytest.approx(1.0)
    assert under_pooled.abs().max() <= 0.011
    assert (under_own - under_pooled).abs().max() > 0.9

    # Round trip under the own row is exact inside the box.
    assert torch.allclose(
        flow_denormalize_chunk(batch, under_own, POOLED, per_dataset=True),
        raw.expand(1, CHUNK, DIM),
    )


def test_per_item_rows_map_each_rig_to_its_own_coordinates() -> None:
    """The two-rig fixture: each item at half its own box lands on the
    SAME normalized coordinate — the per-item map is the whole point —
    and denormalization returns each to its own raw units."""
    raw = torch.stack(
        [torch.full((CHUNK, DIM), 0.5), torch.full((CHUNK, DIM), 50.0)],
    )
    batch = _collator()(
        [
            _item(NARROW, action=raw[0].clone()),
            _item(WIDE, action=raw[1].clone()),
        ],
    )
    rows_q01, rows_q99 = item_flow_quantiles(batch)
    assert rows_q01.shape == (2, 1, DIM)
    assert rows_q99.shape == (2, 1, DIM)
    normalized = flow_normalize_targets(batch, POOLED, per_dataset=True)
    assert torch.allclose(normalized[0], normalized[1])
    assert torch.allclose(
        flow_denormalize_chunk(batch, normalized, POOLED, per_dataset=True),
        raw,
    )


def test_merged_path_is_the_quantile_table_exactly() -> None:
    """per_dataset=False is the family's merged table verbatim, and a
    batch whose items all carry the merged row produces the SAME
    normalized targets under either scheme (the schemes only diverge
    when rows do)."""
    raw = torch.rand(2, CHUNK, DIM) * 150.0 - 75.0
    pooled_box = (
        tuple(POOLED.q01.tolist()),
        tuple(POOLED.q99.tolist()),
    )
    batch = _collator()(
        [
            _item(pooled_box, action=raw[0].clone()),
            _item(pooled_box, action=raw[1].clone()),
        ],
    )
    merged = flow_normalize_targets(batch, POOLED, per_dataset=False)
    assert torch.equal(merged, POOLED.normalize(batch.actions))
    per_item = flow_normalize_targets(batch, POOLED, per_dataset=True)
    assert torch.equal(merged, per_item)
    assert torch.equal(
        flow_denormalize_chunk(batch, merged, POOLED, per_dataset=False),
        flow_denormalize_chunk(batch, merged, POOLED, per_dataset=True),
    )


def test_per_dataset_refuses_quantile_less_batches() -> None:
    """Loud on the wiring gap: items resolved from a pre-quantile
    checkpoint stats table cannot train or serve the scheme — no
    mean/std fallback."""
    batch = _collator()(
        [
            _item(NARROW, with_quantiles=False),
            _item(WIDE, with_quantiles=False),
        ],
    )
    with pytest.raises(SystemExit, match="cannot fall back"):
        item_flow_quantiles(batch)
    with pytest.raises(SystemExit, match="cannot fall back"):
        flow_normalize_targets(batch, POOLED, per_dataset=True)


def test_section_tag_round_trips_and_gates_the_build() -> None:
    """The scheme is ONE recorded fact: the section tag. It round-trips
    through to_dict/from_dict, both known tags build, an unknown tag is
    refused, and :func:`per_dataset_flow_scheme` reads the recorded
    fact at family load."""
    from bijou.sections import MolmoFlowDecoderConfig, build_molmo_flow_decoder

    section = tiny_molmoact2_flow_section()
    build_molmo_flow_decoder(section, device="cpu")  # merged tag builds

    tagged = dataclasses.replace(section, normalization="q01q99_per_dataset")
    round_tripped = MolmoFlowDecoderConfig.from_dict(tagged.to_dict())
    assert round_tripped.normalization == "q01q99_per_dataset"
    build_molmo_flow_decoder(round_tripped, device="cpu")

    with pytest.raises(SystemExit, match="normalization scheme"):
        build_molmo_flow_decoder(
            dataclasses.replace(section, normalization="zscore"),
            device="cpu",
        )

    def metadata_with(config: MolmoFlowDecoderConfig) -> VLAMetadata:
        # per_dataset_flow_scheme reads only components["flow_decoder"]
        # — a structural stand-in exercises the whole function.
        return cast(
            "VLAMetadata",
            SimpleNamespace(
                components={"flow_decoder": {"config": config.to_dict()}},
            ),
        )

    assert per_dataset_flow_scheme(metadata_with(section)) is False
    assert per_dataset_flow_scheme(metadata_with(tagged)) is True
