"""Tests for the shared Collator core and the stats plumbing around it.

Pure CPU/synthetic: a fake InputsCollator stands in for the encoder
strategy, so no processor/checkpoint is needed. What's covered is the
backbone-agnostic half — camera policy, instruction override, NormStats
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


def item(
    *,
    with_quantiles: bool,
    cameras: tuple[str, ...] = ("front",),
    camera_kinds: dict[str, str] | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "task": "pick up the cube",
        "repo_id": "user/rig",
        "observation.state": torch.zeros(DIM),
        "action": torch.zeros(CHUNK, DIM),
        "action_is_pad": torch.zeros(CHUNK, dtype=torch.bool),
        **stats(with_quantiles=with_quantiles).item_tensors(),
    }
    if camera_kinds is not None:
        payload["camera_kinds"] = camera_kinds
    for name in cameras:
        payload[f"observation.images.{name}"] = torch.rand(3, 8, 8)
    return payload


def collator(**overrides: Any) -> Collator[FakeInputs]:
    kwargs: dict[str, Any] = {
        "inputs": fake_inputs_collator,
        "instruction": None,
        "camera_filter": None,
        "max_cameras": None,
        "action_codec": None,
        "aux": None,
        "camera_kind_dropout": 0.0,
        "instruction_augment": 0.0,
        "condition_fields": (),
        "condition_dropout": 0.0,
        "subgoal_condition_dropout": 0.0,
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


def test_camera_kinds_resolve_with_unknown_fallback() -> None:
    """Kinds ride the ITEM (the stats convention); missing map/camera ⇒
    unknown; ORDER stays the sorted camera keys regardless of kind
    (multiple unknowns keep a stable key-derived order)."""
    kinds = {"front": "front", "gripper_cam": "wrist"}
    sample_item = item(
        with_quantiles=True,
        cameras=("zed", "front", "gripper_cam", "aux2"),
        camera_kinds=kinds,
    )
    batch = collator()([sample_item])
    (prompt,) = batch.encoder_inputs.samples
    assert [(c.name, c.kind) for c in prompt.cameras] == [
        ("aux2", "unknown"),
        ("front", "front"),
        ("gripper_cam", "wrist"),
        ("zed", "unknown"),
    ]
    bare = item(with_quantiles=True)  # no camera_kinds key at all
    batch = collator()([bare])
    (prompt,) = batch.encoder_inputs.samples
    assert prompt.cameras[0].kind == "unknown"


def test_instruction_augment_samples_judge_rewrites() -> None:
    """With probability p the recorded task swaps for a uniformly drawn
    judge rewrite; unjudged items (no/empty suggestions) always keep the
    recorded string; the CLI override beats both."""
    suggested = ("grasp the red cube", "pick the cube up and hold it")
    judged = item(with_quantiles=True)
    judged["suggested_instructions"] = suggested

    always = collator(instruction_augment=1.0)
    picked = {
        always([dict(judged)]).encoder_inputs.samples[0].instruction for _ in range(16)
    }
    assert picked <= set(suggested)  # never the recorded string at p=1
    assert len(picked) == 2  # both rewrites appear (uniform draw)

    bare = item(with_quantiles=True)  # no suggestions key at all
    assert always([bare]).encoder_inputs.samples[0].instruction == "pick up the cube"
    off = collator(instruction_augment=0.0)
    assert (
        off([dict(judged)]).encoder_inputs.samples[0].instruction == "pick up the cube"
    )
    override = collator(instruction="do the thing", instruction_augment=1.0)
    assert (
        override([dict(judged)]).encoder_inputs.samples[0].instruction == "do the thing"
    )
    torch.manual_seed(11)
    half = collator(instruction_augment=0.5)
    sequence = [
        half([dict(judged)]).encoder_inputs.samples[0].instruction for _ in range(32)
    ]
    assert "pick up the cube" in sequence  # recorded survives at p=0.5
    assert any(s in suggested for s in sequence)
    torch.manual_seed(11)
    again = collator(instruction_augment=0.5)
    assert [
        again([dict(judged)]).encoder_inputs.samples[0].instruction for _ in range(32)
    ] == sequence  # seeded determinism


def test_condition_fields_render_hindsight_labels() -> None:
    """Configured fields render as trailing brackets from the item's
    labels, template order; None labels render nothing; dropout-0 is
    the TRUE-label probe context; override happens item-side (policy)."""
    from bijou.annotations import ConditionField

    fields = (ConditionField.OUTCOME, ConditionField.SMOOTHNESS)
    labeled = item(with_quantiles=True)
    labeled["condition_outcome"] = "failure"
    labeled["condition_smoothness"] = "medium"
    batch = collator(condition_fields=fields)([labeled])
    (prompt,) = batch.encoder_inputs.samples
    assert prompt.condition_text == "[outcome: failure][smoothness: medium]"

    partial = item(with_quantiles=True)
    partial["condition_outcome"] = "success"  # smoothness unlabeled
    batch = collator(condition_fields=fields)([partial])
    assert batch.encoder_inputs.samples[0].condition_text == "[outcome: success]"

    unlabeled = item(with_quantiles=True)
    batch = collator(condition_fields=fields)([unlabeled])
    assert batch.encoder_inputs.samples[0].condition_text == ""
    # Fields off: labels present but nothing renders.
    batch = collator()([dict(labeled)])
    assert batch.encoder_inputs.samples[0].condition_text == ""
    # Template order is enforced.
    with pytest.raises(ValueError, match="template order"):
        collator(
            condition_fields=(ConditionField.SMOOTHNESS, ConditionField.OUTCOME),
        )


def test_subgoal_condition_resolves_per_frame_with_own_dropout() -> None:
    """C2: the subgoal bracket resolves from the frame's segment label
    (language_persistent), an explicit condition_subgoal override wins
    (planner/CLI), and it drops at its OWN rate — deployment mostly
    runs planner-less."""
    from bijou.annotations import ConditionField

    fields = (ConditionField.SUBGOAL,)
    framed = item(with_quantiles=True)
    framed["timestamp"] = torch.tensor(6.0)
    framed["language_persistent"] = [
        {
            "role": "assistant",
            "content": "reach toward the boat",
            "style": "subtask",
            "timestamp": 0.0,
            "camera": None,
            "tool_calls": None,
        },
    ]
    batch = collator(condition_fields=fields)([dict(framed)])
    assert (
        batch.encoder_inputs.samples[0].condition_text
        == "[subgoal: reach toward the boat]"
    )
    framed["condition_subgoal"] = "place it on the disk"
    batch = collator(condition_fields=fields)([dict(framed)])
    assert (
        batch.encoder_inputs.samples[0].condition_text
        == "[subgoal: place it on the disk]"
    )
    # No segment label and no override: nothing renders.
    bare = item(with_quantiles=True)
    bare["timestamp"] = torch.tensor(6.0)
    batch = collator(condition_fields=fields)([bare])
    assert batch.encoder_inputs.samples[0].condition_text == ""
    # Its own dropout rate: outcome survives while subgoal drops.
    both = dict(framed)
    both["condition_outcome"] = "success"
    torch.manual_seed(5)
    dropped = collator(
        condition_fields=(ConditionField.SUBGOAL, ConditionField.OUTCOME),
        subgoal_condition_dropout=0.9,
    )
    rendered = [
        dropped([dict(both)]).encoder_inputs.samples[0].condition_text
        for _ in range(24)
    ]
    assert "[outcome: success]" in rendered  # subgoal dropped, outcome kept
    assert any("subgoal" in text for text in rendered)  # but not always


def test_condition_dropout_is_seeded_per_field() -> None:
    from bijou.annotations import ConditionField

    labeled = item(with_quantiles=True)
    labeled["condition_outcome"] = "success"
    labeled["condition_smoothness"] = "high"
    fields = (ConditionField.OUTCOME, ConditionField.SMOOTHNESS)
    torch.manual_seed(3)
    half = collator(condition_fields=fields, condition_dropout=0.5)
    rendered = [
        half([dict(labeled)]).encoder_inputs.samples[0].condition_text
        for _ in range(32)
    ]
    assert (
        "" in rendered
        or "[outcome: success]" in rendered
        or ("[smoothness: high]" in rendered)
    )  # some field dropped somewhere
    assert "[outcome: success][smoothness: high]" in rendered  # and some kept
    torch.manual_seed(3)
    again = collator(condition_fields=fields, condition_dropout=0.5)
    assert [
        again([dict(labeled)]).encoder_inputs.samples[0].condition_text
        for _ in range(32)
    ] == rendered


def test_camera_kind_dropout_is_seeded_and_bounded() -> None:
    kinds = {"front": "front"}
    torch.manual_seed(7)
    dropped = collator(camera_kind_dropout=0.5)
    pattern = [
        dropped([item(with_quantiles=True, camera_kinds=kinds)])
        .encoder_inputs.samples[0]
        .cameras[0]
        .kind
        for _ in range(32)
    ]
    assert "unknown" in pattern and "front" in pattern  # both outcomes
    torch.manual_seed(7)
    again = collator(camera_kind_dropout=0.5)
    assert [
        again([item(with_quantiles=True, camera_kinds=kinds)])
        .encoder_inputs.samples[0]
        .cameras[0]
        .kind
        for _ in range(32)
    ] == pattern
    with pytest.raises(ValueError, match="outside"):
        collator(camera_kind_dropout=1.0)


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
