"""Train-time image augmentation (--image-augment, sim2real recipe).

The bitwise pin is the point: p=0 must pass every camera tensor
through by IDENTITY (no clone, no clamp, no dtype round-trip — the
molmoact2 uint8 truncation downstream would amplify any float
epsilon) and consume no RNG, so existing runs' dropout/augment
streams and pixel bytes stay byte-identical to the pre-flag code.
Eval-side constructions never set the field, so the default IS the
eval guarantee. Pure CPU/synthetic through the shared Collator (the
test_state_dropout.py fake-strategy pattern).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Self

import pytest
import torch

from bijou.data import DatasetStats
from bijou.image_augment import ImageAugmentSpec, augment_image
from bijou.interface import Collator, PromptInputs

CHUNK, DIM = 4, 6
SIZE = 16


@dataclass(frozen=True, slots=True)
class FakeInputs:
    samples: tuple[PromptInputs, ...]

    def pin_memory(self) -> Self:
        return self

    def to(self, device: Any, *, non_blocking: bool = False) -> Self:
        return self

    def tensors(self) -> dict[str, torch.Tensor]:
        return {}


def collator(image_augment: float = 0.0) -> Collator[FakeInputs]:
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
        image_augment=image_augment,
    )


def item(seed: int = 0) -> dict[str, Any]:
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
    generator = torch.Generator().manual_seed(seed)
    return {
        "task": "pick up the cube",
        "repo_id": "user/rig",
        "observation.state": torch.full((DIM,), 3.0),
        "action": torch.rand(CHUNK, DIM, generator=generator),
        "action_is_pad": torch.zeros(CHUNK, dtype=torch.bool),
        "observation.images.front": torch.rand(3, SIZE, SIZE, generator=generator),
        **stats.item_tensors(),
    }


def seeded(c: Collator[FakeInputs], seed: int) -> torch.Generator:
    generator = torch.Generator().manual_seed(seed)
    c._generator = generator
    return generator


def frame(batch: Any, row: int) -> torch.Tensor:
    return batch.encoder_inputs.samples[row].cameras[0].image


def test_p_zero_is_identity_and_draws_no_rng() -> None:
    items = [item(i) for i in range(4)]
    inert = collator(image_augment=0.0)
    generator = seeded(inert, 3)
    before = generator.get_state()
    batch = inert(items)
    # No RNG consumed: every pre-existing dropout/augment stream in a
    # p=0 run is byte-identical to the pre-flag code.
    assert torch.equal(before, generator.get_state())
    for row in range(4):
        # The SAME tensor object — the bitwise off-path pin (no clone,
        # no clamp, no dtype round-trip before the uint8 truncation).
        assert frame(batch, row) is items[row]["observation.images.front"]


def test_p_one_augments_every_frame_within_contract() -> None:
    items = [item(i) for i in range(8)]
    always = collator(image_augment=1.0)
    seeded(always, 7)
    batch = always(items)
    for row in range(8):
        out = frame(batch, row)
        source = items[row]["observation.images.front"]
        assert out.shape == source.shape
        assert out.dtype == torch.float32
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        assert not torch.equal(out, source)
        # Rebuilt, never mutated.
        assert out is not source


def test_partial_p_leaves_unfired_frames_by_identity() -> None:
    items = [item(i) for i in range(16)]
    half = collator(image_augment=0.5)
    seeded(half, 11)
    batch = half(items)
    passed = [
        frame(batch, row) is items[row]["observation.images.front"] for row in range(16)
    ]
    assert True in passed and False in passed  # both branches hit


def test_augment_stream_is_deterministic_given_seed() -> None:
    left = collator(image_augment=0.7)
    right = collator(image_augment=0.7)
    seeded(left, 5)
    seeded(right, 5)
    batch_left = left([item(i) for i in range(8)])
    batch_right = right([item(i) for i in range(8)])
    for row in range(8):
        assert torch.equal(frame(batch_left, row), frame(batch_right, row))


def test_augment_never_mutates_source_items() -> None:
    items = [item(i) for i in range(8)]
    originals = [row["observation.images.front"].clone() for row in items]
    always = collator(image_augment=1.0)
    seeded(always, 0)
    always(items)
    for row, original in zip(items, originals, strict=True):
        assert torch.equal(row["observation.images.front"], original)


def test_image_augment_outside_range_rejected() -> None:
    with pytest.raises(ValueError, match="image augment"):
        collator(image_augment=1.5)
    with pytest.raises(ValueError, match="image augment"):
        collator(image_augment=-0.1)
    collator(image_augment=1.0)  # always-on is a valid recipe


def test_probe_clone_pattern_zeroes_image_augment() -> None:
    # The train.py probe-collator convention: dataclasses.replace with
    # image_augment=0.0 must yield an identity clone — probes and
    # evals never see augmented frames.
    train_side = collator(image_augment=1.0)
    probe_side = dataclasses.replace(train_side, image_augment=0.0)
    items = [item(i) for i in range(4)]
    seeded(probe_side, 11)
    batch = probe_side(items)
    for row in range(4):
        assert frame(batch, row) is items[row]["observation.images.front"]


def test_recipe_contract_on_direct_call() -> None:
    generator = torch.Generator().manual_seed(2)
    image = torch.rand(3, 24, 32, generator=generator)
    out = augment_image(image, generator)
    assert out.shape == image.shape
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert not torch.equal(out, image)


def test_recipe_all_subops_fire_and_stay_in_range() -> None:
    spec = ImageAugmentSpec(noise_p=1.0, blur_p=1.0, jpeg_p=1.0)
    generator = torch.Generator().manual_seed(4)
    image = torch.rand(3, SIZE, SIZE, generator=generator)
    out = augment_image(image, generator, spec)
    assert out.shape == image.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert not torch.equal(out, image)


def test_recipe_collapsed_ranges_are_near_identity() -> None:
    # Every range collapsed to its neutral point: the pipeline's only
    # residue is op round-off (hue/contrast conversions), bounded well
    # under one uint8 step — the recipe adds nothing it wasn't asked to.
    spec = ImageAugmentSpec(
        crop_scale_min=1.0,
        brightness_delta=0.0,
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
        hue_delta=0.0,
        gamma_range=(1.0, 1.0),
        noise_p=0.0,
        blur_p=0.0,
        jpeg_p=0.0,
    )
    generator = torch.Generator().manual_seed(6)
    image = torch.rand(3, SIZE, SIZE, generator=generator)
    out = augment_image(image, generator, spec)
    assert torch.allclose(out, image, atol=1.0 / 512)


def test_recipe_rejects_non_chw_float() -> None:
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="float CHW"):
        augment_image(torch.zeros(SIZE, SIZE, 3), generator)  # HWC
    with pytest.raises(ValueError, match="float CHW"):
        augment_image(
            torch.zeros(3, SIZE, SIZE, dtype=torch.uint8),
            generator,
        )
