"""Molmo2 native processor + collator (WP3) — CPU oracles.

Two layers of gating:

- STRUCTURE tests (always run): tiling selection, pooling-group edge
  padding, the token-layout arithmetic, prompt assembly mechanics (state
  splice, left padding, image-type mask) — pure functions, no artifacts.
- GOLDEN tests (skip when the Molmo2-4B snapshot is not in the HF cache):
  byte-for-byte agreement with the shipped ``trust_remote_code``
  processor, whose outputs were banked by
  ``bijou.molmo2.bank_processor_goldens`` in its pinned 4.x environment.
  ids/grids/pooling indices are exact; pixel values are gated on strided
  samples + per-crop stats (the full stacks are not committed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bijou.encoders.molmo2 import (
    BOS_ID,
    IM_END_TEXT_ID,
    IM_START_TEXT_ID,
    PAD_ID,
    Molmo2InputsCollator,
    camera_tag_text,
    hoist_text,
    user_turn_text,
)
from bijou.molmo2.bank_processor_goldens import (
    FIXTURE_DIR,
    PIXEL_STRIDES,
    sample_for,
)
from bijou.molmo2.config import Molmo2TextConfig
from bijou.molmo2.processor import (
    IM_COL_ID,
    IM_END_ID,
    IM_PATCH_ID,
    IM_START_ID,
    LOW_RES_IM_START_ID,
    image_token_ids,
    process_image,
    select_tiling,
)
from bijou.molmo2.testing import GoldenCase, golden_cases


def cached_checkpoint_dir() -> Path | None:
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download("allenai/Molmo2-4B", local_files_only=True),
        )
    except Exception:  # noqa: BLE001 — any cache miss means skip, not fail
        return None


CHECKPOINT = cached_checkpoint_dir()
needs_checkpoint = pytest.mark.skipif(
    CHECKPOINT is None,
    reason="Molmo2-4B snapshot not in the local HF cache",
)


# ---------------------------------------------------------------- structure


def test_select_tiling_matches_reference_cases() -> None:
    # 640x480 minus margins at the 266px crop window: 2x2 (the banked mc8
    # case); max_crops=1 degenerates to 1x1 regardless of size.
    assert select_tiling(480 - 112, 640 - 112, 266, 8) == (2, 2)
    assert select_tiling(480 - 112, 640 - 112, 266, 1) == (1, 1)
    # Wide image prefers more columns than rows.
    assert select_tiling(480 - 112, 1280 - 112, 266, 8) == (2, 4)


def test_image_token_layout_operating_point() -> None:
    # max_crops=1 on any input: grid (14, 14, 14, 14) — 196 + 196 patch
    # tokens; cols on the high-res rows ONLY (shipped processor options).
    crops = process_image(torch.rand(3, 480, 640), max_crops=1)
    assert crops.grid == (14, 14, 14, 14)
    assert crops.crops.shape == (2, 729, 588)
    assert crops.pooled_idx.shape == (392, 4)
    ids = image_token_ids(crops.grid)
    assert len(ids) == 198 + 212
    assert ids.count(IM_PATCH_ID) == 392
    assert ids.count(IM_COL_ID) == 14
    assert ids.count(LOW_RES_IM_START_ID) == 1
    assert ids.count(IM_START_ID) == 1
    assert ids.count(IM_END_ID) == 2
    # Patch-token order == pooling-index row order (the injection
    # contract): global block first.
    assert ids[0] == LOW_RES_IM_START_ID
    # Global-view indices point at view 0 ([0, 729)); crop indices are
    # shifted past it.
    valid_global = crops.pooled_idx[:196][crops.pooled_idx[:196] >= 0]
    valid_crop = crops.pooled_idx[196:][crops.pooled_idx[196:] >= 0]
    assert int(valid_global.max()) < 729
    assert int(valid_crop.min()) >= 729
    # 27 is odd: the centered 2x2 grouping pads bottom/right, so edge
    # groups carry -1 members but every group keeps >= 1 real patch.
    assert bool((crops.pooled_idx == -1).any())
    assert bool(((crops.pooled_idx >= 0).any(dim=-1)).all())


def test_pooling_groups_cover_every_patch_once() -> None:
    crops = process_image(torch.rand(3, 480, 640), max_crops=8)
    valid = crops.pooled_idx[crops.pooled_idx >= 0]
    # Every (view, patch) slot of the resized source grid is pooled
    # exactly once; overlap-margin patches are dropped, so the crop views
    # contribute only their owned windows.
    assert valid.unique().numel() == valid.numel()


# ------------------------------------------------------------------ goldens


def golden(case: GoldenCase) -> dict[str, np.ndarray]:
    path = FIXTURE_DIR / f"{case.name}.npz"
    if not path.exists():
        pytest.fail(
            f"missing golden fixture {path} — run "
            "`uv run --with transformers==4.57.1 "
            "python -m bijou.molmo2.bank_processor_goldens`",
        )
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


@needs_checkpoint
@pytest.mark.parametrize("case", golden_cases(), ids=lambda c: c.name)
def test_native_processor_matches_reference(case: GoldenCase) -> None:
    assert CHECKPOINT is not None
    reference = golden(case)
    sample = sample_for(case.sizes, case.kinds)

    collator = Molmo2InputsCollator(str(CHECKPOINT), max_crops=case.max_crops)
    batch = collator([sample])

    # ids: ours minus the spliced state token == the reference's (the
    # reference knows no state slot). BOS included on both sides.
    ids = batch.input_ids[0].tolist()
    state_slot = batch.state_slot
    assert ids[state_slot] == PAD_ID
    del ids[state_slot]
    assert ids == reference["input_ids"].tolist()

    # Image-type positions == the reference token_type_ids (theirs lacks
    # the state token; drop the same position from ours).
    type_mask = batch.image_type_mask[0].tolist()
    del type_mask[state_slot]
    assert type_mask == [bool(t) for t in reference["token_type_ids"]]

    # Per-image grids and crop counts.
    images = [
        process_image(camera.image, max_crops=case.max_crops)
        for camera in sample.cameras
    ]
    assert [list(image.grid) for image in images] == reference["image_grids"].tolist()
    assert [image.crops.shape[0] for image in images] == reference[
        "image_num_crops"
    ].tolist()

    # Pooling indices: the reference banks them per-image (offsets are
    # applied model-side); ours bake the per-sample offsets in. Check the
    # raw per-image indices exactly, then the offsets.
    raw = torch.cat([image.pooled_idx for image in images], dim=0)
    assert raw.tolist() == reference["image_token_pooling"].tolist()
    offsets = np.cumsum([0] + [image.crops.shape[0] * 729 for image in images])
    expected_rows = []
    for image, offset in zip(images, offsets[:-1], strict=True):
        idx = image.pooled_idx
        expected_rows.append(torch.where(idx >= 0, idx + int(offset), idx))
    expected = torch.cat(expected_rows, dim=0)
    assert torch.equal(batch.pooled_patches_idx[0], expected)

    # Pixels: strided samples bitwise-close, stats tight. Crop stack
    # order = image-by-image, global view first (concatenation order).
    pixels = torch.cat([image.crops for image in images], dim=0).numpy()
    assert list(pixels.shape) == reference["pixel_shape"].tolist()
    samples = pixels[:: PIXEL_STRIDES[0], :: PIXEL_STRIDES[1], :: PIXEL_STRIDES[2]]
    np.testing.assert_allclose(samples, reference["pixel_samples"], atol=1e-6)
    np.testing.assert_allclose(
        pixels.mean(axis=(1, 2)),
        reference["pixel_mean"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pixels.std(axis=(1, 2)),
        reference["pixel_std"],
        atol=1e-6,
    )


# ------------------------------------------------------- collator mechanics


@needs_checkpoint
def test_prompt_assembly_and_state_splice() -> None:
    assert CHECKPOINT is not None
    case = golden_cases()[1]  # two cameras
    sample = sample_for(case.sizes, case.kinds)
    collator = Molmo2InputsCollator(str(CHECKPOINT), max_crops=case.max_crops)
    batch = collator([sample])
    ids = batch.input_ids[0].tolist()

    # Sequence frame: bos first; close tail (state, <|im_end|>, \n) last.
    assert ids[0] == BOS_ID
    assert ids[-3] == PAD_ID  # the state slot
    assert ids[-2] == IM_END_TEXT_ID
    assert batch.state_slot == -3
    assert bool(batch.attention_mask[0, batch.state_slot])  # real token
    # Exactly one user-turn open marker, after the hoisted images.
    assert ids.count(IM_START_TEXT_ID) == 1
    open_at = ids.index(IM_START_TEXT_ID)
    assert ids[:open_at].count(IM_PATCH_ID) == 2 * 392
    # The state slot is NOT an image position.
    assert not bool(batch.image_type_mask[0, batch.state_slot])

    # The rendered text binds kinds to shipped image labels in order.
    text = user_turn_text(sample)
    assert camera_tag_text("wrist", 1) in text
    assert camera_tag_text("overhead", 2) in text
    assert text.index(camera_tag_text("wrist", 1)) < text.index(
        camera_tag_text("overhead", 2),
    )
    assert hoist_text(2) == "Image 1<|image|>Image 2<|image|>"


@needs_checkpoint
def test_batch_left_padding_and_vision_padding() -> None:
    assert CHECKPOINT is not None
    one = sample_for(((640, 480),), ("wrist",))
    two = sample_for(((640, 480), (320, 240)), ("wrist", "overhead"))
    collator = Molmo2InputsCollator(str(CHECKPOINT), max_crops=1)
    batch = collator([one, two])

    assert batch.has_padding
    # The shorter (single-camera) row is LEFT-padded: pads at the front,
    # masked, never image-typed; the real tail still frames correctly.
    row = batch.input_ids[0]
    mask = batch.attention_mask[0]
    pad_width = int((mask == 0).sum())
    assert pad_width > 0
    assert bool((mask[:pad_width] == 0).all()) and bool((mask[pad_width:] == 1).all())
    assert bool((row[:pad_width] == PAD_ID).all())
    assert not bool(batch.image_type_mask[0, :pad_width].any())
    assert row[pad_width].item() == BOS_ID
    assert row[-2].item() == IM_END_TEXT_ID

    # Vision tensors pad per sample: row 0 has 2 views, row 1 has 4; the
    # padding fill is -1 on both crops and pooling indices.
    assert batch.crops.shape[1] == 4
    assert bool((batch.crops[0, 2:] == -1.0).all())
    assert bool((batch.pooled_patches_idx[0, 392:] == -1).all())
    assert bool(((batch.pooled_patches_idx[1, :784] >= 0).any(dim=-1)).all())


def test_collator_refuses_bad_construction() -> None:
    with pytest.raises(ValueError, match="max_crops"):
        Molmo2InputsCollator("allenai/Molmo2-4B", max_crops=0)


def test_fast_block_base_anchors_after_image_specials() -> None:
    config = Molmo2TextConfig.molmo2_4b()
    assert config.fast_block_base == 152_064
    assert config.fast_block_base == config.total_vocab_size
