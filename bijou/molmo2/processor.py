"""Native Molmo2 image/prompt processing (WP3) — no remote code.

The shipped ``Molmo2Processor``/``Molmo2ImageProcessor`` are
``trust_remote_code`` modules pinned to transformers 4.x, which this repo
cannot import (the training environment runs 5.x). This module
reimplements, op-for-op, exactly the pieces the collator needs, from the
fetched sources (``processing_molmo2.py`` / ``image_processing_molmo2.py``,
facts pinned in ``docs/molmo2.md``); the golden-fixture test compares the
outputs against the reference processor run in its own 4.x environment
(``bank_processor_goldens.py``), so drift dies in CI rather than in a
training run.

The image pipeline (per image):

1. GLOBAL VIEW: resize to 378x378 (bilinear, no antialias), normalize
   (mean/std 0.5).
2. CROPS: choose a tiling of up-to-``max_crops`` 378x378 crops whose
   overlap margins (4+4 patches per dim) make neighboring crops share
   borders; resize the image so crops tile it exactly, then slice. With
   ``max_crops=1`` the tiling is 1x1 — the whole image as one crop — which
   is the port plan's "crops off" operating point WITHOUT leaving the
   shipped token-layout distribution (the reference always emits the
   global view + at least one crop).
3. POOLING INDEX: for each output token (2x2 patch group, global view
   first), the up-to-4 member patch indices into the image's flattened
   (view, patch) grid; -1 marks missing members at grid edges. This is the
   ``pooled_patches_idx`` the vision backbone consumes
   (``Molmo2VisionBackbone.forward``).

Token layout (per image, shipped ``processor_config.json`` options:
``image_use_col_tokens=true``, ``use_single_crop_col_tokens=false``,
``use_single_crop_start_token=true``):

    <low_res_im_start> [ <im_patch> * rw ] * rh <im_end>
    <im_start> [ <im_patch> * w  <im_col> ] * h <im_end>

where (rh, rw, h, w) is the image grid — pooled token rows/cols of the
global view and of the crop set. The patch-token order matches the pooling
index rows by construction (the injection contract:
``input_ids == IM_PATCH_ID`` positions receive the features in order).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms
from torch import Tensor

__all__ = [
    "IMAGE_TYPE_IDS",
    "IM_COL_ID",
    "IM_END_ID",
    "IM_PATCH_ID",
    "IM_START_ID",
    "LOW_RES_IM_START_ID",
    "ImageCrops",
    "image_token_ids",
    "process_image",
]

# Special-token ids, pinned from the checkpoint tokenizer (verified against
# the loaded tokenizer at collator build — see Molmo2InputsCollator).
IM_START_ID = 151_936
IM_END_ID = 151_937
IM_PATCH_ID = 151_938
IM_COL_ID = 151_939
LOW_RES_IM_START_ID = 151_940
_IM_LOW_ID = 151_942
_FRAME_START_ID = 151_943
_FRAME_END_ID = 151_944

# The ids the reference processor marks in ``token_type_ids`` — attention
# between any two of these positions is BIDIRECTIONAL (the model's
# token_type_ids mask function); text stays causal. Matches the reference's
# IMAGE_TOKENS list, which includes the video frame markers but NOT the
# ``<|image|>``/``<|video|>`` placeholders (those never survive expansion).
IMAGE_TYPE_IDS = frozenset(
    (
        IM_START_ID,
        IM_END_ID,
        IM_PATCH_ID,
        IM_COL_ID,
        LOW_RES_IM_START_ID,
        _IM_LOW_ID,
        _FRAME_START_ID,
        _FRAME_END_ID,
    ),
)

# Shipped preprocessor_config.json (378x378 views, patch 14, 2x2 pooling,
# 4+4-patch overlap margins, mean/std 0.5, bilinear). Constants, not
# config: the collator refuses nothing here because nothing varies — the
# ONLY knob the operating point exposes is max_crops.
BASE_INPUT_SIZE = 378
PATCH_SIZE = 14
POOL_H = 2
POOL_W = 2
_LEFT_MARGIN = 4
_RIGHT_MARGIN = 4
_CROP_PATCHES = BASE_INPUT_SIZE // PATCH_SIZE  # 27 patches per crop dim
_CROP_WINDOW_PATCHES = _CROP_PATCHES - (_LEFT_MARGIN + _RIGHT_MARGIN)  # 19
_CROP_WINDOW_SIZE = _CROP_WINDOW_PATCHES * PATCH_SIZE  # 266 px
_TOTAL_MARGIN_PIXELS = PATCH_SIZE * (_LEFT_MARGIN + _RIGHT_MARGIN)  # 112 px


@dataclass(frozen=True, slots=True)
class ImageCrops:
    """One image, processed: the ViT inputs and the token-layout facts.

    ``grid`` = (resized_h, resized_w, h, w): pooled token rows/cols of the
    global view and of the crop set — the reference's ``image_grids`` row.

    Shapes:
      - crops: [T, patches, patch_dim]  (T = 1 global view + n crops;
        patches = 27*27, patch_dim = 14*14*3; normalized float32)
      - pooled_idx: [P, pool_group]  (P = rh*rw + h*w tokens, long;
        indices into the flattened [T * patches] grid, -1 = missing)
    """

    grid: tuple[int, int, int, int]
    crops: Tensor
    pooled_idx: Tensor

    @property
    def num_patch_tokens(self) -> int:
        rh, rw, h, w = self.grid
        return rh * rw + h * w


def _resize_normalize(image: Tensor, out_h: int, out_w: int) -> Tensor:
    """Reference resize + normalize: bilinear WITHOUT antialias on the
    float image, clipped, then (x - 0.5) / 0.5 per channel.

    Shapes:
    - ``image``: [3, H, W] float in [0, 1] (the LeRobot decode convention)
    - returns: [out_h, out_w, 3] float32, normalized
    """
    if image.dtype == torch.uint8 or not torch.is_floating_point(image):
        raise ValueError(
            f"expected a float [0, 1] CHW image, got dtype {image.dtype}",
        )
    resized = torchvision.transforms.Resize(
        [out_h, out_w],
        torchvision.transforms.InterpolationMode.BILINEAR,
        antialias=False,
    )(image)
    resized = torch.clip(resized, 0.0, 1.0).to(torch.float32)
    return (resized.permute(1, 2, 0) - 0.5) / 0.5


def select_tiling(
    height: int,
    width: int,
    patch_size: int,
    max_num_crops: int,
) -> tuple[int, int]:
    """The reference's crop-tiling choice: the (rows, cols) grid of
    ``patch_size``-sized windows, up to ``max_num_crops`` total, that fits
    ``height x width`` with the least up- (preferred) or down-scaling.
    Mirrors ``image_processing_molmo2.select_tiling`` including its
    tie-breaks (candidates sorted by area then rows; scales < 1 masked to
    huge before the argmin)."""
    tilings = [
        (i, j)
        for i in range(1, max_num_crops + 1)
        for j in range(1, max_num_crops + 1)
        if i * j <= max_num_crops
    ]
    tilings.sort(key=lambda t: (t[0] * t[1], t[0]))
    # float32 like the reference — tie-breaks must round identically.
    candidates = (np.array(tilings, dtype=np.int32) * patch_size).astype(np.float32)
    original = np.array([height, width], dtype=np.float32)
    with np.errstate(divide="ignore"):
        required_scale = np.min(candidates / original, axis=-1)  # [n]
    if np.all(required_scale < 1):
        index = int(np.argmax(required_scale))
    else:
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        index = int(np.argmin(required_scale))
    return tilings[index]


def _arange_for_pooling(idx_arr: np.ndarray) -> np.ndarray:
    """Group a patch-index grid into 2x2 pooling groups, padding the
    edges with -1 (centered: extra pad goes bottom/right).

    Shapes:
    - ``idx_arr``: [H, W] patch indices
    - returns: [ceil(H/2), ceil(W/2), 4]
    """
    h_pad = POOL_H * -(idx_arr.shape[0] // -POOL_H) - idx_arr.shape[0]
    w_pad = POOL_W * -(idx_arr.shape[1] // -POOL_W) - idx_arr.shape[1]
    idx_arr = np.pad(
        idx_arr,
        [[h_pad // 2, (h_pad + 1) // 2], [w_pad // 2, (w_pad + 1) // 2]],
        mode="constant",
        constant_values=-1,
    )
    height, width = idx_arr.shape[0] // POOL_H, idx_arr.shape[1] // POOL_W
    grouped = idx_arr.reshape(height, POOL_H, width, POOL_W)
    return grouped.transpose(0, 2, 1, 3).reshape(height, width, POOL_H * POOL_W)


def _pixels_to_patches(views: Tensor) -> Tensor:
    """Flatten pixel views into ViT input rows.

    Shapes:
    - ``views``: [T, H, W, 3]
    - returns: [T, (H/14)*(W/14), 14*14*3]
    """
    num, height, width, channels = views.shape
    h_patches, w_patches = height // PATCH_SIZE, width // PATCH_SIZE
    views = views.reshape(num, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE, channels)
    views = views.permute(0, 1, 3, 2, 4, 5)
    return views.reshape(num, h_patches * w_patches, PATCH_SIZE * PATCH_SIZE * channels)


def process_image(image: Tensor, *, max_crops: int) -> ImageCrops:
    """One camera frame -> ViT inputs + token-layout facts (see
    :class:`ImageCrops`). Mirrors the reference
    ``image_to_patches_and_grids`` with the global view FIRST in both the
    crop stack and the pooling index.

    Shapes:
    - ``image``: [3, H, W] float in [0, 1]
    - returns: ImageCrops (crops [T, 729, 588], pooled_idx [P, 4])
    """
    height, width = int(image.shape[1]), int(image.shape[2])

    # Crop tiling, computed as if margins did not exist (they overlap).
    tiling = select_tiling(
        height - _TOTAL_MARGIN_PIXELS,
        width - _TOTAL_MARGIN_PIXELS,
        _CROP_WINDOW_SIZE,
        max_crops,
    )
    src = _resize_normalize(
        image,
        tiling[0] * _CROP_WINDOW_SIZE + _TOTAL_MARGIN_PIXELS,
        tiling[1] * _CROP_WINDOW_SIZE + _TOTAL_MARGIN_PIXELS,
    )
    num_crops = tiling[0] * tiling[1]
    crops = torch.zeros(
        (num_crops, BASE_INPUT_SIZE, BASE_INPUT_SIZE, 3),
        dtype=src.dtype,
    )
    patch_idx_arr = np.zeros(
        (num_crops, _CROP_PATCHES, _CROP_PATCHES),
        dtype=np.int64,
    )
    on_crop = 0
    for i in range(tiling[0]):
        y0 = i * _CROP_WINDOW_SIZE
        for j in range(tiling[1]):
            x0 = j * _CROP_WINDOW_SIZE
            crops[on_crop] = src[y0 : y0 + BASE_INPUT_SIZE, x0 : x0 + BASE_INPUT_SIZE]
            patch_idx = np.arange(_CROP_PATCHES * _CROP_PATCHES).reshape(
                _CROP_PATCHES,
                _CROP_PATCHES,
            )
            patch_idx = patch_idx + on_crop * _CROP_PATCHES * _CROP_PATCHES
            # Overlap regions belong to ONE crop: mask the losing side.
            if i != 0:
                patch_idx[:_LEFT_MARGIN, :] = -1
            if j != 0:
                patch_idx[:, :_LEFT_MARGIN] = -1
            if i != tiling[0] - 1:
                patch_idx[-_RIGHT_MARGIN:, :] = -1
            if j != tiling[1] - 1:
                patch_idx[:, -_RIGHT_MARGIN:] = -1
            patch_idx_arr[on_crop] = patch_idx
            on_crop += 1

    # Transpose crop-by-crop patch indices into one left-to-right grid over
    # the resized source, then drop the masked overlap entries.
    patch_idx_flat = (
        patch_idx_arr.reshape(tiling[0], tiling[1], _CROP_PATCHES, _CROP_PATCHES)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    patch_idx_grid = patch_idx_flat[patch_idx_flat >= 0].reshape(
        src.shape[0] // PATCH_SIZE,
        src.shape[1] // PATCH_SIZE,
    )
    pooling_idx = _arange_for_pooling(patch_idx_grid)
    crops_h, crops_w = pooling_idx.shape[:2]
    pooling_idx = pooling_idx.reshape(-1, POOL_H * POOL_W)

    # Global view — FIRST in the crop stack, so crop patch indices shift.
    resized = _resize_normalize(image, BASE_INPUT_SIZE, BASE_INPUT_SIZE)
    resize_idx = _arange_for_pooling(
        np.arange(_CROP_PATCHES * _CROP_PATCHES).reshape(_CROP_PATCHES, _CROP_PATCHES),
    )
    resized_h, resized_w = resize_idx.shape[:2]
    resize_idx = resize_idx.reshape(-1, POOL_H * POOL_W)

    pooling_idx = np.where(
        pooling_idx >= 0,
        pooling_idx + _CROP_PATCHES * _CROP_PATCHES,
        -1,
    )
    all_views = torch.cat([resized[None], crops], dim=0)
    return ImageCrops(
        grid=(resized_h, resized_w, crops_h, crops_w),
        crops=_pixels_to_patches(all_views),
        pooled_idx=torch.from_numpy(
            np.concatenate([resize_idx, pooling_idx], axis=0),
        ),
    )


def image_token_ids(grid: tuple[int, int, int, int]) -> list[int]:
    """The token expansion of one ``<|image|>`` placeholder: the low-res
    (global view) block then the high-res (crops) block, per the shipped
    processor options (col separators on the high-res rows only, the
    dedicated low-res start marker)."""
    resized_h, resized_w, height, width = grid
    low_res = [LOW_RES_IM_START_ID]
    for _ in range(resized_h):
        low_res.extend([IM_PATCH_ID] * resized_w)
    low_res.append(IM_END_ID)
    high_res = [IM_START_ID]
    for _ in range(height):
        high_res.extend([IM_PATCH_ID] * width)
        high_res.append(IM_COL_ID)
    high_res.append(IM_END_ID)
    return low_res + high_res
