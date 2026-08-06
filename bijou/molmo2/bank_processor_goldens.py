"""Bank reference-processor goldens for the native WP3 processor tests.

The shipped Molmo2 processor is ``trust_remote_code`` pinned to
transformers 4.x, so it cannot run in the repo environment — this script
runs it ONCE in a pinned side environment and freezes its outputs as
fixtures; ``tests/test_molmo2_processor.py`` then gates the native
implementation against them forever, CPU-only, no remote code.

Usage (the parity-harness convention)::

    uv run --with transformers==4.57.1 \\
        python -m bijou.molmo2.bank_processor_goldens

Writes ``tests/fixtures/molmo2_processor/<case>.npz``. Pixel tensors are
stored as strided samples + per-crop stats (full crop stacks are megabytes
and compress poorly); ids/grids/pooling indices are stored exactly. The
test regenerates the deterministic input images, so no image files are
banked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..encoders.molmo2 import hoist_text, user_turn_text
from ..gemma4.loading import resolve_checkpoint_dir
from ..gemma4.testing import synthetic_test_image
from ..interface import CameraFrame, PromptInputs
from .testing import golden_cases

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests/fixtures/molmo2_processor"

# Stride over the [crops, patches, patch_dim] pixel stack — dense enough
# that any resize/normalize/tiling drift shows, small enough to commit.
PIXEL_STRIDES = (1, 13, 11)


def case_images(sizes: tuple[tuple[int, int], ...]) -> list[np.ndarray]:
    """Deterministic float32 HWC [0, 1] inputs — the collator-side dtype
    convention (LeRobot frames are float), regenerated identically by the
    test."""
    return [
        np.asarray(synthetic_test_image(width=w, height=h), dtype=np.float32) / 255.0
        for (w, h) in sizes
    ]


def sample_for(
    sizes: tuple[tuple[int, int], ...],
    kinds: tuple[str, ...],
) -> PromptInputs:
    import torch

    return PromptInputs(
        instruction="pick up the cube and place it in the bin",
        cameras=tuple(
            CameraFrame(
                name=kind,
                kind=kind,
                image=torch.from_numpy(image).permute(2, 0, 1),
            )
            for kind, image in zip(kinds, case_images(sizes), strict=True)
        ),
        condition_text="[outcome|success][generate|actions]",
        state=torch.zeros(6),
    )


def main() -> int:
    from transformers import AutoProcessor

    checkpoint_dir = resolve_checkpoint_dir("allenai/Molmo2-4B")
    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for case in golden_cases():
        sample = sample_for(case.sizes, case.kinds)
        text = (
            hoist_text(len(case.sizes))
            + "<|im_start|>user\n"
            + user_turn_text(sample)
            + "<|im_end|>\n"
        )
        batch = processor(
            images=case_images(case.sizes),
            text=text,
            images_kwargs={"max_crops": case.max_crops},
            return_tensors="np",
        )
        pixels = batch["pixel_values"]
        out = FIXTURE_DIR / f"{case.name}.npz"
        np.savez_compressed(
            out,
            input_ids=np.asarray(batch["input_ids"][0], dtype=np.int64),
            token_type_ids=np.asarray(batch["token_type_ids"][0], dtype=np.int64),
            image_grids=np.asarray(batch["image_grids"], dtype=np.int64),
            image_num_crops=np.asarray(batch["image_num_crops"], dtype=np.int64),
            image_token_pooling=np.asarray(
                batch["image_token_pooling"],
                dtype=np.int64,
            ),
            pixel_samples=pixels[
                :: PIXEL_STRIDES[0],
                :: PIXEL_STRIDES[1],
                :: PIXEL_STRIDES[2],
            ].astype(np.float32),
            pixel_shape=np.asarray(pixels.shape, dtype=np.int64),
            pixel_mean=pixels.mean(axis=(1, 2)).astype(np.float64),
            pixel_std=pixels.std(axis=(1, 2)).astype(np.float64),
        )
        print(
            f"{case.name}: ids {batch['input_ids'].shape} "
            f"crops {pixels.shape} -> {out}",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
