"""Shared test utilities for the verification and benchmark CLIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


def synthetic_test_image(width: int = 640, height: int = 480) -> Image.Image:
    """Deterministic in-memory RGB test image (no RNG, no files).

    A per-pixel color gradient with a red ellipse and a green square --
    enough structure for the vision tower to produce a meaningful
    description. Shape placement is proportional to the requested size; at
    the default 640x480 it matches the geometry of the image used for the
    recorded H100 parity and benchmark results.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    grad = np.zeros((height, width, 3), dtype=np.uint8)
    grad[..., 0] = (np.arange(width, dtype=np.uint32) * 255 // width)[None, :]
    grad[..., 1] = (np.arange(height, dtype=np.uint32) * 255 // height)[:, None]
    grad[..., 2] = 128
    image = Image.fromarray(grad)

    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (width * 5 // 16, height * 5 // 16, width * 11 // 16, height * 11 // 16),
        fill=(220, 40, 40),
    )
    draw.rectangle(
        (width * 3 // 32, height // 8, width * 9 // 32, height * 3 // 8),
        fill=(40, 200, 80),
    )
    return image


def load_test_image(path: str | None) -> tuple[Image.Image, str]:
    """(image, label): opens ``path`` if given, else the synthetic default."""
    from PIL import Image

    if path is not None:
        return Image.open(path).convert("RGB"), path
    image = synthetic_test_image()
    return image, f"synthetic {image.width}x{image.height}"
