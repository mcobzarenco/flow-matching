"""Train-time photometric image augmentation (--image-augment).

The pi0/OpenVLA-class sim2real recipe, applied per camera frame at
collation on the TRAIN side only: photometric jitter (brightness,
contrast, saturation, hue, gamma), Gaussian sensor noise, slight
defocus blur, JPEG compression artifacts, and a small random
crop/translate. The point is robustness to appearance shift between
the training domain (sim renders) and deployment (real cameras) —
lighting, white balance, optics and compression the policy should not
key on.

Contract (the Collator's image convention, bijou/interface.py):
float CHW ``[3, H, W]`` in ``[0, 1]`` in, the same shape/dtype/range
out. All randomness comes from the CALLER's generator (the collator's
per-worker stream) — given a generator state the output is
deterministic, and the caller gates the call, so a frame that is not
augmented never touches this module (identity pass-through, zero RNG:
the bitwise aug-off pin lives at the call site).

Every parameter of one augmented frame is drawn UP FRONT in one
fixed 14-draw block (gates included, fired or not), then the ops
apply in a fixed physical order — crop/translate (camera pose) →
photometric (scene/ISP) → defocus (optics) → sensor noise → JPEG
(compression last, so it compresses the noise like a real camera
pipeline). Only the noise FIELD (one ``randn`` per fired noise gate)
consumes generator state beyond the block.

The v0 parameter ranges below are the pre-registered recipe
(fontaine pre-reg 2026-08-15); training arms cite the spec by
default, not by re-typing numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms.v2 import functional as TF


@dataclass(frozen=True)
class ImageAugmentSpec:
    """One augmented frame's parameter ranges (uniform draws unless
    noted). Sub-ops with a ``*_p`` gate fire per frame; the ungated
    ops (crop, photometric) always apply to an augmented frame."""

    # Random crop/translate: side scale s ~ U(min, 1], crop placed
    # uniformly, bilinear-resized back to the source resolution
    # (≤ ~10% zoom + translate — the pi0-class spatial jitter).
    crop_scale_min: float = 0.90
    # Additive brightness delta ~ U(-d, d) (in [0,1] pixel units).
    brightness_delta: float = 0.15
    # Multiplicative factors ~ U(lo, hi) around 1.
    contrast_range: tuple[float, float] = (0.7, 1.3)
    saturation_range: tuple[float, float] = (0.7, 1.3)
    # Hue shift ~ U(-d, d), torchvision convention (fraction of the
    # color wheel; 0.5 = 180°).
    hue_delta: float = 0.05
    # Gamma ~ exp(U(ln lo, ln hi)) — log-uniform so darkening and
    # brightening curves are symmetric.
    gamma_range: tuple[float, float] = (0.8, 1.25)
    # Gaussian sensor noise: fires with noise_p, sigma ~ U(lo, hi).
    noise_p: float = 0.5
    noise_sigma_range: tuple[float, float] = (0.002, 0.02)
    # Defocus: fires with blur_p, gaussian kernel (blur_kernel odd),
    # sigma ~ U(lo, hi) — "slight", sub-pixel to ~1px.
    blur_p: float = 0.25
    blur_sigma_range: tuple[float, float] = (0.1, 1.2)
    blur_kernel: int = 5
    # JPEG artifacts: fires with jpeg_p, quality ~ U{lo..hi}.
    jpeg_p: float = 0.25
    jpeg_quality_range: tuple[int, int] = (40, 85)


DEFAULT_SPEC = ImageAugmentSpec()


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def augment_image(
    image: Tensor,
    generator: torch.Generator,
    spec: ImageAugmentSpec = DEFAULT_SPEC,
) -> Tensor:
    """Apply the full recipe to ONE frame (the caller has already
    decided this frame is augmented). float CHW [3, H, W] in [0, 1]
    → new tensor, same shape/dtype/range; the input is never
    mutated."""
    if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise ValueError(
            f"augment_image expects float CHW [3, H, W] in [0, 1]; got "
            f"shape {tuple(image.shape)} dtype {image.dtype}",
        )
    height, width = int(image.shape[1]), int(image.shape[2])

    # -- the fixed 14-draw parameter block ---------------------------
    scale = _uniform(generator, spec.crop_scale_min, 1.0)
    crop_h = max(1, round(height * scale))
    crop_w = max(1, round(width * scale))
    top = int(torch.randint(height - crop_h + 1, (), generator=generator))
    left = int(torch.randint(width - crop_w + 1, (), generator=generator))
    brightness = _uniform(generator, -spec.brightness_delta, spec.brightness_delta)
    contrast = _uniform(generator, *spec.contrast_range)
    saturation = _uniform(generator, *spec.saturation_range)
    hue = _uniform(generator, -spec.hue_delta, spec.hue_delta)
    log_lo, log_hi = math.log(spec.gamma_range[0]), math.log(spec.gamma_range[1])
    gamma = math.exp(_uniform(generator, log_lo, log_hi))
    noise_fires = float(torch.rand((), generator=generator)) < spec.noise_p
    noise_sigma = _uniform(generator, *spec.noise_sigma_range)
    blur_fires = float(torch.rand((), generator=generator)) < spec.blur_p
    blur_sigma = _uniform(generator, *spec.blur_sigma_range)
    jpeg_fires = float(torch.rand((), generator=generator)) < spec.jpeg_p
    quality = int(
        torch.randint(
            spec.jpeg_quality_range[0],
            spec.jpeg_quality_range[1] + 1,
            (),
            generator=generator,
        ),
    )

    # -- apply, fixed physical order ---------------------------------
    out = image.to(torch.float32)
    if (crop_h, crop_w) != (height, width):
        out = out[:, top : top + crop_h, left : left + crop_w]
        out = TF.resize(out, [height, width], antialias=True)
    out = (out + brightness).clamp(0.0, 1.0)
    out = TF.adjust_contrast(out, contrast).clamp(0.0, 1.0)
    out = TF.adjust_saturation(out, saturation)
    out = TF.adjust_hue(out, hue)
    out = TF.adjust_gamma(out.clamp(0.0, 1.0), gamma)
    if blur_fires:
        out = TF.gaussian_blur(
            out,
            [spec.blur_kernel, spec.blur_kernel],
            [blur_sigma, blur_sigma],
        )
    if noise_fires:
        noise = torch.randn(out.shape, generator=generator, dtype=out.dtype)
        out = out + noise_sigma * noise
    out = out.clamp(0.0, 1.0)
    if jpeg_fires:
        encoded = encode_jpeg(
            (out * 255.0).round().to(torch.uint8),
            quality=quality,
        )
        decoded = decode_jpeg(encoded)
        # Single-tensor call ⇒ single tensor back (the list overload is
        # the batched form).
        assert isinstance(decoded, Tensor)
        out = decoded.to(torch.float32) / 255.0
    return out
