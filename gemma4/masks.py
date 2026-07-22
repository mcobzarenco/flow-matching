"""Attention mask construction, matching HF's eager-path semantics.

Masks are additive float tensors of shape [B, 1, Sq, Skv]: ``0.0`` where a
position may be attended, ``torch.finfo(dtype).min`` where it may not. When no
mask is needed (single-token decode without padding, or a bidirectional
encoder without padding) ``None`` is returned — adding an all-zero mask is a
bit-exact no-op, so this matches HF's mask-skipping.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .cache import KVCache
from .config import Gemma4TextConfig, LayerType

type MaskMapping = dict[LayerType, Tensor | None]


def _build_mask(
    *,
    batch_size: int,
    q_len: int,
    kv_len: int,
    q_offset: int,
    kv_offset: int,
    sliding_window: int | None,
    padding_mask: Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    q_idx = (torch.arange(q_len, device=device) + q_offset)[None, None, :, None]
    kv_idx = (torch.arange(kv_len, device=device) + kv_offset)[None, None, None, :]
    allowed = kv_idx <= q_idx
    if sliding_window is not None:
        allowed = allowed & (kv_idx > q_idx - sliding_window)
    if padding_mask is not None:
        # padding_mask: [B, total_seen + q_len] bool, True = real token. Slice
        # the columns covered by this mask's kv window.
        cols = padding_mask[:, kv_offset : kv_offset + kv_len].to(device=device)
        allowed = allowed & cols[:, None, None, :]
    allowed = allowed.expand(batch_size, 1, q_len, kv_len)
    min_value = torch.finfo(dtype).min
    return torch.where(
        allowed, torch.tensor(0.0, device=device, dtype=dtype), min_value
    )


def build_text_masks(
    config: Gemma4TextConfig,
    *,
    batch_size: int,
    q_len: int,
    cache: KVCache | None,
    padding_mask: Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> MaskMapping:
    """Per-layer-type causal masks for the decoder.

    ``padding_mask`` is the optional HF-style 2D attention mask of shape
    [B, seen + q_len] with True/1 for real tokens.
    """
    q_offset = cache.seen_tokens if cache is not None else 0
    masks: MaskMapping = {}
    for layer_type in set(config.layer_types):
        if cache is not None:
            kv_len, kv_offset = cache.mask_sizes(layer_type, q_len)
        else:
            kv_len, kv_offset = q_len, 0
        if q_len == 1 and padding_mask is None:
            # Single-token decode: every cached position is visible (sliding
            # caches are already trimmed to the window) => zero mask => skip.
            masks[layer_type] = None
            continue
        masks[layer_type] = _build_mask(
            batch_size=batch_size,
            q_len=q_len,
            kv_len=kv_len,
            q_offset=q_offset,
            kv_offset=kv_offset,
            sliding_window=(
                config.sliding_window if layer_type is LayerType.SLIDING else None
            ),
            padding_mask=padding_mask,
            dtype=dtype,
            device=device,
        )
    return masks


def build_bidirectional_mask(
    valid_mask: Tensor | None, dtype: torch.dtype
) -> Tensor | None:
    """Padding-only bidirectional mask for the vision encoder.

    ``valid_mask``: [B, S] bool, True = real patch. Returns None when nothing
    is masked (bit-exact equivalent of adding zeros).
    """
    if valid_mask is None or bool(valid_mask.all()):
        return None
    min_value = torch.finfo(dtype).min
    mask = torch.where(
        valid_mask[:, None, None, :],
        torch.tensor(0.0, device=valid_mask.device, dtype=dtype),
        min_value,
    )
    return mask.expand(valid_mask.shape[0], 1, valid_mask.shape[1], valid_mask.shape[1])
