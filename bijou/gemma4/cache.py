"""KV cache matching HF's ``DynamicCache`` semantics for Gemma4.

Sliding-window layers store only the trailing ``sliding_window - 1`` positions;
their ``update()`` still returns the full concatenation for the current step,
exactly like HF's ``DynamicSlidingWindowLayer``. KV-shared layers (the last
``num_kv_shared_layers`` of the stack) never own cache entries — they reuse the
states produced by the last non-shared layer of the same type within a forward
pass (see ``text.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .config import Gemma4TextConfig, LayerType


@dataclass
class _LayerKV:
    keys: Tensor | None = None
    values: Tensor | None = None


@dataclass
class KVCache:
    config: Gemma4TextConfig
    layers: list[_LayerKV] = field(init=False)
    # Total number of tokens fed through the model so far (excluding the
    # forward currently in flight).
    seen_tokens: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.layers = [_LayerKV() for _ in range(self.config.num_hidden_layers)]

    def update(
        self,
        layer_idx: int,
        keys: Tensor,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Append new K/V states [B, kv_heads, S, head_dim]; return the
        states to attend to, [B, kv_heads, T, head_dim] (T = seen + S;
        sliding layers then trim their STORED copy to the window, but the
        full pre-trim states are what this call returns)."""
        layer = self.layers[layer_idx]
        if layer.keys is None or layer.values is None:
            full_keys, full_values = keys, values
        else:
            full_keys = torch.cat([layer.keys, keys], dim=-2)
            full_values = torch.cat([layer.values, values], dim=-2)

        if self.config.layer_types[layer_idx] is LayerType.SLIDING:
            window = self.config.sliding_window
            layer.keys = full_keys[:, :, -window + 1 :, :]
            layer.values = full_values[:, :, -window + 1 :, :]
        else:
            layer.keys = full_keys
            layer.values = full_values
        return full_keys, full_values

    def mask_sizes(self, layer_type: LayerType, q_len: int) -> tuple[int, int]:
        """(kv_length, kv_offset) for mask construction, pre-update semantics."""
        if layer_type is LayerType.SLIDING:
            window = self.config.sliding_window
            kv_offset = max(self.seen_tokens - window + 1, 0)
            if self.seen_tokens >= window:
                kv_length = window - 1 + q_len
            else:
                kv_length = self.seen_tokens + q_len
            return kv_length, kv_offset
        return self.seen_tokens + q_len, 0

    def advance(self, q_len: int) -> None:
        """Account for ``q_len`` new tokens; call once per forward, after all
        layers have been updated."""
        self.seen_tokens += q_len
