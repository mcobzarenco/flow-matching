"""KV cache for the Molmo2 (Qwen3) decoder stack.

Far simpler than gemma4's: every layer is an identical full-attention
block — no sliding windows, no KV sharing — so the cache is a plain
per-layer append. It exists for the AR suffix role only (prefill the
multimodal prompt once, continue the suffix against it); design
decision D1 keeps the FLOW path cache-free (taps are its only export).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class _LayerKV:
    keys: Tensor | None = None
    values: Tensor | None = None


@dataclass
class Molmo2KVCache:
    num_layers: int
    layers: list[_LayerKV] = field(init=False)
    # Total number of tokens fed through the model so far (excluding the
    # forward currently in flight).
    seen_tokens: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.layers = [_LayerKV() for _ in range(self.num_layers)]

    def update(
        self,
        layer_idx: int,
        keys: Tensor,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Append new K/V states [B, kv_heads, S, head_dim]; return the
        full states to attend to, [B, kv_heads, T, head_dim]
        (T = seen + S)."""
        layer = self.layers[layer_idx]
        if layer.keys is None or layer.values is None:
            layer.keys, layer.values = keys, values
        else:
            layer.keys = torch.cat([layer.keys, keys], dim=-2)
            layer.values = torch.cat([layer.values, values], dim=-2)
        return layer.keys, layer.values

    def advance(self, q_len: int) -> None:
        """Account for ``q_len`` new tokens; call once per forward, after
        all layers have been updated."""
        self.seen_tokens += q_len
