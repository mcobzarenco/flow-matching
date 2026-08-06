"""Pure-torch Molmo2 text decoder (stock Qwen3-4B geometry).

Faithful reimplementation of the checkpoint's ``modeling_molmo2.py`` text
path (conventions pinned in ``docs/molmo2.md``):

- untied embeddings with a separate extension matrix (``wte.embedding`` +
  ``wte.new_embedding``, concatenated for lookup; NO input scaling),
- 36 identical pre-norm layers: fused-QKV grouped-query attention
  (32 q-heads : 8 kv-heads, head_dim 128, scaling ``head_dim**-0.5``) and a
  fused-gate SwiGLU MLP,
- qwen3-style qk-norm: per-head RMSNorm(head_dim) on q and k BEFORE RoPE,
- one plain rotary embedding for the whole stack (theta 5e6, full head_dim;
  ``rope_scaling_layers`` is null in the 4B SKU and refused at config parse),
- full causal attention everywhere: no sliding windows, no KV sharing, no
  PLE, no softcapping.

The forward supports the ``residual_taps``/``residual_sink`` protocol of
``bijou.gemma4.text.TextModel`` (tap = the hidden state AFTER a layer, post
both residual adds) — under design decision D1 of the port plan this is the
ONLY export the flow expert consumes, so there is no KV cache and no
``kv_stop_layer`` here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..nn import (
    DEFAULT_ATTENTION_BACKEND,
    AttentionBackend,
    DeviceLike,
    MaskSpec,
    RMSNorm,
    RopeParameters,
    RopeType,
    activation_fn,
    apply_rotary_pos_emb,
    attention,
    buffer_device,
    rope_cos_sin,
    rope_inv_freq_from_params,
)
from .cache import Molmo2KVCache
from .config import Molmo2TextConfig


class Molmo2Embedding(nn.Module):
    """Token embedding with a separate extension matrix.

    Ids in ``[0, vocab_size)`` index ``embedding``; the ``additional``
    special ids (image markers etc.) index ``new_embedding``. The two
    matrices are concatenated for lookup, mirroring the reference
    ``Molmo2Embedding`` — they stay separate parameters because the
    checkpoint ships them separately (``wte.embedding`` /
    ``wte.new_embedding``).
    """

    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        embedding_dim: int,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
        )
        self.new_embedding = nn.Parameter(
            torch.empty(num_new_embeddings, embedding_dim, device=device, dtype=dtype),
        )
        with torch.no_grad():
            self.embedding.normal_(std=0.02)
            self.new_embedding.normal_(std=0.02)

    @override
    def forward(self, input_ids: Tensor) -> Tensor:
        """Embed token ids.

        Row-select against the two matrices directly instead of the
        reference's ``cat`` lookup: concatenating materializes a full
        [vocab+ext, hidden] copy (~1.5 GB fp32 at 4B scale) on EVERY
        call — twice per training step and once per decode step.
        Lookup semantics are bitwise-identical (rows are selected, never
        computed); the WP1/WP2 parity suites gate that equivalence.

        Shapes:
          - input_ids: [B, S]  (returns [B, S, embedding_dim])
        """
        base_rows = self.embedding.shape[0]
        embeds = F.embedding(input_ids.clamp(max=base_rows - 1), self.embedding)
        is_extension = input_ids >= base_rows
        if bool(is_extension.any()):
            extension = F.embedding(
                (input_ids - base_rows).clamp(min=0),
                self.new_embedding,
            )
            embeds = torch.where(is_extension[..., None], extension, embeds)
        return embeds


class Molmo2RotaryEmbedding(nn.Module):
    """Single plain-RoPE table shared by every layer; cos/sin in float32."""

    inv_freq: Tensor

    def __init__(self, config: Molmo2TextConfig, *, device: DeviceLike = None) -> None:
        super().__init__()
        params = RopeParameters(
            rope_type=RopeType.DEFAULT,
            rope_theta=config.rope_theta,
            factor=1.0,
            partial_rotary_factor=1.0,
        )
        self.register_buffer(
            "inv_freq",
            rope_inv_freq_from_params(
                params,
                config.head_dim,
                device=buffer_device(device),
            ),
            persistent=False,
        )

    @torch.no_grad()
    @override
    def forward(
        self,
        position_ids: Tensor,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """(cos, sin) tables for the given positions, each [B, S, head_dim]."""
        return rope_cos_sin(self.inv_freq, position_ids, dtype)


def build_causal_mask(
    *,
    batch_size: int,
    q_len: int,
    padding_mask: Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
    seen: int = 0,
) -> MaskSpec:
    """Full causal attention mask for one forward.

    ``padding_mask``: [B, T] (T = seen + q_len), True/1 = real token.
    ``seen`` > 0 is the cache-continuation case (the AR suffix role):
    the q_len query positions sit at logical offsets seen..seen+q_len−1
    and attend causally over all T key positions. The additive tensor is
    always materialized (the eager backend consumes only it); without
    padding or a cache the pattern is exactly lower-triangular and
    ``is_causal`` lets SDPA take its native causal path instead (the
    gemma4 convention).
    """
    kv_len = seen + q_len
    q_idx = torch.arange(q_len, device=device)[None, None, :, None] + seen
    kv_idx = torch.arange(kv_len, device=device)[None, None, None, :]
    allowed = kv_idx <= q_idx
    if padding_mask is not None:
        cols = padding_mask.to(device=device, dtype=torch.bool)
        allowed = allowed & cols[:, None, None, :]
    allowed = allowed.expand(batch_size, 1, q_len, kv_len)
    return MaskSpec(
        tensor=torch.where(
            allowed,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.finfo(dtype).min,
        ),
        is_causal=padding_mask is None and seen == 0,
    )


class TextAttention(nn.Module):
    """Grouped-query attention with fused QKV and qwen3 qk-norm."""

    def __init__(
        self,
        config: Molmo2TextConfig,
        layer_idx: int,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_key_value_groups
        self.scaling = config.head_dim**-0.5
        # Mutable on purpose, matching gemma4 (see set_attention_backend).
        self.attn_backend = attn_backend

        q_dim = config.num_attention_heads * config.head_dim
        kv_dim = config.num_key_value_heads * config.head_dim
        self.fused_dims = (q_dim, kv_dim, kv_dim)
        self.att_proj = nn.Linear(
            config.hidden_size,
            sum(self.fused_dims),
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.attn_out = nn.Linear(
            q_dim,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.q_norm = RMSNorm(
            config.head_dim,
            eps=config.layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.k_norm = RMSNorm(
            config.head_dim,
            eps=config.layer_norm_eps,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: MaskSpec,
        cache: Molmo2KVCache | None = None,
    ) -> Tensor:
        """Grouped-query self-attention; returns [B, S, hidden].

        With a ``cache``, this forward's (post-RoPE) K/V are appended to
        the layer's entry and attention runs over the full T = seen + S
        key positions (the mask must then be [B, 1, S, T]).

        Shapes:
          - hidden_states: [B, S, hidden]
          - position_embeddings: (cos, sin), each [B, S, head_dim]
          - attention_mask.tensor (when present): [B, 1, S, T]
        """
        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, self.head_dim)
        cos, sin = position_embeddings

        query, key, value = self.att_proj(hidden_states).split(self.fused_dims, dim=-1)
        query = self.q_norm(query.view(hidden_shape))
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2)
        query = query.transpose(1, 2)
        key = self.k_norm(key.view(hidden_shape))
        key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2)
        key = key.transpose(1, 2)
        value = value.view(hidden_shape).transpose(1, 2)
        if cache is not None:
            key, value = cache.update(self.layer_idx, key, value)

        attn_output = attention(
            self.attn_backend,
            query,
            key,
            value,
            attention_mask,
            self.num_key_value_groups,
            scaling=self.scaling,
        )
        return self.attn_out(attn_output.reshape(batch, seq_len, -1))


class TextMLP(nn.Module):
    """SwiGLU MLP with the reference's fused gate projection.

    ``ff_proj`` packs [up, gate] along the output dim (in that order — the
    FIRST chunk is the multiplicand, the SECOND is activated).
    """

    def __init__(
        self,
        config: Molmo2TextConfig,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ff_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size * 2,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.ff_out = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.act = activation_fn(config.hidden_act)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Gated GLU MLP; shape-preserving.

        Shapes:
          - x: [B, S, hidden]  (returns [B, S, hidden])
        """
        x, gate = self.ff_proj(x).chunk(2, dim=-1)
        return self.ff_out(self.act(gate) * x)


class DecoderLayer(nn.Module):
    """Pre-norm block: ``x += attn(attn_norm(x)); x += mlp(ff_norm(x))``."""

    def __init__(
        self,
        config: Molmo2TextConfig,
        layer_idx: int,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        eps = config.layer_norm_eps
        self.self_attn = TextAttention(
            config,
            layer_idx,
            attn_backend=attn_backend,
            device=device,
            dtype=dtype,
        )
        self.mlp = TextMLP(config, device=device, dtype=dtype)
        self.attn_norm = RMSNorm(
            config.hidden_size,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self.ff_norm = RMSNorm(
            config.hidden_size,
            eps=eps,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: MaskSpec,
        cache: Molmo2KVCache | None = None,
    ) -> Tensor:
        """self-attn -> MLP with pre-norms; returns [B, S, hidden]."""
        hidden_states = hidden_states + self.self_attn(
            self.attn_norm(hidden_states),
            position_embeddings,
            attention_mask,
            cache,
        )
        return hidden_states + self.mlp(self.ff_norm(hidden_states))


class Molmo2Transformer(nn.Module):
    """Decoder stack without the LM head (module names mirror the
    checkpoint: ``wte`` / ``blocks`` / ``ln_f``)."""

    def __init__(
        self,
        config: Molmo2TextConfig,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.wte = Molmo2Embedding(
            config.vocab_size,
            config.additional_vocab_size,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.blocks = nn.ModuleList(
            DecoderLayer(
                config,
                layer_idx,
                attn_backend=attn_backend,
                device=device,
                dtype=dtype,
            )
            for layer_idx in range(config.num_hidden_layers)
        )
        self.ln_f = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.rotary_emb = Molmo2RotaryEmbedding(config, device=device)

    @override
    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        inputs_embeds: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_mask: MaskSpec | None = None,
        padding_mask: Tensor | None = None,
        cache: Molmo2KVCache | None = None,
        residual_taps: Sequence[int] = (),
        residual_sink: dict[int, Tensor] | None = None,
    ) -> Tensor:
        """Returns the final-norm'd hidden states [B, S, hidden].

        Exactly one of ``input_ids`` / ``inputs_embeds`` must be given
        (``inputs_embeds`` is the multimodal path — WP2 adds image features
        into the embedding sequence before calling this).

        ``cache``: prefill/continuation for the AR suffix role — each
        layer appends its (post-RoPE) K/V and attends over the full
        T = seen + S keys. With a non-empty cache and no explicit
        ``attention_mask``, the default mask is the shifted-causal
        continuation mask; ``padding_mask`` is then [B, T] (the prompt's
        real-token mask extended with ones for fed suffix positions).
        Explicit ``position_ids`` are required under left padding either
        way (logical positions of the real tokens).

        ``residual_taps``/``residual_sink``: record the hidden state AFTER
        each listed layer (post both residual adds — the residual stream the
        next layer consumes, WITHOUT ``ln_f``) into the caller's sink dict,
        [B, S, hidden] per tap. Both or neither.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            inputs_embeds = self.wte(input_ids)
        assert inputs_embeds is not None

        batch, q_len, _ = inputs_embeds.shape
        seen = cache.seen_tokens if cache is not None else 0
        if position_ids is None:
            position_ids = (
                torch.arange(
                    q_len,
                    device=inputs_embeds.device,
                )
                + seen
            ).unsqueeze(0)
        if attention_mask is None:
            attention_mask = build_causal_mask(
                batch_size=batch,
                q_len=q_len,
                padding_mask=padding_mask,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                seen=seen,
            )

        if bool(residual_taps) != (residual_sink is not None):
            raise ValueError("residual_taps and residual_sink travel together")
        taps = frozenset(residual_taps)
        for tap in taps:
            if not 0 <= tap < len(self.blocks):
                raise ValueError(
                    f"residual tap {tap} outside the stack ({len(self.blocks)} layers)",
                )

        position_embeddings = self.rotary_emb(position_ids, inputs_embeds.dtype)
        hidden_states = inputs_embeds
        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                position_embeddings,
                attention_mask,
                cache,
            )
            if i in taps:
                assert residual_sink is not None  # validated above
                residual_sink[i] = hidden_states
        if cache is not None:
            cache.advance(q_len)

        return self.ln_f(hidden_states)


class Molmo2TextModel(nn.Module):
    """Transformer plus optional LM head.

    The mount (truncated prefix, taps-only) never predicts tokens, so it is
    built without the head; the full-depth parity harness needs it. Module
    names mirror the checkpoint after stripping the ``model.`` prefix
    (``lm_head`` sits OUTSIDE ``model.`` in the checkpoint, beside it here).
    """

    def __init__(
        self,
        config: Molmo2TextConfig,
        *,
        lm_head: bool,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.transformer = Molmo2Transformer(
            config,
            attn_backend=attn_backend,
            device=device,
            dtype=dtype,
        )
        self.lm_head: nn.Linear | None = None
        if lm_head:
            # The head covers the BASE vocab only (the extension ids are
            # input-side specials), matching the reference.
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
            )

    @override
    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        inputs_embeds: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_mask: MaskSpec | None = None,
        padding_mask: Tensor | None = None,
        residual_taps: Sequence[int] = (),
        residual_sink: dict[int, Tensor] | None = None,
    ) -> Tensor:
        """Logits [B, S, vocab] with a head, else hidden states [B, S, hidden]."""
        hidden_states = self.transformer(
            input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            residual_taps=residual_taps,
            residual_sink=residual_sink,
        )
        if self.lm_head is None:
            return hidden_states
        return self.lm_head(hidden_states)
