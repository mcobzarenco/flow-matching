"""Pure-torch Gemma 4 text decoder.

Faithful reimplementation of HF's ``Gemma4TextModel``:

- scaled word embeddings (``sqrt(hidden_size)``),
- Per-Layer Embeddings (PLE): a packed auxiliary embedding table plus a
  projection of the input embeddings feed a small residual signal into every
  decoder layer,
- hybrid attention: sliding-window layers with plain RoPE over ``head_dim``
  and global layers with p-RoPE (partial rotation) over the wider
  ``global_head_dim``,
- KV sharing: the last ``num_kv_shared_layers`` layers have no K/V projections
  and reuse the states of the last non-shared layer of the same type,
- optionally double-wide MLPs on the KV-shared layers (``use_double_wide_mlp``,
  used by E2B but not E4B),
- Q/K RMSNorm with scale, V RMSNorm without scale, attention scaling 1.0.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .cache import KVCache
from .config import Gemma4TextConfig, LayerType
from .layers import (
    DEFAULT_ATTENTION_BACKEND,
    AttentionBackend,
    DeviceLike,
    RMSNorm,
    apply_rotary_pos_emb,
    attention,
    buffer_device,
    rope_cos_sin,
    rope_inv_freq_from_params,
)
from .masks import MaskMapping, MaskSpec, build_text_masks

type SharedKV = dict[LayerType, tuple[Tensor, Tensor]]
type RopeMapping = dict[LayerType, tuple[Tensor, Tensor]]


def activation_fn(name: str) -> nn.Module:
    if name == "gelu_pytorch_tanh":
        return nn.GELU(approximate="tanh")
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}")


class ScaledEmbedding(nn.Embedding):
    """Word embedding whose output is multiplied by a constant scale.

    The scale is materialized in the weight dtype (bf16 rounding of e.g.
    ``sqrt(1536)`` is intentional and matches the reference implementation).
    """

    embed_scale: Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            num_embeddings, embedding_dim, padding_idx, device=device, dtype=dtype
        )
        self.register_buffer(
            "embed_scale",
            torch.tensor(embed_scale, device=buffer_device(device)),
            persistent=False,
        )

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input) * self.embed_scale.to(self.weight.dtype)


def rope_inv_freq(
    config: Gemma4TextConfig, layer_type: LayerType, *, device: DeviceLike = None
) -> Tensor:
    """Float32 inverse frequencies for a layer type (see
    :func:`bijou.gemma4.layers.rope_inv_freq_from_params`)."""
    return rope_inv_freq_from_params(
        config.rope_parameters[layer_type],
        config.head_dim_for_type(layer_type),
        device=device,
    )


class TextRotaryEmbedding(nn.Module):
    """Precomputed inverse frequencies per layer type; cos/sin in float32."""

    def __init__(self, config: Gemma4TextConfig, *, device: DeviceLike = None) -> None:
        super().__init__()
        for layer_type in set(config.layer_types):
            self.register_buffer(
                f"inv_freq_{layer_type.name}",
                rope_inv_freq(config, layer_type, device=buffer_device(device)),
                persistent=False,
            )

    def inv_freq(self, layer_type: LayerType) -> Tensor:
        buffer = getattr(self, f"inv_freq_{layer_type.name}")
        assert isinstance(buffer, Tensor)
        return buffer

    @torch.no_grad()
    def forward(
        self, position_ids: Tensor, layer_types: set[LayerType], dtype: torch.dtype
    ) -> RopeMapping:
        return {
            layer_type: rope_cos_sin(self.inv_freq(layer_type), position_ids, dtype)
            for layer_type in layer_types
        }


class TextAttention(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type is LayerType.SLIDING
        self.head_dim = config.head_dim_for_layer(layer_idx)
        self.is_kv_shared_layer = config.is_kv_shared_layer(layer_idx)
        self.is_kv_source_layer = config.is_kv_source_layer(layer_idx)
        # Mutable on purpose: see bijou.gemma4.model.set_attention_backend.
        self.attn_backend = attn_backend

        use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
        if use_alternative_attention:
            num_kv_heads = config.num_global_key_value_heads
            assert num_kv_heads is not None
        else:
            num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // num_kv_heads
        self.k_eq_v = use_alternative_attention

        hidden = config.hidden_size
        bias = config.attention_bias
        self.q_proj = nn.Linear(
            hidden,
            config.num_attention_heads * self.head_dim,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.q_norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            hidden,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        # KV-shared layers have no K/V weights at all.
        self.k_proj: nn.Linear | None = None
        self.v_proj: nn.Linear | None = None
        self.k_norm: RMSNorm | None = None
        self.v_norm: RMSNorm | None = None
        if not self.is_kv_shared_layer:
            self.k_proj = nn.Linear(
                hidden,
                num_kv_heads * self.head_dim,
                bias=bias,
                device=device,
                dtype=dtype,
            )
            if not self.k_eq_v:
                self.v_proj = nn.Linear(
                    hidden,
                    num_kv_heads * self.head_dim,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
            self.k_norm = RMSNorm(
                self.head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype
            )
            self.v_norm = RMSNorm(
                self.head_dim,
                eps=config.rms_norm_eps,
                with_scale=False,
                device=device,
                dtype=dtype,
            )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: MaskSpec,
        shared_kv: SharedKV,
        cache: KVCache | None,
    ) -> Tensor:
        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, self.head_dim)
        cos, sin = position_embeddings

        query = self.q_proj(hidden_states).view(hidden_shape)
        query = self.q_norm(query)
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2)
        query = query.transpose(1, 2)

        if self.is_kv_shared_layer:
            key, value = shared_kv[self.layer_type]
        else:
            assert self.k_proj is not None
            assert self.k_norm is not None and self.v_norm is not None
            key = self.k_proj(hidden_states).view(hidden_shape)
            value = (
                self.v_proj(hidden_states).view(hidden_shape)
                if self.v_proj is not None
                else key
            )
            key = self.k_norm(key)
            key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2)
            key = key.transpose(1, 2)
            value = self.v_norm(value)
            value = value.transpose(1, 2)

            if cache is not None:
                key, value = cache.update(self.layer_idx, key, value)
            if self.is_kv_source_layer:
                shared_kv[self.layer_type] = (key, value)

        attn_output = attention(
            self.attn_backend,
            query,
            key,
            value,
            attention_mask,
            self.num_key_value_groups,
            scaling=1.0,
        )
        return self.o_proj(attn_output.reshape(batch, seq_len, -1))


class TextMLP(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        intermediate = config.intermediate_size_for_layer(layer_idx)
        self.gate_proj = nn.Linear(
            hidden, intermediate, bias=False, device=device, dtype=dtype
        )
        self.up_proj = nn.Linear(
            hidden, intermediate, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate, hidden, bias=False, device=device, dtype=dtype
        )
        self.act_fn = activation_fn(config.hidden_activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    layer_scalar: Tensor

    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        eps = config.rms_norm_eps
        self.self_attn = TextAttention(
            config, layer_idx, attn_backend=attn_backend, device=device, dtype=dtype
        )
        self.mlp = TextMLP(config, layer_idx, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(hidden, eps=eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(
            hidden, eps=eps, device=device, dtype=dtype
        )
        self.pre_feedforward_layernorm = RMSNorm(
            hidden, eps=eps, device=device, dtype=dtype
        )
        self.post_feedforward_layernorm = RMSNorm(
            hidden, eps=eps, device=device, dtype=dtype
        )
        # Persistent buffer, loaded from the checkpoint.
        self.register_buffer("layer_scalar", torch.ones(1, device=device, dtype=dtype))

        self.act_fn = activation_fn(config.hidden_activation)
        self.per_layer_input_gate = nn.Linear(
            hidden,
            config.hidden_size_per_layer_input,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.per_layer_projection = nn.Linear(
            config.hidden_size_per_layer_input,
            hidden,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.post_per_layer_input_norm = RMSNorm(
            hidden, eps=eps, device=device, dtype=dtype
        )

    def forward(
        self,
        hidden_states: Tensor,
        per_layer_input: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: MaskSpec,
        shared_kv: SharedKV,
        cache: KVCache | None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings, attention_mask, shared_kv, cache
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # PLE: gate the hidden state, mix in this layer's slice of the
        # per-layer inputs, project back up and add residually.
        residual = hidden_states
        hidden_states = self.per_layer_input_gate(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = self.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class TextModel(nn.Module):
    """Decoder stack without the LM head."""

    def __init__(
        self,
        config: Gemma4TextConfig,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = ScaledEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            embed_scale=config.hidden_size**0.5,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            DecoderLayer(
                config,
                layer_idx,
                attn_backend=attn_backend,
                device=device,
                dtype=dtype,
            )
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype
        )
        self.rotary_emb = TextRotaryEmbedding(config, device=device)

        # PLE tables/projections.
        self.embed_tokens_per_layer = ScaledEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            padding_idx=config.pad_token_id,
            embed_scale=config.hidden_size_per_layer_input**0.5,
            device=device,
            dtype=dtype,
        )
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.per_layer_projection_norm = RMSNorm(
            config.hidden_size_per_layer_input,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.per_layer_input_scale = 2.0**-0.5
        self.per_layer_model_projection_scale = config.hidden_size**-0.5

    def get_per_layer_inputs(self, input_ids: Tensor) -> Tensor:
        """Token-identity component of PLE: [B, S] -> [B, S, L, ple_dim]."""
        config = self.config
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            config.num_hidden_layers,
            config.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self, inputs_embeds: Tensor, per_layer_inputs: Tensor
    ) -> Tensor:
        """Combine the context projection of ``inputs_embeds`` with the
        token-identity component: ``(proj + identity) / sqrt(2)``."""
        config = self.config
        projection = (
            self.per_layer_model_projection(inputs_embeds)
            * self.per_layer_model_projection_scale
        )
        projection = projection.reshape(
            *inputs_embeds.shape[:-1],
            config.num_hidden_layers,
            config.hidden_size_per_layer_input,
        )
        projection = self.per_layer_projection_norm(projection)
        return (projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        inputs_embeds: Tensor | None = None,
        per_layer_inputs: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_masks: MaskMapping | None = None,
        padding_mask: Tensor | None = None,
        cache: KVCache | None = None,
    ) -> Tensor:
        """Returns the final hidden states [B, S, hidden].

        Exactly one of ``input_ids`` / ``inputs_embeds`` must be given; when
        passing ``inputs_embeds`` (the multimodal path), ``per_layer_inputs``
        (the raw output of :meth:`get_per_layer_inputs`) is required as well.
        ``padding_mask`` is a 2D HF-style attention mask [B, seen + S]
        (True/1 = real token) used to build masks when ``attention_masks`` is
        not provided.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids)
        elif per_layer_inputs is None:
            raise ValueError(
                "per_layer_inputs is required when passing inputs_embeds "
                "(this implementation does not reverse the embedding table)"
            )
        assert inputs_embeds is not None
        per_layer_inputs = self.project_per_layer_inputs(
            inputs_embeds, per_layer_inputs
        )

        batch, q_len, _ = inputs_embeds.shape
        if position_ids is None:
            past = cache.seen_tokens if cache is not None else 0
            position_ids = (
                torch.arange(q_len, device=inputs_embeds.device) + past
            ).unsqueeze(0)
        if attention_masks is None:
            attention_masks = build_text_masks(
                self.config,
                batch_size=batch,
                q_len=q_len,
                cache=cache,
                padding_mask=padding_mask,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        position_embeddings = self.rotary_emb(
            position_ids, set(self.config.layer_types), inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        shared_kv: SharedKV = {}
        for i, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            hidden_states = layer(
                hidden_states,
                per_layer_inputs[:, :, i, :],
                position_embeddings[layer_type],
                attention_masks[layer_type],
                shared_kv,
                cache,
            )
        if cache is not None:
            cache.advance(q_len)

        return self.norm(hidden_states)
