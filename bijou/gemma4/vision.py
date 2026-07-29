"""Pure-torch Gemma 4 vision tower (encoder-free patch pipeline).

``pixel_values`` are rows of raw RGB patches in [0, 1] (one row per
``patch_size²`` tile, e.g. 3·16² = 768 features for the E-series) as produced
by the Gemma4 image processor. The tower is:

  patch embedder (linear + learned 2D position embeddings)
  -> 16 bidirectional transformer layers with 2D RoPE and clipped linears
  -> spatial average pooling to the soft-token budget, scaled by sqrt(hidden)
  -> (in the top-level model) RMSNorm + projection into LM space.
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import Gemma4VisionConfig
from .layers import (
    DEFAULT_ATTENTION_BACKEND,
    AttentionBackend,
    DeviceLike,
    RMSNorm,
    activation_fn,
    apply_rotary_pos_emb,
    attention,
    buffer_device,
)
from .masks import MaskSpec, build_bidirectional_mask


class ClippableLinear(nn.Module):
    """Linear layer whose inputs/outputs are clamped to checkpoint-provided
    bounds (used by the vision tower for numerical stability)."""

    input_min: Tensor
    input_max: Tensor
    output_min: Tensor
    output_max: Tensor

    def __init__(
        self,
        config: Gemma4VisionConfig,
        in_features: int,
        out_features: int,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if self.use_clipped_linears:
            # Persistent buffers: the actual clip bounds come from the checkpoint.
            for name, value in (
                ("input_min", -float("inf")),
                ("input_max", float("inf")),
                ("output_min", -float("inf")),
                ("output_max", float("inf")),
            ):
                self.register_buffer(
                    name,
                    torch.tensor(value, device=device, dtype=dtype),
                )

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.use_clipped_linears:
            x = torch.clamp(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipped_linears:
            x = torch.clamp(x, self.output_min, self.output_max)
        return x


class VisionPatchEmbedder(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        patch_features = 3 * config.patch_size**2
        self.input_proj = nn.Linear(
            patch_features,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.position_embedding_table = nn.Parameter(
            torch.ones(
                2,
                config.position_embedding_size,
                config.hidden_size,
                device=device,
                dtype=dtype,
            ),
        )

    @override
    def forward(
        self,
        pixel_values: Tensor,
        pixel_position_ids: Tensor,
        padding_positions: Tensor,
    ) -> Tensor:
        """Embed raw patches + 2D learned position; returns
        [images, patches, hidden].

        Shapes (``images`` leads throughout the tower — each camera image
        is one batch row here, Σ per-sample cameras at the model level):
          - pixel_values: [images, patches, 3·patch_size²]  (in [0, 1])
          - pixel_position_ids: [images, patches, 2]  ((x, y), (-1, -1) = pad)
          - padding_positions: [images, patches]  (bool, True = pad)
        """
        pixel_values = 2 * (pixel_values - 0.5)
        pixel_values = pixel_values.to(self.input_proj.weight.dtype)
        hidden_states = self.input_proj(pixel_values)

        clamped = pixel_position_ids.clamp(min=0)
        x_emb = F.embedding(clamped[..., 0], self.position_embedding_table[0])
        y_emb = F.embedding(clamped[..., 1], self.position_embedding_table[1])
        position_embeddings = torch.where(
            padding_positions.unsqueeze(-1),
            0.0,
            x_emb + y_emb,
        )
        return hidden_states + position_embeddings


def vision_rope_inv_freq(
    config: Gemma4VisionConfig,
    *,
    device: DeviceLike = None,
) -> Tensor:
    """Per-spatial-dimension inverse frequencies (x and y share the range)."""
    spatial_dim = config.head_dim // 2
    exponent = torch.arange(0, spatial_dim, 2, dtype=torch.int64, device=device)
    return 1.0 / (config.rope_theta ** (exponent.to(dtype=torch.float) / spatial_dim))


class VisionRotaryEmbedding(nn.Module):
    inv_freq: Tensor

    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        device: DeviceLike = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            vision_rope_inv_freq(config, device=buffer_device(device)),
            persistent=False,
        )

    @torch.no_grad()
    @override
    def forward(
        self,
        position_ids: Tensor,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """2D spatial RoPE tables.

        Shapes:
          - position_ids: [images, patches, 2]
          - returns (cos, sin), each [images, patches, head_dim]  (the two
            spatial dimensions' tables concatenated)
        """
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        all_cos: list[Tensor] = []
        all_sin: list[Tensor] = []
        for i in range(2):
            dim_position_ids = position_ids[:, :, i][:, None, :].float()
            freqs = (inv_freq_expanded.float() @ dim_position_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        return (
            torch.cat(all_cos, dim=-1).to(dtype),
            torch.cat(all_sin, dim=-1).to(dtype),
        )


def apply_multidimensional_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    ndim: int = 2,
) -> Tensor:
    """Split the head dim into ``ndim`` equal parts and rotate each with its
    spatial dimension's cos/sin table. x [images, patches, heads, head_dim];
    cos/sin [images, patches, head_dim]."""
    per_dim = 2 * (x.shape[-1] // (2 * ndim))
    split_sizes = [per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    rotated = [
        apply_rotary_pos_emb(x_parts[k], cos_parts[k], sin_parts[k], unsqueeze_dim=2)
        for k in range(ndim)
    ]
    return torch.cat(rotated, dim=-1)


class VisionAttention(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        # Mutable on purpose: see bijou.gemma4.model.set_attention_backend.
        self.attn_backend = attn_backend
        hidden = config.hidden_size
        self.q_proj = ClippableLinear(
            config,
            hidden,
            config.num_attention_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )
        self.k_proj = ClippableLinear(
            config,
            hidden,
            config.num_key_value_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_proj = ClippableLinear(
            config,
            hidden,
            config.num_key_value_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )
        self.o_proj = ClippableLinear(
            config,
            config.num_attention_heads * self.head_dim,
            hidden,
            device=device,
            dtype=dtype,
        )
        self.q_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.v_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            with_scale=False,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: MaskSpec,
    ) -> Tensor:
        """Bidirectional patch self-attention; returns
        [images, patches, hidden].

        Shapes:
          - hidden_states: [images, patches, hidden]
          - position_embeddings: (cos, sin), each [images, patches, head_dim]
          - attention_mask.tensor: [images, 1, patches, patches]
        """
        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, self.head_dim)
        cos, sin = position_embeddings

        query = self.q_proj(hidden_states).view(hidden_shape)
        query = self.q_norm(query)
        query = apply_multidimensional_rope(query, cos, sin)
        query = query.transpose(1, 2)

        key = self.k_proj(hidden_states).view(hidden_shape)
        key = self.k_norm(key)
        key = apply_multidimensional_rope(key, cos, sin)
        key = key.transpose(1, 2)

        value = self.v_proj(hidden_states).view(hidden_shape)
        value = self.v_norm(value)
        value = value.transpose(1, 2)

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


class VisionMLP(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden, intermediate = config.hidden_size, config.intermediate_size
        self.gate_proj = ClippableLinear(
            config,
            hidden,
            intermediate,
            device=device,
            dtype=dtype,
        )
        self.up_proj = ClippableLinear(
            config,
            hidden,
            intermediate,
            device=device,
            dtype=dtype,
        )
        self.down_proj = ClippableLinear(
            config,
            intermediate,
            hidden,
            device=device,
            dtype=dtype,
        )
        self.act_fn = activation_fn(config.hidden_activation)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Gated GLU MLP; shape-preserving.

        Shapes:
          - x: [images, patches, hidden]  (returns [images, patches, hidden])
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.self_attn = VisionAttention(
            config,
            attn_backend=attn_backend,
            device=device,
            dtype=dtype,
        )
        self.mlp = VisionMLP(config, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(hidden, eps=eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(
            hidden,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self.pre_feedforward_layernorm = RMSNorm(
            hidden,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self.post_feedforward_layernorm = RMSNorm(
            hidden,
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
    ) -> Tensor:
        """self-attn -> MLP encoder block; returns [images, patches, hidden].

        Shapes:
          - hidden_states: [images, patches, hidden]
          - position_embeddings: (cos, sin), each [images, patches, head_dim]
          - attention_mask.tensor: [images, 1, patches, patches]
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class VisionEncoder(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.rotary_emb = VisionRotaryEmbedding(config, device=device)
        self.layers = nn.ModuleList(
            VisionEncoderLayer(
                config,
                attn_backend=attn_backend,
                device=device,
                dtype=dtype,
            )
            for _ in range(config.num_hidden_layers)
        )

    @override
    def forward(
        self,
        inputs_embeds: Tensor,
        valid_mask: Tensor,
        pixel_position_ids: Tensor,
    ) -> Tensor:
        """Run the ViT encoder; returns [images, patches, hidden].

        Shapes:
          - inputs_embeds: [images, patches, hidden]
          - valid_mask: [images, patches]  (bool, True = real patch)
          - pixel_position_ids: [images, patches, 2]
        """
        attention_mask = MaskSpec(
            tensor=build_bidirectional_mask(valid_mask, inputs_embeds.dtype),
        )
        position_embeddings = self.rotary_emb(pixel_position_ids, inputs_embeds.dtype)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)
        return hidden_states


class VisionPooler(nn.Module):
    """Average-pools patch features spatially down to the soft-token budget
    and scales by sqrt(hidden_size) in float32 (output stays float32)."""

    def __init__(self, config: Gemma4VisionConfig) -> None:
        super().__init__()
        self.root_hidden_size = config.hidden_size**0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: Tensor,
        pixel_position_ids: Tensor,
        length: int,
    ) -> tuple[Tensor, Tensor]:
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        if k * k * length != input_seq_len:
            raise ValueError(
                f"cannot pool {input_seq_len} patches to {length} soft tokens (k={k})",
            )
        clamped = pixel_position_ids.clamp(min=0)
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / (k * k)
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    @override
    def forward(
        self,
        hidden_states: Tensor,
        pixel_position_ids: Tensor,
        padding_positions: Tensor,
        output_length: int,
    ) -> tuple[Tensor, Tensor]:
        """Spatial 3x3 average-pool patches to soft tokens.

        Shapes:
          - hidden_states: [images, patches, hidden]
          - pixel_position_ids: [images, patches, 2]
          - padding_positions: [images, patches]  (bool, True = pad)
          - output_length: soft_tokens-per-image (int, = patches // kernel²)
          - returns (pooled [images, output_length, hidden],
            valid_mask [images, output_length] bool)
        """
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"cannot output more soft tokens ({output_length}) than patches "
                f"({hidden_states.shape[1]})",
            )
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, valid_mask = self._avg_pool_by_positions(
                hidden_states,
                pixel_position_ids,
                output_length,
            )
        else:
            valid_mask = ~padding_positions
        return hidden_states.float() * self.root_hidden_size, valid_mask


class VisionModel(nn.Module):
    """Full vision tower: soft tokens (padding stripped) in float32."""

    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.patch_embedder = VisionPatchEmbedder(config, device=device, dtype=dtype)
        self.encoder = VisionEncoder(
            config,
            attn_backend=attn_backend,
            device=device,
            dtype=dtype,
        )
        self.pooler = VisionPooler(config)

    @override
    def forward(self, pixel_values: Tensor, pixel_position_ids: Tensor) -> Tensor:
        """Patch-embed -> ViT encoder -> spatial pool; returns the valid
        soft tokens flattened across the images.

        Shapes:
          - pixel_values: [images, patches, 3·patch_size²]
          - pixel_position_ids: [images, patches, 2]  ((-1, -1) = pad)
          - returns: [soft_tokens, hidden]  (padding stripped)
        """
        kernel = self.config.pooling_kernel_size
        output_length = pixel_values.shape[-2] // (kernel * kernel)

        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(
            pixel_values,
            pixel_position_ids,
            padding_positions,
        )
        hidden_states = self.encoder(
            inputs_embeds,
            ~padding_positions,
            pixel_position_ids,
        )
        pooled, valid_mask = self.pooler(
            hidden_states,
            pixel_position_ids,
            padding_positions,
            output_length,
        )
        # Strip padded soft tokens and cast back to the working dtype.
        return pooled[valid_mask].to(inputs_embeds.dtype)


class MultimodalEmbedder(nn.Module):
    """Projects vision soft tokens into the language-model embedding space."""

    def __init__(
        self,
        multimodal_hidden_size: int,
        text_hidden_size: int,
        eps: float,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding_pre_projection_norm = RMSNorm(
            multimodal_hidden_size,
            eps=eps,
            with_scale=False,
            device=device,
            dtype=dtype,
        )
        self.embedding_projection = nn.Linear(
            multimodal_hidden_size,
            text_hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(self, inputs_embeds: Tensor) -> Tensor:
        """Project vision soft tokens into the LM embedding space.

        Shapes:
          - inputs_embeds: [soft_tokens, vision_hidden]
          - returns: [soft_tokens, hidden]  (text hidden size)
        """
        return self.embedding_projection(
            self.embedding_pre_projection_norm(inputs_embeds),
        )
