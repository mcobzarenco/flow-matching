"""Bijou action expert: a narrow flow-matching decoder over action chunks.

The expert consumes ``[state][action_1..action_chunk]`` tokens and, per layer,
cross-attends one *global-attention* K/V stream exported from the frozen
Gemma 4 prefix (layers 4/9/14 for E2B), then self-attends the suffix, then
runs a gated MLP. It predicts the flow-matching velocity of the action chunk
at flow time τ.

Design notes (see the design discussion in the repo history):

- Cross-attention queries adopt the backbone's global-attention geometry so
  the exported K/V are consumed exactly as the backbone's own deep layers
  consume them: head_dim = ``global_head_dim`` (512), q-RMSNorm, p-RoPE at
  positions continuing after each sample's REAL (unpadded) prefix,
  attention scaling 1.0.
- The per-layer stream assignment is the ``cross_attention_schedule`` tuple
  (its length is the expert depth), e.g. blocks ``(4,4,4,4, 9,9,9,9,
  14,...)``; cycle/hybrid schedules are config diffs, not code paths.
- Self-attention over the suffix is bidirectional or causal-over-actions
  (state visible to and from everything in both modes) — an explicit ablation
  knob, ``SelfAttentionMode``.
- Flow time τ enters as a sinusoidal embedding, MLP-transformed and added to
  the action token embeddings (π0-style); the state token is not
  time-conditioned.

Flow-matching convention (matches lerobot's π0/SmolVLA):
``x_τ = τ·ε + (1−τ)·actions`` with ε ~ N(0, I), so τ=1 is pure noise; the
velocity target is ``u = ε − actions``; sampling integrates from τ=1 to 0
with steps of ``dτ = −1/num_steps``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import cast, override

import torch
from torch import Tensor, nn

from .gemma4.config import RopeParameters, RopeType
from .gemma4.layers import (
    DEFAULT_ATTENTION_BACKEND,
    AttentionBackend,
    DeviceLike,
    RMSNorm,
    activation_fn,
    apply_rotary_pos_emb,
    attention,
    buffer_device,
    rope_cos_sin,
    rope_inv_freq_from_params,
)
from .gemma4.masks import MaskSpec

type StreamKV = dict[int, tuple[Tensor, Tensor]]


@dataclass(frozen=True, slots=True)
class PrefixKV:
    """Exported prefix K/V streams: {backbone_layer_idx: (K, V)}, each
    [B, kv_heads, P, head_dim], plus the (padded) prefix length P and, for
    padded batches, the True-means-real padding mask [B, P]. Per-sample
    real lengths (expert query positions) derive from the mask; ``length``
    is the KV width and the position base only for unpadded batches."""

    streams: StreamKV
    length: int
    padding_mask: Tensor | None = None

    @property
    def batch_size(self) -> int:
        first_key = next(iter(self.streams.values()))[0]
        return first_key.shape[0]


class SelfAttentionMode(StrEnum):
    """Masking of the expert's self-attention over ``[state][actions]``.

    The state token attends and is attended by everything in both modes;
    in CAUSAL_ACTIONS each action token only attends earlier actions (the
    SmolVLA ablation found this beats bidirectional; π0 uses bidirectional).
    """

    BIDIRECTIONAL = "bidirectional"
    CAUSAL_ACTIONS = "causal_actions"


@dataclass(frozen=True, slots=True)
class ExpertConfig:
    """Architecture of the action expert. Use
    :func:`bijou.loading.default_expert_config` to derive one from a backbone
    config with the blocks-schedule knobs."""

    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    hidden_activation: str
    rms_norm_eps: float

    self_attention_mode: SelfAttentionMode
    self_attention_rope_theta: float

    # Cross-attention geometry, copied from the backbone's global layers.
    cross_attention_heads: int
    cross_attention_head_dim: int
    cross_attention_rope: RopeParameters
    # Backbone layer index each expert layer cross-attends; the length of
    # this tuple is the expert depth.
    cross_attention_schedule: tuple[int, ...]

    action_dim: int
    state_dim: int
    chunk_size: int
    time_embed_dim: int

    def __post_init__(self) -> None:
        if not self.cross_attention_schedule:
            raise ValueError("cross_attention_schedule must not be empty")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("self-attention head_dim must be even (RoPE)")
        if self.time_embed_dim % 2:
            raise ValueError("time_embed_dim must be even")

    @property
    def num_layers(self) -> int:
        return len(self.cross_attention_schedule)

    @property
    def streams(self) -> tuple[int, ...]:
        """Backbone layers whose K/V the expert consumes, ascending."""
        return tuple(sorted(set(self.cross_attention_schedule)))

    @property
    def suffix_length(self) -> int:
        return 1 + self.chunk_size


def sinusoidal_time_embedding(
    time: Tensor,
    dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> Tensor:
    """[B] flow times in [0, 1] -> [B, dim] float32 sin/cos features.

    Geometric period range tuned for the unit interval (π0's choice), rather
    than the 10k-period convention used for token positions.
    """
    half = dim // 2
    fraction = torch.arange(half, dtype=torch.float32, device=time.device) / half
    period = min_period * (max_period / min_period) ** fraction
    angle = time[:, None].float() / period[None, :] * (2 * math.pi)
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class ExpertCrossAttention(nn.Module):
    """Queries in backbone global-attention geometry over an exported stream."""

    def __init__(
        self,
        config: ExpertConfig,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = config.cross_attention_heads
        self.head_dim = config.cross_attention_head_dim
        self.attn_backend = attn_backend
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.q_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(
        self,
        hidden_states: Tensor,
        stream: tuple[Tensor, Tensor],
        position_embeddings: tuple[Tensor, Tensor],
        mask: MaskSpec,
    ) -> Tensor:
        batch, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        query = self.q_proj(hidden_states).view(batch, seq_len, -1, self.head_dim)
        query = self.q_norm(query)
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2)
        query = query.transpose(1, 2)

        key, value = stream  # [B, kv_heads, P, head_dim], already normed+rope'd
        attn_output = attention(
            self.attn_backend,
            query,
            key,
            value,
            mask,
            num_key_value_groups=self.num_heads // key.shape[1],
            scaling=1.0,
        )
        return self.o_proj(attn_output.reshape(batch, seq_len, -1))


class ExpertSelfAttention(nn.Module):
    """Gemma-flavored self-attention over the ``[state][actions]`` suffix."""

    def __init__(
        self,
        config: ExpertConfig,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attn_backend = attn_backend
        hidden = config.hidden_size
        eps = config.rms_norm_eps
        self.q_proj = nn.Linear(hidden, hidden, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden, hidden, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden, hidden, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(self.head_dim, eps=eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=eps, device=device, dtype=dtype)
        self.v_norm = RMSNorm(
            self.head_dim,
            eps=eps,
            with_scale=False,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        mask: MaskSpec,
    ) -> Tensor:
        batch, seq_len, _ = hidden_states.shape
        shape = (batch, seq_len, self.num_heads, self.head_dim)
        cos, sin = position_embeddings

        query = self.q_proj(hidden_states).view(shape)
        query = self.q_norm(query)
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2)
        key = self.k_proj(hidden_states).view(shape)
        key = self.k_norm(key)
        key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2)
        value = self.v_norm(self.v_proj(hidden_states).view(shape))

        attn_output = attention(
            self.attn_backend,
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            mask,
            num_key_value_groups=1,
            scaling=1.0,
        )
        return self.o_proj(attn_output.reshape(batch, seq_len, -1))


class ExpertMLP(nn.Module):
    def __init__(
        self,
        config: ExpertConfig,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden, intermediate = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(
            hidden,
            intermediate,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up_proj = nn.Linear(
            hidden,
            intermediate,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.down_proj = nn.Linear(
            intermediate,
            hidden,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.act_fn = activation_fn(config.hidden_activation)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ExpertLayer(nn.Module):
    """cross-attention -> self-attention -> MLP, Gemma-style sandwich norms."""

    def __init__(
        self,
        config: ExpertConfig,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.cross_attn = ExpertCrossAttention(
            config,
            attn_backend=attn_backend,
            device=device,
            dtype=dtype,
        )
        self.self_attn = ExpertSelfAttention(
            config,
            attn_backend=attn_backend,
            device=device,
            dtype=dtype,
        )
        self.mlp = ExpertMLP(config, device=device, dtype=dtype)
        self.pre_cross_attention_layernorm = RMSNorm(
            hidden,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self.post_cross_attention_layernorm = RMSNorm(
            hidden,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self.pre_self_attention_layernorm = RMSNorm(
            hidden,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self.post_self_attention_layernorm = RMSNorm(
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
        stream: tuple[Tensor, Tensor],
        cross_position_embeddings: tuple[Tensor, Tensor],
        cross_attention_mask: MaskSpec,
        self_position_embeddings: tuple[Tensor, Tensor],
        self_attention_mask: MaskSpec,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.pre_cross_attention_layernorm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states,
            stream,
            cross_position_embeddings,
            cross_attention_mask,
        )
        hidden_states = self.post_cross_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_self_attention_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            self_position_embeddings,
            self_attention_mask,
        )
        hidden_states = self.post_self_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class ActionExpert(nn.Module):
    """Velocity network over an action chunk, conditioned on prefix KV
    streams, robot state and flow time. Freshly initialized (never loaded
    from the backbone checkpoint)."""

    cross_inv_freq: Tensor
    self_inv_freq: Tensor

    def __init__(
        self,
        config: ExpertConfig,
        *,
        attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size

        self.state_proj = nn.Linear(
            config.state_dim,
            hidden,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.action_in_proj = nn.Linear(
            config.action_dim,
            hidden,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.time_in_proj = nn.Linear(
            config.time_embed_dim,
            hidden,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.time_out_proj = nn.Linear(
            hidden,
            hidden,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.time_act = nn.SiLU()

        self.layers = nn.ModuleList(
            ExpertLayer(config, attn_backend=attn_backend, device=device, dtype=dtype)
            for _ in range(config.num_layers)
        )
        self.norm = RMSNorm(hidden, eps=config.rms_norm_eps, device=device, dtype=dtype)
        # Zero-initialized so the initial velocity field is 0 (standard for
        # flow/diffusion heads).
        self.action_out_proj = nn.Linear(
            hidden,
            config.action_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

        self.register_buffer(
            "cross_inv_freq",
            rope_inv_freq_from_params(
                config.cross_attention_rope,
                config.cross_attention_head_dim,
                device=buffer_device(device),
            ),
            persistent=False,
        )
        self.register_buffer(
            "self_inv_freq",
            rope_inv_freq_from_params(
                RopeParameters(
                    rope_type=RopeType.DEFAULT,
                    rope_theta=config.self_attention_rope_theta,
                    factor=1.0,
                    partial_rotary_factor=1.0,
                ),
                config.hidden_size // config.num_attention_heads,
                device=buffer_device(device),
            ),
            persistent=False,
        )
        if device is None or torch.device(device).type != "meta":
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # cast past the stubs: torch types Linear.bias as Parameter,
                # but bias=False modules carry None at runtime.
                bias = cast("nn.Parameter | None", module.bias)
                if bias is not None:
                    nn.init.zeros_(bias)
        nn.init.zeros_(self.action_out_proj.weight)
        assert self.action_out_proj.bias is not None
        nn.init.zeros_(self.action_out_proj.bias)

    def _self_attention_mask(
        self,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> MaskSpec:
        if self.config.self_attention_mode is SelfAttentionMode.BIDIRECTIONAL:
            return MaskSpec()
        length = self.config.suffix_length
        idx = torch.arange(length, device=device)
        # State (position 0) attends and is attended by everything; actions
        # are causal among themselves.
        allowed = (
            (idx[:, None] == 0) | (idx[None, :] == 0) | (idx[None, :] <= idx[:, None])
        )
        min_value = torch.finfo(dtype).min
        tensor = torch.where(
            allowed,
            torch.tensor(0.0, device=device, dtype=dtype),
            min_value,
        )
        return MaskSpec(tensor=tensor[None, None].expand(batch, 1, length, length))

    def _cross_attention_mask(
        self,
        prefix: PrefixKV,
        dtype: torch.dtype,
        device: torch.device,
    ) -> MaskSpec:
        if prefix.padding_mask is None:
            return MaskSpec()
        real = prefix.padding_mask.to(device=device, dtype=torch.bool)
        min_value = torch.finfo(dtype).min
        tensor = torch.where(
            real[:, None, None, :],
            torch.tensor(0.0, device=device, dtype=dtype),
            min_value,
        )
        return MaskSpec(tensor=tensor)

    @override
    def forward(
        self,
        prefix: PrefixKV,
        state: Tensor,
        noisy_actions: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Velocity of the action chunk at flow time τ.

        state: [B, state_dim]; noisy_actions: [B, chunk_size, action_dim];
        time: [B] flow times in [0, 1]. Returns [B, chunk_size, action_dim].
        Inputs and prefix streams are cast to the expert's own dtype (the
        backbone may run in a different precision, e.g. bf16 vs fp32 expert).
        """
        config = self.config
        batch = state.shape[0]
        if noisy_actions.shape[1] != config.chunk_size:
            raise ValueError(
                f"expected chunk of {config.chunk_size} actions, "
                f"got {noisy_actions.shape[1]}",
            )
        dtype = self.state_proj.weight.dtype

        state_embeds = self.state_proj(state.to(dtype))[:, None, :]
        action_embeds = self.action_in_proj(noisy_actions.to(dtype))
        time_embeds = sinusoidal_time_embedding(time, config.time_embed_dim)
        time_embeds = self.time_out_proj(
            self.time_act(self.time_in_proj(time_embeds.to(dtype))),
        )
        action_embeds = action_embeds + time_embeds[:, None, :]
        hidden_states = torch.cat([state_embeds, action_embeds], dim=1)

        device = hidden_states.device
        streams = {
            idx: (k.to(dtype), v.to(dtype)) for idx, (k, v) in prefix.streams.items()
        }
        suffix_positions = torch.arange(config.suffix_length, device=device)
        # Cross-attention queries continue after each sample's REAL prefix.
        # Using the padded batch width here would shift every query->key
        # RoPE distance by that sample's padding, making predictions depend
        # on batch-mates' prompt lengths (measured: max|delta| 0.55 on the
        # expert alone, outputs/probe_effect1_fix.py).
        if prefix.padding_mask is not None:
            real_lengths = prefix.padding_mask.to(device=device, dtype=torch.long).sum(
                dim=1,
            )
            cross_positions = real_lengths[:, None] + suffix_positions[None, :]
        else:
            cross_positions = (prefix.length + suffix_positions)[None, :]
        cross_position_embeddings = rope_cos_sin(
            self.cross_inv_freq,
            cross_positions,
            dtype,
        )
        self_position_embeddings = rope_cos_sin(
            self.self_inv_freq,
            suffix_positions[None, :],
            dtype,
        )
        self_attention_mask = self._self_attention_mask(batch, dtype, device)
        cross_attention_mask = self._cross_attention_mask(prefix, dtype, device)

        for layer, stream_idx in zip(
            self.layers,
            config.cross_attention_schedule,
            strict=True,
        ):
            hidden_states = layer(
                hidden_states,
                streams[stream_idx],
                cross_position_embeddings,
                cross_attention_mask,
                self_position_embeddings,
                self_attention_mask,
            )

        hidden_states = self.norm(hidden_states)
        return self.action_out_proj(hidden_states[:, 1:, :])
