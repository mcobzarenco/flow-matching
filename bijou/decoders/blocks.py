"""Shared decoder building blocks: the Gemma-style sandwich layer.

Both action decoders (flow matching, AR FAST) are stacks of the same
block — cross-attention over one ObservationMemory stream, self-attention
over the suffix, gated-GLU MLP, sandwich RMSNorms — differing only in the
suffix content, masks, and heads. ExpertConfig carries the shared
geometry; τ-conditioning (adaRMS) is built only when requested and is a
None no-op otherwise.

Attribute names inside these modules are frozen: they are safetensors key
segments of every existing flow checkpoint (tests/test_state_dict_keys.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import override

import torch
from torch import Tensor, nn

from ..interface import MemoryStream, ObservationMemory
from ..nn import (
    DEFAULT_ATTENTION_BACKEND,
    AttentionBackend,
    DeviceLike,
    MaskSpec,
    RMSNorm,
    RopeParameters,
    activation_fn,
    apply_rotary_pos_emb,
    attention,
)


class SelfAttentionMode(StrEnum):
    """Masking of the expert's self-attention over ``[state][actions]``.

    The state token attends and is attended by everything in both modes;
    in CAUSAL_ACTIONS each action token only attends earlier actions (the
    SmolVLA ablation found this beats bidirectional; π0 uses bidirectional).
    """

    BIDIRECTIONAL = "bidirectional"
    CAUSAL_ACTIONS = "causal_actions"


class TimeConditioning(StrEnum):
    """How flow time τ conditions the expert.

    ADDITIVE: the π0/SmolVLA scheme currently shipped — τ's MLP embedding
    is added to the action token embeddings at the input; the state token
    is not time-conditioned; layers are unconditioned.
    ADARMS: DiT-style adaptive RMSNorm — τ's embedding drives a per-layer
    zero-initialized head producing a per-channel SCALE on each sublayer's
    (RMS)norm output and a GATE on its residual contribution (no shift —
    RMSNorm carries no bias; the gate is the additive-injection route).
    Identity at init. The intended successor to ADDITIVE.
    """

    ADDITIVE = "additive"
    ADARMS = "adarms"


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
    time_conditioning: TimeConditioning

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
        stream: MemoryStream,
        position_embeddings: tuple[Tensor, Tensor],
        mask: MaskSpec,
    ) -> Tensor:
        """Cross-attend one exported memory stream; returns [B, suffix, hidden].

        Shapes:
          - hidden_states: [B, suffix, hidden]  (queries: the expert suffix)
          - stream.key/value: [B, kv_heads, P, head_dim]
          - position_embeddings: (cos, sin), each [B, suffix, head_dim]
            (padded batches: per-sample positions; [1, suffix, head_dim]
            broadcast otherwise)
          - mask.tensor (when present): [B, 1, 1, P]  (padding-only — every
            query sees the same real memory columns, broadcast over queries)
        """
        batch, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        query = self.q_proj(hidden_states).view(batch, seq_len, -1, self.head_dim)
        query = self.q_norm(query)
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2)
        query = query.transpose(1, 2)

        # Keys arrive already normed + position-encoded by the producer.
        attn_output = attention(
            self.attn_backend,
            query,
            stream.key,
            stream.value,
            mask,
            num_key_value_groups=self.num_heads // stream.key.shape[1],
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
        """Self-attend the suffix; returns [B, suffix, hidden].

        Shapes:
          - hidden_states: [B, suffix, hidden]
          - position_embeddings: (cos, sin), each [1, suffix, head_dim]
            (suffix positions are sample-independent — broadcast over B)
          - mask.tensor (when present): [B, 1, suffix, suffix]
        """
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
        """Gated GLU MLP; shape-preserving.

        Shapes:
          - x: [B, suffix, hidden]  (returns [B, suffix, hidden])
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# adaRMS application, factored so the additive path is a clean ``None``
# no-op (delete these guards, and the two ``None`` cases below, to drop
# additive support once adaRMS wins). Modulation vectors are per-sample
# [B, hidden], broadcast across the suffix positions.
def apply_scale(hidden_states: Tensor, scale: Tensor | None) -> Tensor:
    """RMSNorm-output scale: ``norm(x)*(1+γ)``; identity when scale is None.
    hidden_states [B, suffix, hidden]; scale [B, hidden]."""
    if scale is None:
        return hidden_states
    return hidden_states * (1.0 + scale[:, None, :])


def apply_gate(hidden_states: Tensor, gate: Tensor | None) -> Tensor:
    """Sublayer-contribution gate: ``g*out``; identity (pass-through, gate
    conceptually 1) when gate is None. hidden_states [B, suffix, hidden];
    gate [B, hidden]."""
    if gate is None:
        return hidden_states
    return hidden_states * gate[:, None, :]


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
        # adaRMS only: SiLU -> Linear(hidden, 6*hidden), zero-initialized
        # (ActionExpert.reset_parameters), producing (scale, gate) for each
        # of cross-attn / self-attn / MLP. None => additive (unconditioned).
        self.modulation: nn.Sequential | None = None
        if config.time_conditioning is TimeConditioning.ADARMS:
            self.modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    config.hidden_size,
                    6 * config.hidden_size,
                    device=device,
                    dtype=dtype,
                ),
            )
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
        stream: MemoryStream,
        cross_position_embeddings: tuple[Tensor, Tensor],
        cross_attention_mask: MaskSpec,
        self_position_embeddings: tuple[Tensor, Tensor],
        self_attention_mask: MaskSpec,
        condition: Tensor | None = None,
    ) -> Tensor:
        """cross-attn -> self-attn -> MLP; returns [B, suffix, hidden].

        ``condition`` is the τ embedding in adaRMS mode (drives this layer's
        scale/gate head); None in additive mode, where every modulation slot
        is None and each block is byte-identical to the pre-adaRMS path
        (``apply_scale``/``apply_gate`` are then no-ops).

        Shapes:
          - hidden_states: [B, suffix, hidden]
          - stream.key/value: [B, kv_heads, P, head_dim]
          - cross_position_embeddings: (cos, sin), each
            [B or 1, suffix, head_dim]  (B when padded, broadcast otherwise)
          - cross_attention_mask.tensor (when present): [B, 1, 1, P]
          - self_position_embeddings: (cos, sin), each [1, suffix, head_dim]
          - self_attention_mask.tensor (when present): [B, 1, suffix, suffix]
          - condition (adaRMS only): [B, hidden]
        """
        if self.modulation is None:
            cross_scale = cross_gate = self_scale = None
            self_gate = mlp_scale = mlp_gate = None
        else:
            assert condition is not None, "adaRMS layer requires a condition"
            (
                cross_scale,
                cross_gate,
                self_scale,
                self_gate,
                mlp_scale,
                mlp_gate,
            ) = self.modulation(condition).chunk(6, dim=-1)

        residual = hidden_states
        hidden_states = apply_scale(
            self.pre_cross_attention_layernorm(hidden_states),
            cross_scale,
        )
        hidden_states = self.cross_attn(
            hidden_states,
            stream,
            cross_position_embeddings,
            cross_attention_mask,
        )
        hidden_states = apply_gate(
            self.post_cross_attention_layernorm(hidden_states),
            cross_gate,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = apply_scale(
            self.pre_self_attention_layernorm(hidden_states),
            self_scale,
        )
        hidden_states = self.self_attn(
            hidden_states,
            self_position_embeddings,
            self_attention_mask,
        )
        hidden_states = apply_gate(
            self.post_self_attention_layernorm(hidden_states),
            self_gate,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = apply_scale(
            self.pre_feedforward_layernorm(hidden_states),
            mlp_scale,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = apply_gate(
            self.post_feedforward_layernorm(hidden_states),
            mlp_gate,
        )
        return residual + hidden_states


def cross_attention_mask(
    memory: ObservationMemory,
    dtype: torch.dtype,
    device: torch.device,
) -> MaskSpec:
    """Padding-only cross-attention mask: [B, 1, 1, P] (every suffix query
    sees the same real memory columns), or an empty spec when the memory
    is unpadded."""
    if memory.padding_mask is None:
        return MaskSpec()
    real = memory.padding_mask.to(device=device, dtype=torch.bool)
    min_value = torch.finfo(dtype).min
    tensor = torch.where(
        real[:, None, None, :],
        torch.tensor(0.0, device=device, dtype=dtype),
        min_value,
    )
    return MaskSpec(tensor=tensor)
