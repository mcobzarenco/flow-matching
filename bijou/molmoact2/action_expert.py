"""MolmoAct2 action expert — faithful port of their flow-matching head.

Source: ``molmoact2/experiments/olmo/nn/action_expert.py`` (982 LOC) at
the pinned clone; math preserved exactly, trainer-machinery
(FSDP/compile/activation-checkpoint wrappers, flash-attn dispatch,
non-finite guard raises) dropped — attention always goes through
``F.scaled_dot_product_attention``, which is their own fallback path
(their flash path is a kernel choice, not different math; the G1
parity gate measures the bf16 gap).

Architecture (577.6M at the released SO-100/101 size): a DiT-style
stack of ``num_layers`` blocks over the noisy action chunk
``(batch, horizon, action_dim)``; each block = AdaLN-modulated
self-attention (RoPE, QK-RMSNorm) + cross-attention into per-layer
backbone KV + SwiGLU MLP, all gated by a 9-way modulation of the
timestep embedding. Backbone conditioning enters as one
``(k, v)`` pair per block: the raw Molmo2 KV states are projected by
the SHARED ``context_k_proj``/``context_v_proj`` (bias-free), RMS-
normed, reshaped to expert heads, and (optionally) concatenated with
the encoded proprioceptive state.

Checkpoint compatibility: HF exports store the expert under
``model.action_expert.*`` with EXACTLY this module's names.
``state_encoder`` (identity), ``state_norm`` (weightless) and the
frozen, inactive per-block ``cross_attn.kv_proj`` are absent from
checkpoints — ``load_action_expert_state`` injects the same
compatibility tensors their loader does.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import cast, override

import torch
import torch.nn.functional as F
from torch import nn


def _modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _round_up_multiple(value: int, multiple_of: int) -> int:
    if multiple_of <= 0:
        return value
    return int(math.ceil(value / multiple_of) * multiple_of)


def _init_linear(linear: nn.Linear, *, zero: bool = False, scale: float = 1.0) -> None:
    if zero:
        nn.init.zeros_(linear.weight)
    else:
        nn.init.xavier_uniform_(linear.weight)
        if scale != 1.0:
            with torch.no_grad():
                linear.weight.mul_(scale)
    # cast past the stubs: torch types Linear.bias as Parameter, but
    # bias=False modules carry None at runtime.
    bias = cast("nn.Parameter | None", linear.bias)
    if bias is not None:
        nn.init.zeros_(bias)


class ActionExpertRMSNorm(nn.Module):
    """Their RMSNorm: fp32 variance under a disabled-autocast block,
    optional elementwise affine (all live instances are weightless)."""

    def __init__(
        self,
        size: int,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ) -> None:
        super().__init__()
        self.size = size
        self.eps = eps
        self.weight: nn.Parameter | None
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(size))
        else:
            self.register_parameter("weight", None)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            dtype = x.dtype
            x_float = x.to(torch.float32)
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            out = x_float * torch.rsqrt(variance + self.eps)
            out = out.to(dtype)
        if self.weight is not None:
            out = out * self.weight
        return out


class ActionExpertRotaryEmbedding(nn.Module):
    """Half-split RoPE (cat convention, not interleaved), recomputed
    per forward in fp32 and cast to the activation dtype."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        self.head_dim = head_dim
        self.base = base

    @override
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, half_dim, device=q.device, dtype=torch.float32)
                / max(half_dim, 1)
            )
        )
        positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(dtype=q.dtype).view(1, 1, seq_len, half_dim)
        sin = freqs.sin().to(dtype=q.dtype).view(1, 1, seq_len, half_dim)

        def _apply(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _apply(q), _apply(k)


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
) -> torch.Tensor:
    """(b, s, h, d) in/out — their SDPA fallback path verbatim."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )
    return out.transpose(1, 2).contiguous()


class ActionExpertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.q_norm = (
            ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        )
        self.k_norm = (
            ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        )
        self.rope = ActionExpertRotaryEmbedding(self.head_dim) if use_rope else None
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    @override
    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # QK norm + RoPE run in (b, h, s, d) layout, matching source.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        out = _sdpa(
            q,
            k,
            v.contiguous(),
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.reshape(bsz, seq_len, self.hidden_size)
        return self.out_drop(self.out_proj(out))


class ActionExpertCrossAttention(nn.Module):
    """Query from the action stream; K/V arrive pre-projected per
    block (the only supported conditioning path — ``kv_proj`` exists
    for checkpoint-shape compatibility and is frozen/inactive)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.q_norm = (
            ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        )
        self.k_norm = (
            ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        )
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.kv_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    @override
    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_k: torch.Tensor,
        kv_v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, tgt_len, _ = x.shape
        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim)
        k, v = kv_k, kv_v
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        out = _sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.reshape(bsz, tgt_len, self.hidden_size)
        return self.out_drop(self.out_proj(out))


class ActionExpertMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        mlp_ratio: float,
        multiple_of: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = _round_up_multiple(int(hidden_size * mlp_ratio), multiple_of)
        self.up_proj = nn.Linear(hidden_size, inner_dim)
        self.gate_proj = nn.Linear(hidden_size, inner_dim)
        self.down_proj = nn.Linear(inner_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        return self.dropout(self.down_proj(x))


class ActionExpertModulation(nn.Module):
    def __init__(self, hidden_size: int, num_chunks: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, num_chunks * hidden_size)

    @override
    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(conditioning))


class ActionExpertBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        ffn_multiple_of: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        rope: bool = True,
    ) -> None:
        super().__init__()
        self.self_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.cross_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.ff_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.self_attn = ActionExpertSelfAttention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            use_rope=rope,
        )
        self.cross_attn = ActionExpertCrossAttention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
        )
        self.mlp = ActionExpertMLP(
            hidden_size,
            mlp_ratio=mlp_ratio,
            multiple_of=ffn_multiple_of,
            dropout=dropout,
        )
        self.modulation = ActionExpertModulation(hidden_size, 9)

    @override
    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        cross_kv: tuple[torch.Tensor, torch.Tensor],
        self_attn_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.modulation(conditioning).chunk(9, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            _modulate(self.self_norm(x), shift_msa, scale_msa),
            attn_mask=self_attn_mask,
            is_causal=is_causal,
        )
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            _modulate(self.cross_norm(x), shift_mca, scale_mca),
            kv_k=cross_kv[0],
            kv_v=cross_kv[1],
            attn_mask=attn_mask,
        )
        return x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.ff_norm(x), shift_mlp, scale_mlp),
        )


class ActionExpertFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        self.norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.modulation = ActionExpertModulation(hidden_size, 2)
        self.linear = nn.Linear(hidden_size, output_dim)

    @override
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @override
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() > 1:
            timesteps = timesteps.view(timesteps.shape[0], -1)[:, 0]
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
            * (-math.log(10000.0) / max(half_dim - 1, 1)),
        )
        args = timesteps[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


@dataclass
class ActionExpertConfig:
    """Defaults = the released MolmoAct2 SO-100/101 expert, measured
    off the HF export + config.json (577,564,448 params): h768, 36
    blocks, 8 heads (head_dim 96), MLP inner 3072, conditioned on the
    Molmo2 4B KV dim 1024."""

    max_horizon: int = 30
    max_action_dim: int = 32
    hidden_size: int = 768
    num_layers: int = 36
    num_heads: int = 8
    mlp_ratio: float = 4.0
    ffn_multiple_of: int = 256
    timestep_embed_dim: int = 256
    dropout: float = 0.0
    attn_dropout: float = 0.0
    context_layer_norm: bool = True
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    rope: bool = True
    causal_attn: bool = False

    def build(self, llm_kv_dim: int) -> ActionExpert:
        return ActionExpert(self, llm_kv_dim=llm_kv_dim)


class ActionExpert(nn.Module):
    """The v-field: ``forward(noisy_actions, t, per-layer backbone KV,
    state) -> velocity`` over the action chunk."""

    def __init__(self, config: ActionExpertConfig, *, llm_kv_dim: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.llm_kv_dim = llm_kv_dim
        self.action_head_dim = config.hidden_size // config.num_heads

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.action_embed = nn.Linear(config.max_action_dim, config.hidden_size)
        self.state_encoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.state_norm = ActionExpertRMSNorm(config.hidden_size, eps=1e-6)
        self.context_k_proj = nn.Linear(self.llm_kv_dim, config.hidden_size, bias=False)
        self.context_v_proj = nn.Linear(self.llm_kv_dim, config.hidden_size, bias=False)
        self.context_norm = (
            ActionExpertRMSNorm(config.hidden_size, eps=1e-6)
            if config.context_layer_norm
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(
                    config.hidden_size,
                    config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    ffn_multiple_of=config.ffn_multiple_of,
                    attn_dropout=config.attn_dropout,
                    dropout=config.dropout,
                    qk_norm=config.qk_norm,
                    qk_norm_eps=config.qk_norm_eps,
                    rope=config.rope,
                )
                for _ in range(config.num_layers)
            ],
        )
        # kv_proj is checkpoint-shape compatibility only (see class doc).
        for block in self.iter_blocks():
            block.cross_attn.kv_proj.weight.requires_grad = False
            kv_bias = cast("nn.Parameter | None", block.cross_attn.kv_proj.bias)
            if kv_bias is not None:
                kv_bias.requires_grad = False
        self.final_layer = ActionExpertFinalLayer(
            config.hidden_size,
            config.max_action_dim,
        )
        self.reset_parameters()

    def iter_blocks(self) -> Iterator[ActionExpertBlock]:
        for block in self.blocks:
            assert isinstance(block, ActionExpertBlock)
            yield block

    def _reshape_hidden_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(
            x.shape[0],
            x.shape[1],
            self.config.num_heads,
            self.action_head_dim,
        )

    def reset_parameters(self) -> None:
        for module in self.time_embed.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)
        _init_linear(self.action_embed)
        _init_linear(self.state_encoder)
        self.context_k_proj.reset_parameters()
        self.context_v_proj.reset_parameters()
        residual_scale = (2 * max(self.config.num_layers, 1)) ** -0.5
        for block in self.iter_blocks():
            _init_linear(block.self_attn.qkv)
            _init_linear(block.self_attn.out_proj, scale=residual_scale)
            _init_linear(block.cross_attn.q_proj)
            _init_linear(block.cross_attn.kv_proj)
            _init_linear(block.cross_attn.out_proj, scale=residual_scale)
            _init_linear(block.mlp.up_proj)
            _init_linear(block.mlp.gate_proj)
            _init_linear(block.mlp.down_proj, scale=residual_scale)
            _init_linear(block.modulation.linear, zero=True)
        _init_linear(self.final_layer.modulation.linear, zero=True)
        _init_linear(self.final_layer.linear, zero=True)

    def _time_conditioning(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Their HF inference semantics: the sinusoid runs at the
        timestep dtype (the flow loop feeds fp32 grids), then the
        embedding is cast to the MLP weight dtype — a no-op at uniform
        dtype, load-bearing for fp32 timesteps on a bf16 expert."""
        sinusoid, first_linear = self.time_embed[0], self.time_embed[1]
        conditioning = sinusoid(timesteps)
        assert isinstance(first_linear, nn.Linear)
        conditioning = conditioning.to(dtype=first_linear.weight.dtype)
        for module in list(self.time_embed.children())[1:]:
            conditioning = module(conditioning)
        return conditioning

    def _encode_states(self, states: torch.Tensor | None) -> torch.Tensor | None:
        if states is None:
            return None
        if states.dim() == 2:
            states = states.unsqueeze(1)
        if states.shape[-1] != self.hidden_size:
            feat_dim = states.shape[-1]
            if feat_dim < self.hidden_size:
                states = F.pad(states, (0, self.hidden_size - feat_dim))
            else:
                states = states[..., : self.hidden_size]
        return self.state_norm(self.state_encoder(states))

    def _prepare_kv_context(
        self,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
        encoded_states: torch.Tensor | None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if len(encoder_kv_states) != len(self.blocks):
            raise ValueError(
                "expected one KV state per action expert block "
                f"(got {len(encoder_kv_states)}, expected {len(self.blocks)})",
            )
        state_heads = (
            self._reshape_hidden_to_heads(encoded_states)
            if encoded_states is not None
            else None
        )
        kv_contexts = []
        for k_in, v_in in encoder_kv_states:
            k_ctx = self._reshape_hidden_to_heads(
                self.context_norm(self.context_k_proj(k_in)),
            )
            v_ctx = self._reshape_hidden_to_heads(
                self.context_norm(self.context_v_proj(v_in)),
            )
            if state_heads is not None:
                k_ctx = torch.cat([k_ctx, state_heads], dim=1)
                v_ctx = torch.cat([v_ctx, state_heads], dim=1)
            kv_contexts.append((k_ctx, v_ctx))
        return kv_contexts

    def _build_cross_attention_mask(
        self,
        encoder_attention_mask: torch.Tensor | None,
        encoded_states: torch.Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        # Source quirk preserved: a state-only context with no encoder
        # mask returns None (states are never masked anyway).
        if encoder_attention_mask is None:
            return None
        state_seq_len = 0 if encoded_states is None else encoded_states.shape[1]
        if encoder_attention_mask.dim() == 2:
            mask = encoder_attention_mask[:, None, None, :].to(dtype=dtype)
        else:
            mask = encoder_attention_mask.to(dtype=dtype)
        if state_seq_len > 0:
            ones = torch.ones(
                batch_size,
                1,
                1,
                state_seq_len,
                device=mask.device,
                dtype=mask.dtype,
            )
            mask = torch.cat([mask, ones], dim=-1)
        return (1.0 - mask) * torch.finfo(dtype).min

    def _build_self_attention_mask(
        self,
        action_attention_mask: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        mask = None
        if action_attention_mask is not None:
            valid = action_attention_mask.to(device=device, dtype=torch.bool)
            if valid.ndim != 2 or valid.shape[1] != seq_len:
                raise ValueError(
                    f"expected action_attention_mask shape (batch, {seq_len}), "
                    f"got {tuple(valid.shape)}",
                )
            mask = (~valid)[:, None, None, :].to(dtype=dtype) * torch.finfo(dtype).min
        if self.config.causal_attn:
            causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(
                diagonal=1,
            )
            causal = (
                causal.unsqueeze(0).unsqueeze(0).to(dtype=dtype)
                * torch.finfo(dtype).min
            )
            mask = causal if mask is None else (mask + causal)
        return mask

    @override
    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
        encoder_attention_mask: torch.Tensor | None = None,
        action_attention_mask: torch.Tensor | None = None,
        state_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(encoder_kv_states) == 0:
            raise ValueError("expected at least one encoder KV state")
        bsz, seq_len, _ = actions.shape
        if seq_len > self.config.max_horizon:
            raise ValueError(
                f"action sequence length {seq_len} exceeds max_horizon={self.config.max_horizon}",
            )

        conditioning = self._time_conditioning(timesteps)
        encoded_states = self._encode_states(state_embeddings)
        x = self.action_embed(actions)
        valid_action = None
        if action_attention_mask is not None:
            valid_action = action_attention_mask.to(
                device=x.device,
                dtype=x.dtype,
            ).unsqueeze(-1)
            x = x * valid_action

        kv_contexts = self._prepare_kv_context(encoder_kv_states, encoded_states)
        cross_mask = self._build_cross_attention_mask(
            encoder_attention_mask,
            encoded_states,
            bsz,
            x.dtype,
        )
        self_mask = self._build_self_attention_mask(
            action_attention_mask,
            seq_len,
            x.device,
            x.dtype,
        )

        for block, kv_context in zip(self.iter_blocks(), kv_contexts, strict=True):
            x = block(
                x,
                conditioning,
                cross_kv=kv_context,
                self_attn_mask=self_mask,
                attn_mask=cross_mask,
                is_causal=self.config.causal_attn,
            )
            if valid_action is not None:
                x = x * valid_action

        out = self.final_layer(x, conditioning)
        if valid_action is not None:
            out = out * valid_action
        return out


def load_action_expert_state(
    expert: ActionExpert,
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str = "model.action_expert.",
) -> None:
    """Load an HF-export state dict (``model.action_expert.*`` keys),
    injecting the compatibility tensors their loader adds: identity
    ``state_encoder`` and zero per-block ``cross_attn.kv_proj`` (both
    absent from checkpoints; kv_proj is frozen/inactive)."""
    stripped = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    if not stripped:
        raise ValueError(f"no keys under prefix {prefix!r}")
    sample = next(iter(stripped.values()))
    hidden = expert.config.hidden_size
    if "state_encoder.weight" not in stripped:
        stripped["state_encoder.weight"] = torch.eye(hidden, dtype=sample.dtype)
        stripped["state_encoder.bias"] = torch.zeros(hidden, dtype=sample.dtype)
    for idx in range(len(expert.blocks)):
        kv_w = f"blocks.{idx}.cross_attn.kv_proj.weight"
        if kv_w not in stripped:
            stripped[kv_w] = torch.zeros(hidden * 2, hidden, dtype=sample.dtype)
            stripped[f"blocks.{idx}.cross_attn.kv_proj.bias"] = torch.zeros(
                hidden * 2,
                dtype=sample.dtype,
            )
    expert.load_state_dict(stripped, strict=True)
