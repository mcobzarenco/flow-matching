"""MolmoAct2's action expert as a first-class bijou decoder (§8.13 step 3).

FLOW CONVENTION (ascending t — the repo standard for new flow code):
``x_t = (1−t)·ε + t·x`` with ε ~ N(0, I), so t=0 is pure noise and t=1
is data; the velocity target is ``u = x − ε``; sampling integrates
0 → 1 (left-endpoint Euler, their serving default; Heun optional).
This is the MIRROR of ``decoders/flow.py``'s π0 convention (τ = 1−t,
target ε − x, integrate 1 → 0) — the two modules deliberately share no
code, so a sign can never be "harmonized" across them by accident. The
training t-law is theirs verbatim: ``t = offset + scale·Beta(α, β)``
(released params 0.001 + 0.999·Beta(1, 1.5)) — mass at the NOISE end,
and unlike flow.py's law it includes pure data (t=1) and excludes pure
noise; the direction test pins both facts.

ARCHITECTURE (their ``nn/action_expert.py``, owned here per §8.13
decision 3 — ``bijou/molmoact2/action_expert.py`` is the frozen parity
reference and the byte-parity oracle pins the two while both exist):
a DiT-style stack over the noisy action chunk; each block =
adaLN-Zero-modulated self-attention (half-split RoPE, QK-RMSNorm) +
cross-attention into per-layer trunk KV + SwiGLU MLP, 9-way modulation
per block, 2-way on the final layer, zero-initialized so the initial
velocity field is exactly 0. Conditioning is trunk-neutral: one
post-RoPE ``(K, V)`` pair per block, flattened ``[B, S, llm_kv_dim]``,
projected by ONE shared bias-free ``context_k_proj``/``context_v_proj``
— the whole prompt cache as conditioning surface, no residual taps, no
per-stream adapters. "Molmo" names the architecture's provenance, not
a trunk requirement.

TENSOR-NAME CONTRACT: attribute names are the converter's
``expert.safetensors`` keys (= their HF export minus the
``model.action_expert.`` prefix), pinned by
``tests/test_convert_molmoact2.py`` — ``load_expert_state`` injects the
compat tensors exports omit (identity ``state_encoder``, zero per-block
``cross_attn.kv_proj``) exactly like their loader. ``kv_proj`` and
``state_encoder`` are frozen at construction: the shipped conditioning
path projects K/V through the SHARED context projections, and state is
discrete-in-prompt (their ``freeze_continuous_state_conditioning``);
un-freezing ``state_encoder`` is the deliberate future
continuous-recondition knob, not a default.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..interface import (
    BijouPrediction,
    CollatedBatch,
    ObservationMemory,
    SamplingMethod,
)
from ..molmo2.cache import Molmo2KVCache


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN modulation.

    Shapes:
    - ``x``: [B, T, hidden] normalized stream
    - ``shift``: [B, hidden] additive term (broadcast over T)
    - ``scale``: [B, hidden] multiplicative term (broadcast over T)
    - returns: [B, T, hidden]
    """
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


class MolmoFlowRMSNorm(nn.Module):
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
    def forward(self, x: Tensor) -> Tensor:
        """Shapes:
        - ``x``: [..., size] any leading dims
        - returns: [..., size] in ``x``'s dtype (variance in fp32)
        """
        with torch.autocast(enabled=False, device_type=x.device.type):
            dtype = x.dtype
            x_float = x.to(torch.float32)
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            out = x_float * torch.rsqrt(variance + self.eps)
            out = out.to(dtype)
        if self.weight is not None:
            out = out * self.weight
        return out


class MolmoFlowRotaryEmbedding(nn.Module):
    """Half-split RoPE (cat convention, not interleaved), recomputed
    per forward in fp32 and cast to the activation dtype."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        self.head_dim = head_dim
        self.base = base

    @override
    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Shapes:
        - ``q``: [B, heads, T, head_dim]
        - ``k``: [B, heads, T, head_dim] (same T as q — self-attention only)
        - returns: (q, k) rotated, same shapes/dtypes
        """
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

        def _apply(x: Tensor) -> Tensor:
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _apply(q), _apply(k)


def _sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
) -> Tensor:
    """Their SDPA fallback path verbatim.

    Shapes:
    - ``q``: [B, T_q, heads, head_dim]
    - ``k``: [B, T_kv, heads, head_dim]
    - ``v``: [B, T_kv, heads, head_dim]
    - ``attn_mask``: [B, 1, T_q, T_kv] additive (or None)
    - returns: [B, T_q, heads, head_dim] contiguous
    """
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )
    return out.transpose(1, 2).contiguous()


class MolmoFlowSelfAttention(nn.Module):
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
            MolmoFlowRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        )
        self.k_norm = (
            MolmoFlowRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        )
        self.rope = MolmoFlowRotaryEmbedding(self.head_dim) if use_rope else None
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    @override
    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Shapes:
        - ``x``: [B, T, hidden] action stream
        - ``attn_mask``: [B, 1, T, T] additive (or None)
        - returns: [B, T, hidden]
        """
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


class MolmoFlowCrossAttention(nn.Module):
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
            MolmoFlowRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        )
        self.k_norm = (
            MolmoFlowRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        )
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.kv_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    @override
    def forward(
        self,
        x: Tensor,
        *,
        kv_k: Tensor,
        kv_v: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Shapes:
        - ``x``: [B, T, hidden] action stream (queries)
        - ``kv_k``: [B, S_ctx, heads, head_dim] pre-projected context keys
        - ``kv_v``: [B, S_ctx, heads, head_dim] pre-projected context values
        - ``attn_mask``: [B, 1, 1, S_ctx] additive (or None)
        - returns: [B, T, hidden]
        """
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


class MolmoFlowMLP(nn.Module):
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
    def forward(self, x: Tensor) -> Tensor:
        """Shapes:
        - ``x``: [B, T, hidden]
        - returns: [B, T, hidden] (SwiGLU through inner_dim)
        """
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        return self.dropout(self.down_proj(x))


class MolmoFlowModulation(nn.Module):
    def __init__(self, hidden_size: int, num_chunks: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, num_chunks * hidden_size)

    @override
    def forward(self, conditioning: Tensor) -> Tensor:
        """Shapes:
        - ``conditioning``: [B, hidden] timestep embedding
        - returns: [B, num_chunks * hidden] stacked shift/scale/gate rows
        """
        return self.linear(self.act(conditioning))


class MolmoFlowBlock(nn.Module):
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
        self.self_norm = MolmoFlowRMSNorm(hidden_size, eps=1e-6)
        self.cross_norm = MolmoFlowRMSNorm(hidden_size, eps=1e-6)
        self.ff_norm = MolmoFlowRMSNorm(hidden_size, eps=1e-6)
        self.self_attn = MolmoFlowSelfAttention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            use_rope=rope,
        )
        self.cross_attn = MolmoFlowCrossAttention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
        )
        self.mlp = MolmoFlowMLP(
            hidden_size,
            mlp_ratio=mlp_ratio,
            multiple_of=ffn_multiple_of,
            dropout=dropout,
        )
        self.modulation = MolmoFlowModulation(hidden_size, 9)

    @override
    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
        *,
        cross_kv: tuple[Tensor, Tensor],
        self_attn_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Shapes:
        - ``x``: [B, T, hidden] action stream
        - ``conditioning``: [B, hidden] timestep embedding
        - ``cross_kv``: ([B, S_ctx, heads, head_dim], same) context K/V
        - ``self_attn_mask``: [B, 1, T, T] additive (or None)
        - ``attn_mask``: [B, 1, 1, S_ctx] additive cross mask (or None)
        - returns: [B, T, hidden]
        """
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


class MolmoFlowFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        self.norm = MolmoFlowRMSNorm(hidden_size, eps=1e-6)
        self.modulation = MolmoFlowModulation(hidden_size, 2)
        self.linear = nn.Linear(hidden_size, output_dim)

    @override
    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        """Shapes:
        - ``x``: [B, T, hidden]
        - ``conditioning``: [B, hidden] timestep embedding
        - returns: [B, T, output_dim] velocity rows
        """
        shift, scale = self.modulation(conditioning).chunk(2, dim=1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class MolmoFlowTimeEmbedding(nn.Module):
    """Classic 10k sinusoid (their DiT inheritance — NOT flow.py's
    π0 geometric-period embedding; the conventions never mix)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @override
    def forward(self, timesteps: Tensor) -> Tensor:
        """Shapes:
        - ``timesteps``: [B] (multi-dim inputs collapse to their first
          column, their convention)
        - returns: [B, dim] sin|cos features at ``timesteps``' dtype
        """
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


@dataclass(frozen=True, slots=True)
class TimeLaw:
    """The training t-distribution, ``t = offset + scale·Beta(α, β)``
    (their ``_sample_beta_timesteps``): recorded per checkpoint by the
    converter, never assumed. Released params: 0.001 + 0.999·Beta(1, 1.5)
    — support [0.001, 1.0]: pure data in, pure noise out."""

    offset: float
    scale: float
    beta_alpha: float
    beta_beta: float

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Shapes:
        - returns: [batch_size] fp32 on ``device``

        ``generator`` seeds the draw explicitly (styleguide); None
        matches the reference trainer's ambient-RNG behavior. Beta is
        sampled as a Gamma ratio so a generator CAN be threaded
        (torch.distributions takes none): G(α)/(G(α)+G(β)), the
        standard construction — same law, not the same byte stream as
        torch.distributions.Beta.
        """
        if generator is None:
            b = (
                torch.distributions.Beta(self.beta_alpha, self.beta_beta)
                .sample(torch.Size((batch_size,)))
                .to(device=device, dtype=torch.float32)
            )
        else:
            gamma_a = torch._standard_gamma(  # pyright: ignore[reportPrivateImportUsage] — no public generator-threaded gamma; stable ATen op
                torch.full(
                    (batch_size,),
                    self.beta_alpha,
                    device=generator.device,
                    dtype=torch.float32,
                ),
                generator=generator,
            )
            gamma_b = torch._standard_gamma(  # pyright: ignore[reportPrivateImportUsage] — same stub gap as above
                torch.full(
                    (batch_size,),
                    self.beta_beta,
                    device=generator.device,
                    dtype=torch.float32,
                ),
                generator=generator,
            )
            b = (gamma_a / (gamma_a + gamma_b)).to(device=device)
        return self.offset + self.scale * b


@dataclass(frozen=True, slots=True)
class MolmoFlowRuntime:
    """Deployment facts a built decoder needs beyond its architecture —
    set by the loader (``MolmoFlowDecoder.configure``) from the
    checkpoint's decoder section, never guessed: the REAL action
    geometry of the tag, their serving flow-step count, and the
    recorded training t-law."""

    action_dim: int
    action_horizon: int
    n_action_steps: int
    num_flow_steps: int
    mask_action_dim_padding: bool
    time_law: TimeLaw


@dataclass(frozen=True, slots=True)
class MolmoFlowConfig:
    """Architecture of the molmo_flow expert. No field defaults — the
    checkpoint bridge (bijou.loading) and tests spell out every field.
    ``llm_kv_dim`` is the trunk's flattened KV width (kv_heads *
    head_dim; 1024 for Molmo2-4B) — the ONE trunk-derived number."""

    max_horizon: int
    max_action_dim: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    ffn_multiple_of: int
    timestep_embed_dim: int
    dropout: float
    attn_dropout: float
    context_layer_norm: bool
    qk_norm: bool
    qk_norm_eps: float
    rope: bool
    causal_attn: bool
    llm_kv_dim: int

    @staticmethod
    def released_so100_101() -> MolmoFlowConfig:
        """The released ``allenai/MolmoAct2-SO100_101`` expert on its
        Molmo2-4B trunk (kv_dim 1024). The literals' home is the port's
        ``ActionExpertConfig.released_so100_101``; the sync test pins
        this mirror field-for-field until the port retires and the
        literals move here."""
        return MolmoFlowConfig(
            max_horizon=30,
            max_action_dim=32,
            hidden_size=768,
            num_layers=36,
            num_heads=8,
            mlp_ratio=4.0,
            ffn_multiple_of=256,
            timestep_embed_dim=256,
            dropout=0.0,
            attn_dropout=0.0,
            context_layer_norm=True,
            qk_norm=True,
            qk_norm_eps=1e-6,
            rope=True,
            causal_attn=False,
            llm_kv_dim=1024,
        )

    def build(self) -> MolmoFlowDecoder:
        return MolmoFlowDecoder(self)


class MolmoFlowDecoder(nn.Module):
    """The v-field: ``forward(noisy_actions, t, per-layer trunk KV,
    state) -> velocity`` over the action chunk. Attribute names are the
    converter's safetensors keys (module docstring)."""

    def __init__(self, config: MolmoFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.llm_kv_dim = config.llm_kv_dim
        self.action_head_dim = config.hidden_size // config.num_heads

        self.time_embed = nn.Sequential(
            MolmoFlowTimeEmbedding(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.action_embed = nn.Linear(config.max_action_dim, config.hidden_size)
        self.state_encoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.state_norm = MolmoFlowRMSNorm(config.hidden_size, eps=1e-6)
        self.context_k_proj = nn.Linear(
            config.llm_kv_dim,
            config.hidden_size,
            bias=False,
        )
        self.context_v_proj = nn.Linear(
            config.llm_kv_dim,
            config.hidden_size,
            bias=False,
        )
        self.context_norm = (
            MolmoFlowRMSNorm(config.hidden_size, eps=1e-6)
            if config.context_layer_norm
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                MolmoFlowBlock(
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
        # kv_proj is checkpoint-shape compatibility only; state_encoder
        # is the dormant continuous-state path (frozen under the shipped
        # discrete-state scope — unfreezing is a deliberate experiment
        # knob, §8.13 decision 10).
        for block in self.iter_blocks():
            block.cross_attn.kv_proj.weight.requires_grad = False
            kv_bias = cast("nn.Parameter | None", block.cross_attn.kv_proj.bias)
            if kv_bias is not None:
                kv_bias.requires_grad = False
        self.state_encoder.weight.requires_grad = False
        state_bias = cast("nn.Parameter | None", self.state_encoder.bias)
        if state_bias is not None:
            state_bias.requires_grad = False
        self.final_layer = MolmoFlowFinalLayer(
            config.hidden_size,
            config.max_action_dim,
        )
        # Deployment configuration (``configure``): the loader supplies
        # the checkpoint's action geometry + t-law, the q01/q99 clamp
        # table, and the write-side schema dict (loading owns the
        # schema; a decoder cannot import it, so the dict is stashed
        # here by the loader). The table is PLAIN fp32 CPU tensors —
        # deliberately NOT buffers: ``module.to(bfloat16)`` would sweep
        # buffers to bf16 and round the denorm constants (~3 significant
        # digits over quantile spans up to ~280 raw units — a measured
        # 0.027 pooled divergence vs the reference on the released arm,
        # step-5 gate diagnosis 2026-08-11). The reference unnormalizes
        # with fp32 JSON values; so do we, at every expert dtype. The
        # table's serialized home is the checkpoint metadata's
        # normalization row, stored ONCE.
        self.runtime: MolmoFlowRuntime | None = None
        self.checkpoint_schema: dict[str, object] | None = None
        self.action_q01 = torch.zeros(config.max_action_dim, dtype=torch.float32)
        self.action_q99 = torch.zeros(config.max_action_dim, dtype=torch.float32)
        self.reset_parameters()

    def configure(
        self,
        runtime: MolmoFlowRuntime,
        *,
        action_q01: Tensor,
        action_q99: Tensor,
        checkpoint_schema: dict[str, object],
    ) -> None:
        """Attach the deployment facts (loader-called; predict/loss refuse
        an unconfigured decoder loudly).

        Shapes:
        - ``action_q01``/``action_q99``: [action_dim] raw-unit quantiles
          (the checkpoint normalization row's, or the run's recomputed
          merge — padded here to max_action_dim with 0/1 inert rows)
        """
        if action_q01.shape != (runtime.action_dim,) or action_q99.shape != (
            runtime.action_dim,
        ):
            raise ValueError(
                f"expected [{runtime.action_dim}] quantile rows, got "
                f"{tuple(action_q01.shape)} / {tuple(action_q99.shape)}",
            )
        pad = self.config.max_action_dim - runtime.action_dim
        self.action_q01 = F.pad(
            action_q01.to(torch.float32).cpu(),
            (0, pad),
            value=0.0,
        )
        # Padded dims get a unit-width box (0..1) so the normalize/
        # unnormalize maps stay finite there; the dim mask keeps them
        # out of every loss and the sampler zeroes them anyway.
        self.action_q99 = F.pad(
            action_q99.to(torch.float32).cpu(),
            (0, pad),
            value=1.0,
        )
        self.runtime = runtime
        self.checkpoint_schema = checkpoint_schema

    def _configured(self) -> MolmoFlowRuntime:
        if self.runtime is None:
            raise ValueError(
                "MolmoFlowDecoder is unconfigured — the loader must call "
                "configure() with the checkpoint's action geometry and "
                "q01/q99 table before predict/loss",
            )
        return self.runtime

    def dim_pad_mask(self) -> Tensor:
        """[max_action_dim] bool CPU, True = padded dim (their
        ``action_dim_is_pad``), derived from the configured geometry;
        consumers broadcast it to their device."""
        runtime = self._configured()
        mask = torch.ones(self.config.max_action_dim, dtype=torch.bool)
        mask[: runtime.action_dim] = False
        return mask

    @torch.no_grad()
    def predict_chunk(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[Any],
        *,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        method: SamplingMethod = SamplingMethod.EULER,
    ) -> BijouPrediction:
        """RAW-unit chunk prediction — their serving tail on our seam:
        extract per-layer KV off the prefix cache, integrate (default =
        the checkpoint's recorded ``num_flow_steps``, Euler — their
        deployment operating point), slice the real action width, clamp
        + q01/q99-unnormalize, and the reference's dtype round-trip.
        ``batch`` stats are deliberately unused — normalization is
        decoder-owned (§8.13 decision 6).

        Shapes:
        - ``noise`` (when given): [B, action_horizon, max_action_dim]
        - returns: BijouPrediction actions [B, n_action_steps,
          action_dim] fp32 raw units; generations None
        """
        del batch  # stats intentionally unused; signature mirrors flow's
        runtime = self._configured()
        kv_states = layer_kv_pairs(memory, num_blocks=len(self.blocks))
        conditioning = conditioning_mask_of(memory)
        if noise is None:
            source = kv_states[0][0]
            noise = torch.randn(
                (source.shape[0], runtime.action_horizon, self.config.max_action_dim),
                device=source.device,
                dtype=self.action_embed.weight.dtype,
                generator=generator,
            )
        chunk = self.sample_actions(
            kv_states,
            encoder_attention_mask=conditioning,
            action_horizon=runtime.action_horizon,
            action_dim_is_pad=self.dim_pad_mask(),
            num_steps=runtime.num_flow_steps if num_steps is None else num_steps,
            method=method,
            mask_action_dim_padding=runtime.mask_action_dim_padding,
            noise=noise,
        )
        sliced = chunk[:, : runtime.n_action_steps, : runtime.action_dim]
        unnormalized = unnormalize_chunk(
            sliced.cpu(),
            self.action_q01[: runtime.action_dim].cpu(),
            self.action_q99[: runtime.action_dim].cpu(),
        )
        # The reference's output dtype path: unnormalize in fp32, cast
        # BACK to the sampled dtype, then fp32 — the bf16 quantization is
        # part of the reference output when the expert runs bf16.
        actions = unnormalized.to(sliced.dtype).to(torch.float32)
        return BijouPrediction(actions=actions, generations=None, noise=noise)

    def iter_blocks(self) -> Iterator[MolmoFlowBlock]:
        for block in self.blocks:
            assert isinstance(block, MolmoFlowBlock)
            yield block

    def _reshape_hidden_to_heads(self, x: Tensor) -> Tensor:
        """Shapes:
        - ``x``: [B, S, hidden]
        - returns: [B, S, num_heads, head_dim] view
        """
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

    def _time_conditioning(self, timesteps: Tensor) -> Tensor:
        """Their HF inference semantics: the sinusoid runs at the
        timestep dtype (the flow loop feeds fp32 grids), then the
        embedding is cast to the MLP weight dtype — a no-op at uniform
        dtype, load-bearing for fp32 timesteps on a bf16 expert.

        Shapes:
        - ``timesteps``: [B] flow times
        - returns: [B, hidden] conditioning at the MLP weight dtype
        """
        sinusoid, first_linear = self.time_embed[0], self.time_embed[1]
        conditioning = sinusoid(timesteps)
        assert isinstance(first_linear, nn.Linear)
        conditioning = conditioning.to(dtype=first_linear.weight.dtype)
        for module in list(self.time_embed.children())[1:]:
            conditioning = module(conditioning)
        return conditioning

    def _encode_states(self, states: Tensor | None) -> Tensor | None:
        """Shapes:
        - ``states``: [B, D] or [B, S_state, D]; D pads/truncates to hidden
        - returns: [B, S_state, hidden] (S_state = 1 for 2-D input), or None
        """
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
        encoder_kv_states: Sequence[tuple[Tensor, Tensor]],
        encoded_states: Tensor | None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Shapes:
        - ``encoder_kv_states``: one ([B, S_ctx, llm_kv_dim], same) pair
          per block
        - ``encoded_states``: [B, S_state, hidden] (or None)
        - returns: per block ([B, S_ctx + S_state, heads, head_dim], same)
        """
        if len(encoder_kv_states) != len(self.blocks):
            raise ValueError(
                "expected one KV state per molmo_flow block "
                f"(got {len(encoder_kv_states)}, expected {len(self.blocks)})",
            )
        state_heads = (
            self._reshape_hidden_to_heads(encoded_states)
            if encoded_states is not None
            else None
        )
        kv_contexts = []
        # The trunk may run a different precision than the expert (bf16
        # mount vs fp32 expert — the FlowDecoder stream convention); the
        # cast is a no-op at uniform dtype, so byte-parity with the
        # reference (which only ever runs uniform) is preserved.
        dtype = self.context_k_proj.weight.dtype
        for k_in, v_in in encoder_kv_states:
            k_ctx = self._reshape_hidden_to_heads(
                self.context_norm(self.context_k_proj(k_in.to(dtype))),
            )
            v_ctx = self._reshape_hidden_to_heads(
                self.context_norm(self.context_v_proj(v_in.to(dtype))),
            )
            if state_heads is not None:
                k_ctx = torch.cat([k_ctx, state_heads], dim=1)
                v_ctx = torch.cat([v_ctx, state_heads], dim=1)
            kv_contexts.append((k_ctx, v_ctx))
        return kv_contexts

    def _build_cross_attention_mask(
        self,
        encoder_attention_mask: Tensor | None,
        encoded_states: Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
    ) -> Tensor | None:
        """Shapes:
        - ``encoder_attention_mask``: [B, S_ctx] bool/int (or pre-built
          [B, 1, 1, S_ctx]); True/1 = attendable
        - ``encoded_states``: [B, S_state, hidden] (or None) — states are
          appended as always-attendable columns
        - returns: [B, 1, 1, S_ctx + S_state] additive, or None
        """
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
        action_attention_mask: Tensor | None,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        """Shapes:
        - ``action_attention_mask``: [B, T] (1 = real action row), or None
        - returns: [B, 1, T, T] additive (or [1, 1, T, T] pure-causal), or
          None when nothing masks
        """
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
        actions: Tensor,
        timesteps: Tensor,
        encoder_kv_states: Sequence[tuple[Tensor, Tensor]],
        encoder_attention_mask: Tensor | None = None,
        action_attention_mask: Tensor | None = None,
        state_embeddings: Tensor | None = None,
    ) -> Tensor:
        """Predict the velocity field at ``timesteps``.

        Shapes:
        - ``actions``: [B, T <= max_horizon, max_action_dim] noisy chunk
        - ``timesteps``: [B] flow times (ascending convention: 0 = noise)
        - ``encoder_kv_states``: one ([B, S_ctx, llm_kv_dim], same) pair
          per block (post-RoPE trunk K/V, flattened)
        - ``encoder_attention_mask``: [B, S_ctx] bool/int, or None
        - ``action_attention_mask``: [B, T], 1 = real row, or None
        - ``state_embeddings``: [B, D] or [B, S_state, D], or None
        - returns: [B, T, max_action_dim] velocity toward data
        """
        if len(encoder_kv_states) == 0:
            raise ValueError("expected at least one encoder KV state")
        bsz, seq_len, _ = actions.shape
        if seq_len > self.config.max_horizon:
            raise ValueError(
                f"action sequence length {seq_len} exceeds "
                f"max_horizon={self.config.max_horizon}",
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

    @torch.no_grad()
    def sample_actions(
        self,
        encoder_kv_states: Sequence[tuple[Tensor, Tensor]],
        *,
        encoder_attention_mask: Tensor | None = None,
        action_horizon: int | None = None,
        action_dim_is_pad: Tensor | None = None,
        num_steps: int = 10,
        method: SamplingMethod = SamplingMethod.EULER,
        mask_action_dim_padding: bool = True,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Integrate the velocity field from t=0 (noise) to t=1 (data).

        EULER at ``num_steps`` (their serving loop: t = k/n ascending,
        ``x += v/n``, padded dims re-zeroed on noise, velocity AND
        trajectory every step — byte-compatible with the reference,
        parity-gated) or HEUN (2 evaluations per step; its corrector
        evaluates at exactly t=1, in-support under this convention).
        ``generator``/``noise`` for deterministic evaluation; the
        generator must live on the KV device.

        Shapes:
        - ``encoder_kv_states``: one ([B, S_ctx, llm_kv_dim], same) pair
          per block
        - ``encoder_attention_mask``: [B, S_ctx] bool (or None)
        - ``action_dim_is_pad``: [D] or [B, D] bool (or None)
        - ``noise`` (when given): [B, horizon, max_action_dim]
        - returns: [B, horizon, max_action_dim] normalized action chunk
        """
        if len(encoder_kv_states) == 0:
            raise ValueError("expected at least one encoder KV state")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        horizon = self.config.max_horizon if action_horizon is None else action_horizon
        if not 1 <= horizon <= self.config.max_horizon:
            raise ValueError(
                f"action_horizon must be in [1, {self.config.max_horizon}], "
                f"got {horizon}",
            )
        source = encoder_kv_states[0][0]
        batch_size, device = source.shape[0], source.device
        if noise is None:
            noise = torch.randn(
                (batch_size, horizon, self.config.max_action_dim),
                device=device,
                dtype=self.action_embed.weight.dtype,
                generator=generator,
            )
        trajectory = _mask_padded_dims(
            noise,
            action_dim_is_pad=action_dim_is_pad,
            enabled=mask_action_dim_padding,
        )

        def velocity_at(x: Tensor, t: float) -> Tensor:
            times = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            velocity = self(
                x,
                times,
                encoder_kv_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            return _mask_padded_dims(
                velocity,
                action_dim_is_pad=action_dim_is_pad,
                enabled=mask_action_dim_padding,
            )

        dt = 1.0 / num_steps
        for k in range(num_steps):
            t = k / num_steps
            velocity = velocity_at(trajectory, t)
            if method is SamplingMethod.HEUN:
                predicted = trajectory + dt * velocity
                velocity_next = velocity_at(predicted, (k + 1) / num_steps)
                velocity = 0.5 * (velocity + velocity_next)
            trajectory = _mask_padded_dims(
                trajectory + dt * velocity,
                action_dim_is_pad=action_dim_is_pad,
                enabled=mask_action_dim_padding,
            )
        return trajectory


def layer_kv_pairs(
    memory: ObservationMemory,
    *,
    num_blocks: int,
    detach: bool = False,
) -> list[tuple[Tensor, Tensor]]:
    """Per-layer conditioning pairs off the prefix cache in the memory —
    the trunk-specific extraction at the seam (the ``ar_molmo2``
    precedent: this decoder consumes the Molmo2 cache directly; the
    cache layout [B, kv_heads, S, head_dim] is OUR contract, so the
    flatten needs no layout inference). ``detach`` is the knowledge-
    insulation seam (§8.13 decision 8): detached pairs carry no graph,
    so flow gradients into every trunk parameter are exactly zero.

    Shapes:
    - ``memory.cache`` layers: K/V [B, kv_heads, S, head_dim] each
    - returns: ``num_blocks`` pairs of [B, S, kv_heads * head_dim]
    """
    cache = memory.cache
    if cache is None:
        raise ValueError(
            "ObservationMemory carries no prefix cache — encode with "
            "retain_cache=True (BijouModel does this for molmo_flow)",
        )
    if not isinstance(cache, Molmo2KVCache):
        raise TypeError(
            f"molmo_flow conditions on the Molmo2 prefix cache; the memory "
            f"carries {type(cache).__name__}",
        )
    pairs: list[tuple[Tensor, Tensor]] = []
    for layer_idx, layer in enumerate(cache.layers):
        keys, values = layer.keys, layer.values
        if keys is None or values is None:
            raise ValueError(
                f"cache layer {layer_idx} has no K/V — the prompt forward "
                "must fill every layer before KV extraction",
            )
        batch, kv_heads, seq_len, head_dim = keys.shape
        key = keys.permute(0, 2, 1, 3).reshape(batch, seq_len, kv_heads * head_dim)
        value = values.permute(0, 2, 1, 3).reshape(batch, seq_len, kv_heads * head_dim)
        if detach:
            key, value = key.detach(), value.detach()
        pairs.append((key, value))
    if len(pairs) != num_blocks:
        raise ValueError(
            f"expected {num_blocks} KV layers (one per molmo_flow block), "
            f"got {len(pairs)}",
        )
    return pairs


def conditioning_mask_of(memory: ObservationMemory) -> Tensor | None:
    """The expert's prompt mask: the encoder-computed conditioning mask
    (the ``action_mode`` flavor) when present, else the padding mask —
    None only for unpadded memories with no flavor (everything
    attendable, their unbatched serving case).

    Shapes:
    - returns: [B, S] bool, or None
    """
    if memory.conditioning_mask is not None:
        return memory.conditioning_mask
    if memory.padding_mask is not None:
        return memory.padding_mask.to(dtype=torch.bool)
    return None


def molmo_flow_loss_sums(
    decoder: MolmoFlowDecoder,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
    *,
    insulate: bool = False,
    times: Tensor | None = None,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """The batch-facing objective (BijouModel dispatches here): extract
    KV off the prefix cache (detached under ``insulate`` — their
    post-train KI), q01/q99-clamp-normalize the RAW batch actions with
    the DECODER'S table (decision 6: per-sample stats deliberately
    unused), pad to max_action_dim, draw t from the RECORDED law and
    ε ~ N(0, I) (ambient RNG — the trainer's seeded stream, matching
    every other decoder's loss), and return the sum-form objective.
    ``times``/``noise`` overrides are for oracles.

    Episode-end repeated actions train as real targets — the same call
    as flow.py's loss and their reference wrapper (lerobot's clamped
    delta-timestamps already hold the reach-and-hold target).

    Shapes:
    - ``batch.actions``: [B, T == action_horizon, action_dim] raw units
    - returns: (scalar loss sum with graph, scalar position count B*T)
    """
    runtime = decoder._configured()
    kv_states = layer_kv_pairs(
        memory,
        num_blocks=len(decoder.blocks),
        detach=insulate,
    )
    enc_mask = conditioning_mask_of(memory)
    actions = batch.actions.to(torch.float32)
    if actions.shape[-1] != runtime.action_dim:
        raise ValueError(
            f"batch action width {actions.shape[-1]} != the configured "
            f"action_dim {runtime.action_dim}",
        )
    if actions.shape[1] != runtime.action_horizon:
        raise ValueError(
            f"batch chunk length {actions.shape[1]} != the configured "
            f"action_horizon {runtime.action_horizon} — --chunk-size must "
            "match the checkpoint's horizon",
        )
    normalized = normalize_targets(
        actions,
        decoder.action_q01[: runtime.action_dim].to(actions.device),
        decoder.action_q99[: runtime.action_dim].to(actions.device),
    )
    padded = F.pad(
        normalized,
        (0, decoder.config.max_action_dim - runtime.action_dim),
    )
    dim_is_pad = decoder.dim_pad_mask().to(actions.device).expand(actions.shape[0], -1)
    if times is None:
        times = runtime.time_law.sample(actions.shape[0], device=actions.device)
    if noise is None:
        noise = torch.randn_like(padded)
    return flow_matching_loss_sums(
        decoder,
        kv_states=kv_states,
        enc_mask=enc_mask,
        actions_norm=padded,
        action_dim_is_pad=dim_is_pad,
        times=times,
        noise=noise,
    )


def molmo_flow_loss(
    decoder: MolmoFlowDecoder,
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
    *,
    insulate: bool = False,
) -> Tensor:
    """Mean form of :func:`molmo_flow_loss_sums` — the unchunked
    objective (sum / count; same contract as flow.py's pair)."""
    loss_sum, count = molmo_flow_loss_sums(
        decoder,
        memory,
        batch,
        insulate=insulate,
    )
    return loss_sum / count


def _padded_dim_mask(target: Tensor, action_dim_is_pad: Tensor | None) -> Tensor | None:
    """Broadcastable bool mask of VALID action dims (their
    ``_action_dim_valid_mask``), or None when nothing is padded.

    Shapes:
    - ``target``: [B, ..., D] tensor the mask must broadcast against
    - ``action_dim_is_pad``: [D] or [B, D] bool, True = padded dim
    - returns: [B, 1..., D] bool (True = valid), or None
    """
    if action_dim_is_pad is None:
        return None
    mask = ~action_dim_is_pad.to(device=target.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"action_dim_is_pad width {mask.shape[-1]} does not match "
            f"action width {target.shape[-1]}",
        )
    if mask.shape[0] == 1 and target.shape[0] != 1:
        mask = mask.expand(target.shape[0], -1)
    if mask.shape[0] != target.shape[0]:
        raise ValueError(
            f"action_dim_is_pad batch {mask.shape[0]} does not match "
            f"batch {target.shape[0]}",
        )
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(1)
    return mask


def _mask_padded_dims(
    tensor: Tensor,
    *,
    action_dim_is_pad: Tensor | None,
    enabled: bool,
) -> Tensor:
    """Shapes:
    - ``tensor``: [B, T, D]
    - ``action_dim_is_pad``: [D] or [B, D] bool (or None)
    - returns: [B, T, D] with padded dims zeroed (or unchanged)
    """
    if not enabled:
        return tensor
    valid = _padded_dim_mask(tensor, action_dim_is_pad)
    if valid is None:
        return tensor
    return tensor.masked_fill(~valid, 0)


def flow_matching_loss_sums(
    decoder: MolmoFlowDecoder,
    *,
    kv_states: Sequence[tuple[Tensor, Tensor]],
    enc_mask: Tensor | None,
    actions_norm: Tensor,
    action_dim_is_pad: Tensor,
    times: Tensor,
    noise: Tensor,
) -> tuple[Tensor, Tensor]:
    """Their flow objective in sum form (chunked-backward exact): the
    per-position mean over VALID action dims, summed over [B, T]; mean
    form = sum / count. Padded dims are zeroed on both actions and
    noise and excluded from the mean.

    ASCENDING convention (module docstring): ``x_t = (1−t)·ε + t·x``,
    target ``x − ε`` — the mirror of flow.py's objective; the direction
    test ties this to the sampler so they can never disagree about
    which end is data.

    Shapes:
    - ``kv_states``: one ([B, S, llm_kv_dim], same) pair per block
    - ``enc_mask``: [B, S] bool (or None)
    - ``actions_norm``: [B, T, max_action_dim] fp32 normalized+clamped
    - ``action_dim_is_pad``: [B, max_action_dim] bool
    - ``times``: [B] fp32
    - ``noise``: [B, T, max_action_dim] fp32
    - returns: (scalar loss sum with graph, scalar position count B*T)
    """
    valid = ~action_dim_is_pad  # [B, D]
    dim_mask = valid[:, None, :].to(actions_norm.dtype)  # [B, 1, D]
    actions_masked = actions_norm * dim_mask
    noise_masked = noise * dim_mask
    t = times[:, None, None]  # [B, 1, 1]
    xt = (1.0 - t) * noise_masked + t * actions_masked
    target = actions_masked - noise_masked
    pred = decoder(
        xt,
        times,
        encoder_kv_states=kv_states,
        encoder_attention_mask=enc_mask,
    )
    err = F.mse_loss(pred.to(torch.float32), target, reduction="none")
    per_dim = err * dim_mask
    counts = valid.sum(dim=-1).clamp(min=1)[:, None]  # [B, 1]
    per_position = per_dim.sum(dim=-1) / counts  # [B, T]
    count = torch.tensor(
        per_position.numel(),
        device=per_position.device,
        dtype=torch.float32,
    )
    return per_position.sum(), count


def normalize_targets(actions: Tensor, q01: Tensor, q99: Tensor) -> Tensor:
    """Their input-side action path: q01/q99 normalize then clamp to
    [-1, 1] (zero-width ranges replaced by eps, their guard).

    Shapes:
    - ``actions``: [..., D] raw units
    - ``q01``/``q99``: [D]
    - returns: [..., D] float32 in [-1, 1]
    """
    actions = torch.as_tensor(actions, dtype=torch.float32)
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.tensor(1e-8, dtype=torch.float32), denom)
    return (2.0 * (actions - q01) / denom - 1.0).clamp(-1.0, 1.0)


def unnormalize_chunk(chunk: Tensor, q01: Tensor, q99: Tensor) -> Tensor:
    """Their output tail, in their order: clamp the sampled normalized
    chunk to [-1, 1], then invert the q01/q99 map back to raw units in
    fp32 (the caller applies the reference's dtype round-trip).

    Shapes:
    - ``chunk``: [..., D] sampled normalized chunk
    - ``q01``/``q99``: [D]
    - returns: [..., D] float32 raw units
    """
    chunk = torch.as_tensor(chunk, dtype=torch.float32).clamp(-1.0, 1.0)
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.tensor(1e-8, dtype=torch.float32), denom)
    return (chunk + 1.0) * denom / 2.0 + q01


def load_expert_state(
    decoder: MolmoFlowDecoder,
    state_dict: dict[str, Tensor],
) -> None:
    """Load a converter ``expert.safetensors`` state dict (unprefixed
    names), injecting the compat tensors exports omit — identity
    ``state_encoder``, zero per-block ``cross_attn.kv_proj`` — exactly
    like their loader (and the port's ``load_action_expert_state``,
    modulo the converter's already-stripped prefix)."""
    if len(state_dict) == 0:
        raise ValueError("empty expert state dict")
    loaded = dict(state_dict)
    sample = next(iter(loaded.values()))
    hidden = decoder.config.hidden_size
    if "state_encoder.weight" not in loaded:
        loaded["state_encoder.weight"] = torch.eye(hidden, dtype=sample.dtype)
        loaded["state_encoder.bias"] = torch.zeros(hidden, dtype=sample.dtype)
    for idx in range(len(decoder.blocks)):
        kv_w = f"blocks.{idx}.cross_attn.kv_proj.weight"
        if kv_w not in loaded:
            loaded[kv_w] = torch.zeros(hidden * 2, hidden, dtype=sample.dtype)
            loaded[f"blocks.{idx}.cross_attn.kv_proj.bias"] = torch.zeros(
                hidden * 2,
                dtype=sample.dtype,
            )
    decoder.load_state_dict(loaded, strict=True)
