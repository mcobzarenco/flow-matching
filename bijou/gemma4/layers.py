"""Shared building blocks: RMSNorm, rotary embeddings, eager attention.

Every op here mirrors the reference HF implementation expression-for-expression
so that outputs are bit-identical in bf16.
"""

from __future__ import annotations

from enum import StrEnum

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import RopeParameters, RopeType
from .masks import MaskSpec

type DeviceLike = torch.device | str | None


class AttentionBackend(StrEnum):
    """How attention is computed. A runtime choice passed at construction
    (never part of the checkpoint config); switchable in place afterwards via
    ``bijou.gemma4.model.set_attention_backend``.

    EAGER mirrors HF's reference implementation op-for-op (fp32 softmax) and
    is the parity baseline. SDPA uses ``F.scaled_dot_product_attention``
    (fused kernels; numerics differ at bf16-ULP scale, greedy tokens verified
    identical — see bijou/gemma4/verify_parity.py) and is the default:
    measured on H100 it is ~1.9x faster / 4.7x leaner on 8k-token text
    prefill and ~1.6x on image prefill, with decode dispatched to eager at
    q_len==1 (faster there).
    """

    EAGER = "eager"
    SDPA = "sdpa"


DEFAULT_ATTENTION_BACKEND = AttentionBackend.SDPA


def buffer_device(device: DeviceLike) -> DeviceLike:
    """Device for *computed* non-persistent buffers (rope frequencies, embed
    scales). Meta-device construction — used by the checkpoint loader — would
    lose their values, and they cannot be restored from a checkpoint, so meta
    requests materialize them on CPU instead; the loader's final ``.to(device)``
    sweep moves them to the target device."""
    if device is not None and torch.device(device).type == "meta":
        return "cpu"
    return device


class RMSNorm(nn.Module):
    """Gemma4 RMSNorm: computed in float32, optional learned scale.

    Note: unlike Gemma 2/3, the scale is applied as ``x * w`` (not ``x * (1+w)``).
    """

    weight: nn.Parameter | None

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        with_scale: bool = True,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        mean_squared = xf.pow(2).mean(-1, keepdim=True) + self.eps
        normed = xf * torch.pow(mean_squared, -0.5)
        if self.weight is not None:
            normed = normed * self.weight.float()
        return normed.type_as(x)


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 2
) -> Tensor:
    """Rotate ``x`` of shape [B, S, H, D] with cos/sin of shape [B, S, D]."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Expand KV heads: [B, KV, S, D] -> [B, KV*n_rep, S, D]."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    num_key_value_groups: int,
    scaling: float = 1.0,
) -> Tensor:
    """Eager attention, softmax in float32; mirrors HF's reference path
    op-for-op. Shapes: q [B, H, Sq, D], k/v [B, KV, Skv, D], additive mask
    [B, 1, Sq, Skv] or None.

    Returns [B, Sq, H, D].
    """
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous()


# Fused SDPA kernels (flash, cudnn) top out at head_dim 256 on H100 as of
# torch 2.11. Above that only the mem-efficient backend remains, and it does
# not support enable_gqa -- without the workaround below, Gemma4's global
# layers (global_head_dim 512 across the E-series) silently fall back to the
# math backend (~3x slower, measured in bijou/gemma4/bench.py).
_SDPA_FUSED_MAX_HEAD_DIM = 256


def sdpa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    num_key_value_groups: int,
    scaling: float = 1.0,
    is_causal: bool = False,
) -> Tensor:
    """Fused attention via ``F.scaled_dot_product_attention``. Same contract
    as :func:`eager_attention`; numerics differ at bf16-ULP scale (fused
    kernels accumulate in fp32 but do not round-trip the softmax through
    fp32->bf16). With ``is_causal`` no mask tensor is passed, keeping the
    flash kernel eligible."""
    if query.shape[-1] > _SDPA_FUSED_MAX_HEAD_DIM and num_key_value_groups > 1:
        # Materialize KV heads so the mem-efficient backend is eligible.
        key = repeat_kv(key, num_key_value_groups)
        value = repeat_kv(value, num_key_value_groups)
        enable_gqa = False
    else:
        enable_gqa = num_key_value_groups > 1
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None if is_causal else attention_mask,
        is_causal=is_causal,
        scale=scaling,
        enable_gqa=enable_gqa,
    )
    return attn_output.transpose(1, 2).contiguous()


def attention(
    backend: AttentionBackend,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: MaskSpec,
    num_key_value_groups: int,
    scaling: float = 1.0,
) -> Tensor:
    """Dispatch to the configured attention implementation.

    Perf policy (measured on H100, see bijou/gemma4/bench.py): single-token decode
    always takes the eager path — at q_len == 1 the fused SDPA kernels are
    launch-bound and ~2x slower than two small gemms. Both paths are
    semantically identical; they differ only at bf16-ULP scale.
    """
    if backend is AttentionBackend.SDPA and query.shape[2] > 1:
        return sdpa_attention(
            query,
            key,
            value,
            mask.tensor,
            num_key_value_groups,
            scaling,
            is_causal=mask.is_causal,
        )
    return eager_attention(
        query, key, value, mask.tensor, num_key_value_groups, scaling
    )


def rope_inv_freq_from_params(
    params: RopeParameters, head_dim: int, *, device: DeviceLike = None
) -> Tensor:
    """Float32 inverse frequencies for one attention geometry.

    ``default``: standard RoPE over the full head_dim.
    ``proportional`` (p-RoPE): only ``partial_rotary_factor * head_dim``
    dimensions carry non-zero frequencies; the rest get frequency 0, which
    makes them rotation-free (cos=1, sin=0).
    """
    base = params.rope_theta
    if params.rope_type is RopeType.DEFAULT:
        exponent = torch.arange(0, head_dim, 2, dtype=torch.int64, device=device)
        inv_freq = 1.0 / (base ** (exponent.to(dtype=torch.float) / head_dim))
    elif params.rope_type is RopeType.PROPORTIONAL:
        rope_angles = int(params.partial_rotary_factor * head_dim // 2)
        exponent = torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64, device=device)
        inv_freq_rotated = 1.0 / (base ** (exponent.to(dtype=torch.float) / head_dim))
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat(
                (
                    inv_freq_rotated,
                    torch.zeros(nope_angles, dtype=torch.float32, device=device),
                ),
                dim=0,
            )
        else:
            inv_freq = inv_freq_rotated
    else:
        raise ValueError(f"unsupported rope type: {params.rope_type}")
    return inv_freq / params.factor


def rope_cos_sin(
    inv_freq: Tensor, position_ids: Tensor, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """cos/sin tables for RoPE, computed in float32 then cast.

    ``inv_freq``: [D/2] float32; ``position_ids``: [B, S] int; returns two
    [B, S, D] tensors of ``dtype``.
    """
    inv_freq_expanded = (
        inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    )
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)
