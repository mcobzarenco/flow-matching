"""Pure-torch Molmo2 vision tower + connector (SigLIP-so400m class).

Faithful reimplementation of the checkpoint's ``modeling_molmo2.py`` vision
path, op-for-op (conventions read from the raw file):

- ViT: linear patch embedding over flattened 14x14x3 patches (bias), learned
  positional embedding added (no class token), pre-norm blocks with standard
  ``nn.LayerNorm`` (NOT RMSNorm), separate biased wq/wk/wv/wo projections,
  attention computed in float32 (``float32_attention``), plain MLP (w1/w2,
  gelu_pytorch_tanh) — and NO final layernorm: the adapter taps raw block
  outputs.
- The backbone instantiates only ``max(vit_layers) + 1`` blocks (25 of 27 —
  the released checkpoint ships exactly those), taps the block outputs named
  by ``adapter_config.vit_layers`` ([-3, -9] -> blocks 24, 18) and
  concatenates them feature-wise in that order (-> 2304).
- 2x2 attention pooling (``image_pooling_2d``): for each output token the
  processor supplies the member-patch indices (``pooled_patches_idx``,
  -1-padded); the query is the (validity-masked) mean of the member
  features, keys/values are the members, output is adapter hidden (1152).
- Gated projector (``image_projector``): ``w2(act(w1(x)) * w3(x))``,
  bias-free, silu, -> text hidden 2560. The caller adds the result into the
  input-embedding sequence at ``image_patch_id`` positions (single
  injection at layer 0 — the property the residual-tap protocol relies on).

The crop geometry (how ``pooled_patches_idx`` is built, overlap trimming,
global view) is the PROCESSOR's job and lands with WP3; this module is
parity-gated at the backbone boundary on real processor outputs.
"""

from __future__ import annotations

import math
from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..nn import DeviceLike, activation_fn
from .config import Molmo2AdapterConfig, Molmo2VitConfig


class ViTAttention(nn.Module):
    """Bidirectional multi-head attention with biased separate projections
    and optional float32 attention math, mirroring the reference
    ``ViTMultiHeadDotProductAttention`` (eager path expression-for-
    expression; queries and keys/values may differ — the pooling reuses
    this class cross-attention style)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        input_dim: int | None = None,
        float32_attention: bool,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.float32_attention = float32_attention
        input_dim = input_dim or hidden_size

        self.wq = nn.Linear(
            input_dim,
            num_heads * head_dim,
            device=device,
            dtype=dtype,
        )
        self.wk = nn.Linear(
            input_dim,
            num_heads * head_dim,
            device=device,
            dtype=dtype,
        )
        self.wv = nn.Linear(
            input_dim,
            num_heads * head_dim,
            device=device,
            dtype=dtype,
        )
        self.wo = nn.Linear(
            num_heads * head_dim,
            hidden_size,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(
        self,
        inputs_q: Tensor,  # [B, Q, input_dim]
        inputs_kv: Tensor | None = None,  # [B, K, input_dim]; None = self-attn
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Attention over full sequences (no causality).

        ``attn_mask`` True = attend — the reference passes it only on
        the SDPA path; the eager path applies it additively for
        identical semantics.

        Shapes:
        - ``inputs_q``: [B, Q, input_dim]
        - ``inputs_kv``: [B, K, input_dim] (None = self-attention)
        - ``attn_mask``: bool, broadcastable to [B, H, Q, K] (or None)
        - returns: [B, Q, hidden_size]
        """
        if inputs_kv is None:
            inputs_kv = inputs_q
        batch, q_len, _ = inputs_q.shape
        query = self.wq(inputs_q).reshape(  # [B, Q, H, head_dim]
            batch,
            q_len,
            self.num_heads,
            self.head_dim,
        )
        key = self.wk(inputs_kv).reshape(batch, -1, self.num_heads, self.head_dim)
        value = self.wv(inputs_kv).reshape(batch, -1, self.num_heads, self.head_dim)

        original_dtype = query.dtype
        if self.float32_attention:
            query = query.float()
            key = key.float()

        # Reference eager path: einsum over [B, S, H, D], fp32 softmax,
        # weights cast to the VALUE dtype for the second matmul.
        scores = torch.einsum(  # [B, H, Q, K], materialized
            "...qhd,...khd->...hqk",
            query / math.sqrt(query.size(-1)),
            key,
        )
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.einsum(  # [B, Q, H, head_dim]
            "...hqk,...khd->...qhd",
            weights.to(value.dtype),
            value,
        )

        attn_output = attn_output.to(original_dtype)
        attn_output = attn_output.reshape(batch, q_len, -1)  # [B, Q, H*head_dim]
        return self.wo(attn_output)


class ViTMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_act: str,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, device=device, dtype=dtype)
        self.w2 = nn.Linear(hidden_dim, dim, device=device, dtype=dtype)
        self.act = activation_fn(hidden_act)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Shapes:
        - ``x``: [B, N, dim]
        - returns: [B, N, dim]
        """
        return self.w2(self.act(self.w1(x)))


class VisionBlock(nn.Module):
    """Pre-norm ViT block with standard LayerNorm."""

    def __init__(
        self,
        config: Molmo2VitConfig,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attention = ViTAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            float32_attention=config.float32_attention,
            device=device,
            dtype=dtype,
        )
        self.feed_forward = ViTMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            device=device,
            dtype=dtype,
        )
        self.attention_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.ffn_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Shapes:
        - ``x``: [B, N, hidden]
        - returns: [B, N, hidden]
        """
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))


class BlockCollection(nn.Module):
    """The reference's ``transformer`` submodule (named for key parity)."""

    def __init__(
        self,
        config: Molmo2VitConfig,
        num_layers: int,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            VisionBlock(config, device=device, dtype=dtype) for _ in range(num_layers)
        )

    @override
    def forward(self, x: Tensor) -> list[Tensor]:
        """Shapes:
        - ``x``: [B, N, hidden]
        - returns: one [B, N, hidden] per block, in depth order
        """
        hidden_states: list[Tensor] = []
        for block in self.resblocks:
            x = block(x)
            hidden_states.append(x)
        return hidden_states


class Molmo2VisionTransformer(nn.Module):
    """Patch embed + learned positions + N pre-norm blocks; returns EVERY
    block's output (the backbone picks its taps)."""

    def __init__(
        self,
        config: Molmo2VitConfig,
        num_layers: int,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.positional_embedding = nn.Parameter(
            torch.zeros(
                config.image_num_pos,
                config.hidden_size,
                device=device,
                dtype=dtype,
            ),
        )
        self.patch_embedding = nn.Linear(
            config.patch_dim,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.transformer = BlockCollection(
            config,
            num_layers,
            device=device,
            dtype=dtype,
        )

    @override
    def forward(self, x: Tensor) -> list[Tensor]:
        """Dynamic grids (the reference's bicubic position-embedding
        interpolation, video path) are not implemented and refuse loudly.

        Shapes:
        - ``x``: [B, N, patch_dim] flattened patch pixels; N must equal
          ``image_num_pos``
        - returns: one [B, N, hidden] per block
        """
        if x.shape[1] != self.config.image_num_pos:
            raise NotImplementedError(
                f"{x.shape[1]} patches != image_num_pos "
                f"{self.config.image_num_pos}: dynamic-grid position "
                "interpolation is not implemented (image crops are always "
                "full 378x378 views)",
            )
        x = self.patch_embedding(x)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        return self.transformer(x)


class ImageProjectorMLP(nn.Module):
    """Gated bias-free MLP into text hidden: ``w2(act(w1(x)) * w3(x))``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_act: str,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(
            input_dim,
            hidden_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            output_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.w3 = nn.Linear(
            input_dim,
            hidden_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.act = activation_fn(hidden_act)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Shapes:
        - ``x``: [..., input_dim]
        - returns: [..., output_dim]
        """
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class Molmo2VisionBackbone(nn.Module):
    """Tower taps -> 2x2 attention pooling -> gated projector.

    Module names mirror the checkpoint (``image_vit`` / ``image_pooling_2d``
    / ``image_projector``) so loading stays "strip the prefix".
    """

    def __init__(
        self,
        vit_config: Molmo2VitConfig,
        adapter_config: Molmo2AdapterConfig,
        *,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vit_config = vit_config
        self.adapter_config = adapter_config
        self.vit_layers = tuple(
            layer if layer >= 0 else layer + vit_config.num_hidden_layers
            for layer in adapter_config.vit_layers
        )
        # Blocks past the deepest tap are never consumed; the reference
        # truncates at build time and the checkpoint ships only these.
        num_layers = max(self.vit_layers) + 1
        self.image_vit = Molmo2VisionTransformer(
            vit_config,
            num_layers,
            device=device,
            dtype=dtype,
        )
        pool_dim = vit_config.hidden_size * len(adapter_config.vit_layers)
        self.image_pooling_2d = ViTAttention(
            hidden_size=adapter_config.hidden_size,
            num_heads=adapter_config.num_attention_heads,
            head_dim=adapter_config.head_dim,
            input_dim=pool_dim,
            float32_attention=adapter_config.float32_attention,
            device=device,
            dtype=dtype,
        )
        self.image_projector = ImageProjectorMLP(
            adapter_config.hidden_size,
            adapter_config.intermediate_size,
            adapter_config.text_hidden_size,
            adapter_config.hidden_act,
            device=device,
            dtype=dtype,
        )

    def encode_image(self, images: Tensor) -> Tensor:
        """Tower features at the adapter taps, concatenated.

        Shapes:
        - ``images``: [B, T, N, patch_dim] (T = crops/views per sample)
        - returns: [B, T, N, hidden * num_taps]
        """
        batch, views, num_patches, _ = images.shape
        hidden_states = self.image_vit(images.view(batch * views, num_patches, -1))
        features = torch.cat(  # [B*T, N, hidden * num_taps]
            [hidden_states[layer] for layer in self.vit_layers],
            dim=-1,
        )
        return features.view(batch, views, num_patches, -1)

    @override
    def forward(
        self,
        images: Tensor,
        pooled_patches_idx: Tensor,
    ) -> Tensor:
        """Pooled, projected image features for the valid output tokens.

        ``pooled_patches_idx`` names, for each of P output tokens, the G
        member patches in the sample's FLATTENED (view, patch) feature
        grid; -1 marks missing members (crop edges) and all--1 rows mark
        padding tokens, which are dropped from the output — mirroring the
        reference exactly (the scatter target is the caller's
        ``input_ids == image_patch_id`` positions, which the processor
        emits in the same order).

        Shapes:
        - ``images``: [B, T, N, patch_dim]
        - ``pooled_patches_idx``: [B, P, G] long, -1 padded
        - returns: [num_valid_tokens, text_hidden]
        """
        batch = images.shape[0]
        weight = self.image_vit.patch_embedding.weight
        images = images.to(device=weight.device, dtype=weight.dtype)
        image_features = self.encode_image(images)
        dim = image_features.shape[-1]

        valid = pooled_patches_idx >= 0  # [B, P, G]
        valid_token = torch.any(valid, -1)  # [B, P]

        batch_idx = torch.arange(
            batch,
            dtype=torch.long,
            device=pooled_patches_idx.device,
        )
        batch_idx = torch.tile(
            batch_idx.view(batch, 1, 1),
            [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]],
        )
        to_pool = image_features.reshape(batch, -1, dim)[  # [B, P, G, dim]
            batch_idx,
            torch.clip(pooled_patches_idx, 0),
        ]
        to_pool = to_pool * valid.to(to_pool.dtype)[:, :, :, None]
        to_pool = to_pool.reshape(
            [-1, pooled_patches_idx.shape[-1], dim],
        )  # [B*P, G, dim]

        if self.adapter_config.pooling_attention_mask:
            attn_mask = valid.reshape([-1, 1, 1, valid.shape[-1]])
            denom = valid.view(-1, to_pool.shape[-2]).float().sum(-1)
            denom = torch.where(denom == 0, 1, denom)
            query = to_pool.sum(-2, keepdim=True) / denom[:, None, None].to(
                to_pool.dtype,
            )
        else:
            attn_mask = None
            query = to_pool.mean(-2, keepdim=True)
        pooled = self.image_pooling_2d(
            query,
            to_pool,
            attn_mask=attn_mask,
        )  # [B*P, 1, ah]
        pooled = pooled.reshape([batch, -1, pooled.shape[-1]])  # [B, P, adapter_hidden]

        pooled = self.image_projector(pooled)  # [B, P, text_hidden]
        return pooled.view(-1, pooled.shape[-1])[valid_token.flatten()]
