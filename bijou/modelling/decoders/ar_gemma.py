"""The Gemma backbone's suffix role — the AR-first decoder concrete on
the Gemma trunk (scaffold: :class:`~.ar_suffix.ARSuffixDecoder`).

What differs from the other concretes is exactly the trunk compute:
scaled embedding lookup with the PLE (per-layer-embedding) tables, the
FAST block living in Gemma's unused vocabulary tail (fresh trainable
rows patched over it), and continuation against the Gemma
:class:`~..gemma4.cache.KVCache`.
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..aux_text import (
    GENERATION_OPENER,
    SUFFIX_FORMAT,
    AuxRuntime,
    TextTokenizer,
)
from ..codecs import ActionCodec
from ..gemma4.cache import KVCache
from ..gemma4.config import Gemma4TextConfig
from ..gemma4.model import Gemma4Model
from ..interface import ObservationMemory
from ..nn import DeviceLike
from .ar_suffix import ARDecoderConfig, ARSuffixDecoder, suffix_positions


class GemmaARDecoder(ARSuffixDecoder[Gemma4Model]):
    """The Gemma backbone's suffix role (see the module docstring).

    ``text_config`` is the FULL backbone architecture — construction
    validates the block placement and that the KV-shared deep half is
    present (a truncated backbone has none; this decoder is
    definitionally the full stack's suffix role)."""

    def __init__(
        self,
        config: ARDecoderConfig,
        text_config: Gemma4TextConfig,
        codec: ActionCodec,
        *,
        tokenizer: TextTokenizer | None,
        aux_runtime: AuxRuntime | None = None,
        aux_loss_weight: float = 1.0,
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            config,
            codec,
            tokenizer=tokenizer,
            opener_text=GENERATION_OPENER,
            aux_runtime=aux_runtime,
            aux_loss_weight=aux_loss_weight,
        )
        if config.suffix_format != SUFFIX_FORMAT:
            raise ValueError(
                f"suffix format {config.suffix_format} is not the "
                f"value-line scaffold ({SUFFIX_FORMAT}) this concrete "
                "implements — format-6 checkpoints load MolmoAct2ARDecoder",
            )
        if config.block_base + config.vocab_total > text_config.vocab_size:
            raise ValueError(
                f"FAST block [{config.block_base}, "
                f"{config.block_base + config.vocab_total}) does not fit the "
                f"backbone vocabulary ({text_config.vocab_size})",
            )
        if text_config.num_kv_shared_layers == 0:
            raise ValueError(
                "ar_backbone needs the FULL backbone (its suffix runs the "
                "KV-shared deep half); this config is a truncated prefix — "
                "load with depth=full",
            )
        hidden = text_config.hidden_size
        self.num_layers = text_config.num_hidden_layers
        self.ple_dim = text_config.hidden_size_per_layer_input
        # Same scale asymmetry as the backbone's own tying: input lookups
        # are multiplied by √dim (ScaledEmbedding), the head uses raw rows.
        self.embed_scale = hidden**0.5
        self.ple_scale = self.ple_dim**0.5
        self.fast_embed = nn.Embedding(
            config.vocab_total,
            hidden,
            device=device,
            dtype=dtype,
        )
        self.fast_ple = nn.Embedding(
            config.vocab_total,
            self.num_layers * self.ple_dim,
            device=device,
            dtype=dtype,
        )
        if device is None or torch.device(device).type != "meta":
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Fallback init: patch tables at the text-embedding-typical 0.02
        std. Training warm-up should prefer
        :meth:`init_tables_from_backbone`."""
        nn.init.normal_(self.fast_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fast_ple.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    @override
    def init_tables_from_backbone(self, backbone: Gemma4Model) -> None:
        """Re-init the patch rows around the REAL tables' row mean
        (+0.02 noise) — the block's logits then start near the average
        text logit instead of at an arbitrary offset, which matters under
        full-vocabulary CE (the data contract's extended-vocab recipe)."""
        text = backbone.language_model
        embed_mean = text.embed_tokens.weight.float().mean(dim=0)
        ple_mean = text.embed_tokens_per_layer.weight.float().mean(dim=0)
        for table, mean in (
            (self.fast_embed.weight, embed_mean),
            (self.fast_ple.weight, ple_mean),
        ):
            noise = torch.randn_like(table) * 0.02
            table.copy_(mean.to(table.dtype)[None, :] + noise)

    def _suffix_inputs_backbone_ids(
        self,
        backbone: Gemma4Model,
        tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """(inputs_embeds, per_layer_inputs) for BACKBONE-id ``tokens``
        [B, T], routed by range: text ids (< block_base — opener + value
        lines) embed through the frozen embed_tokens/PLE tables, block
        ids through the FAST patch. An all-block suffix reproduces the
        pre-aux computation bitwise (torch.where with an all-False mask
        returns the block side elementwise). Returns
        ([B, T, hidden], [B, T, num_layers, ple_dim]) in the backbone's
        dtype."""
        text = backbone.language_model
        target_dtype = text.embed_tokens.weight.dtype
        is_text = (tokens < self.config.block_base)[..., None]
        block_ids = (tokens - self.config.block_base).clamp(min=0)
        # Text-side lookups use the pad row at block positions
        # (discarded by the select) — every id stays in range for every
        # table.
        text_ids = torch.where(
            is_text[..., 0],
            tokens,
            torch.full_like(tokens, text.config.pad_token_id),
        )
        embeds = torch.where(
            is_text,
            text.embed_tokens(text_ids).float(),
            self.fast_embed(block_ids) * self.embed_scale,
        )
        ple = torch.where(
            is_text[..., None],
            text.get_per_layer_inputs(text_ids).float(),
            (self.fast_ple(block_ids) * self.ple_scale).view(
                tokens.shape[0],
                tokens.shape[1],
                self.num_layers,
                self.ple_dim,
            ),
        )
        return embeds.to(target_dtype), ple.to(target_dtype)

    def _patched_logits(self, backbone: Gemma4Model, hidden: Tensor) -> Tensor:
        """Full-vocabulary logits with the FAST block's columns computed
        from the trainable patch, softcapped AFTER the overwrite so the
        block is capped identically to text (Gemma4Model.forward
        semantics). hidden [B, S, hidden] → [B, S, vocab_size].

        Memory discipline (a [B, S, 262k] tensor is ~1.2 GiB fp32 at
        B10 — an out-of-place chain here OOM'd the first full-recipe
        run): the block columns are written IN PLACE (identical values
        and gradient routing to the old cat splice — the overwritten
        head columns received no gradient there either); the softcap's
        div runs in place (scalar backward saves nothing) and tanh_ in
        place, but the trailing scale must be OUT-of-place — tanh_
        saves its output for backward, and a further in-place op on it
        trips the version counter (measured, not theorized). Two
        full-vocab tensors live instead of the old ~four."""
        base = self.config.block_base
        end = base + self.config.vocab_total
        logits = backbone.lm_head(hidden)
        block = hidden @ self.fast_embed.weight.to(hidden.dtype).T
        logits[..., base:end] = block
        softcap = backbone.config.text.final_logit_softcapping
        if softcap is not None:
            logits = logits.div_(softcap).tanh_() * softcap
        return logits

    def _continue_suffix(
        self,
        backbone: Gemma4Model,
        memory: ObservationMemory,
        embeds: Tensor,
        per_layer: Tensor,
        fed: int,
    ) -> Tensor:
        """Run S suffix embeddings through ALL layers against the prefix
        cache (which they extend in place). ``fed`` = suffix positions
        already in the cache from previous calls (decode loop); positions
        continue per-sample after each REAL prompt length + fed.
        Returns final-normed hidden states [B, S, hidden]."""
        cache = memory.cache
        if cache is None:
            raise ValueError(
                "ObservationMemory carries no prefix cache — encode with "
                "retain_cache=True (suffix-decoder families do)",
            )
        if not isinstance(cache, KVCache):
            # The seam types the cache opaquely (trunk-private contract);
            # this decoder continues the GEMMA stack through it.
            raise TypeError(
                f"ar_backbone continues the Gemma prefix cache; the memory "
                f"carries {type(cache).__name__}",
            )
        batch, seq_len, _ = embeds.shape
        positions, full_mask = suffix_positions(
            memory,
            batch=batch,
            seq_len=seq_len,
            fed=fed,
            device=embeds.device,
        )
        # cuDNN's fused-attention graph intermittently fails to EXECUTE
        # its backward on the suffix geometry (bf16 head_dim-512 queries
        # at ragged lengths against the prefix cache) — the
        # pytorch/pytorch#122695 'mha_graph.execute is_good()' assert
        # family. It killed the fullstack run twice (steps 10440,
        # ~20500) starting exactly when the suffix went bf16 and thus
        # became cuDNN-eligible; the TORCH_CUDNN_SDPA_ENABLED env var is
        # NOT honored by this torch (2.11) — verified the hard way. Pin
        # THIS call to the non-cuDNN kernels (the backend chosen at
        # forward time also selects the backward); the prefix encode —
        # ~80% of compute, crash-free for >100k steps — keeps the full
        # dispatcher including cuDNN. No-op on CPU/eager paths.
        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
        ):
            return backbone.language_model(
                inputs_embeds=embeds,
                per_layer_inputs=per_layer,
                position_ids=positions,
                padding_mask=full_mask,
                cache=cache,
            )

    @override
    def _suffix_hidden(
        self,
        backbone: Gemma4Model,
        memory: ObservationMemory,
        tokens: Tensor,
        fed: int,
    ) -> Tensor:
        embeds, per_layer = self._suffix_inputs_backbone_ids(backbone, tokens)
        return self._continue_suffix(backbone, memory, embeds, per_layer, fed)

    @override
    def _logits(self, backbone: Gemma4Model, hidden: Tensor) -> Tensor:
        return self._patched_logits(backbone, hidden)
