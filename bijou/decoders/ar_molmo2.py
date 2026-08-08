"""The Molmo2 (Qwen3) trunk's suffix role — the AR-first phase-1 decoder
(port plan §6 amendment, owner-confirmed 2026-08-06).

Rides the trunk-generic :class:`~bijou.decoders.ar_backbone.ARSuffixDecoder`
scaffold; what differs from the Gemma concrete is exactly the trunk
compute:

- **Embedding**: no scaled lookup, no PLE. Text ids (< block_base —
  ChatML opener + value lines) embed through the frozen ``wte``
  (base + image-special extension matrices); FAST block ids through the
  trainable ``fast_embed`` rows.
- **FAST block placement**: Qwen3's ~271-id unused tail cannot hold the
  1,026 FAST ids, so the block is a SECOND extension id range at
  ``[total_vocab_size, total_vocab_size + vocab_total)`` =
  [152,064, 153,090) for the 4B SKU (``Molmo2TextConfig.fast_block_base``)
  — Molmo2's own ``new_embedding`` pattern. The trunk carries no rows for
  these ids: this module owns them.
- **Head**: the shipped ``lm_head`` is UNTIED and covers the base vocab
  only, and it stays frozen (the 2026-08-06 18:1xZ freezing split: for
  the original vocabulary BOTH sides freeze; aux text reads the frozen
  head with gradients flowing through it into the trunk). The FAST
  columns are FRESH untied rows (``fast_head``), appended — not
  overwritten. The 128 image-special ids sit between the two segments
  and are never legal targets or emissions: their columns are filled
  with the dtype minimum (softmax weight → 0, argmax never selects).
- **Continuation**: the suffix continues through all 36 layers against
  the prefix :class:`~bijou.molmo2.cache.Molmo2KVCache` (built by
  ``Molmo2Encoder.encode(retain_cache=True)``); suffix queries are
  text-typed, so plain shifted-causal masking is exactly the reference
  ``or_mask`` semantics for them (image-block bidirectionality concerns
  image QUERIES, which all live in the prefill).

Trainable set under this decoder + ``--backbone-text-lr``:
``fast_embed`` + ``fast_head`` (here) + decoder layers + ``ln_f``
(encoder ``param_groups``) + ``state_proj`` (prompt-side);
``wte.embedding``, ``wte.new_embedding``, ``lm_head`` frozen.
"""

from __future__ import annotations

from typing import override

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..aux_text import AuxRuntime, TextTokenizer
from ..fast.codec import ActionCodec
from ..interface import ObservationMemory
from ..molmo2.cache import Molmo2KVCache
from ..molmo2.config import Molmo2TextConfig
from ..molmo2.model import Molmo2Model
from ..nn import DeviceLike
from .ar_backbone import ARBackboneConfig, ARSuffixDecoder

# The ChatML assistant-turn opener — the trunk's own generation prompt
# bytes (chat_template.jinja's add_generation_prompt tail), the analogue
# of Gemma's aux_text.GENERATION_OPENER. A trained contract: change only
# with a MOLMO2_PROMPT_FORMAT bump.
MOLMO2_GENERATION_OPENER = "<|im_start|>assistant\n"


class Molmo2ARDecoder(ARSuffixDecoder[Molmo2Model]):
    """The Molmo2 trunk's suffix role (see the module docstring).

    ``text_config`` is the FULL decoder architecture — construction
    validates the extension-block anchoring against the config's own
    ``fast_block_base`` (a mismatched block would silently shear every
    action target off its embedding row)."""

    def __init__(
        self,
        config: ARBackboneConfig,
        text_config: Molmo2TextConfig,
        codec: ActionCodec,
        *,
        tokenizer: TextTokenizer | None,
        aux_runtime: AuxRuntime | None = None,
        aux_loss_weight: float = 1.0,
        newline_carrier_ids: frozenset[int] = frozenset(),
        device: DeviceLike = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            config,
            codec,
            tokenizer=tokenizer,
            opener_text=MOLMO2_GENERATION_OPENER,
            aux_runtime=aux_runtime,
            aux_loss_weight=aux_loss_weight,
            newline_carrier_ids=newline_carrier_ids,
        )
        if config.block_base != text_config.fast_block_base:
            raise ValueError(
                f"FAST block base {config.block_base} != the trunk's "
                f"extension anchor {text_config.fast_block_base} "
                "(vocab + image specials) — the Molmo2 block is a second "
                "extension range, not a tail reuse",
            )
        self.text_vocab_size = text_config.vocab_size
        # Columns between the shipped head (base vocab) and the FAST
        # block: the image-special input ids — never targets, never
        # emitted; masked to the dtype minimum in _logits.
        self.gap = config.block_base - text_config.vocab_size
        hidden = text_config.hidden_size
        # No input-embedding scaling (Molmo2 embeds unscaled) and no PLE.
        self.fast_embed = nn.Embedding(
            config.vocab_total,
            hidden,
            device=device,
            dtype=dtype,
        )
        # Fresh untied head rows for the block (the shipped lm_head has
        # none and stays frozen).
        self.fast_head = nn.Linear(
            hidden,
            config.vocab_total,
            bias=False,
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
        nn.init.normal_(self.fast_head.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    @override
    def init_tables_from_backbone(self, backbone: Molmo2Model) -> None:
        """Re-init the patch rows around the REAL tables' row mean
        (+0.02 noise) — block logits then start near the average text
        logit instead of at an arbitrary offset, which matters under
        full-vocabulary CE (the Gemma concrete's recipe, one table per
        side of the untied pair)."""
        lm_head = backbone.text.lm_head
        assert lm_head is not None  # Molmo2Model requires the full decoder
        embed_mean = backbone.text.transformer.wte.embedding.float().mean(dim=0)
        head_mean = lm_head.weight.float().mean(dim=0)
        for table, mean in (
            (self.fast_embed.weight, embed_mean),
            (self.fast_head.weight, head_mean),
        ):
            noise = torch.randn_like(table) * 0.02
            table.copy_(mean.to(table.dtype)[None, :] + noise)

    @override
    def _suffix_hidden(
        self,
        backbone: Molmo2Model,
        memory: ObservationMemory,
        tokens: Tensor,
        fed: int,
    ) -> Tensor:
        cache = memory.cache
        if cache is None:
            raise ValueError(
                "ObservationMemory carries no prefix cache — encode with "
                "retain_cache=True (BijouModel does this for AR suffix "
                "decoders)",
            )
        if not isinstance(cache, Molmo2KVCache):
            # The seam types the cache opaquely (trunk-private contract);
            # this decoder continues the MOLMO2 stack through it.
            raise TypeError(
                f"Molmo2ARDecoder continues the Molmo2 prefix cache; the "
                f"memory carries {type(cache).__name__}",
            )
        transformer = backbone.text.transformer
        is_text = (tokens < self.config.block_base)[..., None]
        block_ids = (tokens - self.config.block_base).clamp(min=0)
        # Text-side lookups clamp block positions to the last extension
        # row (discarded by the select) — every id stays in range for
        # every table.
        text_ids = tokens.clamp(max=self.config.block_base - 1)
        # The trainable FAST patch stays fp32 while the mounted trunk may
        # be bf16; torch.where would silently promote the mixed pair to
        # fp32 and the first attention matmul rejects it. Same seam as
        # _logits' fast_head cast.
        text_embeds = transformer.wte(text_ids)
        embeds = torch.where(
            is_text,
            text_embeds,
            self.fast_embed(block_ids).to(text_embeds.dtype),
        )
        batch, seq_len = tokens.shape
        device = embeds.device
        offsets = torch.arange(seq_len, device=device)[None, :] + fed
        if memory.padding_mask is not None:
            real = memory.padding_mask.to(device=device, dtype=torch.bool)
            positions = real.long().sum(dim=1, keepdim=True) + offsets
            full_mask = torch.cat(
                [
                    real,
                    torch.ones(
                        (batch, fed + seq_len),
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=1,
            )
        else:
            positions = torch.full((batch, 1), memory.length, device=device) + offsets
            full_mask = None
        # Training re-admits cuDNN (perf pass-1 pre-reg, parity-gated):
        # the teacher-forced suffix always carries a dense mask (rejects
        # FLASH) and enable_gqa (rejects EFFICIENT), so the inherited
        # non-cuDNN pin left only MATH — 13x/layer measured. Decode
        # keeps the inherited list (the Gemma ragged-geometry crash
        # guard, pytorch#122695 family): banked eval byte-anchors assume
        # decode numerics are frozen across code versions.
        backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        if self.training:
            backends.insert(0, SDPBackend.CUDNN_ATTENTION)
        with sdpa_kernel(backends):
            return transformer(
                inputs_embeds=embeds,
                position_ids=positions,
                padding_mask=full_mask,
                cache=cache,
            )

    @override
    def _logits(self, backbone: Molmo2Model, hidden: Tensor) -> Tensor:
        """Full-id-space logits [B, T, block_base + vocab_total]: frozen
        shipped head over the base vocabulary (gradients flow through it
        into the trunk), dtype-min gap columns for the image specials,
        fresh trainable rows for the FAST block."""
        lm_head = backbone.text.lm_head
        assert lm_head is not None  # Molmo2Model requires the full decoder
        base_logits = lm_head(hidden)
        gap = base_logits.new_full(
            (*hidden.shape[:-1], self.gap),
            torch.finfo(base_logits.dtype).min,
        )
        block = self.fast_head(hidden.to(self.fast_head.weight.dtype))
        return torch.cat([base_logits, gap, block.to(base_logits.dtype)], dim=-1)
