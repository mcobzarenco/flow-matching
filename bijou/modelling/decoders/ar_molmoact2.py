"""The MolmoAct2 trunk's discrete-head suffix role — an
`ARSuffixDecoder` concrete (checkpoint decoder kind ``ar_backbone``;
the trunk axis is the PROMPT kind, §2.2a of architecture.md).

**Zero new parameters.** The release trained its action rows into the
trunk's OWN base matrices: suffix ids embed through the frozen-or-live
``wte`` and logits read the shipped ``lm_head`` directly — no
extension tables, no column surgery (grammar legality is the decode
mask's job). The trainable surface under this decoder is the trunk
itself (``--backbone-text-lr``); the decoder owns nothing.

**Suffix format 6** (``MOLMOACT2_SUFFIX_FORMAT``): the emission is
``<action_start>`` + bins + ``<action_end>`` continued directly from
their serving prompt, which already ends
``…<|im_start|>assistant\\n<action_output>`` — so the opener is EMPTY
(the scaffold's first fed token is BOA ≡ ``<action_start>``, exactly
the reference decode's ``advance(opener, 0)``), there are no value
lines (``aux`` must be None), and the specials live BELOW the block:
:class:`~bijou.fast.molmoact2.MolmoAct2ActionCodec` maps them at
offsets −2/−1 so ``block_base`` stays ``action_token_start_id`` — the
frozen capture/TokenRow convention, no ±2 rebase anywhere. In batch
lockstep, finished rows feed ``<action_end>`` (the pad offset, legal
only at symbol budget 0) while others decode on — at B=1 the loop
exits without feeding it, byte-matching the reference's
appended-not-fed close.

**Scope: release-class checkpoints only** (``action_mode='both'``).
The rig-ft exports are ``'continuous'`` — their fine-tune never
trained the discrete head; loading refuses them by name before this
class is ever constructed, and the construction guards here hold the
line for direct builders (probes, tests).

**Construction guards** (the id-space seam made unrepresentable —
bins are block-relative while emissions are backbone ids, and a ±2
rebase error between them is silent everywhere except here):

- the codec must carry the below-block special offsets (−2/−1);
- the block plus both specials must sit INSIDE the base matrices
  (``block_base − 2 ≥ 0``, ``block_base + vocab_total ≤ vocab_size``)
  — a straddle into ``new_embedding`` or past the head would silently
  train/read the wrong rows;
- the trunk tokenizer's actual ``<action_start>``/``<action_end>``/
  ``<action_0>`` ids must equal ``block_base − 2 / − 1 / + 0``
  (release: 151932/151933/151934), verified against the REAL
  tokenizer, never assumed.

Continuation IS the shared Molmo2-stack half
(:func:`~bijou.modelling.decoders.ar_molmo2.continue_molmo2_suffix`):
all-layer continuation against the prefix
:class:`~bijou.modelling.molmo2.cache.Molmo2KVCache` (every
``MolmoAct2Encoder.encode`` product carries it), text-typed suffix
queries under shifted-causal masking, and the non-cuDNN sdpa pin
(cheap insurance; if the decode-parity fixture gate ever disagrees,
that pin and the mount dtype are the first suspects — the reference
implementation decoded under the full kernel dispatcher).
"""

from __future__ import annotations

from typing import override

from torch import Tensor

from ..codecs import ActionCodec
from ..encoders.molmo2 import Molmo2Memory
from ..molmo2.config import Molmo2TextConfig
from ..molmo2.model import Molmo2Model
from ..molmo2.tokenizer import Molmo2TextTokenizer
from .ar_molmo2 import continue_molmo2_suffix
from .ar_suffix import MOLMOACT2_SUFFIX_FORMAT, ARDecoderConfig, ARSuffixDecoder

# The trunk-tokenizer anchors verified at construction: token string →
# codec-relative offset from block_base. A trained contract of the
# release vocabulary; the stub ids in tests mirror the same layout.
_ACTION_TOKEN_ANCHORS: tuple[tuple[str, int], ...] = (
    ("<action_start>", -2),
    ("<action_end>", -1),
    ("<action_0>", 0),
)


class MolmoAct2ARDecoder(ARSuffixDecoder[Molmo2Model, Molmo2Memory]):
    """The MolmoAct2 trunk's suffix role (see the module docstring).

    ``text_config`` is the FULL decoder architecture — construction
    validates that the action block and both specials are base-matrix
    rows. ``tokenizer`` is the trunk's own tokenizer (the concrete
    class, not the protocol: the anchor verification reads
    ``token_to_id`` off its backend)."""

    def __init__(
        self,
        config: ARDecoderConfig,
        text_config: Molmo2TextConfig,
        codec: ActionCodec,
        *,
        tokenizer: Molmo2TextTokenizer,
    ) -> None:
        super().__init__(
            config,
            codec,
            tokenizer=tokenizer,
            opener_text="",
            aux_runtime=None,
        )
        if config.suffix_format != MOLMOACT2_SUFFIX_FORMAT:
            raise ValueError(
                f"suffix format {config.suffix_format} is not the "
                f"MolmoAct2 release emission ({MOLMOACT2_SUFFIX_FORMAT}) "
                "— format-5 checkpoints load the Gemma/Molmo2 concretes",
            )
        if config.aux is not None:
            raise ValueError(
                "the MolmoAct2 release emission has no value lines — "
                "aux must be None (narration on this trunk is a prompt-"
                "format extension, not this decoder)",
            )
        if (codec.boa, codec.pad) != (-2, -1):
            raise ValueError(
                f"codec special offsets (boa={codec.boa}, pad={codec.pad}) "
                "are not the below-block layout (-2, -1) this concrete "
                "implements — pass MolmoAct2ActionCodec, not a fitted-"
                "family codec",
            )
        if len(self.opener_ids) != 0:
            raise ValueError(
                f"tokenizer injected {len(self.opener_ids)} id(s) into the "
                "empty opener — the serving prompt already carries the "
                "whole scaffold; a non-empty opener would shift every "
                "suffix position off the reference",
            )
        block_end = config.block_base + config.vocab_total
        if config.block_base - 2 < 0 or block_end > text_config.vocab_size:
            raise ValueError(
                f"action block [{config.block_base}, {block_end}) with "
                f"specials at {config.block_base - 2}/{config.block_base - 1} "
                f"does not sit inside the base matrices (vocab_size "
                f"{text_config.vocab_size}) — a straddle into the extension "
                "tables would silently train the wrong rows",
            )
        backend = tokenizer.tokenizer
        mismatches = [
            f"{token!r} at {actual} (expected {config.block_base + offset})"
            for token, offset in _ACTION_TOKEN_ANCHORS
            if (actual := backend.token_to_id(token)) != config.block_base + offset
        ]
        if mismatches:
            raise ValueError(
                "trunk tokenizer disagrees with block_base "
                f"{config.block_base}: {'; '.join(mismatches)} — the "
                "checkpoint's recorded block base does not anchor this "
                "vocabulary",
            )

    @override
    def init_tables_from_backbone(self, backbone: Molmo2Model) -> None:
        """Nothing to initialize — the head owns ZERO parameters
        (decision 2): embedding and logits are the trunk's own rows,
        trained by the release itself."""

    @override
    def _suffix_hidden(
        self,
        backbone: Molmo2Model,
        memory: Molmo2Memory,
        tokens: Tensor,
        fed: int,
    ) -> Tensor:
        # Trunk-native: every suffix id (bins AND the two specials) is a
        # base-matrix row — plain lookup, no extension-table select, no
        # scaling (Molmo2 embeds unscaled).
        embeds = backbone.text.transformer.wte(tokens)
        return continue_molmo2_suffix(backbone, memory, embeds, fed)

    @override
    def _logits(self, backbone: Molmo2Model, hidden: Tensor) -> Tensor:
        """The shipped head, unmodified: [B, T, vocab_size] — wider than
        ``block_base + vocab_total`` when the vocabulary continues past
        the action block (callers slice; the release trained these
        columns, gradients flow through them into the trunk)."""
        lm_head = backbone.text.lm_head
        assert lm_head is not None  # Molmo2Model requires the full decoder
        return lm_head(hidden)
