"""Action-token codecs: the AR conventions around a fitted tokenizer.

Two layers, one word each (docs/code-styleguide.md): a *tokenizer* owns
the artifact and its math (normalized chunk <-> body token ids — DCT +
BPE, fit/load I/O); a *codec* owns everything around it that the AR
decoders and the collator must agree on: quantile normalization to and
from RAW units, the specials (BOA/PAD) and their placement, and the
per-id symbol expansion lengths that drive constrained decoding's
budget arithmetic.

:class:`ActionCodec` is the Protocol — the codec-layer contract
:class:`~bijou.decoders.ar_backbone.ARSuffixDecoder` and the collator
consume. :class:`FastActionCodec` is the implementation over our own
fitted :class:`FastTokenizer`; the released MolmoAct2 family lives in
``bijou/fast/molmoact2.py``.

FastActionCodec's id space: [0, vocab_size) = BPE tokens; then BOA
(begin-of-actions, the fixed first token of every sequence), then PAD
(batch padding — a reserved id that never appears as a real token, so
masks derive from ``ids != pad``). There is deliberately NO EOA: a valid
sequence expands to exactly time_horizon · action_dim quantized
coefficients, so length is fixed by that grammar and decoding terminates
when the symbol budget reaches zero — generation never samples BOA or
PAD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .tokenizer import FastDecodeError, FastTokenizer, QuantileEntry

# The tokenizer's arrays are float64 (DCT precision); inputs arriving from
# torch items are float32 — accept any float and cast at the boundary.
type AnyFloatArray = npt.NDArray[np.floating]


class ActionCodec(Protocol):
    """The codec-layer contract: how action tokens live inside a trunk
    suffix, independent of which fitted artifact produced them.

    Id conventions: body ids are CODEC-RELATIVE — a decoder places body
    id ``i`` at backbone id ``block_base + i`` — and ``boa``/``pad`` are
    codec-relative OFFSETS on the same axis. Negative offsets are legal:
    our fitted artifacts append the specials after the body
    (``boa = vocab_size``), while the MolmoAct2 release keeps
    ``<action_start>``/``<action_end>`` BELOW its bin block (−2/−1).
    ``pad`` doubles as the batch filler fed to finished rows; it is
    legal to emit only when a row's symbol budget is exhausted and is
    never recorded as a body token."""

    @property
    def vocab_total(self) -> int:
        """Width of the codec-relative body id space — the number of
        trunk vocabulary rows the decoder's block reserves. Specials
        inside the block (our artifacts) count toward it; specials
        below the block (MolmoAct2) do not."""
        ...

    @property
    def boa(self) -> int:
        """Codec-relative offset of begin-of-actions — FED by the
        scaffold to open the action stream, never sampled."""
        ...

    @property
    def pad(self) -> int:
        """Codec-relative offset of the padding/stream-close id (see
        the class docstring)."""
        ...

    @property
    def time_horizon(self) -> int:
        """Chunk rows [T] a full emission decodes to — with
        ``action_dim``, the symbol budget T·D of constrained decoding."""
        ...

    @property
    def action_dim(self) -> int:
        """Action dimensions [D] per chunk row."""
        ...

    @property
    def symbol_lengths(self) -> npt.NDArray[np.int64]:
        """[vocab_total] — DCT coefficients body id ``i`` expands to
        under decode; 0 for ids decode can never produce (specials,
        untrained block rows), which the grammar mask excludes for
        free. Owned by the codec because the measurement is
        family-specific (our fit: BPE piece length; the released
        byte-level BPE: decoded-string length)."""
        ...

    def encode(
        self,
        actions: AnyFloatArray,
        q01: AnyFloatArray,
        q99: AnyFloatArray,
    ) -> list[int]:
        """RAW-unit chunk [time_horizon, action_dim] -> codec-relative
        ids ``[boa, t_1..t_k]`` under the artifact's own normalization
        convention."""
        ...

    def decode(
        self,
        token_ids: list[int],
        q01: AnyFloatArray,
        q99: AnyFloatArray,
    ) -> npt.NDArray[np.float64]:
        """Codec-relative body ids (a leading ``boa`` is tolerated) ->
        RAW-unit chunk [time_horizon, action_dim] float64. Malformed
        streams raise — the caller owns any fallback policy."""
        ...


class FastActionCodec:
    """A fitted :class:`FastTokenizer` plus the specials and
    normalization glue (the :class:`ActionCodec` implementation for our
    own artifacts)."""

    def __init__(self, tokenizer: FastTokenizer) -> None:
        self.tokenizer = tokenizer
        self.boa = tokenizer.vocab_size
        self.pad = tokenizer.vocab_size + 1
        # Constrained decoding needs each token's symbol expansion
        # length (one BPE piece = a run of quantized DCT coefficients).
        # Specials stay 0 and are handled explicitly in the decode mask.
        # For OUR fit the piece string IS the symbol string, so its
        # length is the expansion (moved here from ARSuffixDecoder
        # 2026-08-14 — the loop reads the fitted BPE's pieces, an API
        # only this artifact family satisfies).
        lengths = np.zeros(self.vocab_total, dtype=np.int64)
        for token_id in range(tokenizer.vocab_size):
            piece = tokenizer.bpe.id_to_token(token_id)
            assert piece is not None, f"BPE id {token_id} has no piece"
            lengths[token_id] = len(piece)
        self.symbol_lengths = lengths

    @classmethod
    def load(cls, directory: Path) -> FastActionCodec:
        return cls(FastTokenizer.load(directory))

    @property
    def vocab_total(self) -> int:
        """BPE vocabulary + BOA + PAD — the AR head's output size."""
        return self.tokenizer.vocab_size + 2

    @property
    def time_horizon(self) -> int:
        return self.tokenizer.time_horizon

    @property
    def action_dim(self) -> int:
        return self.tokenizer.action_dim

    def encode(
        self,
        actions: AnyFloatArray,
        q01: AnyFloatArray,
        q99: AnyFloatArray,
    ) -> list[int]:
        """Raw action chunk [time_horizon, action_dim] -> [BOA, t_1..t_k].

        Normalization is the tokenizer-fit convention (QuantileEntry:
        q01..q99 to [-1, 1], constant dims to 0, outliers clipped)."""
        entry = QuantileEntry(
            q01=tuple(float(v) for v in q01),
            q99=tuple(float(v) for v in q99),
        )
        normalized = entry.normalize(actions.astype(np.float64))
        return [self.boa, *self.tokenizer.encode(normalized)]

    def decode(
        self,
        token_ids: list[int],
        q01: AnyFloatArray,
        q99: AnyFloatArray,
    ) -> npt.NDArray[np.float64]:
        """[BOA?, t_1..t_k] -> raw action chunk [time_horizon, action_dim].
        A leading BOA is stripped; a PAD id (or any malformed body) raises
        :class:`FastDecodeError` — the caller owns the fallback policy."""
        body = [t for t in token_ids if t != self.boa]
        if any(t == self.pad for t in body):
            raise FastDecodeError("PAD id inside a decoded sequence")
        entry = QuantileEntry(
            q01=tuple(float(v) for v in q01),
            q99=tuple(float(v) for v in q99),
        )
        return entry.unnormalize(self.tokenizer.decode(body))
