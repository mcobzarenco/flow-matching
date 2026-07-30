"""ActionCodec: FAST tokens + the AR conventions around them.

The FastTokenizer maps normalized chunks to BPE ids in [0, vocab_size);
the codec owns everything around it that the AR decoder and the collator
must agree on: quantile normalization (via QuantileEntry — the same
normalization the tokenizer was fitted under), the BOA/PAD specials
appended AFTER the BPE vocabulary, and the raw-units round trip.

Token id space: [0, vocab_size) = BPE tokens; then BOA (begin-of-actions,
the fixed first token of every sequence), then PAD (batch padding — a
reserved id that never appears as a real token, so masks derive from
``ids != pad``). There is deliberately NO EOA: a valid sequence expands
to exactly time_horizon · action_dim quantized coefficients, so length
is fixed by that grammar and decoding terminates when the symbol budget
reaches zero — generation never samples BOA or PAD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from .tokenizer import FastDecodeError, FastTokenizer, QuantileEntry

# The tokenizer's arrays are float64 (DCT precision); inputs arriving from
# torch items are float32 — accept any float and cast at the boundary.
type AnyFloatArray = npt.NDArray[np.floating]


class ActionCodec:
    """A fitted tokenizer plus the specials and normalization glue."""

    def __init__(self, tokenizer: FastTokenizer) -> None:
        self.tokenizer = tokenizer
        self.boa = tokenizer.vocab_size
        self.pad = tokenizer.vocab_size + 1

    @classmethod
    def load(cls, directory: Path) -> ActionCodec:
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
