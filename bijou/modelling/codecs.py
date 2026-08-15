"""Action-token codecs: the AR conventions around a fitted tokenizer.

Two layers, one word each (docs/code-styleguide.md): a *tokenizer* owns
the artifact and its math (normalized chunk <-> body token ids — DCT +
BPE, fit/load I/O); a *codec* owns everything around it that the AR
decoders and the collator must agree on: quantile normalization to and
from RAW units, the specials (BOA/PAD) and their placement, and the
per-id symbol expansion lengths that drive constrained decoding's
budget arithmetic.

:class:`ActionCodec` is the Protocol — the codec-layer contract
:class:`~bijou.modelling.decoders.ar_suffix.ARSuffixDecoder` and the
collator consume. :class:`FastActionCodec` is the implementation over
our own fitted :class:`FastTokenizer`;
:class:`MolmoAct2ActionCodec` is the released MolmoAct2 family's, over
:class:`~bijou.fast.molmoact2.MolmoAct2FastTokenizer`. Tokenizers stay
in ``bijou/fast/`` (the artifact layer); codecs live here, beside the
interface and decoders whose id-space conventions they carry.

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
import torch

from ..fast.molmoact2 import (
    MolmoAct2FastTokenizer,
    QuantileStats,
    normalize_state,
    unnormalize_action,
)
from ..fast.tokenizer import (
    AnyFloatArray,
    FastDecodeError,
    FastTokenizer,
    QuantileEntry,
)


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


def _quantile_stats(q01: AnyFloatArray, q99: AnyFloatArray) -> QuantileStats:
    """Per-call q01/q99 rows → their float32 stats object (norm_stats
    tables are read as float32; batch stats arrive as fp32/64 arrays)."""
    return QuantileStats(
        q01=torch.tensor(np.asarray(q01), dtype=torch.float32),
        q99=torch.tensor(np.asarray(q99), dtype=torch.float32),
    )


class MolmoAct2ActionCodec:
    """The released family's CODEC layer — the :class:`ActionCodec`
    implementation over
    :class:`~bijou.fast.molmoact2.MolmoAct2FastTokenizer` plus the
    release geometry.

    **Id space.** Codec-relative body ids ARE the block-relative bins
    (backbone id = ``block_base + bin`` with ``block_base =
    action_token_start_id`` — the convention every capture/TokenRow
    surface shares), so the specials sit at NEGATIVE offsets: ``boa =
    −2`` (``<action_start>``, backbone 151932 on the release) and
    ``pad = −1`` (``<action_end>``, 151933). The suffix scaffold's
    ``block_base + offset`` arithmetic then lands on the real trunk
    ids with no ±2 rebase anywhere, and ``<action_end>`` doubles as
    the pad/filler id: legal to emit exactly when a row's symbol
    budget reaches 0 (the release's stream-close semantics) and fed to
    finished rows in batch lockstep. Neither special is ever a CE
    target (``suffix_targets`` ignores pad targets; BOA is fed, not
    predicted) — training covers exactly what masked decoding samples:
    bins.

    **Raw units.** The released implementation's op order, op-for-op:
    decode returns fp64-of-fp32 values so a downstream ``.float()``
    reproduces its executed chunks byte-for-byte (it casts the fp64
    DCT decode to fp32 BEFORE the [-1, 1] clamp + q01/q99 inversion;
    fp32 → fp64 → fp32 is lossless). Encode applies the same
    clamp-normalize map action targets train under
    (:func:`~bijou.fast.molmoact2.normalize_state`) before the DCT,
    then guards the symbol
    budget: a chunk whose coefficients hit one of the 7 released-BPE
    quantization holes cannot round-trip and is REFUSED by default (a
    silently short CE target is the exact failure class the loud path
    exists for); ``allow_quantization_holes`` opts a training collator
    into the released recipe's short tokenization, counted and
    printed.
    """

    def __init__(
        self,
        tokenizer: MolmoAct2FastTokenizer,
        *,
        time_horizon: int,
        action_dim: int,
        allow_quantization_holes: bool = False,
    ) -> None:
        if time_horizon <= 0 or action_dim <= 0:
            raise ValueError(
                f"geometry must be positive, got time_horizon="
                f"{time_horizon}, action_dim={action_dim}",
            )
        self.tokenizer = tokenizer
        self.time_horizon = time_horizon
        self.action_dim = action_dim
        self.boa = -2
        self.pad = -1
        # Encode-side hole policy. False (default): a chunk whose
        # coefficients hit one of the 7 released-BPE holes REFUSES
        # (parity harnesses, round-trip tests — a silently short target
        # is a wiring bug there). True (the TRAINING collator): tokenize
        # the dropped-symbol stream AS-IS — the reference recipe's
        # verbatim behavior — counted and printed, never silent
        # (measured 2026-08-14: real rig chunks DO hit holes; the
        # 0/2996 audit figure was masked DECODES, whose budget
        # arithmetic cannot produce holes by construction).
        self.allow_quantization_holes = allow_quantization_holes
        # PER-PROCESS: the training collator forks into DataLoader
        # workers, so each worker counts (and prints) its own holes —
        # loud, but not a run-total statistic. Aggregate across workers
        # before ever consuming the number.
        self.hole_count = 0
        # [block_vocab] int64; 0 beyond bpe_vocab — the 1043 untrained
        # rows and both specials are excluded by the grammar mask for
        # free (lengths > 0 legality).
        self.symbol_lengths = tokenizer.symbol_lengths

    @property
    def vocab_total(self) -> int:
        """The bare 2048-wide bin block — the specials sit BELOW it and
        do not count (they are trunk vocabulary, not block rows)."""
        return self.tokenizer.block_vocab

    def encode(
        self,
        actions: AnyFloatArray,
        q01: AnyFloatArray,
        q99: AnyFloatArray,
    ) -> list[int]:
        """RAW-unit chunk [time_horizon, action_dim] → ``[boa, bins…]``
        under their clamp-normalize convention. Raises on geometry
        mismatches and on quantization-hole chunks (see class
        docstring)."""
        array = np.asarray(actions, dtype=np.float64)
        if array.shape != (self.time_horizon, self.action_dim):
            raise ValueError(
                f"encode expects [{self.time_horizon}, {self.action_dim}] "
                f"raw chunks, got {array.shape}",
            )
        normalized = normalize_state(
            torch.from_numpy(array),
            _quantile_stats(q01, q99),
        )
        bins = self.tokenizer.encode(normalized.double().numpy())
        budget = self.time_horizon * self.action_dim
        expanded = int(self.symbol_lengths[bins].sum()) if bins else 0
        if expanded != budget:
            if not self.allow_quantization_holes:
                raise ValueError(
                    f"encoded stream expands to {expanded} DCT coefficients, "
                    f"expected {budget} — the chunk hit a released-BPE "
                    "quantization hole (7 symbol values the artifact cannot "
                    "represent) and would decode short; refusing to emit a "
                    "silently truncated target (construct with "
                    "allow_quantization_holes=True for the reference "
                    "recipe's behavior)",
                )
            self.hole_count += 1
            if self.hole_count == 1 or self.hole_count % 100 == 0:
                print(
                    f"[molmoact2-codec] quantization-hole chunk "
                    f"#{self.hole_count}: tokenized SHORT ({expanded}/"
                    f"{budget} coefficients — the reference recipe's "
                    "silent drop, kept loud here)",
                    flush=True,
                )
        return [self.boa, *bins]

    def decode(
        self,
        token_ids: list[int],
        q01: AnyFloatArray,
        q99: AnyFloatArray,
    ) -> npt.NDArray[np.float64]:
        """``[boa?, bins…]`` → RAW-unit chunk [time_horizon, action_dim]
        float64 (of exact fp32 values — see the class docstring's op
        order). Malformed bodies raise; the caller owns any fallback."""
        body = [int(t) for t in token_ids if int(t) != self.boa]
        if any(t == self.pad for t in body):
            raise ValueError(
                "<action_end> (the pad offset) inside a decoded body — "
                "the stream close is scaffold-owned, never a body token",
            )
        normalized = self.tokenizer.decode(
            body,
            time_horizon=self.time_horizon,
            action_dim=self.action_dim,
        )
        raw = unnormalize_action(
            torch.from_numpy(normalized).to(torch.float32),
            _quantile_stats(q01, q99),
        )
        return raw.double().numpy()
