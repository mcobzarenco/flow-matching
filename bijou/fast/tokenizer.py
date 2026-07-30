"""FAST tokenizer implementation: DCT + BPE (Pertsch et al., arXiv:2501.09747).

Owned reimplementation of the reference ``physical-intelligence/fast``
processor (no ``trust_remote_code``), with this repo's conventions:

- ``time_horizon``/``action_dim`` are FIXED at fit time (the reference
  caches them mutably from the last call; our chunks are homogeneous).
- ``decode`` raises :class:`FastDecodeError` on invalid token sequences
  (the reference prints and returns zeros). Autoregressive generations can
  be malformed — the caller decides the fallback, loudly.
- Coefficients outside the fitted alphabet are clipped and COUNTED (the
  reference clips the low end silently); the first clip prints a warning.
- The DCT is an explicit orthonormal DCT-II matrix (time_horizon squared),
  applied along time per action dimension; the inverse is its transpose
  (orthogonal), so round-trips are exact up to quantization. This avoids a
  scipy dependency for one 50-point transform and matches
  ``scipy.fft.dct(..., norm="ortho")`` analytically.

Input chunks must already be normalized (the paper recommends per-dataset
quantile normalization: q01..q99 mapped to [-1, 1]); the tokenizer is
normalization-agnostic and the quantization scale is calibrated for
roughly unit-range signals.

Tokenization pipeline (fit and encode share the first three steps):

  chunk [H, D] --DCT over time--> coefficients [H, D]
    --round(x * scale)--> integers --flatten (freq-major, dims
    interleaved: low frequencies of every dim first)--> alphabet chars
    --BPE--> token ids

BPE is a plain ``tokenizers`` BPE model over a synthetic single-codepoint
alphabet (one char per quantized coefficient value), no pre-tokenizer, no
normalizer: merges therefore span the whole flattened chunk, and decoding
is an exact join of token strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

CONFIG_FILENAME = "fast_config.json"
BPE_FILENAME = "bpe.json"


class FastDecodeError(ValueError):
    """A token sequence does not decode to a valid coefficient matrix."""


# Encoding-convention constants (v1). Both are part of what makes tokens
# meaningful — changing either invalidates every fitted tokenizer and
# trained AR head, so changes require a NEW tokenizer version by policy.
#
# CONSTANT_SPAN (raw units, degrees here): a dimension whose q01..q99 span
# is below this is effectively constant (parked joint, padded dim); it
# carries no information and normalizes to 0 — dividing by a floored span
# instead amplifies encoder jitter and outlier excursions by ~1e6 (observed:
# a 7.3e9-symbol alphabet on the community corpus).
# NORMALIZED_CLIP: quantile normalization puts 98% of mass in [-1, 1];
# excursions beyond this are outliers by the FAST paper's own robustness
# argument and are clipped (callers report clip counts loudly).
CONSTANT_SPAN = 1e-3
NORMALIZED_CLIP = 8.0

# Fit-time alphabet bound: the base alphabet covers this fraction of the
# corpus' quantized coefficients (per-tail (1-x)/2 quantiles); the rest
# clip into the alphabet edge. Derived from min/max instead, a handful of
# worst-case outliers dictate the alphabet and eat the BPE merge budget —
# fast_tokenizer_v1 shipped with 1019 of 1024 slots spent on base symbols
# and FIVE learned merges, ~2x the tokens/chunk of a bounded fit at
# reconstruction error identical to the third decimal
# (docs/fast-tokenizer-v1-review.md). The measured coefficient tail is
# insensitive around this value (0.008% beyond |71|, none beyond |150| on
# a 359k-coefficient probe); 1.0 reproduces the min/max behavior.
DEFAULT_ALPHABET_COVERAGE = 0.99995


@dataclass(frozen=True, slots=True)
class QuantileEntry:
    """One dataset's per-dimension q01/q99 action quantiles — the
    normalization the tokenizer was fitted under. Tokens are only
    meaningful for chunks normalized with the SAME entry."""

    q01: tuple[float, ...]
    q99: tuple[float, ...]

    def _span(self) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
        low = np.asarray(self.q01)
        span = np.asarray(self.q99) - low
        constant = span < CONSTANT_SPAN
        return np.where(constant, 1.0, span), constant

    def normalize(self, chunk: FloatArray) -> FloatArray:
        """Raw action chunk -> [-NORMALIZED_CLIP, NORMALIZED_CLIP] with
        q01..q99 mapped to [-1, 1]; constant dimensions map to 0."""
        span, constant = self._span()
        normalized = (chunk - np.asarray(self.q01)) / span * 2.0 - 1.0
        normalized = np.where(constant, 0.0, normalized)
        return np.clip(normalized, -NORMALIZED_CLIP, NORMALIZED_CLIP)

    def normalized_overflow(self, chunk: FloatArray) -> int:
        """How many values of ``chunk`` fall outside the clip range — for
        loud reporting at fit/encode call sites."""
        span, constant = self._span()
        normalized = (chunk - np.asarray(self.q01)) / span * 2.0 - 1.0
        normalized = np.where(constant, 0.0, normalized)
        return int((np.abs(normalized) > NORMALIZED_CLIP).sum())

    def unnormalize(self, chunk: FloatArray) -> FloatArray:
        """Inverse on the non-clipped range; constant dimensions restore
        their q01..q99 midpoint."""
        span, constant = self._span()
        low = np.asarray(self.q01)
        raw = (chunk + 1.0) / 2.0 * span + low
        midpoint = (low + np.asarray(self.q99)) / 2.0
        return np.where(constant, midpoint, raw)


def quantile_entry_from_stats(stats_path: Path) -> QuantileEntry:
    """Read the action q01/q99 from a dataset's ``meta/stats.json`` — the
    single source of truth for quantile normalization (exact values are
    backfilled by ldtools.backfill_quantile_stats; LeRobot's native
    episode-aggregated quantiles are wrong for corpus use). Checkpoints
    record the entries they trained with; the tokenizer artifact
    deliberately carries none."""
    action = json.loads(stats_path.read_text()).get("action", {})
    if "q01" not in action or "q99" not in action:
        raise ValueError(
            f"{stats_path} lacks action q01/q99 — backfill exact quantiles "
            "first (uv run python -m ldtools.backfill_quantile_stats "
            "--root <collection> --force, in lerobot-dataset-tools)",
        )
    return QuantileEntry(
        q01=tuple(float(v) for v in action["q01"]),
        q99=tuple(float(v) for v in action["q99"]),
    )


def dct_matrix(n: int) -> FloatArray:
    """Orthonormal DCT-II matrix M [n, n]: ``M @ x`` transforms a length-n
    signal; ``M.T`` is the exact inverse (M is orthogonal).

    M[k, i] = c_k * cos(pi * (2i + 1) * k / (2n)),
    c_0 = sqrt(1/n), c_{k>0} = sqrt(2/n) — the ``norm="ortho"`` convention
    of scipy.fft.dct.
    """
    if n < 1:
        raise ValueError(f"DCT size must be >= 1, got {n}")
    k = np.arange(n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    matrix = np.cos(np.pi * (2.0 * i + 1.0) * k / (2.0 * n))
    matrix *= np.sqrt(2.0 / n)
    matrix[0, :] *= np.sqrt(0.5)
    return matrix


class FastTokenizer:
    """A fitted FAST tokenizer. Build with :meth:`fit` or :meth:`load`."""

    def __init__(
        self,
        bpe: Tokenizer,
        *,
        scale: float,
        min_coefficient: int,
        alphabet_size: int,
        time_horizon: int,
        action_dim: int,
    ) -> None:
        self.bpe = bpe
        self.scale = scale
        self.min_coefficient = min_coefficient
        self.alphabet_size = alphabet_size
        self.time_horizon = time_horizon
        self.action_dim = action_dim
        self._dct = dct_matrix(time_horizon)
        self.clipped_coefficients = 0

    @property
    def vocab_size(self) -> int:
        return int(self.bpe.get_vocab_size())

    def _quantize(self, chunk: FloatArray) -> npt.NDArray[np.int64]:
        """[H, D] float chunk -> flattened alphabet indices [H * D]."""
        if chunk.shape != (self.time_horizon, self.action_dim):
            raise ValueError(
                f"expected chunk of shape "
                f"({self.time_horizon}, {self.action_dim}), got {chunk.shape}",
            )
        coefficients = self._dct @ chunk
        quantized = np.round(coefficients * self.scale).astype(np.int64)
        indices = quantized.reshape(-1) - self.min_coefficient
        clipped = np.clip(indices, 0, self.alphabet_size - 1)
        n_clipped = int((clipped != indices).sum())
        if n_clipped:
            if self.clipped_coefficients == 0:
                print(
                    f"FAST: clipping {n_clipped} DCT coefficient(s) outside "
                    f"the fitted alphabet (further clips counted silently "
                    f"in .clipped_coefficients)",
                    flush=True,
                )
            self.clipped_coefficients += n_clipped
        return clipped

    def encode(self, chunk: FloatArray) -> list[int]:
        """Normalized action chunk [time_horizon, action_dim] -> token ids."""
        indices = self._quantize(chunk)
        text = "".join(map(chr, indices.tolist()))
        # tokenizers is untyped; ids is list[int] per its documentation.
        ids: Any = self.bpe.encode(text).ids
        return list(ids)

    def encode_batch(self, chunks: FloatArray) -> list[list[int]]:
        """[N, time_horizon, action_dim] -> N token-id lists (ragged)."""
        if chunks.ndim != 3:
            raise ValueError(f"expected [N, H, D], got shape {chunks.shape}")
        return [self.encode(chunk) for chunk in chunks]

    def decode(self, token_ids: list[int]) -> FloatArray:
        """Token ids -> action chunk [time_horizon, action_dim].

        Raises :class:`FastDecodeError` when the sequence does not decode
        to exactly time_horizon * action_dim coefficients (malformed
        autoregressive generations) or contains unknown ids.
        """
        pieces: list[str] = []
        for token_id in token_ids:
            piece = self.bpe.id_to_token(token_id)
            if piece is None:
                raise FastDecodeError(
                    f"token id {token_id} is outside the vocabulary "
                    f"({self.vocab_size})",
                )
            pieces.append(piece)
        indices = np.array([ord(ch) for ch in "".join(pieces)], dtype=np.int64)
        expected = self.time_horizon * self.action_dim
        if indices.shape[0] != expected:
            raise FastDecodeError(
                f"decoded to {indices.shape[0]} coefficients, expected "
                f"{expected} ({self.time_horizon} x {self.action_dim})",
            )
        quantized = indices.reshape(self.time_horizon, self.action_dim)
        coefficients = (quantized + self.min_coefficient) / self.scale
        return self._dct.T @ coefficients

    @classmethod
    def quantize_chunks(cls, chunks: FloatArray, scale: float) -> IntArray:
        """Normalized chunks [N, H, D] -> quantized DCT coefficients
        [N, H*D] (freq-major flatten, NOT yet alphabet-offset) — the
        dataset-independent half of fitting, so a corpus fit can stream
        per dataset and only the alphabet bounds are global."""
        if chunks.ndim != 3 or chunks.shape[0] == 0:
            raise ValueError(
                f"expected non-empty [N, H, D] chunk array, got {chunks.shape}",
            )
        matrix = dct_matrix(chunks.shape[1])
        coefficients = np.einsum("kh,nhd->nkd", matrix, chunks.astype(np.float64))
        quantized = np.round(coefficients * scale).astype(np.int64)
        return quantized.reshape(quantized.shape[0], -1)

    @classmethod
    def fit_quantized(
        cls,
        quantized: IntArray,
        *,
        scale: float,
        time_horizon: int,
        action_dim: int,
        vocab_size: int = 1024,
        alphabet_coverage: float = DEFAULT_ALPHABET_COVERAGE,
    ) -> FastTokenizer:
        """Fit the BPE on pre-quantized coefficients [N, H*D] (the output
        of :meth:`quantize_chunks`, possibly concatenated across
        datasets).

        The base alphabet is bounded by corpus quantiles covering
        ``alphabet_coverage`` of the coefficients, and the corpus is
        clipped to that alphabet BEFORE the BPE fit — the trainer builds
        its alphabet from the data, so an unclipped corpus would re-admit
        the tail symbols (see DEFAULT_ALPHABET_COVERAGE for why min/max
        derivation is a footgun). Encode-time out-of-alphabet
        coefficients clip identically (:meth:`_quantize`, counted in
        ``clipped_coefficients``), so fit and encode see the same
        distribution."""
        if quantized.ndim != 2 or quantized.shape[0] == 0:
            raise ValueError(
                f"expected non-empty [N, H*D] array, got {quantized.shape}",
            )
        if quantized.shape[1] != time_horizon * action_dim:
            raise ValueError(
                f"{quantized.shape[1]} coefficients per chunk != "
                f"time_horizon * action_dim ({time_horizon} * {action_dim})",
            )
        if not 0.5 < alphabet_coverage <= 1.0:
            raise ValueError(
                f"alphabet_coverage must be in (0.5, 1.0], got {alphabet_coverage}",
            )

        flat = quantized.reshape(-1)
        tail = (1.0 - alphabet_coverage) / 2.0
        low = int(np.quantile(flat, tail, method="lower"))
        high = int(np.quantile(flat, 1.0 - tail, method="higher"))
        n_clipped = int(((flat < low) | (flat > high)).sum())
        if n_clipped:
            print(
                f"FAST fit: alphabet bounded to [{low}, {high}] "
                f"({alphabet_coverage:.3%} coverage); clipping {n_clipped} "
                f"of {flat.size} corpus coefficients "
                f"({n_clipped / flat.size:.5%})",
                flush=True,
            )
        bounded = np.clip(quantized, low, high)
        min_coefficient = int(bounded.min())
        alphabet_size = int(bounded.max()) - min_coefficient + 1
        if alphabet_size > vocab_size // 2:
            raise ValueError(
                f"quantized alphabet ({alphabet_size} symbols) would eat "
                f"the BPE merge budget (vocab_size {vocab_size}) — the "
                "degenerate fast_tokenizer_v1 failure mode. Check the "
                "inputs were quantile-normalized (raw chunks produce huge "
                "DCT coefficients), or lower --alphabet-coverage / "
                "--scale, or raise --vocab-size",
            )

        offset = bounded - min_coefficient

        def corpus() -> Any:
            for row in offset:
                yield "".join(map(chr, row.tolist()))

        bpe = Tokenizer(BPE())
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=False,
            special_tokens=[],
            initial_alphabet=[chr(i) for i in range(alphabet_size)],
            max_token_length=10_000,
        )
        bpe.train_from_iterator(corpus(), trainer=trainer)
        # tokenizers is untyped; get_vocab() is dict[str, int].
        vocab: Any = bpe.get_vocab()
        merges = sum(1 for token in vocab if len(token) > 1)
        print(
            f"FAST fit: alphabet [{low}, {high}] ({alphabet_size} symbols), "
            f"{merges} learned merges, vocab {bpe.get_vocab_size()}",
            flush=True,
        )
        return cls(
            bpe,
            scale=scale,
            min_coefficient=min_coefficient,
            alphabet_size=alphabet_size,
            time_horizon=time_horizon,
            action_dim=action_dim,
        )

    @classmethod
    def fit(
        cls,
        chunks: FloatArray,
        *,
        scale: float = 10.0,
        vocab_size: int = 1024,
        alphabet_coverage: float = DEFAULT_ALPHABET_COVERAGE,
    ) -> FastTokenizer:
        """Fit on normalized chunks [N, time_horizon, action_dim].

        ``scale`` trades reconstruction fidelity against compression
        (paper default 10, insensitive); ``vocab_size`` is the BPE
        vocabulary (paper default 1024); ``alphabet_coverage`` bounds the
        base alphabet by corpus coefficient quantiles (see
        :meth:`fit_quantized`).
        """
        return cls.fit_quantized(
            cls.quantize_chunks(chunks, scale),
            scale=scale,
            time_horizon=chunks.shape[1],
            action_dim=chunks.shape[2],
            vocab_size=vocab_size,
            alphabet_coverage=alphabet_coverage,
        )

    def save(self, directory: Path) -> None:
        """Write ``fast_config.json`` + ``bpe.json`` (hub-uploadable dir)."""
        directory.mkdir(parents=True, exist_ok=True)
        self.bpe.save(str(directory / BPE_FILENAME))
        (directory / CONFIG_FILENAME).write_text(
            json.dumps(
                {
                    "scale": self.scale,
                    "min_coefficient": self.min_coefficient,
                    "alphabet_size": self.alphabet_size,
                    "time_horizon": self.time_horizon,
                    "action_dim": self.action_dim,
                },
                indent=2,
            ),
        )

    @classmethod
    def load(cls, directory: Path) -> FastTokenizer:
        config = json.loads((directory / CONFIG_FILENAME).read_text())
        return cls(
            Tokenizer.from_file(str(directory / BPE_FILENAME)),
            scale=float(config["scale"]),
            min_coefficient=int(config["min_coefficient"]),
            alphabet_size=int(config["alphabet_size"]),
            time_horizon=int(config["time_horizon"]),
            action_dim=int(config["action_dim"]),
        )
