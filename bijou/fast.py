"""FAST action-chunk tokenizer: DCT + BPE (Pertsch et al., arXiv:2501.09747).

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

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

CONFIG_FILENAME = "fast_config.json"
BPE_FILENAME = "bpe.json"
QUANTILE_FILENAME = "quantile_stats.json"
REPORT_FILENAME = "fit_report.json"


class FastDecodeError(ValueError):
    """A token sequence does not decode to a valid coefficient matrix."""


@dataclass(frozen=True, slots=True)
class QuantileEntry:
    """One dataset's per-dimension q01/q99 action quantiles — the
    normalization the tokenizer was fitted under. Tokens are only
    meaningful for chunks normalized with the SAME entry."""

    q01: tuple[float, ...]
    q99: tuple[float, ...]

    def normalize(self, chunk: FloatArray) -> FloatArray:
        """Raw action chunk -> roughly [-1, 1] (q01..q99 mapped exactly)."""
        low = np.asarray(self.q01)
        span = np.maximum(np.asarray(self.q99) - low, 1e-6)
        return (chunk - low) / span * 2.0 - 1.0

    def unnormalize(self, chunk: FloatArray) -> FloatArray:
        low = np.asarray(self.q01)
        span = np.maximum(np.asarray(self.q99) - low, 1e-6)
        return (chunk + 1.0) / 2.0 * span + low


def quantile_entry_for(
    table: dict[str, QuantileEntry],
    repo_id: str,
) -> QuantileEntry:
    """Loud lookup: wrong quantiles silently corrupt token semantics, so
    there is deliberately NO aggregate fallback."""
    entry = table.get(repo_id)
    if entry is None:
        raise ValueError(
            f"no quantile stats for {repo_id!r} in this FAST tokenizer — "
            "compute them (python -m bijou.fast on a corpus including it, "
            "or extend the run's table for a new rig dataset)",
        )
    return entry


def save_quantile_table(
    table: dict[str, QuantileEntry],
    directory: Path,
) -> None:
    (directory / QUANTILE_FILENAME).write_text(
        json.dumps(
            {
                repo_id: {"q01": list(e.q01), "q99": list(e.q99)}
                for repo_id, e in sorted(table.items())
            },
            indent=2,
        ),
    )


def load_quantile_table(directory: Path) -> dict[str, QuantileEntry]:
    data = json.loads((directory / QUANTILE_FILENAME).read_text())
    return {
        repo_id: QuantileEntry(
            q01=tuple(float(v) for v in entry["q01"]),
            q99=tuple(float(v) for v in entry["q99"]),
        )
        for repo_id, entry in data.items()
    }


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
    ) -> FastTokenizer:
        """Fit the BPE on pre-quantized coefficients [N, H*D] (the output
        of :meth:`quantize_chunks`, possibly concatenated across
        datasets)."""
        if quantized.ndim != 2 or quantized.shape[0] == 0:
            raise ValueError(
                f"expected non-empty [N, H*D] array, got {quantized.shape}",
            )
        if quantized.shape[1] != time_horizon * action_dim:
            raise ValueError(
                f"{quantized.shape[1]} coefficients per chunk != "
                f"time_horizon * action_dim ({time_horizon} * {action_dim})",
            )
        min_coefficient = int(quantized.min())
        alphabet_size = int(quantized.max()) - min_coefficient + 1
        if alphabet_size > vocab_size:
            raise ValueError(
                f"quantized alphabet ({alphabet_size} symbols) exceeds "
                f"vocab_size {vocab_size}; lower the scale or raise the "
                "vocabulary",
            )

        offset = quantized - min_coefficient

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
    ) -> FastTokenizer:
        """Fit on normalized chunks [N, time_horizon, action_dim].

        ``scale`` trades reconstruction fidelity against compression
        (paper default 10, insensitive); ``vocab_size`` is the BPE
        vocabulary (paper default 1024).
        """
        return cls.fit_quantized(
            cls.quantize_chunks(chunks, scale),
            scale=scale,
            time_horizon=chunks.shape[1],
            action_dim=chunks.shape[2],
            vocab_size=vocab_size,
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


# --- fit CLI (python -m bijou.fast) ----------------------------------------
# Deliberately self-contained (pathlib + pandas over parquet): pulling in
# bijou.data would drag torch/lerobot into what is a pure-CPU stats pass,
# and the selection guards it needs are two lines (action-dim match).


@dataclass(frozen=True, slots=True)
class DatasetChunks:
    """One dataset's contribution to the fit."""

    repo_id: str
    entry: QuantileEntry
    quantized: IntArray
    normalized_sample: FloatArray
    raw_span: FloatArray


def discover_action_datasets(
    roots: list[Path],
    action_dim: int,
) -> list[tuple[str, Path]]:
    """(repo_id, dir) for every dataset under the roots whose action
    feature matches ``action_dim``; drops are printed with reasons."""
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for root in roots:
        for info_path in sorted(root.glob("**/meta/info.json")):
            dataset_dir = info_path.parent.parent
            repo_id = f"{dataset_dir.parent.name}/{dataset_dir.name}"
            if repo_id in seen:
                print(f"  - {repo_id} (duplicate; keeping first)", flush=True)
                continue
            seen.add(repo_id)
            features = json.loads(info_path.read_text()).get("features") or {}
            action = features.get("action")
            if action is None:
                print(f"  - {repo_id} (no action feature)", flush=True)
                continue
            dim = int(action["shape"][0])
            if dim != action_dim:
                print(f"  - {repo_id} (action dim {dim})", flush=True)
                continue
            found.append((repo_id, dataset_dir))
    if not found:
        raise SystemExit(f"no datasets with action dim {action_dim} under {roots}")
    return found


def episode_actions(dataset_dir: Path) -> list[FloatArray]:
    frames = [
        pd.read_parquet(p, columns=["episode_index", "action"])
        for p in sorted((dataset_dir / "data").rglob("*.parquet"))
    ]
    table = pd.concat(frames)
    episodes: list[FloatArray] = []
    for _, group in table.groupby("episode_index", sort=True):
        # pandas boundary: to_numpy() on a column of vectors yields an
        # object array of per-frame arrays; stack over the plain list.
        rows: list[Any] = list(group["action"].to_numpy())
        episodes.append(np.stack(rows).astype(np.float64))
    return episodes


def dataset_chunks(
    repo_id: str,
    dataset_dir: Path,
    *,
    chunk_size: int,
    stride: int,
    scale: float,
) -> DatasetChunks | None:
    """Exact q01/q99 over ALL frames, then quantized full-window chunks.
    None (printed) when no episode is long enough for a single window."""
    episodes = episode_actions(dataset_dir)
    all_frames = np.concatenate(episodes)
    entry = QuantileEntry(
        q01=tuple(np.quantile(all_frames, 0.01, axis=0).tolist()),
        q99=tuple(np.quantile(all_frames, 0.99, axis=0).tolist()),
    )
    windows: list[FloatArray] = []
    for actions in episodes:
        windows.extend(
            actions[start : start + chunk_size]
            for start in range(0, actions.shape[0] - chunk_size + 1, stride)
        )
    if not windows:
        print(f"  - {repo_id} (no episode reaches {chunk_size} frames)", flush=True)
        return None
    normalized = entry.normalize(np.stack(windows))
    span = np.maximum(np.asarray(entry.q99) - np.asarray(entry.q01), 1e-6)
    return DatasetChunks(
        repo_id=repo_id,
        entry=entry,
        quantized=FastTokenizer.quantize_chunks(normalized, scale),
        normalized_sample=normalized[:: max(len(normalized) // 200, 1)][:200],
        raw_span=span,
    )


def fidelity_report(
    tokenizer: FastTokenizer,
    datasets: list[DatasetChunks],
) -> dict[str, Any]:
    """Per-dataset tokens/chunk and reconstruction error in RAW action
    units on the held-back sample — printed AND returned for
    fit_report.json (measure before claiming, in the artifact itself)."""
    naive = tokenizer.time_horizon * tokenizer.action_dim
    report: dict[str, Any] = {}
    print(
        f"\n{'dataset':44s} {'tok/chunk':>9s} {'p90':>5s} {'compress':>9s} "
        f"{'recon MAE raw':>14s}",
        flush=True,
    )
    for data in datasets:
        tokens = tokenizer.encode_batch(data.normalized_sample)
        lengths = np.array([len(t) for t in tokens])
        decoded = np.stack([tokenizer.decode(t) for t in tokens])
        error_raw = np.abs(decoded - data.normalized_sample) * (data.raw_span / 2.0)
        row = {
            "chunks": int(data.quantized.shape[0]),
            "tokens_mean": float(lengths.mean()),
            "tokens_p90": float(np.quantile(lengths, 0.9)),
            "compression": float(naive / lengths.mean()),
            "recon_mae_raw": float(error_raw.mean()),
            "recon_p99_raw": float(np.quantile(error_raw, 0.99)),
        }
        report[data.repo_id] = row
        print(
            f"{data.repo_id:44s} {row['tokens_mean']:9.1f} "
            f"{row['tokens_p90']:5.0f} {row['compression']:8.1f}x "
            f"{row['recon_mae_raw']:14.3f}",
            flush=True,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bijou.fast",
        description="Fit a FAST action tokenizer (DCT + BPE) on LeRobot v3 "
        "collection roots / dataset dirs, with per-dataset quantile "
        "normalization. Writes an immutable tokenizer directory "
        "(fast_config.json, bpe.json, quantile_stats.json, "
        "fit_report.json) ready for hub upload.",
    )
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="window stride within episodes; BPE fitting is cheap enough "
        "(~0.8s per 7k chunks, near-linear) that dense strides are fine",
    )
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="optional cap on total fitted chunks (uniform per-dataset "
        "subsample, seeded); omit to fit on everything",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(
            f"{args.output} exists and is not empty — tokenizers are "
            "versioned immutably, pick a fresh directory",
        )

    started = time.perf_counter()
    pairs = discover_action_datasets(args.data, args.action_dim)
    print(f"{len(pairs)} datasets with action dim {args.action_dim}", flush=True)

    datasets: list[DatasetChunks] = []
    for index, (repo_id, dataset_dir) in enumerate(pairs, start=1):
        data = dataset_chunks(
            repo_id,
            dataset_dir,
            chunk_size=args.chunk_size,
            stride=args.stride,
            scale=args.scale,
        )
        if data is not None:
            datasets.append(data)
        if index % 25 == 0 or index == len(pairs):
            total = sum(d.quantized.shape[0] for d in datasets)
            print(
                f"  [{index}/{len(pairs)}] {total} chunks, "
                f"{time.perf_counter() - started:.0f}s",
                flush=True,
            )

    rng = np.random.default_rng(args.seed)
    quantized_parts: list[IntArray] = []
    total = sum(d.quantized.shape[0] for d in datasets)
    keep_fraction = 1.0 if args.max_chunks is None else args.max_chunks / total
    for data in datasets:
        part = data.quantized
        if keep_fraction < 1.0:
            keep = max(int(part.shape[0] * keep_fraction), 1)
            part = part[rng.choice(part.shape[0], size=keep, replace=False)]
        quantized_parts.append(part)
    quantized = np.concatenate(quantized_parts)
    print(
        f"\nfitting BPE on {quantized.shape[0]} of {total} chunks "
        f"(scale {args.scale:g}, vocab {args.vocab_size}) ...",
        flush=True,
    )
    fit_started = time.perf_counter()
    tokenizer = FastTokenizer.fit_quantized(
        quantized,
        scale=args.scale,
        time_horizon=args.chunk_size,
        action_dim=args.action_dim,
        vocab_size=args.vocab_size,
    )
    fit_seconds = time.perf_counter() - fit_started
    print(
        f"fitted in {fit_seconds:.0f}s: alphabet {tokenizer.alphabet_size}, "
        f"vocab {tokenizer.vocab_size}",
        flush=True,
    )

    report = fidelity_report(tokenizer, datasets)
    tokenizer.save(args.output)
    save_quantile_table({d.repo_id: d.entry for d in datasets}, args.output)
    (args.output / REPORT_FILENAME).write_text(
        json.dumps(
            {
                "data": [str(p) for p in args.data],
                "datasets": len(datasets),
                "chunks_total": total,
                "chunks_fitted": int(quantized.shape[0]),
                "chunk_size": args.chunk_size,
                "stride": args.stride,
                "scale": args.scale,
                "vocab_size": args.vocab_size,
                "seed": args.seed,
                "fit_seconds": round(fit_seconds, 1),
                "per_dataset": report,
            },
            indent=2,
        ),
    )
    print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
