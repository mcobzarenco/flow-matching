"""Fit CLI for FAST tokenizers (invoked via ``python -m bijou.fast``).

Fits on LeRobot v3 collection roots / dataset dirs with per-dataset
quantile normalization and writes an immutable tokenizer directory
(fast_config.json, bpe.json, quantile_stats.json, fit_report.json) ready
for hub upload. See ``bijou/fast/tokenizer.py`` for the algorithm and
``docs/plan_ar_fast.md`` for the artifact lifecycle.

Deliberately self-contained on the data side (pathlib + pandas over
parquet): pulling in bijou.data would drag torch/lerobot into a pure-CPU
stats pass, and the selection guard it needs is an action-dim match.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .tokenizer import (
    FastTokenizer,
    FloatArray,
    IntArray,
    QuantileEntry,
    quantile_entry_from_stats,
)

REPORT_FILENAME = "fit_report.json"


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
    """Quantized full-window chunks, normalized with the dataset's OWN
    stats.json quantiles (fit normalization must equal training
    normalization, and stats.json is the single source — loud failure
    when a dataset was never backfilled). None (printed) when no episode
    is long enough for a single window."""
    episodes = episode_actions(dataset_dir)
    entry = quantile_entry_from_stats(dataset_dir / "meta" / "stats.json")
    windows: list[FloatArray] = []
    for actions in episodes:
        windows.extend(
            actions[start : start + chunk_size]
            for start in range(0, actions.shape[0] - chunk_size + 1, stride)
        )
    if not windows:
        print(f"  - {repo_id} (no episode reaches {chunk_size} frames)", flush=True)
        return None
    stacked = np.stack(windows)
    overflow = entry.normalized_overflow(stacked)
    if overflow:
        fraction = overflow / stacked.size
        print(
            f"  - {repo_id}: clipped {overflow} outlier values "
            f"({fraction:.5%}) beyond the normalized range",
            flush=True,
        )
    normalized = entry.normalize(stacked)
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
