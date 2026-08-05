"""Metadata-only padding simulation for --bucket-by-length (idea #2).

Reads each dataset's meta/info.json (camera count, frame count) — no
video, no parquet — and compares padded-prompt-token cost per epoch
under (a) global-shuffle batching and (b) LengthBucketedBatchSampler,
using the real sampler. Prompt-length model per frame:
``n_cams * (max_soft_tokens + TAG_TOKENS) + TEXT_TOKENS`` — text is
approximated as a constant, so within-batch text jitter (tens of
tokens, varying per draw) is NOT simulated; the simulated saving is an
upper bound on the padding recovered by camera-count grouping alone.

Usage: uv run python -m fontaine.scripts.bucketing_padding_sim \
    ~/datasets/mcobzarenco/community_curated_v0 --batch-size 10
"""

import argparse
import json
from pathlib import Path

import torch

from bijou.data import LengthBucketedBatchSampler

# Rough prompt constants: the camera tag line ("[kind camera|" + name)
# and the instruction/state/condition text. Only the RELATIVE inflation
# matters; both terms shift padded and true cost together.
TAG_TOKENS = 8
TEXT_TOKENS = 60


def padded_cost(batches: list[list[int]], lengths: torch.Tensor) -> int:
    return int(
        sum(len(b) * int(lengths[b].max()) for b in batches),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-soft-tokens", type=int, default=140)
    parser.add_argument("--seed", type=int, default=0)
    # Mirror the train-selection guards that shape the length census
    # (dataset-level fps / camera-count filters; the finer guards —
    # bespoke features, stats, holdout split — move few frames).
    parser.add_argument("--fps", type=float, nargs="*", default=None)
    parser.add_argument("--camera-counts", type=int, nargs="*", default=None)
    args = parser.parse_args()

    keys: list[int] = []
    for info_path in sorted(args.root.glob("*/*/meta/info.json")):
        info = json.loads(info_path.read_text())
        features = info.get("features") or {}
        cams = sum(
            1 for ft in features.values() if ft.get("dtype") in ("video", "image")
        )
        if not cams:
            continue
        if args.fps is not None and float(info["fps"]) not in args.fps:
            continue
        if args.camera_counts is not None and cams not in args.camera_counts:
            continue
        keys.extend([cams] * int(info["total_frames"]))
    per_camera = args.max_soft_tokens + TAG_TOKENS
    lengths = torch.tensor(keys, dtype=torch.long) * per_camera + TEXT_TOKENS

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(keys), generator=generator)
    shuffled = [
        perm[i : i + args.batch_size].tolist()
        for i in range(0, len(perm) - args.batch_size + 1, args.batch_size)
    ]
    sampler = LengthBucketedBatchSampler(
        keys,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    bucketed = list(iter(sampler))

    census: dict[int, int] = {}
    for key in keys:
        census[key] = census.get(key, 0) + 1
    print(f"frames {len(keys)}, camera-count census {dict(sorted(census.items()))}")
    for name, batches in [("shuffled", shuffled), ("bucketed", bucketed)]:
        cost = padded_cost(batches, lengths)
        true = int(lengths[[i for b in batches for i in b]].sum())
        print(
            f"{name}: {len(batches)} batches of {args.batch_size}, "
            f"padded prompt-tokens/epoch {cost:,} "
            f"(inflation {cost / true - 1:+.2%} over true {true:,})",
        )


if __name__ == "__main__":
    main()
