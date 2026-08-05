"""LengthBucketedBatchSampler: determinism, coverage, grouping, DDP
slicing. Keys are laid out in per-dataset blocks (the shape
``length_bucket_keys`` produces: every frame of a dataset shares its
effective camera count)."""

import pytest

from bijou.data import LengthBucketedBatchSampler

# 800 frames across four "datasets": camera counts 1/2/3 mixed so a
# global shuffle interleaves keys and grouping has real work to do.
KEYS = [2] * 300 + [3] * 250 + [1] * 130 + [2] * 120
BATCH = 8


def batches_of(sampler: LengthBucketedBatchSampler) -> list[list[int]]:
    return list(iter(sampler))


def test_full_batches_cover_every_index_exactly_once() -> None:
    # 800 = 100 full batches of 8: nothing dropped at this geometry.
    sampler = LengthBucketedBatchSampler(KEYS, batch_size=BATCH, seed=0)
    batches = batches_of(sampler)
    assert len(batches) == len(sampler) == 100
    assert all(len(b) == BATCH for b in batches)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(len(KEYS)))


def test_deterministic_within_epoch_reshuffled_across() -> None:
    a = LengthBucketedBatchSampler(KEYS, batch_size=BATCH, seed=0)
    b = LengthBucketedBatchSampler(KEYS, batch_size=BATCH, seed=0)
    assert batches_of(a) == batches_of(b)
    assert batches_of(a) == batches_of(a)  # iter is repeatable
    b.set_epoch(1)
    assert batches_of(a) != batches_of(b)
    c = LengthBucketedBatchSampler(KEYS, batch_size=BATCH, seed=1)
    assert batches_of(a) != batches_of(c)


def test_batches_are_mostly_key_homogeneous() -> None:
    # Each megabatch is stable-sorted by key, so mixed batches occur
    # only at key boundaries: <= (distinct keys - 1) per megabatch.
    # This geometry (800 samples, megabatch 512, 3 keys) admits <= 4
    # mixed batches of 100; without grouping ~every batch mixes.
    sampler = LengthBucketedBatchSampler(KEYS, batch_size=BATCH, seed=0)
    homogeneous = sum(
        len({KEYS[i] for i in batch}) == 1 for batch in batches_of(sampler)
    )
    assert homogeneous >= 96


def test_ddp_ranks_partition_the_global_list() -> None:
    ranks = [
        LengthBucketedBatchSampler(
            KEYS,
            batch_size=BATCH,
            seed=0,
            rank=rank,
            world_size=2,
        )
        for rank in (0, 1)
    ]
    per_rank = [batches_of(s) for s in ranks]
    assert len(per_rank[0]) == len(per_rank[1]) == len(ranks[0]) == 50
    seen = [i for batches in per_rank for b in batches for i in b]
    assert sorted(seen) == list(range(len(KEYS)))  # disjoint + complete


def test_tail_and_truncation_drops_are_bounded() -> None:
    # 803 samples, world 2: the final megabatch's sub-batch tail (3)
    # drops, and the odd 100th batch drops to equalize ranks.
    keys = [*KEYS, 1, 1, 1]
    ranks = [
        LengthBucketedBatchSampler(
            keys,
            batch_size=BATCH,
            seed=0,
            rank=rank,
            world_size=2,
        )
        for rank in (0, 1)
    ]
    seen = [i for s in ranks for b in batches_of(s) for i in b]
    assert len(seen) == len(set(seen)) == 2 * 50 * BATCH


def test_rejects_degenerate_geometry() -> None:
    with pytest.raises(ValueError, match="fewer than"):
        LengthBucketedBatchSampler([1] * 7, batch_size=8, seed=0)
    with pytest.raises(ValueError, match="outside world"):
        LengthBucketedBatchSampler(KEYS, batch_size=8, seed=0, rank=2, world_size=2)
