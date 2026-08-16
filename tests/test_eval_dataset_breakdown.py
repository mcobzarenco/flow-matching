"""--eval-dataset-breakdown oracles (owner work order 2026-08-16):
flag plumbing through TrainArgs, and the probe set's per-dataset
labels — rank-identical bucket order, per-item repo ids, tolerant of
synthetic datasets that carry no repo id."""

from types import SimpleNamespace
from typing import Any, override

import torch
from test_train_args import _parse

from bijou.train.loop import ProbeSet, build_probe_set


def test_flag_defaults_off_and_parses_on() -> None:
    assert _parse(["--family", "gemma_flow"]).eval_dataset_breakdown is False
    args = _parse(["--family", "gemma_flow", "--eval-dataset-breakdown"])
    assert args.eval_dataset_breakdown is True


def test_probe_set_default_fields_stay_compatible() -> None:
    # Direct construction predating the feature (no repo fields) must
    # keep working — the breakdown quietly disables on empty buckets.
    probe = ProbeSet(
        total=0,
        batches=[],
        rich_items=[],
        rich_positions=(),
        outcomes=(),
    )
    assert probe.repo_ids == ()
    assert probe.repo_buckets == ()


class _Member(torch.utils.data.Dataset[dict[str, Any]]):
    """A StatsAttachedDataset stand-in: items carry repo_id, the
    wrapped `.dataset` exposes it (the bucket-derivation surface)."""

    def __init__(self, repo_id: str, n: int) -> None:
        self.dataset = SimpleNamespace(repo_id=repo_id)
        self.repo_id = repo_id
        self.n = n

    def __len__(self) -> int:
        return self.n

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"repo_id": self.repo_id, "index": index}


def test_build_probe_set_labels_and_buckets() -> None:
    # Repeated member (the --dataset-repeat shape) dedupes in buckets;
    # bucket order is sorted, independent of member order.
    concat = torch.utils.data.ConcatDataset(
        [_Member("z/rig", 6), _Member("a/demo", 10), _Member("z/rig", 6)],
    )
    probe = build_probe_set(
        concat,
        lambda items: items,  # type: ignore[arg-type] — batches opaque here
        num_samples=12,
        seed=0,
        rank=0,
        world_size=1,
        batch_size=4,
        keep_rich=False,
    )
    assert probe.repo_buckets == ("a/demo", "z/rig")
    assert len(probe.repo_ids) == 12
    assert set(probe.repo_ids) <= {"a/demo", "z/rig"}
    # Striped shards partition the labels: two ranks see disjoint items
    # but the SAME bucket order (the collective-alignment invariant).
    shard1 = build_probe_set(
        concat,
        lambda items: items,  # type: ignore[arg-type]
        num_samples=12,
        seed=0,
        rank=1,
        world_size=2,
        batch_size=4,
        keep_rich=False,
    )
    assert shard1.repo_buckets == probe.repo_buckets


def test_build_probe_set_without_repo_ids_disables_cleanly() -> None:
    class Bare(torch.utils.data.Dataset[dict[str, Any]]):
        def __len__(self) -> int:
            return 4

        @override
        def __getitem__(self, index: int) -> dict[str, Any]:
            return {"index": index}

    probe = build_probe_set(
        torch.utils.data.ConcatDataset([Bare()]),
        lambda items: items,  # type: ignore[arg-type]
        num_samples=4,
        seed=0,
        rank=0,
        world_size=1,
        batch_size=2,
        keep_rich=False,
    )
    assert probe.repo_buckets == ()
    assert probe.repo_ids == ("", "", "", "")
