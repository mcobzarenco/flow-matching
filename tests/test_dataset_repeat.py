"""--dataset-repeat oracle: spec parsing, resolution precedence, loud
no-match failure, replica expansion, and the er-60k mixture-note
arithmetic the flag exists to realize (a 0.19%-natural-share dataset
lifted to ~5% effective share by a 27x repeat).

Pure-function surface (parse_repeat_specs / resolve_repeats /
repeat_datasets) — no real LeRobot data; expansion is checked through
the same ConcatDataset the trainer consumes.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from bijou.data import (
    DatasetStats,
    StatsAttachedDataset,
    parse_repeat_specs,
    repeat_datasets,
    resolve_repeats,
)

DIM = 6


class StubDataset:
    """Duck-typed LeRobotDataset: repo id + fixed length."""

    def __init__(self, repo_id: str, length: int) -> None:
        self.repo_id = repo_id
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"index": index, "episode_index": torch.tensor(0)}


def wrap(repo_id: str, length: int) -> StatsAttachedDataset:
    stats = DatasetStats(
        action_mean=(0.0,) * DIM,
        action_std=(1.0,) * DIM,
        state_mean=(0.0,) * DIM,
        state_std=(1.0,) * DIM,
        action_q01=(-1.0,) * DIM,
        action_q99=(1.0,) * DIM,
        state_q01=(-1.0,) * DIM,
        state_q99=(1.0,) * DIM,
    )
    # cast: the stub duck-types the LeRobotDataset surface
    # StatsAttachedDataset touches (repo_id, __len__, __getitem__).
    return StatsAttachedDataset(
        cast(LeRobotDataset, StubDataset(repo_id, length)),
        stats,
        {},
        {},
    )


# -- parse_repeat_specs ---------------------------------------------------


def test_parse_valid_specs() -> None:
    assert parse_repeat_specs(("mcobzarenco/*=27", "alice/cubes=1")) == (
        ("mcobzarenco/*", 27),
        ("alice/cubes", 1),
    )


def test_parse_pattern_may_contain_equals() -> None:
    # rpartition: the LAST '=' splits, so patterns with '=' survive.
    assert parse_repeat_specs(("weird=name=3",)) == (("weird=name", 3),)


@pytest.mark.parametrize(
    "spec",
    ["norepeat", "=5", "user/data=", "user/data=2.5", "user/data=zero"],
)
def test_parse_malformed_spec_is_fatal(spec: str) -> None:
    with pytest.raises(ValueError, match="--dataset-repeat"):
        parse_repeat_specs((spec,))


def test_parse_count_below_one_is_fatal() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        parse_repeat_specs(("user/data=0",))


# -- resolve_repeats ------------------------------------------------------

REPOS = ("alice/cubes", "mcobzarenco/so101_a", "mcobzarenco/so101_b")


def test_resolve_maps_matching_repos() -> None:
    assert resolve_repeats((("mcobzarenco/*", 27),), REPOS) == {
        "mcobzarenco/so101_a": 27,
        "mcobzarenco/so101_b": 27,
    }


def test_resolve_first_matching_spec_wins() -> None:
    specs = (("mcobzarenco/so101_a", 9), ("mcobzarenco/*", 27))
    assert resolve_repeats(specs, REPOS) == {
        "mcobzarenco/so101_a": 9,
        "mcobzarenco/so101_b": 27,
    }


def test_resolve_count_one_is_identity() -> None:
    assert resolve_repeats((("alice/*", 1),), REPOS) == {}


def test_resolve_unmatched_pattern_is_fatal() -> None:
    # A silently unapplied oversample (typo, or the target dropped by a
    # filter) would corrupt the registered mixture.
    with pytest.raises(ValueError, match="matches no selected dataset"):
        resolve_repeats((("mcobzarenco/typo_*", 27),), REPOS)


def test_resolve_no_specs_is_identity() -> None:
    assert resolve_repeats((), REPOS) == {}


# -- repeat_datasets ------------------------------------------------------


def test_expansion_is_contiguous_and_shares_objects() -> None:
    passenger = wrap("alice/cubes", 10)
    rig = wrap("mcobzarenco/so101_a", 3)
    expanded = repeat_datasets([passenger, rig], {"mcobzarenco/so101_a": 4})
    assert [sub.dataset.repo_id for sub in expanded] == [
        "alice/cubes",
        *["mcobzarenco/so101_a"] * 4,
    ]
    # Replicas are the SAME object: no duplicated stats/memory, one shared
    # substitution counter.
    assert all(sub is rig for sub in expanded[1:])


def test_concat_sees_each_repeated_frame_count_times() -> None:
    passenger = wrap("alice/cubes", 10)
    rig = wrap("mcobzarenco/so101_a", 3)
    concat: torch.utils.data.ConcatDataset[dict[str, Any]] = (
        torch.utils.data.ConcatDataset(
            repeat_datasets([passenger, rig], {"mcobzarenco/so101_a": 4}),
        )
    )
    assert len(concat) == 10 + 4 * 3
    # Every replica indexes into the same underlying frames.
    repeated = [concat[10 + i]["index"] for i in range(12)]
    assert repeated == [0, 1, 2] * 4


# -- the er-60k mixture-note arithmetic -----------------------------------


def test_er60k_mixture_note_arithmetic() -> None:
    """The pinned lever: rig 36,078 frames of an 18.67M-frame corpus is
    0.19% natural share; a 27x repeat lands ~5% effective share — inside
    the CL-triangle 2-20% replay band."""
    rig, total, count = 36_078, 18_670_000, 27
    natural = rig / total
    effective = count * rig / (total + (count - 1) * rig)
    assert natural == pytest.approx(0.0019, abs=1e-4)
    assert effective == pytest.approx(0.05, abs=0.003)
    assert 0.02 < effective < 0.20
