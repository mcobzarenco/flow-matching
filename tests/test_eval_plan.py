"""Sample-plan contract: deterministic stratified build, JSON roundtrip,
loud resolution failure, filter validation, and the visibility parser.

Pure CPU/synthetic: a stub stands in for StatsAttachedDataset (the plan
builder touches only .dataset.repo_id, .dataset.hf_dataset column reads
and len()).
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from bijou.aux_text import parse_visibility
from bijou.data import DataSelection, StatsAttachedDataset
from bijou.eval.plan import (
    PlanFrame,
    SamplePlan,
    build_plan,
    episode_tables,
    resolve_plan,
    validate_plan,
)

NAN = math.nan


class StubTable:
    """Duck-typed hf_dataset: named columns, list-valued."""

    def __init__(self, columns: dict[str, list[float]]) -> None:
        self._columns = columns
        self.column_names = list(columns)

    def __getitem__(self, key: str) -> list[float]:
        return self._columns[key]

    def with_format(self, _format: str) -> StubTable:
        return self


class StubLeRobot:
    def __init__(self, repo_id: str, columns: dict[str, list[float]]) -> None:
        self.repo_id = repo_id
        self.hf_dataset = StubTable(columns)


class StubDataset:
    """Duck-typed StatsAttachedDataset for the plan builder."""

    def __init__(self, repo_id: str, columns: dict[str, list[float]]) -> None:
        self.dataset = StubLeRobot(repo_id, columns)

    def __len__(self) -> int:
        return len(self.dataset.hf_dataset["episode_index"])


def _selection(datasets: list[StubDataset]) -> DataSelection:
    # Only .datasets is touched by the plan code; the rest is inert.
    return cast(
        DataSelection,
        type("S", (), {"datasets": cast(list[StatsAttachedDataset], datasets)})(),
    )


def _stub_a() -> StubDataset:
    # Two episodes (5 + 3 frames); episode 0 has labeled frames 1 and 3.
    return StubDataset(
        "user/a",
        {
            "episode_index": [0, 0, 0, 0, 0, 1, 1, 1],
            "annotation.progress": [NAN, 0.2, NAN, 0.8, NAN, NAN, NAN, NAN],
        },
    )


def _stub_b() -> StubDataset:
    # One episode, no annotation column (never judged).
    return StubDataset("user/b", {"episode_index": [0, 0, 0, 0]})


def _build(selection: DataSelection) -> SamplePlan:
    return build_plan(
        episode_tables(selection),
        plan_seed=0,
        frames_per_episode=2,
        labeled_per_episode=2,
        episodes="holdout",
        holdout_episodes=0.1,
        split_seed=0,
        fps=[30.0],
        camera_counts=[1, 2],
    )


def test_build_is_deterministic_and_stratified(tmp_path: Path) -> None:
    selection = _selection([_stub_a(), _stub_b()])
    plan = _build(selection)
    again = _build(_selection([_stub_a(), _stub_b()]))
    assert plan.core == again.core
    assert plan.labeled == again.labeled

    # 2 core frames from every episode (3 episodes), all in range.
    by_episode: dict[tuple[str, int], int] = {}
    for frame in plan.core:
        by_episode[frame.repo_id, frame.episode_index] = (
            by_episode.get((frame.repo_id, frame.episode_index), 0) + 1
        )
    assert by_episode == {("user/a", 0): 2, ("user/a", 1): 2, ("user/b", 0): 2}

    # Labeled picks: only user/a episode 0 has labeled frames (1, 3),
    # and they never duplicate core picks.
    assert all(f.repo_id == "user/a" and f.episode_index == 0 for f in plan.labeled)
    assert all(f.frame_index in (1, 3) for f in plan.labeled)
    core_keys = {(f.repo_id, f.episode_index, f.frame_index) for f in plan.core}
    assert all(
        (f.repo_id, f.episode_index, f.frame_index) not in core_keys
        for f in plan.labeled
    )

    # Roundtrip through JSON.
    path = tmp_path / "plan.json"
    plan.save(path)
    assert SamplePlan.load(path) == plan


def test_v2_plan_loads_rows_identically_and_v3_refused(tmp_path: Path) -> None:
    # panel_v2.py plans: version 2 = the v1 payload + exclusions
    # metadata the loader never reads. Rows must parse identically to
    # v1 (first caught live 2026-08-06: the frozen panel-v2 plan was
    # unreadable by every SamplePlan consumer).
    plan = _build(_selection([_stub_a(), _stub_b()]))
    v2 = plan.to_dict()
    v2["version"] = 2
    v2["derived_from"] = "plan.json (v1, byte-identical rows)"
    v2["exclusions"] = {"leaked_episodes": [], "corrupt_repos": [], "counts": {}}
    path = tmp_path / "plan_v2.json"
    path.write_text(json.dumps(v2))
    assert SamplePlan.load(path) == plan

    v3 = dict(v2, version=3)
    path.write_text(json.dumps(v3))
    with pytest.raises(ValueError, match="sample plan version 3"):
        SamplePlan.load(path)


def test_resolve_maps_to_concat_indices_and_splits_core() -> None:
    selection = _selection([_stub_a(), _stub_b()])
    plan = _build(selection)
    indices, core = resolve_plan(plan, episode_tables(selection))
    assert indices == sorted(indices)
    assert core <= set(indices)
    # user/b's frames live at concat offset 8 (after user/a's 8 rows).
    b_frames = [f.frame_index + 8 for f in plan.core if f.repo_id == "user/b"]
    assert set(b_frames) <= core
    # Labeled panel = resolved indices minus core.
    assert len(indices) == len(core) + len(plan.labeled)


def test_resolve_fails_loudly_on_missing_episode() -> None:
    plan = _build(_selection([_stub_a(), _stub_b()]))
    with pytest.raises(SystemExit, match="missing from the selection"):
        resolve_plan(plan, episode_tables(_selection([_stub_a()])))  # user/b gone


def test_resolve_fails_loudly_on_out_of_range_frame() -> None:
    # user/a episode 0 has 5 rows; a plan built against a longer episode
    # (frame 5) must not silently score episode 1's first row — the
    # truncated/re-encoded-episode trap.
    plan = _build(_selection([_stub_a(), _stub_b()]))
    truncated = replace(
        plan,
        core=[PlanFrame(repo_id="user/a", episode_index=0, frame_index=5)],
        labeled=[],
    )
    with pytest.raises(SystemExit, match="missing from the selection"):
        resolve_plan(truncated, episode_tables(_selection([_stub_a(), _stub_b()])))


def test_validate_rejects_filter_mismatch() -> None:
    plan = _build(_selection([_stub_a()]))
    with pytest.raises(SystemExit, match="different selection filters"):
        validate_plan(
            plan,
            episodes="holdout",
            holdout_episodes=0.1,
            split_seed=0,
            fps=None,  # plan was built under fps=[30.0]
            camera_counts=[1, 2],
        )


def test_parse_visibility() -> None:
    assert parse_visibility("object 0,1; gripper none") == (
        frozenset({0, 1}),
        frozenset(),
    )
    assert parse_visibility("object 1,0; gripper 0") == (
        frozenset({0, 1}),
        frozenset({0}),
    )  # order-insensitive
    assert parse_visibility("object none; gripper none") == (frozenset(), frozenset())
    assert parse_visibility("garbage") is None
    assert parse_visibility("object x; gripper 0") is None
    assert parse_visibility("gripper 0; object 1") is None  # fixed order
