"""Stratified eval sample plans (``--sample-plan``).

Uniform-over-frames sampling has two defects on a corpus of ragged
episodes: long episodes dominate (frame count ∝ duration), and
judge-labeled frames are so rare in the draw that aux metrics starve
(n=365 of 16,384 on the rcond-100k eval). A sample plan replaces the
draw with per-episode stratification:

- **core panel**: K frames per episode, drawn uniformly per episode.
  Headline chunk metrics aggregate over EXACTLY these frames — with K
  constant per episode the frame mean IS the episode mean (episodes
  shorter than K contribute what they have).
- **labeled panel**: up to L additional judge-labeled frames per
  episode (finite ``annotation.progress`` — the judge's sampled-frame
  mask), disjoint from the core picks. Scored by every policy like any
  frame but EXCLUDED from headline aggregation (they oversample judged
  frames); they exist to feed the aux metrics (holding/progress/event/
  visibility vs the weak labels) and the Q3 sensitivity diagnostic
  with a usable n.

Determinism: every draw derives from ``Random(f"{plan_seed}:{repo_id}:
{episode}")`` — a pure function of the plan, independent of dataset
order, machine, and the eval ``--seed`` (which keeps governing policy
noise only). Frames are stored as (repo_id, episode_index,
frame_index) triples so the plan survives re-selection; resolution
against a selection FAILS LOUDLY if any referenced episode is missing
(a plan is a frozen panel — silently scoring a subset would unpair
every cross-checkpoint comparison).

The plan is a JSON artifact next to the reports: build once (the CLI
builds and writes it when the file does not exist), commit or archive
it, and every later checkpoint scores the identical frames — deltas
between runs become paired comparisons.
"""

from __future__ import annotations

import datetime
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data import DataSelection

PLAN_VERSION = 1
# v2 plans (fontaine/scripts/panel_v2.py) carry the same row payload +
# selection-filter provenance as v1 and add exclusions metadata this
# loader never reads; the rows parse identically. Writes stay v1.
SUPPORTED_PLAN_VERSIONS = (1, 2)


@dataclass(frozen=True, slots=True)
class PlanFrame:
    """One planned frame, in dataset-absolute coordinates (stable under
    episode-split filtering and selection changes)."""

    repo_id: str
    episode_index: int
    frame_index: int


@dataclass(frozen=True, slots=True)
class SamplePlan:
    """A frozen eval panel plus the provenance needed to validate it.

    ``episodes``/``holdout_episodes``/``split_seed``/``fps``/
    ``camera_counts`` echo the selection filters the plan was built
    under; loading validates them against the invocation (scoring a
    plan under different filters silently redefines the panel).
    """

    plan_seed: int
    frames_per_episode: int
    labeled_per_episode: int
    episodes: str
    holdout_episodes: float
    split_seed: int
    fps: list[float] | None
    camera_counts: list[int] | None
    created_at: str
    core: list[PlanFrame]
    labeled: list[PlanFrame]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": PLAN_VERSION,
            "plan_seed": self.plan_seed,
            "frames_per_episode": self.frames_per_episode,
            "labeled_per_episode": self.labeled_per_episode,
            "episodes": self.episodes,
            "holdout_episodes": self.holdout_episodes,
            "split_seed": self.split_seed,
            "fps": self.fps,
            "camera_counts": self.camera_counts,
            "created_at": self.created_at,
            "core": [[f.repo_id, f.episode_index, f.frame_index] for f in self.core],
            "labeled": [
                [f.repo_id, f.episode_index, f.frame_index] for f in self.labeled
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplePlan:
        """Validating parse at the JSON edge ("parse, don't validate")."""
        version = data["version"]
        if version not in SUPPORTED_PLAN_VERSIONS:
            raise ValueError(
                f"sample plan version {version} != supported {SUPPORTED_PLAN_VERSIONS}",
            )

        def frames(key: str) -> list[PlanFrame]:
            return [
                PlanFrame(
                    repo_id=str(repo_id),
                    episode_index=int(episode),
                    frame_index=int(frame),
                )
                for repo_id, episode, frame in data[key]
            ]

        fps = data["fps"]
        camera_counts = data["camera_counts"]
        return cls(
            plan_seed=int(data["plan_seed"]),
            frames_per_episode=int(data["frames_per_episode"]),
            labeled_per_episode=int(data["labeled_per_episode"]),
            episodes=str(data["episodes"]),
            holdout_episodes=float(data["holdout_episodes"]),
            split_seed=int(data["split_seed"]),
            fps=[float(f) for f in fps] if fps is not None else None,
            camera_counts=(
                [int(c) for c in camera_counts] if camera_counts is not None else None
            ),
            created_at=str(data["created_at"]),
            core=frames("core"),
            labeled=frames("labeled"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path) -> SamplePlan:
        return cls.from_dict(json.loads(path.read_text()))


@dataclass(frozen=True, slots=True)
class EpisodeTable:
    """One dataset's episode layout, scanned once from the arrow table
    (column reads only, no video decode). ``labeled`` is None when the
    dataset carries no materialized ``annotation.progress`` column
    (never judged); otherwise the judge's sampled-frame mask."""

    offset: int  # global concat offset of the dataset's first row
    length: int
    episode_ids: np.ndarray
    starts: np.ndarray  # dataset-local first-row offset per episode
    labeled: np.ndarray | None


def episode_tables(selection: DataSelection) -> dict[str, EpisodeTable]:
    """Scan every dataset's episode layout ONCE — shared by build and
    resolve (the scan reads two columns of ~900 arrow tables and is the
    dominant cost; measured ~8 min on curated-v0, so it must not run
    twice). ``with_format("numpy")`` bypasses the per-row torch
    formatter for whole-column reads."""
    tables: dict[str, EpisodeTable] = {}
    offset = 0
    for dataset in selection.datasets:
        repo_id = str(dataset.dataset.repo_id)
        table = dataset.dataset.hf_dataset.with_format("numpy")
        episode_column = np.asarray(table["episode_index"])
        episode_ids, starts = np.unique(episode_column, return_index=True)
        assert bool(np.all(np.diff(starts) > 0)), (
            f"{repo_id}: episode rows not grouped in ascending order"
        )
        labeled: np.ndarray | None = None
        if "annotation.progress" in table.column_names:
            progress = np.asarray(table["annotation.progress"], dtype=np.float32)
            labeled = np.isfinite(progress)
        tables[repo_id] = EpisodeTable(
            offset=offset,
            length=len(dataset),
            episode_ids=episode_ids,
            starts=starts,
            labeled=labeled,
        )
        offset += len(dataset)
    return tables


def build_plan(
    tables: dict[str, EpisodeTable],
    *,
    plan_seed: int,
    frames_per_episode: int,
    labeled_per_episode: int,
    episodes: str,
    holdout_episodes: float,
    split_seed: int,
    fps: list[float] | None,
    camera_counts: list[int] | None,
) -> SamplePlan:
    """Stratified panel over every episode of the scanned selection (see
    module docstring for the draw semantics)."""
    core: list[PlanFrame] = []
    labeled_frames: list[PlanFrame] = []
    for repo_id, table in tables.items():
        bounds = [*table.starts.tolist(), table.length]
        for position, episode in enumerate(table.episode_ids.tolist()):
            start, end = bounds[position], bounds[position + 1]
            length = end - start
            rng = random.Random(f"{plan_seed}:{repo_id}:{episode}")
            picks = sorted(rng.sample(range(length), min(frames_per_episode, length)))
            core.extend(
                PlanFrame(repo_id=repo_id, episode_index=episode, frame_index=frame)
                for frame in picks
            )
            if table.labeled is None or labeled_per_episode == 0:
                continue
            candidates = [
                frame
                for frame in np.flatnonzero(table.labeled[start:end]).tolist()
                if frame not in picks
            ]
            extra = sorted(
                rng.sample(candidates, min(labeled_per_episode, len(candidates))),
            )
            labeled_frames.extend(
                PlanFrame(repo_id=repo_id, episode_index=episode, frame_index=frame)
                for frame in extra
            )
    return SamplePlan(
        plan_seed=plan_seed,
        frames_per_episode=frames_per_episode,
        labeled_per_episode=labeled_per_episode,
        episodes=episodes,
        holdout_episodes=holdout_episodes,
        split_seed=split_seed,
        fps=fps,
        camera_counts=camera_counts,
        created_at=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        core=core,
        labeled=labeled_frames,
    )


def validate_plan(
    plan: SamplePlan,
    *,
    episodes: str,
    holdout_episodes: float,
    split_seed: int,
    fps: list[float] | None,
    camera_counts: list[int] | None,
) -> None:
    """Filters at scoring time must equal the filters the plan was built
    under — anything else silently redefines the panel."""
    expected = {
        "episodes": (plan.episodes, episodes),
        "holdout_episodes": (plan.holdout_episodes, holdout_episodes),
        "split_seed": (plan.split_seed, split_seed),
        "fps": (plan.fps, fps),
        "camera_counts": (plan.camera_counts, camera_counts),
    }
    mismatched = {
        name: (planned, given)
        for name, (planned, given) in expected.items()
        if planned != given
    }
    if mismatched:
        raise SystemExit(
            f"sample plan was built under different selection filters: "
            f"{mismatched} (plan value first) — rerun with the plan's "
            "filters or build a new plan",
        )


def resolve_plan(
    plan: SamplePlan,
    tables: dict[str, EpisodeTable],
) -> tuple[list[int], set[int]]:
    """(sorted global concat indices of every planned frame, the subset
    that is the core panel). Fails loudly when the scanned selection is
    missing anything the plan references."""
    missing: list[str] = []

    def resolve(frame: PlanFrame) -> int | None:
        table = tables.get(frame.repo_id)
        if table is None:
            missing.append(f"{frame.repo_id} (dataset absent)")
            return None
        position = int(np.searchsorted(table.episode_ids, frame.episode_index))
        if (
            position >= len(table.episode_ids)
            or table.episode_ids[position] != frame.episode_index
        ):
            missing.append(f"{frame.repo_id} episode {frame.episode_index}")
            return None
        start = int(table.starts[position])
        end = (
            int(table.starts[position + 1])
            if position + 1 < len(table.starts)
            else table.length
        )
        # A truncated/re-encoded episode must not silently score its
        # neighbour's rows: a planned frame beyond the episode's end is
        # missing, not offset arithmetic.
        if not 0 <= frame.frame_index < end - start:
            missing.append(
                f"{frame.repo_id} episode {frame.episode_index} frame "
                f"{frame.frame_index} (episode has {end - start} rows)",
            )
            return None
        return table.offset + start + frame.frame_index

    core = [resolve(frame) for frame in plan.core]
    labeled = [resolve(frame) for frame in plan.labeled]
    if missing:
        unique = sorted(set(missing))
        raise SystemExit(
            f"sample plan references {len(unique)} episode(s)/dataset(s) "
            f"missing from the selection (first 10: {unique[:10]}) — the "
            "panel must score complete or not at all",
        )
    core_set = {index for index in core if index is not None}
    indices = sorted(core_set | {index for index in labeled if index is not None})
    return indices, core_set
