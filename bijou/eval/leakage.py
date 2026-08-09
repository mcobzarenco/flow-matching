"""Panel-leakage checker: training selection ∩ panel episodes = ∅.

Panel holdout episodes must never be trained on: nothing on
the holdout side of the panel's split may ever be trained on. The trap
is DERIVED corpora — the episode holdout hashes ``(split_seed,
repo_id)`` and draws from ``total_episodes``, so a filtered/merged/
renamed corpus draws a DIFFERENT split and silently moves former
holdout episodes into training. This checker certifies a corpus
against a frozen plan BEFORE any training touches it; its verdict
line is cited in the corpus's fit report.

What is checked, conservatively:

1. The radioactive set — the FULL holdout side of the panel's split,
   recomputed from the plan's recorded ``(holdout_episodes,
   split_seed)`` over every dataset of the panel corpus (not just the
   episodes the plan sampled). The plan's own episodes must be a
   subset (instrument self-check; a violation means the plan and the
   panel corpus disagree — stop).
2. Every episode the training corpus could train on, resolved to
   PANEL-corpus coordinates ``(source repo_id, source episode)``:

   - a dataset with a ``meta/source_provenance.json`` (the derived-
     corpus contract, below) maps through its recorded provenance;
   - a dataset whose repo id exists in the panel corpus maps
     identically — and the identity claim is VERIFIED, not assumed:
     episode counts must match and, when either side ships per-episode
     length metadata (``meta/episodes.jsonl`` or ``meta/episodes/``
     parquet), the per-episode length sequences must be identical. A
     filtered-and-renumbered corpus that kept its repo id would
     otherwise map through the identity to a false PASS while
     radioactive panel content trains (deep-dive finding 6b);
   - a dataset with neither is an ERROR — unattributable episodes
     cannot be certified (fail loud, never assume disjoint).

   Selection filters (``--fps``/``--camera-counts``) are deliberately
   ignored: they only shrink the training set, so certifying the
   superset is safe and keeps this checker filter-agnostic. When
   ``--holdout-episodes``/``--split-seed`` are given, only the TRAIN
   side of the training run's own split is checked (the run's holdout
   is never trained on); omitted, ALL episodes are checked.

Derived-corpus provenance contract (``meta/source_provenance.json``,
written by whatever builds the corpus):

    {"version": 1,
     "episodes": [{"episode_index": 0,
                   "source_repo_id": "user/dataset",
                   "source_episode_index": 17}, ...]}

Every episode of the derived dataset appears exactly once;
``source_repo_id`` uses panel-corpus ids. CPU-only, metadata-only
(reads ``meta/*.json``), minutes on the full corpus.

Usage:

    uv run python -m bijou.eval.leakage \\
        --plan plans/holdout_curated_v0_k4l2.json \\
        --panel-data ~/datasets/mcobzarenco/community_curated_v0 \\
        --train-data <corpus dir(s)> \\
        [--holdout-episodes 0.1 --split-seed 0]

Exit 0 with ``LEAKAGE CHECK PASSED`` iff the intersection is empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ..data import DatasetInfo, discover_datasets, holdout_episodes, repo_id_of
from .plan import SamplePlan

PROVENANCE_VERSION = 1
PROVENANCE_FILE = "source_provenance.json"


@dataclass(frozen=True, slots=True)
class Episode:
    """One episode in panel-corpus coordinates."""

    repo_id: str
    episode_index: int


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Outcome of a corpus certification against one frozen plan."""

    radioactive: frozenset[Episode]
    checked: frozenset[Episode]
    leaked: frozenset[Episode]
    unattributable: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return len(self.leaked) == 0 and len(self.unattributable) == 0


def radioactive_episodes(
    plan: SamplePlan,
    panel_data: tuple[Path, ...],
) -> frozenset[Episode]:
    """The full holdout side of the plan's split over the panel corpus.

    Recomputed from ``(plan.holdout_episodes, plan.split_seed)`` over
    every discovered dataset — a superset of the episodes the plan
    sampled, because unsampled holdout episodes are just as radioactive
    (the split, not the plan draw, defines what eval may ever see).
    """
    episodes: set[Episode] = set()
    for dataset_dir in discover_datasets(panel_data, exclude=()):
        repo_id = repo_id_of(dataset_dir)
        info = DatasetInfo.from_json(dataset_dir / "meta" / "info.json")
        for index in holdout_episodes(
            repo_id,
            info.total_episodes,
            plan.holdout_episodes,
            plan.split_seed,
        ):
            episodes.add(Episode(repo_id=repo_id, episode_index=index))
    return frozenset(episodes)


def plan_episodes(plan: SamplePlan) -> frozenset[Episode]:
    """Every (repo_id, episode) the plan references (core + labeled)."""
    return frozenset(
        Episode(repo_id=frame.repo_id, episode_index=frame.episode_index)
        for frame in (*plan.core, *plan.labeled)
    )


def _provenance_episodes(path: Path, repo_id: str) -> list[Episode]:
    """Parse a derived dataset's source_provenance.json (loud on any
    shape violation — an uncertifiable corpus must not pass silently)."""
    data = json.loads(path.read_text())
    version = data.get("version")
    if version != PROVENANCE_VERSION:
        raise SystemExit(
            f"{repo_id}: provenance version {version!r} != supported "
            f"{PROVENANCE_VERSION} ({path})",
        )
    records = data.get("episodes")
    if not isinstance(records, list) or len(records) == 0:
        raise SystemExit(f"{repo_id}: provenance 'episodes' missing/empty ({path})")
    return [
        Episode(
            repo_id=str(record["source_repo_id"]),
            episode_index=int(record["source_episode_index"]),
        )
        for record in records
    ]


def _episode_lengths(dataset_dir: Path) -> dict[int, int] | None:
    """Per-episode frame lengths from a dataset's metadata, or None when
    the dataset ships neither ``meta/episodes.jsonl`` (v2 layout) nor a
    ``meta/episodes/`` parquet tree (v3 layout)."""
    jsonl_path = dataset_dir / "meta" / "episodes.jsonl"
    if jsonl_path.exists():
        lengths: dict[int, int] = {}
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            lengths[int(record["episode_index"])] = int(record["length"])
        return lengths
    parquet_dir = dataset_dir / "meta" / "episodes"
    if parquet_dir.is_dir():
        import pyarrow.parquet as pq

        lengths = {}
        for path in sorted(parquet_dir.rglob("*.parquet")):
            table = pq.read_table(path, columns=["episode_index", "length"])
            for index, length in zip(
                table.column("episode_index").to_pylist(),
                table.column("length").to_pylist(),
                strict=True,
            ):
                lengths[int(index)] = int(length)
        return lengths
    return None


def _assert_identity(dataset_dir: Path, panel_dir: Path, repo_id: str) -> None:
    """Verify a same-repo-id training dataset really is the panel dataset
    (finding 6b: the identity claim was previously unchecked — a
    filtered/renumbered corpus keeping its repo id got a false PASS)."""
    if dataset_dir.resolve() == panel_dir.resolve():
        return  # literally the same directory
    train_info = DatasetInfo.from_json(dataset_dir / "meta" / "info.json")
    panel_info = DatasetInfo.from_json(panel_dir / "meta" / "info.json")
    if train_info.total_episodes != panel_info.total_episodes:
        raise SystemExit(
            f"{repo_id}: training copy declares {train_info.total_episodes} "
            f"episodes, panel corpus has {panel_info.total_episodes} — the "
            f"identity mapping is invalid; a filtered/renumbered corpus must "
            f"ship meta/{PROVENANCE_FILE}",
        )
    train_lengths = _episode_lengths(dataset_dir)
    panel_lengths = _episode_lengths(panel_dir)
    if train_lengths is None and panel_lengths is None:
        return  # no length metadata on either side; count check stands alone
    if train_lengths is None or panel_lengths is None:
        missing = dataset_dir if train_lengths is None else panel_dir
        raise SystemExit(
            f"{repo_id}: episode-length metadata present on one side only "
            f"(missing under {missing}/meta) — identity cannot be verified; "
            f"mirror the metadata or ship meta/{PROVENANCE_FILE}",
        )
    if train_lengths != panel_lengths:
        differing = sorted(
            index
            for index in train_lengths.keys() | panel_lengths.keys()
            if train_lengths.get(index) != panel_lengths.get(index)
        )
        raise SystemExit(
            f"{repo_id}: per-episode lengths differ from the panel corpus at "
            f"{len(differing)} episode(s) (e.g. {differing[:5]}) — the "
            f"identity mapping is invalid; a re-encoded/renumbered corpus "
            f"must ship meta/{PROVENANCE_FILE}",
        )


def trainable_episodes(
    train_data: tuple[Path, ...],
    panel_datasets: dict[str, Path],
    holdout_fraction: float,
    split_seed: int,
) -> tuple[frozenset[Episode], tuple[str, ...]]:
    """Every episode a run on ``train_data`` could train on, in panel
    coordinates, plus the repo ids that could not be attributed.

    ``holdout_fraction > 0`` restricts each dataset to the TRAIN side
    of the training run's OWN split (computed on the training corpus's
    repo id and episode count — exactly where a derived corpus's split
    diverges from the panel's).
    """
    checked: set[Episode] = set()
    unattributable: list[str] = []
    for dataset_dir in discover_datasets(train_data, exclude=()):
        repo_id = repo_id_of(dataset_dir)
        info = DatasetInfo.from_json(dataset_dir / "meta" / "info.json")
        held_out = frozenset(
            holdout_episodes(
                repo_id,
                info.total_episodes,
                holdout_fraction,
                split_seed,
            ),
        )
        trained_indices = [
            index for index in range(info.total_episodes) if index not in held_out
        ]
        provenance_path = dataset_dir / "meta" / PROVENANCE_FILE
        if provenance_path.exists():
            mapped = _provenance_episodes(provenance_path, repo_id)
            if len(mapped) != info.total_episodes:
                raise SystemExit(
                    f"{repo_id}: provenance maps {len(mapped)} episodes, "
                    f"info.json declares {info.total_episodes}",
                )
            checked.update(mapped[index] for index in trained_indices)
        elif repo_id in panel_datasets:
            _assert_identity(dataset_dir, panel_datasets[repo_id], repo_id)
            checked.update(
                Episode(repo_id=repo_id, episode_index=index)
                for index in trained_indices
            )
        else:
            unattributable.append(repo_id)
    return frozenset(checked), tuple(unattributable)


def check_leakage(
    plan_path: Path,
    panel_data: tuple[Path, ...],
    train_data: tuple[Path, ...],
    holdout_fraction: float,
    split_seed: int,
) -> LeakageReport:
    """Certify ``train_data`` against the frozen plan at ``plan_path``."""
    plan = SamplePlan.load(plan_path)
    radioactive = radioactive_episodes(plan, panel_data)
    referenced = plan_episodes(plan)
    stray = referenced - radioactive
    if len(stray) > 0:
        sample = sorted(stray, key=lambda e: (e.repo_id, e.episode_index))[:5]
        raise SystemExit(
            f"plan {plan_path} references {len(stray)} episode(s) OUTSIDE "
            f"the recomputed holdout side (e.g. {sample}) — the plan and "
            "the panel corpus disagree; do not train, diagnose first",
        )
    panel_datasets = {
        repo_id_of(d): d for d in discover_datasets(panel_data, exclude=())
    }
    checked, unattributable = trainable_episodes(
        train_data,
        panel_datasets,
        holdout_fraction,
        split_seed,
    )
    return LeakageReport(
        radioactive=radioactive,
        checked=checked,
        leaked=frozenset(radioactive & checked),
        unattributable=unattributable,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="panel-leakage checker: training selection ∩ panel = ∅",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="frozen panel plan JSON (plans/*.json)",
    )
    parser.add_argument(
        "--panel-data",
        type=Path,
        nargs="+",
        required=True,
        help="the corpus the plan was built on (collection roots)",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        nargs="+",
        required=True,
        help="the corpus a training run would load (collection roots)",
    )
    parser.add_argument(
        "--holdout-episodes",
        type=float,
        default=0.0,
        help="the TRAINING run's own holdout fraction (0 = the run "
        "trains on every episode; the conservative default)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="the TRAINING run's own split seed (with --holdout-episodes)",
    )
    args = parser.parse_args()

    report = check_leakage(
        plan_path=args.plan,
        panel_data=tuple(args.panel_data),
        train_data=tuple(args.train_data),
        holdout_fraction=args.holdout_episodes,
        split_seed=args.split_seed,
    )
    print(
        f"radioactive episodes (full holdout side): {len(report.radioactive)}",
    )
    print(f"training episodes checked: {len(report.checked)}")
    for repo_id in report.unattributable:
        print(
            f"  UNATTRIBUTABLE: {repo_id} — not in the panel corpus and no "
            f"meta/{PROVENANCE_FILE}; cannot certify",
        )
    if len(report.leaked) > 0:
        sample = sorted(
            report.leaked,
            key=lambda e: (e.repo_id, e.episode_index),
        )[:20]
        print(f"  LEAKED: {len(report.leaked)} panel episode(s) in training:")
        for episode in sample:
            print(f"    {episode.repo_id} episode {episode.episode_index}")
    print(
        "LEAKAGE CHECK PASSED"
        if report.passed
        else "LEAKAGE CHECK FAILED (see lines above)",
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
