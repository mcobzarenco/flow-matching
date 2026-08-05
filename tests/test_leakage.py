"""Leakage checker: the derived-corpus split trap must be caught.

Synthetic corpora on tmp_path — a panel corpus, an identity training
corpus, a derived corpus whose provenance leaks a panel episode, and a
derived corpus with no provenance at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bijou.data import holdout_episodes
from bijou.eval.leakage import Episode, check_leakage

HOLDOUT_FRACTION = 0.5
SPLIT_SEED = 0


def _write_dataset(root: Path, repo_id: str, total_episodes: int) -> Path:
    dataset_dir = root / repo_id
    (dataset_dir / "meta").mkdir(parents=True)
    info = {
        "fps": 30,
        "total_episodes": total_episodes,
        "features": {
            "action": {"shape": [6], "names": ["a"] * 6},
            "observation.state": {"shape": [6]},
            "observation.images.top": {"dtype": "video"},
        },
    }
    (dataset_dir / "meta" / "info.json").write_text(json.dumps(info))
    return dataset_dir


def _write_plan(path: Path, core: list[list[object]]) -> None:
    plan = {
        "version": 1,
        "plan_seed": 0,
        "frames_per_episode": 1,
        "labeled_per_episode": 0,
        "episodes": "holdout",
        "holdout_episodes": HOLDOUT_FRACTION,
        "split_seed": SPLIT_SEED,
        "fps": None,
        "camera_counts": None,
        "created_at": "2026-08-05T00:00:00+00:00",
        "core": core,
        "labeled": [],
    }
    path.write_text(json.dumps(plan))


@pytest.fixture()
def panel(tmp_path: Path) -> tuple[Path, Path, Episode]:
    """A 2-dataset panel corpus + a plan sampling one holdout episode."""
    panel_root = tmp_path / "panel"
    _write_dataset(panel_root, "alice/cubes", 10)
    _write_dataset(panel_root, "bob/pegs", 4)
    held = holdout_episodes("alice/cubes", 10, HOLDOUT_FRACTION, SPLIT_SEED)
    radioactive = Episode(repo_id="alice/cubes", episode_index=held[0])
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, core=[["alice/cubes", held[0], 3]])
    return panel_root, plan_path, radioactive


def test_identity_corpus_with_matching_split_passes(
    panel: tuple[Path, Path, Episode],
) -> None:
    panel_root, plan_path, _ = panel
    report = check_leakage(
        plan_path=plan_path,
        panel_data=(panel_root,),
        train_data=(panel_root,),
        holdout_fraction=HOLDOUT_FRACTION,
        split_seed=SPLIT_SEED,
    )
    assert report.passed
    assert len(report.radioactive & report.checked) == 0


def test_identity_corpus_training_on_everything_fails(
    panel: tuple[Path, Path, Episode],
) -> None:
    """holdout_fraction=0 trains on every episode — panel included."""
    panel_root, plan_path, radioactive = panel
    report = check_leakage(
        plan_path=plan_path,
        panel_data=(panel_root,),
        train_data=(panel_root,),
        holdout_fraction=0.0,
        split_seed=SPLIT_SEED,
    )
    assert not report.passed
    assert radioactive in report.leaked


def test_derived_corpus_leaking_via_provenance_fails(
    panel: tuple[Path, Path, Episode],
    tmp_path: Path,
) -> None:
    """The charter trap: a renamed corpus draws a different split, and
    its own holdout no longer covers the panel episode."""
    panel_root, plan_path, radioactive = panel
    derived_root = tmp_path / "derived"
    derived_dir = _write_dataset(derived_root, "fontaine/filtered", 3)
    provenance = {
        "version": 1,
        "episodes": [
            {
                "episode_index": index,
                "source_repo_id": "alice/cubes",
                "source_episode_index": source,
            }
            # episode 0 maps to the radioactive panel episode
            for index, source in enumerate([radioactive.episode_index, 1, 2])
        ],
    }
    (derived_dir / "meta" / "source_provenance.json").write_text(
        json.dumps(provenance),
    )
    report = check_leakage(
        plan_path=plan_path,
        panel_data=(panel_root,),
        train_data=(derived_root,),
        holdout_fraction=0.0,
        split_seed=SPLIT_SEED,
    )
    assert not report.passed
    assert radioactive in report.leaked


def test_unattributable_dataset_fails(
    panel: tuple[Path, Path, Episode],
    tmp_path: Path,
) -> None:
    panel_root, plan_path, _ = panel
    derived_root = tmp_path / "mystery"
    _write_dataset(derived_root, "fontaine/unknown-origin", 3)
    report = check_leakage(
        plan_path=plan_path,
        panel_data=(panel_root,),
        train_data=(derived_root,),
        holdout_fraction=0.0,
        split_seed=SPLIT_SEED,
    )
    assert not report.passed
    assert report.unattributable == ("fontaine/unknown-origin",)


def test_plan_outside_recomputed_holdout_is_fatal(
    panel: tuple[Path, Path, Episode],
    tmp_path: Path,
) -> None:
    """A plan referencing a non-holdout episode means the plan and the
    corpus disagree — SystemExit, never a silent pass."""
    panel_root, _, _ = panel
    held = set(holdout_episodes("alice/cubes", 10, HOLDOUT_FRACTION, SPLIT_SEED))
    trained = next(i for i in range(10) if i not in held)
    bad_plan = tmp_path / "bad_plan.json"
    _write_plan(bad_plan, core=[["alice/cubes", trained, 0]])
    with pytest.raises(SystemExit, match="OUTSIDE"):
        check_leakage(
            plan_path=bad_plan,
            panel_data=(panel_root,),
            train_data=(panel_root,),
            holdout_fraction=HOLDOUT_FRACTION,
            split_seed=SPLIT_SEED,
        )
