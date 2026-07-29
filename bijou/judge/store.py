"""Durable per-dataset verdict store: ``meta/judgments.json``.

The envelope is ``{"judgments": [record, ...]}``; each record keeps the
exact ``EpisodeJudgment.to_dict()`` payload under ``"judgment"`` plus
provenance fields, keyed by (episode_index, model, prompt_hash).
Deliberately JSON, not parquet: verdicts are nested, per-dataset counts are
tiny (median ~60 episodes), and JSON round-trips through the
schema-validating dataclass with no flattening layer to maintain.

The file lives inside the dataset directory, so hub upload/download carries
it and train-time consumers read it next to the rest of the metadata —
this module is stdlib-only so they never pull the judging stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

JUDGMENTS_RELPATH = Path("meta") / "judgments.json"


def judgment_key(record: dict[str, Any]) -> tuple[int, str, str]:
    """(episode_index, model, prompt_hash) — what makes verdicts comparable.

    The prompt hash is content-derived (see schema.PROMPT_HASH): editing the
    prompt re-judges automatically, nothing is bumped by hand.
    """
    return (
        int(record["episode_index"]),
        str(record["model"]),
        str(record["prompt_hash"]),
    )


def load_sidecar(dataset_dir: Path) -> list[dict[str, Any]]:
    """Records from a dataset's judgments sidecar ([] when absent).

    A corrupt sidecar is fatal, not empty: treating it as empty would
    re-judge and then overwrite whatever the file still holds.
    """
    path = dataset_dir / JUDGMENTS_RELPATH
    if not path.exists():
        return []
    try:
        judgments = json.loads(path.read_text())["judgments"]
        if not isinstance(judgments, list):
            raise TypeError(
                f"'judgments' must be an array, got {type(judgments).__name__}",
            )
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise SystemExit(
            f"corrupt judgments sidecar {path}: {error}; fix or remove it",
        ) from error
    return judgments


def write_sidecar(dataset_dir: Path, records: list[dict[str, Any]]) -> None:
    """Atomically (re)write a dataset's sidecar, sorted for stable diffs.

    NOTE for the future curation rewrite: lerobot's delete_episodes
    renumbers episode_index — any dataset rewrite must remap (or
    deliberately drop) this file, it does not survive renumbering by itself.
    """
    path = dataset_dir / JUDGMENTS_RELPATH
    ordered = sorted(records, key=judgment_key)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"judgments": ordered}, indent=1))
    tmp.replace(path)


def sidecar_record(journal_record: dict[str, Any]) -> dict[str, Any]:
    """Durable subset of a journal ok-record: key + provenance + the verdict
    payload exactly as EpisodeJudgment.to_dict() produced it (episode
    length, fps, task etc. stay journal-only — they are recoverable from
    the dataset metadata itself)."""
    return {
        "episode_index": int(journal_record["episode"]),
        "model": journal_record["model"],
        "prompt_hash": str(journal_record["prompt_hash"]),
        "judged_at": journal_record["time"],
        "num_timesteps": journal_record.get("num_timesteps"),
        "max_image_dim": journal_record.get("max_image_dim"),
        "usage": journal_record.get("usage"),
        "judgment": journal_record["judgment"],
    }
