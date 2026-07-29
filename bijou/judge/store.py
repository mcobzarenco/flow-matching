"""Durable per-dataset verdict store: ``meta/judgments.json``.

The envelope is ``{"judgments": [record, ...]}``; each record is a
``JudgmentRecord`` — provenance plus the verdict — keyed by
(episode_index, model, prompt_hash). Deliberately JSON, not parquet:
verdicts are nested, per-dataset counts are tiny (median ~60 episodes),
and JSON round-trips through the schema-validating dataclasses with no
flattening layer to maintain.

The file lives inside the dataset directory, so hub upload/download
carries it and train-time consumers read it next to the rest of the
metadata — this module needs nothing beyond the stdlib and the (equally
light) schema module, so consumers never pull the judging stacks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import EpisodeJudgment

JUDGMENTS_RELPATH = Path("meta") / "judgments.json"


@dataclass(frozen=True, slots=True)
class JudgmentRecord:
    """One stored verdict: identity, evidence provenance, and the judgment.

    ``key()`` — (episode_index, model, prompt_hash) — is what makes
    verdicts comparable and sweeps idempotent: the prompt hash is
    content-derived (schema.PROMPT_HASH), so editing the prompt re-judges
    automatically and nothing is bumped by hand.
    """

    episode_index: int
    model: str
    prompt_hash: str
    judged_at: str  # UTC, "%F %T"
    num_timesteps: int  # sampled timesteps the judge saw
    max_image_dim: int  # px, longer side after downscaling
    usage: dict[str, int]  # input/output token counts
    judgment: EpisodeJudgment

    def key(self) -> tuple[int, str, str]:
        return (self.episode_index, self.model, self.prompt_hash)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgmentRecord:
        """Parse + validate a stored record (the nested judgment included —
        loading IS re-validation)."""
        try:
            return cls(
                episode_index=int(data["episode_index"]),
                model=str(data["model"]),
                prompt_hash=str(data["prompt_hash"]),
                judged_at=str(data["judged_at"]),
                num_timesteps=int(data["num_timesteps"]),
                max_image_dim=int(data["max_image_dim"]),
                usage={str(k): int(v) for k, v in data["usage"].items()},
                judgment=EpisodeJudgment.from_dict(data["judgment"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed judgment record: {error}") from error

    @classmethod
    def from_journal(cls, record: dict[str, Any]) -> JudgmentRecord:
        """Build from a sweep journal ok-line (field names differ: the
        journal says ``episode``/``time``). Episode length, fps, task etc.
        stay journal-only — they are recoverable from the dataset metadata
        itself."""
        return cls.from_dict(
            {
                "episode_index": record["episode"],
                "judged_at": record["time"],
                **record,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "judged_at": self.judged_at,
            "num_timesteps": self.num_timesteps,
            "max_image_dim": self.max_image_dim,
            "usage": dict(self.usage),
            "judgment": self.judgment.to_dict(),
        }


def load_sidecar(dataset_dir: Path) -> list[JudgmentRecord]:
    """Parsed, validated records from a dataset's sidecar ([] when absent).

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
        return [JudgmentRecord.from_dict(record) for record in judgments]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(
            f"corrupt judgments sidecar {path}: {error}; fix or remove it",
        ) from error


def write_sidecar(dataset_dir: Path, records: list[JudgmentRecord]) -> None:
    """Atomically (re)write a dataset's sidecar, sorted for stable diffs.

    NOTE for the future curation rewrite: lerobot's delete_episodes
    renumbers episode_index — any dataset rewrite must remap (or
    deliberately drop) this file, it does not survive renumbering by itself.
    """
    path = dataset_dir / JUDGMENTS_RELPATH
    ordered = sorted(records, key=JudgmentRecord.key)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"judgments": [r.to_dict() for r in ordered]}, indent=1))
    tmp.replace(path)
