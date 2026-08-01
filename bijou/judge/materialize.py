"""Materialize judge subgoals as LeRobot-native ``language_persistent`` rows.

Run: ``python -m bijou.judge.materialize --root <dataset>``

Reads the dataset's ``meta/judgments.json`` sidecar (the durable verdict
store) and writes each judged episode's subgoal segments into the data
parquet as ``language_persistent`` rows with ``style="subtask"`` — the
form the online dataset visualizer's Annotations tab renders and
lerobot's ``active_at(t)`` resolver consumes (persistent = active until
superseded, precisely our piecewise-constant semantics). One
``role="assistant"`` row per segment at the segment's exact start-frame
timestamp; segments carry 1-based inclusive ``until_frame``, so segment
k activates at ``frame_timestamps[previous_until]``. Reuses lerobot's
own steerable-pipeline reader/staging/writer (their row validation,
struct schema, per-episode-row-group atomic rewrites) and advertises the
columns in ``meta/info.json`` features (without which non-streaming
LeRobotDataset loads cast against the old schema and fail).

The sidecar stays the source of truth: the projection is idempotent and
re-runnable (later verdicts overwrite). Records are selected by prompt
hash (default: the running code's PROMPT_HASH) and optionally by model;
latest ``judged_at`` wins per episode.

A SARM-columns projection (``sparse_subtask_*`` episode-metadata
columns) existed briefly and was removed: the visualizer never reads
them, no consumer of ours does, and every projection is a mapping to
keep in sync with schema evolution — resurrect from git if a SARM
reward-model experiment ever wants it.

CAVEAT: ``LanguageColumnsWriter`` rewrites whole data files — episodes
sharing a file with the selection but absent from it get EMPTY language
columns (pre-existing rows wiped). Fine for first-time annotation;
re-materialize the full dataset after adding verdicts. This is THE
invariant for the future curation merge to test.

Deliberately NOT materialized: per-frame annotations (progress, holding,
visibility, events) stay sidecar-only for now — see
docs/e2b-ar-data-contract.md for how consumers access them.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from lerobot.annotations.steerable_pipeline.reader import iter_episodes
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter
from lerobot.datasets.io_utils import load_info, write_info
from lerobot.datasets.language import language_feature_info

from .schema import PROMPT_HASH
from .store import JudgmentRecord, load_sidecar


def select_records(
    records: list[JudgmentRecord],
    *,
    prompt_hash: str,
    model: str | None,
) -> dict[int, JudgmentRecord]:
    """Latest matching record per episode (prompt hash exact, model optional)."""
    chosen: dict[int, JudgmentRecord] = {}
    for record in records:
        if record.prompt_hash != prompt_hash:
            continue
        if model is not None and record.model != model:
            continue
        current = chosen.get(record.episode_index)
        if current is None or record.judged_at > current.judged_at:
            chosen[record.episode_index] = record
    return chosen


def materialize_language(root: Path, chosen: dict[int, JudgmentRecord]) -> int:
    """Project subgoals as ``language_persistent`` subtask rows via
    lerobot's steerable-pipeline machinery; returns episodes written."""
    subgoals_by_episode = {
        episode: record.parsed_judgment().subgoals for episode, record in chosen.items()
    }
    written = 0
    with tempfile.TemporaryDirectory(prefix="judge_staging_") as staging_name:
        staging_dir = Path(staging_name)
        episode_records = []
        for record in iter_episodes(
            root,
            only_episodes=tuple(sorted(subgoals_by_episode)),
        ):
            segments = subgoals_by_episode[record.episode_index]
            rows: list[dict[str, Any]] = []
            previous = 0
            for segment in segments:
                # Segment covers 1-based frames (previous, until]; its
                # persistent row activates at the exact timestamp of its
                # first frame (0-based index = previous).
                if previous >= len(record.frame_timestamps):
                    raise ValueError(
                        f"episode {record.episode_index}: segment start frame "
                        f"{previous} beyond {len(record.frame_timestamps)} frames",
                    )
                rows.append(
                    {
                        "role": "assistant",
                        "content": segment.subgoal,
                        "style": "subtask",
                        "timestamp": float(record.frame_timestamps[previous]),
                        "tool_calls": None,
                    },
                )
                previous = segment.until_frame
            # "plan" is the steerable pipeline's module slot for
            # subtask-style rows (its own subtask generator stages there).
            EpisodeStaging(staging_dir, record.episode_index).write("plan", rows)
            episode_records.append(record)
            written += 1
        if episode_records:
            LanguageColumnsWriter().write_all(episode_records, staging_dir, root)
    # Advertise the new columns in meta/info.json features — without this,
    # non-streaming LeRobotDataset loads cast data files against the old
    # schema and fail on the extra columns (lerobot's executor does the
    # same merge after its writer runs).
    info = load_info(root)
    merged = {**info.features, **language_feature_info()}
    if merged != info.features:
        info.features = merged
        write_info(info, root)
    return written


def select_or_die(
    root: Path,
    *,
    prompt_hash: str,
    model: str | None,
) -> tuple[dict[int, JudgmentRecord], int]:
    records = load_sidecar(root)
    if not records:
        raise SystemExit(f"no judgments sidecar (or no records) under {root}")
    chosen = select_records(records, prompt_hash=prompt_hash, model=model)
    skipped = len({record.episode_index for record in records}) - len(chosen)
    if not chosen:
        raise SystemExit(
            f"no records match prompt_hash={prompt_hash}"
            + (f", model={model}" if model else "")
            + f" (sidecar holds {len(records)} record(s) — check --prompt-hash)",
        )
    return chosen, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project judge subgoals into lerobot-native "
        "language_persistent subtask rows.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset directory containing meta/judgments.json (v3.0 format).",
    )
    parser.add_argument(
        "--prompt-hash",
        type=str,
        default=PROMPT_HASH,
        help="Only materialize records judged under this prompt hash "
        "(default: the running code's, %(default)s).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Only materialize records from this judge model (default: any).",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    chosen, skipped = select_or_die(
        root,
        prompt_hash=args.prompt_hash,
        model=args.model,
    )
    written = materialize_language(root, chosen)
    print(f"language_persistent subtask rows for {written} episode(s) in {root}")
    if skipped:
        print(f"({skipped} sidecar episode(s) skipped by hash/model filters)")


if __name__ == "__main__":
    main()
