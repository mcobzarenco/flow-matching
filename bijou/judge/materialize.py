"""Materialize judge subgoals into LeRobot-native subtask annotations.

Run: ``python -m bijou.judge.materialize --root <dataset> [--form ...]``

Reads the dataset's ``meta/judgments.json`` sidecar (the durable verdict
store) and projects each judged episode's subgoal segments into lerobot's
native forms:

- ``--form sarm``: episode-metadata columns in ``meta/episodes/*.parquet``
  (``sparse_subtask_names/_start_times/_end_times/_start_frames/
  _end_frames`` + bare legacy duplicates — lerobot's
  ``save_annotations_to_dataset`` sparse convention, consumed by the SARM
  reward-model tooling and its local PNG visualizer).
- ``--form language``: ``language_persistent`` rows with
  ``style="subtask"`` in the DATA parquet — what the online dataset
  visualizer's Annotations tab actually renders (verified against the
  space source: it reads ``language_persistent``/``language_events`` and
  never the SARM columns). Reuses lerobot's own steerable-pipeline
  reader/staging/writer, so row validation, struct schema and atomic
  rewrites are theirs; one ``role="assistant"`` row per segment at the
  segment's exact start-frame timestamp (persistent = active until
  superseded, precisely our piecewise-constant semantics).
- ``--form both`` (default): both projections.

Mapping: judge segments carry 1-based inclusive ``until_frame``. SARM
columns get 0-based [start, end) frames and seconds; language rows get
one row per segment at ``frame_timestamps[start_frame]``.

The sidecar stays the source of truth: projections are idempotent and
re-runnable (later verdicts overwrite). Records are selected by prompt
hash (default: the running code's PROMPT_HASH) and optionally by model;
latest ``judged_at`` wins per episode. SARM rows are matched on the
``episode_index`` COLUMN, not the DataFrame index — lerobot's own writer
indexes per-file frames with global indices, which breaks on multi-file
datasets.

CAVEAT (language form): ``LanguageColumnsWriter`` rewrites whole data
files — episodes sharing a file with the selection but absent from it get
EMPTY language columns (pre-existing rows wiped). Fine for first-time
annotation; re-materialize the full dataset after adding verdicts.

Deliberately NOT materialized: per-frame annotations (progress, holding,
visibility, events) stay sidecar-only for now — events could become
``language_events`` rows later, but their style vocabulary
(interjection/vqa/trace) doesn't fit mistake-marking yet.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from lerobot.annotations.steerable_pipeline.reader import iter_episodes
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter
from lerobot.datasets.io_utils import load_info, write_info
from lerobot.datasets.language import language_feature_info

from .schema import PROMPT_HASH
from .store import JudgmentRecord, load_sidecar

SUBTASK_FIELDS = (
    "subtask_names",
    "subtask_start_times",
    "subtask_end_times",
    "subtask_start_frames",
    "subtask_end_frames",
)
# The visualizer/SARM sparse convention writes both prefixed and legacy names.
SUBTASK_COLUMNS = tuple(f"sparse_{field}" for field in SUBTASK_FIELDS) + SUBTASK_FIELDS


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


def subtask_cells(record: JudgmentRecord, fps: float) -> dict[str, list[Any]]:
    """One episode's subgoals -> SARM sparse column cells."""
    judgment = record.parsed_judgment()
    names: list[str] = []
    start_times: list[float] = []
    end_times: list[float] = []
    start_frames: list[int] = []
    end_frames: list[int] = []
    previous = 0
    for segment in judgment.subgoals:
        names.append(segment.subgoal)
        start_frames.append(previous)
        end_frames.append(segment.until_frame)
        start_times.append(previous / fps)
        end_times.append(segment.until_frame / fps)
        previous = segment.until_frame
    cells = dict(
        zip(
            SUBTASK_FIELDS,
            (names, start_times, end_times, start_frames, end_frames),
            strict=True,
        ),
    )
    return {f"sparse_{field}": value for field, value in cells.items()} | cells


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


def materialize_sarm(root: Path, chosen: dict[int, JudgmentRecord]) -> int:
    """Write SARM subtask columns for every judged episode; returns
    episodes materialized."""
    fps = float(json.loads((root / "meta" / "info.json").read_text())["fps"])
    cells_by_episode = {
        episode: subtask_cells(record, fps) for episode, record in chosen.items()
    }

    written = 0
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        changed = False
        for column in SUBTASK_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
            frame[column] = frame[column].astype(object)
        for position, episode_index in enumerate(frame["episode_index"]):
            cells = cells_by_episode.get(int(episode_index))
            if cells is None:
                continue
            row = frame.index[position]
            for column, value in cells.items():
                frame.at[row, column] = value
            changed = True
            written += 1
        if changed:
            frame.to_parquet(path, engine="pyarrow", compression="snappy")
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
        description="Project judge subgoals into lerobot-native subtask columns.",
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
    parser.add_argument(
        "--form",
        choices=["sarm", "language", "both"],
        default="both",
        help="sarm = episode-metadata columns (SARM tooling); language = "
        "language_persistent rows in data parquet (online visualizer) "
        "(default: %(default)s).",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    chosen, skipped = select_or_die(
        root,
        prompt_hash=args.prompt_hash,
        model=args.model,
    )
    if args.form in ("sarm", "both"):
        written = materialize_sarm(root, chosen)
        print(f"sarm: subtask columns for {written} episode(s) in {root}")
    if args.form in ("language", "both"):
        written = materialize_language(root, chosen)
        print(f"language: subtask rows for {written} episode(s) in {root}")
    if skipped:
        print(f"({skipped} sidecar episode(s) skipped by hash/model filters)")


if __name__ == "__main__":
    main()
