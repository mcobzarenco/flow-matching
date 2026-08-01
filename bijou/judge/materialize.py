"""Materialize judge subgoals into LeRobot-native subtask annotations.

Run: ``python -m bijou.judge.materialize --root <dataset>``

Reads the dataset's ``meta/judgments.json`` sidecar (the durable verdict
store) and writes each judged episode's subgoal segments into the
``meta/episodes/*.parquet`` columns that lerobot's SARM tooling and the
online dataset visualizer already understand:

    sparse_subtask_names / _start_times / _end_times / _start_frames /
    _end_frames  (+ the bare legacy ``subtask_*`` duplicates, matching
    lerobot's ``save_annotations_to_dataset`` sparse convention)

Mapping: judge segments carry 1-based inclusive ``until_frame``; SARM
columns want 0-based [start, end) frames and seconds. Segment k spanning
1-based frames (prev, until] becomes start_frame=prev, end_frame=until,
start_time=prev/fps, end_time=until/fps.

The sidecar stays the source of truth: this is a projection, idempotent
and re-runnable (later verdicts overwrite the same episode's cells).
Records are selected by prompt hash (default: the running code's
PROMPT_HASH) and optionally by model; latest ``judged_at`` wins per
episode. Rows are matched on the ``episode_index`` COLUMN, not the
DataFrame index — lerobot's own writer indexes per-file frames with
global indices, which breaks on multi-file datasets.

Deliberately NOT materialized here: per-frame annotations (progress,
holding, visibility, events) have no native lerobot home yet — they stay
in the sidecar; the ``language_persistent``/``language_events`` columns
would be the native path but require rewriting every data parquet for a
processor pipeline bijou does not use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

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


def materialize(
    root: Path,
    *,
    prompt_hash: str,
    model: str | None,
) -> tuple[int, int]:
    """Write subtask columns for every judged episode; returns
    (episodes materialized, episodes in sidecar skipped by the filters)."""
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
    return written, skipped


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
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    written, skipped = materialize(
        root,
        prompt_hash=args.prompt_hash,
        model=args.model,
    )
    print(
        f"materialized subtask columns for {written} episode(s) in {root} "
        f"({skipped} sidecar episode(s) skipped by hash/model filters)",
    )


if __name__ == "__main__":
    main()
