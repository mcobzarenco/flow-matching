"""Materialize judge annotations into their LeRobot-native homes.

Run: ``python -m bijou.judge.materialize --root <dataset>``

Reads the dataset's ``meta/judgments.json`` sidecar (the durable verdict
store, which remains the source of truth) and projects each judged
episode's annotations into the form a plain ``LeRobotDataset`` consumer
sees without any sidecar knowledge:

- **subgoals** → ``language_persistent`` rows, ``style="subtask"`` — the
  form the online visualizer's Annotations tab renders and lerobot's
  ``active_at(t)`` resolver consumes (persistent = active until
  superseded, precisely our piecewise-constant semantics). One
  ``role="assistant"`` row per segment at the segment's exact
  start-frame timestamp.
- **events** → ``language_events`` rows, ``style="event"`` (project-local
  style, registered through lerobot's documented import-time extension
  hook). Event rows live on the exact frame row where they fired —
  lerobot's writer buckets them by timestamp; a frame may carry SEVERAL
  rows (one per event), so consumers read the frame's rows directly —
  ``emitted_at`` is a single-row resolver and raises on multi-event
  frames. Point-in-time facts are never broadcast.
- **progress / holding / per-camera visibility** → NaN-masked float32
  feature columns: ``annotation.progress`` and ``annotation.holding``
  (scalars), ``annotation.visible_object`` and
  ``annotation.visible_gripper`` (vectors over the dataset's cameras in
  sorted short-name order, recorded in the feature's ``names``).
  NaN = the judge never saw that frame; supervise through an
  ``isfinite`` mask, never interpolate. Booleans are 0.0/1.0.

Invariants downstream consumers may rely on:

- The finite-value mask of ``annotation.progress`` IS the judge's
  sampled-frame set: a sampled frame with no event row is a true "no
  event" negative; an unsampled frame is unknown, not negative.
- Judge frame numbers are 1-based inclusive; ``frame_index`` is 0-based
  (the off-by-one is resolved here, once).
- Every run rewrites the WHOLE dataset (all data files): unjudged
  episodes get explicitly empty language rows and all-NaN columns, so a
  rerun can never leave stale rows from an earlier selection behind.
- One ``(model filter, prompt_hash)`` selection per materialized
  dataset, stamped in ``meta/judge_annotations.json`` — mixed-provenance
  columns are unrepresentable. Records are selected by prompt hash
  (default: the running code's PROMPT_HASH) and optionally by model;
  latest ``judged_at`` wins per episode.
- Stats are deliberately NOT written for annotation columns: they are
  supervision labels, nothing normalizes them, and NaN would poison
  ``aggregate_stats``.

A SARM-columns projection (``sparse_subtask_*`` episode-metadata
columns) existed briefly and was removed: the visualizer never reads
them, no consumer of ours does, and every projection is a mapping to
keep in sync with schema evolution — resurrect from git if a SARM
reward-model experiment ever wants it.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.annotations.steerable_pipeline.reader import EpisodeRecord, iter_episodes
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter
from lerobot.datasets.io_utils import (
    load_info,
    write_info,
    write_table_one_row_group_per_episode,
)
from lerobot.datasets.language import language_feature_info

from ..annotations import EVENT_STYLE
from .evidence import short_camera
from .schema import PROMPT_HASH, EpisodeJudgment, FrameAnnotation
from .store import JudgmentRecord, load_sidecar

ANNOTATION_PROGRESS = "annotation.progress"
ANNOTATION_HOLDING = "annotation.holding"
ANNOTATION_VISIBLE_OBJECT = "annotation.visible_object"
ANNOTATION_VISIBLE_GRIPPER = "annotation.visible_gripper"
ANNOTATION_COLUMNS = (
    ANNOTATION_PROGRESS,
    ANNOTATION_HOLDING,
    ANNOTATION_VISIBLE_OBJECT,
    ANNOTATION_VISIBLE_GRIPPER,
)
PROVENANCE_RELPATH = Path("meta") / "judge_annotations.json"


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


def _persistent_rows(
    judgment: EpisodeJudgment,
    record: EpisodeRecord,
) -> list[dict[str, Any]]:
    """Subgoal segments as staged persistent subtask rows.

    Segments carry 1-based inclusive ``until_frame``, so segment k
    activates at ``frame_timestamps[previous_until]`` (0-based index of
    its first frame)."""
    rows: list[dict[str, Any]] = []
    previous = 0
    for segment in judgment.subgoals:
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
                "camera": None,
                "tool_calls": None,
            },
        )
        previous = segment.until_frame
    return rows


def _event_rows(
    judgment: EpisodeJudgment,
    record: EpisodeRecord,
) -> list[dict[str, Any]]:
    """Judge events as staged event rows; the writer buckets each onto the
    frame whose timestamp matches exactly (same parquet-sourced floats on
    both sides, so equality is safe)."""
    rows: list[dict[str, Any]] = []
    for annotation in judgment.frame_annotations:
        if not annotation.events:
            continue
        if annotation.frame > len(record.frame_timestamps):
            raise ValueError(
                f"episode {record.episode_index}: annotated frame "
                f"{annotation.frame} beyond {len(record.frame_timestamps)} frames",
            )
        timestamp = float(record.frame_timestamps[annotation.frame - 1])
        rows.extend(
            {
                "role": "assistant",
                "content": event,
                "style": EVENT_STYLE,
                "timestamp": timestamp,
                "camera": None,
                "tool_calls": None,
            }
            for event in annotation.events
        )
    return rows


def materialize_language(
    root: Path,
    judgments: dict[int, EpisodeJudgment],
) -> tuple[int, int]:
    """Write subtask + event rows for judged episodes and explicitly empty
    language columns for every other episode (full-dataset rewrite via
    lerobot's steerable-pipeline writer). Returns (subtask rows, event
    rows) written."""
    subtask_count = 0
    event_count = 0
    with tempfile.TemporaryDirectory(prefix="judge_staging_") as staging_name:
        staging_dir = Path(staging_name)
        episode_records: list[EpisodeRecord] = []
        for record in iter_episodes(root):
            episode_records.append(record)
            judgment = judgments.get(record.episode_index)
            if judgment is None:
                continue
            persistent = _persistent_rows(judgment, record)
            events = _event_rows(judgment, record)
            # One staging file carries both kinds: the writer partitions
            # rows into the two columns by style, not by module slot.
            EpisodeStaging(staging_dir, record.episode_index).write(
                "plan",
                persistent + events,
            )
            subtask_count += len(persistent)
            event_count += len(events)
        if not episode_records:
            raise ValueError(f"no episodes under {root}/data")
        LanguageColumnsWriter().write_all(episode_records, staging_dir, root)
    return subtask_count, event_count


@dataclass(frozen=True, slots=True)
class EpisodeAnnotationArrays:
    """Per-frame annotation columns for one episode (NaN = not sampled).

    Shapes: ``progress``/``holding`` are (num_frames,); the visibility
    matrices are (num_frames, len(cameras)) in the caller's camera order.
    """

    progress: np.ndarray
    holding: np.ndarray
    visible_object: np.ndarray
    visible_gripper: np.ndarray


def episode_annotation_arrays(
    num_frames: int,
    cameras: list[str],
    annotations: Sequence[FrameAnnotation],
) -> EpisodeAnnotationArrays:
    """Place sampled-frame annotations into NaN-initialized per-frame arrays.

    Judge frames are 1-based; row = frame - 1. Booleans become 0.0/1.0.
    A frame outside [1, num_frames] or a visibility camera missing from
    ``cameras`` is a hard error (sidecar/dataset mismatch); a camera the
    judge never saw (single-episode camera filter) simply stays NaN.
    """
    column_of = {camera: i for i, camera in enumerate(cameras)}
    progress = np.full(num_frames, np.nan, dtype=np.float32)
    holding = np.full(num_frames, np.nan, dtype=np.float32)
    visible_object = np.full((num_frames, len(cameras)), np.nan, dtype=np.float32)
    visible_gripper = np.full((num_frames, len(cameras)), np.nan, dtype=np.float32)
    for annotation in annotations:
        if not 1 <= annotation.frame <= num_frames:
            raise ValueError(
                f"annotated frame {annotation.frame} outside 1..{num_frames}",
            )
        unknown = set(annotation.visible) - set(column_of)
        if unknown:
            raise ValueError(
                f"visibility cameras {sorted(unknown)} not among dataset "
                f"cameras {cameras}",
            )
        row = annotation.frame - 1
        progress[row] = annotation.progress
        holding[row] = float(annotation.holding)
        for camera, visibility in annotation.visible.items():
            column = column_of[camera]
            visible_object[row, column] = float(visibility.task_object)
            visible_gripper[row, column] = float(visibility.gripper)
    return EpisodeAnnotationArrays(
        progress=progress,
        holding=holding,
        visible_object=visible_object,
        visible_gripper=visible_gripper,
    )


def _fixed_size_list_array(matrix: np.ndarray) -> pa.Array:
    """(n, width) float32 -> arrow fixed_size_list<float>[width] (the same
    encoding lerobot uses for 'action'/'observation.state')."""
    _, width = matrix.shape
    values = pa.array(matrix.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(values, width)


def _per_camera_array(matrix: np.ndarray) -> pa.Array:
    """(n, n_cameras) -> the arrow column lerobot's feature convention
    expects: shape (1,) features cast as scalar Values, so single-camera
    visibility must be a plain float column — fixed_size_list[1] fails
    LeRobotDataset's schema cast (measured: every 1-camera dataset in the
    corpus). Multi-camera stays fixed_size_list[n], like 'action'."""
    if matrix.shape[1] == 1:
        return pa.array(matrix[:, 0], type=pa.float32())
    return _fixed_size_list_array(matrix)


def materialize_annotation_columns(
    root: Path,
    judgments: dict[int, EpisodeJudgment],
    cameras: list[str],
) -> tuple[int, int]:
    """Rewrite every data file with the four ``annotation.*`` columns
    (replacing any previous materialization). Returns (annotated frames,
    files rewritten). A judged episode absent from every data file is a
    hard error — that is what a renumbering-invalidated sidecar looks
    like."""
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise ValueError(f"no data parquet files under {root}/data")
    pending = set(judgments)
    annotated_frames = 0
    for path in data_files:
        table = pq.read_table(path)
        stale = [name for name in ANNOTATION_COLUMNS if name in table.column_names]
        if stale:
            table = table.drop_columns(stale)
        episode_col = table.column("episode_index").to_numpy(zero_copy_only=False)
        frame_col = table.column("frame_index").to_numpy(zero_copy_only=False)
        num_rows = table.num_rows
        progress = np.full(num_rows, np.nan, dtype=np.float32)
        holding = np.full(num_rows, np.nan, dtype=np.float32)
        visible_object = np.full((num_rows, len(cameras)), np.nan, dtype=np.float32)
        visible_gripper = np.full((num_rows, len(cameras)), np.nan, dtype=np.float32)
        for episode in np.unique(episode_col):
            judgment = judgments.get(int(episode))
            if judgment is None:
                continue
            rows = np.flatnonzero(episode_col == episode)
            if not np.array_equal(frame_col[rows], np.arange(len(rows))):
                raise ValueError(
                    f"{path}: episode {int(episode)} rows are not the "
                    f"contiguous frame range 0..{len(rows) - 1}",
                )
            try:
                arrays = episode_annotation_arrays(
                    len(rows),
                    cameras,
                    judgment.frame_annotations,
                )
            except ValueError as error:
                raise ValueError(f"episode {int(episode)}: {error}") from error
            progress[rows] = arrays.progress
            holding[rows] = arrays.holding
            visible_object[rows] = arrays.visible_object
            visible_gripper[rows] = arrays.visible_gripper
            annotated_frames += int(np.isfinite(arrays.progress).sum())
            pending.discard(int(episode))
        table = table.append_column(
            ANNOTATION_PROGRESS,
            pa.array(progress, type=pa.float32()),
        )
        table = table.append_column(
            ANNOTATION_HOLDING,
            pa.array(holding, type=pa.float32()),
        )
        table = table.append_column(
            ANNOTATION_VISIBLE_OBJECT,
            _per_camera_array(visible_object),
        )
        table = table.append_column(
            ANNOTATION_VISIBLE_GRIPPER,
            _per_camera_array(visible_gripper),
        )
        # Atomic replace, preserving the one-row-group-per-episode layout
        # (plain pq.write_table would collapse row groups and break the
        # readers' random access).
        tmp = path.with_name(path.name + ".tmp")
        write_table_one_row_group_per_episode(table, tmp)
        tmp.replace(path)
    if pending:
        raise ValueError(
            f"judged episodes {sorted(pending)} appear in no data file under "
            f"{root} — episode renumbering without a sidecar remap?",
        )
    return annotated_frames, len(data_files)


def annotation_feature_info(cameras: list[str]) -> dict[str, dict[str, Any]]:
    """``info.json`` feature entries for the annotation columns (fresh dict
    per key: DatasetInfo mutates shapes in place)."""
    features: dict[str, dict[str, Any]] = {}
    for name in (ANNOTATION_PROGRESS, ANNOTATION_HOLDING):
        features[name] = {"dtype": "float32", "shape": (1,), "names": None}
    for name in (ANNOTATION_VISIBLE_OBJECT, ANNOTATION_VISIBLE_GRIPPER):
        features[name] = {
            "dtype": "float32",
            "shape": (len(cameras),),
            "names": list(cameras),
        }
    return features


def write_provenance(
    root: Path,
    *,
    prompt_hash: str,
    model: str | None,
    chosen: dict[int, JudgmentRecord],
) -> None:
    """Stamp what this materialization was built from: one selection per
    dataset, so a loader pinning (model, prompt_hash) can fail loudly on
    mismatch instead of training on stale labels."""
    payload = {
        "prompt_hash": prompt_hash,
        "model_filter": model,
        "models": sorted({record.model for record in chosen.values()}),
        "written_at": time.strftime("%F %T", time.gmtime()),
        "episodes": sorted(chosen),
        "columns": list(ANNOTATION_COLUMNS),
        "event_style": EVENT_STYLE,
        "source": "meta/judgments.json",
    }
    path = root / PROVENANCE_RELPATH
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def dataset_cameras(features: dict[str, dict[str, Any]]) -> list[str]:
    """Camera short names in sorted order — the same order the judge's
    evidence (and thus ``camera_kinds``/visibility keys) uses."""
    return sorted(
        short_camera(key)
        for key, feature in features.items()
        if feature.get("dtype") == "video"
    )


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
        description="Project judge annotations into LeRobot-native form: "
        "subtask rows, event rows, and NaN-masked annotation.* columns.",
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
    judgments = {
        episode: record.parsed_judgment() for episode, record in chosen.items()
    }

    info = load_info(root)
    cameras = dataset_cameras(info.features)
    if not cameras:
        raise SystemExit(f"{root}: no video features — nothing to key visibility on")

    subtask_count, event_count = materialize_language(root, judgments)
    annotated_frames, files_rewritten = materialize_annotation_columns(
        root,
        judgments,
        cameras,
    )

    # Advertise the new columns in meta/info.json features LAST — readers
    # cast data files against features, so the schema promise must only
    # appear once the columns actually exist everywhere.
    info.features = {
        **info.features,
        **language_feature_info(),
        **annotation_feature_info(cameras),
    }
    write_info(info, root)
    write_provenance(
        root,
        prompt_hash=args.prompt_hash,
        model=args.model,
        chosen=chosen,
    )

    print(
        f"{len(judgments)} judged episode(s): {subtask_count} subtask row(s), "
        f"{event_count} event row(s), {annotated_frames} annotated frame(s) "
        f"across {files_rewritten} data file(s) [cameras: {', '.join(cameras)}]",
    )
    print(f"features + provenance written ({PROVENANCE_RELPATH})")
    if skipped:
        print(f"({skipped} sidecar episode(s) skipped by hash/model filters)")


if __name__ == "__main__":
    main()
