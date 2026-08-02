"""Pure-CPU tests for the judge-annotation materialization mapping.

The frame-placement function is where the 1-based-judge / 0-based-lerobot
off-by-one would silently corrupt every label — pin it with synthetic
annotations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from lerobot.annotations.steerable_pipeline.reader import EpisodeRecord

from bijou.judge.materialize import (
    ANNOTATION_COLUMNS,
    EVENT_STYLE,
    _event_rows,
    _per_camera_array,
    annotation_feature_info,
    episode_annotation_arrays,
)
from bijou.judge.schema import (
    CameraVisibility,
    EpisodeJudgment,
    FrameAnnotation,
    InstructionQuality,
    Scores,
    Subgoal,
    TaskCompletion,
    Verdict,
)


def annotation(
    frame: int,
    progress: float,
    *,
    holding: bool,
    front_object: bool = True,
    events: tuple[str, ...] = (),
) -> FrameAnnotation:
    return FrameAnnotation(
        frame=frame,
        progress=progress,
        holding=holding,
        visible={
            "front": CameraVisibility(task_object=front_object, gripper=False),
            "wrist": CameraVisibility(task_object=True, gripper=True),
        },
        events=events,
    )


def judgment_with(annotations: tuple[FrameAnnotation, ...]) -> EpisodeJudgment:
    return EpisodeJudgment(
        overall_score=7,
        verdict=Verdict.KEEP,
        task_completion_visible=TaskCompletion.YES,
        scores=Scores(visual_quality=7, smoothness=7, efficiency=7, camera_framing=7),
        instruction_quality=InstructionQuality.GOOD,
        observed_task="a task",
        suggested_instructions=("do it",),
        subgoals=(Subgoal(until_frame=10, subgoal="do it"),),
        frame_annotations=annotations,
        camera_kinds={},  # not consulted by the event projection
        issues=(),
        summary="",
    )


def episode_record(num_frames: int) -> EpisodeRecord:
    return EpisodeRecord(
        episode_index=0,
        episode_task="a task",
        frame_timestamps=tuple(i / 30 for i in range(num_frames)),
        frame_indices=tuple(range(num_frames)),
        data_path=Path("unused.parquet"),
        row_offset=0,
        row_count=num_frames,
    )


def test_multi_event_frame_yields_one_row_per_event_never_merged() -> None:
    """Several events legitimately anchor to one sampled frame (0.15% of
    corpus frames); each must become its own language_events row at the
    frame's exact timestamp — the class that crashed single-row consumers
    and must never be silently joined or dropped."""
    judgment = judgment_with(
        (
            annotation(1, 0.0, holding=False),
            annotation(3, 0.5, holding=True, events=("box toppled", "reset begins")),
            annotation(10, 1.0, holding=False, events=("task completed",)),
        ),
    )
    rows = _event_rows(judgment, episode_record(10))
    assert [row["content"] for row in rows] == [
        "box toppled",
        "reset begins",
        "task completed",
    ]
    # Both same-frame rows anchor to the SAME timestamp (frame 3, 0-based 2).
    assert rows[0]["timestamp"] == rows[1]["timestamp"] == 2 / 30
    assert rows[2]["timestamp"] == 9 / 30
    assert all(row["style"] == EVENT_STYLE for row in rows)
    assert all(row["role"] == "assistant" for row in rows)


def test_event_beyond_episode_length_is_loud() -> None:
    judgment = judgment_with(
        (annotation(11, 0.5, holding=False, events=("ghost event",)),),
    )
    with pytest.raises(ValueError, match="beyond"):
        _event_rows(judgment, episode_record(10))


def test_placement_and_nan_mask() -> None:
    cameras = ["front", "wrist"]
    arrays = episode_annotation_arrays(
        10,
        cameras,
        [
            annotation(1, 0.0, holding=False),  # 1-based first frame -> row 0
            annotation(10, 1.0, holding=True, front_object=False),  # last -> row 9
        ],
    )
    assert arrays.progress[0] == 0.0
    assert arrays.progress[9] == 1.0
    assert arrays.holding[0] == 0.0
    assert arrays.holding[9] == 1.0
    # Exactly the sampled frames are finite — the mask IS the sample set.
    assert np.isfinite(arrays.progress).sum() == 2
    assert np.isnan(arrays.progress[1:9]).all()
    assert np.isnan(arrays.visible_object[1:9]).all()
    # Camera order: column 0 = front, column 1 = wrist.
    assert arrays.visible_object[9, 0] == 0.0
    assert arrays.visible_object[9, 1] == 1.0
    assert arrays.visible_gripper[0, 0] == 0.0
    assert arrays.visible_gripper[0, 1] == 1.0


def test_frame_out_of_range_is_loud() -> None:
    with pytest.raises(ValueError, match=r"outside 1\.\.5"):
        episode_annotation_arrays(
            5,
            ["front", "wrist"],
            [annotation(6, 0.5, holding=False)],
        )


def test_unknown_visibility_camera_is_loud() -> None:
    with pytest.raises(ValueError, match="wrist"):
        episode_annotation_arrays(
            5,
            ["front"],  # judge answered for a camera the dataset lacks
            [annotation(1, 0.5, holding=False)],
        )


def test_per_camera_encoding_matches_lerobot_feature_convention() -> None:
    """lerobot casts shape-(1,) features to scalar Values — a length-1
    fixed_size_list breaks every single-camera dataset's schema cast
    (measured corpus-wide before the fix)."""
    single = _per_camera_array(np.zeros((4, 1), dtype=np.float32))
    assert single.type == pa.float32()
    multi = _per_camera_array(np.zeros((4, 2), dtype=np.float32))
    assert multi.type == pa.list_(pa.float32(), 2)


def test_feature_info_shapes_and_names() -> None:
    features = annotation_feature_info(["front", "wrist"])
    assert set(features) == set(ANNOTATION_COLUMNS)
    assert features["annotation.progress"]["shape"] == (1,)
    assert features["annotation.visible_object"]["shape"] == (2,)
    assert features["annotation.visible_object"]["names"] == ["front", "wrist"]
