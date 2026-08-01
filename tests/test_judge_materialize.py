"""Pure-CPU tests for the judge-annotation materialization mapping.

The frame-placement function is where the 1-based-judge / 0-based-lerobot
off-by-one would silently corrupt every label — pin it with synthetic
annotations.
"""

from __future__ import annotations

import numpy as np
import pytest

from bijou.judge.materialize import (
    ANNOTATION_COLUMNS,
    annotation_feature_info,
    episode_annotation_arrays,
)
from bijou.judge.schema import CameraVisibility, FrameAnnotation


def annotation(
    frame: int,
    progress: float,
    *,
    holding: bool,
    front_object: bool = True,
) -> FrameAnnotation:
    return FrameAnnotation(
        frame=frame,
        progress=progress,
        holding=holding,
        visible={
            "front": CameraVisibility(task_object=front_object, gripper=False),
            "wrist": CameraVisibility(task_object=True, gripper=True),
        },
        events=(),
    )


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


def test_feature_info_shapes_and_names() -> None:
    features = annotation_feature_info(["front", "wrist"])
    assert set(features) == set(ANNOTATION_COLUMNS)
    assert features["annotation.progress"]["shape"] == (1,)
    assert features["annotation.visible_object"]["shape"] == (2,)
    assert features["annotation.visible_object"]["names"] == ["front", "wrist"]
