"""Joint-frame remap between arm calibration and checkpoint frame
(bijou.rollout_safety.JointFrameTransform).

Pure CPU. The v30-to-v21 preset is pinned BOTH ways against the
official lerobot PR#777 SO-100/101 conversion (the backwardcomp doc's
replay transform: shoulder_lift' = −(x − 90), elbow_flex' = x − 90,
identity elsewhere), so a sign can never be "harmonized" without
failing here — the wrong direction is exactly the slam-into-the-table
failure the remap exists to prevent.
"""

from __future__ import annotations

import pytest
import torch

from bijou.data import DatasetStats
from bijou.rollout_safety import (
    JointFrameTransform,
    envelope_violations,
    state_envelope,
)

# The rig's measured rest pose (sim/so101_sim.py HOME_DEGREES: the
# median first-frame observation.state of so101_pick_place_v2) — arm
# frame under a post-PR#777 (lerobot >= 0.5) calibration.
RIG_HOME = [4.6, -102.7, 97.0, 78.7, 77.6, 3.5]

# The released MolmoAct2 SO-100/101 table's state band, rounded (its
# norm_stats.json q01/q99) — the OLD (pre-PR#777) degrees frame that a
# global-normalization checkpoint bakes in.
OLD_FRAME_Q01 = (-42.0, 44.0, 38.0, 6.0, -63.0, 1.0)
OLD_FRAME_Q99 = (48.0, 185.0, 173.0, 92.0, 43.0, 44.0)


def official_model_to_arm(model: list[float]) -> list[float]:
    """The backwardcomp doc's replay transform, transcribed verbatim
    (old/model-frame values → post-PR#777 arm frame)."""
    arm = list(model)
    arm[1] = -(arm[1] - 90.0)
    arm[2] = arm[2] - 90.0
    return arm


def old_frame_stats() -> DatasetStats:
    mean = tuple(
        (lo + hi) / 2 for lo, hi in zip(OLD_FRAME_Q01, OLD_FRAME_Q99, strict=True)
    )
    return DatasetStats(
        action_mean=mean,
        action_std=(10.0,) * 6,
        state_mean=mean,
        state_std=(10.0,) * 6,
        action_q01=OLD_FRAME_Q01,
        action_q99=OLD_FRAME_Q99,
        state_q01=OLD_FRAME_Q01,
        state_q99=OLD_FRAME_Q99,
    )


def test_preset_state_to_model_inverts_the_official_transform() -> None:
    # arm→model must be the exact inverse of the doc's model→arm on
    # every joint (this is also the reference MolmoAct2 deployment
    # default: signs 1,-1,1,1,1,1 / offsets 0,90,90,0,0,0).
    frame = JointFrameTransform.lerobot_v30_to_v21()
    model = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    assert frame.state_to_model(official_model_to_arm(model)) == pytest.approx(model)


def test_preset_chunk_to_arm_matches_the_official_transform() -> None:
    frame = JointFrameTransform.lerobot_v30_to_v21()
    model_rows = [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [0.0] * 6]
    expected = torch.tensor([official_model_to_arm(row) for row in model_rows])
    assert torch.equal(frame.chunk_to_arm(torch.tensor(model_rows)), expected)


def test_round_trip_recovers_arm_frame() -> None:
    frame = JointFrameTransform.lerobot_v30_to_v21()
    model = frame.state_to_model(RIG_HOME)
    arm = frame.chunk_to_arm(torch.tensor([model], dtype=torch.float64))
    assert arm[0].tolist() == pytest.approx(RIG_HOME)


def test_rig_home_maps_to_old_frame_band_edge() -> None:
    # The measured rest pose lands at the top edge of the released
    # table's state band (q99 shoulder_lift 185.3 / elbow_flex 173.1) —
    # the numeric anchor of the whole frame story.
    frame = JointFrameTransform.lerobot_v30_to_v21()
    model = frame.state_to_model(RIG_HOME)
    assert model[1] == pytest.approx(192.7)
    assert model[2] == pytest.approx(187.0)
    assert model[0] == pytest.approx(4.6)  # identity joints untouched


def test_identity_is_noop_and_passthrough() -> None:
    frame = JointFrameTransform.identity(6)
    assert frame.is_identity
    chunk = torch.randn(5, 6)
    # The tensor passes through UNTOUCHED: the no-remap deployment path
    # stays byte-identical to the pre-flag rollout.
    assert frame.chunk_to_arm(chunk) is chunk
    values = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]
    assert frame.state_to_model(values) == values
    assert not JointFrameTransform.lerobot_v30_to_v21().is_identity


def test_constructor_validates_signs_and_lengths() -> None:
    with pytest.raises(ValueError, match="±1"):
        JointFrameTransform(signs=(2.0,), offsets=(0.0,))
    with pytest.raises(ValueError, match="per joint"):
        JointFrameTransform(signs=(1.0, 1.0), offsets=(0.0,))


def test_state_dim_mismatch_is_loud() -> None:
    with pytest.raises(ValueError):
        JointFrameTransform.lerobot_v30_to_v21().state_to_model([0.0, 0.0])


def test_model_frame_gate_catches_a_missing_remap() -> None:
    """The failure this module exists for: a global-normalization
    checkpoint's state band (old frame) gated against a new-frame arm.
    The raw rest pose must flag exactly shoulder_lift (the sign-flipped
    joint — ~130° below the band even after widening); the remapped
    pose must pass every joint, including wrist_roll's known fleet
    variation (77.6 vs their q99 43, inside the widened band)."""
    envelope = state_envelope(old_frame_stats(), expected_dim=6)
    frame = JointFrameTransform.lerobot_v30_to_v21()
    assert envelope_violations(RIG_HOME, envelope) == [1]
    assert envelope_violations(frame.state_to_model(RIG_HOME), envelope) == []


def test_observation_to_item_maps_state_into_model_frame() -> None:
    from bijou.rollout import SO_MOTORS, observation_to_item

    frame = JointFrameTransform.lerobot_v30_to_v21()
    observation = {
        f"{motor}.pos": value for motor, value in zip(SO_MOTORS, RIG_HOME, strict=True)
    }
    item = observation_to_item(
        observation,
        "task",
        stats=old_frame_stats(),
        chunk_size=30,
        camera_kinds={},
        frame=frame,
    )
    assert item["observation.state"].tolist() == pytest.approx(
        frame.state_to_model(RIG_HOME),
    )
    # Omitted frame = identity (the sim call sites): raw state passes through.
    raw = observation_to_item(
        observation,
        "task",
        stats=old_frame_stats(),
        chunk_size=30,
        camera_kinds={},
    )
    assert raw["observation.state"].tolist() == pytest.approx(RIG_HOME)
