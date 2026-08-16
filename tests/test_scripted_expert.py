"""CPU oracles for the privileged scripted expert (grasp-SFT bootstrap
stage A, pre-reg DRAFT posts/2026-08-14-prereg-grasp-sft-bootstrap.md).

Kinematics-only — the model loads and FK/IK run without GL; the
physics-stepping validation (the registered ≥70% gate read) is the
post-finalization GPU read, not a unit test."""

import mujoco
import numpy as np
import pytest

from sim.scripted_expert import (
    DEMO_SEED_BASE,
    JAW_OPEN_RAD,
    PICKUP_QPOS,
    ExpertPlanner,
    run_expert_episode,
)
from sim.so101_sim import SCENE_PATH


class ModelOnly:
    """The planner's sim-facing surface without constructing SO101Sim
    (no renderer, no post pipeline — pure kinematics)."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        disk = self.model.geom("disk")
        self.disk_center = (float(disk.pos[0]), float(disk.pos[1]))


@pytest.fixture(scope="module")
def planner() -> ExpertPlanner:
    return ExpertPlanner(ModelOnly())


def test_ik_reaches_spawn_band_hovers(planner: ExpertPlanner) -> None:
    # Three spawn-band targets (benchy region around (0.22, -0.08),
    # design constant) at the approach clearance: IK from the pickup
    # seed must land within 5 mm — the reachability oracle.
    for xy in ((0.22, -0.08), (0.19, -0.05), (0.25, -0.11)):
        target = np.array([*xy, 0.055])
        arm, residual = planner.solve_ik(target, PICKUP_QPOS[:5])
        assert residual < 5e-3, (xy, residual)
        assert (arm >= planner.arm_low).all()
        assert (arm <= planner.arm_high).all()


def test_ik_reaches_disk_hover(planner: ExpertPlanner) -> None:
    target = np.array([0.22, 0.11, 0.065])
    _, residual = planner.solve_ik(target, PICKUP_QPOS[:5])
    assert residual < 5e-3


def test_wrist_roll_alignment_is_perpendicular(planner: ExpertPlanner) -> None:
    # For a spread of boat yaws, the aligned jaw axis must sit within
    # ~6 degrees of perpendicular to the hull's long axis (the P4
    # pinch geometry).
    arm, residual = planner.solve_ik(
        np.array([0.22, -0.08, 0.055]),
        PICKUP_QPOS[:5],
    )
    assert residual < 5e-3
    for boat_yaw in np.deg2rad((0.0, 30.0, -45.0, 80.0)):
        aligned = planner.align_wrist_roll(arm, float(boat_yaw))
        axis = planner.jaw_axis(aligned, JAW_OPEN_RAD)
        offset = (axis - (boat_yaw + np.pi / 2) + np.pi / 2) % np.pi - np.pi / 2
        assert abs(np.degrees(offset)) < 6.0, np.degrees(boat_yaw)


def test_planner_never_touches_live_state(planner: ExpertPlanner) -> None:
    # Planning runs on the scratch MjData only.
    live = mujoco.MjData(planner.model)
    before = live.qpos.copy()
    planner.solve_ik(np.array([0.22, -0.08, 0.055]), PICKUP_QPOS[:5])
    planner.align_wrist_roll(PICKUP_QPOS[:5].copy(), 0.3)
    np.testing.assert_array_equal(live.qpos, before)


def test_eval_seed_refusal() -> None:
    # The contamination guard fires before any sim work (sim=None
    # never gets touched).
    with pytest.raises(ValueError, match="frozen eval holdout"):
        run_expert_episode(None, 5)
    with pytest.raises(ValueError, match="frozen eval holdout"):
        run_expert_episode(None, DEMO_SEED_BASE - 1)


class _SlewSim(ModelOnly):
    """ModelOnly plus live data — the surface ``_smooth`` reads."""

    def __init__(self) -> None:
        super().__init__()
        self.data = mujoco.MjData(self.model)


def test_output_slew_bounds_every_command_step() -> None:
    # The output-stage limiter (owner 2026-08-16 16:53Z: smoother
    # traces): commanded per-tick steps are rate-bounded, seeded from
    # the measured pose, and converge to the requested target.
    from typing import Any, cast

    from sim.scripted_expert import ScriptedExpert

    sim = cast("Any", _SlewSim())
    expert = ScriptedExpert(sim)
    expert.SLEW_ARM_DEG, expert.SLEW_JAW_DEG = 6.0, 8.0
    start = np.rad2deg(
        np.concatenate(
            [
                sim.data.qpos[expert.planner.arm_qpos],
                [float(sim.data.qpos[expert.planner.jaw_qpos])],
            ],
        ),
    )
    target = start + np.array([90.0, -40.0, 3.0, 0.0, 200.0, 40.0])
    prev = start
    out = start
    for _ in range(60):
        out = expert._smooth(sim, target)
        step = out - prev
        assert float(np.abs(step[:5]).max()) <= 6.0 + 1e-9
        assert abs(float(step[5])) <= 8.0 + 1e-9
        prev = out
    np.testing.assert_allclose(out, target, atol=1e-9)


def test_output_slew_none_is_legacy_passthrough() -> None:
    from typing import Any, cast

    from sim.scripted_expert import ScriptedExpert

    sim = cast("Any", _SlewSim())
    expert = ScriptedExpert(sim)
    expert.SLEW_ARM_DEG = expert.SLEW_JAW_DEG = None
    cmd = np.array([120.0, -90.0, 45.0, 10.0, -170.0, 41.7])
    np.testing.assert_array_equal(expert._smooth(sim, cmd), cmd)
