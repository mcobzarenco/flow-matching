"""Oracles for the grasp/contact instrument (grpo-reward-patch-prereg,
owner-approved 09:16Z 2026-08-14): `benchy_grip_contacts()` is a pure
mjData read (no RNG draws, no state writes), the settled reset is
contact-free (rides the reset-strike gate), a scripted two-sided close
produces the pinch code, and the EpisodeResult grip/distance derived
channels do the registered arithmetic — including nan (never a silent
0) on rows recorded before the instrument."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.rollout_sim import EpisodeResult, _grip_code
from sim.so101_sim import HOME_DEGREES, JOINTS, SO101Sim


@pytest.fixture(scope="module")
def sim() -> SO101Sim:
    instance = SO101Sim()
    instance.observe = lambda: None  # type: ignore[method-assign]
    return instance


def test_grip_code_encoding() -> None:
    assert _grip_code((False, False)) == 0
    assert _grip_code((True, False)) == 1
    assert _grip_code((False, True)) == 2
    assert _grip_code((True, True)) == 3


def test_reset_is_contact_free_and_query_is_pure(sim: SO101Sim) -> None:
    for seed in (0, 7):
        sim.reset(seed)
        qpos = sim.data.qpos.copy()
        noise_state = sim._noise_rng.bit_generator.state
        assert sim.benchy_grip_contacts() == (False, False)
        # pure read: physics and the shared noise stream are untouched
        np.testing.assert_array_equal(sim.data.qpos, qpos)
        assert sim._noise_rng.bit_generator.state == noise_state


def test_scripted_close_reads_two_sided_pinch(sim: SO101Sim) -> None:
    """Teleport the benchy between the jaws at the settled home pose and
    close the gripper: both sides must register (grip code 3). The
    teleport mirrors reset()'s own benchy placement mechanics."""
    sim.reset(0)
    mid = (sim.data.xpos[sim._gripper_body] + sim.data.xpos[sim._moving_jaw_body]) / 2
    adr = sim._benchy_qpos
    sim.data.qpos[adr : adr + 3] = (mid[0] + 0.02, mid[1], 0.001)
    sim.data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
    vadr = sim.model.joint("benchy_free").dofadr[0]
    sim.data.qvel[vadr : vadr + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    close = np.array(HOME_DEGREES, dtype=float)
    close[JOINTS.index("gripper")] = -10.0
    for _ in range(60):
        sim.step(close)
    assert sim.benchy_grip_contacts() == (True, True)


def _result(distance_cm: list[float], grip: list[int]) -> EpisodeResult:
    return EpisodeResult(
        seed=0,
        initial_cm=distance_cm[0],
        min_cm=min(distance_cm),
        final_cm=distance_cm[-1],
        success_tick=None,
        spawn_xy=(0.0, 0.0),
        reset_strikes=0,
        final_z_mm=1.0,
        final_upright=1.0,
        ticks=len(distance_cm) - 1,
        distance_cm=distance_cm,
        grip=grip,
    )


def test_derived_channels_split_progress_by_grip() -> None:
    # ticks: shove 2 cm closer ungrasped, carry 3 cm closer pinched,
    # then knocked 1 cm away ungrasped
    row = _result([10.0, 8.0, 5.0, 6.0], [0, 1, 3, 0])
    assert row.progress_final_cm == pytest.approx(4.0)
    assert row.grasped_progress_cm == pytest.approx(3.0)
    assert row.ungrasped_displacement_cm == pytest.approx(3.0)  # 2 + 1
    assert row.max_setback_cm == pytest.approx(0.0)


def test_max_setback_sees_mid_episode_excursion() -> None:
    # endpoint reads -0.5 cm (under the -1 knock bar) but the boat was
    # batted 4 cm out mid-episode — the excursion channel must see it
    row = _result([10.0, 14.0, 10.5], [0, 0, 0])
    assert row.progress_final_cm == pytest.approx(-0.5)
    assert row.max_setback_cm == pytest.approx(4.0)
    assert row.grasped_progress_cm == pytest.approx(0.0)
    assert row.ungrasped_displacement_cm == pytest.approx(7.5)


def test_pre_instrument_rows_read_nan_not_zero() -> None:
    row = _result([10.0, 8.0], [])
    assert np.isnan(row.grasped_progress_cm)
    assert np.isnan(row.ungrasped_displacement_cm)
    # trace-only channels still work without a grip trace
    assert row.max_setback_cm == pytest.approx(0.0)
