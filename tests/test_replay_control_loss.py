"""Oracles for the replay control-loss probe (sim.replay_control_loss):
the rotation term must match its closed form — for a relative rotation
by theta, ||R1 - R2||_F = 2 sqrt2 sin(theta/2), so the arcsin term is
exactly theta/2 — and the FK pass must be deterministic, param-free,
and driven by the arm joints alone."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.replay_control_loss import control_loss_terms, ee_trajectory
from sim.so101_sim import SO101Sim
from sim.sysid_servo import MENAGERIE, set_params


@pytest.fixture(scope="module")
def sim() -> SO101Sim:
    return SO101Sim(width=64, height=48, render_style="v0")


def _rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_rotation_term_is_half_geodesic_angle() -> None:
    rng = np.random.default_rng(0)
    thetas = np.array([0.0, 0.1, 0.5, 1.0, np.pi / 2, 3.0])
    base = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    base *= np.sign(np.linalg.det(base))
    mats_a = np.stack([base] * len(thetas))
    mats_b = np.stack([base @ _rotz(t) for t in thetas])
    pos = np.zeros((len(thetas), 3))
    trans, rot = control_loss_terms(pos, mats_a, pos, mats_b)
    np.testing.assert_allclose(trans, 0.0, atol=1e-12)
    np.testing.assert_allclose(rot, thetas / 2.0, atol=1e-9)


def test_identical_trajectories_score_zero(sim: SO101Sim) -> None:
    data = mujoco.MjData(sim.model)
    traj = np.array(
        [[0.0, -30.0, 40.0, 20.0, 10.0, 5.0], [4.0, -20.0, 30.0, 10.0, 0.0, 0.0]],
    )
    pos, mat = ee_trajectory(sim, data, traj)
    trans, rot = control_loss_terms(pos, mat, *ee_trajectory(sim, data, traj))
    assert trans.max() == 0.0
    assert rot.max() == 0.0


def test_fk_ignores_servo_params_and_moves_with_arm(sim: SO101Sim) -> None:
    data = mujoco.MjData(sim.model)
    traj = np.array([[0.0, -30.0, 40.0, 20.0, 10.0, 5.0]])
    pos_a, mat_a = ee_trajectory(sim, data, traj)
    set_params(sim, MENAGERIE)  # actuator/dof params must not touch FK
    pos_b, mat_b = ee_trajectory(sim, data, traj)
    np.testing.assert_array_equal(pos_a, pos_b)
    np.testing.assert_array_equal(mat_a, mat_b)

    # wrist_roll (arm joint 4) by +10 deg rotates the site frame by
    # exactly 10 deg about the roll axis: rot term = half of that.
    rolled = traj.copy()
    rolled[0, 4] += 10.0
    _, rot = control_loss_terms(pos_a, mat_a, *ee_trajectory(sim, data, rolled))
    np.testing.assert_allclose(rot, np.deg2rad(10.0) / 2.0, atol=1e-9)

    # the jaw joint (index 5) moves a child body, not the site
    jawed = traj.copy()
    jawed[0, 5] += 25.0
    trans, rot = control_loss_terms(pos_a, mat_a, *ee_trajectory(sim, data, jawed))
    assert trans.max() == 0.0
    assert rot.max() == 0.0
