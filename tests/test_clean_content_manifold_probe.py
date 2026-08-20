"""Oracles for the mechanism-(a) clean-content manifold probe math."""

import numpy as np
import pytest

from fontaine.scripts.clean_content_manifold_probe import (
    gripper_transitions,
    ks_distance,
    overlap_coefficient,
    within_episode_deltas,
)


def test_ks_identical_samples_is_zero() -> None:
    a = np.linspace(0, 1, 1000)
    assert ks_distance(a, a.copy()) == 0.0


def test_ks_disjoint_supports_is_one() -> None:
    a = np.random.default_rng(0).uniform(0, 1, 500)
    b = np.random.default_rng(1).uniform(5, 6, 700)
    assert ks_distance(a, b) == 1.0


def test_ks_half_shifted_uniform() -> None:
    # U(0,1) vs U(0.5,1.5): true KS distance is 0.5.
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, 200_000)
    b = rng.uniform(0.5, 1.5, 200_000)
    assert ks_distance(a, b) == pytest.approx(0.5, abs=0.01)


def test_ks_matches_known_small_case() -> None:
    # ECDFs of {1,2,3} vs {2,3,4} differ by at most 1/3.
    assert ks_distance(np.array([1.0, 2, 3]), np.array([2.0, 3, 4])) == pytest.approx(
        1 / 3,
    )


def test_ovl_identical_is_one() -> None:
    a = np.random.default_rng(3).normal(0, 1, 50_000)
    assert overlap_coefficient(a, a.copy()) == pytest.approx(1.0, abs=1e-6)


def test_ovl_disjoint_is_zero() -> None:
    a = np.random.default_rng(4).uniform(0, 1, 5000)
    b = np.random.default_rng(5).uniform(10, 11, 5000)
    assert overlap_coefficient(a, b) == pytest.approx(0.0, abs=1e-6)


def test_ovl_half_shifted_uniform() -> None:
    # U(0,1) vs U(0.5,1.5): true overlap coefficient is 0.5.
    rng = np.random.default_rng(6)
    a = rng.uniform(0, 1, 200_000)
    b = rng.uniform(0.5, 1.5, 200_000)
    assert overlap_coefficient(a, b) == pytest.approx(0.5, abs=0.02)


def test_gripper_transitions_square_wave() -> None:
    # Two full open-close cycles = 4 flips (lo=0.25, hi=0.75 on a 0..1 wave).
    traj = np.array([0.0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    assert gripper_transitions(traj, 0.25, 0.75) == 4


def test_gripper_transitions_mid_band_chatter_ignored() -> None:
    # Oscillation inside the hysteresis band must not count as flips.
    traj = np.array([0.0, 0.4, 0.6, 0.4, 0.6, 0.4, 1.0])
    assert gripper_transitions(traj, 0.25, 0.75) == 1


def test_gripper_transitions_flat_is_zero() -> None:
    assert gripper_transitions(np.full(100, 0.9), 0.25, 0.75) == 0


def test_within_episode_deltas_masks_boundaries() -> None:
    arr = np.array([[0.0], [1.0], [10.0], [11.0]])
    ep = np.array([0, 0, 1, 1])
    d = within_episode_deltas(arr, ep)
    # The 1.0 -> 10.0 jump crosses an episode boundary and must be dropped.
    assert d.shape == (2, 1)
    assert d.flatten().tolist() == [1.0, 1.0]
