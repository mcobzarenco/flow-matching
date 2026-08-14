"""Oracles for the arm photometric grade (sim-arm-photometric-links):
the opt-in arm_photometrics='v1' path may ONLY rewrite the link/servo
material grades — the default path stays byte-identical, physics and
every RNG stream stay untouched either way, and the excluded materials
(moving jaws, wrist_roll_follower) keep their recolor values."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.so101_sim import SO101Sim


def _physics_only(sim: SO101Sim) -> SO101Sim:
    sim.observe = lambda: None  # type: ignore[method-assign]
    return sim


@pytest.fixture(scope="module")
def default_sim() -> SO101Sim:
    return _physics_only(SO101Sim(render_style="v3"))


@pytest.fixture(scope="module")
def graded_sim() -> SO101Sim:
    return _physics_only(SO101Sim(render_style="v3", arm_photometrics="v1"))


def _material_state(sim: SO101Sim) -> dict[str, tuple]:
    model = sim.model
    return {
        model.mat(i).name: (
            tuple(np.round(model.mat_rgba[i], 6)),
            float(model.mat_specular[i]),
            float(model.mat_shininess[i]),
        )
        for i in range(model.nmat)
    }


def test_default_path_materials_unchanged(default_sim: SO101Sim) -> None:
    # the flat recolor contract: black arm, orange follower jaw, servo
    # casings at the menagerie 0.1 gray, spec/shin at the shipped 0.5
    state = _material_state(default_sim)
    assert state["upper_arm_so101_v1_material"] == ((0.13, 0.13, 0.14, 1.0), 0.5, 0.5)
    assert state["sts3215_03a_v1_material"][0][:3] == (0.1, 0.1, 0.1)
    assert state["moving_jaw_so101_v1_material"][0][:3] == (0.95, 0.45, 0.1)


def test_v1_grades_link_and_servo_materials(graded_sim: SO101Sim) -> None:
    state = _material_state(graded_sim)
    pla = SO101Sim.ARM_PHOTOMETRICS_V1["pla"]
    servo = SO101Sim.ARM_PHOTOMETRICS_V1["servo"]
    for name in (
        "upper_arm_so101_v1_material",
        "leader-upper_arm_so101_v1_material",
        "base_so101_v2_material",
        "wrist_roll_pitch_so101_v2_material",
    ):
        rgba, spec, shin = state[name]
        assert rgba[:3] == pytest.approx(pla["rgba"])
        assert (spec, shin) == pytest.approx((pla["specular"], pla["shininess"]))
    for name in ("sts3215_03a_v1_material", "leader-sts3215_03a_no_horn_v1_material"):
        rgba, spec, shin = state[name]
        assert rgba[:3] == pytest.approx(servo["rgba"])
        assert (spec, shin) == pytest.approx((servo["specular"], servo["shininess"]))


def test_v1_leaves_excluded_materials_alone(
    default_sim: SO101Sim,
    graded_sim: SO101Sim,
) -> None:
    before = _material_state(default_sim)
    after = _material_state(graded_sim)
    for name in (
        "moving_jaw_so101_v1_material",  # the follower's orange jaw
        "leader-moving_jaw_so101_v1_material",
        "wrist_roll_follower_so101_v1_material",  # gripper/mount territory
        "leader-wrist_roll_follower_so101_v1_material",
    ):
        assert after[name] == before[name]
    # and nothing outside the arm: table/benchy materials untouched
    for name in before:
        if "so101" not in name and "sts3215" not in name:
            assert after[name] == before[name]


def test_v1_consumes_no_rng_draws(
    default_sim: SO101Sim,
    graded_sim: SO101Sim,
) -> None:
    # same (seed, appearance_seed) => bit-identical settled physics AND
    # bit-identical sensor-noise stream state on both paths
    for seed in (0, 7):
        default_sim.reset(seed, appearance_seed=123)
        graded_sim.reset(seed, appearance_seed=123)
        np.testing.assert_array_equal(default_sim.data.qpos, graded_sim.data.qpos)
        assert (
            default_sim._noise_rng.bit_generator.state
            == graded_sim._noise_rng.bit_generator.state
        )


def test_arm_photometrics_validated() -> None:
    with pytest.raises(ValueError, match="arm_photometrics"):
        SO101Sim(arm_photometrics="v2")
