"""Oracles for the camera-mount material split (sim-mount-material-split):
the opt-in mount_material='v1' path may ONLY (a) detach the gripper's
wrist-roll geom from the shared material with its rgba copied — a
render-identical hand-off that makes the material mount-exclusive — and
(b) grade that material. The default path stays byte-identical, physics
and every RNG stream stay untouched either way."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.so101_sim import SO101Sim

SHARED = "wrist_roll_follower_so101_v1_material"


def _physics_only(sim: SO101Sim) -> SO101Sim:
    sim.observe = lambda: None  # type: ignore[method-assign]
    return sim


@pytest.fixture(scope="module")
def default_sim() -> SO101Sim:
    return _physics_only(SO101Sim(render_style="v3"))


@pytest.fixture(scope="module")
def mount_sim() -> SO101Sim:
    return _physics_only(SO101Sim(render_style="v3", mount_material="v1"))


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


def _geoms_using(sim: SO101Sim, mat_id: int) -> set[str]:
    model = sim.model
    return {
        model.body(model.geom_bodyid[g]).name
        for g in range(model.ngeom)
        if model.geom_matid[g] == mat_id
    }


def test_default_path_keeps_shared_material(default_sim: SO101Sim) -> None:
    # default: the material stays shared gripper+mount at recolor black
    model = default_sim.model
    for prefix in ("", "leader-"):
        mat = model.material(prefix + SHARED)
        assert _geoms_using(default_sim, mat.id) == {
            prefix + "gripper",
            prefix + "camera_mount",
        }
        assert tuple(np.round(model.mat_rgba[mat.id], 6)) == (0.13, 0.13, 0.14, 1.0)


def test_v1_splits_and_grades_mount_material(mount_sim: SO101Sim) -> None:
    model = mount_sim.model
    grade = SO101Sim.MOUNT_MATERIAL_V1
    for prefix in ("", "leader-"):
        mat = model.material(prefix + SHARED)
        # the material is now mount-exclusive and graded
        assert _geoms_using(mount_sim, mat.id) == {prefix + "camera_mount"}
        assert tuple(np.round(model.mat_rgba[mat.id, :3], 6)) == pytest.approx(
            grade["rgba"],
        )
        assert float(model.mat_specular[mat.id]) == pytest.approx(grade["specular"])
        assert float(model.mat_shininess[mat.id]) == pytest.approx(grade["shininess"])
        # the detached gripper geom carries the recolor black in geom_rgba
        gripper = model.body(prefix + "gripper").id
        detached = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == gripper
            and model.geom_matid[g] == -1
            # collision geoms are born material-less with default rgba;
            # the detached visual geom carries the copied recolor black
            and tuple(np.round(model.geom_rgba[g], 6)) == (0.13, 0.13, 0.14, 1.0)
        ]
        assert len(detached) == 1


def test_v1_touches_nothing_else(
    default_sim: SO101Sim,
    mount_sim: SO101Sim,
) -> None:
    before = _material_state(default_sim)
    after = _material_state(mount_sim)
    for name in before:
        if "wrist_roll_follower" not in name:
            assert after[name] == before[name], name


def test_v1_consumes_no_rng_draws(
    default_sim: SO101Sim,
    mount_sim: SO101Sim,
) -> None:
    for seed in (0, 7):
        default_sim.reset(seed, appearance_seed=123)
        mount_sim.reset(seed, appearance_seed=123)
        np.testing.assert_array_equal(default_sim.data.qpos, mount_sim.data.qpos)
        assert (
            default_sim._noise_rng.bit_generator.state
            == mount_sim._noise_rng.bit_generator.state
        )


def test_mount_material_validated() -> None:
    with pytest.raises(ValueError, match="mount_material"):
        SO101Sim(mount_material="v2")


def test_split_alone_renders_byte_identical() -> None:
    # the hand-off contract: detaching the gripper geom with the
    # material's rgba copied is invisible — mjv's material-less defaults
    # (spec 0.5 shin 0.5 refl 0 emis 0) equal the shipped material's
    pytest.importorskip("mujoco.egl")
    try:
        plain = SO101Sim(render_style="v3")
        obs_plain = plain.reset(3)
    except Exception as error:  # noqa: BLE001 — any GL failure means skip, not fail
        pytest.skip(f"no GL context: {error}")
    split = SO101Sim(render_style="v3")
    split._split_mount_material()
    obs_split = split.reset(3)
    np.testing.assert_array_equal(obs_plain.top, obs_split.top)
    np.testing.assert_array_equal(obs_plain.wrist, obs_split.wrist)
