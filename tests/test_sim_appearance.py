"""Oracles for the visual-matching v1 changes (prereg 2026-08-12):
appearance is decoupled from physics by construction — same seed must
give bit-identical settled state no matter the appearance seed or
render style, and the spawn stream must still match the banked sim100
v0 run."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.so101_sim import SO101Sim

# spawn_xy for seeds 0..2 as recorded by the sim100 v0 eval
# (outputs/sim/eval100/er60k.json, episodes[0:3].spawn_xy) — the spawn
# RNG stream and draw order must survive appearance changes.
BANKED_SPAWN = {
    0: (0.2427721265491091, 0.007140402119374163),
    1: (0.23338662185251927, 0.03777086633466709),
    2: (0.21462091006869874, 0.008432101453635547),
}


def _physics_only(sim: SO101Sim) -> SO101Sim:
    # reset() returns the first observation, which needs a GL context;
    # these oracles are about the physics state, so stub the render.
    sim.observe = lambda: None  # type: ignore[method-assign]
    return sim


@pytest.fixture(scope="module")
def sim() -> SO101Sim:
    return _physics_only(SO101Sim())


def test_spawn_stream_matches_banked_v0(sim: SO101Sim) -> None:
    for seed, expected in BANKED_SPAWN.items():
        sim.reset(seed)
        assert sim.reset_spawn_xy == pytest.approx(expected, abs=1e-12)


def test_qpos_identical_across_appearance_seeds(sim: SO101Sim) -> None:
    sim.reset(3)
    baseline = sim.data.qpos.copy()
    for appearance_seed in (17, 999):
        sim.reset(3, appearance_seed=appearance_seed)
        np.testing.assert_array_equal(sim.data.qpos, baseline)


def test_qpos_identical_across_render_styles() -> None:
    qpos = {}
    for style in ("v0", "v1", "v2", "v3", "v4"):
        sim = _physics_only(SO101Sim(render_style=style))
        sim.reset(5)
        qpos[style] = sim.data.qpos.copy()
    np.testing.assert_array_equal(qpos["v0"], qpos["v1"])
    np.testing.assert_array_equal(qpos["v0"], qpos["v2"])
    np.testing.assert_array_equal(qpos["v0"], qpos["v3"])
    np.testing.assert_array_equal(qpos["v0"], qpos["v4"])


def test_v3_clutter_draws_are_physics_inert() -> None:
    # The drawn stand-ins are contype/conaffinity 0 in the scene XML;
    # a v3 reset that moves them must leave the settled state and the
    # spawn stream untouched (same-seed check against the v2 style).
    v2 = _physics_only(SO101Sim(render_style="v2"))
    v3 = _physics_only(SO101Sim(render_style="v3"))
    for seed in (0, 7):
        v2.reset(seed)
        v3.reset(seed)
        np.testing.assert_array_equal(v2.data.qpos, v3.data.qpos)
        assert v2.reset_spawn_xy == v3.reset_spawn_xy
    assert set(v3._clutter_drawn) == {"mouse", "mug", "laptop", "pcb"}


def test_appearance_seed_changes_only_appearance(sim: SO101Sim) -> None:
    sim.reset(3, appearance_seed=17)
    tint_a = sim.model.mat_rgba[sim._benchy_mat].copy()
    sun_a = sim.model.light_diffuse[sim._sun].copy()
    sim.reset(3, appearance_seed=999)
    assert not np.array_equal(tint_a, sim.model.mat_rgba[sim._benchy_mat])
    assert not np.array_equal(sun_a, sim.model.light_diffuse[sim._sun])


def test_render_style_validated() -> None:
    with pytest.raises(ValueError, match="render_style"):
        SO101Sim(render_style="v9")
    with pytest.raises(ValueError, match="post_backend"):
        SO101Sim(post_backend="cupy")


@pytest.mark.gpu
def test_torch_post_matches_numpy_reference() -> None:
    # The CUDA post path (owner-approved 08-12) must reproduce the
    # numpy float64 reference to within float32 rounding: same seeded
    # noise stream, arithmetic-only differences, <= 2/255 counts.
    reference = SO101Sim(render_style="v3", post_backend="numpy")
    fast = SO101Sim(render_style="v3", post_backend="torch")
    for seed in (0, 7):
        obs_ref = reference.reset(seed)
        obs_fast = fast.reset(seed)
        for name in ("top", "wrist"):
            a = getattr(obs_ref, name).astype(np.int16)
            b = getattr(obs_fast, name).astype(np.int16)
            diff = np.abs(a - b)
            assert diff.max() <= 2, f"{name} seed {seed}: max diff {diff.max()}"
            assert (diff > 0).mean() < 0.05, f"{name} seed {seed}: widespread drift"


def test_bracket_appearance_validated() -> None:
    with pytest.raises(ValueError, match="bracket_appearance"):
        SO101Sim(bracket_appearance="v2")


def test_bracket_real_is_render_only() -> None:
    # bracket_appearance='real' (owner 2026-08-16): leader bracket mesh
    # hidden, follower camera_box2 surfaced dark — groups/rgba only, so
    # the settled physics state must stay bit-identical to v1.
    v1 = _physics_only(SO101Sim())
    real = _physics_only(SO101Sim(bracket_appearance="real"))

    mesh = [
        g
        for g in range(real.model.ngeom)
        if real.model.geom_bodyid[g] == real.model.body("leader-camera_mount").id
        and real.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    assert [real.model.geom_group[g] for g in mesh] == [3]
    box_v1 = v1.model.geom("camera_box2")
    box_real = real.model.geom("camera_box2")
    assert v1.model.geom_group[box_v1.id] == 3  # v1 untouched
    assert real.model.geom_group[box_real.id] == 2
    assert tuple(real.model.geom_rgba[box_real.id]) == (0.08, 0.08, 0.09, 1.0)
    # leader boxes stay hidden in both variants
    for name in ("leader-camera_box1", "leader-camera_box2"):
        assert real.model.geom_group[real.model.geom(name).id] == 3

    for seed in (0, 3):
        v1.reset(seed)
        real.reset(seed)
        assert np.array_equal(v1.data.qpos, real.data.qpos), f"seed {seed}"


def test_wrist_pose_validated() -> None:
    with pytest.raises(ValueError, match="wrist_pose"):
        SO101Sim(wrist_pose="v2")


def test_clutter_appearance_validated() -> None:
    with pytest.raises(ValueError, match="clutter_appearance"):
        SO101Sim(clutter_appearance="v2")


def test_patched_clutter_preserves_physics_and_streams() -> None:
    # Clutter-patch promotion (2026-08-18): the paste consumes no RNG
    # and the pose/presence draws are unchanged, so the same seed must
    # give bit-identical settled physics, spawn, drawn-clutter slots,
    # plate affine and appearance draws under both modes — the v3
    # slot-pairing guarantee the gate evidence relied on.
    standins = _physics_only(SO101Sim(clutter_appearance="standins"))
    patched = _physics_only(SO101Sim())
    assert patched.clutter_appearance == "patched"  # promotion default
    for seed in (0, 7):
        standins.reset(seed)
        patched.reset(seed)
        np.testing.assert_array_equal(standins.data.qpos, patched.data.qpos)
        assert standins.reset_spawn_xy == patched.reset_spawn_xy
        assert set(standins._clutter_drawn) == set(patched._clutter_drawn)
        for name, (pos, yaw) in standins._clutter_drawn.items():
            np.testing.assert_array_equal(pos, patched._clutter_drawn[name][0])
            assert yaw == patched._clutter_drawn[name][1]
        np.testing.assert_array_equal(standins._active_gain, patched._active_gain)
        np.testing.assert_array_equal(standins._active_bias, patched._active_bias)
        np.testing.assert_array_equal(
            standins.model.mat_rgba[standins._benchy_mat],
            patched.model.mat_rgba[patched._benchy_mat],
        )
        # the stand-ins rest off-frustum under patched, drawn under
        # standins — the top pass drops them from render/mask/shadow
        for name in patched._clutter_base:
            np.testing.assert_array_equal(
                patched.model.geom_pos[patched.model.geom(name).id],
                patched.V3_ABSENT_POS,
            )
            np.testing.assert_array_equal(
                standins.model.geom_pos[standins.model.geom(name).id],
                standins._clutter_drawn[name][0],
            )


def test_clutter_patch_camera_model_frozen() -> None:
    # The paste's inlined camera model must stay pinned to the mining
    # pass's measured values (fontaine/scripts/make_clean_plates.py —
    # "matching sim/so101_sim.py exactly"); a drift here silently
    # misplaces every pasted crop.
    from sim import clutter_patch as cp

    assert pytest.approx((-0.02, -0.125, 0.555)) == cp.CAM_POS
    assert (cp.WIDTH, cp.HEIGHT) == (640, 480)
    assert pytest.approx((480 / 2.0) / np.tan(np.deg2rad(52.0) / 2.0)) == cp.F_DIST
    assert pytest.approx((319.5, 239.5)) == cp.CENTER
    assert cp.ABSENT_POS == SO101Sim.V3_ABSENT_POS
    expected_r = np.stack(
        [
            np.array([0.0, -1.0, 0.0]),
            np.array([0.9063, 0.0, 0.4226]),
            np.cross([0.0, -1.0, 0.0], [0.9063, 0.0, 0.4226]),
        ],
        axis=1,
    )
    np.testing.assert_allclose(cp.CAM_R, expected_r)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Separable Chebyshev (box) binary dilation — covers the blur
    kernel's square footprint."""
    out = mask.copy()
    for axis in (0, 1):
        acc = out.copy()
        for shift in range(1, radius + 1):
            acc |= np.roll(out, shift, axis=axis)
            acc |= np.roll(out, -shift, axis=axis)
        out = acc
    return out


@pytest.mark.gpu
def test_patched_clutter_render_oracles() -> None:
    # Promotion render oracles (registered with the queue item): the
    # wrist frame is bit-exact across modes (it renders the canonical
    # scene either way), and the top frame is bit-exact outside the
    # clutter-affected pixels — the stand-ins' screen footprint plus
    # the paste delta, dilated by the composite blur radius (weight
    # and foreground differences spread that far, nothing else moves;
    # the noise stream is seed-identical across modes).
    import mujoco as mj

    standins = SO101Sim(
        render_style="v3",
        post_backend="numpy",
        clutter_appearance="standins",
    )
    patched = SO101Sim(render_style="v3", post_backend="numpy")
    radius = int(np.ceil(2.5 * SO101Sim.V1_BLUR_SIGMA)) + 1
    clutter_ids = np.array(
        sorted(standins.model.geom(n).id for n in standins._clutter_base),
    )
    saw_clutter = False
    for seed in (0, 7):
        obs_standins = standins.reset(seed)
        obs_patched = patched.reset(seed)
        np.testing.assert_array_equal(obs_standins.wrist, obs_patched.wrist)
        # stand-in footprint from a segmentation pass at the drawn
        # poses (the standins sim rests there after observe())
        renderer = standins.renderer
        renderer.enable_segmentation_rendering()
        renderer.update_scene(standins.data, camera="top_cam")
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        footprint_src = (seg[..., 1] == mj.mjtObj.mjOBJ_GEOM.value) & np.isin(
            seg[..., 0],
            clutter_ids,
        )
        # the RGB render anti-aliases clutter edges ~1px past the hard
        # segmentation footprint; then the composite remaps the source
        # render through the fisheye — dilate in source space, push
        # through the same gather into output space
        footprint = (
            standins._remap(
                _dilate(footprint_src, 2)[..., None].astype(np.float64),
            )[..., 0]
            > 0
        )
        paste_delta = np.any(
            standins._active_top_plate != patched._active_top_plate,
            axis=-1,
        )
        affected = _dilate(footprint | paste_delta, radius)
        outside = ~affected
        np.testing.assert_array_equal(
            obs_standins.top[outside],
            obs_patched.top[outside],
        )
        if affected.any():
            saw_clutter = True
            assert np.any(obs_standins.top != obs_patched.top)
    assert saw_clutter  # at least one seed drew present clutter


@pytest.mark.gpu
def test_patched_clutter_v4_wrist_bit_exact() -> None:
    # v4 promotion: the shadow pass is top-only, so the wrist stays
    # bit-exact across modes there too.
    standins = SO101Sim(
        render_style="v4",
        post_backend="numpy",
        clutter_appearance="standins",
    )
    patched = SO101Sim(render_style="v4", post_backend="numpy")
    for seed in (0,):
        np.testing.assert_array_equal(
            standins.reset(seed).wrist,
            patched.reset(seed).wrist,
        )


def test_wrist_pose_refit_is_camera_only() -> None:
    # wrist_pose='refit' (queue wrist-cam-pose-refit, fitted 2026-08-17):
    # a camera pos/quat change only — the settled physics state must stay
    # bit-identical to v1, the v1 pose oracle-pinned, and the refit pose
    # must match the shipped fit record (outputs/sim/wrist_refit/fit.json).
    v1 = _physics_only(SO101Sim())
    refit = _physics_only(SO101Sim(wrist_pose="refit"))

    cam_v1 = v1.model.camera("wrist_cam")
    cam_refit = refit.model.camera("wrist_cam")
    assert cam_v1.pos == pytest.approx((0.02416, -0.05504, 0.03225))
    assert cam_v1.quat == pytest.approx((-0.24345, -0.05192, 0.02663, 0.96816))
    assert cam_refit.pos == pytest.approx((0.00475, -0.08292, 0.00056))
    assert cam_refit.quat == pytest.approx((-0.14755, -0.10195, -0.20516, 0.96216))
    # the fitted quat ships normalized, and the lens-source fovy is frozen
    # across the flag (the fisheye init owns it; the pose never touches it)
    assert float(np.linalg.norm(cam_refit.quat)) == pytest.approx(1.0, abs=1e-4)
    assert cam_refit.fovy[0] == cam_v1.fovy[0]

    for seed in (0, 3):
        v1.reset(seed)
        refit.reset(seed)
        assert np.array_equal(v1.data.qpos, refit.data.qpos), f"seed {seed}"
