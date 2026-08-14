"""Oracles for the arm micro-texture (sim-arm-texture-followup): the
opt-in arm_texture='v1' path builds deterministic static fields at init
from a PRIVATE pinned Generator — the spawn/appearance/noise streams
stay untouched, physics is bit-identical either way — and the pixel
math is identity at zero parameters, arm-population-local by masking."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.so101_sim import SO101Sim


def _physics_only(sim: SO101Sim) -> SO101Sim:
    sim.observe = lambda: None  # type: ignore[method-assign]
    return sim


@pytest.fixture(scope="module")
def graded_sim() -> SO101Sim:
    return _physics_only(SO101Sim(render_style="v3", arm_photometrics="v1"))


@pytest.fixture(scope="module")
def textured_sim() -> SO101Sim:
    return _physics_only(
        SO101Sim(render_style="v3", arm_photometrics="v1", arm_texture="v1"),
    )


def test_arm_texture_validated() -> None:
    with pytest.raises(ValueError, match="arm_texture"):
        SO101Sim(arm_texture="v2")
    # gated as the combination fitted on top of the grade
    with pytest.raises(ValueError, match="arm_photometrics"):
        SO101Sim(render_style="v3", arm_texture="v1")
    # composite path only
    with pytest.raises(ValueError, match="composite"):
        SO101Sim(render_style="v1", arm_photometrics="v1", arm_texture="v1")


def test_fields_deterministic_and_normalized(textured_sim: SO101Sim) -> None:
    mod = textured_sim._texture_mod
    assert mod.shape == textured_sim._render_size
    assert abs(float(mod.mean())) < 1e-9
    assert abs(float(mod.std()) - 1.0) < 1e-9
    # speckle: pla is texture-only (no glints registered), servo carries
    # the fitted glint tail at its pinned density
    params = SO101Sim.ARM_TEXTURE_V1
    for name, speckle in textured_sim._texture_speckle.items():
        density = params[name]["speckle_density"]
        lit = float((speckle > 0).mean())
        assert speckle.min() >= 0.0 and speckle.max() <= 1.0
        if density <= 0.0:
            assert lit == 0.0
        else:
            # the softening passes spread faint skirts well past the
            # binary-stage density; the STRONG cores stay density-bounded
            # and the peaks at full strength
            assert lit >= density
            assert (speckle >= 0.5).mean() <= 2.0 * density
            assert speckle.max() == 1.0
    # deterministic: a second instance builds bit-identical fields
    other = _physics_only(
        SO101Sim(render_style="v3", arm_photometrics="v1", arm_texture="v1"),
    )
    np.testing.assert_array_equal(mod, other._texture_mod)
    for name in textured_sim._texture_speckle:
        np.testing.assert_array_equal(
            textured_sim._texture_speckle[name],
            other._texture_speckle[name],
        )


def test_texture_geom_sets_pinned(textured_sim: SO101Sim) -> None:
    counts = {name: len(ids) for name, ids in textured_sim._texture_geoms.items()}
    assert counts == SO101Sim.ARM_TEXTURE_GEOM_COUNTS
    # populations are disjoint
    overlap = set(textured_sim._texture_geoms["pla"]) & set(
        textured_sim._texture_geoms["servo"],
    )
    assert not overlap


def test_pixel_math_identity_at_zero() -> None:
    rng = np.random.default_rng(0)
    pixels = rng.uniform(0, 255, size=(500, 3))
    mod = rng.standard_normal(500)
    speckle = rng.random(500)
    zero = {"amplitude": 0.0, "speckle_density": 0.0, "speckle_gain": 0.0}
    np.testing.assert_array_equal(
        SO101Sim._texture_pixels(pixels, mod, speckle, zero),
        pixels,
    )


def test_pixel_math_speckle_pushes_toward_white() -> None:
    pixels = np.full((100, 3), 40.0)
    mod = np.zeros(100)
    speckle = np.linspace(0.0, 1.0, 100)
    params = {"amplitude": 0.0, "speckle_density": 0.05, "speckle_gain": 0.8}
    out = SO101Sim._texture_pixels(pixels, mod, speckle, params)
    assert np.all(out >= pixels)  # never darkens
    assert np.all(out <= 255.0)  # never overshoots white
    assert np.all(np.diff(out[:, 0]) >= 0)  # monotone in speckle strength


def test_v1_consumes_no_rng_draws(
    graded_sim: SO101Sim,
    textured_sim: SO101Sim,
) -> None:
    # same (seed, appearance_seed) => bit-identical settled physics AND
    # bit-identical sensor-noise stream state on both paths
    for seed in (0, 7):
        graded_sim.reset(seed, appearance_seed=123)
        textured_sim.reset(seed, appearance_seed=123)
        np.testing.assert_array_equal(graded_sim.data.qpos, textured_sim.data.qpos)
        assert (
            graded_sim._noise_rng.bit_generator.state
            == textured_sim._noise_rng.bit_generator.state
        )


# --- arm_texture='v2' (sim-arm-surface-texture-mjspec): true surface
# texture via the mjSpec recompile path. The HARD BAR (registered before
# any read): the recompiled model is physics-identical to the
# from_xml_path baseline — every physics field bit-equal, ids
# unrenumbered, qpos trajectories bit-equal, shared RNG streams
# untouched; the texture is PLA-local in the render.

# every MjModel field that feeds physics (or the id maps the paired
# reads key on); mat_* / tex_* fields are the only intended delta
PHYSICS_FIELDS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "body_pos",
    "body_quat",
    "dof_damping",
    "dof_armature",
    "dof_frictionloss",
    "jnt_range",
    "jnt_stiffness",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_ctrlrange",
    "actuator_forcerange",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_friction",
    "geom_contype",
    "geom_conaffinity",
    "geom_solref",
    "geom_solimp",
    "geom_margin",
    "geom_gap",
    "geom_bodyid",
    "geom_matid",
    "geom_sameframe",
    "qpos0",
)


@pytest.fixture(scope="module")
def surface_sim() -> SO101Sim:
    return _physics_only(
        SO101Sim(render_style="v3", arm_photometrics="v1", arm_texture="v2"),
    )


def test_v2_validated() -> None:
    with pytest.raises(ValueError, match="arm_texture"):
        SO101Sim(arm_texture="v3")
    with pytest.raises(ValueError, match="arm_photometrics"):
        SO101Sim(render_style="v3", arm_texture="v2")
    with pytest.raises(ValueError, match="composite"):
        SO101Sim(render_style="v1", arm_photometrics="v1", arm_texture="v2")


def test_v2_model_physics_bit_equal(
    graded_sim: SO101Sim,
    surface_sim: SO101Sim,
) -> None:
    # same reset first: reset() jitters benchy/table/light materials on
    # the appearance stream, so the module-shared instances' scene
    # materials only match in a matched reset state
    graded_sim.reset(0, appearance_seed=123)
    surface_sim.reset(0, appearance_seed=123)
    base, patched = graded_sim.model, surface_sim.model
    assert (base.nq, base.nbody, base.ngeom, base.nmat, base.nu) == (
        patched.nq,
        patched.nbody,
        patched.ngeom,
        patched.nmat,
        patched.nu,
    )
    for field in PHYSICS_FIELDS:
        np.testing.assert_array_equal(
            getattr(base, field),
            getattr(patched, field),
            err_msg=f"recompiled model differs in {field}",
        )
    # the intended delta and nothing else on the material side: PLA
    # materials gain the texture and the mean-compensated albedo
    assert patched.ntex == base.ntex + 1
    mean = surface_sim._surface_tex_mean
    pla_rgba = np.asarray(SO101Sim.ARM_PHOTOMETRICS_V1["pla"]["rgba"])
    for index in range(base.nmat):
        name = base.mat(index).name
        if SO101Sim._is_pla_material(name):
            assert patched.mat_texid[index][mujoco.mjtTextureRole.mjTEXROLE_RGB] >= 0
            np.testing.assert_allclose(
                patched.mat_rgba[index, :3] * mean,
                pla_rgba,
                atol=1e-12,
            )
        else:
            np.testing.assert_array_equal(
                patched.mat_rgba[index],
                base.mat_rgba[index],
                err_msg=f"non-PLA material {name} drifted",
            )
            assert (
                patched.mat_texid[index][mujoco.mjtTextureRole.mjTEXROLE_RGB]
                == base.mat_texid[index][mujoco.mjtTextureRole.mjTEXROLE_RGB]
            )


def test_v2_qpos_trajectory_bit_equal(
    graded_sim: SO101Sim,
    surface_sim: SO101Sim,
) -> None:
    # settled reset AND a scripted 60-tick excursion: bit-equal qpos at
    # every tick, shared RNG streams untouched
    for seed in (0, 7):
        graded_sim.reset(seed, appearance_seed=123)
        surface_sim.reset(seed, appearance_seed=123)
        np.testing.assert_array_equal(graded_sim.data.qpos, surface_sim.data.qpos)
        assert (
            graded_sim._noise_rng.bit_generator.state
            == surface_sim._noise_rng.bit_generator.state
        )
    from sim.so101_sim import HOME_DEGREES

    for tick in range(60):
        action = HOME_DEGREES + 8.0 * np.sin(tick / 9.0) * np.ones(6)
        graded_sim.step(action)
        surface_sim.step(action)
        np.testing.assert_array_equal(graded_sim.data.qpos, surface_sim.data.qpos)


def test_v2_texture_image_deterministic_and_bounded() -> None:
    params = SO101Sim.ARM_SURFACE_TEXTURE_V2
    tex = SO101Sim._surface_texture_image(params)
    assert tex.shape == (params["size"], params["size"])
    # tanh soft-bound: strictly inside [0, 1], ZERO clipped texels —
    # clipping would silently break the grade-time mean compensation
    assert float(tex.min()) > 0.0 and float(tex.max()) < 1.0
    clipped = float(((tex <= 0.0) | (tex >= 1.0)).mean())
    assert clipped == 0.0
    assert abs(float(tex.mean()) - params["center"]) < 0.02
    # the generator refuses amplitudes that would clip
    with pytest.raises(ValueError, match="headroom"):
        SO101Sim._surface_texture_image({**params, "amplitude": 0.5})
    np.testing.assert_array_equal(tex, SO101Sim._surface_texture_image(params))
    # anisotropy: layer lines vary along rows, coherent along columns
    row_var = float(tex.mean(axis=1).std())
    col_var = float(tex.mean(axis=0).std())
    assert row_var > 3.0 * col_var


def test_v2_texture_is_pla_local_in_render(
    graded_sim: SO101Sim,
    surface_sim: SO101Sim,
) -> None:
    # raw top source renders differ only on PLA-material pixels plus an
    # antialiasing halo (MSAA blends geom boundaries)
    graded_sim.reset(3, appearance_seed=99)
    surface_sim.reset(3, appearance_seed=99)
    frames = {}
    masks = {}
    for label, sim in (("base", graded_sim), ("tex", surface_sim)):
        renderer = mujoco.Renderer(sim.model, height=480, width=640)
        renderer.update_scene(sim.data, camera="top_cam")
        frames[label] = renderer.render()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(sim.data, camera="top_cam")
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        renderer.close()
        pla = [
            g
            for g in range(sim.model.ngeom)
            if sim.model.geom_matid[g] >= 0
            and SO101Sim._is_pla_material(sim.model.mat(sim.model.geom_matid[g]).name)
        ]
        is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
        masks[label] = is_geom & np.isin(seg[..., 0], pla)
    np.testing.assert_array_equal(masks["base"], masks["tex"])
    halo = masks["base"].astype(np.float64)
    for _ in range(2):
        padded = np.pad(halo, 1, mode="edge")
        halo = sum(
            padded[1 + dy : 481 + dy, 1 + dx : 641 + dx]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        )
    outside = halo == 0.0
    diff = (frames["base"].astype(int) != frames["tex"].astype(int)).any(axis=-1)
    # REFLECTION RIDER (registered 2026-08-14, mechanism confirmed by
    # zeroing mat_reflectance -> 0 out-of-halo diffs): a TRUE surface
    # texture rides every physical light path, including the tabletop's
    # planar reflection of the arm (mat_reflectance 0.02) — exactly as
    # it does in production observations. Out-of-halo diffs are
    # therefore allowed ONLY on reflective-material geoms, magnitude
    # bounded far below the on-arm signal, small pixel count. Any other
    # out-of-halo diff is a locality leak and fails.
    seg_base = None
    renderer = mujoco.Renderer(graded_sim.model, height=480, width=640)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(graded_sim.data, camera="top_cam")
    seg_base = renderer.render()
    renderer.close()
    reflective = [
        g
        for g in range(graded_sim.model.ngeom)
        if graded_sim.model.geom_matid[g] >= 0
        and float(graded_sim.model.mat_reflectance[graded_sim.model.geom_matid[g]]) > 0
    ]
    is_geom = seg_base[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
    on_reflective = is_geom & np.isin(seg_base[..., 0], reflective)
    # a 1-px MSAA fringe around the reflective surface blends with it
    fringe = on_reflective.astype(np.float64)
    padded = np.pad(fringe, 1, mode="edge")
    fringe = sum(
        padded[1 + dy : 481 + dy, 1 + dx : 641 + dx]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    )
    leak = diff & outside & (fringe == 0.0)
    assert not leak.any(), f"{int(leak.sum())} out-of-halo px off reflective geoms"
    rider = diff & outside
    magnitude = np.abs(frames["base"].astype(int) - frames["tex"].astype(int)).max(
        axis=-1,
    )
    assert float(rider.mean()) < 0.005  # < 0.5% of the frame
    if rider.any():
        assert int(magnitude[rider].max()) <= 24  # reflectance-scale, not signal
    assert diff[masks["base"]].any()  # the texture is actually visible
