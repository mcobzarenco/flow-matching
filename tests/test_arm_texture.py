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
