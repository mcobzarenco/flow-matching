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
