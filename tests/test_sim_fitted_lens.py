"""Oracles for the fitted-lens wrist render path (sim-fit-real-lens-model
leg (b)): the cubemap source + fitted theta(rho) gather behind
``lens_model="fitted"``. Conventions are pinned two independent ways —
a map-level reconstruction that re-derives every stored sample point
from the fitted model, and a render-level equivalence run where the
cubemap machinery, fed the deployed ideal-equidistant parameters, must
reproduce the deployed single-source path. The registered gates ride
along: the top-cam path must stay bit-identical, and the deployed
default must not change at all."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.so101_sim import SO101Sim

# The deployed ideal-equidistant lens expressed in the fitted model's
# parameterization: center at the 640x480 image midpoint used by
# _init_fisheye, no polynomial terms.
IDEAL_PARAMS = {"cx": 319.5, "cy": 239.5, "k2": 0.0, "k4": 0.0}


def test_lens_model_validated() -> None:
    with pytest.raises(ValueError, match="lens_model"):
        SO101Sim(lens_model="barrel")
    with pytest.raises(ValueError, match="fisheye style"):
        SO101Sim(render_style="v0", lens_model="fitted")


def test_default_path_untouched() -> None:
    # The deployed default carries no fitted-lens state: same fovy,
    # same shared equidistant map, no face renderer.
    sim = SO101Sim()
    assert sim.lens_model == "equidistant"
    assert sim._wrist_lens is None
    assert float(sim.model.camera("wrist_cam").fovy[0]) == SO101Sim.V1_SRC_FOVY


def test_fitted_map_matches_lens_model() -> None:
    # Reconstruct each stored sample point back into a ray (face pixel
    # -> face-frame direction -> base frame via the face quat) and
    # check it against the fitted model evaluated directly at the
    # output pixel: polar angle theta(rho) and pixel-space azimuth.
    sim = SO101Sim(lens_model="fitted")
    lens = sim._wrist_lens
    assert lens is not None
    x0, y0, wx, wy = lens.grid
    size = lens.face_size
    sx = x0 + wx[..., 0]
    sy = (y0 % size) + wy[..., 0]
    slot = y0 // size
    center = (size - 1) / 2.0
    f_face = (size / 2.0) / np.tan(np.deg2rad(SO101Sim.LENS_FACE_HALF_DEG))
    height, width = x0.shape
    fit = SO101Sim.WRIST_LENS_FIT
    f_dist = (height / 2.0) / np.tan(np.deg2rad(SO101Sim.V1_CENTER_FOVY) / 2.0)
    rotations = []
    for quat in lens.face_quats:
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, quat)
        rotations.append(mat.reshape(3, 3))
    rng = np.random.default_rng(0)
    v = rng.integers(0, height, 3000)
    u = rng.integers(0, width, 3000)
    # Force the four output corners in: they exercise the side faces.
    v = np.concatenate([v, [0, 0, height - 1, height - 1]])
    u = np.concatenate([u, [0, width - 1, 0, width - 1]])
    d_face = np.stack(
        [
            (sx[v, u] - center) / f_face,
            -(sy[v, u] - center) / f_face,
            -np.ones(len(v)),
        ],
    )
    d_face /= np.linalg.norm(d_face, axis=0)
    d = np.stack(
        [rotations[s] @ d_face[:, i] for i, s in enumerate(slot[v, u])],
        axis=1,
    )
    theta = np.arccos(np.clip(-d[2], -1.0, 1.0))
    x = u - fit["cx"]
    y = v - fit["cy"]
    rho = np.hypot(x, y) / f_dist
    expected = rho * (1 + fit["k2"] * rho**2 + fit["k4"] * rho**4)
    # Bilinear-grid quantization bounds the reconstruction at the
    # sample points themselves only through float roundoff.
    np.testing.assert_allclose(theta, expected, atol=1e-9)
    azimuth = np.arctan2(d[0], -d[1])  # image +x, +y(down) components
    expected_az = np.arctan2(x, y)
    mismatch = np.abs(np.angle(np.exp(1j * (azimuth - expected_az))))
    np.testing.assert_allclose(mismatch, 0.0, atol=1e-9)


@pytest.mark.gpu
def test_cubemap_machinery_reproduces_equidistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With the fit parameters overridden to the deployed ideal lens,
    # the cubemap path must reproduce the deployed single-source remap
    # up to bilinear resampling from the different source grids (this
    # catches any quat/axis/v-flip convention error, which would warp
    # the frame wholesale). Same seed + same draw order -> identical
    # noise, so the residual is resampling only. Measured seed-0/3
    # values: mean 0.054/0.058, p99 1, max 20 (isolated wood-grain
    # edge pixels); thresholds carry >10x headroom on the mean.
    monkeypatch.setattr(SO101Sim, "WRIST_LENS_FIT", IDEAL_PARAMS)
    ref = SO101Sim(render_style="v3", post_backend="numpy")
    cube = SO101Sim(render_style="v3", post_backend="numpy", lens_model="fitted")
    for seed in (0, 3):
        obs_ref = ref.reset(seed)
        obs_cube = cube.reset(seed)
        assert np.array_equal(obs_ref.top, obs_cube.top)
        diff = np.abs(
            obs_ref.wrist.astype(np.int16) - obs_cube.wrist.astype(np.int16),
        )
        assert diff.mean() < 1.0, f"seed {seed}: mean {diff.mean():.2f}"
        assert np.percentile(diff, 99) <= 6, (
            f"seed {seed}: p99 {np.percentile(diff, 99)}"
        )


@pytest.mark.gpu
def test_rotated_cubemap_layout_agrees(monkeypatch: pytest.MonkeyPatch) -> None:
    # The ideal-params equivalence above only exercises the FRONT face
    # (an ideal centered lens keeps every ray within 45 deg), so it
    # cannot catch a side-face convention error. Render the same
    # fitted lens through a second cubemap layout with every face
    # pre-rotated 20 deg about the camera y axis: large regions of the
    # frame switch faces between the layouts, and both must resample
    # the same underlying scene — any error in the quat composition or
    # the face-frame projection breaks the agreement wholesale (before
    # the base-axis headlight re-point in _render_wrist_source this
    # read mean 6.77 — the seam the oracle exists to catch). Measured:
    # mean 0.266, p99 4, max 34 (resampling from 20-deg-rotated grids).
    standard = SO101Sim(render_style="v3", post_backend="numpy", lens_model="fitted")
    tilt = np.array([np.cos(np.deg2rad(10)), 0.0, np.sin(np.deg2rad(10)), 0.0])
    rotated_faces = {}
    for name, quat in SO101Sim.LENS_FACES.items():
        composed = np.empty(4)
        mujoco.mju_mulQuat(composed, tilt, np.array(quat))
        rotated_faces[name] = tuple(composed)
    monkeypatch.setattr(SO101Sim, "LENS_FACES", rotated_faces)
    rotated = SO101Sim(render_style="v3", post_backend="numpy", lens_model="fitted")
    assert standard._wrist_lens is not None
    assert rotated._wrist_lens is not None
    assert len(rotated._wrist_lens.face_quats) >= 2
    obs_std = standard.reset(2)
    obs_rot = rotated.reset(2)
    assert np.array_equal(obs_std.top, obs_rot.top)
    diff = np.abs(obs_std.wrist.astype(np.int16) - obs_rot.wrist.astype(np.int16))
    assert diff.mean() < 1.0, f"mean {diff.mean():.2f}"
    assert np.percentile(diff, 99) <= 6, f"p99 {np.percentile(diff, 99)}"


@pytest.mark.gpu
def test_top_path_bit_identical_under_fitted_lens() -> None:
    # Registered gate: the fitted lens is wrist-only — the top plate
    # already carries the real top lens. Top frames must not move by
    # a single count, across styles with and without the composite.
    for style in ("v1", "v3"):
        ref = SO101Sim(render_style=style, post_backend="numpy")
        fit = SO101Sim(render_style=style, post_backend="numpy", lens_model="fitted")
        obs_ref = ref.reset(1)
        obs_fit = fit.reset(1)
        assert np.array_equal(obs_ref.top, obs_fit.top), style
        assert not np.array_equal(obs_ref.wrist, obs_fit.wrist), style


@pytest.mark.gpu
def test_fitted_reset_deterministic_and_camera_restored() -> None:
    # Two constructions, same seed -> bit-identical fitted frames; and
    # the per-face camera re-aiming restores the base wrist pose.
    a = SO101Sim(render_style="v3", post_backend="numpy", lens_model="fitted")
    base = a.model.camera("wrist_cam").quat.copy()
    obs_a = a.reset(5)
    assert np.array_equal(a.model.camera("wrist_cam").quat, base)
    b = SO101Sim(render_style="v3", post_backend="numpy", lens_model="fitted")
    obs_b = b.reset(5)
    assert np.array_equal(obs_a.wrist, obs_b.wrist)
    assert np.array_equal(obs_a.top, obs_b.top)


@pytest.mark.gpu
def test_fitted_torch_post_matches_numpy() -> None:
    # Same contract as the deployed-path oracle in
    # test_sim_appearance.py: the CUDA gather must reproduce the numpy
    # float64 reference within float32 rounding.
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    reference = SO101Sim(
        render_style="v3",
        post_backend="numpy",
        lens_model="fitted",
    )
    fast = SO101Sim(render_style="v3", post_backend="torch", lens_model="fitted")
    obs_ref = reference.reset(7)
    obs_fast = fast.reset(7)
    for name in ("top", "wrist"):
        a = getattr(obs_ref, name).astype(np.int16)
        b = getattr(obs_fast, name).astype(np.int16)
        diff = np.abs(a - b)
        assert diff.max() <= 2, f"{name}: max diff {diff.max()}"
        assert (diff > 0).mean() < 0.05, f"{name}: widespread drift"
