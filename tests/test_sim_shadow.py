"""Oracles for the v4 contact-shadow pass (sim.shadow +
SO101Sim render_style="v4", sim-composite-contact-shadows leg (a)).

Geometry oracles run against a synthetic overhead camera where every
projection is analytic: camera at (0, 0, 1) looking straight down
(cam_mat = identity: x_cam = x_world, y_cam = y_world, camera looks
along -z_world), f = 200, so a surface at height h seen at pixel u
casts, under vertical light, a shadow at cx + (u - cx) * (1 - h)
(the foot is at the same world x/y but reprojects with depth 1), and
tilting the light by zenith angle t along +x moves the foot by
h * tan(t) in world = f * h * tan(t) pixels.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.shadow import gaussian_blur_2d, shadow_map
from sim.so101_sim import SO101Sim

SIZE = 128
F = 200.0
CX = CY = (SIZE - 1) / 2.0
INTRINSICS = (F, CX, CY)
CAM_POS = np.array([0.0, 0.0, 1.0])
CAM_MAT = np.eye(3)


def overhead_scene(
    height: float,
    blob: tuple[int, int, int] = (90, 100, 8),
) -> tuple[np.ndarray, np.ndarray]:
    """Depth + dynamic mask: a square object surface at world height
    ``height`` covering pixels blob = (v0, u0, half_width)."""
    depth = np.full((SIZE, SIZE), 1.0)
    dynamic = np.zeros((SIZE, SIZE))
    v0, u0, half = blob
    depth[v0 - half : v0 + half, u0 - half : u0 + half] = 1.0 - height
    dynamic[v0 - half : v0 + half, u0 - half : u0 + half] = 1.0
    return depth, dynamic


def centroid(mapped: np.ndarray) -> tuple[float, float]:
    v, u = np.nonzero(mapped > 1e-6)
    w = mapped[v, u]
    return float((v * w).sum() / w.sum()), float((u * w).sum() / w.sum())


def test_vertical_light_shadow_shrinks_toward_center() -> None:
    height = 0.3
    depth, dynamic = overhead_scene(height)
    mapped = shadow_map(
        depth,
        dynamic,
        INTRINSICS,
        CAM_POS,
        CAM_MAT,
        np.array([0.0, 0.0, -1.0]),
        sigma_px=1.0,
    )
    assert mapped.max() > 0.1
    cv, cu = centroid(mapped)
    # Foot at same world x/y, reprojected from depth 1 instead of 0.7.
    assert abs(cu - (CX + (100 - CX) * (1.0 - height))) < 1.5
    assert abs(cv - (CY + (90 - CY) * (1.0 - height))) < 1.5


def test_oblique_light_offsets_shadow_by_h_tan_zenith() -> None:
    height = 0.3
    depth, dynamic = overhead_scene(height)
    zenith = np.deg2rad(30.0)
    vertical = shadow_map(
        depth,
        dynamic,
        INTRINSICS,
        CAM_POS,
        CAM_MAT,
        np.array([0.0, 0.0, -1.0]),
        sigma_px=1.0,
    )
    oblique = shadow_map(
        depth,
        dynamic,
        INTRINSICS,
        CAM_POS,
        CAM_MAT,
        np.array([np.sin(zenith), 0.0, -np.cos(zenith)]),
        sigma_px=1.0,
    )
    _, cu_vertical = centroid(vertical)
    cv_oblique, cu_oblique = centroid(oblique)
    expected = F * height * np.tan(zenith)
    assert abs((cu_oblique - cu_vertical) - expected) < 1.5
    assert abs(cv_oblique - centroid(vertical)[0]) < 1.5  # +x light: no v shift


def test_flush_points_cast_nothing() -> None:
    depth, dynamic = overhead_scene(0.001)  # below min_height
    mapped = shadow_map(
        depth,
        dynamic,
        INTRINSICS,
        CAM_POS,
        CAM_MAT,
        np.array([0.0, 0.0, -1.0]),
        sigma_px=1.0,
    )
    assert mapped.max() == 0.0


def test_bounds_clip_feet() -> None:
    depth, dynamic = overhead_scene(0.3)
    mapped = shadow_map(
        depth,
        dynamic,
        INTRINSICS,
        CAM_POS,
        CAM_MAT,
        np.array([0.0, 0.0, -1.0]),
        sigma_px=1.0,
        bounds_xy=(5.0, 6.0, 5.0, 6.0),  # nowhere near the feet
    )
    assert mapped.max() == 0.0


def test_upward_light_rejected() -> None:
    depth, dynamic = overhead_scene(0.3)
    with pytest.raises(ValueError, match="downward"):
        shadow_map(
            depth,
            dynamic,
            INTRINSICS,
            CAM_POS,
            CAM_MAT,
            np.array([0.0, 0.0, 1.0]),
            sigma_px=1.0,
        )


def test_blur_preserves_interior_mass() -> None:
    image = np.zeros((64, 64))
    image[30:34, 30:34] = 1.0
    blurred = gaussian_blur_2d(image, 3.0)
    assert abs(blurred.sum() - image.sum()) < 1e-6
    assert blurred.max() < 1.0


@pytest.mark.gpu
def test_v4_wrist_bit_identical_to_v3_top_differs() -> None:
    v3 = SO101Sim(render_style="v3", post_backend="numpy")
    v4 = SO101Sim(render_style="v4", post_backend="numpy")
    for seed in (0, 7):
        obs3, obs4 = v3.reset(seed), v4.reset(seed)
        np.testing.assert_array_equal(obs3.wrist, obs4.wrist)
        assert not np.array_equal(obs3.top, obs4.top)
        # Shadow only ever darkens the plate: no pixel gets brighter
        # by more than the sensor-noise re-rounding wiggle.
        diff = obs4.top.astype(np.int16) - obs3.top.astype(np.int16)
        assert diff.max() <= 1


@pytest.mark.gpu
def test_v4_zero_strength_is_v3() -> None:
    v3 = SO101Sim(render_style="v3", post_backend="numpy")
    v4 = SO101Sim(render_style="v4", post_backend="numpy")
    v4.V4_SHADOW_STRENGTH = 0.0  # instance override
    obs3, obs4 = v3.reset(0), v4.reset(0)
    np.testing.assert_array_equal(obs3.top, obs4.top)


@pytest.mark.gpu
def test_v4_torch_matches_numpy_reference() -> None:
    reference = SO101Sim(render_style="v4", post_backend="numpy")
    fast = SO101Sim(render_style="v4", post_backend="torch")
    for seed in (0, 7):
        obs_ref, obs_fast = reference.reset(seed), fast.reset(seed)
        for name in ("top", "wrist"):
            a = getattr(obs_ref, name).astype(np.int16)
            b = getattr(obs_fast, name).astype(np.int16)
            diff = np.abs(a - b)
            assert diff.max() <= 2, f"{name} seed {seed}: max diff {diff.max()}"
            assert (diff > 0).mean() < 0.05, f"{name} seed {seed}: widespread drift"
