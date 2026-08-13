"""Contact-shadow projection for the composite render styles.

The v2/v3 composites paste the rendered arm over a real clean plate;
the one physics law every real frame obeys that no composite frame
does is the arm darkening the table under it (lit 0823,
fontaine/blog/src/papers/composite-shadows.md). This module holds the
pure geometry shared by the v4 render style (sim.so101_sim) and the
light-direction fit instrument (fontaine/scripts/fit_contact_shadow.py):
unproject the camera-visible dynamic pixels through the depth buffer,
slide them along the dominant light direction onto the table plane,
and splat the feet back through the same camera as a soft occupancy
map in source-pinhole space (the caller remaps it through the fisheye
grid exactly like the dynamic-content mask).

Camera conventions (verified against MuJoCo renders before this module
was written: table-plane pixels reconstruct to z = 0.0003 +/- 0.0001 m,
the benchy centroid to its xpos): the depth buffer is planar (distance
along -z_cam), and pixel (u, v) maps to the camera-frame direction
((u - cx) / f, -(v - cy) / f, -1).

Known second-order omissions, accepted for the one-parameter recipe:
the map darkens only the plate contribution (rendered foreground never
receives a cast shadow), and surfaces hidden from the camera but lit
cast nothing (visible-surface silhouette; with a near-overhead light
and the top camera the coverage loss is small and the soft edge hides
the seams).
"""

from __future__ import annotations

import numpy as np


def gaussian_blur_2d(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable edge-padded Gaussian on a single-channel [H, W] float
    image (the SO101Sim._blur kernel, parameterized sigma)."""
    if sigma <= 0:
        return image
    radius = max(1, int(np.ceil(2.5 * sigma)))
    taps = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (taps / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(image, ((radius, radius), (0, 0)), mode="edge")
    rows = np.zeros_like(image)
    for i, k in enumerate(kernel):
        rows += padded[i : i + image.shape[0]] * k
    padded = np.pad(rows, ((0, 0), (radius, radius)), mode="edge")
    out = np.zeros_like(image)
    for i, k in enumerate(kernel):
        out += padded[:, i : i + image.shape[1]] * k
    return out


def unproject(
    depth: np.ndarray,
    pixels: tuple[np.ndarray, ...],
    intrinsics: tuple[float, float, float],
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
) -> np.ndarray:
    """World-frame [N, 3] points for (v, u) pixel arrays of a planar
    depth buffer. intrinsics = (f, cx, cy) of the source pinhole."""
    f, cx, cy = intrinsics
    v, u = pixels
    d = depth[v, u]
    cam = np.stack([(u - cx) / f * d, -(v - cy) / f * d, -d])
    return cam_pos[None, :] + (cam_mat @ cam).T


def shadow_map(
    depth: np.ndarray,
    dynamic_mask: np.ndarray,
    intrinsics: tuple[float, float, float],
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    light_dir: np.ndarray,
    sigma_px: float,
    *,
    plane_z: float = 0.0,
    bounds_xy: tuple[float, float, float, float] | None = None,
    min_height: float = 0.003,
    max_points: int | None = None,
) -> np.ndarray:
    """[H, W] float soft shadow-occupancy map in source pinhole space.

    Dynamic pixels (dynamic_mask > 0.5) are unprojected, slid along
    ``light_dir`` (unit-normalized here; z component must be negative:
    the direction light TRAVELS) onto ``plane_z``, and the feet are
    reprojected and bilinearly splatted, then softened with a Gaussian
    of ``sigma_px``. Points within ``min_height`` of the plane cast
    nothing (they are already resting on it, and depth quantization at
    mask edges would splat noise). ``bounds_xy`` = (xmin, xmax, ymin,
    ymax) clips feet to the table extent so nothing darkens plate
    pixels that show the floor beyond the table edge.
    """
    light = np.asarray(light_dir, dtype=np.float64)
    light = light / np.linalg.norm(light)
    if light[2] >= 0:
        raise ValueError(f"light_dir must point downward, got z={light[2]:.3f}")
    height, width = depth.shape
    out = np.zeros((height, width), dtype=np.float64)
    pixels = np.nonzero(dynamic_mask > 0.5)
    if len(pixels[0]) == 0:
        return out
    if max_points is not None and len(pixels[0]) > max_points:
        stride = int(np.ceil(len(pixels[0]) / max_points))
        pixels = (pixels[0][::stride], pixels[1][::stride])
    points = unproject(depth, pixels, intrinsics, cam_pos, cam_mat)
    points = points[points[:, 2] > plane_z + min_height]
    if len(points) == 0:
        return out
    t = (plane_z - points[:, 2]) / light[2]
    feet = points + t[:, None] * light[None, :]
    if bounds_xy is not None:
        xmin, xmax, ymin, ymax = bounds_xy
        keep = (
            (feet[:, 0] >= xmin)
            & (feet[:, 0] <= xmax)
            & (feet[:, 1] >= ymin)
            & (feet[:, 1] <= ymax)
        )
        feet = feet[keep]
        if len(feet) == 0:
            return out
    f, cx, cy = intrinsics
    local = (feet - cam_pos[None, :]) @ cam_mat  # == cam_mat.T @ (p - c)
    in_front = local[:, 2] < -1e-6
    local = local[in_front]
    if len(local) == 0:
        return out
    u = cx + f * local[:, 0] / -local[:, 2]
    v = cy - f * local[:, 1] / -local[:, 2]
    inside = (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    u, v = u[inside], v[inside]
    if len(u) == 0:
        return out
    u0 = np.clip(np.floor(u).astype(np.int64), 0, width - 2)
    v0 = np.clip(np.floor(v).astype(np.int64), 0, height - 2)
    wu, wv = u - u0, v - v0
    np.add.at(out, (v0, u0), (1 - wu) * (1 - wv))
    np.add.at(out, (v0, u0 + 1), wu * (1 - wv))
    np.add.at(out, (v0 + 1, u0), (1 - wu) * wv)
    np.add.at(out, (v0 + 1, u0 + 1), wu * wv)
    np.minimum(out, 1.0, out=out)
    return gaussian_blur_2d(out, sigma_px)
