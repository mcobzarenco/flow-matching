"""Paste the mined real clutter crops into a bank plate at the drawn
poses (fg appearance pass leg (b), pre-reg in-channel 05:23Z
2026-08-13).

Per drawn-present object the paste is an inverse warp through the
same analytic fisheye camera model the bank mining pass verified
against a segmentation render (make_clean_plates selfcheck <= 2 cm):
each target pixel is unprojected to the object's height plane, moved
by the rigid drawn->mined transform in world space, reprojected, and
bilinearly sampled from the RGBA crop — translation, yaw jitter and
the fisheye's local scale all ride the camera model. Transform
conventions per draw mode:

  absolute               target anchor = the drawn xy (the draw ranges
                         were measured in the same blob-centroid world
                         convention as the crop anchor)
  delta_about_canonical  target anchor = mined anchor + (drawn xy -
                         canonical xy) — the laptop's absolute centroid
                         is crop-biased, only the delta is trusted
  fixed_canonical        identity (the object never moves; the crop
                         stays at its real measured location)

The crop RGB is stored normalized to global-plate lighting; the paste
applies the active episode's affine (gain, bias) so patch and plate
share the reset's lighting state, exactly like the rendered
foreground does. The paste consumes no RNG.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from make_clean_plates import CAM_POS, CAM_R, CENTER, F_DIST

ABSENT_POS = (2.5, 0.0, -1.0)  # SO101Sim.V3_ABSENT_POS
BBOX_PAD = 4


def project_v(world: np.ndarray) -> np.ndarray:
    """[N, 3] world points -> [N, 2] distorted output pixels."""
    v_cam = (np.asarray(world, dtype=np.float64) - CAM_POS) @ CAM_R
    norm = np.linalg.norm(v_cam, axis=-1)
    theta = np.arccos(np.clip(-v_cam[..., 2] / norm, -1.0, 1.0))
    phi = np.arctan2(v_cam[..., 1], v_cam[..., 0])
    r = F_DIST * theta
    return CENTER + np.stack([r * np.cos(phi), -r * np.sin(phi)], axis=-1)


def unproject_v(pixels: np.ndarray, height_m: float) -> np.ndarray:
    """[N, 2] distorted output pixels -> [N, 2] world xy on z=height_m."""
    d = np.asarray(pixels, dtype=np.float64) - CENTER
    r = np.hypot(d[..., 0], d[..., 1])
    theta = r / F_DIST
    phi = np.arctan2(-d[..., 1], d[..., 0])
    sin_t = np.sin(theta)
    d_cam = np.stack(
        [sin_t * np.cos(phi), sin_t * np.sin(phi), -np.cos(theta)],
        axis=-1,
    )
    d_world = d_cam @ CAM_R.T
    t = (height_m - CAM_POS[2]) / d_world[..., 2]
    return (CAM_POS + t[..., None] * d_world)[..., :2]


def rot2(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


class ClutterCrops:
    """The mined RGBA crops + their paste metadata."""

    def __init__(self, crops_dir: Path) -> None:
        from PIL import Image

        self.manifest = json.loads((crops_dir / "crops_manifest.json").read_text())
        self.crops: dict[str, dict] = {}
        for name, meta in self.manifest["objects"].items():
            rgba = np.asarray(Image.open(crops_dir / f"{name}.png"), dtype=np.float64)
            self.crops[name] = {**meta, "rgba": rgba}

    def paste(
        self,
        plate: np.ndarray,
        drawn: dict[str, tuple[np.ndarray, float]],
        base: dict[str, tuple[np.ndarray, np.ndarray, float]],
        gain: np.ndarray,
        bias: np.ndarray,
    ) -> np.ndarray:
        """New [H, W, 3] float plate with every drawn-present object's
        crop warped in. ``drawn``/``base`` are SO101Sim._clutter_drawn /
        _clutter_base; ``gain``/``bias`` the active episode affine."""
        out = plate.astype(np.float64).copy()
        height, width = out.shape[:2]
        for name, meta in self.crops.items():
            pos, yaw = drawn[name]
            if tuple(np.round(pos, 6)) == ABSENT_POS:
                continue
            base_pos, _, base_yaw = base[name]
            anchor_mined = np.array(meta["anchor_world_xy"], dtype=np.float64)
            delta_yaw = float(yaw - base_yaw)
            if meta["mode"] == "absolute":
                anchor_target = np.asarray(pos, dtype=np.float64)[:2]
            elif meta["mode"] == "delta_about_canonical":
                anchor_target = anchor_mined + (
                    np.asarray(pos, dtype=np.float64)[:2] - base_pos[:2]
                )
            else:  # fixed_canonical: the object never moves
                anchor_target = anchor_mined
                delta_yaw = 0.0
            h_m = float(meta["height_m"])
            x0, y0, w, h = meta["bbox_xywh"]
            corners_px = np.array(
                [[x0, y0], [x0 + w, y0], [x0, y0 + h], [x0 + w, y0 + h]],
                dtype=np.float64,
            )
            corners_world = unproject_v(corners_px, h_m)
            fwd = (corners_world - anchor_mined) @ rot2(delta_yaw).T + anchor_target
            target_px = project_v(
                np.concatenate([fwd, np.full((4, 1), h_m)], axis=-1),
            )
            tx0 = max(0, int(np.floor(target_px[:, 0].min())) - BBOX_PAD)
            tx1 = min(width, int(np.ceil(target_px[:, 0].max())) + 1 + BBOX_PAD)
            ty0 = max(0, int(np.floor(target_px[:, 1].min())) - BBOX_PAD)
            ty1 = min(height, int(np.ceil(target_px[:, 1].max())) + 1 + BBOX_PAD)
            if tx1 <= tx0 or ty1 <= ty0:
                continue
            uu, vv = np.meshgrid(np.arange(tx0, tx1), np.arange(ty0, ty1))
            grid = np.stack([uu, vv], axis=-1).reshape(-1, 2).astype(np.float64)
            world = unproject_v(grid, h_m)
            back = (world - anchor_target) @ rot2(-delta_yaw).T + anchor_mined
            src = project_v(
                np.concatenate([back, np.full((len(back), 1), h_m)], axis=-1),
            )
            su = src[:, 0] - x0
            sv = src[:, 1] - y0
            rgba = meta["rgba"]
            inside = (su >= 0) & (su <= w - 1.001) & (sv >= 0) & (sv <= h - 1.001)
            iu = np.clip(su, 0, w - 1.001)
            iv = np.clip(sv, 0, h - 1.001)
            u0 = np.floor(iu).astype(int)
            v0 = np.floor(iv).astype(int)
            fu = (iu - u0)[:, None]
            fv = (iv - v0)[:, None]
            sample = (
                rgba[v0, u0] * (1 - fu) * (1 - fv)
                + rgba[v0, u0 + 1] * fu * (1 - fv)
                + rgba[v0 + 1, u0] * (1 - fu) * fv
                + rgba[v0 + 1, u0 + 1] * fu * fv
            )
            alpha = (sample[:, 3:4] / 255.0) * inside[:, None]
            graded = np.clip(sample[:, :3] * gain + bias, 0.0, 255.0)
            block = out[ty0:ty1, tx0:tx1].reshape(-1, 3)
            out[ty0:ty1, tx0:tx1] = (block * (1 - alpha) + graded * alpha).reshape(
                ty1 - ty0,
                tx1 - tx0,
                3,
            )
        return out
