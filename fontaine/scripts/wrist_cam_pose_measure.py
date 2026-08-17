"""Wrist-cam matched-pair measurements (queue: wrist-cam-pose-refit,
stage 2 of the lens-plumbline pattern; stage 1 = wrist_cam_matched_pairs.py).

Measures the SAME gripper-geometry quantities on both sides of every
matched pair (identical kinematics, so any discrepancy is camera
pose/optics):

  moving jaw   real: hue segmentation — the rig's moving jaw is the only
               salmon-orange object in the wrist view; blue-deficit
               (r-b) separates the chroma-bearing core from every wood
               tone by >3x (disk r-b p99 12, table p99 9, jaw p25 49;
               probed on ep000), and the blown-highlight jaw body
               (251/251/251 — zero chroma at this exposure, probed on
               ep012) is recovered by bounded dilation growth from that
               seed into bright pixels, so the centroid is not biased
               into the shadow side.
               sim: exact segmentation mask of the moving_jaw body's
               visible geoms, pushed through the production lens remap
               (wrist_arm_mask pattern).
  fixed jaw    real: the serrated jaw is near-black indoors (ep000
               v<=45) but a BLUE-tinted gray under daylight (ep012
               b-r p25 54 at v~130 — every wood tone is warm-neutral,
               seams b-r ~ -3); union of the two, filtered to chunky
               components (table seams are line-thin, bbox fill < 0.2)
               AND proximity-gated to the rig's jaw geometry: the fixed
               jaw opposes the moving jaw image-RIGHT at similar height,
               so only components whose centroid falls in that window
               count — blue-gray mount prints and room background are
               the same color family and fire the raw threshold in
               lifted-arm frames (QC'd on ep012/ep020 mid-episode).
               Undefined (None) when the moving jaw is not visible.
               sim: exact mask of the gripper body's visible geoms (the
               fixed jaw and housing are one body), no gate — the mask
               IS the jaw.

Per side, per frame:
  visible            component area >= MIN_AREA (px, at 640x480)
  area_frac          mask area / frame area
  centroid x, y      normalized [0, 1] image coordinates
  axis_deg           moving jaw only: PCA major-axis angle, degrees
                     from image-vertical in [-90, 90) (0 = tip points
                     straight up the image)
  bottom_occ         mask pixels in the bottom BAND_FRAC rows / band px
  both_jaws_visible  moving & fixed visible

Components are kept only if they touch the bottom BAND_TOUCH rows of
the frame — the jaws enter from the bottom edge by construction; this
kills residual wood-knot / shadow bleed on the real side.

Outputs (out-dir, default outputs/sim/wrist_refit):
  measurements.json   per-pair rows + per-metric matched-set summary
  qc/qc_*.png         (--qc) real overlay | sim overlay strips for
                      eyeball validation of the real-side detectors

Usage:
  uv run python fontaine/scripts/wrist_cam_pose_measure.py --qc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.so101_sim import SO101Sim

PAIRS_DIR = Path("outputs/sim/wrist_refit/matched_pairs")

# Real-side detector thresholds (probed on ep000/ep012, see docstring).
ORANGE_R_MINUS_B = 30
ORANGE_R_MINUS_G = 10
ORANGE_V_MIN = 110
GROW_V_MIN = 235  # blown-highlight recovery: bright...
GROW_ROUNDS = 18  # ...and within this many 3x3 dilations of the seed
DARK_V_MAX = 45
BLUE_B_MINUS_R = 15  # daylight fixed jaw: blue-tinted gray
BLUE_V_MAX = 190
DARK_FILL_MIN = 0.2  # bbox fill ratio; table seams are line-thin
FIXED_DX = (0.02, 0.40)  # fixed-jaw window, right of the moving centroid
FIXED_DY = 0.25  # ...and within this height band of it

MIN_AREA = 400  # px at 640x480; below this a part is "not visible"
BAND_FRAC = 0.25  # bottom-band occupancy band
BAND_TOUCH = 0.98  # component must reach into the bottom 2% rows


def _components(
    mask: np.ndarray,
    *,
    min_fill: float = 0.0,
    bottom_touch: bool = True,
    near: tuple[float, float] | None = None,
) -> np.ndarray:
    """Union of connected components that are large enough, (optionally)
    touch the bottom edge of the frame, fill their bbox, and sit inside
    the fixed-jaw window right of ``near`` (normalized moving-jaw
    centroid)."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    height, width = mask.shape
    keep = np.zeros(n, dtype=bool)
    for index in range(1, n):
        _x, y, w, h, area = stats[index]
        if area < MIN_AREA:
            continue
        if bottom_touch and y + h < BAND_TOUCH * height:
            continue
        if min_fill and area / float(w * h) < min_fill:
            continue
        if near is not None:
            dx = centroids[index][0] / width - near[0]
            dy = centroids[index][1] / height - near[1]
            if not (FIXED_DX[0] <= dx <= FIXED_DX[1] and abs(dy) <= FIXED_DY):
                continue
        keep[index] = True
    return keep[labels]


def real_moving_jaw(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.int16)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    v = f.max(axis=-1)
    seed = (
        (r - b >= ORANGE_R_MINUS_B) & (r - g >= ORANGE_R_MINUS_G) & (v >= ORANGE_V_MIN)
    )
    # Blown-highlight recovery: the jaw body saturates to chroma-free
    # white at this exposure; grow the confident seed into bright pixels
    # a bounded number of dilation rounds, so a bright disk or table
    # patch that merely ABUTS the jaw cannot swallow the mask.
    grow = seed | (v >= GROW_V_MIN)
    mask = seed.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(GROW_ROUNDS):
        mask = cv2.dilate(mask, kernel) & grow
    return _components(mask.astype(bool))


def real_fixed_jaw(
    rgb: np.ndarray,
    moving_centroid: tuple[float, float] | None,
) -> np.ndarray | None:
    """None (undefined) when the moving jaw is not visible — the window
    that separates the jaw from same-colored prints needs its anchor."""
    if moving_centroid is None:
        return None
    f = rgb.astype(np.int16)
    r, b = f[..., 0], f[..., 2]
    v = f.max(axis=-1)
    raw = (v <= DARK_V_MAX) | ((b - r >= BLUE_B_MINUS_R) & (v <= BLUE_V_MAX))
    return _components(
        raw,
        min_fill=DARK_FILL_MIN,
        bottom_touch=False,
        near=moving_centroid,
    )


class SimMasker:
    """Per-part sim masks in the FINAL wrist frame: segmentation pass of
    the wrist source render, remapped through the production lens
    (wrist_arm_mask pattern, restricted to one body's visible geoms)."""

    def __init__(self, sim: SO101Sim) -> None:
        self.sim = sim
        self.parts = {
            part: self._visible_geoms(body)
            for part, body in (
                ("moving", "moving_jaw_so101_v1"),
                ("fixed", "gripper"),
                ("mount", "camera_mount"),
            )
        }

    def _visible_geoms(self, body: str) -> np.ndarray:
        model = self.sim.model
        body_id = model.body(body).id
        return np.array(
            [
                index
                for index in range(model.ngeom)
                if model.geom_bodyid[index] == body_id and model.geom_group[index] <= 2
            ],
        )

    def masks_at(self, state_deg: np.ndarray) -> dict[str, np.ndarray]:
        """Boolean per-part masks at the given joint state (kinematic
        qpos set + mj_forward, exactly the stage-1 render path)."""
        sim = self.sim
        sim.data.qpos[sim._joint_qpos] = np.deg2rad(state_deg)
        mujoco.mj_forward(sim.model, sim.data)
        if sim.render_style in ("v3", "v4"):
            sim._set_clutter(drawn=False)
        renderer = sim.renderer
        renderer.enable_segmentation_rendering()
        renderer.update_scene(sim.data, camera="wrist_cam")
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        if sim.render_style in ("v3", "v4"):
            sim._set_clutter(drawn=True)
        is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
        out = {}
        for part, geoms in self.parts.items():
            mask = (is_geom & np.isin(seg[..., 0], geoms)).astype(np.float64)
            if sim.render_style != "v0":
                mask = np.clip(sim._remap(mask[..., None])[..., 0], 0.0, 1.0)
            out[part] = mask >= 0.5
        return out


def metrics(mask: np.ndarray, *, axis: bool = False) -> dict:
    """The per-part scalar reads; None-valued when not visible."""
    height, width = mask.shape
    area = int(mask.sum())
    row = {
        "visible": bool(area >= MIN_AREA),
        "area_frac": round(area / mask.size, 5),
    }
    if not row["visible"]:
        row.update({"cx": None, "cy": None, "bottom_occ": None})
        if axis:
            row["axis_deg"] = None
        return row
    ys, xs = np.nonzero(mask)
    row["cx"] = round(float(xs.mean()) / width, 4)
    row["cy"] = round(float(ys.mean()) / height, 4)
    band = mask[round((1 - BAND_FRAC) * height) :]
    row["bottom_occ"] = round(float(band.sum()) / band.size, 4)
    if axis:
        pts = np.stack([xs - xs.mean(), ys - ys.mean()])
        cov = pts @ pts.T / len(xs)
        values, vectors = np.linalg.eigh(cov)
        vx, vy = vectors[:, np.argmax(values)]
        # Angle from image-vertical, sign = clockwise lean, in [-90, 90).
        angle = np.degrees(np.arctan2(vx, -vy))
        if angle >= 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0
        row["axis_deg"] = round(float(angle), 2)
    return row


def measure_pair(
    real_rgb: np.ndarray,
    sim_masks: dict[str, np.ndarray],
) -> tuple[dict, np.ndarray, np.ndarray | None]:
    real_moving = real_moving_jaw(real_rgb)
    moving_metrics = metrics(real_moving, axis=True)
    anchor = (
        (moving_metrics["cx"], moving_metrics["cy"])
        if moving_metrics["visible"]
        else None
    )
    real_fixed = real_fixed_jaw(real_rgb, anchor)
    row = {
        "real": {
            "moving": moving_metrics,
            "fixed": metrics(real_fixed) if real_fixed is not None else None,
        },
        "sim": {
            "moving": metrics(sim_masks["moving"], axis=True),
            "fixed": metrics(sim_masks["fixed"]),
            "mount": metrics(sim_masks["mount"]),
        },
    }
    row["real"]["both_jaws_visible"] = (
        None if row["real"]["fixed"] is None else bool(row["real"]["fixed"]["visible"])
    )
    row["sim"]["both_jaws_visible"] = bool(
        row["sim"]["moving"]["visible"] and row["sim"]["fixed"]["visible"],
    )
    return row, real_moving, real_fixed


def summarize(rows: list[dict]) -> dict:
    """Matched-set summary: per-metric mean |real - sim| over pairs where
    both sides are visible, plus the visibility-rate gap."""

    def rate(side: str, key: str) -> float | None:
        values = [
            r["measure"][side][key]["visible"]
            for r in rows
            if r["measure"][side][key] is not None
        ]
        return round(float(np.mean(values)), 4) if values else None

    def pair_rate(side: str) -> dict:
        values = [
            r["measure"][side]["both_jaws_visible"]
            for r in rows
            if r["measure"][side]["both_jaws_visible"] is not None
        ]
        return {"defined": len(values), "rate": round(float(np.mean(values)), 4)}

    summary = {
        "pairs": len(rows),
        "visible_rate": {
            side: {part: rate(side, part) for part in ("moving", "fixed")}
            for side in ("real", "sim")
        },
        "both_jaws_visible_rate": {side: pair_rate(side) for side in ("real", "sim")},
        "sim_mount_visible_rate": round(
            float(np.mean([r["measure"]["sim"]["mount"]["visible"] for r in rows])),
            4,
        ),
    }
    for metric in ("cx", "cy", "axis_deg", "bottom_occ", "area_frac"):
        deltas = [
            r["measure"]["sim"]["moving"][metric]
            - r["measure"]["real"]["moving"][metric]
            for r in rows
            if r["measure"]["real"]["moving"]["visible"]
            and r["measure"]["sim"]["moving"]["visible"]
        ]
        summary[f"moving_{metric}"] = {
            "matched": len(deltas),
            "mean_delta_sim_minus_real": round(float(np.mean(deltas)), 4),
            "mean_abs_delta": round(float(np.mean(np.abs(deltas))), 4),
        }
    return summary


def overlay(
    rgb: np.ndarray,
    moving: np.ndarray,
    fixed: np.ndarray | None,
) -> np.ndarray:
    out = rgb.copy()
    out[moving] = (0.4 * out[moving] + 0.6 * np.array([60, 255, 60])).astype(np.uint8)
    if fixed is not None:
        out[fixed] = (0.4 * out[fixed] + 0.6 * np.array([60, 140, 255])).astype(
            np.uint8,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, default=PAIRS_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sim/wrist_refit"))
    parser.add_argument("--qc", action="store_true", help="write overlay strips")
    parser.add_argument("--qc-every", type=int, default=24)
    args = parser.parse_args()

    manifest = json.loads((args.pairs_dir / "manifest.json").read_text())
    sim = SO101Sim(render_style=manifest["render_style"])
    sim.reset(manifest["scene_seed"])
    masker = SimMasker(sim)

    rows = []
    qc_dir = args.out_dir / "qc"
    if args.qc:
        qc_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(manifest["rows"]):
        real_rgb = np.asarray(Image.open(args.pairs_dir / row["real"]))
        sim_masks = masker.masks_at(np.array(row["state_deg"]))
        measure, real_moving, real_fixed = measure_pair(real_rgb, sim_masks)
        rows.append(
            {
                "episode": row["episode"],
                "frame": row["frame"],
                "measure": measure,
            },
        )
        if args.qc and index % args.qc_every == 0:
            sim_rgb = np.asarray(Image.open(args.pairs_dir / row["sim"]))
            strip = np.concatenate(
                [
                    overlay(real_rgb, real_moving, real_fixed),
                    np.full((real_rgb.shape[0], 8, 3), 24, dtype=np.uint8),
                    overlay(sim_rgb, sim_masks["moving"], sim_masks["fixed"]),
                ],
                axis=1,
            )
            stem = f"ep{row['episode']:03d}_f{row['frame']:05d}"
            Image.fromarray(strip).save(qc_dir / f"qc_{stem}.png")
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(manifest['rows'])} pairs measured")

    out = {
        "pairs_dir": str(args.pairs_dir),
        "wrist_cam": manifest["wrist_cam"],
        "thresholds": {
            "orange_r_minus_b": ORANGE_R_MINUS_B,
            "orange_r_minus_g": ORANGE_R_MINUS_G,
            "orange_v_min": ORANGE_V_MIN,
            "dark_v_max": DARK_V_MAX,
            "dark_fill_min": DARK_FILL_MIN,
            "min_area": MIN_AREA,
            "band_frac": BAND_FRAC,
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "measurements.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
