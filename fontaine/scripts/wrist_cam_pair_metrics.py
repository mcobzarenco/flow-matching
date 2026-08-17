"""Wrist-cam refit stages 2+3: measure the matched pairs, fit the pose
(queue: wrist-cam-pose-refit; owner priority 2026-08-16 23:56Z "How
will we fix it? Let's prioritise this work, use the local machine").

Stage 2 (--measure): on every matched pair from
wrist_cam_matched_pairs.py, measure the ORANGE MOVING JAW both sides —
real via hue segmentation (the one rig color that segments robustly),
sim via an exact segmentation render (moving-jaw geom ids) pushed
through the same equidistant lens remap as the deployed wrist frame.
Per-side metrics:
  visible      mask area >= AREA_MIN px (of 640x480)
  angle_deg    principal-axis angle from image-vertical (PCA), [-90,90)
  bottom_occ   mask coverage of the bottom-quarter band
  cx, cy       mask centroid, normalized [0,1]
The real columns are the TARGET distribution; sim-minus-real is the
defect as a number.

Stage 3 (--fit): grid-search a mount-local camera correction applied
to the current _repose_wrist_cam pose — roll (about the optical axis)
and tilt (about the image-horizontal axis), optionally pan/fovy — by
re-rendering the sim jaw mask at each candidate on the FIT episodes
and scoring against the real metrics; validates the winner on the
HELD-OUT episodes it never saw. Loss per pair (real-visible pairs):
  |d angle|/10 deg + 10*|d bottom_occ| + 4*|d cx| + 40*vis_mismatch
(explicit weights; per-term means reported so the aggregate hides
nothing). The visibility penalty strictly dominates the other terms'
worst case (~30) — the first fit run scored jaw-hiding poses at 2.0
and every top-grid candidate was degenerate-invisible, so a candidate
that hides the jaw must always lose to any visible one. Physics
untouched throughout — pose is render-only.

Usage:
  uv run python fontaine/scripts/wrist_cam_pair_metrics.py --measure
  uv run python fontaine/scripts/wrist_cam_pair_metrics.py --fit
  uv run python fontaine/scripts/wrist_cam_pair_metrics.py --fit --pan --fovy
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.so101_sim import SO101Sim

PAIRS_DIR = Path("outputs/sim/wrist_refit/matched_pairs")
OUT_DIR = Path("outputs/sim/wrist_refit")
AREA_MIN = 300
BAND_ROWS = 120  # bottom quarter of 480

# Held-out split: fit on episodes 0-17, validate on 18-25 (queue plan:
# "fit ... minimizing the matched-set discrepancy, validate held-out").
FIT_EPISODES = frozenset(range(18))

# Real-side orange-jaw hue bands (cv2 HSV, H in 0..180). Calibrated on
# ep0 f0 jaw-patch percentiles 2026-08-17: the jaw reads SALMON under
# rig lighting — H wraps the red boundary (5..95% span 5..176), S is
# only moderate (43..66 IQR) but V is bright (196+), while the wood
# table/disk are near-gray (S 4..12, V ~110). Discriminator: warm hue
# (either side of the wrap) + S>=25 + V>=140.
HSV_BANDS = (
    ((0, 25, 140), (25, 255, 255)),
    ((160, 25, 140), (180, 255, 255)),
)


def mask_metrics(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    area = len(xs)
    if area < AREA_MIN:
        return {"visible": False, "area": area}
    pts = np.stack([xs - xs.mean(), ys - ys.mean()])
    cov = pts @ pts.T / area
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, int(np.argmax(evals))]
    # angle from image-vertical (up), in [-90, 90)
    angle = np.degrees(np.arctan2(vx, -vy))
    if angle >= 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    h, w = mask.shape
    band = mask[h - BAND_ROWS :, :]
    return {
        "visible": True,
        "area": area,
        "angle_deg": float(angle),
        "bottom_occ": float(band.mean()),
        "cx": float(xs.mean() / w),
        "cy": float(ys.mean() / h),
    }


def real_jaw_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for lo, hi in HSV_BANDS:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi)).astype(bool)
    # largest connected component only — kills stray warm speckle
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if n <= 1:
        return np.zeros_like(mask)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


class SimJawRenderer:
    """Moving-jaw segmentation masks at recorded states, under an
    optional mount-local camera correction on top of the deployed
    _repose_wrist_cam pose."""

    def __init__(self) -> None:
        # v1: the lens machinery exists (the mask must ride the same
        # equidistant remap as the pair renders) without the composite
        # path's clutter bookkeeping.
        self.sim = SO101Sim(render_style="v1")
        self.sim.reset(0)
        cam = self.sim.model.camera("wrist_cam")
        self.cam = cam
        self.base_quat = cam.quat.copy()
        self.base_fovy = float(cam.fovy[0])
        body = self.sim.model.body("moving_jaw_so101_v1").id
        self.jaw_geoms = np.array(
            [
                g
                for g in range(self.sim.model.ngeom)
                if self.sim.model.geom_bodyid[g] == body
            ],
        )

    def set_correction(
        self,
        roll_deg: float = 0.0,
        tilt_deg: float = 0.0,
        pan_deg: float = 0.0,
        dfovy: float = 0.0,
    ) -> None:
        """Compose local-frame rotations onto the base pose. Camera
        frame: looks along -z, image-right +x, image-up +y => tilt
        about +x, pan about +y, roll about +z (optical axis)."""

        def quat_about(axis: np.ndarray, deg: float) -> np.ndarray:
            half = np.radians(deg) / 2.0
            return np.concatenate([[np.cos(half)], np.sin(half) * axis])

        def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            w1, x1, y1, z1 = a
            w2, x2, y2, z2 = b
            return np.array(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ],
            )

        q = self.base_quat.copy()
        q = qmul(q, quat_about(np.array([1.0, 0.0, 0.0]), tilt_deg))
        q = qmul(q, quat_about(np.array([0.0, 1.0, 0.0]), pan_deg))
        q = qmul(q, quat_about(np.array([0.0, 0.0, 1.0]), roll_deg))
        self.cam.quat[:] = q
        self.cam.fovy[0] = self.base_fovy + dfovy

    def jaw_mask(self, state_deg: list[float]) -> np.ndarray:
        sim = self.sim
        sim.data.qpos[sim._joint_qpos] = np.deg2rad(state_deg)
        mujoco.mj_forward(sim.model, sim.data)
        renderer = sim.renderer
        renderer.enable_segmentation_rendering()
        renderer.update_scene(sim.data, camera="wrist_cam")
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
        mask = (is_geom & np.isin(seg[..., 0], self.jaw_geoms)).astype(np.float64)
        return sim._remap(mask[..., None])[..., 0] > 0.5


def pair_loss(real: dict, sim: dict) -> dict | None:
    """Per-pair discrepancy terms; None when the real jaw is hidden
    (nothing to match against)."""
    if not real["visible"]:
        return None
    if not sim["visible"]:
        return {"vis_mismatch": 1.0, "angle": 0.0, "occ": 0.0, "cx": 0.0}
    return {
        "vis_mismatch": 0.0,
        "angle": abs(real["angle_deg"] - sim["angle_deg"]),
        "occ": abs(real["bottom_occ"] - sim["bottom_occ"]),
        "cx": abs(real["cx"] - sim["cx"]),
    }


def aggregate(losses: list[dict]) -> dict:
    def mean(key: str) -> float:
        return float(np.mean([t[key] for t in losses]))

    terms = {k: mean(k) for k in ("angle", "occ", "cx", "vis_mismatch")}
    terms["loss"] = (
        terms["angle"] / 10.0
        + 10.0 * terms["occ"]
        + 4.0 * terms["cx"]
        + 40.0 * terms["vis_mismatch"]
    )
    terms["pairs"] = len(losses)
    return terms


def measure_real(rows: list[dict]) -> dict[str, dict]:
    from PIL import Image

    out = {}
    for row in rows:
        rgb = np.asarray(Image.open(PAIRS_DIR / row["real"]))
        out[row["sbs"]] = mask_metrics(real_jaw_mask(rgb))
    return out


def measure_sim(rows: list[dict], renderer: SimJawRenderer) -> dict[str, dict]:
    return {
        row["sbs"]: mask_metrics(renderer.jaw_mask(row["state_deg"])) for row in rows
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--pan", action="store_true", help="fit pan too")
    parser.add_argument("--fovy", action="store_true", help="fit dfovy too")
    args = parser.parse_args()

    manifest = json.loads((PAIRS_DIR / "manifest.json").read_text())
    rows = manifest["rows"]
    real = measure_real(rows)
    renderer = SimJawRenderer()

    if args.measure:
        sim = measure_sim(rows, renderer)
        losses = [
            t for row in rows if (t := pair_loss(real[row["sbs"]], sim[row["sbs"]]))
        ]
        summary = {
            "pairs": len(rows),
            "real_visible": sum(m["visible"] for m in real.values()),
            "sim_visible": sum(m["visible"] for m in sim.values()),
            "real_angle_mean": float(
                np.mean([m["angle_deg"] for m in real.values() if m["visible"]]),
            ),
            "sim_angle_mean": float(
                np.mean([m["angle_deg"] for m in sim.values() if m["visible"]]),
            ),
            "real_bottom_occ_mean": float(
                np.mean([m["bottom_occ"] for m in real.values() if m["visible"]]),
            ),
            "sim_bottom_occ_mean": float(
                np.mean([m["bottom_occ"] for m in sim.values() if m["visible"]]),
            ),
            "discrepancy": aggregate(losses),
        }
        out = OUT_DIR / "pair_metrics.json"
        out.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "real": real,
                    "sim": sim,
                    "hsv_bands": HSV_BANDS,
                },
                indent=1,
            ),
        )
        print(json.dumps(summary, indent=1))
        print(f"-> {out}")

    if args.fit:
        fit_rows = [r for r in rows if r["episode"] in FIT_EPISODES]
        held_rows = [r for r in rows if r["episode"] not in FIT_EPISODES]
        rolls = np.arange(-25.0, 25.1, 5.0)
        tilts = np.arange(-25.0, 25.1, 5.0)
        pans = np.arange(-10.0, 10.1, 5.0) if args.pan else [0.0]
        fovys = [-8.0, 0.0, 8.0] if args.fovy else [0.0]
        results = []
        for cand in itertools.product(rolls, tilts, pans, fovys):
            roll, tilt, pan, dfovy = (float(v) for v in cand)
            renderer.set_correction(roll, tilt, pan, dfovy)
            sim = measure_sim(fit_rows, renderer)
            losses = [
                t
                for row in fit_rows
                if (t := pair_loss(real[row["sbs"]], sim[row["sbs"]]))
            ]
            agg = aggregate(losses)
            results.append(
                {"roll": roll, "tilt": tilt, "pan": pan, "dfovy": dfovy, **agg},
            )
        results.sort(key=lambda r: r["loss"])
        best = results[0]
        print("top 5 candidates (fit split):")
        for r in results[:5]:
            print(
                f"  roll {r['roll']:+6.1f} tilt {r['tilt']:+6.1f} "
                f"pan {r['pan']:+5.1f} dfovy {r['dfovy']:+5.1f} "
                f"loss {r['loss']:.3f} (angle {r['angle']:.1f} occ {r['occ']:.3f} "
                f"cx {r['cx']:.3f} vis {r['vis_mismatch']:.2f})",
            )
        # refine at half-step around the winner
        fine = []
        for cand in itertools.product(
            np.arange(best["roll"] - 2.5, best["roll"] + 2.6, 2.5),
            np.arange(best["tilt"] - 2.5, best["tilt"] + 2.6, 2.5),
        ):
            roll, tilt = (float(v) for v in cand)
            renderer.set_correction(roll, tilt, best["pan"], best["dfovy"])
            sim = measure_sim(fit_rows, renderer)
            losses = [
                t
                for row in fit_rows
                if (t := pair_loss(real[row["sbs"]], sim[row["sbs"]]))
            ]
            fine.append({**best, "roll": roll, "tilt": tilt, **aggregate(losses)})
        fine.sort(key=lambda r: r["loss"])
        best = fine[0]

        # held-out validation: baseline vs winner on episodes 18-25
        def held_terms(roll: float, tilt: float, pan: float, dfovy: float) -> dict:
            renderer.set_correction(roll, tilt, pan, dfovy)
            sim = measure_sim(held_rows, renderer)
            losses = [
                t
                for row in held_rows
                if (t := pair_loss(real[row["sbs"]], sim[row["sbs"]]))
            ]
            return aggregate(losses)

        held_base = held_terms(0.0, 0.0, 0.0, 0.0)
        held_best = held_terms(
            best["roll"],
            best["tilt"],
            best["pan"],
            best["dfovy"],
        )
        out = {
            "winner": best,
            "grid_top10": results[:10],
            "refine_top5": fine[:5],
            "held_out": {"baseline": held_base, "winner": held_best},
            "base_pose": {
                "quat": [float(v) for v in renderer.base_quat],
                "fovy": renderer.base_fovy,
            },
        }
        path = OUT_DIR / "pose_fit.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"\nwinner: {json.dumps(best, indent=1)}")
        print(
            f"held-out baseline {held_base['loss']:.3f} -> winner {held_best['loss']:.3f}",
        )
        print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
