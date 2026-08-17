"""Wrist-cam pose fit + held-out validation (queue: wrist-cam-pose-refit,
stage 3; pre-registered in-channel before running — msg 1538759641591324747).

Fits a mount-local wrist-camera pose correction — rotation deltas
(pitch, yaw, roll, camera-frame axes) plus a camera-frame position
offset — minimizing the matched-set discrepancy of the stage-2
moving-jaw reads. fovy is FROZEN: on the deployed equidistant path
cam.fovy is the fisheye remap's source constant (the lens model), not a
free optical parameter.

Objective, per train pair (real reads precomputed by stage 2):
  both visible   (cx_s-cx_r)^2 + (cy_s-cy_r)^2
                 + 0.5 * (wrap180(axis_s-axis_r)/90)^2
                 + (sqrt(area_s)-sqrt(area_r))^2
  real-only      MISS_MOVING (sim lost the jaw the real camera sees)
  fixed jaw      MISS_FIXED when real/sim disagree on visibility
                 (small: the real detector is itself ~7% noisy)

Split: the 26 reference episodes sorted, every 3rd (index % 3 == 2)
HELD OUT — the fit never sees them. Gates (pre-registered) evaluate on
held-out pairs only:
  G1  mean moving-jaw centroid error improves >= 50% vs the v1 pose
  G2  sim both-jaws-visible rate within 15 points of real
  G3  mean |bottom_occ delta| improves >= 50%
  (sim mount-visible rate: report-only)

Search: coarse rotation-only grid on a train subsample, then greedy
pattern search over all 6 parameters with shrinking steps (no scipy in
the env; ~150 evaluations).

Outputs (out-dir, default outputs/sim/wrist_refit):
  fit.json            fit record: search trace tail, fitted pose
                      (mount-local pos + quat, ship-ready), gate table
  refit_sbs/*.png     real | sim v1 | sim refit strips (held-out eps)

Usage:
  uv run python fontaine/scripts/wrist_cam_pose_fit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fontaine.scripts.wrist_cam_pose_measure import SimMasker, metrics
from sim.so101_sim import SO101Sim

# MISS_MOVING must dominate the worst visible-frame cost (centroid can
# reach ~2.0, axis 0.25, area 0.25): the first fit run priced it at
# 0.08 and the optimizer discovered that pointing the camera away from
# the jaw entirely (every frame a miss, 0.0989/frame) beat v1's
# visible-but-displaced cost (0.147/frame) — a degenerate optimum.
AXIS_WEIGHT = 0.25
MISS_MOVING = 0.5
MISS_FIXED = 0.05
ROT_BOUND = 35.0  # deg, per rotation delta
POS_BOUND = 0.03  # m, per camera-frame offset axis

GRID_PITCH = (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0)
GRID_YAW = (-5.0, 0.0, 5.0)
GRID_ROLL = (-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0)


def wrap180(delta: float) -> float:
    """Axis angles live on a 180-deg circle."""
    return (delta + 90.0) % 180.0 - 90.0


def pose_from_params(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(pitch, yaw, roll deg, dx, dy, dz m) -> mount-local pos, quat.
    Rotations compose in the CAMERA frame (right-multiply); the offset
    is expressed in camera axes and rotated out to the mount frame."""
    pitch, yaw, roll = np.deg2rad(params[:3])
    quat = base_quat.copy()
    for axis, angle in (((1, 0, 0), pitch), ((0, 1, 0), yaw), ((0, 0, 1), roll)):
        if angle == 0.0:
            continue
        delta = np.empty(4)
        mujoco.mju_axisAngle2Quat(delta, np.array(axis, dtype=float), angle)
        out = np.empty(4)
        mujoco.mju_mulQuat(out, quat, delta)
        quat = out
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, base_quat)
    pos = base_pos + rot.reshape(3, 3) @ params[3:]
    return pos, quat


class Evaluator:
    def __init__(self, sim: SO101Sim, rows: list[dict], states: dict) -> None:
        self.sim = sim
        self.masker = SimMasker(sim)
        self.rows = rows
        self.states = states
        cam = sim.model.camera("wrist_cam")
        self.base_pos = cam.pos.copy()
        self.base_quat = cam.quat.copy()
        self.cache: dict[tuple, dict] = {}

    def set_pose(self, params: np.ndarray) -> None:
        pos, quat = pose_from_params(self.base_pos, self.base_quat, params)
        cam = self.sim.model.camera("wrist_cam")
        cam.pos[:] = pos
        cam.quat[:] = quat

    def sim_reads(
        self,
        params: np.ndarray,
        rows: list[dict],
        *,
        mount: bool = False,
    ) -> list[dict]:
        """Per-row sim moving/fixed(/mount) metrics at the candidate
        pose; the mount mask (a third lens remap) only when asked —
        it is report-only and skipping it saves ~25% per search eval."""
        self.set_pose(params)
        full = self.masker.parts
        if not mount:
            self.masker.parts = {k: full[k] for k in ("moving", "fixed")}
        try:
            reads = []
            for row in rows:
                masks = self.masker.masks_at(
                    self.states[(row["episode"], row["frame"])],
                )
                read = {
                    "moving": metrics(masks["moving"], axis=True),
                    "fixed": metrics(masks["fixed"]),
                }
                if mount:
                    read["mount"] = metrics(masks["mount"])
                reads.append(read)
        finally:
            self.masker.parts = full
        return reads

    pos_bound = POS_BOUND

    def loss(self, params: np.ndarray, rows: list[dict]) -> float:
        if np.any(np.abs(params[:3]) > ROT_BOUND) or np.any(
            np.abs(params[3:]) > self.pos_bound,
        ):
            return 1e6
        key = (*np.round(params, 6), len(rows))
        if key in self.cache:
            return self.cache[key]
        total = 0.0
        for row, sim_read in zip(rows, self.sim_reads(params, rows), strict=True):
            real = row["measure"]["real"]
            if not real["moving"]["visible"]:
                continue
            moving = sim_read["moving"]
            if not moving["visible"]:
                total += MISS_MOVING
            else:
                total += (moving["cx"] - real["moving"]["cx"]) ** 2
                total += (moving["cy"] - real["moving"]["cy"]) ** 2
                total += (
                    AXIS_WEIGHT
                    * (wrap180(moving["axis_deg"] - real["moving"]["axis_deg"]) / 90.0)
                    ** 2
                )
                total += (
                    np.sqrt(moving["area_frac"]) - np.sqrt(real["moving"]["area_frac"])
                ) ** 2
            if real["fixed"] is not None and (
                real["fixed"]["visible"] != sim_read["fixed"]["visible"]
            ):
                total += MISS_FIXED
        value = total / len(rows)
        self.cache[key] = value
        return value


def validation(evaluator: Evaluator, params: np.ndarray, rows: list[dict]) -> dict:
    """The registered metric table at one pose, on one split."""
    reads = evaluator.sim_reads(params, rows, mount=True)
    centroid, bottom, axis = [], [], []
    fixed_agree, sim_pair_visible, mount_visible = [], [], []
    for row, sim_read in zip(rows, reads, strict=True):
        real = row["measure"]["real"]
        mount_visible.append(sim_read["mount"]["visible"])
        if real["moving"]["visible"] and sim_read["moving"]["visible"]:
            centroid.append(
                float(
                    np.hypot(
                        sim_read["moving"]["cx"] - real["moving"]["cx"],
                        sim_read["moving"]["cy"] - real["moving"]["cy"],
                    ),
                ),
            )
            bottom.append(
                abs(sim_read["moving"]["bottom_occ"] - real["moving"]["bottom_occ"]),
            )
            axis.append(
                abs(
                    wrap180(
                        sim_read["moving"]["axis_deg"] - real["moving"]["axis_deg"],
                    ),
                ),
            )
        if real["fixed"] is not None:
            fixed_agree.append(real["fixed"]["visible"] == sim_read["fixed"]["visible"])
        sim_pair_visible.append(
            sim_read["moving"]["visible"] and sim_read["fixed"]["visible"],
        )
    real_pair = [
        r["measure"]["real"]["both_jaws_visible"]
        for r in rows
        if r["measure"]["real"]["both_jaws_visible"] is not None
    ]
    return {
        "pairs": len(rows),
        "centroid_err_mean": round(float(np.mean(centroid)), 4),
        "bottom_occ_absdelta_mean": round(float(np.mean(bottom)), 4),
        "axis_absdelta_mean_deg": round(float(np.mean(axis)), 2),
        "both_jaws_visible_rate": {
            "real": round(float(np.mean(real_pair)), 4),
            "sim": round(float(np.mean(sim_pair_visible)), 4),
        },
        "fixed_visibility_agreement": round(float(np.mean(fixed_agree)), 4),
        "sim_mount_visible_rate": round(float(np.mean(mount_visible)), 4),
    }


def pattern_search(
    evaluator: Evaluator,
    start: np.ndarray,
    rows: list[dict],
    trace: list,
) -> np.ndarray:
    """Greedy coordinate pattern search, shrinking steps."""
    best = start.copy()
    best_loss = evaluator.loss(best, rows)
    for rot_step, pos_step in ((4.0, 0.008), (2.0, 0.004), (1.0, 0.002), (0.5, 0.001)):
        improved = True
        while improved:
            improved = False
            for index in range(6):
                step = rot_step if index < 3 else pos_step
                for sign in (1.0, -1.0):
                    candidate = best.copy()
                    candidate[index] += sign * step
                    loss = evaluator.loss(candidate, rows)
                    if loss < best_loss - 1e-9:
                        best, best_loss = candidate, loss
                        improved = True
            trace.append(
                {
                    "step": [rot_step, pos_step],
                    "params": [round(float(v), 5) for v in best],
                    "loss": round(best_loss, 6),
                },
            )
    print(f"pattern search done: loss {best_loss:.6f} at {np.round(best, 4)}")
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sim/wrist_refit"))
    parser.add_argument("--sbs-count", type=int, default=8)
    parser.add_argument(
        "--pos-bound",
        type=float,
        default=POS_BOUND,
        help="per-axis camera-frame offset bound, m",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="skip the grid; start the pattern search from fit.json's params",
    )
    args = parser.parse_args()

    measurements = json.loads((args.out_dir / "measurements.json").read_text())
    pairs_dir = Path(measurements["pairs_dir"])
    manifest = json.loads((pairs_dir / "manifest.json").read_text())
    states = {
        (row["episode"], row["frame"]): np.array(row["state_deg"])
        for row in manifest["rows"]
    }

    episodes = sorted({row["episode"] for row in measurements["rows"]})
    held_out = set(episodes[2::3])
    train_rows = [r for r in measurements["rows"] if r["episode"] not in held_out]
    held_rows = [r for r in measurements["rows"] if r["episode"] in held_out]
    print(
        f"split: {len(train_rows)} train / {len(held_rows)} held-out pairs "
        f"(held-out eps {sorted(held_out)})",
    )

    sim = SO101Sim(render_style=manifest["render_style"])
    sim.reset(manifest["scene_seed"])
    evaluator = Evaluator(sim, measurements["rows"], states)

    evaluator.pos_bound = args.pos_bound
    if args.warm_start:
        prior = json.loads((args.out_dir / "fit.json").read_text())["params"]
        best_grid = np.array(
            [prior["pitch_deg"], prior["yaw_deg"], prior["roll_deg"]]
            + prior["offset_cam_frame_m"],
        )
        print(f"warm start from fit.json: {best_grid}")
    else:
        # Coarse rotation-only grid on a train subsample.
        subsample = train_rows[::4]
        best_grid, best_loss = np.zeros(6), np.inf
        for pitch in GRID_PITCH:
            for yaw in GRID_YAW:
                for roll in GRID_ROLL:
                    params = np.array([pitch, yaw, roll, 0.0, 0.0, 0.0])
                    loss = evaluator.loss(params, subsample)
                    if loss < best_loss:
                        best_grid, best_loss = params, loss
                        print(f"grid best {loss:.6f} at {params[:3]}")

    # Pattern search on every 2nd train pair (loss is a 100+-pair mean;
    # the halved set tracks the full one and halves the render bill).
    trace: list = []
    fitted = pattern_search(evaluator, best_grid, train_rows[::2], trace)

    v1 = np.zeros(6)
    report = {
        "train": {
            "v1": validation(evaluator, v1, train_rows),
            "refit": validation(evaluator, fitted, train_rows),
        },
        "held_out": {
            "v1": validation(evaluator, v1, held_rows),
            "refit": validation(evaluator, fitted, held_rows),
        },
    }
    held_v1, held_fit = report["held_out"]["v1"], report["held_out"]["refit"]
    gates = {
        "G1_centroid_halved": held_fit["centroid_err_mean"]
        <= 0.5 * held_v1["centroid_err_mean"],
        "G2_pair_visibility_within_15pts": abs(
            held_fit["both_jaws_visible_rate"]["sim"]
            - held_fit["both_jaws_visible_rate"]["real"],
        )
        <= 0.15,
        "G3_bottom_occ_halved": held_fit["bottom_occ_absdelta_mean"]
        <= 0.5 * held_v1["bottom_occ_absdelta_mean"],
    }

    pos, quat = pose_from_params(evaluator.base_pos, evaluator.base_quat, fitted)
    record = {
        "pre_registration": "discord msg 1538759641591324747",
        "split": {"held_out_episodes": sorted(held_out)},
        "params": {
            "pitch_deg": round(float(fitted[0]), 3),
            "yaw_deg": round(float(fitted[1]), 3),
            "roll_deg": round(float(fitted[2]), 3),
            "offset_cam_frame_m": [round(float(v), 5) for v in fitted[3:]],
        },
        "pose": {
            "base_pos": [round(float(v), 5) for v in evaluator.base_pos],
            "base_quat": [round(float(v), 5) for v in evaluator.base_quat],
            "pos": [round(float(v), 5) for v in pos],
            "quat": [round(float(v), 5) for v in quat / np.linalg.norm(quat)],
        },
        "grid_best": [round(float(v), 3) for v in best_grid[:3]],
        "search_trace": trace,
        "report": report,
        "gates": gates,
    }
    (args.out_dir / "fit.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({"report": report, "gates": gates}, indent=1))

    # Held-out side-by-side strips: real | sim v1 | sim refit.
    sbs_dir = args.out_dir / "refit_sbs"
    sbs_dir.mkdir(parents=True, exist_ok=True)
    picks = held_rows[:: max(1, len(held_rows) // args.sbs_count)][: args.sbs_count]
    for row in picks:
        stem = f"ep{row['episode']:03d}_f{row['frame']:05d}"
        real = np.asarray(Image.open(pairs_dir / f"real_{stem}.png"))
        panels = [real]
        for params in (v1, fitted):
            evaluator.set_pose(params)
            sim.data.qpos[sim._joint_qpos] = np.deg2rad(
                states[(row["episode"], row["frame"])],
            )
            mujoco.mj_forward(sim.model, sim.data)
            panels.append(sim.observe().wrist)
        gap = np.full((real.shape[0], 8, 3), 24, dtype=np.uint8)
        strip = np.concatenate(
            [panels[0], gap, panels[1], gap, panels[2]],
            axis=1,
        )
        Image.fromarray(strip).save(sbs_dir / f"refit_{stem}.png")
    print(f"-> {args.out_dir / 'fit.json'}, {len(picks)} strips in {sbs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
