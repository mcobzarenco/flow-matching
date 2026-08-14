"""W3 mask visual spot-check on 3 banked pose slots — the stage-0
CPU-side oracle of the wrist-transfer screen (pre-reg
posts/2026-08-14-prereg-wrist-transfer-screen.md §1; queue
`wrist-transfer-stage0-cpu-prep`).

Re-derives the honesty probe's 100 manipulation-pose slots exactly as
the banked rollout-pose read did (held episodes >= 26, mid-band,
pick_evenly — deterministic, no RNG), poses the production v3 sim at
3 spread slots, and renders per slot: the wrist frame, the
`wrist_arm_mask()` overlay, and the full `arm_blur` treatment. A human
eyeballs the panels: the mask must cover the arm+gripper (and only
them), and the treatment must visibly kill arm texture while leaving
background and silhouette alone.

In-script checks: mask in [0, 1]; manipulation-pose coverage in a sane
band (the gripper fills a real fraction of the frame); the treated
frame bit-matches the raw frame outside the (hard) mask support.

GPU fence (owner reserve): render on Mesa's software EGL —
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \\
  LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= \\
  uv run python fontaine/scripts/wrist_transform_spotcheck.py \\
      --out-dir reports/wrist_transform_stage0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

from sim_rollout_pose_wrist_read import (
    HELD_EPISODE_MIN,
    N_DRAWS,
    N_SLOTS,
    load_tables,
    mid_band_pool,
    pick_evenly,
)

SPOT_SLOTS = (0, 50, 99)  # spread over the 100-slot grid
# Sanity band for arm+gripper coverage at MANIPULATION poses. Measured
# fact (this spot-check, slot 0): a legit manipulation pose can show
# only a corner of the moving jaw — coverage 0.011 with the mask
# pixel-exact on it — so the floor only catches a mask that missed the
# arm ENTIRELY; the ceiling catches one that swallowed the scene.
COVERAGE_BAND = (0.005, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=Path("~/datasets/mcobzarenco/so101_pick_place_v2").expanduser(),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/wrist_transform_stage0"),
    )
    return parser.parse_args()


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Red-tinted mask overlay (alpha rides the soft mask)."""
    alpha = 0.55 * mask[..., None]
    red = np.array([255.0, 40.0, 40.0])
    return np.clip(
        frame.astype(np.float64) * (1 - alpha) + red * alpha,
        0,
        255,
    ).astype(np.uint8)


def main() -> int:
    args = parse_args()
    import mujoco
    from PIL import Image

    from sim.so101_sim import SO101Sim
    from sim.wrist_transform import ArmBlurTransform

    data, _eps = load_tables(args.v2_root)
    n_episodes = int(data.episode_index.max()) + 1
    held_pool = mid_band_pool(data, list(range(HELD_EPISODE_MIN, n_episodes)))
    held_rows = pick_evenly(held_pool, N_SLOTS)
    states_deg = np.stack(held_rows["observation.state"].to_list()).astype(np.float64)
    print(f"held pool {len(held_pool)} rows -> {N_SLOTS} slots (banked derivation)")

    sim = SO101Sim(post_backend="numpy")  # production v3, deployed lens
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for slot in SPOT_SLOTS:
        seed, draw = slot // N_DRAWS, slot % N_DRAWS
        start = time.perf_counter()
        sim.reset(seed, appearance_seed=1000 * draw + seed)
        sim.data.qpos[sim._joint_qpos] = np.clip(
            np.deg2rad(states_deg[slot]),
            sim._ctrl_low,
            sim._ctrl_high,
        )
        mujoco.mj_forward(sim.model, sim.data)
        obs = sim.observe()
        transform = ArmBlurTransform(sim)
        treated = transform(obs)
        mask = np.asarray(sim.wrist_arm_mask())
        coverage = transform.coverage[0]
        elapsed = time.perf_counter() - start

        if not (mask.min() >= 0.0 and mask.max() <= 1.0):
            raise SystemExit(f"slot {slot}: mask outside [0, 1]")
        if not (COVERAGE_BAND[0] <= coverage <= COVERAGE_BAND[1]):
            raise SystemExit(
                f"slot {slot}: coverage {coverage:.3f} outside {COVERAGE_BAND}",
            )
        untouched = mask == 0.0
        if not np.array_equal(treated.wrist[untouched], obs.wrist[untouched]):
            raise SystemExit(f"slot {slot}: pixels outside the mask changed")

        panel = np.concatenate(
            [obs.wrist, overlay(obs.wrist, mask), treated.wrist],
            axis=1,
        )
        path = args.out_dir / f"slot{slot:03d}_wrist_mask_armblur.png"
        Image.fromarray(panel).save(path)
        row = {
            "slot": slot,
            "seed": seed,
            "draw": draw,
            "coverage": round(coverage, 4),
            "mask_soft_px": int((mask > 0).sum()),
            "mask_hard_px": int((mask > 0.5).sum()),
            "render_s": round(elapsed, 2),
            "panel": str(path),
        }
        summary.append(row)
        print(json.dumps(row))

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {args.out_dir}/summary.json — eyeball the 3 panels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
