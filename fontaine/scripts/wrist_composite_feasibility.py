"""Wrist-compositing feasibility read (owner steer 22:21Z 08-12).

CPU-only, no renders: forward-kinematics + analytic fisheye ray
geometry answer the three facts a wrist-composite design needs.

1. Plate-pose spread: wrist_cam world pose at the episode-start plate
   window (frame 0) of each A-half reference episode — how far apart
   the poses a static plate would average over actually are.
2. Eval-pose wander: at replan cadence (every 30 ticks), distance from
   the wrist pose to the NEAREST plate pose — what a nearest-plate
   homography warp has to bridge.
3. Analytic coverage: fraction of wrist fisheye rays that hit the
   table plane at all (the plane-warpable fraction), and of those, the
   fraction whose table point lies inside the nearest plate camera's
   fisheye cone (the warp-fillable fraction). Off-plane and
   out-of-cone pixels can never come from a plane-warped plate.

Rays use the v1 output fisheye mapping (theta = r_px / F_DIST with
F_DIST pinned to the 52-deg pinhole focal) on the deployed 640x480
grid; camera convention: MuJoCo cameras look along -Z of cam_xmat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fontaine.scripts.make_clean_plates import a_half_episodes
from sim.so101_sim import SO101Sim

WIDTH, HEIGHT = 640, 480
REPLAN_TICKS = 30
GRID = (64, 48)  # ray subsample of the 640x480 frame


def wrist_pose(sim: SO101Sim, state_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.clip(np.deg2rad(state_deg), sim._ctrl_low, sim._ctrl_high)
    sim.data.qpos[sim._joint_qpos] = q
    mujoco.mj_forward(sim.model, sim.data)
    cam = sim.model.camera("wrist_cam").id
    return sim.data.cam_xpos[cam].copy(), sim.data.cam_xmat[cam].reshape(3, 3).copy()


def fisheye_rays() -> np.ndarray:
    """Unit ray directions (camera frame, -Z forward) on the output grid."""
    f_dist = (HEIGHT / 2.0) / np.tan(np.deg2rad(SO101Sim.V1_CENTER_FOVY) / 2.0)
    u = np.linspace(-WIDTH / 2.0, WIDTH / 2.0, GRID[0])
    v = np.linspace(-HEIGHT / 2.0, HEIGHT / 2.0, GRID[1])
    uu, vv = np.meshgrid(u, v)
    r = np.hypot(uu, vv)
    theta = r / f_dist
    phi = np.arctan2(vv, uu)
    d = np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            -np.cos(theta),
        ],
        axis=-1,
    )
    return d.reshape(-1, 3)


def main() -> None:
    root = Path("~/datasets/mcobzarenco/so101_pick_place_v2").expanduser()
    episodes, _total = a_half_episodes(root)
    ep_ids = [int(e) for e in episodes["episode_index"]]

    df = pd.read_parquet(
        root / "data" / "chunk-000" / "file-000.parquet",
        columns=["episode_index", "frame_index", "observation.state"],
    )
    df = df[df["episode_index"].isin(ep_ids)]

    sim = SO101Sim(render_style="v0")  # physics only, renderer never built
    table = sim.model.geom("table")
    z_table = float(table.pos[2] + table.size[2])
    tx, ty = float(table.pos[0]), float(table.pos[1])
    sx, sy = float(table.size[0]), float(table.size[1])

    rays = fisheye_rays()
    theta_max = float(
        np.hypot(WIDTH / 2, HEIGHT / 2)
        / ((HEIGHT / 2.0) / np.tan(np.deg2rad(SO101Sim.V1_CENTER_FOVY) / 2.0)),
    )

    plate_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    eval_poses: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        states = np.stack(g["observation.state"].to_list()).astype(np.float64)
        plate_poses[int(ep)] = wrist_pose(sim, states[0])
        for t in range(0, len(states), REPLAN_TICKS):
            pos, mat = wrist_pose(sim, states[t])
            eval_poses.append((int(ep), t, pos, mat))

    # 1. plate-pose spread
    pos_arr = np.stack([p for p, _ in plate_poses.values()])
    axes = np.stack([-m[:, 2] for _, m in plate_poses.values()])  # look dirs
    centroid_dist_mm = np.linalg.norm(pos_arr - pos_arr.mean(0), axis=1) * 1000
    mean_axis = axes.mean(0)
    mean_axis /= np.linalg.norm(mean_axis)
    angles_deg = np.degrees(np.arccos(np.clip(axes @ mean_axis, -1, 1)))
    print(
        f"plate poses (n={len(pos_arr)}): pos spread mm "
        f"median {np.median(centroid_dist_mm):.1f} p90 {np.percentile(centroid_dist_mm, 90):.1f} "
        f"max {centroid_dist_mm.max():.1f}; look-axis deg "
        f"median {np.median(angles_deg):.2f} p90 {np.percentile(angles_deg, 90):.2f} "
        f"max {angles_deg.max():.2f}",
    )

    # 2 + 3. eval-pose wander and analytic coverage
    plate_pos = pos_arr
    plate_axes = axes
    nn_mm, nn_deg, plane_fracs, fill_fracs = [], [], [], []
    for _ep, _t, pos, mat in eval_poses:
        d_pos = np.linalg.norm(plate_pos - pos, axis=1)
        j = int(d_pos.argmin())
        look = -mat[:, 2]
        nn_mm.append(d_pos[j] * 1000)
        nn_deg.append(
            float(np.degrees(np.arccos(np.clip(look @ plate_axes[j], -1, 1)))),
        )

        d_world = rays @ mat.T
        dz = d_world[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_hit = (z_table - pos[2]) / dz
        pts = pos + t_hit[:, None] * d_world
        on_plane = (
            (t_hit > 0) & (np.abs(pts[:, 0] - tx) < sx) & (np.abs(pts[:, 1] - ty) < sy)
        )
        plane_fracs.append(float(on_plane.mean()))

        pp, pa = plate_pos[j], plate_axes[j]
        v = pts[on_plane] - pp
        vn = v / np.linalg.norm(v, axis=1, keepdims=True)
        in_cone = np.degrees(np.arccos(np.clip(vn @ pa, -1, 1))) < np.degrees(theta_max)
        fill_fracs.append(float(in_cone.mean()) if on_plane.any() else 0.0)

    def q(x: list[float]) -> str:
        a = np.asarray(x)
        return f"median {np.median(a):.3g} p10 {np.percentile(a, 10):.3g} p90 {np.percentile(a, 90):.3g}"

    print(f"eval poses (n={len(eval_poses)}, replan cadence):")
    print(f"  nearest-plate pos mm: {q(nn_mm)}")
    print(f"  nearest-plate look deg: {q(nn_deg)}")
    print(f"  table-plane ray fraction: {q(plane_fracs)}")
    print(f"  warp-fillable (of plane rays, nearest plate cone): {q(fill_fracs)}")
    print(
        f"  theta_max {np.degrees(theta_max):.1f} deg; table z {z_table:.4f} m; "
        f"episodes {len(plate_poses)}",
    )

    out = Path("outputs/sim/wrist_composite_feasibility.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "plate_pos_spread_mm": centroid_dist_mm.tolist(),
                "plate_axis_spread_deg": angles_deg.tolist(),
                "nn_mm": nn_mm,
                "nn_deg": nn_deg,
                "plane_frac": plane_fracs,
                "fill_frac": fill_fracs,
            },
        ),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
