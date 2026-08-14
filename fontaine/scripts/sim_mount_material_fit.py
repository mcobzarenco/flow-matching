"""Camera-mount material — mine real mount-pixel statistics and fit the
mount grade (queue `sim-mount-material-split`; arm-split named the mount
the per-pixel most sim-distinctive class: no_mount is the ONLY removal
moving v3 toward real, 0.713->0.654 on ~0.66% px; the rider finding is
that the mount is WHITE in reality while the sim recolors it black).

Reuses the link-photometrics machinery (sim_arm_photometric_fit) with
one structural change: the mount is WHITE, so the darkness-snap that
registers the link masks cannot apply to it. Instead the mount mask
RIDES the dark gripper-cluster snap — the mount is rigidly attached to
the gripper body (camera_mount is its child, centimeters away at the
same chain depth), so the cluster's darkness lock registers the mount
too. Cluster union: the wrist + gripper bodies' rendered geoms minus
the mount and the orange moving jaw (black hardware in reality, same
guards as the link mine: ring ratio + absolute darkness).

Two subcommands (same shapes as the link fit):

  mine  Pose the sim at recorded real joints (26 reference-half v2
        episodes, 6 frames each), project follower cluster + mount
        masks through the production fisheye, snap the cluster by
        darkness, harvest real pixels under the shifted eroded mount
        mask.

  fit   Split the mount material off the gripper (the byte-identical
        detach in SO101Sim._split_mount_material), then fit albedo RGB
        + specular x shininess by matching the production v3 composite's
        mount-pixel statistics to the mined real ones (the link-fit
        objective: luma percentiles + channel medians, highlight tail
        weighted; albedo by 2-point linear solve, spec x shin by grid).

Output JSON feeds MOUNT_MATERIAL_V1 in sim/so101_sim.py — the opt-in
`mount_material="v1"` path gated by the pinned 20x5 probe
(sim_mount_material_read.py).

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_mount_material_fit.py \
      mine --out reports/analysis__mount_material_mine.json \
      --dump-overlays reports/mount_mine_overlays
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_mount_material_fit.py \
      fit --mined reports/analysis__mount_material_mine.json \
      --out reports/analysis__mount_material_fit.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
from sim_arm_photometric_fit import (
    EPISODES,
    FIT_LOOKS,
    FIT_POSES,
    FRAME_SPAN,
    FRAMES_PER_EPISODE,
    MASK_LUMA_MAX,
    MIN_BODY_PIXELS,
    RING_LUMA_MAX_RATIO,
    SHIN_GRID,
    SNAP_RANGE,
    SPEC_GRID,
    clipped_shift,
    decode_frames,
    episode_video_offsets,
    erode_mask,
    load_states,
    local_contrast,
    luma,
    pixel_stats,
    render_masks,
    set_real_pose,
    snap_shift,
    stats_loss,
)
from sim_top_gap_decomposition import dilate

MOUNT_ERODE = 2
# Snap bodies tried in order; the mount rides the FIRST that locks.
# gripper first — the mount's parent body, rigidly closest. A single
# wrist+gripper union mask saturated its darkness search on ~2/3 of
# frames (the ±60 px window reaches other dark blobs — the leader's
# identical gripper cluster is in frame); per-body masks lock on ~half
# the frames each and their union covers well past the frame bar.
CLUSTER_BODIES = ("gripper", "wrist")
MIN_FRAMES_KEPT = 60
MIN_MOUNT_POOL = 20_000  # ~0.66% of px/frame, thin part — smaller pool bar
# The mount is WHITE hardware riding a DARK cluster's lock: reject
# harvests whose mount pixels do not read clearly brighter than the
# locked dark body (a wrong lock lands the mount mask on arbitrary
# content; true mount pixels read ~2x the black hardware's luma).
MOUNT_OVER_CLUSTER_MIN_RATIO = 1.4
ALBEDO_PROBES = (0.3, 0.8)  # bright part: probe the upper albedo range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("mine", "fit"))
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=Path("~/datasets/mcobzarenco/so101_pick_place_v2").expanduser(),
    )
    parser.add_argument("--mined", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dump-overlays",
        type=Path,
        default=None,
        help="write mask-overlay PNGs for visual registration checks",
    )
    return parser.parse_args()


def mount_sets(sim) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Follower snap-anchor bodies (dark gripper/wrist geoms, separate
    masks) and the mount (harvest target); rendered geoms only
    (material-less geoms are the invisible collision set)."""
    model = sim.model
    sets: dict[str, list[int]] = {name: [] for name in (*CLUSTER_BODIES, "mount")}
    for geom in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[geom]).name
        if body_name.startswith("leader-") or int(model.geom_matid[geom]) < 0:
            continue
        if body_name == "camera_mount":
            sets["mount"].append(geom)
        elif body_name in CLUSTER_BODIES:
            sets[body_name].append(geom)
    out = {name: np.array(sorted(ids)) for name, ids in sets.items()}
    if len(out["mount"]) < 1 or any(len(out[b]) < 2 for b in CLUSTER_BODIES):
        raise SystemExit(
            f"set sizes look wrong: { {k: len(v) for k, v in out.items()} }",
        )
    return out


def mine(args: argparse.Namespace) -> dict:
    from sim.so101_sim import SO101Sim

    sim = SO101Sim(render_style="v3", post_backend="numpy")
    sim.reset(0)
    sets = mount_sets(sim)
    print({name: len(ids) for name, ids in sets.items()})

    states = load_states(args.v2_root)
    video_dir = args.v2_root / "videos" / "observation.images.front" / "chunk-000"
    offsets = episode_video_offsets(args.v2_root)

    by_file: dict[int, list[tuple[int, int, int]]] = {}
    for episode in EPISODES:
        traj = states[episode]
        file_index, start = offsets[episode]
        lo, hi = (int(len(traj) * f) for f in FRAME_SPAN)
        for t in np.linspace(lo, hi, FRAMES_PER_EPISODE).astype(int).tolist():
            by_file.setdefault(file_index, []).append((episode, t, start + t))

    pixels: list[np.ndarray] = []
    contrast: list[np.ndarray] = []
    kept, dropped, frame_log = 0, 0, []
    for file_index, entries in sorted(by_file.items()):
        decoded = decode_frames(
            video_dir / f"file-{file_index:03d}.mp4",
            [absolute for _, _, absolute in entries],
        )
        for episode, t, absolute in entries:
            traj = states[episode]
            frame = decoded[absolute]
            set_real_pose(sim, traj[t])
            raw = render_masks(sim, sets)
            lum = luma(frame)
            record = {"episode": episode, "frame": int(t), "kept": False}
            frame_log.append(record)
            lock: tuple[str, tuple[int, int], np.ndarray, float] | None = None
            for body in CLUSTER_BODIES:
                body_raw = raw[body]
                if body_raw.sum() < MIN_BODY_PIXELS:
                    continue
                shift = snap_shift(lum, body_raw)
                if shift is None or max(abs(s) for s in shift) >= SNAP_RANGE - 2:
                    continue  # saturated search = no confident lock
                moved = clipped_shift(body_raw, *shift)
                grown = moved.astype(np.float64)
                for _ in range(8):
                    grown = dilate(grown)
                near = dilate(dilate(dilate(moved.astype(np.float64)))) > 0.5
                ring = (grown > 0.5) & ~near
                if not ring.any() or not moved.any():
                    continue
                mask_luma = float(np.median(lum[moved]))
                ratio = mask_luma / max(np.median(lum[ring]), 1e-6)
                record[f"{body}_ratio"] = round(ratio, 3)
                if ratio >= RING_LUMA_MAX_RATIO or mask_luma >= MASK_LUMA_MAX:
                    continue
                lock = (body, shift, moved, mask_luma)
                break
            if lock is None:
                dropped += 1
                continue
            body, shift, moved, cluster_luma = lock
            record["lock_body"] = body
            record["shift"] = list(shift)
            # the mount rides its rigid dark neighbor's lock
            mount_mask = erode_mask(
                clipped_shift(raw["mount"], *shift),
                MOUNT_ERODE,
            )
            record["mount_px"] = int(mount_mask.sum())
            if mount_mask.sum() < MIN_BODY_PIXELS:
                dropped += 1
                continue
            mount_luma = float(np.median(lum[mount_mask]))
            record["mount_luma_median"] = round(mount_luma, 1)
            record["mount_over_cluster"] = round(
                mount_luma / max(cluster_luma, 1e-6),
                3,
            )
            if mount_luma < MOUNT_OVER_CLUSTER_MIN_RATIO * cluster_luma:
                dropped += 1
                continue  # white part must read clearly brighter than the lock
            kept += 1
            record["kept"] = True
            pixels.append(frame[mount_mask].astype(np.float64))
            contrast.append(local_contrast(lum)[mount_mask])
            if args.dump_overlays is not None and kept <= 12:
                from PIL import Image

                overlay = frame.copy()
                for mask, color in (
                    (moved, [128, 0, 0]),
                    (mount_mask, [0, 0, 128]),
                ):
                    overlay[mask] = (0.5 * overlay[mask] + color).astype(np.uint8)
                args.dump_overlays.mkdir(parents=True, exist_ok=True)
                Image.fromarray(overlay).save(
                    args.dump_overlays / f"ep{episode:03d}_f{t:04d}.png",
                )
        print(f"file {file_index}: kept {kept} dropped {dropped} (cumulative)")

    total = kept + dropped
    if kept < MIN_FRAMES_KEPT:
        raise SystemExit(
            f"ABORT: only {kept}/{total} frames pass the cluster lock — "
            "projected masks are not landing on the real gripper",
        )
    pooled = np.concatenate(pixels)
    if len(pooled) < MIN_MOUNT_POOL:
        raise SystemExit(
            f"ABORT: mount pool {len(pooled)} px < {MIN_MOUNT_POOL}",
        )
    return {
        "populations": {"mount": pixel_stats(pooled, np.concatenate(contrast))},
        "frames": {"kept": kept, "dropped": dropped, "log": frame_log},
        "geoms": {name: [int(g) for g in ids] for name, ids in sets.items()},
    }


def mount_mat_ids(model) -> list[int]:  # noqa: ANN001
    return [
        model.material(prefix + "wrist_roll_follower_so101_v1_material").id
        for prefix in ("", "leader-")
    ]


def set_mount_grade(model, mats: list[int], grade: dict) -> None:  # noqa: ANN001
    for mat in mats:
        model.mat_rgba[mat, :3] = grade["rgba"]
        model.mat_specular[mat] = grade["specular"]
        model.mat_shininess[mat] = grade["shininess"]


def sample_mount_stats(
    sim,  # noqa: ANN001
    combos: list[tuple[np.ndarray, int]],
    mount_geoms: np.ndarray,
) -> dict:
    """Pooled production-composite mount-pixel stats over the fit combos
    (the same statistics the mine step took from real; sim frames are
    self-registered — no snap, straight erosion)."""
    pixels: list[np.ndarray] = []
    contrast: list[np.ndarray] = []
    for state_deg, look in combos:
        sim.reset(1000 + look, appearance_seed=7000 + look)
        set_real_pose(sim, state_deg)
        obs = sim.observe()
        mask = erode_mask(
            render_masks(sim, {"mount": mount_geoms})["mount"],
            MOUNT_ERODE,
        )
        if mask.any():
            lum = luma(obs.top)
            pixels.append(obs.top[mask].astype(np.float64))
            contrast.append(local_contrast(lum)[mask])
    return pixel_stats(np.concatenate(pixels), np.concatenate(contrast))


def solve_albedo(
    sim,  # noqa: ANN001
    combos,  # noqa: ANN001
    mount_geoms: np.ndarray,
    mats: list[int],
    real: dict,
    spec_shin: tuple[float, float],
) -> np.ndarray:
    """2-point per-channel linear solve (the link-fit shape): composited
    channel median responds ~affinely to albedo through the fixed
    grade/blur chain."""
    responses = []
    for albedo in ALBEDO_PROBES:
        set_mount_grade(
            sim.model,
            mats,
            {
                "rgba": (albedo,) * 3,
                "specular": spec_shin[0],
                "shininess": spec_shin[1],
            },
        )
        responses.append(sample_mount_stats(sim, combos, mount_geoms))
    lo = np.array(responses[0]["channel_median"])
    hi = np.array(responses[1]["channel_median"])
    target = np.array(real["channel_median"])
    slope = (hi - lo) / (ALBEDO_PROBES[1] - ALBEDO_PROBES[0])
    slope = np.where(np.abs(slope) < 1e-3, 1e-3, slope)
    return np.clip(ALBEDO_PROBES[0] + (target - lo) / slope, 0.02, 0.98)


def fit(args: argparse.Namespace) -> dict:
    from sim.so101_sim import SO101Sim

    mined = json.loads(args.mined.read_text())
    real = mined["populations"]["mount"]

    sim = SO101Sim(render_style="v3", post_backend="numpy")
    sim._split_mount_material()  # the byte-identical detach; mats now mount-only
    sim.reset(0)
    # mount_sets' size guard assumes the pre-split model (the detached
    # gripper geom is material-less afterwards) — look up the mount
    # geoms directly here
    model = sim.model
    mount_geoms = np.array(
        sorted(
            g
            for g in range(model.ngeom)
            if model.body(model.geom_bodyid[g]).name == "camera_mount"
            and int(model.geom_matid[g]) >= 0
        ),
    )
    if len(mount_geoms) != 1:
        raise SystemExit(f"expected 1 rendered mount geom, got {mount_geoms}")
    mats = mount_mat_ids(sim.model)
    print("mount geoms", mount_geoms.tolist(), "mats", mats)

    states = load_states(args.v2_root)
    rng = np.random.default_rng(0)
    pose_eps = rng.choice(EPISODES, size=FIT_POSES, replace=False)
    combos = []
    for k, episode in enumerate(pose_eps):
        traj = states[int(episode)]
        t = int(rng.integers(int(len(traj) * 0.15), int(len(traj) * 0.9)))
        combos.extend((traj[t], k * FIT_LOOKS + look) for look in range(FIT_LOOKS))

    baseline = sample_mount_stats(sim, combos, mount_geoms)  # recolor black
    print("baseline (recolor-black mount):", json.dumps(baseline, indent=1))

    albedo = solve_albedo(sim, combos, mount_geoms, mats, real, (0.5, 0.5))
    print("albedo pass 1:", albedo.round(3).tolist())

    grid_results = []
    best: tuple[tuple[float, float] | None, float] = (None, np.inf)
    for spec in SPEC_GRID:
        for shin in SHIN_GRID:
            set_mount_grade(
                sim.model,
                mats,
                {"rgba": tuple(albedo), "specular": spec, "shininess": shin},
            )
            stats = sample_mount_stats(sim, combos, mount_geoms)
            loss = stats_loss(stats, real)
            grid_results.append({"specular": spec, "shininess": shin, "loss": loss})
            if loss < best[1]:
                best = ((spec, shin), loss)
            print(f"spec {spec} shin {shin}: {loss:.1f}")

    chosen = best[0]
    albedo = solve_albedo(sim, combos, mount_geoms, mats, real, chosen)
    final_grade = {
        "rgba": tuple(round(float(v), 4) for v in albedo),
        "specular": chosen[0],
        "shininess": chosen[1],
    }
    set_mount_grade(sim.model, mats, final_grade)
    final_stats = sample_mount_stats(sim, combos, mount_geoms)
    return {
        "fitted": {"mount": final_grade},
        "final_sim_stats": final_stats,
        "final_loss": stats_loss(final_stats, real),
        "baseline_sim_stats": baseline,
        "baseline_loss": stats_loss(baseline, real),
        "real_stats": real,
        "grid": grid_results,
        "material_ids": mats,
        "combos": [{"look": look} for _, look in combos],
    }


def main() -> int:
    args = parse_args()
    payload = mine(args) if args.command == "mine" else fit(args)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    payload["config"] = {
        "command": args.command,
        "v2_root": str(args.v2_root),
        "episodes": list(EPISODES),
        "commit": commit,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
