"""Fit the room's dominant light from the real arm shadows.

`sim-composite-contact-shadows` leg (a) (lit 0823,
fontaine/blog/src/papers/composite-shadows.md): the v3 composite's
pasted arm casts no shadow on the real clean plate. Before wiring a
shadow pass into the composite we measure, from the real episodes
themselves, (1) whether the real arm casts a measurable shadow at all,
(2) the dominant light direction, (3) the shadow's strength and
softness — the three constants the v4 render style needs.

Method. For frames of the bank episodes (per-episode ghost-free clean
plates, assets/real_plates/bank): the frame/plate luminance ratio is a
darkening map D = 1 - ratio in which the arm's real shadow is the
dominant structure once the arm itself, the leader arm + operator hand
halo, the manifest-recorded clutter (+ a shadow halo around each), and
object-replacement artifacts (|ratio - 1| large) are excluded. The
recorded observation.state replayed through the sim (leader mirrors
the follower, the teleop identity) gives the exact arm geometry; a
candidate light direction turns it into a predicted shadow region via
sim.shadow.shadow_map — the same projector the v4 composite uses — so
the fitted optimum transfers to the render style verbatim.

Fit. Grid over (zenith, azimuth): score = mean over frames of
mean(D[predicted shadow]) - mean(D[control ring around it]). Softness
sigma by correlation of the continuous map with D at the winning
direction; strength by least squares D ~ s * map through the origin.
Frame bootstrap for CIs; the axis dies here (no probe spend) if the
contrast CI includes 0.

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/fit_contact_shadow.py \
      --out reports/analysis__contact_shadow_fit.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.shadow import shadow_map
from sim.so101_sim import JOINTS, SO101Sim

ZENITH_COARSE = (0.0, 10.0, 20.0, 30.0, 40.0)
AZIMUTH_COARSE = tuple(range(0, 360, 30))
SIGMA_CANDIDATES = (4.0, 8.0, 16.0, 24.0, 32.0, 48.0)
MAP_SIGMA_FIT = 2.0  # sharp maps while fitting direction
SHADOW_THRESHOLD = 0.15
RATIO_KEEP = (0.55, 1.15)  # outside = object replacement, not lighting
MIN_REGION_PX = 400
N_BOOTSTRAP = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=Path("~/datasets/mcobzarenco/so101_pick_place_v2").expanduser(),
    )
    parser.add_argument(
        "--bank-dir",
        type=Path,
        default=Path("assets/real_plates/bank"),
    )
    parser.add_argument("--frames-per-episode", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dump-diagnostics",
        type=Path,
        default=None,
        help="write per-frame D / predicted-map PNGs for a few frames",
    )
    parser.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    return parser.parse_args()


def light_vector(zenith_deg: float, azimuth_deg: float) -> np.ndarray:
    """Direction light TRAVELS (unit, z < 0). Azimuth 0 = +x world,
    measured toward +y; zenith 0 = straight down."""
    z, a = np.deg2rad(zenith_deg), np.deg2rad(azimuth_deg)
    return np.array([np.sin(z) * np.cos(a), np.sin(z) * np.sin(a), -np.cos(z)])


def episode_frames(
    root: Path,
    episode_row: pd.Series,
    fractions: np.ndarray,
) -> tuple[list[np.ndarray], list[int]]:
    """Decode the frames at the given fractions of one episode's span
    from its front-camera video (single sequential pass)."""
    length = int(episode_row["length"])
    t0 = float(episode_row["videos/observation.images.front/from_timestamp"])
    file_index = int(episode_row["videos/observation.images.front/file_index"])
    wanted = sorted({min(length - 1, int(length * f)) for f in fractions})
    times = [t0 + w / 30.0 for w in wanted]
    path = (
        root
        / "videos/observation.images.front/chunk-000"
        / f"file-{file_index:03d}.mp4"
    )
    container = av.open(str(path))
    stream = container.streams.video[0]
    if stream.time_base is None:
        raise SystemExit(f"{path}: stream has no time base")
    container.seek(int(times[0] / stream.time_base), stream=stream)
    frames: list[np.ndarray] = []
    cursor = 0
    for frame in container.decode(video=0):
        while cursor < len(times) and frame.time >= times[cursor] - 1e-3:
            frames.append(frame.to_ndarray(format="rgb24"))
            cursor += 1
        if cursor == len(times):
            break
    container.close()
    if len(frames) != len(wanted):
        raise SystemExit(f"episode {episode_row['episode_index']}: decode short")
    return frames, wanted


def clutter_exclusion(manifest_episode: dict, shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask of manifest-recorded clutter + a halo generous
    enough to swallow each object's own real shadow."""
    out = np.zeros(shape, dtype=np.uint8)
    for record in manifest_episode["clutter"].values():
        if not record.get("present"):
            continue
        u, v = record["px"]
        radius = int(2.2 * np.sqrt(record["area_px"] / np.pi)) + 12
        cv2.circle(out, (round(u), round(v)), radius, 1, -1)
    return out.astype(bool)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


class FramePack:
    """Everything the direction grid needs about one real frame."""

    __slots__ = ("darkening", "episode", "frame_index", "points_px", "valid")

    def __init__(
        self,
        episode: int,
        frame_index: int,
        darkening: np.ndarray,
        valid: np.ndarray,
        points_px: tuple[np.ndarray, ...],
    ) -> None:
        self.episode = episode
        self.frame_index = frame_index
        self.darkening = darkening  # [H, W] float32, 1 - normalized ratio
        self.valid = valid  # [H, W] bool, table & not excluded
        self.points_px = points_px  # dynamic (v, u) pixel arrays (pinhole)


def main() -> int:
    args = parse_args()
    import mujoco
    from PIL import Image

    sim = SO101Sim(render_style="v3", post_backend="numpy")
    model, data = sim.model, sim.data
    height, width = sim._render_size
    sim.reset(0)  # builds the renderer + fisheye grid, settles a scene

    manifest = json.loads((args.bank_dir / "bank_manifest.json").read_text())
    episodes_meta = pd.read_parquet(args.v2_root / "meta/episodes/chunk-000")
    state_frames = pd.concat(
        pd.read_parquet(
            p,
            columns=["episode_index", "frame_index", "observation.state"],
        )
        for p in sorted((args.v2_root / "data/chunk-000").glob("*.parquet"))
    ).set_index(["episode_index", "frame_index"])

    # Static camera: intrinsics + pose once.
    cam_id = model.camera("top_cam").id
    f = (height / 2.0) / np.tan(np.deg2rad(float(model.cam_fovy[cam_id])) / 2.0)
    intrinsics = (f, (width - 1) / 2.0, (height - 1) / 2.0)
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
    table = model.geom("table")
    bounds = (
        float(table.pos[0] - table.size[0]),
        float(table.pos[0] + table.size[0]),
        float(table.pos[1] - table.size[1]),
        float(table.pos[1] + table.size[1]),
    )
    table_id = table.id

    # Geom id sets: follower vs leader arm (bodies prefixed "leader-").
    follower_geoms, leader_geoms = [], []
    for gid in range(model.ngeom):
        body = model.body(model.geom_bodyid[gid])
        if body.id == 0 or body.id == sim._benchy_body:
            continue
        (leader_geoms if body.name.startswith("leader-") else follower_geoms).append(
            gid,
        )
    follower_geoms = np.array(follower_geoms)
    leader_geoms = np.array(leader_geoms)
    leader_qpos = np.array(
        [model.joint(f"leader-{name}").qposadr[0] for name in JOINTS],
    )

    # Park the benchy far outside both frusta (its real pose is unknown)
    # and the clutter stand-ins too: reset(0) drew v3 poses that are not
    # this episode's real ones — the REAL clutter is excluded by the
    # manifest px-halo masks instead.
    adr = sim._benchy_qpos
    data.qpos[adr : adr + 3] = (2.4, 1.5, 0.001)
    data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
    for name in SO101Sim.V2_DYNAMIC_STATICS:
        model.geom_pos[model.geom(name).id] = SO101Sim.V3_ABSENT_POS

    def remap_bool(mask: np.ndarray, threshold: float = 0.25) -> np.ndarray:
        return sim._remap(mask.astype(np.float64)[..., None])[..., 0] > threshold

    packs: list[FramePack] = []
    fractions = np.linspace(0.08, 0.92, args.frames_per_episode)
    for key in sorted(manifest["episodes"], key=int):
        episode = int(key)
        plate_path = args.bank_dir / f"top_ep{episode:03d}.png"
        if not plate_path.exists():
            continue
        plate = np.asarray(Image.open(plate_path), dtype=np.float32)
        row = episodes_meta[episodes_meta.episode_index == episode].iloc[0]
        frames, indices = episode_frames(args.v2_root, row, fractions)
        clutter = clutter_exclusion(manifest["episodes"][key], (height, width))
        for frame, frame_index in zip(frames, indices, strict=True):
            state = np.asarray(
                state_frames.loc[(episode, frame_index), "observation.state"],
            )
            data.qpos[sim._joint_qpos] = np.deg2rad(state)
            data.qpos[leader_qpos] = np.deg2rad(state)
            mujoco.mj_forward(model, data)

            renderer = sim.renderer
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera="top_cam")
            seg = renderer.render()
            renderer.disable_segmentation_rendering()

            is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
            follower = is_geom & np.isin(seg[..., 0], follower_geoms)
            leader = is_geom & np.isin(seg[..., 0], leader_geoms)
            table_px = is_geom & (seg[..., 0] == table_id)

            ratio = (frame.astype(np.float32).mean(axis=2) + 1.0) / (
                plate.mean(axis=2) + 1.0
            )
            valid = (
                remap_bool(table_px)
                & ~dilate(remap_bool(follower), 12)
                & ~dilate(remap_bool(leader), 80)
                & ~clutter
                & (ratio > RATIO_KEEP[0])
                & (ratio < RATIO_KEEP[1])
            )
            if valid.sum() < 20 * MIN_REGION_PX:
                continue
            # Per-frame photometric normalization: the person moving
            # around the room drifts the global level within episodes.
            ratio = ratio / np.median(ratio[valid])
            packs.append(
                FramePack(
                    episode,
                    frame_index,
                    (1.0 - ratio).astype(np.float32),
                    valid,
                    np.nonzero(follower | leader),
                ),
            )
        print(f"episode {episode}: {len(packs)} frames packed", flush=True)

    if not packs:
        raise SystemExit("no usable frames")
    depth_cache: dict[tuple[int, int], np.ndarray] = {}

    def predicted_map(
        pack: FramePack,
        direction: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Remapped (fisheye-space) continuous shadow map for a pack."""
        key = (pack.episode, pack.frame_index)
        if key not in depth_cache:
            raise KeyError
        depth = depth_cache[key]
        dyn = np.zeros(depth.shape)
        dyn[pack.points_px] = 1.0
        pin = shadow_map(
            depth,
            dyn,
            intrinsics,
            cam_pos,
            cam_mat,
            direction,
            sigma,
            bounds_xy=bounds,
            max_points=20000,
        )
        return sim._remap(pin[..., None])[..., 0]

    # Second pass: cache depth buffers (replaying states again costs a
    # render but keeps peak memory to one float depth per frame).
    for pack in packs:
        state = np.asarray(
            state_frames.loc[(pack.episode, pack.frame_index), "observation.state"],
        )
        data.qpos[sim._joint_qpos] = np.deg2rad(state)
        data.qpos[leader_qpos] = np.deg2rad(state)
        mujoco.mj_forward(model, data)
        renderer = sim.renderer
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="top_cam")
        depth_cache[(pack.episode, pack.frame_index)] = renderer.render().astype(
            np.float32,
        )
        renderer.disable_depth_rendering()

    directions: list[tuple[float, float]] = [
        (z, float(a)) for z in ZENITH_COARSE for a in (AZIMUTH_COARSE if z else (0,))
    ]

    def contrast(pack: FramePack, zenith: float, azimuth: float) -> float:
        mapped = predicted_map(pack, light_vector(zenith, azimuth), MAP_SIGMA_FIT)
        shadow = (mapped > SHADOW_THRESHOLD) & pack.valid
        ring = dilate(shadow, 90) & ~dilate(shadow, 25) & pack.valid
        if shadow.sum() < MIN_REGION_PX or ring.sum() < MIN_REGION_PX:
            return np.nan
        return float(
            pack.darkening[shadow].mean() - pack.darkening[ring].mean(),
        )

    matrix = np.full((len(packs), len(directions)), np.nan)
    for j, (zenith, azimuth) in enumerate(directions):
        for i, pack in enumerate(packs):
            matrix[i, j] = contrast(pack, zenith, azimuth)
        print(
            f"dir zen {zenith:>4.1f} az {azimuth:>5.1f}: "
            f"contrast {np.nanmean(matrix[:, j]):+.4f}",
            flush=True,
        )

    scores = np.nanmean(matrix, axis=0)
    best = int(np.nanargmax(scores))
    best_zenith, best_azimuth = directions[best]

    # Refine azimuth at the winning zenith (skip for the vertical pole).
    if best_zenith > 0:
        refine = [
            (best_zenith + dz, (best_azimuth + da) % 360)
            for dz in (-5.0, 0.0, 5.0)
            for da in (-15.0, -7.5, 0.0, 7.5, 15.0)
            if (dz, da) != (0.0, 0.0) and 0 < best_zenith + dz <= 45
        ]
        for zenith, azimuth in refine:
            column = np.array([contrast(p, zenith, azimuth) for p in packs])
            directions.append((zenith, azimuth))
            matrix = np.column_stack([matrix, column])
        scores = np.nanmean(matrix, axis=0)
        best = int(np.nanargmax(scores))
        best_zenith, best_azimuth = directions[best]
    best_direction = light_vector(best_zenith, best_azimuth)
    print(
        f"best: zenith {best_zenith} azimuth {best_azimuth} "
        f"contrast {scores[best]:+.4f}",
    )

    # Softness + strength at the winning direction.
    sums: dict[float, list[tuple[float, float, float]]] = {
        s: [] for s in SIGMA_CANDIDATES
    }
    for pack in packs:
        for sigma in SIGMA_CANDIDATES:
            mapped = predicted_map(pack, best_direction, sigma)
            m, d = mapped[pack.valid], pack.darkening[pack.valid].astype(np.float64)
            if m.std() < 1e-6:
                continue
            corr = float(np.corrcoef(m, d)[0, 1])
            sums[sigma].append((corr, float((m * d).sum()), float((m * m).sum())))
    sigma_corr = {s: float(np.mean([c for c, _, _ in v])) for s, v in sums.items() if v}
    best_sigma = max(sigma_corr, key=lambda s: sigma_corr[s])
    md = np.array([[md_ for _, md_, _ in sums[best_sigma]]]).ravel()
    mm = np.array([[mm_ for _, _, mm_ in sums[best_sigma]]]).ravel()
    strength = float(md.sum() / mm.sum())

    rng = np.random.default_rng(args.seed)
    boot_contrast, boot_strength, boot_dir = [], [], []
    per_frame_best = matrix[:, best]
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(len(packs), size=len(packs))
        idx_s = rng.integers(len(md), size=len(md))
        boot_contrast.append(np.nanmean(per_frame_best[idx]))
        boot_strength.append(md[idx_s].sum() / mm[idx_s].sum())
        boot_dir.append(int(np.nanargmax(np.nanmean(matrix[idx], axis=0))))

    def ci(values: list[float]) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    dir_counts = {
        f"zen{directions[k][0]:g}_az{directions[k][1]:g}": int(n)
        for k, n in zip(*np.unique(boot_dir, return_counts=True), strict=True)
    }

    if args.dump_diagnostics is not None:
        args.dump_diagnostics.mkdir(parents=True, exist_ok=True)
        for pack in packs[:: max(1, len(packs) // 6)]:
            mapped = predicted_map(pack, best_direction, best_sigma)
            dark = np.clip(pack.darkening * 4.0, -1, 1)
            panel = np.concatenate(
                [
                    ((dark * 0.5 + 0.5) * 255).astype(np.uint8),
                    (np.clip(mapped, 0, 1) * 255).astype(np.uint8),
                    (pack.valid * 255).astype(np.uint8),
                ],
                axis=1,
            )
            Image.fromarray(panel).save(
                args.dump_diagnostics
                / f"ep{pack.episode:03d}_f{pack.frame_index:05d}.png",
            )

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    payload = {
        "config": {
            "commit": commit,
            "frames": len(packs),
            "episodes": len({p.episode for p in packs}),
            "frames_per_episode": args.frames_per_episode,
            "ratio_keep": RATIO_KEEP,
            "map_sigma_fit": MAP_SIGMA_FIT,
            "shadow_threshold": SHADOW_THRESHOLD,
            "grid": {"zenith": ZENITH_COARSE, "azimuth_step": 30, "refine": "az 7.5"},
            "exclusions": "follower+12px, leader+80px (operator hand), "
            "manifest clutter x2.2 halo, ratio outside [0.55, 1.15]",
        },
        "direction": {
            "zenith_deg": best_zenith,
            "azimuth_deg": best_azimuth,
            "light_dir_world": [float(x) for x in best_direction],
            "bootstrap_direction_counts": dir_counts,
        },
        "contrast": {
            "mean": float(scores[best]),
            "ci95": ci(boot_contrast),
            "per_direction_coarse": {
                f"zen{z:g}_az{a:g}": float(s)
                for (z, a), s in zip(directions, scores, strict=True)
            },
        },
        "softness": {
            "sigma_px": best_sigma,
            "correlation_by_sigma": sigma_corr,
        },
        "strength": {"fit": strength, "ci95": ci(boot_strength)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")
    print(
        f"SUMMARY: contrast {scores[best]:+.4f} CI {ci(boot_contrast)} | "
        f"zen {best_zenith} az {best_azimuth} | sigma {best_sigma} | "
        f"strength {strength:.3f} CI {ci(boot_strength)}",
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
