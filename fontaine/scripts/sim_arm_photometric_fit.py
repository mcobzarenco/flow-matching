"""Arm link photometrics — mine real link-pixel statistics and fit the
material grade (queue `sim-arm-photometric-links`; arm-split leg named
LINKS at 88% of the arm's keep-only delta, mounts the record-only rider).

Two subcommands:

  mine  Project the sim's own segmentation masks onto REAL v2 top frames
        at their recorded joint poses (observation.state degrees ->
        follower qpos, mj_forward, seg render, production fisheye remap)
        and pool the real pixels under the eroded masks. Two populations,
        split per-geom by material: the printed-PLA link surfaces and the
        STS3215 servo casings. Follower instance only (the real leader's
        pose is operator-held and unrecorded); reference-half episodes
        only (the bank's 0-25). Per-frame registration guard: the masked
        real pixels must read darker than the surrounding ring (the arm
        is black hardware on a light table) else the frame is dropped.

  fit   Fit one material grade per population — albedo RGB, specular,
        shininess — by rendering the PRODUCTION v3 composite (numpy post
        backend) at real-registered poses under drawn appearances and
        matching the composited link-pixel statistics to the mined real
        ones (luma percentiles p25-p99 + channel medians; highlight tail
        weighted). Albedo by 2-point linear solve per channel, specular x
        shininess by grid, albedo re-solved at the winning cell.

Output JSON feeds the frozen constants in sim/so101_sim.py
(ARM_PHOTOMETRICS_V1) — the opt-in `arm_photometrics="v1"` render path
gated by the pinned 20x5 probe (sim_arm_photometric_read.py).

Usage:
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_arm_photometric_fit.py \
      mine --out reports/analysis__arm_photometric_mine.json
  MUJOCO_GL=egl uv run python fontaine/scripts/sim_arm_photometric_fit.py \
      fit --mined reports/analysis__arm_photometric_mine.json \
      --out reports/analysis__arm_photometric_fit.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
from sim_top_gap_decomposition import dilate, erode

LINK_BODIES = ("base", "shoulder", "upper_arm", "lower_arm", "wrist")
MOUNT_BODIES = ("camera_mount",)
EPISODES = tuple(range(26))  # bank reference half
FRAMES_PER_EPISODE = 6
FRAME_SPAN = (0.12, 0.92)  # strided inside the episode, away from the ends
ERODE_STEPS = {"pla": 4, "servo": 3, "mount": 2}
# Mining bodies: wrist excluded — its neighborhood (PCB, cables, the
# orange-printed gripper cluster) offers dark distractors the snap can
# lock onto; the link material is identical along the chain, so the
# base/shoulder/upper/lower pools characterize it fully. The mount is
# excluded from pixel pools too (white in reality — darkness machinery
# does not apply; the rider read is the overlay + the color constant).
MINE_BODIES = ("base", "shoulder", "upper_arm", "lower_arm")
RING_LUMA_MAX_RATIO = 0.8  # masked arm must read darker than its ring
MASK_LUMA_MAX = 100.0  # ... and absolutely dark (black hardware)
SNAP_KEEP_FRACTION = 0.8  # sliding the mask off-frame is not a lock
# Pose registration at arm height is off by tens of px in image space
# (camera fit is table-plane, servo calibration offsets) and the error
# GROWS along the chain — one rigid shift cannot serve base and wrist at
# once. Snap PER BODY: each link body's mask slides to its own darkness
# minimum (black hardware against the light table) before its ring guard
# judges it. The mount is white in reality, so darkness-snap would be
# wrong for it — it rides the wrist body's shift, record-only.
SNAP_RANGE = 60
MIN_BODY_PIXELS = 80
# One confidently locked body (unsaturated snap, in-frame, dark in both
# the ring-relative and absolute senses) is a valid partial harvest —
# quality control is per-body; the abort bar is the harvested pixel
# mass, not the frame count.
MIN_BODIES_KEPT = 1
MIN_FRAMES_KEPT = 60
MIN_POOL = {"pla": 200_000, "servo": 40_000}
LUMA_W = np.array([0.2126, 0.7152, 0.0722])
PCTS = (10, 25, 50, 75, 90, 97, 99)

# fit sampling: poses x appearance draws per candidate
FIT_POSES = 8
FIT_LOOKS = 2
SPEC_GRID = (0.0, 0.3, 0.6, 1.0)
SHIN_GRID = (0.1, 0.4, 0.7, 1.0)
ALBEDO_PROBES = (0.10, 0.45)


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


def load_states(root: Path) -> dict[int, np.ndarray]:
    """episode -> [T, 6] float64 degrees (observation.state)."""
    frames: dict[int, list] = {}
    for file in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(
            file,
            columns=["episode_index", "frame_index", "observation.state"],
        )
        for episode, part in df.groupby("episode_index"):
            frames.setdefault(int(episode), []).append(part)
    return {
        episode: np.stack(
            pd.concat(parts).sort_values("frame_index")["observation.state"].to_list(),
        ).astype(np.float64)
        for episode, parts in frames.items()
    }


def decode_frames(path: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    """Absolute frame indices -> decoded frames, one sequential pass."""
    want = set(wanted)
    out: dict[int, np.ndarray] = {}
    container = av.open(str(path))
    for index, frame in enumerate(container.decode(video=0)):
        if index in want:
            out[index] = frame.to_ndarray(format="rgb24")
            if len(out) == len(want):
                break
    container.close()
    if len(out) != len(want):
        raise SystemExit(f"{path.name}: decoded {len(out)}/{len(want)} frames")
    return out


def episode_video_offsets(root: Path) -> dict[int, tuple[int, int]]:
    """episode -> (video file index, absolute frame offset) for the top
    camera; LeRobot v3 concatenates episodes into shared mp4 files, the
    per-episode start rides meta/episodes from_timestamp at the fps."""
    meta = pd.concat(
        pd.read_parquet(f)
        for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))
    )
    fps = 30
    key = "videos/observation.images.front"
    return {
        int(row["episode_index"]): (
            int(row[f"{key}/file_index"]),
            round(float(row[f"{key}/from_timestamp"]) * fps),
        )
        for _, row in meta.iterrows()
    }


def population_geoms(sim) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Follower links-class geoms split per-material — printed PLA vs
    STS3215 servo casings — plus the camera mount as a RECORD-ONLY
    population (the rider read; its material is shared with the gripper's
    wrist-roll piece, so a fix there is out of this leg's scope).
    Material-less geoms are the invisible collision set (group 3) —
    never rendered, never mined."""
    model = sim.model
    pops: dict[str, list[int]] = {"pla": [], "servo": [], "mount": []}
    for geom in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[geom]).name
        if body_name.startswith("leader-"):
            continue
        matid = int(model.geom_matid[geom])
        if matid < 0:
            continue
        if body_name in MOUNT_BODIES:
            pops["mount"].append(geom)
        elif body_name in LINK_BODIES:
            mat_name = model.mat(matid).name
            pops["servo" if "sts3215" in mat_name else "pla"].append(geom)
    out = {name: np.array(sorted(ids)) for name, ids in pops.items()}
    if len(out["pla"]) < 6 or len(out["servo"]) < 3 or len(out["mount"]) < 1:
        raise SystemExit(
            f"population sizes look wrong: { {k: len(v) for k, v in out.items()} }",
        )
    return out


def set_real_pose(sim, state_deg: np.ndarray) -> None:  # noqa: ANN001
    """Kinematic pose from a recorded real frame (sysid degree
    convention); forward only — no stepping, physics untouched."""
    import mujoco

    sim.data.qpos[sim._joint_qpos] = np.deg2rad(state_deg)
    mujoco.mj_forward(sim.model, sim.data)


def render_masks(
    sim,  # noqa: ANN001
    sets: dict,
) -> dict:
    """Raw 0/1 masks in output pixel space for arbitrary geom-id sets:
    one seg render at the current pose, production fisheye remap per set,
    threshold."""
    import mujoco

    renderer = sim.renderer
    renderer.enable_segmentation_rendering()
    renderer.update_scene(sim.data, camera="top_cam")
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value
    return {
        key: sim._remap(
            (is_geom & np.isin(seg[..., 0], ids)).astype(np.float64)[..., None],
        )[..., 0]
        > 0.5
        for key, ids in sets.items()
    }


def body_populations(sim) -> dict[tuple[str, str], np.ndarray]:  # noqa: ANN001
    """(body, population) -> follower geom ids: each link body's PLA and
    servo geoms separately (per-body snap), plus the mount."""
    model = sim.model
    sets: dict[tuple[str, str], list[int]] = {}
    for geom in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[geom]).name
        if body_name.startswith("leader-"):
            continue
        matid = int(model.geom_matid[geom])
        if matid < 0:
            continue
        if body_name in MOUNT_BODIES:
            sets.setdefault((body_name, "mount"), []).append(geom)
        elif body_name in LINK_BODIES:
            pop = "servo" if "sts3215" in model.mat(matid).name else "pla"
            sets.setdefault((body_name, pop), []).append(geom)
    return {key: np.array(sorted(ids)) for key, ids in sets.items()}


def clipped_shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Zero-fill 2D shift — pixels pushed past the frame edge vanish
    (np.roll would wrap the arm's edge pixels to the opposite side)."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys = slice(max(dy, 0), min(h + dy, h))
    xs = slice(max(dx, 0), min(w + dx, w))
    ys_src = slice(max(-dy, 0), min(h - dy, h))
    xs_src = slice(max(-dx, 0), min(w - dx, w))
    out[ys, xs] = mask[ys_src, xs_src]
    return out


def snap_shift(lum: np.ndarray, union: np.ndarray) -> tuple[int, int] | None:
    """Exact argmin over ALL integer shifts in +-SNAP_RANGE of the mean
    luma under the shifted mask, via FFT cross-correlation (zero-padded,
    so nothing wraps) — one body's mask slides to its darkness minimum."""
    height, width = lum.shape
    pad = SNAP_RANGE
    image = np.zeros((height + 2 * pad, width + 2 * pad))
    valid = np.zeros_like(image)
    mask = np.zeros_like(image)
    image[pad : pad + height, pad : pad + width] = lum
    valid[pad : pad + height, pad : pad + width] = 1.0
    mask[pad : pad + height, pad : pad + width] = union
    mask_f = np.conj(np.fft.fft2(mask))
    num = np.fft.ifft2(np.fft.fft2(image) * mask_f).real
    den = np.fft.ifft2(np.fft.fft2(valid) * mask_f).real
    keep_floor = max(MIN_BODY_PIXELS, SNAP_KEEP_FRACTION * union.sum())
    best: tuple[float, int, int] | None = None
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            count = den[dy % den.shape[0], dx % den.shape[1]]
            if count < keep_floor:
                continue
            score = num[dy % num.shape[0], dx % num.shape[1]] / count
            if best is None or score < best[0]:
                best = (score, dy, dx)
    if best is None:
        return None
    return best[1], best[2]


def erode_mask(mask: np.ndarray, steps: int) -> np.ndarray:
    eroded = mask.astype(np.float64)
    for _ in range(steps):
        eroded = erode(eroded)
    return eroded > 0.5


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float64) @ LUMA_W


def local_contrast(lum: np.ndarray, radius: int = 2) -> np.ndarray:
    """Per-pixel std of the (2r+1)^2 neighborhood, via shifted sums."""
    total = np.zeros_like(lum)
    total2 = np.zeros_like(lum)
    count = (2 * radius + 1) ** 2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(lum, dy, axis=0), dx, axis=1)
            total += shifted
            total2 += shifted**2
    mean = total / count
    return np.sqrt(np.maximum(total2 / count - mean**2, 0.0))


def pixel_stats(rgb_pixels: np.ndarray, contrast_pixels: np.ndarray) -> dict:
    lum = rgb_pixels @ LUMA_W
    return {
        "n_pixels": len(lum),
        "channel_median": [float(np.median(rgb_pixels[:, c])) for c in range(3)],
        "luma_percentiles": {str(p): float(np.percentile(lum, p)) for p in PCTS},
        "highlight_fraction": float((lum > 1.5 * np.median(lum)).mean()),
        "local_contrast_median": float(np.median(contrast_pixels)),
    }


def mine(args: argparse.Namespace) -> dict:
    from sim.so101_sim import SO101Sim

    sim = SO101Sim(render_style="v3", post_backend="numpy")
    sim.reset(0)
    pops = population_geoms(sim)
    body_sets = body_populations(sim)
    print({name: len(ids) for name, ids in pops.items()})

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

    pixels: dict[str, list[np.ndarray]] = {name: [] for name in pops}
    contrast: dict[str, list[np.ndarray]] = {name: [] for name in pops}
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
            raw = render_masks(sim, body_sets)
            lum = luma(frame)
            contrast_map = local_contrast(lum)
            body_shift: dict[str, tuple[int, int]] = {}
            body_ratio: dict[str, float] = {}
            frame_masks: dict[str, np.ndarray] = {
                name: np.zeros(lum.shape, dtype=bool) for name in pops
            }
            for body in MINE_BODIES:
                union_raw = np.zeros(lum.shape, dtype=bool)
                for pop in ("pla", "servo"):
                    if (body, pop) in raw:
                        union_raw |= raw[(body, pop)]
                if union_raw.sum() < MIN_BODY_PIXELS:
                    continue
                shift = snap_shift(lum, union_raw)
                if shift is None or max(abs(s) for s in shift) >= SNAP_RANGE - 2:
                    continue  # saturated search = no confident lock
                moved_raw = clipped_shift(union_raw, *shift)
                grown = moved_raw.astype(np.float64)
                for _ in range(8):
                    grown = dilate(grown)
                near = dilate(dilate(dilate(moved_raw.astype(np.float64)))) > 0.5
                ring = (grown > 0.5) & ~near
                if not ring.any() or not moved_raw.any():
                    continue
                mask_luma = float(np.median(lum[moved_raw]))
                ratio = mask_luma / max(np.median(lum[ring]), 1e-6)
                body_ratio[body] = round(ratio, 3)
                if ratio >= RING_LUMA_MAX_RATIO or mask_luma >= MASK_LUMA_MAX:
                    continue
                body_shift[body] = shift
                for pop in ("pla", "servo"):
                    if (body, pop) in raw:
                        frame_masks[pop] |= erode_mask(
                            clipped_shift(raw[(body, pop)], *shift),
                            ERODE_STEPS[pop],
                        )
            ok = len(body_shift) >= MIN_BODIES_KEPT
            frame_log.append(
                {
                    "episode": episode,
                    "frame": int(t),
                    "shifts": {b: list(s) for b, s in body_shift.items()},
                    "ratios": body_ratio,
                    "kept": ok,
                },
            )
            if not ok:
                dropped += 1
                continue
            kept += 1
            for name in pops:
                mask = frame_masks[name]
                if mask.any():
                    pixels[name].append(frame[mask].astype(np.float64))
                    contrast[name].append(contrast_map[mask])
            if args.dump_overlays is not None and kept <= 12:
                from PIL import Image

                overlay = frame.copy()
                tint = {"pla": [0, 128, 0], "servo": [128, 0, 0], "mount": [0, 0, 128]}
                for name, color in tint.items():
                    mask = frame_masks[name]
                    overlay[mask] = (0.5 * overlay[mask] + color).astype(np.uint8)
                args.dump_overlays.mkdir(parents=True, exist_ok=True)
                Image.fromarray(overlay).save(
                    args.dump_overlays / f"ep{episode:03d}_f{t:04d}.png",
                )
        print(f"file {file_index}: kept {kept} dropped {dropped} (cumulative)")

    total = kept + dropped
    if kept < MIN_FRAMES_KEPT:
        raise SystemExit(
            f"ABORT: only {kept}/{total} frames pass the registration guard — "
            "projected masks are not landing on the real arm",
        )
    stats = {}
    for name, chunks in pixels.items():
        pooled = np.concatenate(chunks) if chunks else np.empty((0, 3))
        if name == "mount":  # record-only rider population
            if len(pooled) < 300:
                stats[name] = {"n_pixels": len(pooled), "insufficient": True}
                continue
        elif len(pooled) < MIN_POOL[name]:
            raise SystemExit(
                f"ABORT: population {name} pooled {len(pooled)} px < {MIN_POOL[name]}",
            )
        stats[name] = pixel_stats(pooled, np.concatenate(contrast[name]))
    return {
        "populations": stats,
        "frames": {"kept": kept, "dropped": dropped, "log": frame_log},
        "geoms": {name: [int(g) for g in ids] for name, ids in pops.items()},
    }


def set_grade(model, mat_ids: dict[str, list[int]], grades: dict) -> None:  # noqa: ANN001
    for name, ids in mat_ids.items():
        grade = grades[name]
        for mat in ids:
            model.mat_rgba[mat, :3] = grade["rgba"]
            model.mat_specular[mat] = grade["specular"]
            model.mat_shininess[mat] = grade["shininess"]


def arm_material_ids(model) -> dict[str, list[int]]:  # noqa: ANN001
    """The patch surface (both instances): the link bodies' printed-PLA
    materials plus the STS3215 servo materials (the servo material is
    shared with gripper-body servo geoms — the same physical casing in
    reality). Excluded: the moving jaws (gripper class; the follower's
    is orange) and wrist_roll_follower (shared between the gripper's
    wrist-roll piece and the camera mount — out of this leg's scope)."""
    ids: dict[str, list[int]] = {"pla": [], "servo": []}
    for mat in range(model.nmat):
        name = model.mat(mat).name
        if "sts3215" in name:
            ids["servo"].append(mat)
        elif (
            "so101" in name
            and "moving_jaw" not in name
            and "wrist_roll_follower" not in name
        ):
            ids["pla"].append(mat)
    return ids


def sample_sim_stats(
    sim,  # noqa: ANN001
    combos: list[tuple[np.ndarray, int]],
    pops: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Composite the CURRENT materials over the fit combos; pooled
    per-population stats of the production top output under the eroded
    remapped masks (the same statistics the mine step took from real)."""
    fit_pops = {name: ids for name, ids in pops.items() if name != "mount"}
    pixels: dict[str, list[np.ndarray]] = {name: [] for name in fit_pops}
    contrast: dict[str, list[np.ndarray]] = {name: [] for name in fit_pops}
    for state_deg, look in combos:
        sim.reset(1000 + look, appearance_seed=7000 + look)
        set_real_pose(sim, state_deg)
        obs = sim.observe()
        # sim frames are self-registered: no snap, straight erosion
        masks = {
            name: erode_mask(mask, ERODE_STEPS[name])
            for name, mask in render_masks(sim, fit_pops).items()
        }
        lum = luma(obs.top)
        contrast_map = local_contrast(lum)
        for name in fit_pops:
            mask = masks[name]
            if mask.any():
                pixels[name].append(obs.top[mask].astype(np.float64))
                contrast[name].append(contrast_map[mask])
    return {
        name: pixel_stats(np.concatenate(pixels[name]), np.concatenate(contrast[name]))
        for name in pixels
    }


def stats_loss(sim_stats: dict, real_stats: dict) -> float:
    """Weighted match on luma percentiles + channel medians; the
    highlight tail (p90+) doubled — specular structure is the point."""
    loss = 0.0
    for p in PCTS:
        weight = 2.0 if p >= 90 else 1.0
        diff = (
            sim_stats["luma_percentiles"][str(p)]
            - real_stats["luma_percentiles"][str(p)]
        )
        loss += weight * diff**2
    for c in range(3):
        loss += (sim_stats["channel_median"][c] - real_stats["channel_median"][c]) ** 2
    return float(loss)


def solve_albedo(
    sim,  # noqa: ANN001
    combos,  # noqa: ANN001
    pops,  # noqa: ANN001
    mat_ids,  # noqa: ANN001
    real: dict,
    spec_shin: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """2-point per-channel linear solve: composited channel median
    responds ~affinely to albedo through the fixed grade/blur chain."""
    responses = []
    for albedo in ALBEDO_PROBES:
        grades = {
            name: {
                "rgba": (albedo,) * 3,
                "specular": spec_shin[name][0],
                "shininess": spec_shin[name][1],
            }
            for name in mat_ids
        }
        set_grade(sim.model, mat_ids, grades)
        responses.append(sample_sim_stats(sim, combos, pops))
    solved = {}
    for name in mat_ids:
        lo = np.array(responses[0][name]["channel_median"])
        hi = np.array(responses[1][name]["channel_median"])
        target = np.array(real[name]["channel_median"])
        slope = (hi - lo) / (ALBEDO_PROBES[1] - ALBEDO_PROBES[0])
        slope = np.where(np.abs(slope) < 1e-3, 1e-3, slope)
        solved[name] = np.clip(
            ALBEDO_PROBES[0] + (target - lo) / slope,
            0.02,
            0.95,
        )
    return solved


def fit(args: argparse.Namespace) -> dict:
    from sim.so101_sim import SO101Sim

    mined = json.loads(args.mined.read_text())
    real = mined["populations"]

    sim = SO101Sim(render_style="v3", post_backend="numpy")
    sim.reset(0)
    pops = population_geoms(sim)
    mat_ids = arm_material_ids(sim.model)
    print({name: len(ids) for name, ids in mat_ids.items()})

    states = load_states(args.v2_root)
    rng = np.random.default_rng(0)
    pose_eps = rng.choice(EPISODES, size=FIT_POSES, replace=False)
    combos = []
    for k, episode in enumerate(pose_eps):
        traj = states[int(episode)]
        t = int(rng.integers(int(len(traj) * 0.15), int(len(traj) * 0.9)))
        combos.extend((traj[t], k * FIT_LOOKS + look) for look in range(FIT_LOOKS))

    baseline = sample_sim_stats(sim, combos, pops)  # untouched materials
    print("baseline (current flat materials):", json.dumps(baseline, indent=1))

    start = dict.fromkeys(mat_ids, (0.5, 0.5))
    albedo = solve_albedo(sim, combos, pops, mat_ids, real, start)
    print("albedo pass 1:", {k: v.round(3).tolist() for k, v in albedo.items()})

    grid_results = []
    best = dict.fromkeys(mat_ids, (None, np.inf))
    for spec in SPEC_GRID:
        for shin in SHIN_GRID:
            grades = {
                name: {"rgba": tuple(albedo[name]), "specular": spec, "shininess": shin}
                for name in mat_ids
            }
            set_grade(sim.model, mat_ids, grades)
            stats = sample_sim_stats(sim, combos, pops)
            cell = {"specular": spec, "shininess": shin, "loss": {}}
            for name in mat_ids:
                loss = stats_loss(stats[name], real[name])
                cell["loss"][name] = loss
                if loss < best[name][1]:
                    best[name] = ((spec, shin), loss)
            grid_results.append(cell)
            print(
                f"spec {spec} shin {shin}: { {n: round(c, 1) for n, c in cell['loss'].items()} }",
            )

    chosen = {name: best[name][0] for name in mat_ids}
    albedo = solve_albedo(sim, combos, pops, mat_ids, real, chosen)
    final_grades = {
        name: {
            "rgba": tuple(round(float(v), 4) for v in albedo[name]),
            "specular": chosen[name][0],
            "shininess": chosen[name][1],
        }
        for name in mat_ids
    }
    set_grade(sim.model, mat_ids, final_grades)
    final_stats = sample_sim_stats(sim, combos, pops)
    return {
        "fitted": final_grades,
        "final_sim_stats": final_stats,
        "final_loss": {
            name: stats_loss(final_stats[name], real[name]) for name in mat_ids
        },
        "baseline_sim_stats": baseline,
        "baseline_loss": {
            name: stats_loss(baseline[name], real[name]) for name in mat_ids
        },
        "real_stats": real,
        "grid": grid_results,
        "material_ids": mat_ids,
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
