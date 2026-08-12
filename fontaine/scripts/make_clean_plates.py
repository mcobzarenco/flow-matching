"""Mine per-camera clean plates from real_v2 for the v2 inpainting
render style (prereg posts/2026-08-12-prereg-sim-visual-inpainting.md).

Per camera: per-pixel median over frames drawn ONLY from episodes
lying wholly inside the encoder probe's reference half A — the probe
holds out B = strided timeline indices >= 150 * (total // 300), so
every episode whose frames all precede that boundary is pixel-disjoint
from the held-out set by construction.

  - top ("front" video key): every --top-stride-th frame of each A
    episode — the arm, boat and operator move within/across episodes
    and median away; the static disk (and clutter) stays.
  - wrist: the first --wrist-window frames of each A episode — the
    wrist camera moves with the arm, so only the episode-start rest
    pose (the settled-reset viewpoint the probe renders) admits a
    static plate. The boat varies across episodes and medians away;
    the parked jaws are shared by every start window and stay in the
    plate (the rendered jaws overlay them at reset).

Outputs (assets/real_plates/): {top,wrist}_plate.png, a per-pixel
coverage sidecar {top,wrist}_coverage.png (fraction of source frames
within --coverage-delta mean-abs of the median; dark = the median is
a minority view there), and manifest.json (episodes, counts, args).

Usage:
  uv run python fontaine/scripts/make_clean_plates.py \
      --out assets/real_plates

--bank mode (sim-content-diversity pre-reg, 2026-08-12): one TOP
plate per A-half episode instead of one global plate. The naive
per-episode median bakes in every parked object (the boat resting on
the disk, both arm rest poses, the operator's hand), so the mining
pass keeps only per-pixel samples that agree with the per-episode
gain/bias-corrected GLOBAL plate (inliers = lighting-shifted
background); pixels with too few inliers fall back to the corrected
global plate, feathered. The same pass measures the between-episode
clutter spread: static-novelty blobs (naive median far from the
corrected global plate) matched to each contype-0 stand-in's
canonical position, unprojected through the sim's own camera model
(equidistant fisheye, verified against a segmentation render by
--selfcheck) to table-plane world xy. Outputs
assets/real_plates/bank/: top_epNNN.png x26 + bank_manifest.json
(gains, fallback fractions, per-object per-episode world xy, draw
ranges = empirical min/max, presence frequencies).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import av
import numpy as np
import pandas as pd

CAMERA_KEYS = {"top": "observation.images.front", "wrist": "observation.images.wrist"}
# The probe's held-out boundary: B = strided frames 150.. of stride
# total // 300 (fontaine/scripts/sim_encoder_ood_probe.py).
N_STRIDED = 300
N_REFERENCE = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=Path("~/datasets/mcobzarenco/so101_pick_place_v2").expanduser(),
    )
    parser.add_argument("--out", type=Path, default=Path("assets/real_plates"))
    parser.add_argument("--top-stride", type=int, default=16)
    parser.add_argument("--wrist-window", type=int, default=12)
    parser.add_argument("--coverage-delta", type=float, default=12.0)
    parser.add_argument(
        "--bank",
        action="store_true",
        help="mine the per-episode top-plate bank + clutter spread "
        "instead of the global plates",
    )
    parser.add_argument(
        "--inlier-delta",
        type=float,
        default=25.0,
        help="bank mode: channel-mean deviation from the corrected "
        "global plate below which a sample counts as background",
    )
    parser.add_argument(
        "--novelty-delta",
        type=float,
        default=40.0,
        help="bank mode: naive-median deviation above which a pixel "
        "is a static novelty (clutter measurement)",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="bank mode: verify the analytic camera model against a "
        "segmentation render (needs MUJOCO_GL=egl) before mining",
    )
    return parser.parse_args()


def a_half_episodes(root: Path) -> tuple[pd.DataFrame, int]:
    episodes = pd.read_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    total = int(episodes["dataset_to_index"].max())
    boundary = N_REFERENCE * (total // N_STRIDED)
    inside = episodes[episodes["dataset_to_index"] <= boundary]
    print(
        f"{len(inside)}/{len(episodes)} episodes wholly inside the "
        f"A half (boundary frame {boundary} of {total})",
    )
    return inside, boundary


def episode_frames(
    root: Path,
    episodes: pd.DataFrame,
    camera: str,
    stride: int,
    window: int | None,
) -> list[np.ndarray]:
    """Decode the selected frames of every episode, grouped by video
    file so each file is read once. ``window`` limits to the first N
    frames of each episode (wrist); else every ``stride``-th frame."""
    key = CAMERA_KEYS[camera]
    frames: list[np.ndarray] = []
    for file_index, group in episodes.groupby(f"videos/{key}/file_index"):
        spans = sorted(
            (
                float(row[f"videos/{key}/from_timestamp"]),
                float(row[f"videos/{key}/to_timestamp"]),
            )
            for _, row in group.iterrows()
        )
        path = root / "videos" / key / "chunk-000" / f"file-{int(file_index):03d}.mp4"
        container = av.open(str(path))
        span_pos = 0
        in_episode = -1
        for frame in container.decode(video=0):
            t = float(frame.time)
            while span_pos < len(spans) and t >= spans[span_pos][1]:
                span_pos += 1
                in_episode = -1
            if span_pos == len(spans):
                break
            if t < spans[span_pos][0]:
                continue
            in_episode += 1
            if window is not None:
                take = in_episode < window
            else:
                take = in_episode % stride == 0
            if take:
                frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
    return frames


# --- bank mode -------------------------------------------------------
# Top camera model, matching sim/so101_sim.py exactly: top_cam pose
# from assets/robotstudio_so101/bijou_pickplace.xml, equidistant
# fisheye with center magnification pinned to the 52-deg pinhole
# (SO101Sim.V1_CENTER_FOVY / _init_fisheye).
CAM_POS = np.array([-0.02, -0.125, 0.555])
_CAM_X = np.array([0.0, -1.0, 0.0])
_CAM_Y = np.array([0.9063, 0.0, 0.4226])
CAM_R = np.stack([_CAM_X, _CAM_Y, np.cross(_CAM_X, _CAM_Y)], axis=1)
WIDTH, HEIGHT = 640, 480
F_DIST = (HEIGHT / 2.0) / np.tan(np.deg2rad(52.0) / 2.0)
CENTER = np.array([(WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0])

# Clutter stand-ins (sim scene canonical world xy), matching radius
# around the canonical image projection (px), unprojection height (m),
# and plausible blob area band (px) — the arms/boat park far from the
# up-table objects; the area band keeps arm-sized blobs off the PCB.
CLUTTER = {
    "mouse": {"canon": (0.50, -0.085), "h": 0.018, "r_px": 80, "area": (250, 8000)},
    "mug": {"canon": (0.50, 0.055), "h": 0.06, "r_px": 80, "area": (250, 8000)},
    "laptop": {"canon": (0.30, -0.52), "h": 0.01, "r_px": 130, "area": (1500, 60000)},
    "pcb": {"canon": (0.10, -0.035), "h": 0.002, "r_px": 70, "area": (600, 12000)},
}


def project(world: np.ndarray) -> np.ndarray:
    """World point [3] -> distorted output pixel [2] (u, v)."""
    v_cam = CAM_R.T @ (np.asarray(world, dtype=np.float64) - CAM_POS)
    theta = np.arccos(np.clip(-v_cam[2] / np.linalg.norm(v_cam), -1.0, 1.0))
    phi = np.arctan2(v_cam[1], v_cam[0])
    r = F_DIST * theta
    return CENTER + np.array([r * np.cos(phi), -r * np.sin(phi)])


def unproject(pixel: np.ndarray, height_m: float) -> np.ndarray:
    """Distorted output pixel [2] -> world xy on the z=height_m plane."""
    dx, dy = np.asarray(pixel, dtype=np.float64) - CENTER
    r = float(np.hypot(dx, dy))
    theta = r / F_DIST
    phi = np.arctan2(-dy, dx)
    d_cam = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)],
    )
    d_world = CAM_R @ d_cam
    t = (height_m - CAM_POS[2]) / d_world[2]
    return (CAM_POS + t * d_world)[:2]


def seg_centroid_out(sim: object, name: str) -> np.ndarray:
    """Segmentation-render centroid of a geom, mapped from the source
    pinhole space into the distorted output space (where real-frame
    blobs are measured)."""
    import mujoco

    renderer = sim.renderer  # type: ignore[attr-defined]
    renderer.enable_segmentation_rendering()
    renderer.update_scene(sim.data, camera="top_cam")  # type: ignore[attr-defined]
    seg = renderer.render()
    renderer.disable_segmentation_rendering()
    gid = sim.model.geom(name).id  # type: ignore[attr-defined]
    mask = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value) & (seg[..., 0] == gid)
    if not mask.any():
        raise SystemExit(f"seg calibration: {name} not visible")
    ys, xs = np.nonzero(mask)
    f_src = (HEIGHT / 2.0) / np.tan(np.deg2rad(72.0) / 2.0)
    src = np.array([xs.mean(), ys.mean()]) - CENTER
    r_src = float(np.hypot(*src))
    r_out = F_DIST * np.arctan(r_src / f_src)
    return CENTER + src * (r_out / r_src)


def calibrate_offsets(*, selfcheck: bool) -> dict[str, np.ndarray]:
    """Per-object pixel offset between the rendered blob centroid and
    the projected geom center, measured on the sim's own segmentation
    render at canonical pose. Subtracting it from a measured real-blob
    centroid before unprojection cancels both the centroid-vs-center
    bias of each shape and the residual camera-model error (raw
    analytic-vs-seg disagreement is up to ~12 px on the flat shapes).

    With ``selfcheck``: displace mouse and pcb by a known world delta,
    re-render, and require the calibrated pipeline to recover the
    displacement to <= 2 cm before any measurement is credited."""
    import mujoco

    from sim.so101_sim import SO101Sim

    sim = SO101Sim(render_style="v1")
    sim.reset(0)
    mujoco.mj_forward(sim.model, sim.data)
    offsets: dict[str, np.ndarray] = {}
    for name, spec in CLUTTER.items():
        proj = project(np.array([*spec["canon"], spec["h"]]))
        blob = seg_centroid_out(sim, name)
        offsets[name] = blob - proj
        print(f"calibration {name}: blob-center offset {offsets[name].round(1)} px")
    if selfcheck:
        for name, delta in (("mouse", (0.05, 0.06)), ("pcb", (-0.04, 0.05))):
            spec = CLUTTER[name]
            gid = sim.model.geom(name).id
            base = sim.model.geom_pos[gid].copy()
            sim.model.geom_pos[gid][:2] = base[:2] + delta
            mujoco.mj_forward(sim.model, sim.data)
            blob = seg_centroid_out(sim, name)
            recovered = unproject(blob - offsets[name], spec["h"])
            truth = np.array(spec["canon"]) + delta
            err = float(np.hypot(*(recovered - truth)))
            print(
                f"camera selfcheck {name}: moved to {truth.round(3)}, "
                f"recovered {recovered.round(3)} -> {err * 100:.1f} cm",
            )
            sim.model.geom_pos[gid] = base
            if err > 0.02:
                raise SystemExit("camera selfcheck failed (> 2 cm) - not mining")
        mujoco.mj_forward(sim.model, sim.data)
    return offsets


def per_episode_frames(
    root: Path,
    episodes: pd.DataFrame,
    camera: str,
    stride: int,
) -> dict[int, list[np.ndarray]]:
    """Every stride-th frame of each episode, keyed by episode index."""
    key = CAMERA_KEYS[camera]
    out: dict[int, list[np.ndarray]] = {}
    for file_index, group in episodes.groupby(f"videos/{key}/file_index"):
        spans = sorted(
            (
                float(row[f"videos/{key}/from_timestamp"]),
                float(row[f"videos/{key}/to_timestamp"]),
                int(row["episode_index"]),
            )
            for _, row in group.iterrows()
        )
        path = root / "videos" / key / "chunk-000" / f"file-{int(file_index):03d}.mp4"
        container = av.open(str(path))
        span_pos = 0
        in_episode = -1
        for frame in container.decode(video=0):
            t = float(frame.time)
            while span_pos < len(spans) and t >= spans[span_pos][1]:
                span_pos += 1
                in_episode = -1
            if span_pos == len(spans):
                break
            if t < spans[span_pos][0]:
                continue
            in_episode += 1
            if in_episode % stride == 0:
                out.setdefault(spans[span_pos][2], []).append(
                    frame.to_ndarray(format="rgb24"),
                )
        container.close()
    return out


def fit_affine(
    naive: np.ndarray,
    global_plate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (gain, bias) mapping the global plate onto this
    episode's naive median, fitted where the two roughly agree (the
    episode's exposure / white-balance state)."""
    close = np.abs(naive - global_plate).mean(axis=-1) < 15.0
    gain = np.empty(3)
    bias = np.empty(3)
    for c in range(3):
        gain[c], bias[c] = np.polyfit(global_plate[close][:, c], naive[close][:, c], 1)
    return gain, bias


def feather(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Separable gaussian blur of a [H, W] float mask."""
    radius = max(1, int(np.ceil(2.5 * sigma)))
    taps = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (taps / sigma) ** 2)
    kernel /= kernel.sum()
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(mask, pad, mode="edge")
        mask = np.zeros_like(mask)
        for i, k in enumerate(kernel):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(i, i + mask.shape[axis])
            mask += padded[tuple(sl)] * k
    return mask


def components(mask: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """4-connected components of a bool mask -> (centroid_uv, area),
    plain BFS (no scipy on this box)."""
    visited = np.zeros_like(mask, dtype=bool)
    out: list[tuple[np.ndarray, int]] = []
    height, width = mask.shape
    for sy, sx in zip(*np.nonzero(mask), strict=True):
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = True
        ys_sum = xs_sum = area = 0
        while stack:
            y, x = stack.pop()
            ys_sum += y
            xs_sum += x
            area += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                inside = 0 <= ny < height and 0 <= nx < width
                if inside and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        out.append((np.array([xs_sum / area, ys_sum / area]), area))
    return out


def measure_clutter(
    novelty: np.ndarray,
    naive: np.ndarray,
    episode: int,
    offsets: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    """Locate each stand-in's real twin among the static-novelty blobs
    and unproject the (offset-calibrated) centroid to world xy.

    Per-object rules tuned on the 26 A-half novelty maps (blob_diag
    inspection, 08-12): color separates the dark mouse from the white
    up-table item the mug stands in for; image zones separate up-table
    clutter (top band) from the wandering disk (center), the laptop
    (right edge) and the arm/operator rest ghosts (bottom right). The
    disk itself is measured record-only (it is task geometry, not a
    draw axis) with the boat usually resting on it at episode end."""
    reads: dict[str, dict[str, object]] = {}

    def blob_color(centroid: np.ndarray) -> float:
        x0, y0 = int(centroid[0]), int(centroid[1])
        patch = naive[max(0, y0 - 4) : y0 + 5, max(0, x0 - 4) : x0 + 5]
        return float(np.median(patch.reshape(-1, 3), axis=0).mean())

    blobs = [
        (centroid, area, blob_color(centroid))
        for centroid, area in components(novelty)
        if area >= 250
    ]

    def pick(
        name: str,
        rule: object,
        height_m: float,
    ) -> None:
        matches = [b for b in blobs if rule(*b)]  # type: ignore[operator]
        if not matches:
            reads[name] = {"present": False}
        else:
            centroid, area, color = max(matches, key=lambda b: b[1])
            offset = offsets.get(name, np.zeros(2))
            world = unproject(centroid - offset, height_m)
            reads[name] = {
                "present": True,
                "world_xy": [round(float(v), 4) for v in world],
                "px": [round(float(v), 1) for v in centroid],
                "area_px": area,
                "rgb_mean": round(color, 1),
            }
        print(f"  ep {episode:2d} {name}: {reads[name]}")

    pick(
        "mouse",
        lambda c, a, col: (
            col < 130 and 800 <= a <= 9000 and c[1] < 140 and 250 <= c[0] <= 520
        ),
        CLUTTER["mouse"]["h"],
    )
    pick(
        "mug",
        lambda c, a, col: col > 175 and 800 <= a <= 9000 and c[1] < 140 and c[0] <= 350,
        CLUTTER["mug"]["h"],
    )
    # Laptop: zone centroid of every dark novelty pixel at the right
    # edge (the blob splits into components across episodes).
    zone = novelty.copy()
    zone[:, :560] = False
    zone[:250] = False
    zone[470:] = False
    dark = naive.mean(axis=-1) < 140
    zone &= dark
    if zone.sum() >= 600:
        ys, xs = np.nonzero(zone)
        centroid = np.array([xs.mean(), ys.mean()])
        world = unproject(centroid - offsets["laptop"], CLUTTER["laptop"]["h"])
        reads["laptop"] = {
            "present": True,
            "world_xy": [round(float(v), 4) for v in world],
            "px": [round(float(v), 1) for v in centroid],
            "area_px": int(zone.sum()),
        }
    else:
        reads["laptop"] = {"present": False}
    print(f"  ep {episode:2d} laptop: {reads['laptop']}")
    canon_pcb = project(np.array([*CLUTTER["pcb"]["canon"], CLUTTER["pcb"]["h"]]))
    canon_pcb = canon_pcb + offsets["pcb"]
    pcb = [
        (c, a, col)
        for c, a, col in blobs
        if col < 130 and 600 <= a <= 12000 and float(np.hypot(*(c - canon_pcb))) <= 70
    ]
    if pcb:
        centroid, area, _col = min(
            pcb,
            key=lambda b: float(np.hypot(*(b[0] - canon_pcb))),
        )
        world = unproject(centroid - offsets["pcb"], CLUTTER["pcb"]["h"])
        reads["pcb"] = {
            "present": True,
            "world_xy": [round(float(v), 4) for v in world],
            "px": [round(float(v), 1) for v in centroid],
            "area_px": area,
        }
    else:
        reads["pcb"] = {"present": False}
    print(f"  ep {episode:2d} pcb: {reads['pcb']}")
    # Record-only: the real disk (+ the boat usually parked on it at
    # episode end) — bright, center zone. Feeds the out-of-scope
    # disk-position item, drawn by nothing in this pass.
    pick(
        "disk_record_only",
        lambda c, a, col: (
            col > 175
            and 1500 <= a <= 9000
            and 140 <= c[1] <= 400
            and 120 <= c[0] <= 380
        ),
        0.006,
    )
    return reads


def main_bank(args: argparse.Namespace) -> int:
    from PIL import Image

    offsets = calibrate_offsets(selfcheck=args.selfcheck)
    out_dir = args.out / "bank"
    out_dir.mkdir(parents=True, exist_ok=True)
    global_plate = np.asarray(
        Image.open(args.out / "top_plate.png"),
        dtype=np.float64,
    )
    episodes, boundary = a_half_episodes(args.v2_root)
    frames = per_episode_frames(args.v2_root, episodes, "top", args.top_stride)
    manifest: dict[str, object] = {
        "v2_root": str(args.v2_root),
        "a_boundary_frame": boundary,
        "top_stride": args.top_stride,
        "inlier_delta": args.inlier_delta,
        "novelty_delta": args.novelty_delta,
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
    }
    per_episode: dict[str, dict[str, object]] = {}
    clutter_xy: dict[str, list[list[float]]] = {name: [] for name in CLUTTER}
    for episode, stack_list in sorted(frames.items()):
        stack = np.stack(stack_list).astype(np.float64)
        naive = np.median(stack, axis=0)
        gain, bias = fit_affine(naive, global_plate)
        corrected = np.clip(global_plate * gain + bias, 0.0, 255.0)
        # max over channels: the mean is blind to hue-level change
        # (the operator's hand on warm wood passed a mean test and
        # smeared pink into the ep24 plate on the first mining run).
        deviation = np.abs(stack - corrected).max(axis=-1)  # [N, H, W]
        inlier = deviation < args.inlier_delta
        counts = inlier.sum(axis=0)
        # Median over inlier samples only: sort with outliers pushed to
        # +inf, then index the per-pixel inlier count's midpoint.
        ranked = np.where(inlier[..., None], stack, np.inf).astype(np.float64)
        ranked.sort(axis=0)
        mid = np.maximum(counts - 1, 0) // 2
        inlier_median = np.take_along_axis(
            ranked,
            mid[None, ..., None].repeat(3, axis=-1),
            axis=0,
        )[0]
        n_min = max(4, int(np.ceil(0.30 * len(stack))))
        fallback = counts < n_min
        weight = np.clip(feather(fallback.astype(np.float64)), 0.0, 1.0)[..., None]
        base = np.where(fallback[..., None], corrected, inlier_median)
        plate = (1.0 - weight) * base + weight * corrected
        Image.fromarray(np.clip(plate, 0, 255).astype(np.uint8)).save(
            out_dir / f"top_ep{episode:03d}.png",
        )
        novelty = np.abs(naive - corrected).mean(axis=-1) > args.novelty_delta
        reads = measure_clutter(novelty, naive, episode, offsets)
        for name, read in reads.items():
            if read["present"]:
                clutter_xy.setdefault(name, []).append(read["world_xy"])  # type: ignore[arg-type]
        per_episode[str(episode)] = {
            "n_frames": len(stack),
            "gain": [round(float(g), 4) for g in gain],
            "bias": [round(float(b), 2) for b in bias],
            "fallback_frac": round(float(fallback.mean()), 4),
            "clutter": reads,
        }
        print(
            f"ep {episode:2d}: n={len(stack):2d} fallback "
            f"{fallback.mean():.3f} gain {gain.round(3)}",
        )
    # Draw modes consumed by SO101Sim v3: absolute boxes where the
    # object is fully visible (mouse, mug), deltas about the sim
    # canonical for the crop-biased laptop (its real center is past
    # the frame edge; only the visible-part centroid is measurable,
    # so the spread is trustworthy but the absolute position is not),
    # fixed canonical for the pcb (cabled to the arms, visually
    # near-static; the blob matcher rarely separates it from the
    # parked-arm ghosts). disk_record_only is exactly that.
    modes = {
        "mouse": "absolute",
        "mug": "absolute",
        "laptop": "delta_about_canonical",
        "pcb": "fixed_canonical",
        "disk_record_only": "record_only",
    }
    ranges: dict[str, dict[str, object]] = {}
    for name, points in sorted(clutter_xy.items()):
        presence = len(points) / len(frames)
        entry: dict[str, object] = {
            "mode": modes[name],
            "presence": round(presence, 3),
            "n_present": len(points),
        }
        if len(points) >= 2:
            arr = np.array(points)
            entry["xy_min"] = [round(float(v), 4) for v in arr.min(axis=0)]
            entry["xy_max"] = [round(float(v), 4) for v in arr.max(axis=0)]
            if modes[name] == "delta_about_canonical":
                mean = arr.mean(axis=0)
                entry["xy_delta_min"] = [
                    round(float(v), 4) for v in arr.min(axis=0) - mean
                ]
                entry["xy_delta_max"] = [
                    round(float(v), 4) for v in arr.max(axis=0) - mean
                ]
        ranges[name] = entry
        print(f"{name}: {entry}")
    manifest["calibration_offsets_px"] = {
        name: [round(float(v), 2) for v in off] for name, off in offsets.items()
    }
    manifest["episodes"] = per_episode
    manifest["clutter_ranges"] = ranges
    (out_dir / "bank_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out_dir}/bank_manifest.json ({len(frames)} plates)")
    return 0


def main() -> int:
    args = parse_args()
    if args.bank:
        return main_bank(args)
    episodes, boundary = a_half_episodes(args.v2_root)
    args.out.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    manifest: dict[str, object] = {
        "v2_root": str(args.v2_root),
        "a_boundary_frame": boundary,
        "episodes": [int(e) for e in episodes["episode_index"]],
        "top_stride": args.top_stride,
        "wrist_window": args.wrist_window,
        "coverage_delta": args.coverage_delta,
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
    }
    for camera in CAMERA_KEYS:
        window = args.wrist_window if camera == "wrist" else None
        frames = episode_frames(
            args.v2_root,
            episodes,
            camera,
            args.top_stride,
            window,
        )
        # In-place partition instead of np.median (which would copy
        # the ~1 GB stack to float64); the coverage read only needs
        # each pixel's value multiset, which partitioning preserves.
        stack = np.stack(frames)
        del frames
        stack.partition(len(stack) // 2, axis=0)
        plate = stack[len(stack) // 2].copy()
        residual = np.abs(stack.astype(np.int16) - plate.astype(np.int16))
        coverage = (residual.mean(axis=-1) < args.coverage_delta).mean(axis=0)
        Image.fromarray(plate).save(args.out / f"{camera}_plate.png")
        Image.fromarray((coverage * 255).astype(np.uint8)).save(
            args.out / f"{camera}_coverage.png",
        )
        manifest[f"{camera}_frames"] = len(stack)
        manifest[f"{camera}_coverage_mean"] = float(coverage.mean())
        print(
            f"{camera}: {len(stack)} frames -> plate "
            f"{plate.shape[1]}x{plate.shape[0]}, "
            f"coverage mean {coverage.mean():.3f}",
        )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
