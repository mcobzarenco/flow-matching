"""Mine real-pixel RGBA crops of the clutter objects for the
foreground appearance pass leg (b) (pre-reg in-channel 05:23Z
2026-08-13).

Per object (mouse, mug, laptop, pcb): the source episode is the
bank-manifest episode with the largest present ``area_px`` (registered
rule). The crop's RGB is that episode's naive per-pixel median —
static content survives the median, so the object arrives with its
real texture, optics and contact shadow — normalized back to
global-plate lighting via the episode's fitted gain/bias (the paste
site re-grades it with the active episode's affine). Alpha is the
feathered static-novelty blob (naive median vs the gain/bias-corrected
global plate), the same statistic and threshold the bank mining pass
used to localize the objects.

In-run oracles: the refitted per-episode gain/bias must match the bank
manifest (the source data has not drifted), and the recomputed blob
centroid must sit within a few px of the manifest ``px`` (the crop is
the same object the pass measured).

Outputs (assets/real_plates/bank/crops/): {name}.png (RGBA),
crops_manifest.json (per object: source episode, bbox, anchor world
xy / px, mode, source gain/bias), and crops_strip.png (visual gate:
crops over a checker plus a paste-identity preview vs the naive
median).

Usage:
  uv run python fontaine/scripts/make_clutter_crops.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_clean_plates import (
    CLUTTER,
    a_half_episodes,
    components,
    feather,
    fit_affine,
    per_episode_frames,
)

TARGETS = ("mouse", "mug", "laptop", "pcb")
FEATHER_SIGMA = 2.0
BBOX_PAD = 8
NOVELTY_DELTA = 40.0  # == the bank pass default (manifest-checked)
CENTROID_TOL_PX = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=Path("~/datasets/mcobzarenco/so101_pick_place_v2").expanduser(),
    )
    parser.add_argument(
        "--plates",
        type=Path,
        default=Path("assets/real_plates"),
    )
    return parser.parse_args()


def object_mask(
    name: str,
    novelty: np.ndarray,
    naive: np.ndarray,
    anchor_px: np.ndarray,
) -> np.ndarray:
    """Bool mask of the object's pixels, by the bank pass's per-object
    rules: largest novelty component in the anchor window, except the
    laptop (zone pixels — its blob splits across components)."""
    if name == "laptop":
        zone = novelty.copy()
        zone[:, :560] = False
        zone[:250] = False
        zone[470:] = False
        return zone & (naive.mean(axis=-1) < 140)
    r = CLUTTER[name]["r_px"] + 20
    window = np.zeros_like(novelty)
    y0, x0 = int(anchor_px[1]), int(anchor_px[0])
    window[max(0, y0 - r) : y0 + r, max(0, x0 - r) : x0 + r] = True
    boxed = novelty & window
    # pcb replicates the bank rule (nearest candidate in the area band
    # — parked-arm ghosts nearby are LARGER); mouse/mug take the
    # largest component near the anchor.
    lo, hi = CLUTTER[name]["area"]
    candidates = [
        (centroid, area)
        for centroid, area in components(boxed)
        if float(np.hypot(*(centroid - anchor_px))) <= r and lo <= area <= hi
    ]
    if not candidates:
        raise SystemExit(f"{name}: no novelty component near {anchor_px}")
    if name == "pcb":
        best = min(candidates, key=lambda b: float(np.hypot(*(b[0] - anchor_px))))[0]
    else:
        best = max(candidates, key=lambda b: b[1])[0]
    # rebuild just the winning component (components() only returns
    # centroids): BFS from the pixel of the component nearest the
    # winning centroid
    from collections import deque

    seed = None
    ys, xs = np.nonzero(boxed)
    d2 = (xs - best[0]) ** 2 + (ys - best[1]) ** 2
    seed = (int(ys[d2.argmin()]), int(xs[d2.argmin()]))
    keep = np.zeros_like(boxed)
    queue = deque([seed])
    keep[seed] = True
    height, width = boxed.shape
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= ny < height
                and 0 <= nx < width
                and boxed[ny, nx]
                and not keep[ny, nx]
            ):
                keep[ny, nx] = True
                queue.append((ny, nx))
    return keep


def checker(height: int, width: int, cell: int = 12) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    tile = ((yy // cell + xx // cell) % 2).astype(np.float64)
    return (110 + 60 * tile)[..., None].repeat(3, axis=-1)


def main() -> int:
    from PIL import Image

    args = parse_args()
    bank_dir = args.plates / "bank"
    manifest = json.loads((bank_dir / "bank_manifest.json").read_text())
    if manifest["novelty_delta"] != NOVELTY_DELTA:
        raise SystemExit("bank manifest novelty_delta drifted from the mining default")
    global_plate = np.asarray(
        Image.open(args.plates / "top_plate.png"),
        dtype=np.float64,
    )

    sources: dict[str, int] = {}
    for name in TARGETS:
        present = [
            (int(ep), entry["clutter"][name])
            for ep, entry in manifest["episodes"].items()
            if entry["clutter"][name]["present"]
        ]
        if not present:
            raise SystemExit(f"{name}: present in no bank episode")
        sources[name] = max(present, key=lambda row: row[1]["area_px"])[0]
    print(f"source episodes (max area_px): {sources}")

    episodes, _ = a_half_episodes(args.v2_root)
    chosen = episodes[episodes["episode_index"].isin(set(sources.values()))]
    frames = per_episode_frames(
        args.v2_root,
        chosen,
        "top",
        manifest["top_stride"],
    )

    out_dir = bank_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_manifest: dict[str, object] = {
        "prereg": "fg appearance pass leg (b), in-channel 05:23Z 2026-08-13",
        "novelty_delta": NOVELTY_DELTA,
        "feather_sigma": FEATHER_SIGMA,
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "objects": {},
    }
    previews: list[np.ndarray] = []
    for name in TARGETS:
        episode = sources[name]
        entry = manifest["episodes"][str(episode)]
        read = entry["clutter"][name]
        naive = np.median(np.stack(frames[episode]).astype(np.float64), axis=0)
        gain, bias = fit_affine(naive, global_plate)
        if not (
            np.allclose(gain, entry["gain"], atol=5e-3)
            and np.allclose(bias, entry["bias"], atol=0.5)
        ):
            raise SystemExit(f"{name}: refitted gain/bias drifted from the manifest")
        corrected = np.clip(global_plate * gain + bias, 0.0, 255.0)
        novelty = np.abs(naive - corrected).mean(axis=-1) > NOVELTY_DELTA
        anchor_px = np.array(read["px"])
        mask = object_mask(name, novelty, naive, anchor_px)
        ys, xs = np.nonzero(mask)
        centroid = np.array([xs.mean(), ys.mean()])
        drift = float(np.hypot(*(centroid - anchor_px)))
        if drift > CENTROID_TOL_PX:
            raise SystemExit(
                f"{name}: recomputed centroid {centroid.round(1)} is "
                f"{drift:.1f} px from the manifest px {anchor_px} (> "
                f"{CENTROID_TOL_PX})",
            )
        alpha = np.clip(feather(mask.astype(np.float64), FEATHER_SIGMA), 0.0, 1.0)
        x0 = max(0, int(xs.min()) - BBOX_PAD)
        x1 = min(mask.shape[1], int(xs.max()) + 1 + BBOX_PAD)
        y0 = max(0, int(ys.min()) - BBOX_PAD)
        y1 = min(mask.shape[0], int(ys.max()) + 1 + BBOX_PAD)
        rgb_norm = np.clip((naive - bias) / gain, 0.0, 255.0)
        rgba = np.dstack(
            [rgb_norm[y0:y1, x0:x1], alpha[y0:y1, x0:x1] * 255.0],
        ).astype(np.uint8)
        Image.fromarray(rgba).save(out_dir / f"{name}.png")
        crops_manifest["objects"][name] = {  # type: ignore[index]
            "episode": episode,
            "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
            "anchor_world_xy": read["world_xy"],
            "anchor_px": read["px"],
            "area_px": int(mask.sum()),
            "manifest_area_px": read["area_px"],
            "height_m": CLUTTER[name]["h"],
            "mode": manifest["clutter_ranges"][name]["mode"],
            "source_gain": [round(float(g), 4) for g in gain],
            "source_bias": [round(float(b), 2) for b in bias],
        }
        print(
            f"{name}: ep {episode} bbox ({x0},{y0},{x1 - x0},{y1 - y0}) "
            f"area {int(mask.sum())} (manifest {read['area_px']}) "
            f"centroid drift {drift:.1f} px",
        )
        # preview row: crop on checker | naive median bbox | graded
        # global plate bbox with the crop alpha-blended back (identity
        # paste — should visually match the naive median)
        board = checker(y1 - y0, x1 - x0)
        over = (
            board * (1 - alpha[y0:y1, x0:x1, None])
            + rgb_norm[y0:y1, x0:x1] * alpha[y0:y1, x0:x1, None]
        )
        pasted = corrected.copy()
        blend = alpha[..., None]
        pasted = (
            pasted * (1 - blend) + np.clip(rgb_norm * gain + bias, 0.0, 255.0) * blend
        )
        row = np.concatenate(
            [over, naive[y0:y1, x0:x1], pasted[y0:y1, x0:x1]],
            axis=1,
        )
        previews.append(row)

    width = max(row.shape[1] for row in previews)
    strip = np.concatenate(
        [np.pad(row, ((0, 4), (0, width - row.shape[1]), (0, 0))) for row in previews],
        axis=0,
    )
    Image.fromarray(np.clip(strip, 0, 255).astype(np.uint8)).save(
        out_dir / "crops_strip.png",
    )
    (out_dir / "crops_manifest.json").write_text(json.dumps(crops_manifest, indent=1))
    print(f"wrote {out_dir}/crops_manifest.json + crops + strip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
