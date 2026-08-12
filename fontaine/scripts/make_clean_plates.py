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


def main() -> int:
    args = parse_args()
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
