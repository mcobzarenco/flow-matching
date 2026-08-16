"""Wrist-cam matched-pairs instrument (queue: wrist-cam-pose-refit,
owner ask 2026-08-16 21:43Z, plan agreed in-channel 22:0xZ).

Stage 1 of the lens-plumbline pattern (instrument first, fit second):
replay rig-v2 recorded per-frame joint STATES into the sim
kinematically (no servo dynamics — qpos set + mj_forward) and render
the wrist camera at identical kinematics, next to the decoded real
wrist frame at that exact timestamp. The discrepancy the pairs expose
is then PURE camera pose/optics (mount-local roll/tilt, fovy), not
kinematics: both sides show the same joint configuration.

Motivating defect (eyeballed 08-16): rig v2 wrist frames show BOTH
jaw tips symmetric from the bottom edge; the sim wrist
(sim-wrist-periphery-fix pose, so101_sim._repose_wrist_cam) shows one
clockwise-leaning orange tip — the re-tune overcorrected.

Outputs (out-dir, default outputs/sim/wrist_refit/matched_pairs):
  real_ep{E:03d}_f{F:05d}.png   decoded rig wrist frame (rgb)
  sim_ep{E:03d}_f{F:05d}.png    sim wrist render at the same qpos
  sbs_ep{E:03d}_f{F:05d}.png    side-by-side (real | sim)
  manifest.json                 rows: episode, frame, state_deg,
                                subtask (when annotated), file paths;
                                plus the sim config + camera pose
                                actually rendered under.

The scene background differs by construction (real clutter/lighting vs
sim plates) — stage-2 measurements (jaw-axis angle, bottom-band
occupancy, both-jaws-visible rate) are gripper-geometry reads robust
to background. The scene is reset once at a fixed seed (--scene-seed,
default 0) so every pair shares one background; the seed is recorded
in the manifest.

Usage:
  uv run python fontaine/scripts/wrist_cam_matched_pairs.py           # 26 eps x 12
  uv run python fontaine/scripts/wrist_cam_matched_pairs.py --episodes 0 1 --frames-per-episode 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.replay_control_loss import REFERENCE_EPISODES, V2_REPO
from sim.so101_sim import SO101Sim
from sim.sysid_servo import DATASETS_ROOT, load_episode

WRIST_KEY = "videos/observation.images.wrist"
FPS = 30.0


def episodes_meta(root: Path) -> dict[int, dict]:
    import pandas as pd

    rows: dict[int, dict] = {}
    for file in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(
            file,
            columns=[
                "episode_index",
                "length",
                f"{WRIST_KEY}/chunk_index",
                f"{WRIST_KEY}/file_index",
                f"{WRIST_KEY}/from_timestamp",
                "subtask_names",
                "subtask_start_frames",
                "subtask_end_frames",
            ],
        )
        for _, row in df.iterrows():
            rows[int(row["episode_index"])] = row.to_dict()
    return rows


def decode_wrist_frames(
    root: Path,
    meta: dict,
    frame_indices: list[int],
) -> list[np.ndarray]:
    """Decode the wrist frames at the given episode-local indices
    (single sequential pass, fit_contact_shadow pattern)."""
    t0 = float(meta[f"{WRIST_KEY}/from_timestamp"])
    chunk = int(meta[f"{WRIST_KEY}/chunk_index"])
    file = int(meta[f"{WRIST_KEY}/file_index"])
    times = [t0 + f / FPS for f in frame_indices]
    path = root / WRIST_KEY / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
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
    if len(frames) != len(frame_indices):
        raise SystemExit(f"episode {meta['episode_index']}: decode short")
    return frames


def subtask_at(meta: dict, frame: int) -> str | None:
    names = meta.get("subtask_names")
    if names is None or not hasattr(names, "__len__"):
        return None
    starts, ends = meta["subtask_start_frames"], meta["subtask_end_frames"]
    for name, s, e in zip(names, starts, ends, strict=True):
        if s <= frame < e:
            return str(name)
    return None


def render_at_state(sim: SO101Sim, state_deg: np.ndarray) -> np.ndarray:
    """Wrist render at the recorded joint state: kinematic qpos set on
    the 6 follower joints (everything else at qpos0) + mj_forward, then
    the deployed observe() wrist path (lens + grade), so the sim side
    is byte-for-byte what the training pipeline would see at this
    kinematic configuration."""
    sim.data.qpos[sim._joint_qpos] = np.deg2rad(state_deg)
    mujoco.mj_forward(sim.model, sim.data)
    return sim.observe().wrist


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=list(REFERENCE_EPISODES),
        help="v2 episode ids (default: the 26 reference episodes)",
    )
    parser.add_argument("--frames-per-episode", type=int, default=12)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/sim/wrist_refit/matched_pairs"),
    )
    parser.add_argument(
        "--render-style",
        default="v3",
        help="SO101Sim render_style; wrist rides the v1 lens path in all",
    )
    parser.add_argument("--scene-seed", type=int, default=0)
    args = parser.parse_args()

    root = DATASETS_ROOT / V2_REPO
    metas = episodes_meta(root)
    sim = SO101Sim(render_style=args.render_style)
    sim.reset(args.scene_seed)
    cam = sim.model.camera("wrist_cam")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "repo": V2_REPO,
        "render_style": args.render_style,
        "scene_seed": args.scene_seed,
        "wrist_cam": {
            "pos": [float(v) for v in cam.pos],
            "quat": [float(v) for v in cam.quat],
            "fovy": float(cam.fovy[0]),
        },
        "frames_per_episode": args.frames_per_episode,
        "rows": [],
    }
    for episode in args.episodes:
        meta = metas[episode]
        _, states = load_episode(V2_REPO, episode)
        length = min(int(meta["length"]), len(states))
        picks = sorted(
            {int(i) for i in np.linspace(0, length - 1, args.frames_per_episode)},
        )
        real = decode_wrist_frames(root, meta, picks)
        for frame_index, real_frame in zip(picks, real, strict=True):
            state = states[frame_index]
            sim_frame = render_at_state(sim, state)
            stem = f"ep{episode:03d}_f{frame_index:05d}"
            Image.fromarray(real_frame).save(args.out_dir / f"real_{stem}.png")
            Image.fromarray(sim_frame).save(args.out_dir / f"sim_{stem}.png")
            gap = np.full((real_frame.shape[0], 8, 3), 24, dtype=np.uint8)
            sbs = np.concatenate([real_frame, gap, sim_frame], axis=1)
            Image.fromarray(sbs).save(args.out_dir / f"sbs_{stem}.png")
            manifest["rows"].append(
                {
                    "episode": episode,
                    "frame": frame_index,
                    "state_deg": [round(float(v), 3) for v in state],
                    "subtask": subtask_at(meta, frame_index),
                    "real": f"real_{stem}.png",
                    "sim": f"sim_{stem}.png",
                    "sbs": f"sbs_{stem}.png",
                },
            )
        print(f"episode {episode}: {len(picks)} pairs")
    out = args.out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest['rows'])} pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
