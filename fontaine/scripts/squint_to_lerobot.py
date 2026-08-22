"""Twin episodes -> LeRobot v3 dataset + the pre-registered conversion
oracle (posts/2026-08-22-prereg-squint-twin-screen.md, Gate-1
"Conversion" bullet). Main venv (lerobot 0.6.0).

Input: outputs/squint_screen/{lift,place}/episodes/ep_*.npz from
squint_expert_collect.py (10 Hz twin steps: absolute radian targets,
qpos, wrist+front 224 frames, per-step predicates), kept_ids.json.

Output dataset: ~/datasets/fontaine/squint_twin_demos_v1 — fps 30 by
repeat-3 upsampling, so the deploy adapter's subsample-every-3rd exactly
inverts the conversion (self-consistent round trip). Actions/state ride
the inverse adapter back to our servo-degree convention: arm rad2deg,
gripper inverse-affine twin sim [-10,120] deg -> ours [0, 41.69]. Camera
keys mirror grasp_demos_v2: observation.images.wrist (twin base_camera,
wrist mount) + observation.images.front (third_camera, kind tag front).

Oracle (receipts to conversion_oracle.json):
  - action round trip: dataset ep 0 actions, subsample-by-3 + forward
    adapter -> max |delta| vs the original twin targets (< 1e-5 rad).
  - state round trip: same path on observation.state (< 1e-5 rad).
  - frames: writer input IS the re-render output (bit-exact by
    construction at the pre-encode boundary); decoded-video fidelity
    PSNR recorded as a fact, not a gate (our own demos ride the same
    mp4 path).

Run: uv run python fontaine/scripts/squint_to_lerobot.py
"""

import json
import shutil
from pathlib import Path

import numpy as np

OUT_ROOT = Path("/home/ubuntu/flow-matching/outputs/squint_screen")
DS_ROOT = Path("/home/ubuntu/datasets/fontaine/squint_twin_demos_v1")
REPO_ID = "fontaine/squint_twin_demos_v1"

# Frozen adapter constants (Gate 0 / preflight-2 receipts).
OUR_OPEN = 41.69
SIM_MIN, SIM_MAX = -10.0, 120.0
REPEAT = 3
FPS = 30

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Frozen instruction strings (finalization amendment slots).
INSTRUCTIONS = {
    "lift": "Pick up the red cube.",
    "place": "Pick up the red cube and place it in the bin.",
}


def from_sim_rad(q_rad: np.ndarray) -> np.ndarray:
    """Twin absolute radians (T, 6) -> our servo degrees (T, 6)."""
    deg = np.rad2deg(q_rad.astype(np.float64))
    frac = (deg[..., 5] - SIM_MIN) / (SIM_MAX - SIM_MIN)
    deg[..., 5] = frac * OUR_OPEN
    return deg.astype(np.float32)


def to_sim_rad(a_deg: np.ndarray) -> np.ndarray:
    """Our servo degrees -> twin radians (the deploy adapter direction)."""
    out = np.deg2rad(a_deg.astype(np.float64))
    frac = a_deg[..., 5].astype(np.float64) / OUR_OPEN
    out[..., 5] = np.deg2rad(frac * (SIM_MAX - SIM_MIN) + SIM_MIN)
    return out


def build() -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if DS_ROOT.exists():
        shutil.rmtree(DS_ROOT)

    features = {
        "action": {"dtype": "float32", "shape": [6], "names": JOINT_NAMES},
        "observation.state": {"dtype": "float32", "shape": [6], "names": JOINT_NAMES},
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [224, 224, 3],
            "names": ["height", "width", "channels"],
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": [224, 224, 3],
            "names": ["height", "width", "channels"],
        },
    }
    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        root=DS_ROOT,
        robot_type="so101",
        use_videos=True,
    )

    counts, ep0 = {}, None
    for task in ("lift", "place"):
        tdir = OUT_ROOT / task
        if not (tdir / "kept_ids.json").exists():
            print(f"[convert] no kept episodes for {task} — skipping")
            counts[task] = 0
            continue
        kept = json.loads((tdir / "kept_ids.json").read_text())
        counts[task] = len(kept)
        for i in kept:
            e = np.load(tdir / "episodes" / f"ep_{i:04d}.npz")
            act10 = from_sim_rad(e["targets"])
            st10 = from_sim_rad(e["qpos"])
            if ep0 is None:
                ep0 = {
                    "task": task,
                    "targets_rad": e["targets"].copy(),
                    "qpos_rad": e["qpos"].copy(),
                    "wrist0": e["wrist"][0].copy(),
                    "front0": e["front"][0].copy(),
                }
            for t in range(len(act10) * REPEAT):
                s = t // REPEAT
                ds.add_frame(
                    {
                        "action": act10[s],
                        "observation.state": st10[s],
                        "observation.images.wrist": e["wrist"][s],
                        "observation.images.front": e["front"][s],
                        "task": INSTRUCTIONS[task],
                    },
                )
            ds.save_episode()
    if ep0 is None:
        msg = "no episodes converted — refusing to finalize an empty dataset"
        raise SystemExit(msg)
    ds.finalize()
    return {"episode_counts": counts, "ep0": ep0}


def oracle(ep0: dict) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(REPO_ID, root=DS_ROOT)
    n0 = len(ep0["targets_rad"]) * REPEAT
    rows = [ds[t] for t in range(n0)]
    acts = np.stack([r["action"].numpy() for r in rows])
    states = np.stack([r["observation.state"].numpy() for r in rows])

    # Deploy-adapter direction: subsample every 3rd row, forward adapter.
    act_rt = to_sim_rad(acts[::REPEAT])
    st_rt = to_sim_rad(states[::REPEAT])
    d_act = float(np.abs(act_rt - ep0["targets_rad"]).max())
    d_st = float(np.abs(st_rt - ep0["qpos_rad"]).max())

    def psnr(a: np.ndarray, b: np.ndarray) -> float:
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
        return float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf")

    def img(row_val: "object") -> np.ndarray:
        arr = row_val.numpy() if hasattr(row_val, "numpy") else np.asarray(row_val)
        if arr.ndim == 3 and arr.shape[0] == 3:  # CHW float 0..1 -> HWC uint8
            arr = (arr.transpose(1, 2, 0) * 255).round()
        return arr.astype(np.uint8)

    return {
        "ep0_task": ep0["task"],
        "ep0_rows_30hz": n0,
        "action_roundtrip_max_rad": d_act,
        "state_roundtrip_max_rad": d_st,
        "gate_roundtrip_lt_1e5_rad": bool(d_act < 1e-5 and d_st < 1e-5),
        "video_psnr_frame0": {
            "wrist": psnr(img(rows[0]["observation.images.wrist"]), ep0["wrist0"]),
            "front": psnr(img(rows[0]["observation.images.front"]), ep0["front0"]),
        },
        "fps": FPS,
        "repeat": REPEAT,
        "instructions": INSTRUCTIONS,
    }


def main() -> None:
    b = build()
    facts = oracle(b["ep0"])
    facts["episode_counts"] = b["episode_counts"]
    out = OUT_ROOT / "conversion_oracle.json"
    out.write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
