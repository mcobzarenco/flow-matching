"""Squint preflight-2: the CPU receipts for the twin qualification screen.

Exec item `squint-twin-screen-exec` part (a); pre-reg
posts/2026-08-22-prereg-squint-twin-screen.md §"Preflight-2 receipts".
Receipts produced here (facts + frames under outputs/squint_preflight2/):

  R1 dual-camera subclass — wrist + third-person in ONE env at 224x224
     (the setup the preflight-1 note priced as "the one thing that
     genuinely needs a small subclass"), both frames saved.
  R2 determinism entry gate — same seed + same action sequence twice on
     the dual env, sensor bytes and qpos bit-equal (wrist-screen
     precedent; DR off).
  R3 Gate-0 rig-episode replay — grasp_demos_v2/merged episode 0 action
     trace (servo-degree LeRobot convention, extracted to
     replay_trace_ep0.npz by the main venv) driven through the frozen
     adapter mapping: arm joints deg->rad, gripper affine ours
     {0..41.69} -> twin sim {-10..120} deg (constants from the twin's
     own deploy path, deploy_utils/manipulator.py), 30->10 Hz
     subsample-by-3. Reads: tracking p50/p95/max over arm joints,
     joint-limit clip count, commanded vs achieved gripper transitions.
     Pre-registered pass line: tracking p50 < 0.05 rad, zero limit
     violations on the 5 arm joints.
  R4 train_squint.py smoke — the SAC-expert entry point parses (--help
     rc 0) in the twin venv; no GPU touched.

Run from the squint checkout with its isolated venv (GPU guard: the
gripfix battery owns gpu0 — CUDA_VISIBLE_DEVICES empty + lavapipe):

  cd ~/squint && VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python \
    ~/flow-matching/fontaine/scripts/squint_preflight2.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

OUT = Path("/home/ubuntu/flow-matching/outputs/squint_preflight2")
OUT.mkdir(parents=True, exist_ok=True)
SQUINT = Path("/home/ubuntu/squint")

# The frozen adapter constants (pre-reg Gate 0). Twin side from
# deploy_utils/manipulator.py (sim gripper convention in degrees); our
# side from the banked demos convention (bang-bang {0, 41.69}).
OUR_GRIPPER_CLOSED_DEG = 0.0
OUR_GRIPPER_OPEN_DEG = 41.69
SIM_GRIPPER_MIN_DEG = -10.0
SIM_GRIPPER_MAX_DEG = 120.0
SUBSAMPLE = 3  # 30 Hz chunks -> 10 Hz twin control
ARM_JOINTS = 5
TRACK_P50_GATE_RAD = 0.05


def to_sim_rad(action_servo_deg: "Any") -> "Any":
    """Our 6-dim servo-degree action -> twin absolute-radian target."""
    import numpy as np

    out = np.deg2rad(np.asarray(action_servo_deg, dtype=np.float64)).copy()
    g = float(action_servo_deg[5])
    frac = (g - OUR_GRIPPER_CLOSED_DEG) / (
        OUR_GRIPPER_OPEN_DEG - OUR_GRIPPER_CLOSED_DEG
    )
    sim_deg = frac * (SIM_GRIPPER_MAX_DEG - SIM_GRIPPER_MIN_DEG) + SIM_GRIPPER_MIN_DEG
    out[5] = np.deg2rad(sim_deg)
    return out


def save_png(path: Path, rgb: Any) -> None:
    import cv2
    import numpy as np

    arr = rgb
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr)
    while arr.ndim > 3:
        arr = arr[0]
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def register_dual_env() -> None:
    """The dual-camera subclass: wrist (inherited, mounted + per-step
    updated) + a static third-person camera on the base env's existing
    camera_mount, poses/fov from ThirdCameraEnv's published constants."""
    import numpy as np
    import sapien
    from envs.base_random_env import ThirdCameraEnv
    from envs.lift import LiftCube
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.registration import register_env

    @register_env("SO101LiftCubeDual-v1", max_episode_steps=50)
    class LiftCubeDual(LiftCube):
        @property
        def _default_sensor_configs(self) -> list:
            wrist = super()._default_sensor_configs
            third = CameraConfig(
                "third_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=ThirdCameraEnv.DEFAULT_CAMERA_FOV,
                near=0.01,
                far=100,
                mount=self.camera_mount,
            )
            return [*wrist, third]

        def _initialize_episode(self, env_idx: Any, options: Any) -> None:
            super()._initialize_episode(env_idx, options)
            self.camera_mount.set_pose(
                sapien_utils.look_at(
                    eye=ThirdCameraEnv.DEFAULT_CAMERA_POS,
                    target=ThirdCameraEnv.DEFAULT_CAMERA_TARGET,
                ),
            )

    _ = np  # keep the import local-style consistent


def make(env_id: str, **kw: Any) -> Any:
    import gymnasium as gym

    defaults = {
        "num_envs": 1,
        "control_mode": "pd_joint_pos",
        "sim_backend": "physx_cpu",
        "domain_randomization": False,
    }
    defaults.update(kw)
    return gym.make(env_id, **defaults)


def obs_frames(obs: dict) -> dict:
    return {name: d["rgb"] for name, d in obs["sensor_data"].items()}


def r1_r2_dual_camera(facts: dict) -> None:
    import numpy as np
    import torch

    env = make(
        "SO101LiftCubeDual-v1",
        obs_mode="rgb",
        domain_randomization_config={"apply_overlay": False},
        sensor_configs={"width": 224, "height": 224},
    )
    obs, _ = env.reset(seed=7)
    frames = obs_frames(obs)
    shapes = {k: tuple(v.shape) for k, v in frames.items()}
    for name, frame in frames.items():
        save_png(OUT / f"dual_{name}_224.png", frame)

    # A few hold steps so the wrist mount update path runs, then save a
    # moved-arm wrist frame for the record.
    qpos = env.unwrapped.agent.robot.get_qpos().cpu().numpy()[0]
    hold = qpos.copy()
    for _ in range(5):
        obs, *_ = env.step(torch.tensor(hold[None], dtype=torch.float32))
    save_png(OUT / "dual_base_camera_after_hold.png", obs_frames(obs)["base_camera"])

    # R2 determinism: same seed, same 10-step action sequence, twice.
    def rollout_bytes() -> tuple:
        obs, _ = env.reset(seed=7)
        rng = np.random.default_rng(3)
        acc = []
        target = env.unwrapped.agent.robot.get_qpos().cpu().numpy()[0].copy()
        for _ in range(10):
            target = target + rng.uniform(-0.02, 0.02, size=target.shape)
            obs, *_ = env.step(torch.tensor(target[None], dtype=torch.float32))
            fr = obs_frames(obs)
            acc.append(
                (
                    {k: v.cpu().numpy().tobytes() for k, v in fr.items()},
                    env.unwrapped.agent.robot.get_qpos().cpu().numpy().tobytes(),
                ),
            )
        return acc

    a, b = rollout_bytes(), rollout_bytes()
    bit_equal = all(
        xa[1] == xb[1] and all(xa[0][k] == xb[0][k] for k in xa[0])
        for xa, xb in zip(a, b, strict=True)
    )
    env.close()
    facts["r1_dual_camera"] = {
        "sensor_shapes": {k: list(v) for k, v in shapes.items()},
        "both_224": all(tuple(s[-3:-1]) == (224, 224) for s in shapes.values()),
    }
    facts["r2_determinism_bit_equal"] = bool(bit_equal)


def r3_replay(facts: dict) -> None:
    import numpy as np
    import torch

    trace = np.load(OUT / "replay_trace_ep0.npz")
    actions = trace["actions"]  # (T, 6) servo degrees @ 30 Hz

    env = make("SO101LiftCube-v1", obs_mode="state")
    env.reset(seed=0)
    robot = env.unwrapped.agent.robot
    joint_names = [j.name for j in robot.active_joints]
    low = env.action_space.low.astype(np.float64)
    high = env.action_space.high.astype(np.float64)

    # Teleport to the trace's initial state so the read measures
    # steady-state tracking, not the home-pose transient.
    q0 = to_sim_rad(trace["states"][0])
    robot.set_qpos(torch.tensor(np.clip(q0, low, high)[None], dtype=torch.float32))

    errs, clips = [], 0
    clip_mag = np.zeros(ARM_JOINTS)
    clip_cnt = np.zeros(ARM_JOINTS, dtype=int)
    grip_targets, grip_achieved = [], []
    t0 = time.perf_counter()
    for k in range(0, len(actions), SUBSAMPLE):
        raw = to_sim_rad(actions[k])
        clipped = np.clip(raw, low, high)
        d = np.abs(raw[:ARM_JOINTS] - clipped[:ARM_JOINTS])
        clips += int((d > 1e-9).sum())
        clip_cnt += d > 1e-9
        clip_mag = np.maximum(clip_mag, d)
        env.step(torch.tensor(clipped[None], dtype=torch.float32))
        q = robot.get_qpos().cpu().numpy()[0].astype(np.float64)
        errs.append(np.abs(q[:ARM_JOINTS] - clipped[:ARM_JOINTS]))
        grip_targets.append(clipped[5])
        grip_achieved.append(q[5])
    wall = time.perf_counter() - t0
    env.close()

    errs_arr = np.asarray(errs)
    grip_t = np.asarray(grip_targets)
    grip_a = np.asarray(grip_achieved)
    mid = np.deg2rad((SIM_GRIPPER_MIN_DEG + SIM_GRIPPER_MAX_DEG) / 2)
    cmd_transitions = int(np.abs(np.diff(grip_t > mid)).sum())
    ach_transitions = int(np.abs(np.diff(grip_a > mid)).sum())

    p50 = float(np.median(errs_arr))
    facts["r3_replay"] = {
        "trace": "grasp_demos_v2/merged episode 0 (449 frames @ 30 Hz)",
        "steps_at_10hz": len(errs),
        "joint_names": joint_names,
        "action_space_low": low.round(4).tolist(),
        "action_space_high": high.round(4).tolist(),
        "tracking_rad": {
            "p50": p50,
            "p95": float(np.quantile(errs_arr, 0.95)),
            "max": float(errs_arr.max()),
            "per_joint_p50": np.median(errs_arr, axis=0).round(4).tolist(),
        },
        "arm_limit_clips": int(clips),
        "arm_limit_clip_count_per_joint": clip_cnt.tolist(),
        "arm_limit_clip_max_rad_per_joint": clip_mag.round(4).tolist(),
        "gripper_transitions_commanded": cmd_transitions,
        "gripper_transitions_achieved": ach_transitions,
        "wall_s": round(wall, 2),
        "gate_p50_lt_0.05_rad": bool(p50 < TRACK_P50_GATE_RAD),
        "gate_zero_arm_clips": bool(clips == 0),
    }


def r4_train_smoke(facts: dict) -> None:
    proc = subprocess.run(
        [sys.executable, "train_squint.py", "--help"],
        cwd=SQUINT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    facts["r4_train_squint_help_rc"] = proc.returncode
    if proc.returncode != 0:
        facts["r4_stderr_tail"] = proc.stderr[-500:]


def main() -> None:
    facts: dict = {
        "adapter_constants": {
            "our_gripper_deg": [OUR_GRIPPER_CLOSED_DEG, OUR_GRIPPER_OPEN_DEG],
            "sim_gripper_deg": [SIM_GRIPPER_MIN_DEG, SIM_GRIPPER_MAX_DEG],
            "arm_joints": "plain deg2rad (manipulator.py precedent, no offsets for SO-101)",
            "rate": f"30 Hz -> 10 Hz, subsample every {SUBSAMPLE}rd action",
        },
    }
    register_dual_env()
    r1_r2_dual_camera(facts)
    r3_replay(facts)
    r4_train_smoke(facts)
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
