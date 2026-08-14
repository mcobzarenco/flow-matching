"""Squint SO-101 twin preflight (queue item squint-twin-preflight, 08-14).

CPU-only feasibility probe of github.com/aalmuzairee/squint (MIT):
  1. all 8 SO101*-v1 envs register + step headless (state obs, pd_joint_pos)
  2. absolute-joint controller end-to-end: scripted hold + random walk
  3. wrist frame at 224x224, apply_overlay=False (+ overlay=True contrast)
  4. third-person frame at 224x224 via the DefaultCameraEnv alias flip
     (separate process: `--mode third`)
  5. per-step wall time at 1 env CPU, state mode and rgb mode

Run from the squint checkout with its isolated venv:
  cd ~/squint && VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    CUDA_VISIBLE_DEVICES= .venv/bin/python \
    ~/flow-matching/fontaine/scripts/squint_preflight.py --mode main
  ... --mode third

GPU guard: CUDA_VISIBLE_DEVICES empty + lavapipe Vulkan ICD -> the owner
reserve stays at 0 MiB; physics is PhysX CPU (num_envs=1).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

OUT = Path("/home/ubuntu/flow-matching/outputs/squint_preflight")
OUT.mkdir(parents=True, exist_ok=True)

ENV_IDS = [
    f"SO101{task}{obj}-v1"
    for task in ("Reach", "Lift", "Place", "Stack")
    for obj in ("Cube", "Can")
]

NO_DR = {"domain_randomization": False}
NO_OVERLAY = {"apply_overlay": False}


def make(env_id: str, **kw: Any) -> Any:
    import gymnasium as gym

    defaults = {
        "num_envs": 1,
        "control_mode": "pd_joint_pos",
        "sim_backend": "physx_cpu",
    }
    defaults.update(kw)
    return gym.make(env_id, **defaults)


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


def sensor_rgb(obs: dict) -> Any:
    return obs["sensor_data"]["base_camera"]["rgb"]


def mode_main() -> None:
    import envs  # noqa: F401  (registers SO101*-v1)
    import numpy as np
    import torch

    facts = {"envs": {}, "controller": {}, "timing": {}, "render": {}}

    # --- 1. registration + headless step, all 8 envs (state obs) ---
    for env_id in ENV_IDS:
        t0 = time.perf_counter()
        env = make(env_id, obs_mode="state", **NO_DR)
        t_make = time.perf_counter() - t0
        obs, info = env.reset(seed=0)
        rows = {}
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, _terminated, _truncated, info = env.step(action)
        rows["obs_shape"] = (
            list(np.asarray(obs.shape if hasattr(obs, "shape") else ()).tolist())
            if hasattr(obs, "shape")
            else str(type(obs))
        )
        rows["action_space"] = {
            "shape": list(env.action_space.shape),
            "low": np.round(env.action_space.low, 4).tolist(),
            "high": np.round(env.action_space.high, 4).tolist(),
        }
        rows["reward_sample"] = float(torch.as_tensor(reward).flatten()[0])
        rows["info_keys"] = sorted(info.keys())
        rows["success_in_info"] = "success" in info
        rows["make_seconds"] = round(t_make, 2)
        facts["envs"][env_id] = rows
        env.close()
        print(f"[step-ok] {env_id}  make={t_make:.1f}s  info={sorted(info.keys())}")

    # --- 2. absolute-joint controller: hold + random walk (LiftCube) ---
    env = make("SO101LiftCube-v1", obs_mode="state", **NO_DR)
    obs, info = env.reset(seed=0)
    agent = env.unwrapped.agent
    qpos0 = agent.robot.get_qpos()[0].cpu().numpy().copy()
    qlim = agent.robot.get_qlimits()[0].cpu().numpy()

    # hold: command the current qpos for 30 steps; drift should be ~0
    hold_action = qpos0.astype(np.float32)
    max_drift = 0.0
    for _ in range(30):
        obs, reward, term, trunc, info = env.step(hold_action)
        q = agent.robot.get_qpos()[0].cpu().numpy()
        max_drift = max(max_drift, float(np.abs(q - qpos0).max()))
    facts["controller"]["qpos0"] = np.round(qpos0, 4).tolist()
    facts["controller"]["qlimits"] = np.round(qlim, 4).tolist()
    facts["controller"]["hold_max_drift_rad"] = round(max_drift, 5)

    # random walk: absolute targets = qpos + cumulative bounded deltas;
    # verify tracking, 50-step truncation, success/info plumbing
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=1)
    target = agent.robot.get_qpos()[0].cpu().numpy().copy()
    track_errs, steps = [], 0
    term = trunc = False
    while not (term or trunc):
        target = target + rng.uniform(-0.04, 0.04, size=target.shape)
        target = np.clip(target, qlim[:, 0], qlim[:, 1])
        obs, reward, term, trunc, info = env.step(target.astype(np.float32))
        q = agent.robot.get_qpos()[0].cpu().numpy()
        track_errs.append(float(np.abs(q - target).max()))
        steps += 1
        term = bool(torch.as_tensor(term).flatten()[0])
        trunc = bool(torch.as_tensor(trunc).flatten()[0])
    facts["controller"]["walk_steps_to_truncation"] = steps
    facts["controller"]["walk_terminated"] = term
    facts["controller"]["walk_truncated"] = trunc
    facts["controller"]["walk_final_info"] = {
        k: (
            float(torch.as_tensor(v).flatten()[0])
            if hasattr(v, "flatten") or isinstance(v, (int, float, bool))
            else str(v)
        )
        for k, v in info.items()
        if k != "final_observation"
    }
    p50 = round(float(np.median(track_errs)), 4)
    facts["controller"]["walk_track_err_p50_rad"] = p50
    facts["controller"]["walk_track_err_max_rad"] = round(float(np.max(track_errs)), 4)

    # --- 3. state-mode step timing (same env, 200 steps) ---
    obs, info = env.reset(seed=2)
    action = qpos0.astype(np.float32)
    t0 = time.perf_counter()
    n = 200
    for i in range(n):
        env.step(action)
        if (i + 1) % 50 == 0:
            env.reset(seed=2)
    dt = time.perf_counter() - t0
    facts["timing"]["state_ms_per_step"] = round(1000 * dt / n, 2)
    env.close()

    # --- 4. wrist render at 224, overlay off + on; rgb step timing ---
    # NOTE: the overlay hook silently no-ops unless the obs mode includes
    # BOTH rgb and segmentation (base_random_env._get_obs_sensor_data), so
    # the overlay-on contrast frame must use obs_mode="rgb+segmentation".
    for overlay, tag in ((False, "off"), (True, "on")):
        env = make(
            "SO101LiftCube-v1",
            obs_mode="rgb+segmentation" if overlay else "rgb",
            sensor_configs={"width": 224, "height": 224},
            domain_randomization=False,
            domain_randomization_config={"apply_overlay": overlay},
        )
        obs, info = env.reset(seed=0)
        rgb = sensor_rgb(obs)
        facts["render"][f"wrist_overlay_{tag}_shape"] = list(rgb.shape)
        save_png(OUT / f"wrist_224_overlay_{tag}.png", rgb)
        if not overlay:
            t0 = time.perf_counter()
            n = 50
            for _ in range(n):
                obs, *_ = env.step(hold_action)
            dt = time.perf_counter() - t0
            facts["timing"]["rgb224_ms_per_step"] = round(1000 * dt / n, 2)
        env.close()
        print(f"[render-ok] wrist 224 overlay={tag}")

    (OUT / "facts_main.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


def mode_third() -> None:
    # Third-person camera is the repo's documented module-level switch
    # (CAMERA_TYPE in envs/base_random_env.py) — a per-process constant,
    # not a gym.make kwarg. An in-process alias flip does NOT work: any
    # `import envs.<sub>` runs the package __init__ first, which binds
    # DefaultCameraEnv into every task class before user code can touch
    # it. The driver seds CAMERA_TYPE to "third" before this mode and
    # reverts after; we assert the flip actually took.
    import envs  # noqa: F401

    facts = {}
    env = make(
        "SO101LiftCube-v1",
        obs_mode="rgb",
        sensor_configs={"width": 224, "height": 224},
        domain_randomization=False,
        domain_randomization_config={"apply_overlay": False},
    )
    obs, _info = env.reset(seed=0)
    rgb = sensor_rgb(obs)
    facts["third_shape"] = list(rgb.shape)
    base = env.unwrapped.__class__.__mro__
    facts["mro_has_third"] = any("ThirdCameraEnv" in c.__name__ for c in base)
    assert facts["mro_has_third"], (
        "CAMERA_TYPE flip did not take — sed envs/base_random_env.py first"
    )
    save_png(OUT / "third_224_overlay_off.png", rgb)
    t0 = time.perf_counter()
    n = 50
    import numpy as np

    action = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32)
    for _ in range(n):
        env.step(action)
    facts["third_rgb224_ms_per_step"] = round(1000 * (time.perf_counter() - t0) / n, 2)
    env.close()
    (OUT / "facts_third.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["main", "third"], required=True)
    args = ap.parse_args()
    {"main": mode_main, "third": mode_third}[args.mode]()
