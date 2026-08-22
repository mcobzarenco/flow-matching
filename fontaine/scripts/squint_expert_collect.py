"""Squint twin demo collection — SAC-expert rollouts re-rendered through
the frozen adapter config (Gate-1 demo source for the qualification
screen, posts/2026-08-22-prereg-squint-twin-screen.md).

Two stages per task, both GPU (sim_backend "gpu"; the H100 window the
exec item priced):

  rollout  — load the trained expert (runs/{exp}/ckpt.pt, train_squint
             recipe DR-off) and roll batched episodes in its NATIVE env
             config (rgb+segmentation 128->16 + jitter, target-delta
             control). Per episode record: reset seed, the full initial
             env state dict, the controller's ABSOLUTE target qpos per
             step (radians — target-delta control is PDJointPosController
             with use_target/use_delta, so `_target_qpos` is the
             pd_joint_pos-equivalent command), qpos trace, and
             success_at_end. Keep candidates until --n-candidates
             successes are banked.
  rerender — replay each success through OUR adapter env: the dual-camera
             (wrist base_camera + third_camera) 224x224 raw DR-off
             subclass, control_mode pd_joint_pos, same seed + initial
             state set via set_state_dict, actions = the recorded
             absolute targets. Capture both camera streams, servo-state
             trace, per-step honest predicates (reached_object /
             is_item_grasped / item_lifted / success — the labeled-rollout
             corpus, idea #6). The FINAL success label is the re-render's
             own end-state success; episodes that lose success in replay
             are dropped (keep-rate is a receipt). First --keep survivors
             become the dataset; conversion to LeRobot is
             squint_to_lerobot.py (main venv).

Run from the squint checkout with its venv, GPU visible:
  cd ~/squint && PYTHONPATH=. .venv/bin/python \
    ~/flow-matching/fontaine/scripts/squint_expert_collect.py \
    --task lift --stage all --ckpt runs/squint_expert_lift/ckpt.pt
"""

import argparse
import json
import time
from pathlib import Path

OUT_ROOT = Path("/home/ubuntu/flow-matching/outputs/squint_screen")

TASKS = {
    "lift": {
        "env_id": "SO101LiftCube-v1",
        "dual_id": "SO101LiftCubeDualC-v1",
        "horizon": 50,
    },
    "place": {
        "env_id": "SO101PlaceCube-v1",
        "dual_id": "SO101PlaceCubeDualC-v1",
        "horizon": 50,
    },
}
SEED0 = {"lift": 10_000, "place": 20_000}


def register_dual_envs() -> None:
    """Dual-camera subclasses for both tasks — the preflight-2 R1 recipe
    (wrist inherited, static third_camera on the base env's camera_mount
    at ThirdCameraEnv's published pose/FOV), registered fresh ids."""
    import sapien
    from envs.base_random_env import ThirdCameraEnv
    from envs.lift import LiftCube
    from envs.place import PlaceCube
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.registration import register_env

    def dualize(base_cls: type, env_id: str, max_steps: int) -> type:
        @register_env(env_id, max_episode_steps=max_steps)
        class Dual(base_cls):
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

            def _initialize_episode(self, env_idx: "object", options: "object") -> None:
                super()._initialize_episode(env_idx, options)
                self.camera_mount.set_pose(
                    sapien_utils.look_at(
                        eye=ThirdCameraEnv.DEFAULT_CAMERA_POS,
                        target=ThirdCameraEnv.DEFAULT_CAMERA_TARGET,
                    ),
                )

        return Dual

    dualize(LiftCube, TASKS["lift"]["dual_id"], TASKS["lift"]["horizon"])
    dualize(PlaceCube, TASKS["place"]["dual_id"], TASKS["place"]["horizon"])


def get_abs_targets(base_env: "object") -> "object":
    """Controller absolute target qpos (B, n_joints) in active_joints
    order — asserts the single-controller layout the smoke verified."""
    import torch

    ctrl = base_env.agent.controller
    assert not hasattr(ctrl, "controllers"), "expected single flat controller"
    t = ctrl._target_qpos
    t = t.clone() if isinstance(t, torch.Tensor) else torch.as_tensor(t).clone()
    names = [j.name for j in ctrl.joints]
    active = [j.name for j in base_env.agent.robot.active_joints]
    assert names == active, f"joint order mismatch: {names} vs {active}"
    return t


def state_slice(sd: "object", i: int) -> "object":
    import torch

    if isinstance(sd, dict):
        return {k: state_slice(v, i) for k, v in sd.items()}
    if isinstance(sd, torch.Tensor):
        return sd[i : i + 1].clone()
    return sd


def state_cat(sds: list) -> "object":
    import torch

    first = sds[0]
    if isinstance(first, dict):
        return {k: state_cat([s[k] for s in sds]) for k in first}
    if isinstance(first, torch.Tensor):
        return torch.cat(sds, dim=0)
    return first


def stage_rollout(
    task: str,
    ckpt: str,
    n_candidates: int,
    batch: int,
    *,
    smoke: bool = False,
) -> None:
    import gymnasium as gym
    import numpy as np
    import torch
    import utils as squtils
    from mani_skill.utils.wrappers.flatten import (
        FlattenActionSpaceWrapper,
        FlattenRGBDObservationWrapper,
    )
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from train_squint import Actor, CNNEncoder

    cfg = TASKS[task]
    out = OUT_ROOT / task
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    envs = gym.make(
        cfg["env_id"],
        num_envs=batch,
        obs_mode="rgb+segmentation",
        render_mode="all",
        sim_backend="gpu",
        sensor_configs={"width": 128, "height": 128},
    )
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=True)
    envs = squtils.DownsampleObsWrapper(envs, target_size=16)
    envs = squtils.ColorJitterWrapper(envs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(
        envs,
        batch,
        ignore_terminations=True,
        record_metrics=True,
    )
    base = envs.unwrapped

    n_act = int(np.prod(envs.unwrapped.single_action_space.shape))
    n_channels = envs.unwrapped.single_observation_space["rgb"].shape[2]
    n_state = int(np.prod(envs.unwrapped.single_observation_space["state"].shape))
    encoder = CNNEncoder(n_obs=(16, 16, n_channels), device=device)
    actor = Actor(
        envs,
        n_obs=encoder.repr_dim,
        n_state=n_state,
        n_act=n_act,
        device=device,
    )
    ck = torch.load(ckpt, map_location=device)
    encoder.load_state_dict(ck["encoder"])
    actor.load_state_dict(ck["actor"])
    encoder.eval()
    actor.eval()
    print(f"[rollout:{task}] expert loaded from {ckpt} @ step {ck['global_step']}")

    horizon = cfg["horizon"]
    kept, round_idx, t0 = [], 0, time.perf_counter()
    attempts = 0
    while len(kept) < n_candidates and round_idx < 8:
        seeds = [SEED0[task] + round_idx * batch + i for i in range(batch)]
        obs, _ = envs.reset(seed=seeds)
        init_state = base.get_state_dict()
        targets = torch.zeros(batch, horizon, 6)
        qpos_tr = torch.zeros(batch, horizon, 6)
        infos = {}
        for t in range(horizon):
            with torch.no_grad():
                act = actor.get_eval_action(encoder(obs["rgb"]), obs["state"])
            obs, _, _, _, infos = envs.step(act)
            targets[:, t] = get_abs_targets(base).cpu()
            qpos_tr[:, t] = base.agent.robot.get_qpos()[:, :6].cpu()
        success = infos["final_info"]["episode"]["success_at_end"].reshape(-1).cpu()
        if smoke:
            success = torch.ones_like(success.float()).bool()
        attempts += batch
        # The vector env TRUNCATES + AUTO-RESETS on the final step: the
        # post-step target/qpos of step horizon-1 are the NEW episode's
        # reset state, not the episode's last command (04:3xZ 08-22
        # incident: the reset pose's OPEN gripper replayed as a
        # release -> 0/130 re-render successes). Drop that row; the
        # banked episode is horizon-1 real steps.
        targets = targets[:, :-1]
        qpos_tr = qpos_tr[:, :-1]
        for i in torch.nonzero(success).reshape(-1).tolist():
            if len(kept) >= n_candidates:
                break
            kept.append(
                {
                    "seed": seeds[i],
                    "targets": targets[i].numpy(),
                    "qpos": qpos_tr[i].numpy(),
                    "state": state_slice(init_state, i),
                },
            )
        print(
            f"[rollout:{task}] round {round_idx}: {int(success.sum())}/{batch} success, "
            f"banked {len(kept)}/{n_candidates}",
            flush=True,
        )
        round_idx += 1
    envs.close()

    if not kept:
        (out / "rollout_facts.json").write_text(
            json.dumps({"task": task, "attempts": attempts, "candidates": 0}),
        )
        print(f"[rollout:{task}] NO successes banked after {attempts} attempts")
        return

    np.savez_compressed(
        out / "rollout_candidates.npz",
        seeds=np.array([k["seed"] for k in kept]),
        targets=np.stack([k["targets"] for k in kept]),
        qpos=np.stack([k["qpos"] for k in kept]),
    )
    torch.save([k["state"] for k in kept], out / "rollout_states.pt")
    facts = {
        "task": task,
        "ckpt_step": int(ck["global_step"]),
        "attempts": attempts,
        "candidates": len(kept),
        "rollout_success_rate": len(kept) / max(attempts, 1),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (out / "rollout_facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts))


def stage_rerender(task: str, keep: int, batch: int, *, smoke: bool = False) -> None:
    import gymnasium as gym
    import numpy as np
    import torch

    cfg = TASKS[task]
    out = OUT_ROOT / task
    eps_dir = out / "episodes"
    eps_dir.mkdir(parents=True, exist_ok=True)

    if not (out / "rollout_candidates.npz").exists():
        print(f"[rerender:{task}] no candidates file — skipping")
        return
    data = np.load(out / "rollout_candidates.npz")
    states = torch.load(out / "rollout_states.pt", weights_only=False)
    seeds, targets, qpos_roll = data["seeds"], data["targets"], data["qpos"]
    n = len(seeds)
    # Sized from the DATA, not the env registration: rollout banks
    # horizon-1 real steps (the final row is the truncation-reset
    # artifact, dropped there).
    horizon = targets.shape[1]

    env = gym.make(
        cfg["dual_id"],
        num_envs=batch,
        control_mode="pd_joint_pos",
        obs_mode="rgb",
        sim_backend="gpu",
        domain_randomization=False,
        domain_randomization_config={"apply_overlay": False},
        sensor_configs={"width": 224, "height": 224},
    )
    base = env.unwrapped

    kept_ids, divergences, t0 = [], [], time.perf_counter()
    for lo in range(0, n, batch):
        idx = list(range(lo, min(lo + batch, n)))
        pad = [idx[-1]] * (batch - len(idx))
        rows = idx + pad
        env.reset(seed=[int(seeds[i]) for i in rows])
        base.set_state_dict(state_cat([states[i] for i in rows]))
        obs = base.get_obs()

        frames_w = np.zeros((batch, horizon, 224, 224, 3), dtype=np.uint8)
        frames_f = np.zeros_like(frames_w)
        qpos_tr = np.zeros((batch, horizon, 6), dtype=np.float32)
        preds = {
            k: np.zeros((batch, horizon), dtype=bool)
            for k in ("reached_object", "is_item_grasped", "item_lifted", "success")
        }
        info = {}
        for t in range(horizon):
            frames_w[:, t] = obs["sensor_data"]["base_camera"]["rgb"].cpu().numpy()
            frames_f[:, t] = obs["sensor_data"]["third_camera"]["rgb"].cpu().numpy()
            qpos_tr[:, t] = base.agent.robot.get_qpos()[:, :6].cpu().numpy()
            act = torch.tensor(
                targets[rows, t],
                dtype=torch.float32,
                device=base.device,
            )
            obs, _, _, _, info = env.step(act)
            for k, buf in preds.items():
                if k in info:
                    buf[:, t] = info[k].reshape(-1).cpu().numpy()
        succ = info["success"].reshape(-1).cpu().numpy()
        if smoke:
            succ = np.ones_like(succ, dtype=bool)
        div = np.abs(qpos_tr[:, :, :5] - qpos_roll[rows][:, :, :5])
        for j, i in enumerate(idx):
            divergences.append(float(np.median(div[j])))
            if not succ[j]:
                continue
            kept_ids.append(i)
            np.savez_compressed(
                eps_dir / f"ep_{i:04d}.npz",
                seed=seeds[i],
                targets=targets[i],
                qpos=qpos_tr[j],
                wrist=frames_w[j],
                front=frames_f[j],
                **{f"pred_{k}": v[j] for k, v in preds.items()},
            )
        print(
            f"[rerender:{task}] {lo + len(idx)}/{n} replayed, kept {len(kept_ids)}",
            flush=True,
        )
    env.close()

    final = kept_ids[:keep]
    facts = {
        "task": task,
        "candidates": n,
        "rerender_success": len(kept_ids),
        "keep_rate": len(kept_ids) / max(n, 1),
        "kept_for_dataset": len(final),
        "replay_qpos_divergence_rad": {
            "p50": float(np.median(divergences)),
            "p95": float(np.quantile(divergences, 0.95)),
        },
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (out / "kept_ids.json").write_text(json.dumps(final))
    (out / "rerender_facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--stage", choices=["rollout", "rerender", "all"], default="all")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-candidates", type=int, default=130)
    ap.add_argument("--keep", type=int, default=100)
    ap.add_argument("--rollout-batch", type=int, default=128)
    ap.add_argument("--rerender-batch", type=int, default=16)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="bank/keep episodes regardless of success (pipeline smoke only)",
    )
    args = ap.parse_args()

    register_dual_envs()
    if args.stage in ("rollout", "all"):
        assert args.ckpt, "--ckpt required for rollout"
        stage_rollout(
            args.task,
            args.ckpt,
            args.n_candidates,
            args.rollout_batch,
            smoke=args.smoke,
        )
    if args.stage in ("rerender", "all"):
        stage_rerender(args.task, args.keep, args.rerender_batch, smoke=args.smoke)


if __name__ == "__main__":
    main()
