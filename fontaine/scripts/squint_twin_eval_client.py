"""Squint twin policy-eval client — the Gate-1/2 rollout harness
(queue item squint-gate2-harness; pre-reg
posts/2026-08-22-prereg-squint-twin-screen.md).

Two-process split, and it is the DEPLOY path on purpose: the policy is
served by ``bijou.policy_server`` in the MAIN venv (GPU, one arm per
server run, port 8145 — the owner's rig server owns 8144 and is never
touched), and THIS client runs in the twin's venv where the ManiSkill
env lives, speaking the raw wire protocol (schema_version 1, GET /spec
+ POST /predict, base64-JPEG frames). No bijou imports on this side —
the venvs never have to merge. JPEG on the wire is the same lossy path
the rig rides; identical across arms, recorded not gated.

Per replan (adapter frozen at finalization): wrist + front 224 frames +
servo-degree state -> /predict -> [chunk, 6] servo-degree 30 Hz chunk
-> take the first 30 rows, subsample every 3rd -> forward adapter
(deg2rad + gripper affine ours [0,41.69] -> twin sim [-10,120] deg) ->
clip to action space -> 10 pd_joint_pos env steps. 5 replans x 10 steps
covers the 50-step twin horizon. Env: the dual-camera 224 raw DR-off
subclass, physx_cpu (the R2/R3 receipt backend), num_envs 1, paired
seeds seed0..seed0+n-1 identical across arms.

Per-episode banked: success (final-step, the honest end-state read),
per-step predicates reached_object / is_item_grasped / item_lifted /
success (the KM/KS co-primary consumes these), qpos + target traces.

Stats row: --stats-repo-id picks the worn row from /spec's
per_dataset_stats (loud list on a miss). Adapted arms wear their own
recomputed twin row; the unadapted record-only rider wears the sim100
worn-row rule (the rig merged row).

Run (server already up in the main venv):
  cd ~/squint && PYTHONPATH=. .venv/bin/python \
    ~/flow-matching/fontaine/scripts/squint_twin_eval_client.py \
    --task lift --arm-name adapt_onerig \
    --stats-repo-id fontaine/squint_twin_demos_v1 \
    --num-seeds 100
"""

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

OUT_ROOT = Path("/home/ubuntu/flow-matching/outputs/squint_screen/eval")
WIRE_SCHEMA_VERSION = 1

# Frozen adapter constants (finalization amendment).
OUR_OPEN = 41.69
SIM_MIN, SIM_MAX = -10.0, 120.0
SUBSAMPLE = 3
EXEC_STEPS = 10  # twin env-steps per replan (30 chunk rows / 3)
HORIZON = 50
PRED_KEYS = ("reached_object", "is_item_grasped", "item_lifted", "success")

TASKS = {
    "lift": {
        "dual_id": "SO101LiftCubeDualC-v1",
        "instruction": "Pick up the red cube.",
    },
    "place": {
        "dual_id": "SO101PlaceCubeDualC-v1",
        "instruction": "Pick up the red cube and place it in the bin.",
    },
}


def http_json(url: str, payload: "dict | None" = None, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def encode_jpeg_b64(rgb: "object") -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok, "JPEG encode failed"
    return base64.b64encode(buf.tobytes()).decode()


def to_sim_rad(a_deg: "object") -> "object":
    import numpy as np

    out = np.deg2rad(np.asarray(a_deg, dtype=np.float64))
    frac = np.asarray(a_deg, dtype=np.float64)[..., 5] / OUR_OPEN
    out[..., 5] = np.deg2rad(frac * (SIM_MAX - SIM_MIN) + SIM_MIN)
    return out


def from_sim_rad(q_rad: "object") -> "object":
    import numpy as np

    deg = np.rad2deg(np.asarray(q_rad, dtype=np.float64))
    frac = (deg[..., 5] - SIM_MIN) / (SIM_MAX - SIM_MIN)
    deg[..., 5] = frac * OUR_OPEN
    return deg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--arm-name", required=True, help="output stem, e.g. adapt_onerig")
    ap.add_argument("--server", default="http://127.0.0.1:8145")
    ap.add_argument("--stats-repo-id", required=True)
    ap.add_argument("--num-seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--replans", type=int, default=5)
    ap.add_argument("--sample-steps", type=int, default=10)
    ap.add_argument("--method", default="euler")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gymnasium as gym
    import numpy as np
    import torch
    from squint_expert_collect import register_dual_envs

    spec = http_json(f"{args.server}/spec")
    if spec.get("schema_version") != WIRE_SCHEMA_VERSION:
        msg = f"wire schema {spec.get('schema_version')} != {WIRE_SCHEMA_VERSION}"
        raise SystemExit(msg)
    tables = spec["per_dataset_stats"]
    if args.stats_repo_id not in tables:
        msg = (
            f"--stats-repo-id {args.stats_repo_id!r} not in the checkpoint's "
            f"per-dataset table — available: {sorted(tables)}"
        )
        raise SystemExit(msg)
    stats = tables[args.stats_repo_id]
    chunk_size = int(spec["chunk_size"])
    print(
        f"[eval:{args.task}:{args.arm_name}] serving {spec['checkpoint']} "
        f"@ step {spec['step']} (family {spec['family']}, chunk {chunk_size}), "
        f"worn row {args.stats_repo_id}",
        flush=True,
    )

    register_dual_envs()
    cfg = TASKS[args.task]
    env = gym.make(
        cfg["dual_id"],
        num_envs=1,
        control_mode="pd_joint_pos",
        obs_mode="rgb",
        sim_backend="physx_cpu",
        domain_randomization=False,
        domain_randomization_config={"apply_overlay": False},
        sensor_configs={"width": 224, "height": 224},
    )
    base = env.unwrapped
    low = env.action_space.low.astype(np.float64)
    high = env.action_space.high.astype(np.float64)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows, t_start = [], time.perf_counter()
    for seed in range(args.seed0, args.seed0 + args.num_seeds):
        obs, _ = env.reset(seed=seed)
        preds = {k: [] for k in PRED_KEYS}
        qpos_tr, tgt_tr, latencies = [], [], []
        info = {}
        for replan in range(args.replans):
            frames = {
                "wrist": obs["sensor_data"]["base_camera"]["rgb"].cpu().numpy()[0],
                "front": obs["sensor_data"]["third_camera"]["rgb"].cpu().numpy()[0],
            }
            qpos = base.agent.robot.get_qpos().cpu().numpy()[0].astype(np.float64)
            state_deg = from_sim_rad(qpos)
            t0 = time.perf_counter()
            resp = http_json(
                f"{args.server}/predict",
                {
                    "task": cfg["instruction"],
                    "stats": stats,
                    "state": state_deg.tolist(),
                    "images": {k: encode_jpeg_b64(v) for k, v in frames.items()},
                    "camera_kinds": {"wrist": "wrist", "front": "front"},
                    "index": replan,
                    "options": {
                        "num_steps": args.sample_steps,
                        "method": args.method,
                    },
                },
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            chunk = np.asarray(resp["actions"], dtype=np.float64)
            exec_rows = chunk[: EXEC_STEPS * SUBSAMPLE : SUBSAMPLE]
            for act_deg in exec_rows:
                raw = to_sim_rad(act_deg)
                clipped = np.clip(raw, low, high)
                obs, _, _, _, info = env.step(
                    torch.tensor(clipped[None], dtype=torch.float32),
                )
                tgt_tr.append(clipped.tolist())
                qpos_tr.append(
                    base.agent.robot.get_qpos().cpu().numpy()[0, :6].tolist(),
                )
                for k in PRED_KEYS:
                    preds[k].append(bool(info[k].reshape(-1)[0]))
        rows.append(
            {
                "seed": seed,
                "success": bool(info["success"].reshape(-1)[0]),
                "predicates": preds,
                "first_true_step": {
                    k: (preds[k].index(True) if True in preds[k] else None)
                    for k in PRED_KEYS
                },
                "predict_ms_mean": float(np.mean(latencies)),
                "qpos_trace": qpos_tr,
                "target_trace": tgt_tr,
            },
        )
        done = seed - args.seed0 + 1
        if done % 10 == 0 or done == args.num_seeds:
            n_succ = sum(r["success"] for r in rows)
            print(
                f"[eval:{args.task}:{args.arm_name}] {done}/{args.num_seeds} seeds, "
                f"{n_succ} successes, {(time.perf_counter() - t_start) / done:.1f} s/seed",
                flush=True,
            )
    env.close()

    out = OUT_ROOT / f"{args.arm_name}_{args.task}.json"
    out.write_text(
        json.dumps(
            {
                "arm": args.arm_name,
                "task": args.task,
                "instruction": cfg["instruction"],
                "checkpoint": spec["checkpoint"],
                "step": spec["step"],
                "server_git_rev": spec.get("git_rev"),
                "stats_repo_id": args.stats_repo_id,
                "config": {
                    "num_seeds": args.num_seeds,
                    "seed0": args.seed0,
                    "replans": args.replans,
                    "sample_steps": args.sample_steps,
                    "method": args.method,
                    "sim_backend": "physx_cpu",
                    "horizon": HORIZON,
                    "exec_steps": EXEC_STEPS,
                    "subsample": SUBSAMPLE,
                },
                "successes": sum(r["success"] for r in rows),
                "rows": rows,
            },
            indent=2,
        ),
    )
    print(f"[eval:{args.task}:{args.arm_name}] wrote {out}")


if __name__ == "__main__":
    main()
