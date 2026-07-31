"""Replay the async policy-server inference pipeline for pi05 WITHOUT a
robot: synthetic observation -> raw_observation_to_observation ->
preprocessor -> predict_action_chunk -> postprocessor. Mirrors
policy_server.SendPolicyInstructions + _predict_action_chunk so serving
failures reproduce here instead of through robot client round trips.

Usage (box): CUDA_VISIBLE_DEVICES=1 .venv/bin/python probe_pi05_infer.py \
    /home/ubuntu/flow-matching/outputs/train/<run>/checkpoints/<step>/pretrained_model
"""

import sys
import time

import numpy as np
import torch
from lerobot.async_inference.helpers import raw_observation_to_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy

MOTORS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
CAMERAS = ["base_0_rgb", "left_wrist_0_rgb"]
TASK = "Pick up the toy boat and place it on the wooden disk."


def main() -> int:
    pretrained = sys.argv[1]
    device = "cuda"

    policy = PI05Policy.from_pretrained(pretrained)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=pretrained,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )

    # What map_robot_keys_to_lerobot_features(robot) produces for our rig
    # (hw_to_dataset_features with use_video=False).
    lerobot_features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": MOTORS,
        },
    }
    for cam in CAMERAS:
        lerobot_features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }

    # Raw robot observation: motor floats + HWC uint8 frames + task.
    rng = np.random.default_rng(0)
    raw_obs: dict[str, object] = {
        m: float(v) for m, v in zip(MOTORS, rng.normal(0, 30, 6), strict=True)
    }
    for cam in CAMERAS:
        raw_obs[cam] = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    raw_obs["task"] = TASK

    observation = raw_observation_to_observation(
        raw_obs,
        lerobot_features,
        policy.config.image_features,
    )
    observation = preprocessor(observation)

    chunk = torch.empty(0)
    for attempt in range(2):  # first call includes CUDA warmup
        start = time.perf_counter()
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(observation)
        elapsed = time.perf_counter() - start
        print(
            f"predict_action_chunk[{attempt}]: {elapsed:.3f}s shape={tuple(chunk.shape)}"
        )

    if chunk.ndim != 3:
        chunk = chunk.unsqueeze(0)
    processed = [postprocessor(chunk[:, i, :]) for i in range(chunk.shape[1])]
    actions = torch.stack(processed, dim=1).squeeze(0).cpu()
    print(f"postprocessed: shape={tuple(actions.shape)}")
    print(f"first action (degrees): {actions[0].tolist()}")
    print(f"chunk range: min={actions.min():.2f} max={actions.max():.2f}")
    print("PROBE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
