#!/usr/bin/env bash
# pi0.5 robot client (laptop, arm + cameras attached).
# Prereqs, each in its own terminal:
#   1. tunnel:  ssh -N -L 8080:localhost:8080 ubuntu@192.222.54.70
#   2. server:  (on box) bash pi05_server.sh
#   3. this script
#
# The checkpoint path is a BOX path (the server loads it); pass the run
# dir and step: bash pi05_client.sh pi05_ft_rig_v2_frozen 000500
# Cameras are named with pi0 slot names directly — see README (async
# server resizes by policy-feature key before the checkpoint's rename
# processor runs). video6 = front → base_0, video4 = wrist → left_wrist_0,
# the same physical mapping training used.
set -euo pipefail
RUN="${1:-pi05_ft_rig_v2_frozen}"
STEP="${2:-last}"
CKPT="/home/ubuntu/flow-matching/outputs/train/${RUN}/checkpoints/${STEP}/pretrained_model"
cd ~/w/flow-matching/baselines
.venv/bin/python -m lerobot.async_inference.robot_client \
  --policy_type=pi05 \
  --pretrained_name_or_path="$CKPT" \
  --server_address=localhost:8080 \
  --policy_device=cuda \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower0 \
  --robot.max_relative_target=80 \
  --robot.cameras='{ base_0_rgb: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}, left_wrist_0_rgb: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}' \
  --task="Pick up the toy boat and place it on the wooden disk." \
  --fps=30 \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5
