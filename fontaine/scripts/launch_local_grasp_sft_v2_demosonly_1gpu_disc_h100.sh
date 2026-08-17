#!/usr/bin/env bash
# PREPARED (owner-gated): the 1-GPU drift discriminator — LOCAL H100
# adaptation of the frozen box launcher
# (fontaine/scripts/box/launch_box_grasp_sft_v2_demosonly_1gpu_discriminator.sh).
# The box was deleted by the owner 2026-08-17 18:09Z; this is the same
# run re-pointed at the only remaining GPU. Read rule and verdict
# bounds are UNCHANGED (frozen in sft_drift_saga_charts.py:
# HEALTHY_BOUND=+0.30, DRIFT_FRACTION=0.5 => drift_min +1.0158).
#
# Deltas vs the frozen box header — platform only, zero recipe deltas:
#   1. Hardware: 1x H100 80GB (local) instead of 1x A100 80GB (box
#      GPU 0). Same 80 GiB budget; recipe class measured to fit
#      (micro-12 + act-ckpt). Pace: box estimate was ~25-32 s/step;
#      H100 should be at or under that — first-poll util/rate check
#      per standing rule, plus free -g (1-GPU DataLoader carries the
#      full batch-96 buffers; host has 221 GB).
#   2. Data: ~/datasets/fontaine/grasp_demos_v2/merged is the local
#      snapshot of mcobzarenco/fontaine-grasp-demos-v2 (the HF mirror
#      verified ~= the box merged copy at evacuation, 17:20Z 08-17).
#   3. Guard: aborts if ANY compute process holds the GPU — the owner
#      policy-server (port 8144) silently claims the H100 for rig
#      serving and must never be preempted or killed.
# Everything below the guard is the box script verbatim (same
# hyperparameters, same default seed, same eff-96 = micro-12 x 8
# backward chunks, same --recompute-stats / --image-augment 0.8 /
# --insulate-flow joint objective, same init checkpoint).
#
# LAUNCH ONLY on the owner GO, pre-reg posted first:
#   systemd-run --user --unit=fontaine-demosonly-1gpu-disc \
#     --setenv=HOME=/home/ubuntu \
#     --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
#     --working-directory=/home/ubuntu/flow-matching \
#     bash ~/flow-matching/fontaine/scripts/launch_local_grasp_sft_v2_demosonly_1gpu_disc_h100.sh
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${DISC_GPU:-0}"
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "ABORT: GPU busy (owner policy-server?) — compute apps present:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 2
fi

STEPS="${STEPS:-1000}"
RUN_NAME="${RUN_NAME:-grasp_sft_v2_demosonly_1gpu_disc}"
SAVE_DIR="$HOME/checkpoints/finetune/$RUN_NAME"

uv run python -m bijou.train \
  --train-data ~/datasets/fontaine/grasp_demos_v2/merged \
  --init-from ~/checkpoints/molmoact2-so101-released \
  --objective joint --joint-ce-weight 1.0 --insulate-flow \
  --recompute-stats \
  --flow-decoder-init inherit \
  --image-augment 0.8 \
  --decoder-lr 5e-5 --backbone-text-lr 1e-5 \
  --steps "$STEPS" --batch-size 96 --backward-chunks 8 \
  --activation-checkpointing --offload-optim \
  --holdout-episodes 0.1 --eval-every 250 --eval-samples 256 \
  --eval-dataset-breakdown \
  --save-every 500 \
  --wandb-project fontaine --wandb-run-name "$RUN_NAME" \
  --save-dir "$SAVE_DIR" \
  2>&1 | tee -a "outputs/train/${RUN_NAME}.log"
