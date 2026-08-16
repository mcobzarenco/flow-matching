#!/usr/bin/env bash
# Grasp-SFT v1 joint — 8xA100-80GB box (147.224.218.164).
# Owner GO 17:07:47Z 2026-08-16: "start the train run with the command
# you have above, only change is ... global effective batch of 96
# instead of 128 for faster steps."
# Command = the 16:39:18Z consolidated post (in-channel), verbatim,
# with the two owner deltas:
#   1. --batch-size 12  (eff. 96 = 12 x 8 ranks; was 16)
#   2. --init-from ~/checkpoints/molmoact2-so101-released — the
#      owner's re-converted released ckpt (joint-mapping fix +
#      --remap-stats v21-to-v30, reuploaded 17:07Z; pulled from
#      mcobzarenco/molmoact2-so101-released).
# Recipe class: route-C, the only MEASURED joint config (66.5 GiB
# peak at micro-16 + act-ckpt on one 80-GiB card; micro-12 is
# strictly smaller). Fit smoke first per house rules:
#   STEPS=20 SMOKE=1 bash launch_box_grasp_sft_v1_joint_8xa100.sh
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

STEPS="${STEPS:-3000}"
BATCH="${BATCH:-12}"
RUN_NAME="${RUN_NAME:-grasp_sft_v1_joint_8xa100}"
SAVE_DIR="$HOME/checkpoints/finetune/$RUN_NAME"
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_NAME="${RUN_NAME}_smoke"
  SAVE_DIR=/tmp/${RUN_NAME}
fi

uv run torchrun --standalone --nproc-per-node=8 -m bijou.train \
  --train-data ~/datasets/fontaine/grasp_demos_v1/merged \
               ~/datasets/mcobzarenco/so101_pick_place_v2 \
               ~/datasets/mcobzarenco/so101_pick_place_clean \
  --dataset-repeat 'mcobzarenco/so101_pick_place*=4' \
  --init-from ~/checkpoints/molmoact2-so101-released \
  --objective joint --joint-ce-weight 1.0 --insulate-flow \
  --flow-decoder-init inherit \
  --image-augment 0.8 \
  --decoder-lr 5e-5 --backbone-text-lr 1e-5 \
  --steps "$STEPS" --batch-size "$BATCH" --backward-chunks 2 \
  --chunk-grad-allreduce --zero1 --activation-checkpointing \
  --holdout-episodes 0.1 --eval-every 250 --eval-samples 256 \
  --eval-dataset-breakdown \
  --save-every 500 \
  --wandb-project fontaine --wandb-run-name "$RUN_NAME" \
  --save-dir "$SAVE_DIR" \
  2>&1 | tee -a "outputs/train/${RUN_NAME}.log"
