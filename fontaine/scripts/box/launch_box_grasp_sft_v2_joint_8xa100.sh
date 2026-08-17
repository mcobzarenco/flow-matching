#!/usr/bin/env bash
# Grasp-SFT v2 joint — 8xA100-80GB box (147.224.218.164).
# Owner GO 09:08:27Z 2026-08-17 ("start train run v2"), recipe locked
# 09:23:42Z: IDENTICAL hyperparameters to run 2, NO per-dataset norm
# (one merged --recompute-stats table for CE/state/flow — the released
# checkpoint's own wide-mixture convention). Pre-reg:
# posts/2026-08-17-prereg-grasp-sft-v2-joint.md, posted before launch.
#
# This is launch_box_grasp_sft_v1_joint_8xa100.sh with exactly two
# deltas (both pre-registered):
#   1. --train-data grasp_demos_v2/merged (the regen corpus, expert
#      v1.3 + real bracket + refit wrist pose; SHIPPED 08:30Z)
#   2. run/save/log names grasp_sft_v2_joint_8xa100
# Same command/seed otherwise (seed policy: same run on better data =
# same seed, deltas attribute to the corpus). Fit smoke first per
# house rules:
#   STEPS=20 SMOKE=1 bash launch_box_grasp_sft_v2_joint_8xa100.sh
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

STEPS="${STEPS:-3000}"
BATCH="${BATCH:-12}"
RUN_NAME="${RUN_NAME:-grasp_sft_v2_joint_8xa100}"
SAVE_DIR="$HOME/checkpoints/finetune/$RUN_NAME"
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_NAME="${RUN_NAME}_smoke"
  SAVE_DIR=/tmp/${RUN_NAME}
fi

uv run torchrun --standalone --nproc-per-node=8 -m bijou.train \
  --train-data ~/datasets/fontaine/grasp_demos_v2/merged \
               ~/datasets/mcobzarenco/so101_pick_place_v2 \
               ~/datasets/mcobzarenco/so101_pick_place_clean \
  --dataset-repeat 'mcobzarenco/so101_pick_place*=4' \
  --init-from ~/checkpoints/molmoact2-so101-released \
  --objective joint --joint-ce-weight 1.0 --insulate-flow \
  --recompute-stats \
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
