#!/usr/bin/env bash
# Grasp-SFT RIG-ONLY joint — 8xA100-80GB box (147.224.218.164).
# Owner order 13:30Z 2026-08-17: kill the demosonly run (done at step
# ~1350; drift reproduced, saves 500/1000 kept) and run a SHORT
# 1000-step run on JUST their rig datasets (pick_place_v2 +
# pick_place_clean), no demo datasets, same hyperparams otherwise,
# save every 250. Targets the data axis with the distributed stack
# held constant: known-good rig data drifting too would convict the
# recipe/stack; rig-only healthy implicates the sim-demo corpus.
#
# Deltas vs launch_box_grasp_sft_v2_demosonly_8xa100.sh (all
# owner-ordered in-channel 13:30Z):
#   1. --train-data = the two rig datasets ONLY
#   2. --steps 1000
#   3. --save-every 250
#   4. run/save/log names grasp_sft_rigonly_8xa100
# (--dataset-repeat stays dropped: it existed to boost the rig share
# inside a mix; rig-only IS the full corpus.) Same command/seed
# otherwise; --recompute-stats = rig-native table.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

STEPS="${STEPS:-1000}"
BATCH="${BATCH:-12}"
RUN_NAME="${RUN_NAME:-grasp_sft_rigonly_8xa100}"
SAVE_DIR="$HOME/checkpoints/finetune/$RUN_NAME"

uv run torchrun --standalone --nproc-per-node=8 -m bijou.train \
  --train-data ~/datasets/mcobzarenco/so101_pick_place_v2 \
               ~/datasets/mcobzarenco/so101_pick_place_clean \
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
  --save-every 250 \
  --wandb-project fontaine --wandb-run-name "$RUN_NAME" \
  --save-dir "$SAVE_DIR" \
  2>&1 | tee -a "outputs/train/${RUN_NAME}.log"
