#!/usr/bin/env bash
# Grasp-SFT v2 DEMOS-ONLY joint — 8xA100-80GB box (147.224.218.164).
# Owner order 11:27-11:28Z 2026-08-17: kill the v2 mixed run (done at
# step ~1150 — train MAE rising again, run-1b signature) and restart
# on JUST the v2 demo dataset, same hyperparams, --recompute-stats so
# the quantiles match the sim demos exactly, no smoke. This is the
# isolation grid's "demos-native table" cell: with one dataset the
# recomputed table IS the demos' own ranges.
#
# launch_box_grasp_sft_v2_joint_8xa100.sh with exactly three deltas
# (owner-ordered in-channel, delta posted 11:37Z before launch):
#   1. --train-data grasp_demos_v2/merged ONLY (both rig datasets out)
#   2. --dataset-repeat dropped (existed only for the rig sets)
#   3. run/save/log names grasp_sft_v2_demosonly_8xa100
# Same command/seed otherwise.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

STEPS="${STEPS:-3000}"
BATCH="${BATCH:-12}"
RUN_NAME="${RUN_NAME:-grasp_sft_v2_demosonly_8xa100}"
SAVE_DIR="$HOME/checkpoints/finetune/$RUN_NAME"

uv run torchrun --standalone --nproc-per-node=8 -m bijou.train \
  --train-data ~/datasets/fontaine/grasp_demos_v2/merged \
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
