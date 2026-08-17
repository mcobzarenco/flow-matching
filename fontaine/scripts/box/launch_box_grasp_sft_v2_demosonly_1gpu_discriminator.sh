#!/usr/bin/env bash
# PREPARED (owner-gated): the 1-GPU drift discriminator.
#
# Every drifting run in this family (run-1b, run-2, mixed v2,
# demosonly) is 8xA100 distributed (torchrun + zero1 +
# --chunk-grad-allreduce); every healthy run (44/100 joint probe,
# 28/100 stage-C) was single-GPU. This launcher reruns the demosonly
# recipe on ONE box GPU with the SAME effective batch 96 and the SAME
# micro-batch 12 (--batch-size 96 --backward-chunks 8 = micro 12,
# exactly one 8x rank's shard), same seed, same --image-augment,
# same --recompute-stats, same init. The ONLY delta vs the drifting
# run is the distributed machinery.
#
# Read: MAE probes every 250 steps. The 8x runs rise monotonically
# from step 500 on BOTH slices (demosonly: eval 3.46/3.24/4.22/5.27/
# 6.17, train 3.69/3.32/3.86/4.60/5.62 at 250..1250). If this run's
# curve falls/holds through step 1000 -> the distributed path is
# CONVICTED; if it drifts the same -> distributed exonerated,
# remaining deltas vs the healthy probe are image-augment, eff-96,
# --recompute-stats-vs-baked-table, init checkpoint, corpus scale.
#
# Pace: single GPU carries the full eff-96 step, expect ~25-32 s/step
# => 1000 steps ~ 7-9 h wall, ~7-9 GPU-h (vs 8 GPU-h/h when the 8x
# run holds the box — this discriminator is CHEAPER per wall-clock
# insight than riding the drifting run was).
#
# LAUNCH ONLY after the owner's kill call frees the box:
#   systemd-run --user --unit=fontaine-demosonly-1gpu-disc \
#     --setenv=HOME=/home/ubuntu \
#     --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
#     bash ~/flow-matching/fontaine/scripts/box/launch_box_grasp_sft_v2_demosonly_1gpu_discriminator.sh
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${DISC_GPU:-0}"
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

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
