#!/usr/bin/env bash
# Squint qualification screen — GPU leg B: Gate-1 adaptation, BOTH arms
# chained in one unit (exec item squint-twin-screen-exec part (c);
# frozen command block = the finalization amendment on
# posts/2026-08-22-prereg-squint-twin-screen.md, Slot 6). ONE recipe,
# identical across arms: 500 steps ~= 2.25 GPU-h/arm at the measured
# 16.2 s/step (hard cap 2.5/arm), init each arm's step_003000, twin
# demo dataset only, seed = bijou default (same both arms), no tuning,
# no second attempt.
#
# Launch gate: leg A rc 0 + conversion_oracle.json verified
# (round-trip < 1e-5 rad). Announce in-channel at launch.
#
# Fit smoke first per house rules:
#   STEPS=20 SMOKE=1 bash launch_squint_adapt_arms.sh
#
# Launch:
#   systemd-run --user --unit=fontaine-squint-adapt \
#     --setenv=HOME=/home/ubuntu \
#     --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
#     --working-directory=/home/ubuntu/flow-matching \
#     bash ~/flow-matching/fontaine/scripts/launch_squint_adapt_arms.sh
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${ADAPT_GPU:-0}"
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "ABORT: GPU busy (owner policy-server?) — compute apps present:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 2
fi

STEPS="${STEPS:-500}"
ARMS="${ARMS:-onerig democlean}"

for arm in $ARMS; do
  RUN_NAME="grasp_sft_v2_squint_adapt_${arm}"
  SAVE_DIR="$HOME/checkpoints/finetune/$RUN_NAME"
  if [[ "${SMOKE:-0}" == "1" ]]; then
    RUN_NAME="${RUN_NAME}_smoke"
    SAVE_DIR=/tmp/${RUN_NAME}
  fi
  echo "=== adapt arm $arm ($STEPS steps) $(date -u +%FT%TZ)"
  uv run python -m bijou.train \
    --train-data ~/datasets/fontaine/squint_twin_demos_v1 \
    --init-from ~/checkpoints/finetune/grasp_sft_v2_joint_1gpu_pdnorm_${arm}/step_003000 \
    --objective joint --joint-ce-weight 1.0 --insulate-flow \
    --recompute-stats \
    --per-dataset-flow-norm \
    --flow-decoder-init inherit \
    --image-augment 0.8 \
    --decoder-lr 5e-5 --backbone-text-lr 1e-5 \
    --steps "$STEPS" --batch-size 96 --backward-chunks 8 \
    --activation-checkpointing --offload-optim \
    --prune-superseded-optim \
    --holdout-episodes 0.1 --eval-every 100 --eval-samples 256 \
    --save-every 250 \
    --wandb-project fontaine --wandb-run-name "$RUN_NAME" \
    --save-dir "$SAVE_DIR" \
    2>&1 | tee -a "outputs/train/${RUN_NAME}.log"
done

echo "=== leg B done $(date -u +%FT%TZ)"
