#!/usr/bin/env bash
# Squint leg B RECOVERY (incident 08:10Z 08-22): the original
# fontaine-squint-adapt unit trained arm 1 (onerig) to 500/500 but the
# final async checkpoint save died on ENOSPC (disk 99%, staging needs
# ~44 GiB) — the unit exited 1 before arm 2 started; arm 1's step-500
# weights were lost (2.4 GiB .tmp), step_000250 (full, with optimizer)
# survived. Disk since cleared to ~196 GiB free (weights-only
# INTERMEDIATES of the three closed pdnorm runs pruned; endpoints and
# the squint step_000250 untouched).
#
# Recovery = arm 1 resumed step_000250 -> 500 (--resume, fresh seed 1
# per the enforced fresh-seed-on-resume convention: steps 250-500 see
# a different shuffle than the original — recorded lineage note, the
# probe curve of the lost segment stays record-only), then arm 2
# (democlean) verbatim per the frozen Slot 6 recipe
# (launch_squint_adapt_arms.sh). Under --resume the objective KIND,
# stats table and flow-norm scheme are checkpoint-inherited — but
# --insulate-flow is a plain CLI passthrough (bijou/train/args.py:971,
# NOT reconstructed from the recorded payload despite the field doc):
# attempt 1 of this recovery (08:40-09:0xZ) omitted it, the seam came
# back OPEN, the trunk took flow-loss gradients against CE-only Adam
# moments and the flow head collapsed (flow loss 0.09 -> 1.44, probe
# 2.80@300 -> 9.19@300; poisoned rows snipped to
# train_log_poisoned_resume_0840Z.jsonl). The flag MUST be re-declared
# here; lr/batch flags repeat the original values so the restored
# optimizer group check passes. Validation gate: if the first resumed
# records do not continue flow ~0.09-level, kill and full-retrain.
#
# Launch:
#   systemd-run --user --unit=fontaine-squint-adapt-r2 \
#     --setenv=HOME=/home/ubuntu \
#     --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
#     --working-directory=/home/ubuntu/flow-matching \
#     bash ~/flow-matching/fontaine/scripts/launch_squint_adapt_recover.sh
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

# Disk guard — the failure class this script recovers from. A full
# checkpoint save stages ~44 GiB; refuse to start without 3x headroom.
avail_gib=$(df --output=avail -B G /home/ubuntu | tail -1 | tr -dc 0-9)
if [ "$avail_gib" -lt 130 ]; then
  echo "ABORT: only ${avail_gib} GiB free (< 130) — clear disk first" >&2
  exit 3
fi

STEPS="${STEPS:-500}"

echo "=== adapt arm onerig RESUME 250->$STEPS (fresh seed 1) $(date -u +%FT%TZ)"
RUN_NAME="grasp_sft_v2_squint_adapt_onerig"
SAVE_DIR="$HOME/checkpoints/finetune/$RUN_NAME"
uv run python -m bijou.train \
  --train-data ~/datasets/fontaine/squint_twin_demos_v1 \
  --resume "$SAVE_DIR/step_000250" \
  --seed 1 \
  --joint-ce-weight 1.0 --insulate-flow \
  --image-augment 0.8 \
  --decoder-lr 5e-5 --backbone-text-lr 1e-5 \
  --steps "$STEPS" --batch-size 96 --backward-chunks 8 \
  --activation-checkpointing --offload-optim \
  --prune-superseded-optim \
  --holdout-episodes 0.1 --eval-every 100 --eval-samples 256 \
  --save-every 250 \
  --wandb-project fontaine --wandb-run-name "${RUN_NAME}_r2" \
  --save-dir "$SAVE_DIR" \
  2>&1 | tee -a "outputs/train/${RUN_NAME}.log"

echo "=== adapt arm democlean (fresh, frozen recipe) $(date -u +%FT%TZ)"
ARMS="democlean" STEPS="$STEPS" \
  bash ~/flow-matching/fontaine/scripts/launch_squint_adapt_arms.sh

echo "=== leg B recovery done $(date -u +%FT%TZ)"
