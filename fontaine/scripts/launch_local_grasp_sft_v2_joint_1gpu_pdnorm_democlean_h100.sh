#!/usr/bin/env bash
# grasp-SFT v2 demos + clean ONLY — the poison-pinning cell. LOCAL
# H100, single GPU. STAGED with the pre-reg DRAFT 2026-08-20
# (posts/2026-08-20-prereg-demos-plus-clean.md, queue item
# prereg-draft-demos-plus-clean). Launch is DELEGATED (standing
# no-GO-ask rule 2026-08-18): fires at the next free GPU window
# AFTER the grpo-r2 lane closes its boundary; announce in-channel
# at launch.
#
# Recipe = the pdnorm mixed recipe
# (launch_local_grasp_sft_v2_joint_1gpu_pdnorm_h100.sh, CONVICTED
# cell 1/100 on 2026-08-19) with exactly ONE recipe delta:
# so101_pick_place_v2 is DROPPED from the mix — the mirror image of
# the onerig cell (28/100, mix-exonerated 2026-08-20), which dropped
# clean. Identically: demos + clean = "full mix minus v2"; the two
# follow-ups named by the onerig pre-reg are the same cell.
# Everything else verbatim: per-dataset flow norm, joint objective +
# insulate-flow, recompute-stats, repeat-4 on the remaining rig
# dataset (clean raw share 0.70% here vs 0.65 inside the convicted
# mix — dose held constant), eff-96 = micro-12 x 8 chunks, act-ckpt
# + offload-optim + prune-superseded-optim, seed 0 (same-seed
# comparability policy), STEPS default 3000.
#
# Guard: aborts if any compute process holds the GPU — the owner
# policy-server (port 8144) silently claims the H100 for rig serving
# and must never be preempted or killed.
#
# Fit smoke first per house rules:
#   STEPS=20 SMOKE=1 bash launch_local_grasp_sft_v2_joint_1gpu_pdnorm_democlean_h100.sh
#
# Launch (at the delegated window; announce in-channel):
#   systemd-run --user --unit=fontaine-v2-joint-pdnorm-democlean \
#     --setenv=HOME=/home/ubuntu \
#     --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
#     --working-directory=/home/ubuntu/flow-matching \
#     bash ~/flow-matching/fontaine/scripts/launch_local_grasp_sft_v2_joint_1gpu_pdnorm_democlean_h100.sh
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${PDNORM_GPU:-0}"
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "ABORT: GPU busy (owner policy-server?) — compute apps present:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 2
fi

STEPS="${STEPS:-3000}"
RUN_NAME="${RUN_NAME:-grasp_sft_v2_joint_1gpu_pdnorm_democlean}"
SAVE_DIR="$HOME/checkpoints/finetune/$RUN_NAME"
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_NAME="${RUN_NAME}_smoke"
  SAVE_DIR=/tmp/${RUN_NAME}
fi

uv run python -m bijou.train \
  --train-data ~/datasets/fontaine/grasp_demos_v2/merged \
               ~/datasets/mcobzarenco/so101_pick_place_clean \
  --dataset-repeat 'mcobzarenco/so101_pick_place*=4' \
  --init-from ~/checkpoints/molmoact2-so101-released \
  --objective joint --joint-ce-weight 1.0 --insulate-flow \
  --recompute-stats \
  --per-dataset-flow-norm \
  --flow-decoder-init inherit \
  --image-augment 0.8 \
  --decoder-lr 5e-5 --backbone-text-lr 1e-5 \
  --steps "$STEPS" --batch-size 96 --backward-chunks 8 \
  --activation-checkpointing --offload-optim \
  --prune-superseded-optim \
  --holdout-episodes 0.1 --eval-every 250 --eval-samples 256 \
  --eval-dataset-breakdown \
  --save-every 500 \
  --wandb-project fontaine --wandb-run-name "$RUN_NAME" \
  --save-dir "$SAVE_DIR" \
  2>&1 | tee -a "outputs/train/${RUN_NAME}.log"
