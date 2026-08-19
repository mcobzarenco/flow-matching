#!/usr/bin/env bash
# grasp-SFT v2 demos + ONE rig dataset isolation cell — LOCAL
# H100, single GPU. STAGED with the pre-reg DRAFT 2026-08-19
# (posts/2026-08-19-prereg-demos-plus-one-rig.md, queue item
# prereg-draft-demos-plus-one-rig). EXECUTION IS AN OWNER CALL per
# the pdnorm pre-reg's registered grid text ("Next isolation is an
# owner call") — do NOT launch from the draft item.
#
# Recipe = the pdnorm mixed recipe
# (launch_local_grasp_sft_v2_joint_1gpu_pdnorm_h100.sh, CONVICTED
# cell 1/100 on 2026-08-19) with exactly ONE recipe delta:
# so101_pick_place_clean is DROPPED from the mix. Everything else
# verbatim: per-dataset flow norm, joint objective + insulate-flow,
# recompute-stats, repeat-4 on the remaining rig dataset (v2 share
# 6.31% vs 6.26% inside the convicted mix — dose held constant),
# eff-96 = micro-12 x 8 chunks, act-ckpt + offload-optim, seed 0
# (same-seed comparability policy), STEPS default 3000.
#
# Disk policy (2026-08-19 root-disk-full incident follow-up): a
# sidecar pruner deletes superseded offload-optim optimizer.pt
# mirrors (~31 GiB per save), keeping the latest TWO saves
# resume-capable. Weights are never touched.
#
# Guard: aborts if any compute process holds the GPU — the owner
# policy-server (port 8144) silently claims the H100 for rig serving
# and must never be preempted or killed.
#
# Fit smoke first per house rules:
#   STEPS=20 SMOKE=1 bash launch_local_grasp_sft_v2_joint_1gpu_pdnorm_onerig_h100.sh
#
# Launch (only after the owner isolation call; announce in-channel):
#   systemd-run --user --unit=fontaine-v2-joint-pdnorm-onerig \
#     --setenv=HOME=/home/ubuntu \
#     --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
#     --working-directory=/home/ubuntu/flow-matching \
#     bash ~/flow-matching/fontaine/scripts/launch_local_grasp_sft_v2_joint_1gpu_pdnorm_onerig_h100.sh
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
RUN_NAME="${RUN_NAME:-grasp_sft_v2_joint_1gpu_pdnorm_onerig}"
SAVE_DIR="$HOME/checkpoints/finetune/$RUN_NAME"
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_NAME="${RUN_NAME}_smoke"
  SAVE_DIR=/tmp/${RUN_NAME}
fi

# Superseded-optimizer pruner: every 5 min drop optimizer.pt from all
# but the latest TWO step_* saves (latest may be mid-write; the one
# before it stays resume-capable).
(
  while true; do
    sleep 300
    ls -d "$SAVE_DIR"/step_* 2>/dev/null | sort | head -n -2 | while read -r d; do
      rm -f "$d/optimizer.pt"
    done
  done
) &
PRUNE_PID=$!
trap 'kill "$PRUNE_PID" 2>/dev/null || true' EXIT

uv run python -m bijou.train \
  --train-data ~/datasets/fontaine/grasp_demos_v2/merged \
               ~/datasets/mcobzarenco/so101_pick_place_v2 \
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
  --holdout-episodes 0.1 --eval-every 250 --eval-samples 256 \
  --eval-dataset-breakdown \
  --save-every 500 \
  --wandb-project fontaine --wandb-run-name "$RUN_NAME" \
  --save-dir "$SAVE_DIR" \
  2>&1 | tee -a "outputs/train/${RUN_NAME}.log"
