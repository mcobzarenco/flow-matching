#!/usr/bin/env bash
# PREPARED (owner-gated): grasp-SFT v2 mixed rerun with
# --per-dataset-flow-norm — LOCAL H100, single GPU.
#
# Recipe = the mixed-v2 box recipe
# (fontaine/scripts/box/launch_box_grasp_sft_v2_joint_8xa100.sh)
# with exactly ONE recipe delta: --per-dataset-flow-norm (the 08-17
# isolation verdict's recommendation; enabler 6a6a0aa, family-level
# port d3dd4d0). Everything else about the command is the box recipe
# re-platformed through the discriminator run's PROVEN single-GPU
# form (launch_local_grasp_sft_v2_demosonly_1gpu_disc_h100.sh:
# eff-96 = micro-12 x 8 backward chunks, act-ckpt + offload-optim,
# measured 62.26/78 GiB + healthy verdict on this host):
#   - same 3-dataset mix + --dataset-repeat 'so101_pick_place*=4'
#   - same --recompute-stats (CE/state tables stay merged-recomputed;
#     the flag moves ONLY the flow codec to per-dataset rows)
#   - same joint objective / lrs / augment / holdout / eval cadence
#   - same default seed (comparability policy)
#   - STEPS default 3000 (the mixed-v2 recipe's registered length)
#   - torchrun/zero1/chunk-grad-allreduce DELETED: the drift
#     discriminator CONVICTED that path 2026-08-18 00:42Z
#
# Guard: aborts if any compute process holds the GPU — the owner
# policy-server (port 8144) silently claims the H100 for rig serving
# and must never be preempted or killed.
#
# Fit smoke first per house rules (the mix adds rig-video decode to
# the loader but the model-side footprint matches the discriminator):
#   STEPS=20 SMOKE=1 bash launch_local_grasp_sft_v2_joint_1gpu_pdnorm_h100.sh
#
# LAUNCH ONLY on the owner GO, pre-reg posted first:
#   systemd-run --user --unit=fontaine-v2-joint-pdnorm \
#     --setenv=HOME=/home/ubuntu \
#     --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
#     --working-directory=/home/ubuntu/flow-matching \
#     bash ~/flow-matching/fontaine/scripts/launch_local_grasp_sft_v2_joint_1gpu_pdnorm_h100.sh
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
RUN_NAME="${RUN_NAME:-grasp_sft_v2_joint_1gpu_pdnorm}"
SAVE_DIR="$HOME/checkpoints/finetune/$RUN_NAME"
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_NAME="${RUN_NAME}_smoke"
  SAVE_DIR=/tmp/${RUN_NAME}
fi

uv run python -m bijou.train \
  --train-data ~/datasets/fontaine/grasp_demos_v2/merged \
               ~/datasets/mcobzarenco/so101_pick_place_v2 \
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
  --holdout-episodes 0.1 --eval-every 250 --eval-samples 256 \
  --eval-dataset-breakdown \
  --save-every 500 \
  --wandb-project fontaine --wandb-run-name "$RUN_NAME" \
  --save-dir "$SAVE_DIR" \
  2>&1 | tee -a "outputs/train/${RUN_NAME}.log"
