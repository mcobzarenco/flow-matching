#!/usr/bin/env bash
# grasp-SFT v2 demos + clean_ch0fix_n — the ch0 (shoulder-pan) affine
# isolation cell, carrier-hunt ladder rung 2. LOCAL H100, single GPU.
# Pre-reg 2026-08-22 (posts/2026-08-22-prereg-clean-ch0-affine.md,
# queue item ch0-affine-exec). Launch is DELEGATED (standing no-GO-ask
# rule 2026-08-18): sequenced BEHIND the squint screen's GPU claim —
# fires at the first free window after the squint leg C boundary;
# announce in-channel at launch.
#
# Recipe = the democlean launcher
# (launch_local_grasp_sft_v2_joint_1gpu_pdnorm_democlean_h100.sh,
# 8/100 banked) with exactly ONE delta: so101_pick_place_clean ->
# so101_pick_place_clean_ch0fix_n in --train-data — the 7 clean
# episodes with ch0 (action AND state) mapped through the frozen
# moment-matched affine
#   x' = 0.0923439813196304 + (x - 1.481974338423806) * 2.755193138766973
# (materializer make_clean_ch0fix_dataset.py, all oracles green
# 2026-08-22; holdout draw pre-verified (2,) — clean-side train split
# episode-identical to democlean's). The --dataset-repeat glob
# 'mcobzarenco/so101_pick_place*=4' matches the new name unchanged.
# Everything else verbatim: per-dataset flow norm, joint objective +
# insulate-flow, recompute-stats (gives ch0fix its own pdnorm row —
# ch0 scale should move x2.755, all else pinned, the live materializer
# oracle), eff-96 = micro-12 x 8 chunks, act-ckpt + offload-optim +
# prune-superseded-optim, seed 0 (same-seed comparability), STEPS
# default 3000.
#
# Guard: aborts if any compute process holds the GPU — the owner
# policy-server (port 8144) silently claims the H100 for rig serving
# and must never be preempted or killed.
#
# Fit smoke first per house rules:
#   STEPS=20 SMOKE=1 bash launch_local_grasp_sft_v2_joint_1gpu_pdnorm_ch0fix_h100.sh
#
# Launch (at the delegated window; announce in-channel):
#   systemd-run --user --unit=fontaine-v2-joint-pdnorm-ch0fix \
#     --setenv=HOME=/home/ubuntu \
#     --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
#     --working-directory=/home/ubuntu/flow-matching \
#     bash ~/flow-matching/fontaine/scripts/launch_local_grasp_sft_v2_joint_1gpu_pdnorm_ch0fix_h100.sh
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
RUN_NAME="${RUN_NAME:-grasp_sft_v2_joint_1gpu_pdnorm_ch0fix}"
SAVE_DIR="$HOME/checkpoints/finetune/$RUN_NAME"
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_NAME="${RUN_NAME}_smoke"
  SAVE_DIR=/tmp/${RUN_NAME}
fi

uv run python -m bijou.train \
  --train-data ~/datasets/fontaine/grasp_demos_v2/merged \
               ~/datasets/mcobzarenco/so101_pick_place_clean_ch0fix_n \
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
