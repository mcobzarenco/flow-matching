#!/usr/bin/env bash
# Grasp-SFT STAGE C — AR PRIMARY (pre-reg §6 frozen, 2026-08-15).
# Pre-reg: fontaine/blog/src/posts/2026-08-14-prereg-grasp-sft-bootstrap.md
# Launches ONLY after stage B's gate reads green (>= 300 kept) — the
# preflight below refuses otherwise. Recipe = rig-ft run 1
# (launch_local_molmoact2_rig_ft.sh, banked 2026-08-10) VERBATIM-CLASS.
#
# ARG DIFF vs the banked rig-ft r1 command line (the verbatim receipt —
# every line not listed here is byte-identical):
#   so101_rig                          -> so101_grasp_sft   (§6: demo set,
#                                         recomputed per-tag q01/q99, no shim)
#   --wandb.name=fontaine_so101_rig_ae_r1
#                                      -> fontaine_grasp_sft_stagec_ar
#   --max_duration=2000                -> 3000               (§6: demo set is
#                                         45-90k frames, 2-4 epochs at gb64)
#   --save_folder=checkpoints/finetune/fontaine_so101_rig_ae_r1
#                                      -> .../fontaine_grasp_sft_stagec_ar
#   unit fontaine-molmoact2-rig-ft     -> fontaine-grasp-sft-stagec-ar
#   log  molmoact2_rig_ft.log          -> molmoact2_grasp_sft_stagec_ar.log
# NO other deltas: ft_action_expert=true only (ft_vlm=false,
# ft_embedding=none, lora_enable=false), AE LR 5e-5, gb64 (device 8),
# save every 500 keep 20, dynamic seq len, no packing, workers 12.
# Note: no shim flag exists anywhere in this arg list (§6 item 4).
set -euo pipefail

cd /home/ubuntu/flow-matching
.venv/bin/python fontaine/scripts/grasp_sft_stagec_preflight.py

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi

EXP=/home/ubuntu/molmoact2/experiments
LOG=/home/ubuntu/logs/molmoact2_grasp_sft_stagec_ar.log

systemd-run --user --unit=fontaine-grasp-sft-stagec-ar \
  --working-directory="$EXP" \
  --setenv=HOME=/home/ubuntu \
  --setenv=PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  --setenv=PYTHONPATH="$EXP:$EXP/../lerobot/src" \
  --setenv=LEROBOT_DATA_ROOT=/home/ubuntu/datasets \
  --setenv=MOLMO_DATA_DIR=/home/ubuntu/molmoact2/molmo_data \
  --setenv=WANDB_PROJECT=fontaine \
  --setenv=WANDB_ENTITY=aristotle1337 \
  --setenv=HF_HOME=/home/ubuntu/.cache/huggingface \
  bash -c "$EXP/../.venv/bin/torchrun --standalone --nproc-per-node=1 \
    launch_scripts/train_lerobot.py \
    allenai/MolmoAct2-SO100_101 \
    so101_grasp_sft \
    --wandb.name=fontaine_grasp_sft_stagec_ar \
    --max_duration=3000 \
    --device_batch_size=8 \
    --global_batch_size=64 \
    --num_workers=12 --prefetch_factor=4 --pin_memory=true \
    --save_interval=500 \
    --save_num_checkpoints_to_keep=20 \
    --save_folder=checkpoints/finetune/fontaine_grasp_sft_stagec_ar \
    --packing=false \
    --dynamic_seq_len=true \
    --ft_vlm=false \
    --ft_action_expert=true \
    --ft_embedding=none \
    --lora_enable=false \
    --action_expert_learning_rate=5e-5 \
    >> $LOG 2>&1"

echo "launched unit fontaine-grasp-sft-stagec-ar, log $LOG"
echo "NEXT: activate the PREPARED grasp_sft_stageC_ar babysit entry (started_utc = now)"
