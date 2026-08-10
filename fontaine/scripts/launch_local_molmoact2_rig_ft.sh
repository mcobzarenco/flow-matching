#!/usr/bin/env bash
# MolmoAct2 rig fine-tune, rung 1 (action-expert-only) — local H100.
# Pre-reg: fontaine/blog/src/posts/2026-08-10-prereg-molmoact2-rig-finetune.md
# (owner GO 15:24:16Z 08-10; param sheet in-channel 16:20Z; Amendment 1
# 16:2xZ; silence-window close 17:50Z). Their stack verbatim
# (~/molmoact2 branch fontaine-so101-rig), so101_rig mixture, rig-only
# q01/q99, their README AE-only recipe with the pre-reg's named deltas.
set -euo pipefail

EXP=/home/ubuntu/molmoact2/experiments
LOG=/home/ubuntu/logs/molmoact2_rig_ft.log

systemd-run --user --unit=fontaine-molmoact2-rig-ft \
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
    so101_rig \
    --wandb.name=fontaine_so101_rig_ae_r1 \
    --max_duration=2000 \
    --device_batch_size=8 \
    --global_batch_size=64 \
    --num_workers=12 --prefetch_factor=4 --pin_memory=true \
    --save_interval=500 \
    --save_num_checkpoints_to_keep=20 \
    --save_folder=checkpoints/finetune/fontaine_so101_rig_ae_r1 \
    --packing=false \
    --dynamic_seq_len=true \
    --ft_vlm=false \
    --ft_action_expert=true \
    --ft_embedding=none \
    --lora_enable=false \
    --action_expert_learning_rate=5e-5 \
    >> $LOG 2>&1"

echo "launched unit fontaine-molmoact2-rig-ft, log $LOG"
