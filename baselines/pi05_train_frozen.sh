#!/usr/bin/env bash
# pi0.5 frozen-trunk rig fine-tune (box): --policy.train_expert_only=true
# freezes the entire VLM (vision tower + PaliGemma LM); only the ~300M
# action expert + projections train. Motivation: every live-trunk arm on
# this ~50-episode corpus memorized by step ~500-1000, while frozen-trunk
# fts kept improving; 2500 steps, checkpoints every 250 to catch the
# holdout-free sweet spot. Otherwise identical to the unfrozen run
# (effective batch 64, pi05 AdamW preset peak 2.5e-5, warmup 200).
set -euo pipefail
cd ~/flow-matching/baselines
.venv/bin/accelerate launch --num_processes 2 --mixed_precision no \
  .venv/bin/lerobot-train \
  --dataset.repo_id=mcobzarenco/so101_pick_place_v2 \
  --dataset.root=/home/ubuntu/datasets/mcobzarenco/so101_pick_place_v2 \
  --rename_map='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --policy.path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub=false \
  --policy.scheduler_warmup_steps=200 \
  --output_dir=/home/ubuntu/flow-matching/outputs/train/pi05_ft_rig_v2_frozen \
  --job_name=pi05-ft-rig-v2-frozen \
  --steps=2500 \
  --log_freq=50 \
  --save_freq=250 \
  --batch_size=32 \
  --num_workers=8 \
  --seed=23 \
  --tolerance_s=0.0167 \
  --wandb.enable=true \
  --wandb.project=bijou-dev \
  2>&1 | tee /tmp/pi05_train_frozen.log
