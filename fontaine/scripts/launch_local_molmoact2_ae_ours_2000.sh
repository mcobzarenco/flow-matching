#!/usr/bin/env bash
# MolmoAct2 AE fine-tune in OUR trainer (port item 4, gate G4) — local H100.
# Pre-reg: fontaine/blog/src/posts/2026-08-10-prereg-molmoact2-firstclass-port.md
# (G4 frozen 21:0xZ 08-10; execution note posted in-channel before this
# launch, 2026-08-11). bijou.molmoact2.train: their recipe verbatim
# (2000 steps, global 64 = 8x8 micro, AE-only lr 5e-5, warmup 200 ->
# cosine 0.1x, clip 1.0, rig-only q01/q99, their img_aug=full) with the
# two named deltas (deterministic frozen trunk; per-frame sqrt-weighted
# sampling with replacement). Gate <= 6 GPU-h (train ~2.5 measured-class
# + rung reads).
set -euo pipefail

cd /home/ubuntu/flow-matching

BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$BUSY" -gt 1024 ]; then
  echo "local GPU busy (${BUSY} MiB) — abort" >&2
  exit 1
fi

RUN=molmoact2_ae_ours_r1
LOG=/home/ubuntu/logs/${RUN}.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m bijou.molmoact2.train \
  --checkpoint allenai/MolmoAct2-SO100_101 \
  --norm-stats /home/ubuntu/checkpoints/molmoact2-so101-rig-r1-step2000-hf/norm_stats.json \
  --norm-tag so100_so101_molmoact2 \
  --train-data /home/ubuntu/datasets/mcobzarenco/so101_pick_place_clean \
               /home/ubuntu/datasets/mcobzarenco/so101_pick_place_v2 \
  --steps 2000 \
  --global-batch 64 --micro-batch 8 \
  --lr 5e-5 --warmup-steps 200 --alpha-f 0.1 --grad-clip 1.0 \
  --save-every 500 \
  --save-dir outputs/train/${RUN} \
  --num-workers 8 --prefetch-factor 2 \
  --log-every 20 \
  --seed 0 \
  --img-aug full \
  >> "$LOG" 2>&1

echo "training rc=0; G4 rung read follows (chained in-unit)" >> "$LOG"
.venv/bin/python fontaine/scripts/molmoact2_ours_ft_rung_read.py \
  --run-dir outputs/train/${RUN} \
  >> "$LOG" 2>&1
