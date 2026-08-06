#!/usr/bin/env bash
# E4B pre-launch memory smoke (pre-reg checklist item 4) — 1xGPU,
# the exact E4B recipe, 60 steps at loader B12, peak VRAM recorded.
# Usage: ./smoke_e4b_b12.sh [backward_chunks]   (default 1 = B12 direct)
# Ladder (pre-registered, decided HERE, never mid-run):
#   rung 1: B12 direct fits with >=3 GiB headroom (peak <= ~76.5 GiB)
#   rung 2-4: first of 2/3/4 chunks that fits
#   none fit: DO NOT LAUNCH; post the finding, owner call.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
cd /home/ubuntu/flow-matching

CHUNKS="${1:-1}"
CHUNK_ARGS=()
if [ "$CHUNKS" -gt 1 ]; then CHUNK_ARGS=(--backward-chunks "$CHUNKS"); fi

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi

# VRAM sampler (2 s cadence) — peak printed at the end.
SMOKE_TAG="e4b_b12_chunks${CHUNKS}"
VRAM_LOG="/home/ubuntu/smoke_${SMOKE_TAG}_vram.log"
: > "$VRAM_LOG"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 >> "$VRAM_LOG"
    sleep 2
  done ) &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

set +e
.venv/bin/python -m bijou.train \
    --train-data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --fps 30 --camera-counts 1 2 \
    --holdout-episodes 0.1 --split-seed 0 \
    --decoder ar_backbone \
    --backbone google/gemma-4-e4b-it \
    --fast-tokenizer mcobzarenco/bijou-checkpoints/fast_tokenizer_v2 \
    --aux-fields subgoal holding progress event visible \
    --aux-dropout 0.0 --field-dropout 0.1 \
    --condition-fields subgoal outcome smoothness \
    --condition-dropout 0.1 --subgoal-dropout 0.5 \
    --instruction-augment 0.5 \
    --camera-kind-dropout 0.1 \
    --decoder-lr 1e-4 --backbone-text-lr 2e-5 --grad-clip 100 \
    --steps 60 --warmup-steps 1000 --batch-size 12 \
    "${CHUNK_ARGS[@]}" \
    --num-workers 20 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 2500 --log-every 20 \
    --seed 0 \
    --save-dir outputs/train/smoke_${SMOKE_TAG} \
    2>&1 | tee "/home/ubuntu/smoke_${SMOKE_TAG}.log"
TRAIN_RC=$?
set -e

kill "$SAMPLER_PID" 2>/dev/null || true
PEAK=$(sort -n "$VRAM_LOG" | tail -1)
echo "=== SMOKE ${SMOKE_TAG}: train rc=${TRAIN_RC}, peak VRAM ${PEAK} MiB (sampler 2 s; OOM => rc!=0) ==="
