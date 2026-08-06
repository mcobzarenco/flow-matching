#!/usr/bin/env bash
# fontaine — Molmo2-4B AR memory smoke (all 4 GPUs, DDP4, B/rank arg).
# GATE for the Molmo2 AR 4xDDP launch (port plan §6 amendment,
#   owner-confirmed AR-first + "smoke with ddp enabled" 19:11Z):
#   150 steps of the EXACT launch recipe — live 4.85B trunk (fp32
#   masters, bf16 autocast), full aux + conditioning, eval at step 100
#   (probe decode + wandb table path, WANDB offline so the full metric
#   assembly runs without polluting the project), save at step 100
#   (Molmo2PromptConfig writer + full-trunk snapshot).
# Usage: ./smoke_molmo2_ar_ddp4.sh [batch] [backward_chunks] [zero1]
#   (defaults 12 2 1 — B12 chunked 2x6 with ZeRO-1; the suffix CE vocab
#   is 153k vs Gemma's 262k, so loss-side tensors are ~40% lighter at
#   matched B).
# Pass rule: rc=0 AND peak <= ~75000 MiB (>=5 GiB headroom on 80 GiB).
#   History: the no-zero1 chunk ladder is EXHAUSTED (B12 direct 77.5
#   GiB at first step; 2x6 and 6x2 both OOM — static budget ~76-77
#   GiB/rank, measured 19:5x-20:0xZ). ZeRO-1 (--zero1) shards Adam
#   moments 29.1 -> ~7.3 GiB/rank => static ~55 GiB, ~24 GiB headroom;
#   optimizer semantics exact.
#   Rate: record s/step (last 5 windows); projected 40k wall must be
#   < 30 h or the pre-reg takes a shorter schedule.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
cd /home/ubuntu/flow-matching

BATCH="${1:-12}"
BACKWARD_CHUNKS="${2:-2}"
ZERO1="${3:-1}"
TAG="molmo2_ar_ddp4_b${BATCH}"
CHUNK_ARGS=()
if [ "$BACKWARD_CHUNKS" -gt 1 ]; then
    CHUNK_ARGS=(--backward-chunks "$BACKWARD_CHUNKS")
    TAG="${TAG}c${BACKWARD_CHUNKS}"
fi
if [ "$ZERO1" -eq 1 ]; then
    CHUNK_ARGS+=(--zero1)
    TAG="${TAG}z1"
fi

for gpu in 0 1 2 3; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
  if [ "$mem" -gt 1024 ]; then echo "GPU ${gpu} busy (${mem} MiB) — abort"; exit 1; fi
done

VRAM_LOG="/home/ubuntu/smoke_${TAG}_vram.log"
: > "$VRAM_LOG"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 \
      | paste -sd' ' >> "$VRAM_LOG"
    sleep 2
  done ) &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

set +e
.venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.train \
    --train-data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --fps 30 --camera-counts 1 2 \
    --holdout-episodes 0.1 --split-seed 0 \
    --decoder ar_backbone \
    --backbone allenai/Molmo2-4B \
    --max-crops 1 \
    --fast-tokenizer mcobzarenco/bijou-checkpoints/fast_tokenizer_v2 \
    --aux-fields subgoal holding progress event visible \
    --aux-dropout 0.0 --field-dropout 0.1 \
    --condition-fields subgoal outcome smoothness \
    --condition-dropout 0.1 --subgoal-dropout 0.5 \
    --instruction-augment 0.5 \
    --camera-kind-dropout 0.1 \
    --decoder-lr 1e-4 --backbone-text-lr 2e-5 --grad-clip 100 \
    --steps 150 --warmup-steps 1000 --batch-size "$BATCH" \
    "${CHUNK_ARGS[@]}" \
    --num-workers 20 --prefetch-factor 4 \
    --eval-samples 64 --eval-every 100 --save-every 100 --log-every 20 \
    --seed 0 --wandb-project fontaine \
    --wandb-run-name "smoke_${TAG}" \
    --save-dir "outputs/train/smoke_${TAG}" \
    2>&1 | tee "/home/ubuntu/smoke_${TAG}.log"
TRAIN_RC=$?
set -e

kill "$SAMPLER_PID" 2>/dev/null || true
PEAK=$(tr ' ' '\n' < "$VRAM_LOG" | sort -n | tail -1)
RATE=$(grep -o '"s_per_step": [0-9.]*' "/home/ubuntu/smoke_${TAG}.log" | tail -5 | awk '{s+=$2} END {if (NR) printf "%.3f", s/NR; else print "n/a"}')
# Verdict tee'd into the smoke log too — a tmux pane dies with the
# session, the log is the record (lesson: first arch run's echoes died).
{
  echo "=== MOLMO2 AR SMOKE ${TAG}: rc=${TRAIN_RC}, peak VRAM ${PEAK} MiB (any GPU), s/step(last5) ${RATE} ==="
  echo "=== pass: rc=0 AND peak <= ~75000 MiB; projected 40k wall = 40000*rate <= 108000 s (30 h) ==="
} | tee -a "/home/ubuntu/smoke_${TAG}.log"
