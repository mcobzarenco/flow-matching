#!/usr/bin/env bash
# fontaine — arch-batch-1 F1 MEMORY SMOKE (GPUs 1-3, DDP3, B32/rank).
# PRE-REG: fontaine/blog/src/posts/2026-08-06-prereg-arch-batch-1.md
#   gate F1 (Amendment 1: two configs A+B; Amendment 2: arm A := 280).
# Usage: ./smoke_arch_ddp3_b32.sh A|B [batch]   (batch default 32)
#   200-step smoke of the EXACT arm recipe; peak VRAM per GPU sampled
#   at 2 s; rate (s_per_step) read from the tee'd log afterwards.
# F1 rule: any OOM => the WHOLE batch drops to the largest
#   B in {24, 16} that fits BOTH configs with >=5 GiB headroom
#   (peak <= ~75 GiB on 80 GiB cards) — one eff-batch for the whole
#   batch, never per-arm. Re-run this script for both arms at the
#   lower B before any launch.
# F2 rule (arm A only): projected 40k wall = 40000 * s_per_step;
#   > 30 h => arm A reduces to a 10k screen at 280 (screen-rung label).
# Priors (these are what the smoke tests): arms ~0.7-0.9 s/step (B),
#   arm A ~1.5-2x that at 280 tokens; B64 flow was ~40 GiB on 1xH100
#   => B32 + 2x prefix (A) / +15 res-adapter streams (B) unknown.
# EXECUTION: requires GPUs 1-3 quiet (guard below); GPU 0 may be
#   running arm C's chained evals — different GPUs, no co-location.
#   Code: live checkout at fontaine HEAD (synced at the arm-C-free
#   boundary; stage-0 re-verify green before this runs).
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1,2,3
export WANDB_MODE=disabled
cd /home/ubuntu/flow-matching

ARM="${1:?usage: smoke_arch_ddp3_b32.sh A|B [batch]}"
BATCH="${2:-32}"
case "$ARM" in
  A) ARM_ARGS=(--max-soft-tokens 280 --stream-counts 4 4 7)
     TAG="archA_img280_b${BATCH}" ;;
  B) ARM_ARGS=(--conditioning-streams residual)   # schedule structural: no --stream-counts
     TAG="archB_fullresid_b${BATCH}" ;;
  *) echo "arm must be A or B"; exit 2 ;;
esac

for gpu in 1 2 3; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
  if [ "$mem" -gt 1024 ]; then echo "GPU ${gpu} busy (${mem} MiB) — abort"; exit 1; fi
done

VRAM_LOG="/home/ubuntu/smoke_${TAG}_vram.log"
: > "$VRAM_LOG"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,2,3 \
      | paste -sd' ' >> "$VRAM_LOG"
    sleep 2
  done ) &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

set +e
.venv/bin/torchrun --standalone --nproc-per-node=3 -m bijou.train \
    --train-data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
                 /home/ubuntu/datasets/mcobzarenco/so101_pick_place_v2 \
                 /home/ubuntu/datasets/mcobzarenco/so101_pick_place_clean \
    --fps 30 --camera-counts 1 2 \
    --holdout-episodes 0.1 --split-seed 0 \
    --decoder flow --prompt-generate-bracket \
    --backbone google/gemma-4-e2b-it \
    --backbone-init-from outputs/train/bijou_arb_rcond_100k_ddp4/step_100000 \
    --self-attention-mode bidirectional --time-conditioning adarms \
    --decoder-hidden 1024 --decoder-heads 8 \
    --decoder-intermediate 4096 --decoder-cross-heads 8 \
    --chunk-size 50 \
    --camera-kind-dropout 0.1 --instruction-augment 0.5 \
    --condition-fields subgoal outcome smoothness \
    --condition-dropout 0.1 --subgoal-dropout 0.5 \
    --decoder-lr 1e-4 --warmup-steps 500 --weight-decay 1e-5 \
    --grad-clip 10.0 \
    --steps 200 --batch-size "$BATCH" \
    "${ARM_ARGS[@]}" \
    --num-workers 20 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 2500 --log-every 20 \
    --seed 0 --eval-seed 0 \
    --save-dir outputs/train/smoke_${TAG} \
    2>&1 | tee "/home/ubuntu/smoke_${TAG}.log"
TRAIN_RC=$?
set -e

kill "$SAMPLER_PID" 2>/dev/null || true
PEAK=$(tr ' ' '\n' < "$VRAM_LOG" | sort -n | tail -1)
RATE=$(grep -o '"s_per_step": [0-9.]*' "/home/ubuntu/smoke_${TAG}.log" | tail -5 | awk '{s+=$2} END {if (NR) printf "%.3f", s/NR; else print "n/a"}')
# Verdict tee'd into the smoke log too — a tmux pane dies with the
# session, the log is the record (lesson: first run's echoes were lost).
{
  echo "=== F1 SMOKE ${TAG}: rc=${TRAIN_RC}, peak VRAM ${PEAK} MiB (any GPU), s/step(last5) ${RATE} ==="
  echo "=== F1 pass: rc=0 AND peak <= ~75000 MiB. F2 (arm A): 40000*rate <= 108000 s (30 h) ==="
} | tee -a "/home/ubuntu/smoke_${TAG}.log"
