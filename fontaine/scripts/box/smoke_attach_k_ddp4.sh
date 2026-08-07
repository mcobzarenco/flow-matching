#!/usr/bin/env bash
# fontaine — attach-screen ARM K memory smoke LADDER (box 4xDDP).
# PRE-REG instrument item 4 (2026-08-07-prereg-molmo2-attach-screen.md):
#   the K launcher's K_MEM_READY guard demands this ladder GREEN at the
#   launch batch before any K launch — phase 1 alone sat at 67.07/71 GiB
#   with no expert riding, so "it probably fits checkpointed" is not a
#   measurement.
# WHAT IT RUNS: the EXACT K recipe (launch_box_fontaine_molmo2_attach_K_
#   10k_ddp4.sh flags verbatim — live trunk warm-started from the 40k
#   endpoint, --joint-ce, --seam-stop-grad, --activation-checkpointing,
#   zero1 + chunked backward + chunk-grad-allreduce) for 150 steps per
#   rung, with eval at step 100 (probe decode path) and save at step 100
#   (joint_ce writer + full-trunk snapshot) so every memory shape of the
#   real run is exercised, not just the bare step.
# LADDER (chunk microbatch pinned at 2 on every rung — the memory shape
#   per chunk stays matched):
#     rung 1: B12 c6  (eff-48 — the pre-registered batch)
#     rung 2: B8  c4  (eff-32)
#     rung 3: B6  c3  (eff-24)
#   First green rung wins and the ladder stops. A green rung BELOW B12
#   is a MATCHED DOWNSHIFT: BOTH arms take the same BATCH/BACKWARD_CHUNKS
#   (pre-reg: never K alone) — echoed loudly below.
# PASS RULE per rung: rc=0 AND max vram_alloc_peak_gib over the rung's
#   jsonl rows <= 71.0 (the pre-reg gate babysit enforces on the real
#   run; torch alloc peak, NOT nvidia-smi reserved — the phase-1 ladder
#   lesson: expandable_segments' reserved pool never shrinks and shadows
#   the live gap). nvidia-smi peak across all 4 GPUs is echoed as the
#   secondary headroom read (advisory bar ~75000 MiB on 80 GiB).
# ON GREEN: writes fontaine/harness/state/k_mem_ready (batch, chunks,
#   peak, rate, log path) — the durable record behind setting
#   K_MEM_READY=1 at launch. Launch with:
#     K_MEM_READY=1 BATCH=<green B> BACKWARD_CHUNKS=<green c> ./launch_..._attach_F...
#     (F first, same BATCH/BACKWARD_CHUNKS, then K.)
# ON ALL-RED: no marker, loud abort — the attach screen does not launch
#   at any laddered batch without owner steer.
# Rate: last-5 s_per_step echoed + 10k-arm GPU-h projection (advisory;
#   attach_rate_gate.py at launch is the binding 70 GPU-h gate).
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
cd /home/ubuntu/flow-matching

ENDPOINT=outputs/train/fontaine_molmo2_ar_40k_ddp4/step_040000
MARKER=fontaine/harness/state/k_mem_ready
[ -d "$ENDPOINT" ] || { echo "no endpoint checkpoint $ENDPOINT — the ladder runs the real warm start; abort"; exit 1; }
mkdir -p fontaine/harness/state
rm -f "$MARKER"
SAMPLER_PID=""
trap '[ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true' EXIT

wait_gpus_free() { # $1 = attempts of 10 s; rc 1 if still busy
    for _ in $(seq 1 "$1"); do
        local busy=0 g mem
        for g in 0 1 2 3; do
            mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
            [ "$mem" -gt 1024 ] && busy=1
        done
        [ "$busy" -eq 0 ] && return 0
        sleep 10
    done
    return 1
}
wait_gpus_free 1 || { echo "GPUs busy — abort (ladder never preempts a live run)"; exit 1; }

run_rung() { # $1 = batch, $2 = backward chunks; sets RUNG_* facts
    local BATCH=$1 CHUNKS=$2
    local TAG="attach_k_b${BATCH}c${CHUNKS}ckpt"
    local LOG="/home/ubuntu/smoke_${TAG}.log"
    local VRAM_LOG="/home/ubuntu/smoke_${TAG}_vram.log"
    local JSONL="outputs/train/smoke_${TAG}/train_log.jsonl"
    rm -rf "outputs/train/smoke_${TAG}"
    : > "$VRAM_LOG"
    ( while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 \
          | paste -sd' ' >> "$VRAM_LOG"
        sleep 2
      done ) &
    SAMPLER_PID=$!
    set +e
    # Memory forensics: per-rank allocation-history snapshot dumped at
    # OOM (smoke-only instrument; the launcher never sets this).
    BIJOU_MEM_SNAPSHOT="/home/ubuntu/smoke_${TAG}_mem" \
    .venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.train \
        --train-data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
        --fps 30 --camera-counts 1 2 \
        --holdout-episodes 0.1 --split-seed 0 \
        --decoder flow --prompt-generate-bracket \
        --backbone allenai/Molmo2-4B --max-crops 1 \
        --backbone-init-from "$ENDPOINT" \
        --conditioning-streams residual \
        --seam-stop-grad --joint-ce \
        --fast-tokenizer mcobzarenco/bijou-checkpoints/fast_tokenizer_v2 \
        --aux-fields subgoal holding progress event visible \
        --aux-dropout 0.0 --field-dropout 0.1 \
        --self-attention-mode bidirectional --time-conditioning adarms \
        --decoder-hidden 1024 --decoder-heads 8 \
        --decoder-intermediate 4096 --decoder-cross-heads 8 \
        --chunk-size 50 \
        --camera-kind-dropout 0.1 --instruction-augment 0.5 \
        --condition-fields subgoal outcome smoothness \
        --condition-dropout 0.1 --subgoal-dropout 0.5 \
        --decoder-lr 1e-4 --backbone-text-lr 2e-5 \
        --warmup-steps 500 --weight-decay 1e-5 \
        --grad-clip 100 \
        --steps 150 --batch-size "$BATCH" \
        --zero1 --backward-chunks "$CHUNKS" --chunk-grad-allreduce \
        --activation-checkpointing \
        --num-workers 20 --prefetch-factor 4 \
        --eval-samples 64 --eval-every 100 --save-every 100 --log-every 20 \
        --seed 0 --eval-seed 0 \
        --wandb-project fontaine --wandb-run-name "smoke_${TAG}" \
        --save-dir "outputs/train/smoke_${TAG}" \
        2>&1 | tee "$LOG"
    RUNG_RC=$?
    set -e
    kill "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=""
    # `|| true` inside every substitution: pipefail + set -e must not
    # kill the LADDER when an OOMed rung left no jsonl/log rows to grep.
    RUNG_SMI_PEAK=$( { tr ' ' '\n' < "$VRAM_LOG" || true; } | sort -n | tail -1)
    RUNG_ALLOC_PEAK=$( { grep -o '"vram_alloc_peak_gib": [0-9.]*' "$JSONL" 2>/dev/null || true; } \
        | awk '{if ($2 > m) m = $2} END {if (NR) print m; else print "n/a"}')
    RUNG_RATE=$( { grep -o '"s_per_step": [0-9.]*' "$LOG" || true; } | tail -5 \
        | awk '{s+=$2} END {if (NR) printf "%.3f", s/NR; else print "n/a"}')
    RUNG_LOG="$LOG"
    RUNG_PASS=0
    if [ "$RUNG_RC" -eq 0 ] && [ "$RUNG_ALLOC_PEAK" != "n/a" ] \
        && awk -v p="$RUNG_ALLOC_PEAK" 'BEGIN {exit !(p <= 71.0)}'; then
        RUNG_PASS=1
    fi
    # Verdict tee'd into the rung log — a tmux pane dies with the
    # session, the log is the record.
    {
      echo "=== K SMOKE RUNG B${BATCH}c${CHUNKS}: rc=${RUNG_RC}, vram_alloc_peak ${RUNG_ALLOC_PEAK} GiB (gate <= 71.0), nvidia-smi peak ${RUNG_SMI_PEAK} MiB (advisory ~75000), s/step(last5) ${RUNG_RATE} => $([ "$RUNG_PASS" -eq 1 ] && echo GREEN || echo RED) ==="
      if [ "$RUNG_RATE" != "n/a" ]; then
        awk -v r="$RUNG_RATE" 'BEGIN {printf "=== projected K 10k train: %.1f GPU-h of the 70 GPU-h batch gate (advisory; attach_rate_gate.py binds at launch) ===\n", 10000 * r * 4 / 3600}'
      fi
    } | tee -a "$LOG"
}

for RUNG in "12 6" "8 4" "6 3"; do
    read -r B C <<< "$RUNG"
    echo "=== K SMOKE LADDER: rung B${B}c${C} (eff-$((B * 4))) ==="
    run_rung "$B" "$C"
    if [ "$RUNG_PASS" -eq 1 ]; then
        {
          echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
          echo "batch=$B"
          echo "backward_chunks=$C"
          echo "vram_alloc_peak_gib=$RUNG_ALLOC_PEAK"
          echo "nvidia_smi_peak_mib=$RUNG_SMI_PEAK"
          echo "s_per_step_last5=$RUNG_RATE"
          echo "log=$RUNG_LOG"
        } > "$MARKER"
        echo "=== K SMOKE LADDER GREEN at B${B}c${C} — marker written: $MARKER ==="
        if [ "$B" -ne 12 ]; then
            echo "=== MATCHED DOWNSHIFT: eff-48 does NOT fit checkpointed — BOTH arms run BATCH=$B BACKWARD_CHUNKS=$C (pre-reg: matched, never K alone) ==="
        fi
        echo "=== launch (F first, then K, SAME env): K_MEM_READY=1 BATCH=$B BACKWARD_CHUNKS=$C ==="
        exit 0
    fi
    echo "=== rung B${B}c${C} RED — draining GPUs before the next rung ==="
    pkill -KILL -f "smoke_attach_k_b${B}c${C}ckpt" 2>/dev/null || true
    wait_gpus_free 30 || { echo "GPUs did not drain after RED rung — abort, inspect by hand"; exit 1; }
done

echo "=== K SMOKE LADDER ALL-RED (B12/B8/B6 checkpointed) — NO marker; K does not launch; owner steer required (pre-reg escalation) ==="
exit 1
