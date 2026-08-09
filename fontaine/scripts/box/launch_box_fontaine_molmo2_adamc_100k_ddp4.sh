#!/usr/bin/env bash
# fontaine Molmo2-4B AdamC 100k from BASE — fontaine_molmo2_adamc_100k_ddp4.
# OWNER SPEC 2026-08-09 12:37:56Z, approved in-channel 13:19Z with
#   overrides folded (λ=0.01, text+vision 2e-5 confirmed, seed 1,
#   save-every 5000, NO smoke — direct launch, restart-on-OOM policy).
# PRE-REG / PARAMETER SHEET:
#   fontaine/blog/src/posts/2026-08-09-prereg-molmo2-adamc-100k.md
#   (amendment records the owner's decisions; posted before launch).
# Deltas vs the 40k lineage launcher (launch_box_fontaine_molmo2_ar_40k_ddp4.sh):
#   steps 100k, BATCH 8 (eff-batch 32; chunks 4 keeps per-chunk
#   microbatch 2), vision tower unfrozen from step 0 at 2e-5,
#   --optimizer adamc --weight-decay 0.01, seed 1, save-every 5000.
#   Everything else byte-identical (data recipe, aux/conditioning,
#   ZeRO-1 + chunk-grad-allreduce, warmup 1000, clip 100).
# E1 hard gate (train banner): 878 datasets / 38,571 episodes /
#   18,636,749 frames / dims 6/6. Any deviation => abort before step 1.
# Kill lines (babysit adamc_100k entry): NaN/inf; probe not below its
#   own @2500 value by 10k; probe > 25 sustained x3 after 5k; vram
#   alloc peak > 77 GiB (near-OOM watch — no smoke ran, so the sheet's
#   71 was replaced by an OOM-guard bar; actual recorded at first
#   poll). Grad-norm trajectory is a record-only AdamC watch.
# OOM policy (owner 13:19Z): no smoke; if the run OOMs, relaunch with
#   BACKWARD_CHUNKS=8 (microbatch 1), same effective batch.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

BATCH="${BATCH:-8}"
STEPS="${STEPS:-100000}"
RUN_NAME="fontaine_molmo2_adamc_100k_ddp4"
ENDPOINT_STEP=$(printf "%06d" "$STEPS")
BACKWARD_CHUNKS="${BACKWARD_CHUNKS:-4}"
CHUNK_ARGS=(--zero1)
if [ "$BACKWARD_CHUNKS" -gt 1 ]; then
    CHUNK_ARGS+=(--backward-chunks "$BACKWARD_CHUNKS" --chunk-grad-allreduce)
fi

# GPU-clear guard: all four GPUs must be free.
for g in 0 1 2 3; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$mem" -gt 1024 ]; then echo "GPU $g busy (${mem} MiB) — abort"; exit 1; fi
done

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
    --decoder-lr 1e-4 --backbone-text-lr 2e-5 --backbone-vision-lr 2e-5 \
    --optimizer adamc --weight-decay 0.01 \
    --grad-clip 100 \
    --steps "$STEPS" --warmup-steps 1000 --batch-size "$BATCH" \
    "${CHUNK_ARGS[@]}" \
    --num-workers 20 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 5000 --log-every 20 \
    --seed 1 --wandb-project fontaine \
    --wandb-run-name "$RUN_NAME" \
    --save-dir "outputs/train/$RUN_NAME" \
    2>&1 | tee "/home/ubuntu/train_${RUN_NAME}.log"

# Endpoint panel — chained, sharded, with dumps + HTML report (owner
# standing rule: --report on every important checkpoint).
.venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2.json \
    --checkpoint outputs/train/$RUN_NAME/step_$ENDPOINT_STEP \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --dump-predictions "reports/eval__${RUN_NAME}__step_${ENDPOINT_STEP}__panel_curated_v0_k4l2.npz" \
    --output-json "reports/eval__${RUN_NAME}__step_${ENDPOINT_STEP}__panel_curated_v0_k4l2.json" \
    --report "reports/eval__${RUN_NAME}__step_${ENDPOINT_STEP}__panel_curated_v0_k4l2.html" \
    2>&1 | tee "/home/ubuntu/eval_${RUN_NAME}.log"
echo "=== MOLMO2 ADAMC 100K DONE (train + endpoint panel eval) ==="
