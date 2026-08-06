#!/usr/bin/env bash
# fontaine Molmo2-4B AR phase-1 run — fontaine_molmo2_ar_40k_ddp4.
# PRE-REG: fontaine/blog/src/posts/2026-08-06-prereg-molmo2-ar-40k.md
#   (posted before launch; finalization amendment carries the smoke
#   peak/rate + the batch rung). Owner steering 18:10/18:12/19:11Z.
# Recipe: mainline arb_rcond verbatim where the trunk allows +
#   --backbone allenai/Molmo2-4B --max-crops 1, e4b-screen scale-out
#   (4xDDP, B12/rank, global 48). One trunk variable + declared
#   batch confound (pre-reg §2).
# E1 hard gate (train banner): 878 datasets / 38,571 episodes /
#   18,636,749 frames / dims 6/6 (42,872 episodes pre-holdout).
#   Any deviation => abort before step 1.
# K1 kill line (pre-reg §4): NaN/inf loss; probe not below its own
#   @2500 value by 10k; probe > 25 sustained x3 evals after 5k.
#   Kills wait for save boundaries. Batch semantics FROZEN at launch.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

BATCH="${BATCH:-12}"
# F2 branch (pre-reg §3): if the smoke rate projects the 40k wall over
# 30 h, the schedule SHRINKS to a 10k screen — STEPS=10000 relabels
# run name, save dir, log and the endpoint step consistently.
STEPS="${STEPS:-40000}"
RUN_NAME="fontaine_molmo2_ar_$((STEPS / 1000))k_ddp4"
ENDPOINT_STEP=$(printf "%06d" "$STEPS")
# Rung from the F1 smoke (pre-reg §3 rung 8): 6x2 chunked backward +
# ZeRO-1 (--zero1, Adam moments ~1/4 per rank, semantics exact) +
# --chunk-grad-allreduce, which now SKIPS the DDP wrapper entirely —
# the snapshot showed DDP's reducer buckets (13.6 GiB, a full fp32
# grad copy) are allocated at CONSTRUCTION. Rank-0 state broadcast at
# init + one explicit in-place grad allreduce per step; gradient
# equal to the DDP sync up to fp reduction order (oracle-tested).
# Global 48 unchanged.
BACKWARD_CHUNKS="${BACKWARD_CHUNKS:-6}"
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
    --decoder-lr 1e-4 --backbone-text-lr 2e-5 --grad-clip 100 \
    --steps "$STEPS" --warmup-steps 1000 --batch-size "$BATCH" \
    "${CHUNK_ARGS[@]}" \
    --num-workers 20 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 2500 --log-every 20 \
    --seed 0 --wandb-project fontaine \
    --wandb-run-name "$RUN_NAME" \
    --save-dir "outputs/train/$RUN_NAME" \
    2>&1 | tee "/home/ubuntu/train_${RUN_NAME}.log"

# Endpoint panel — the family voice, 4-GPU sharded, with dumps for the
# paired per-frame Δ vs the A-s0 anchor npz (pre-reg §5 stems).
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
echo "=== MOLMO2 AR 40K DONE (train + endpoint panel eval) ==="
