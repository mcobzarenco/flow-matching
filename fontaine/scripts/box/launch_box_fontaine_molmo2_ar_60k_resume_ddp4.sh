#!/usr/bin/env bash
# fontaine Molmo2-4B AR CONTINUATION — fontaine_molmo2_ar_60k_ddp4.
# PRE-REG: fontaine/blog/src/posts/2026-08-08-prereg-molmo2-ar-60k-continuation.md
#   (owner GO 09:04Z 08-08, prioritized over the attach screen).
# LAUNCH (on the box, via its run_detached wrapper):
#   fontaine/scripts/run_detached.sh fontaine-molmo2-60k \
#       bash fontaine/scripts/box/launch_box_fontaine_molmo2_ar_60k_resume_ddp4.sh
# Design: --resume the 40k endpoint (weights+optimizer+step), +20k to
#   60k total, --rewarmup-steps 1000 (re-warm + re-decay: cosine over
#   60k puts step-40k at 0.332x peak, floor 1e-5 at 60k), --seed 1
#   (fresh shuffle; check_resume_seed aborts on seed reuse). EVERY
#   other flag byte-identical to launch_box_fontaine_molmo2_ar_40k_ddp4.sh.
# E1 hard gate (train banner): 878 datasets / 38,571 episodes /
#   18,636,749 frames / dims 6/6 — any deviation => abort before step 1.
#   Resume banner must show 'resumed optimizer/scheduler at step 40000'.
# K1 kill lines (pre-reg): NaN/inf loss; probe > 8.2075 sustained x3
#   evals after step 41,500; vram 71 GiB. Judged at save boundaries.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

BATCH="${BATCH:-12}"
STEPS=60000
RUN_NAME="fontaine_molmo2_ar_60k_ddp4"
ENDPOINT_STEP=$(printf "%06d" "$STEPS")
RESUME_FROM="outputs/train/fontaine_molmo2_ar_40k_ddp4/step_040000"
BACKWARD_CHUNKS="${BACKWARD_CHUNKS:-6}"
CHUNK_ARGS=(--zero1)
if [ "$BACKWARD_CHUNKS" -gt 1 ]; then
    CHUNK_ARGS+=(--backward-chunks "$BACKWARD_CHUNKS" --chunk-grad-allreduce)
fi

[ -f "$RESUME_FROM/optimizer.pt" ] || {
    echo "no optimizer.pt in $RESUME_FROM — not a resumable save; abort"; exit 1; }

# GPU-clear guard: all four GPUs must be free.
for g in 0 1 2 3; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$mem" -gt 1024 ]; then echo "GPU $g busy (${mem} MiB) — abort"; exit 1; fi
done

# Amendment 1 (owner 10:06Z): the two staged SO101 rig datasets join
# the mix (57 eps / 36,078 frames ≈ 0.19% of the corpus; panel plan
# untouched — evals stay comparable). Expected E1 banner becomes
# 880 datasets / 38,628 episodes / 18,672,827 frames; a banner still
# reading 878 means the rig sets were filtered out (camera/fps) — stop
# and report, don't proceed silently.
.venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.train \
    --train-data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
        /home/ubuntu/datasets/mcobzarenco/so101_pick_place_clean \
        /home/ubuntu/datasets/mcobzarenco/so101_pick_place_v2 \
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
    --steps "$STEPS" --warmup-steps 1000 --rewarmup-steps 1000 \
    --resume "$RESUME_FROM" \
    --batch-size "$BATCH" \
    "${CHUNK_ARGS[@]}" \
    --num-workers 20 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 2500 --log-every 20 \
    --seed 1 --wandb-project fontaine \
    --wandb-run-name "$RUN_NAME" \
    --save-dir "outputs/train/$RUN_NAME" \
    2>&1 | tee "/home/ubuntu/train_${RUN_NAME}.log"

# Endpoint panel — same stems pattern as the 40k launcher, dumps
# retained for the paired per-frame Δ vs the banked 40k endpoint npz
# (pre-reg read 1).
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
echo "=== MOLMO2 AR 60K CONTINUATION DONE (train + endpoint panel eval) ==="
