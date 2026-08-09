#!/usr/bin/env bash
# fontaine Molmo2-ER 60k AR run — fontaine_molmo2_er_60k_ddp4.
# PRE-REG: fontaine/blog/src/posts/2026-08-09-prereg-molmo2-er-60k.md
#   (owner spec 22:14Z 08-09; go 22:36Z; param sheet approved 22:45Z:
#   "Uniform sampling across all data, including my datasets is fine,
#   I'll fine-tune later. Parameters look good").
# Recipe: 40k AR launcher verbatim (launch_box_fontaine_molmo2_ar_40k_ddp4.sh)
#   with the pre-reg's named deltas:
#   1. --steps 60000
#   2. --backbone allenai/Molmo2-ER (byte-verified drop-in vs Molmo2-4B)
#   3. rig datasets in --train-data from step 0 (natural/uniform share
#      per owner 22:45Z — no oversampling)
#   4. --save-every 5000 (~12 saves at 60k)
#   5. --seed 0 — owner override 22:46:40Z ('let's use the same seed
#      too'): SAME shuffle seed as the 40k run; the fresh-seed standing
#      rule is explicitly owner-overridden for this run. First launch
#      22:50Z at seed 2 was stopped PRE-STEP-1 at 22:52Z and relaunched.
# E1 banner expectation (train banner, verify at first poll):
#   880 datasets / 38,628 episodes / 18,672,827 frames / dims 6/6
#   pre-holdout (curated_v0 878/38,571/18,636,749 + rig clean 7 ep
#   3,399 fr + rig v2 50 ep 32,679 fr). Deviation => kill + diagnose.
# Kill lines (finalized pre-reg): NaN/inf loss; probe bars re-derived
#   from the 40k curve; vram near-OOM bar per first-poll actuals.
#   Kills wait for save boundaries.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

BATCH="${BATCH:-12}"
STEPS="${STEPS:-60000}"
RUN_NAME="fontaine_molmo2_er_$((STEPS / 1000))k_ddp4"
ENDPOINT_STEP=$(printf "%06d" "$STEPS")
# Memory rung inherited from the 40k run (measured green there at
# identical batch/chunks/trunk-shape): 6-chunk backward + ZeRO-1 +
# --chunk-grad-allreduce (skips the DDP wrapper; oracle-tested).
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
                 /home/ubuntu/datasets/mcobzarenco/so101_pick_place_clean \
                 /home/ubuntu/datasets/mcobzarenco/so101_pick_place_v2 \
    --fps 30 --camera-counts 1 2 \
    --holdout-episodes 0.1 --split-seed 0 \
    --decoder ar_backbone \
    --backbone allenai/Molmo2-ER \
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
    --eval-samples 256 --eval-every 500 --save-every 5000 --log-every 20 \
    --seed 0 --wandb-project fontaine \
    --wandb-run-name "$RUN_NAME" \
    --save-dir "outputs/train/$RUN_NAME" \
    2>&1 | tee "/home/ubuntu/train_${RUN_NAME}.log"

# Endpoint panel — k4l2 panel_v2, sharded, with dumps for the paired
# per-frame CI95 vs the banked 40k endpoint and 60k-continuation npz.
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
echo "=== MOLMO2 ER 60K DONE (train + endpoint panel eval) ==="
