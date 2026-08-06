#!/usr/bin/env bash
# fontaine E4B trunk-swap screen — fontaine_arb_rcond_e4b_100k_ddp4.
# PRE-REG: fontaine/blog/src/posts/2026-08-05-prereg-e4b-screen.md
#   (posted 2026-08-05 ~22:4xZ before launch; Amendment 1 = chunked
#   backward mechanism; finalization amendment REQUIRED before launch:
#   sigma_seed from tonight's replicate panels, ladder rung from the
#   B12 memory smoke, measured smoke peak, disk check).
# Recipe: VERBATIM mainline launch_arb_rcond_100k.sh (B12/eff-48 as
#   in the recipe's launch command — NOT the post-OOM B10 resume edit)
#   + --backbone google/gemma-4-e4b-it. One variable changed.
# E1 hard gate: selection 878 datasets / 42,872 episodes / dims 6/6
#   (identical to tonight's four arms); model line = E4B geometry
#   (42 layers / hidden 2560), block base 262144 - vocab_total (same
#   value as E2B). Any selection deviation => abort before step 1.
# E2 first poll: record s/step + peak VRAM. Expected 0.9-1.1 s/step
#   at B12-equivalent; slowness is data, NOT a kill. Starving util =>
#   input fix at a safe boundary, logged.
# E3 gates: @10k record-only (kill only on divergence: probe >15
#   falling-then-rising, or NaN). @30k DECISION: kill if probe >7.07
#   AND the 25k panel does not contradict. @50k: kill if >6.29, same
#   cross-check. E2B refs: 7.54@10k, 6.57@30k, 5.79@50k (+0.5 floor).
# KILLS wait for save boundaries. Ladder rung is FROZEN at launch —
#   never change batch semantics mid-run.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train

# Ladder rung (finalization amendment fills this from the smoke):
# 1 = B12 direct; 2/3/4 = chunked backward 2x6 / 3x4 / 4x3.
BACKWARD_CHUNKS="${BACKWARD_CHUNKS:?set by finalization amendment (1=B12 direct, 2, 3, or 4)}"
CHUNK_ARGS=()
if [ "$BACKWARD_CHUNKS" -gt 1 ]; then
    CHUNK_ARGS=(--backward-chunks "$BACKWARD_CHUNKS")
fi

# GPU-clear guard: all four GPUs must be free (box batch + evals done).
for g in 0 1 2 3; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$mem" -gt 1024 ]; then echo "GPU $g busy (${mem} MiB) — abort"; exit 1; fi
done

.venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.train \
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
    --steps 100000 --warmup-steps 1000 --batch-size 12 \
    "${CHUNK_ARGS[@]}" \
    --num-workers 20 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 2500 --log-every 20 \
    --seed 0 --wandb-project fontaine \
    --wandb-run-name fontaine_arb_rcond_e4b_100k_ddp4 \
    --save-dir outputs/train/fontaine_arb_rcond_e4b_100k_ddp4 \
    2>&1 | tee /home/ubuntu/train_fontaine_arb_rcond_e4b_100k_ddp4.log

# E5 endpoint panel — matched eval command, 4-GPU sharded, with dumps
# (per-frame paired analysis vs the 5.8026 E2B anchor npz).
.venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2.json \
    --checkpoint outputs/train/fontaine_arb_rcond_e4b_100k_ddp4/step_100000 \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --dump-predictions reports/eval__fontaine_arb_rcond_e4b_100k_ddp4__step_100000__panel_curated_v0_k4l2.npz \
    --output-json reports/eval__fontaine_arb_rcond_e4b_100k_ddp4__step_100000__panel_curated_v0_k4l2.json \
    --report reports/eval__fontaine_arb_rcond_e4b_100k_ddp4__step_100000__panel_curated_v0_k4l2.html \
    2>&1 | tee /home/ubuntu/eval_fontaine_arb_rcond_e4b_100k_ddp4_100k.log
echo "=== E4B DONE (train + endpoint panel eval) ==="
