#!/usr/bin/env bash
# fontaine box batch — A-s0 (ctl, seed 0) on GPU 0.
# PRE-REG: fontaine/blog/src/posts/2026-08-05-prereg-box-batch-4xh100.md
#   (commit cc0b922; science: 2026-08-05-prereg-paired-auxoff-40k.md)
# E1 hard gate: 878 datasets / 42,872 episodes / dims 6/6 — any
#   deviation => abort the whole batch. B-s0: no loss_aux in logs.
# E2: 0.4-0.7 s/step @ B10 (contention allowance); VRAM < 76 GiB.
# E3: probe <12 @10k, <9 @30k (ctl); kill: >15 @10k after
#   falling-then-rising; NaN; second OOM after standing B-1 resume.
#   A-s0 killed => kill B-s0 too (pair void); replicates continue.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
cd /home/ubuntu/flow-matching

# GPU-clear guard (this GPU only)
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi

.venv/bin/python -m bijou.train \
    --train-data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --fps 30 --camera-counts 1 2 \
    --holdout-episodes 0.1 --split-seed 0 \
    --decoder ar_backbone \
    --fast-tokenizer mcobzarenco/bijou-checkpoints/fast_tokenizer_v2 \
    --aux-fields subgoal holding progress event visible \
    --aux-dropout 0.0 --field-dropout 0.1 \
    --condition-fields subgoal outcome smoothness \
    --condition-dropout 0.1 --subgoal-dropout 0.5 \
    --instruction-augment 0.5 --camera-kind-dropout 0.1 \
    --decoder-lr 1e-4 --backbone-text-lr 2e-5 --grad-clip 100 \
    --steps 40000 --warmup-steps 1000 --batch-size 10 \
    --num-workers 16 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 5000 --log-every 20 \
    --seed 0 --wandb-project fontaine \
    --wandb-run-name fontaine_arb_rcond_40k_1xh100 \
    --save-dir outputs/train/fontaine_arb_rcond_40k_1xh100 \
    2>&1 | tee /home/ubuntu/train_fontaine_arb_rcond_40k_1xh100.log

.venv/bin/python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2.json \
    --checkpoint outputs/train/fontaine_arb_rcond_40k_1xh100/step_040000 \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --dump-predictions reports/eval__fontaine_arb_rcond_40k_1xh100__step_040000__panel_curated_v0_k4l2.npz \
    --output-json reports/eval__fontaine_arb_rcond_40k_1xh100__step_040000__panel_curated_v0_k4l2.json \
    --report reports/eval__fontaine_arb_rcond_40k_1xh100__step_040000__panel_curated_v0_k4l2.html \
    2>&1 | tee /home/ubuntu/eval_fontaine_arb_rcond_40k_1xh100_40k.log
echo "=== A-s0 DONE (train + panel eval) ==="
