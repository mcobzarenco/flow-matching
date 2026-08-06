#!/usr/bin/env bash
# fontaine — arm C: state-dropout 0.8 paired against banked A-s0 (GPU 0).
# PRE-REG: fontaine/blog/src/posts/2026-08-06-prereg-state-dropout-40k.md
#   (ideas #9; branch rule fired by the state-reliance probe, D=+0.702).
# ONE variable vs A-s0 (launch_box_gpu0_fontaine_arb_rcond_40k_1xh100.sh
#   @ cc0b922): --state-dropout 0.8. Same seed 0, same 40k/B10, same data.
# E1 hard gate: 878 datasets / 42,872 episodes / dims 6/6 — deviation =>
#   abort before step 1. Banner must show "state dropout: p=0.8".
# E2: 0.4-0.6 s/step @ B10 (idle box — no contention allowance); VRAM
#   < 76 GiB; sustained > 0.8 s/step => input-pipeline fix at a boundary.
# E3 (probe scores INTACT state — comparable to A-s0's trajectory):
#   expect <13 @10k, <10.5 @30k; kill: >13 @10k after falling-then-
#   rising; NaN; second OOM after standing B-1 resume. Formal final
#   gate: probe < 10 @40k (else p=0.8 too aggressive at this rung).
# Chained after training: full panel eval (paired vs A-s0's banked npz)
#   then masked-subset eval (state-probe q4 plan) — the reliance readout.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
cd /home/ubuntu/flow-matching

# GPU-clear guard (this GPU only)
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi

# Subset-plan integrity (the masked eval consumes it at the end; assert
# BEFORE burning 5 h of training).
echo "876c39c8fe2b3cb16945a40c35ec157c907b4f7417e7dfd0b6cf46dd47355ef5  plans/holdout_curated_v0_k4l2_stateprobe_q4.json" | sha256sum -c -

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
    --state-dropout 0.8 \
    --decoder-lr 1e-4 --backbone-text-lr 2e-5 --grad-clip 100 \
    --steps 40000 --warmup-steps 1000 --batch-size 10 \
    --num-workers 16 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 5000 --log-every 20 \
    --seed 0 --wandb-project fontaine \
    --wandb-run-name fontaine_arb_rcond_statedrop80_40k_1xh100 \
    --save-dir outputs/train/fontaine_arb_rcond_statedrop80_40k_1xh100 \
    2>&1 | tee /home/ubuntu/train_fontaine_arb_rcond_statedrop80_40k_1xh100.log

.venv/bin/python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2.json \
    --checkpoint outputs/train/fontaine_arb_rcond_statedrop80_40k_1xh100/step_040000 \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --dump-predictions reports/eval__fontaine_arb_rcond_statedrop80_40k_1xh100__step_040000__panel_curated_v0_k4l2.npz \
    --output-json reports/eval__fontaine_arb_rcond_statedrop80_40k_1xh100__step_040000__panel_curated_v0_k4l2.json \
    --report reports/eval__fontaine_arb_rcond_statedrop80_40k_1xh100__step_040000__panel_curated_v0_k4l2.html \
    2>&1 | tee /home/ubuntu/eval_fontaine_arb_rcond_statedrop80_40k_1xh100_40k.log

.venv/bin/python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2_stateprobe_q4.json \
    --mask-state \
    --checkpoint outputs/train/fontaine_arb_rcond_statedrop80_40k_1xh100/step_040000 \
    --batch-size 32 --num-workers 20 \
    --dump-predictions reports/eval__fontaine_arb_rcond_statedrop80_40k_1xh100__step_040000__stateprobe_q4_state-masked.npz \
    --output-json reports/eval__fontaine_arb_rcond_statedrop80_40k_1xh100__step_040000__stateprobe_q4_state-masked.json \
    --report reports/eval__fontaine_arb_rcond_statedrop80_40k_1xh100__step_040000__stateprobe_q4_state-masked.html \
    2>&1 | tee /home/ubuntu/eval_fontaine_arb_rcond_statedrop80_40k_1xh100_40k_statemasked.log
echo "=== ARM C DONE (train + panel eval + masked reliance eval) ==="
