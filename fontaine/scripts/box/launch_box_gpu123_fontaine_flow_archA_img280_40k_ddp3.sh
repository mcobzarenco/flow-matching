#!/usr/bin/env bash
# fontaine — arch-batch-1 ARM A: bigger images, --max-soft-tokens 280
#   (GPUs 1-3, DDP3, B32/rank = eff-96, 40k steps).
# PRE-REG: fontaine/blog/src/posts/2026-08-06-prereg-arch-batch-1.md
#   (+ Amendment 1: control := teacher@40k, K1 vs teacher probe curve;
#    + Amendment 2, owner 12:59Z: primary rung 280, 560 demoted).
# ONE variable vs control recipe: --max-soft-tokens 280 (teacher trained
#   at 140). Everything else teacher-verbatim per the pre-reg common
#   recipe; stated deltas only: 40k vs 80k, DDP3xB32 vs DDP2xB48 (same
#   eff-96), seed 0, our box, num-workers 20/rank.
# GATES (pre-registered):
#   F1/F2: smoke_arch_ddp3_b32.sh A must have passed at B32 (else this
#     launcher's BATCH line is edited to the smoke-decided B for BOTH
#     arms before either launches; OOM mid-run => bug, NOT a B-1 resume
#     — the B-1 ladder is FORBIDDEN here, it would break matching).
#   K1 (kill): in-run probe > teacher probe@matched step + 3.0 at any
#     eval >= 5k => kill at next save boundary. Teacher curve:
#     reports/teacher_artrunk40k_probe_curve.json (9.1306@5000);
#     babysits run arch_batch_results.py --k1-train-log.
#   E (endpoint band, control eval banked 13:47Z): teacher@40k panel-v2
#     ctrl = 7.1041/2.0720, inside [6.7, 7.9]/[1.90, 2.35]. Arm A modal
#     outcome |dchunk| < 0.15 (explore arm); tail worth buying is
#     dfirst -0.1 to -0.3.
# CHAINED after training: panel-v2 endpoint eval on rank-0 GPU (1),
#   Heun-30, draws=1, --noise-key stable (pinned EXPLICITLY per the
#   d9dd385 lesson, never inherited), seed 0, npz+json+html, stems per
#   arch_batch_results.py instrument:
#   eval__fontaine_flow_archA_img280_40k_ddp3__step_040000__panel_v2_heun30_draws1_stable
# COST: ~12-16 h train (F2-checked at smoke) + ~1-2 h eval; disk ~30-45
#   GB at save-every 2500 (prune to endpoint+latest after reads,
#   uploads before deletions).
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1,2,3
cd /home/ubuntu/flow-matching

RUN=fontaine_flow_archA_img280_40k_ddp3
BATCH=32   # F1-decided; edit ONLY per the smoke's whole-batch rule

for gpu in 1 2 3; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
  if [ "$mem" -gt 1024 ]; then echo "GPU ${gpu} busy (${mem} MiB) — abort"; exit 1; fi
done

.venv/bin/torchrun --standalone --nproc-per-node=3 -m bijou.train \
    --train-data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
                 /home/ubuntu/datasets/mcobzarenco/so101_pick_place_v2 \
                 /home/ubuntu/datasets/mcobzarenco/so101_pick_place_clean \
    --fps 30 --camera-counts 1 2 \
    --holdout-episodes 0.1 --split-seed 0 \
    --decoder flow --prompt-generate-bracket \
    --backbone google/gemma-4-e2b-it \
    --backbone-init-from outputs/train/bijou_arb_rcond_100k_ddp4/step_100000 \
    --max-soft-tokens 280 \
    --stream-counts 4 4 7 \
    --self-attention-mode bidirectional --time-conditioning adarms \
    --decoder-hidden 1024 --decoder-heads 8 \
    --decoder-intermediate 4096 --decoder-cross-heads 8 \
    --chunk-size 50 \
    --camera-kind-dropout 0.1 --instruction-augment 0.5 \
    --condition-fields subgoal outcome smoothness \
    --condition-dropout 0.1 --subgoal-dropout 0.5 \
    --decoder-lr 1e-4 --warmup-steps 500 --weight-decay 1e-5 \
    --grad-clip 10.0 \
    --steps 40000 --batch-size "$BATCH" \
    --num-workers 20 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 2500 --log-every 20 \
    --seed 0 --eval-seed 0 \
    --wandb-project fontaine --wandb-run-name "$RUN" \
    --save-dir outputs/train/${RUN} \
    2>&1 | tee /home/ubuntu/train_${RUN}.log

# Chained endpoint eval (rank-0 GPU only; GPUs 2-3 free for arm B prep).
export CUDA_VISIBLE_DEVICES=1
name="eval__${RUN}__step_040000__panel_v2_heun30_draws1_stable"
.venv/bin/python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2_panel_v2.json \
    --checkpoint outputs/train/${RUN}/step_040000 \
    --sample-draws 1 --sample-steps 30 --sample-method heun \
    --noise-key stable \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --output-json "reports/${name}.json" \
    --dump-predictions "reports/${name}.npz" \
    --report "reports/${name}.html" \
    2>&1 | tee "/home/ubuntu/${name}.log"
echo "=== ARM A DONE (train + panel-v2 endpoint eval: reports/${name}.json) ==="
