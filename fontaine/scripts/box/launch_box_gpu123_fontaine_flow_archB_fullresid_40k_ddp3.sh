#!/usr/bin/env bash
# fontaine — arch-batch-1 ARM B: full-residual conditioning
#   (--conditioning-streams residual, res0..res14 learned adapters
#   ~23.62M params replace kv4/9/14; GPUs 1-3, DDP3, B32/rank = eff-96,
#   40k steps). LAUNCH ORDER: A -> B — this waits for arm A's chain.
# PRE-REG: fontaine/blog/src/posts/2026-08-06-prereg-arch-batch-1.md
#   (+ Amendments 1-2). ONE variable vs control recipe: the
#   conditioning-stream architecture (no --stream-counts: the schedule
#   is structural under residual mode, 1:1 ascending, parser-enforced).
# PRE-LAUNCH GATE (pre-reg): all five arm-B oracles landed green at
#   d77ed58-lineage HEAD (tests/test_residual_streams.py, 11 CPU tests
#   in check.py) + box stage-0 re-verify green after the boundary sync.
#   F1: smoke_arch_ddp3_b32.sh B must have passed at the SAME B as arm
#   A (whole-batch rule; OOM mid-run => bug, B-1 ladder FORBIDDEN).
# K1 (kill): probe > teacher probe@matched step + 3.0 at any eval
#   >= 5k => kill at next save boundary
#   (reports/teacher_artrunk40k_probe_curve.json, 9.1306@5000).
# Expectations: genuinely open (mainline never tested it) — a null
#   closes mainline #4's caveat with data; adopt iff dchunk <= -0.15
#   CI95-excl-0 paired vs teacher@40k ctrl npz.
# CHAINED: panel-v2 endpoint eval, Heun-30, draws=1, --noise-key stable
#   (pinned explicitly), stems per arch_batch_results.py.
# COST: ~8-10 h train + ~1-2 h eval.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1,2,3
cd /home/ubuntu/flow-matching

RUN=fontaine_flow_archB_fullresid_40k_ddp3
BATCH=32   # must equal arm A's launched B (whole-batch rule)

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
    --conditioning-streams residual \
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
echo "=== ARM B DONE (train + panel-v2 endpoint eval: reports/${name}.json) ==="
