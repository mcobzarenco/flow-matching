#!/usr/bin/env bash
# fontaine — SnapFlow student -> rig fine-tune (1-NFE, owner-steered
#   16:35Z "queue asap"), local 1xH100.
# PRE-REG: fontaine/blog/src/posts/2026-08-06-prereg-snapflow-ftrig.md
# Usage: ./launch_local_snapflow_ftrig_4k_1xh100.sh [r0|train]
#   r0    — R0 baseline: un-tuned student @30k on the rig holdout,
#           1-NFE euler-1, draws 1 + mean-of-10, stable keying (the
#           paired "before"; REQUIRED banked before train launches).
#   train — the 4k ft + chained after-reads (rig holdout draws 1/10 +
#           community panel-v2 1-NFE forgetting read).
# Recipe: student train_args verbatim + EXACTLY the pre-reg deltas
#   (rig-only data, init-from student, 4k steps, LR 1e-5, save 500).
# E1: banner must show 2 datasets, dims 6/6, distill snapflow, phi_s
#   present, strict init load (NO extension branch). E2: 0.45-0.65
#   s/step @ B24. K1: NaN, or probe > first-read+3.0 at 3 consecutive
#   evals >= 1.5k => kill at next save boundary.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching

MODE="${1:?usage: launch_local_snapflow_ftrig_4k_1xh100.sh r0|train}"
STUDENT=outputs/train/fontaine_flow_snapdistill_h1024_30k_1xh100/step_030000
RUN=fontaine_flow_snapdistill_ftrig_4k_1xh100
RIG_DATA=(/home/ubuntu/datasets/mcobzarenco/so101_pick_place_v2
          /home/ubuntu/datasets/mcobzarenco/so101_pick_place_clean)

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi

rig_eval () {  # rig_eval <checkpoint> <stem> <draws>
  .venv/bin/python -m bijou.eval \
      --data "${RIG_DATA[@]}" \
      --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
      --fps 30 --camera-counts 1 2 \
      --checkpoint "$1" \
      --sample-draws "$3" --sample-steps 1 --sample-method euler \
      --target-time zero --noise-key stable \
      --batch-size 32 --num-workers 8 --seed 0 \
      --report-samples 32 \
      --dump-predictions "reports/$2.npz" \
      --output-json "reports/$2.json" \
      --report "reports/$2.html" \
      2>&1 | tee "/home/ubuntu/$2.log"
}

if [ "$MODE" = "r0" ]; then
  rig_eval "$STUDENT" \
    "eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__rig_holdout_1nfe_euler1_stable" 1
  rig_eval "$STUDENT" \
    "eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__rig_holdout_1nfe_euler1_stable_draws10" 10
  echo "=== R0 BANKED (rig-holdout before-reads, draws 1 + 10) ==="
  exit 0
fi

[ "$MODE" = "train" ] || { echo "mode must be r0 or train"; exit 2; }
for d in 1 10; do
  stem="eval__fontaine_flow_snapdistill_h1024_30k_1xh100__step_030000__rig_holdout_1nfe_euler1_stable"
  [ "$d" = 10 ] && stem="${stem}_draws10"
  [ -f "reports/${stem}.json" ] || { echo "R0 read missing (${stem}) — run r0 first"; exit 1; }
done

.venv/bin/python -m bijou.train \
    --train-data "${RIG_DATA[@]}" \
    --fps 30 --camera-counts 1 2 \
    --holdout-episodes 0.1 --split-seed 0 \
    --decoder flow --prompt-generate-bracket \
    --backbone google/gemma-4-e2b-it \
    --init-from "$STUDENT" \
    --distill snapflow \
    --stream-counts 4 4 7 \
    --self-attention-mode bidirectional --time-conditioning adarms \
    --decoder-hidden 1024 --decoder-heads 8 \
    --decoder-intermediate 4096 --decoder-cross-heads 8 \
    --chunk-size 50 \
    --camera-kind-dropout 0.1 --instruction-augment 0.5 \
    --condition-fields subgoal outcome smoothness \
    --condition-dropout 0.1 --subgoal-dropout 0.5 \
    --decoder-lr 1e-5 --warmup-steps 500 --weight-decay 1e-5 \
    --grad-clip 1.0 \
    --steps 4000 --batch-size 24 \
    --num-workers 20 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 500 --log-every 20 \
    --seed 0 --eval-seed 0 \
    --wandb-project fontaine --wandb-run-name "$RUN" \
    --save-dir outputs/train/${RUN} \
    2>&1 | tee /home/ubuntu/train_${RUN}.log

# Chained after-reads: transfer (rig holdout, paired vs R0) + guard
# (community panel-v2 1-NFE forgetting read).
rig_eval "outputs/train/${RUN}/step_004000" \
  "eval__${RUN}__step_004000__rig_holdout_1nfe_euler1_stable" 1
rig_eval "outputs/train/${RUN}/step_004000" \
  "eval__${RUN}__step_004000__rig_holdout_1nfe_euler1_stable_draws10" 10

name="eval__${RUN}__step_004000__panel_v2_1nfe_euler1_stable"
.venv/bin/python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2_panel_v2.json \
    --checkpoint "outputs/train/${RUN}/step_004000" \
    --sample-draws 1 --sample-steps 1 --sample-method euler \
    --target-time zero --noise-key stable \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --output-json "reports/${name}.json" \
    --dump-predictions "reports/${name}.npz" \
    --report "reports/${name}.html" \
    2>&1 | tee "/home/ubuntu/${name}.log"
echo "=== FT-RIG DONE (train + rig after-reads + panel-v2 forgetting read) ==="
