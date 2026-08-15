#!/usr/bin/env bash
# Grasp-SFT STAGE C — OPTIONAL FLOW ARM (pre-reg §6 frozen, 2026-08-15).
# Pre-reg: fontaine/blog/src/posts/2026-08-14-prereg-grasp-sft-bootstrap.md
# CONDITIONAL: launches only after the AR primary chain lands AND the
# remaining <= 13 GPU-h pre-reg gate affords the ~1.3 GPU-h (§6: flow
# arm retained — wrist screen closed F-instrument, the §5 drop clause
# did not fire). Recipe = ftrig4k
# (launch_local_snapflow_ftrig_4k_1xh100.sh train mode, banked
# 2026-08-06 pre-reg) VERBATIM with EXACTLY the frozen §6 delta:
#
# ARG DIFF vs the banked ftrig4k train command (the verbatim receipt):
#   --train-data <2 rig repos>   -> /home/ubuntu/datasets/fontaine/grasp_sft_demos_v0
#                                   (§6: "dataset swapped to the demo set")
#   run name fontaine_flow_snapdistill_ftrig_4k_1xh100
#                                -> fontaine_grasp_sft_stagec_flow_4k
# NO other deltas: init-from student @30k, --distill snapflow, 4k steps,
# decoder LR 1e-5, batch 24, save/eval every 500, seed 0, camera-counts
# 1 2, holdout 0.1 split-seed 0. The ftrig4k launcher's chained
# after-reads are NOT inherited — stage D (sim100) is this arm's read.
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit> bash <this script>
# or systemd-run (driver guard: bare launches die at unit teardown).
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching

.venv/bin/python fontaine/scripts/grasp_sft_stagec_preflight.py

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi

STUDENT=outputs/train/fontaine_flow_snapdistill_h1024_30k_1xh100/step_030000
RUN=fontaine_grasp_sft_stagec_flow_4k
DEMO_DATA=(/home/ubuntu/datasets/fontaine/grasp_sft_demos_v0)

.venv/bin/python -m bijou.train \
    --train-data "${DEMO_DATA[@]}" \
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

echo "=== STAGE-C FLOW ARM DONE (train only; stage D sim100 is the read) ==="
