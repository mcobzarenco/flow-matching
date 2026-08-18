#!/usr/bin/env bash
# Browsable HTML panel for grasp_sft_v2_demosonly_1gpu_disc/step_001000
# (queue: disc-step1000-html-report; owner standing rule — every
# important checkpoint gets an HTML report linked from reports/).
# Step-1000 is the first non-drifting v2-corpus checkpoint (verdict
# HEALTHY 00:42Z 08-18, banked on fontaine-checkpoints).
#
# CURRENT-stack eval on the probe-matched pins — every flag mirrors the
# discriminator's in-train probe (and stack_parity_probe.sh):
#   --data ~/datasets/fontaine/grasp_demos_v2/merged   (same local snapshot)
#   --episodes holdout --holdout-episodes 0.1 --split-seed 0
#   --num-samples 256 --seed 0
#   --chunk-size 30
#   --sample-method euler --sample-steps 10
#   --batch-size 12
# Integrity anchor: the in-train probe read 5.8989 @1000 on this exact
# instrument — the report's headline chunk-MAE must reproduce it.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching

# Owner policy-server guard (port 8144 claims the H100 silently).
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "ABORT: GPU busy — compute apps present:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 2
fi

RUN=grasp_sft_v2_demosonly_1gpu_disc
STEP=step_001000
NAME="eval__${RUN}__${STEP}__demos_holdout256_euler10"
mkdir -p reports

uv run python -m bijou.eval \
    --data ~/datasets/fontaine/grasp_demos_v2/merged \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --checkpoint ~/checkpoints/finetune/$RUN/$STEP \
    --num-samples 256 --seed 0 \
    --chunk-size 30 \
    --sample-method euler --sample-steps 10 \
    --batch-size 12 --num-workers 4 \
    --report-samples 32 \
    --output-json "reports/${NAME}.json" \
    --report "reports/${NAME}.html" \
    2>&1 | tee "/home/ubuntu/${NAME}.log"
echo "=== disc-1000 HTML report eval DONE ==="
