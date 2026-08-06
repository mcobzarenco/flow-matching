#!/usr/bin/env bash
# fontaine — SnapFlow distill @10k record-only 1-NFE probe.
# PRE-REG: fontaine/blog/src/posts/2026-08-06-prereg-snapflow-distill.md
#   1-NFE (euler-1, --target-time zero) on the stride-7 probe subset
#   (2,458 frames) against a saved distill checkpoint. RECORD-ONLY:
#   the kill line fires only if the probe chunk-MAE exceeds the
#   teacher's own Heun-30 probe read by > 3.0 (catastrophic
#   non-convergence — SnapFlow's claim is endpoint near-parity, so
#   mid-run reads inside that margin never kill).
# EXECUTION NOTE (charter §3, no co-located GPU jobs): the local GPU is
#   busy with the distill run at @10k. Take this read from a quiet GPU
#   only — either push step_010000 to a free box GPU, or run it
#   retroactively right after the run ends (still the pre-registered
#   probe-curve evidence if the endpoint misses). The in-run
#   eval_chunk_mae (s=t, every 500 steps) is the live divergence watch.
# Usage: ./probe_snapflow_10k_1nfe.sh [checkpoint_dir]
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

CKPT=${1:-outputs/train/bijou_flow_snapdistill_h1024_30k_1xh100/step_010000}
STEP=$(basename "$CKPT")
name="eval__snapdistill__${STEP}__probe_s7_1nfe_euler1"
uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2_drawsprobe_s7.json \
    --checkpoint "$CKPT" \
    --sample-steps 1 --sample-method euler --target-time zero \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${name}.json" \
    2>&1 | tee "/home/ubuntu/${name}.log"
echo "record-only read banked: reports/${name}.json"
echo "kill line: probe chunk-MAE > (teacher Heun-30 stride-7 read) + 3.0"
