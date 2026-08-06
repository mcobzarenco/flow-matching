#!/usr/bin/env bash
# fontaine — SnapFlow distill @10k record-only 1-NFE probe (BOX variant).
# PRE-REG: fontaine/blog/src/posts/2026-08-06-prereg-snapflow-distill.md
#   1-NFE (euler-1, --target-time zero) on the stride-7 probe subset
#   (2,458 frames). RECORD-ONLY: kill line fires only if probe
#   chunk-MAE > teacher Heun-30 stride-7 read (6.6755) + 3.0 = 9.6755.
# EXECUTION (charter §3): runs on box GPU 1 — arm C owns GPU 0; this
#   never co-locates (separate GPU, CUDA_VISIBLE_DEVICES pins it).
#   Checkpoint assembled on-box: teacher backbone byte-identical
#   (sha256 3095474d… verified vs local) + rsynced expert/config/prompt.
#   Box code bcbf101 (has --target-time; predates the live arm C run —
#   no code sync performed, per never-sync-under-live-run).
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

mem=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU 1 busy (${mem} MiB) — abort"; exit 1; fi

CKPT=${1:-outputs/train/fontaine_flow_snapdistill_h1024_30k_1xh100/step_010000}
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
echo "kill line: probe chunk-MAE > 9.6755 (teacher 6.6755 + 3.0)"
