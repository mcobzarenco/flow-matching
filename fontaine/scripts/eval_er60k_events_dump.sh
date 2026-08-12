#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh eval-er60k-events-dump bash <this script>
# fontaine — er_60k @60000 events one-off DUMP PASS (owner request
# 12:44Z 08-11; rides posts/2026-08-09-prereg-molmo2-er-60k.md,
# record-only). Launch note with pinned invocation + confusion/probe
# spec posted in-channel 14:1xZ 08-11 before any GPU minute.
#
# Single narrated arm via explicit --generate (all trained fields):
# byte-identical greedy voice to the banked standard eval's +fields
# arm, so the aux-vs-weak-labels event acc printed at the end must
# reproduce the endpoint eval's 0.858 — the instrument oracle for the
# --dump-generations retention path (commit 7f43c54).
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=fontaine_molmo2_er_60k_ddp4
CKPT=outputs/train/${RUN}/step_060000
name="eval__${RUN}__step_060000__panel_curated_v0_k4l2_events"
uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2.json \
    --checkpoint "$CKPT" \
    --generate subgoal holding progress event visible \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${name}.json" \
    --dump-generations "reports/${name}_generations.json" \
    2>&1 | tee "/home/ubuntu/logs/${name}.log"
echo "=== er60k events dump pass DONE (rc=0) ==="
