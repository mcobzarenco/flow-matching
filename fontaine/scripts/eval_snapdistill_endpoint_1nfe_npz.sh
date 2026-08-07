#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
# fontaine — SnapFlow endpoint 1-NFE primary, npz-dump addendum.
# WHY THIS EXISTS: the launcher's chained stage-4 endpoint evals
#   (launch_local_snapflow_distill_30k_1xh100.sh) dump JSON+HTML only —
#   no npz. The pre-reg's per-step horizon read (paired-analysis
#   protocol, ships with the results post) and the panel-v2 descriptive
#   column both need per-frame predictions. The launcher was already
#   running when the gap was caught (2026-08-06 09:xxZ), and editing a
#   live bash script is unsafe — so this addendum re-runs the PRIMARY
#   eval (1-NFE single draw) with --dump-predictions, at the boundary
#   AFTER the chained evals finish. snapflow_results.py cross-checks
#   its pooled numbers against the chained primary JSON (the registered
#   read) and computes the per-step + v2 reads from the npz.
# SEMANTICS PINNED EXPLICITLY (the d9dd385 lesson — no silent
#   inheritance): --noise-key index is the REGISTERED keying for every
#   SnapFlow comparator (the pre-reg predates the #18.2 stable-key
#   adoption; in-flight reads finish as registered). A future default
#   flip must not shift this eval.
# COST: 1-NFE single draw over the 25.8k-frame panel ≈ 30-40 min
#   (cheaper than the ~34-min heun-10 draws-1 run 5 precedent).
# Usage: ./eval_snapdistill_endpoint_1nfe_npz.sh [checkpoint_dir]
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=fontaine_flow_snapdistill_h1024_30k_1xh100
CKPT=${1:-outputs/train/${RUN}/step_030000}
STEP=$(basename "$CKPT")
name="eval__${RUN}__${STEP}__panel_curated_v0_k4l2_1nfe_euler1_npz"
uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2.json \
    --checkpoint "$CKPT" \
    --sample-draws 1 --sample-steps 1 --sample-method euler \
    --target-time zero --noise-key index \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${name}.json" \
    --dump-predictions "reports/${name}.npz" \
    2>&1 | tee "/home/ubuntu/${name}.log"
echo "npz addendum banked: reports/${name}.npz"
echo "next: fontaine/scripts/snapflow_results.py --npz reports/${name}.npz reports/${name}.json"
