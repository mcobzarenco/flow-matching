#!/usr/bin/env bash
# k4l2 panel_v2 leg for grasp_sft_v2_demosonly_1gpu_disc/step_001000
# (queue: disc1000-k4l2-panel-leg) — pre-banks the baseline side of the
# pdnorm pre-reg's tertiary read ("paired vs the discriminator's banked
# step-1000 on the shared frames", +0.05 CI guard, per-motor deltas).
#
# PROTOCOL PIN (the pdnorm endpoint leg must copy these exactly —
# the pairing is only valid protocol-matched):
#   - plans/holdout_curated_v0_k4l2_panel_v2.json (panel_v2, 15,056
#     core frames pooled), community_curated_v0, holdout 0.1 /
#     split-seed 0, fps 30, cameras 1+2 — the house panel_v2 pins
#   - euler-10 single draw, stable noise keying — the molmoact2
#     lineage's serving operating point (probe/sim/HTML report all
#     read euler-10; heun-30 is the gemma-flow panel convention, NOT
#     this family's)
#   - --chunk-size 30 explicit (spec.chunk_size; eval default is 50)
#   - batch 32, workers 20 (RELAUNCH 04:3xZ: batch-12/workers-8 was
#     input-starved — 66 f/min, 38-57% util, projected 5.7 GPU-h vs
#     the 3 gate; killed at 4.7 min per the first-poll starvation
#     rule. 32 is ~35 GiB-class on this checkpoint, far under the 78
#     bar; the probe-batch OOM was at 96)
#   - npz prediction dump = the pairing substrate
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
NAME="eval__${RUN}__${STEP}__panel_v2_k4l2_euler10_draws1_stable"
mkdir -p reports

uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2_panel_v2.json \
    --checkpoint ~/checkpoints/finetune/$RUN/$STEP \
    --sample-draws 1 --sample-steps 10 --sample-method euler \
    --noise-key stable \
    --chunk-size 30 \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --output-json "reports/${NAME}.json" \
    --dump-predictions "reports/${NAME}.npz" \
    --report "reports/${NAME}.html" \
    2>&1 | tee "/home/ubuntu/${NAME}.log"
echo "=== disc-1000 k4l2 panel_v2 leg DONE $(date -u +%FT%TZ) ==="
