#!/usr/bin/env bash
# k4l2 panel_v2 leg for the PRE-SFT released checkpoint
# (~/checkpoints/molmoact2-so101-released) — queue:
# released-ckpt-k4l2-panel-row, PRE-GO record-only. The pdnorm pre-reg
# names "vs the pre-SFT released checkpoint's panel row" as an
# informative endpoint comparison; this banks that row. The checkpoint
# wears its OWN released source table (metadata: normalization q01q99,
# global `stats`, per_dataset_stats empty — the default eval path IS
# its honest wear; nothing to override).
#
# PROTOCOL PIN — verbatim from eval_disc1000_k4l2_panel.sh (the
# pairing/comparison is only valid protocol-matched):
#   - plans/holdout_curated_v0_k4l2_panel_v2.json (panel_v2, 15,056
#     core frames pooled), community_curated_v0, holdout 0.1 /
#     split-seed 0, fps 30, cameras 1+2
#   - euler-10 single draw, stable noise keying
#   - --chunk-size 30 explicit (spec.chunk_size; eval default is 50)
#   - batch 32, workers 20 (starvation-tuned on the disc-1000 r2 leg:
#     96% util; batch 12/workers 8 was input-starved)
#   - npz prediction dump = comparison substrate
# Record-only read (frozen in queue.json before launch):
#   released ~<=15 => SFT destroyed real community competence;
#   released ~>=25 (at/above the 25.15 midpoint null) => community
#   data was never in reach and the endpoint read reweights toward
#   the serving-window mechanism.
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

NAME="eval__molmoact2_so101_released__panel_v2_k4l2_euler10_draws1_stable"
mkdir -p reports

uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2_panel_v2.json \
    --checkpoint ~/checkpoints/molmoact2-so101-released \
    --sample-draws 1 --sample-steps 10 --sample-method euler \
    --noise-key stable \
    --chunk-size 30 \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --output-json "reports/${NAME}.json" \
    --dump-predictions "reports/${NAME}.npz" \
    --report "reports/${NAME}.html" \
    2>&1 | tee "/home/ubuntu/${NAME}.log"
echo "=== released k4l2 panel_v2 leg DONE $(date -u +%FT%TZ) ==="
