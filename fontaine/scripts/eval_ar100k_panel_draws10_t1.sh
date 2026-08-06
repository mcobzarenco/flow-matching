#!/usr/bin/env bash
# fontaine — AR sampled-draws eval, gemma4 arm (ideas #19).
# Pre-reg: fontaine/blog/src/posts/2026-08-06-prereg-ar-sampled-draws.md
#   (incl. the pre-launch amendment: the arm runs on AR-100k
#   `bijou_arb_rcond_100k_ddp4/step_100000` — the checkpoint that owns
#   the banked greedy anchor 5.8026 — not A-s0, whose greedy is 7.7966).
# Instrument: --ar-temperature 1.0 --sample-draws 10 (commit 78c9f56) —
#   Gumbel-max over the grammar-masked softmax, T=1.0 pinned/untuned,
#   10 draws share one prefill, chunks averaged in raw units; policy
#   row gains `_draws10_t1`. Aux value lines stay greedy; narrated
#   pass skipped under sampling.
# Row pairing: same plan file as the banked greedy run
#   (plans/holdout_curated_v0_k4l2.json, its recorded sample_plan) —
#   frozen reads pair per-row against
#   reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2.npz.
# COST GATE (pre-registered, not improvised): measure rate over the
#   first ~200 frames; if full-panel (25,800 rows) projects > 24 GPU-h
#   (< ~18 f/min), KILL and relaunch on the frozen q4 subset
#   (plans/holdout_curated_v0_k4l2_stateprobe_q4.json, 4,301 rows);
#   the switch is recorded in now.md + Discord, never silent.
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_arb_rcond_100k_ddp4
CKPT=/home/ubuntu/checkpoints/bijou-checkpoints/${RUN}/step_100000
name="eval__${RUN}__step_100000__panel_k4l2_draws10_t1"
uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2.json \
    --checkpoint "$CKPT" \
    --ar-temperature 1.0 --sample-draws 10 \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${name}.json" \
    --dump-predictions "reports/${name}.npz" \
    --report "reports/${name}.html" \
    2>&1 | tee "/home/ubuntu/${name}.log"
echo "=== AR-100k draws10_t1 panel eval DONE (rc=0) ==="
