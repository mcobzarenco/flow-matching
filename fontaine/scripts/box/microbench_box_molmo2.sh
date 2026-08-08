#!/usr/bin/env bash
# fontaine — molmo2 decode-cost microbench rows, ONE command on the box
# (queue item molmo2-decode-cost-microbench, 2026-08-08): runs the
# SHARED microbench harness (leaderboard_decode_microbench.py, pre-reg
# 2026-08-07-prereg-leaderboard-decode-microbench.md, record-only) for
# the two molmo2 configs only — retires the 40k leaderboard row's
# mtime-derived cost caveat with same-harness measured numbers
# (batched panel config + batch-1 single-stream latency).
#
# GPU minutes ride a pre-registered box eval window per the queue
# item's boundary: run this AFTER the #19 draws-arm chain finishes and
# BEFORE the next training launch claims the GPUs (~15 min total; the
# harness guards abort if any GPU is busy). Launch via run_detached.sh:
#   fontaine/scripts/run_detached.sh fontaine-microbench-molmo2 \
#       bash fontaine/scripts/box/microbench_box_molmo2.sh
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
cd /home/ubuntu/flow-matching

for g in 0 1 2 3; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$mem" -gt 1024 ]; then echo "GPU $g busy (${mem} MiB) — abort"; exit 1; fi
done

.venv/bin/python fontaine/scripts/leaderboard_decode_microbench.py \
    --configs molmo2_greedy molmo2_draws10_t1 \
    --out reports/analysis__leaderboard_decode_microbench_molmo2.json

echo "=== MOLMO2 MICROBENCH DONE (rc=0) — merge rows into the leaderboard cost column ==="
