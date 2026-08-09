#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh fontaine-ftrig-ticket64 bash <this script>
# fontaine — ftrig rig-holdout ticket-bank scoring (owner request
# 2026-08-09 15:25Z: "give me the optimized-for vector" for physical
# rollouts of snapdistill_ftrig_4k @ step_004000 under a fixed noise
# ticket).
#
# WHAT IT IS: the R1 golden-ticket screen pattern (one batched draws-M
# eval scores every bank candidate: draw d at every frame IS
# tickets[d]) pointed at the DEPLOYMENT checkpoint and ITS data — the
# ftrig student, 1-NFE euler-1 target-time-zero, the exact rig-holdout
# split its fine-tune banked R0/R2 on (launch_local_snapflow_ftrig_
# 4k_1xh100.sh's rig_eval, verbatim recipe apart from the noise
# source). Exploratory, record-only, owner-requested deployment
# support; NOT a leaderboard read — the _ticket-suffixed policy name
# and the bank sha ride in all outputs (#18.1 provenance).
# The winner export + selection read runs AFTER via
# ftrig_ticket_winner.py (oracle-gated; argmin per-ticket pooled MAE).
# COST: 3,647 holdout frames x 64 draws at 1-NFE on the idle local
# H100 — the 25.8k-frame 1-NFE panel ran ~30-40 min at draws 1;
# per-frame here is 64x chunks through the h1024 expert but the
# trunk prefix encodes once per frame. Expect well under 2 h; abort
# and downshift --batch-size if the first nvidia-smi poll shows
# near-OOM (77+ GiB).
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RIG_DATA=(/home/ubuntu/datasets/mcobzarenco/so101_pick_place_v2
          /home/ubuntu/datasets/mcobzarenco/so101_pick_place_clean)
CKPT=outputs/train/fontaine_flow_snapdistill_ftrig_4k_1xh100/step_004000
STEM=eval__fontaine_flow_snapdistill_ftrig_4k_1xh100__step_004000__rig_holdout_1nfe_euler1_ticketbank64

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi

.venv/bin/python -m bijou.eval \
    --data "${RIG_DATA[@]}" \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --checkpoint "$CKPT" \
    --sample-draws 64 --sample-steps 1 --sample-method euler \
    --target-time zero \
    --noise-tickets plans/tickets_goldenticket_m64.npz \
    --batch-size 32 --num-workers 8 --seed 0 \
    --report-samples 0 \
    --dump-draws "reports/${STEM}_draws.npz" \
    --output-json "reports/${STEM}.json" \
    2>&1 | tee "/home/ubuntu/${STEM}.log"

echo "=== TICKET BANK SCORED (ftrig@4000, rig holdout, 64 tickets) ==="
