#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh fontaine-ftrig-ticket64-v2all bash <this script>
# fontaine — ftrig ticket-bank scoring over the ENTIRE so101_pick_place_v2
# dataset, training episodes included (owner request 2026-08-09 15:44Z).
# Same recipe as eval_ftrig_rig_ticketbank64.sh (which scored the 2-repo
# holdout) except: v2 only, --episodes all (50 episodes, 32,679 frames).
# Deployment support, record-only; NOT a generalization read — training
# rows are deliberately in the pool at the owner's request.
# COST: ~600 f/min observed at this exact decode => ~55 min local H100.
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=outputs/train/fontaine_flow_snapdistill_ftrig_4k_1xh100/step_004000
STEM=eval__fontaine_flow_snapdistill_ftrig_4k_1xh100__step_004000__rigv2all_1nfe_euler1_ticketbank64

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi

.venv/bin/python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/so101_pick_place_v2 \
    --episodes all \
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

echo "=== TICKET BANK SCORED (ftrig@4000, rig v2 ALL episodes, 64 tickets) ==="
