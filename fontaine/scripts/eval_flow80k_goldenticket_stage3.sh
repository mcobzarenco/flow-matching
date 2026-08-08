#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
#
# fontaine — golden-ticket noise screen STAGE 3 (#1, R2-gated):
#   the "both" cell — mean of the TOP-10 stage-1 tickets, full panel,
#   --sample-draws 10 with ticket noise (draw d = top-10 ticket d):
#   does searched noise beat random noise INSIDE the ensembling
#   regime? Gated open by R2 = REAL (2026-08-08 05:1xZ: complement
#   delta -0.924 [CI95 -0.985, -0.866] vs line -0.05).
#   R3 (frozen): pooled delta (mean-of-top-10 - banked mean-of-10
#   5.3645); interesting iff <= -0.02 (beyond the tie band); either
#   way RECORD-ONLY in this screen — mean-of-10's row is not
#   displaced without a paired follow-up.
# Pre-reg: fontaine/blog/src/posts/2026-08-07-prereg-golden-ticket-screen.md
#   Cost: ~2.9 GPU-h; screen total ~1.7 (s1) + ~0.85 (s2) + 2.9 ≈ 5.5
#   of the pre-registered <= 6 GPU-h gate.
# Provenance: top10 npz = bank[[33,2,0,51,10,59,38,28,15,36]] (the
#   stage-1 argsort order, analysis__goldenticket_stage1.json),
#   byte-verified at materialization; bank sha 9bb13bc4....
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a /home/ubuntu/eval__goldenticket_stage3.log) 2>&1

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_flow_artrunk_h1024_40k_ddp2
CKPT=outputs/train/${RUN}/step_080000
PLAN=plans/holdout_curated_v0_k4l2.json
TICKETS=plans/tickets_goldenticket_top10.npz

[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
e537f4cde57b7d2b789f1a4e821fee3fc5ea21ee439913f3747d6d0401d36453  plans/tickets_goldenticket_top10.npz
SHAS

name="eval__${RUN}__step_080000__panel_curated_v0_k4l2_top10tickets_heun30"
echo "=== golden-ticket stage 3: mean-of-top-10 tickets, full panel ($name) ==="
uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan "$PLAN" \
    --checkpoint "$CKPT" \
    --sample-steps 30 --sample-method heun \
    --sample-draws 10 --noise-tickets "$TICKETS" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${name}.json" \
    --dump-predictions "reports/${name}.npz" \
    --dump-draws "reports/${name}_draws.npz"

echo "=== GOLDENTICKET STAGE 3 DONE (rc=0) — R3 pooled read vs banked 5.3645 (record-only) ==="
