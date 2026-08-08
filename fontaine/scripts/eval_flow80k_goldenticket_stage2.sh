#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
#
# fontaine — golden-ticket noise screen STAGE 2 (#1, R1-gated):
#   confirmatory full-panel eval of the stage-1 WINNER ticket (index 33
#   of the pinned m64 bank; R1 verdict CONFIRM 2026-08-08 04:2xZ —
#   sd 0.82252 > 0.0785, min 5.70564 < 6.52401, both clauses open).
#   --sample-draws 1 --noise-tickets <winner-only npz>: every frame
#   integrates from ticket 33. The R2 read (paired per-frame Δ vs the
#   banked stable-key npz on COMPLEMENT core rows — panel core minus
#   the probe plan's frame-identity triples; REAL iff Δ ≤ −0.05 with
#   CI95 excluding 0) runs from this eval's dump; this script never
#   adjudicates. Stage 3 (mean-of-top-10) is a separate launch, gated
#   on R2.
# Pre-reg: fontaine/blog/src/posts/2026-08-07-prereg-golden-ticket-screen.md
#   Cost: ~0.9 GPU-h of the pre-registered <= 6 GPU-h total (stage 1
#   spent ~1.7).
# Provenance: winner npz is bank[33:34] byte-verified at materialization
#   (bank sha 9bb13bc4..., materialized 2026-08-08 04:2xZ work session).
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a /home/ubuntu/eval__goldenticket_stage2.log) 2>&1

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_flow_artrunk_h1024_40k_ddp2
CKPT=outputs/train/${RUN}/step_080000
PLAN=plans/holdout_curated_v0_k4l2.json
TICKETS=plans/tickets_goldenticket_winner33.npz

[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
a392d630f264c3061ce7f0e246a8803ca8d1c50c64f112a6667a989fe4af1fa5  plans/tickets_goldenticket_winner33.npz
SHAS

name="eval__${RUN}__step_080000__panel_curated_v0_k4l2_ticket33_heun30"
echo "=== golden-ticket stage 2: winner ticket 33, full panel ($name) ==="
uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan "$PLAN" \
    --checkpoint "$CKPT" \
    --sample-steps 30 --sample-method heun \
    --sample-draws 1 --noise-tickets "$TICKETS" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${name}.json" \
    --dump-predictions "reports/${name}.npz"

echo "=== GOLDENTICKET STAGE 2 DONE (rc=0) — R2 read runs offline from the npz ==="
