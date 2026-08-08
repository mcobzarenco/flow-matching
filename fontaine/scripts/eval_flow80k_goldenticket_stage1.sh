#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
#
# fontaine — golden-ticket noise screen STAGE 1 (#1, teacher-first):
#   ONE batched draws-64 eval on the drawsprobe_s7 probe where the
#   draws ARE the 64 candidate tickets (--noise-tickets: draw m at
#   every frame uses ticket m), then ticket_scores.py pools per-ticket
#   core chunk MAE + adjudicates the frozen R1 kill line
#   (sd > 0.0785 OR min < mean − 0.22). Stage 2 (winner, full panel,
#   complement rows) and stage 3 (mean-of-top-10) are SEPARATE
#   launches, gated on R1 — this script never runs them.
# Pre-reg: fontaine/blog/src/posts/2026-08-07-prereg-golden-ticket-screen.md
#   Anchors banked: stable-key single draw 6.5997/1.9355, mean-of-10
#   5.3645/1.4242, sigma_probe 0.0669. Cost: stage 1 ~1.5 GPU-h of the
#   pre-registered <= 6 GPU-h total.
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a /home/ubuntu/eval__goldenticket_stage1.log) 2>&1

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_flow_artrunk_h1024_40k_ddp2
CKPT=outputs/train/${RUN}/step_080000
PLAN=plans/holdout_curated_v0_k4l2_drawsprobe_s7.json
TICKETS=plans/tickets_goldenticket_m64.npz

[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
3d7c5b02923f10b17d6c928c603bd6aff9bdc59ec076307ad523854d0c571dbd  plans/holdout_curated_v0_k4l2_drawsprobe_s7.json
9bb13bc47a92f7cc764e81022a9a7b05dbb9ec391eb9ba8ab14d675c955cc7c0  plans/tickets_goldenticket_m64.npz
SHAS

name="eval__${RUN}__step_080000__drawsprobe_s7_ticket_draws64_heun30"
echo "=== golden-ticket stage 1: draws-64 ticket search ($name) ==="
uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan "$PLAN" \
    --checkpoint "$CKPT" \
    --sample-steps 30 --sample-method heun \
    --sample-draws 64 --noise-tickets "$TICKETS" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${name}.json" \
    --dump-draws "reports/${name}_draws.npz"

echo "=== stage-1 scorer + R1 kill line ==="
uv run python fontaine/scripts/ticket_scores.py \
    --npz "reports/${name}_draws.npz" \
    --out reports/analysis__goldenticket_stage1.json

echo "=== GOLDENTICKET STAGE 1 DONE (rc=0) — R1 verdict above gates stage 2 ==="
