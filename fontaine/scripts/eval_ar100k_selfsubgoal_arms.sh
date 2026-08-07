#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
#
# fontaine — self-subgoal probe STAGE-2 ARMS (#6 rung (a)), sequential:
#   1. oracle-subgoal arm, full k4l2 panel (~35 min) — runs regardless
#      of the stage-1 verdict (pre-reg: it answers whether the slot is
#      live at all);
#   2. self-subgoal arm, full panel, two passes (~70 min; pass 1 IS the
#      narrated-subgoal arm, retained in the same npz) — gated on the
#      stage-1 go marker fontaine/harness/state/selfsubgoal_stage1_go,
#      written ONLY after the human go/no-go on the validity table
#      (pre-reg criteria (a) (b) (c)). No marker -> oracle arm only.
# Pre-reg: fontaine/blog/src/posts/2026-08-07-prereg-selfsubgoal-probe.md
#   Baseline 5.8026/2.1431 is BANKED (panel_k4l2 npz) and never re-run.
# PREREQUISITE: eval_ar100k_selfsubgoal_preflight.sh completed with
#   live oracles GREEN (this script refuses to run without its outputs).
# Cost: ~2 GPU-h of the pre-registered <= 8 GPU-h total (banked rate
#   0.081 s/frame x 25,800 rows; q4 fallback clause not expected to bind
#   — preflight measured the real rate first).
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a /home/ubuntu/eval__ar100k_selfsubgoal_arms.log) 2>&1

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_arb_rcond_100k_ddp4
CKPT=/home/ubuntu/checkpoints/bijou-checkpoints/${RUN}/step_100000
PLAN_FULL=plans/holdout_curated_v0_k4l2.json
GO_MARKER=fontaine/harness/state/selfsubgoal_stage1_go

[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
SHAS
# Preflight must have adjudicated GREEN (its stage-4 table only exists
# after the abort-on-red oracle step passed).
[ -f reports/analysis__selfsubgoal_stage1_table.json ] || {
    echo "no stage-1 table — preflight has not completed; abort"; exit 1; }

DATA_ARGS=(
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0
    --episodes holdout --holdout-episodes 0.1 --split-seed 0
    --fps 30 --camera-counts 1 2
)

orc="eval__${RUN}__step_100000__panel_k4l2_oraclesubgoal"
echo "=== arm 1/2: oracle-subgoal, full panel ($orc) ==="
uv run python -m bijou.eval \
    "${DATA_ARGS[@]}" \
    --sample-plan "$PLAN_FULL" \
    --checkpoint "$CKPT" \
    --subgoal-mode oracle \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${orc}.json" \
    --dump-predictions "reports/${orc}.npz" \
    --report "reports/${orc}.html"

if [ ! -f "$GO_MARKER" ]; then
    echo "=== NO STAGE-1 GO MARKER ($GO_MARKER) — self arm NOT run ==="
    echo "=== (pre-reg: stage-1 fail IS the rung-(a) generation result) ==="
    echo "=== SELFSUBGOAL ARMS DONE (oracle only, rc=0) ==="
    exit 0
fi

slf="eval__${RUN}__step_100000__panel_k4l2_selfsubgoal"
echo "=== arm 2/2: self-subgoal two-pass, full panel ($slf) ==="
uv run python -m bijou.eval \
    "${DATA_ARGS[@]}" \
    --sample-plan "$PLAN_FULL" \
    --checkpoint "$CKPT" \
    --subgoal-mode self \
    --dump-subgoals "reports/${slf}_subgoals.json" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${slf}.json" \
    --dump-predictions "reports/${slf}.npz" \
    --report "reports/${slf}.html"

echo "=== SELFSUBGOAL ARMS DONE (oracle + self/narrated, rc=0) ==="
