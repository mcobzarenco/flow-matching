#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
#
# fontaine — self-subgoal probe PREFLIGHT (#6 rung (a)).
# Pre-reg: fontaine/blog/src/posts/2026-08-07-prereg-selfsubgoal-probe.md
# Runs, in order, everything the pre-reg gates BEFORE the stage-2 arms:
#   1. q4-subset SELF run with --selfsubgoal-force-empty + --dump-subgoals
#      (live oracle (i) input: the no-hint limit must be bit-exact vs the
#      banked baseline rows; the dump also carries per-frame TRUE labels =
#      the label mask for oracle (ii));
#   2. q4-subset ORACLE run (live oracle (ii) input: label-less rows must
#      be bit-exact vs baseline; labeled rows must move);
#   3. fontaine/scripts/selfsubgoal_live_oracles.py — abort-on-red
#      byte-match adjudication (state-copy rows included). NO scalars.
#   4. stage-1 validity table (selfsubgoal_stage1.py, 60 stratified
#      frames, generation-only) — the go/no-go judgment itself is human
#      and happens in-session; this script only produces the evidence.
# Cost: ~0.4 GPU-h of the pre-registered <= 8 GPU-h total.
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a /home/ubuntu/eval__ar100k_selfsubgoal_preflight.log) 2>&1

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_arb_rcond_100k_ddp4
CKPT=/home/ubuntu/checkpoints/bijou-checkpoints/${RUN}/step_100000
PLAN_Q4=plans/holdout_curated_v0_k4l2_stateprobe_q4.json
PLAN_FULL=plans/holdout_curated_v0_k4l2.json
BASELINE_NPZ=reports/eval__${RUN}__step_100000__panel_k4l2.npz

[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
[ -f "$BASELINE_NPZ" ] || { echo "no banked baseline $BASELINE_NPZ — abort"; exit 1; }
sha256sum -c - <<'SHAS'
876c39c8fe2b3cb16945a40c35ec157c907b4f7417e7dfd0b6cf46dd47355ef5  plans/holdout_curated_v0_k4l2_stateprobe_q4.json
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
SHAS

DATA_ARGS=(
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0
    --episodes holdout --holdout-episodes 0.1 --split-seed 0
    --fps 30 --camera-counts 1 2
)

empty="eval__${RUN}__step_100000__stateprobe_q4_selfsubgoal_emptyhint"
echo "=== preflight 1/4: q4 forced-empty self run ($empty) ==="
uv run python -m bijou.eval \
    "${DATA_ARGS[@]}" \
    --sample-plan "$PLAN_Q4" \
    --checkpoint "$CKPT" \
    --subgoal-mode self --selfsubgoal-force-empty \
    --dump-subgoals "reports/${empty}_subgoals.json" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${empty}.json" \
    --dump-predictions "reports/${empty}.npz"

orc="eval__${RUN}__step_100000__stateprobe_q4_oraclesubgoal"
echo "=== preflight 2/4: q4 oracle-subgoal run ($orc) ==="
uv run python -m bijou.eval \
    "${DATA_ARGS[@]}" \
    --sample-plan "$PLAN_Q4" \
    --checkpoint "$CKPT" \
    --subgoal-mode oracle \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${orc}.json" \
    --dump-predictions "reports/${orc}.npz"

echo "=== preflight 3/4: live-oracle adjudication (abort-on-red) ==="
uv run python fontaine/scripts/selfsubgoal_live_oracles.py \
    --baseline-npz "$BASELINE_NPZ" \
    --emptyhint-npz "reports/${empty}.npz" \
    --emptyhint-json "reports/${empty}.json" \
    --oracle-npz "reports/${orc}.npz" \
    --oracle-json "reports/${orc}.json" \
    --subgoals-json "reports/${empty}_subgoals.json"

echo "=== preflight 4/4: stage-1 validity table (60 frames, generation-only) ==="
uv run python fontaine/scripts/selfsubgoal_stage1.py \
    "${DATA_ARGS[@]}" \
    --checkpoint "$CKPT" \
    --sample-plan "$PLAN_FULL" \
    --num-frames 60 --seed 0 \
    --output-md reports/analysis__selfsubgoal_stage1_table.md \
    --output-json reports/analysis__selfsubgoal_stage1_table.json

echo "=== SELFSUBGOAL PREFLIGHT DONE (rc=0) — live oracles GREEN; ==="
echo "=== stage-1 table awaits the human go/no-go (pre-reg criteria ==="
echo "=== (a) >=90% non-empty, (b) no string >50%, (c) eyes).      ==="
