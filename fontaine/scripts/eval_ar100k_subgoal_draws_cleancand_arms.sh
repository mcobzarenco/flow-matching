#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
#
# fontaine — clean-list subgoal-draws STAGE-2 ARMS (#6 rung (b′)).
# Pre-reg: fontaine/blog/src/posts/2026-08-08-prereg-subgoal-draws-cleanlist.md
# Rung (b) inherited verbatim except the frozen eligible-list rule:
# budget-truncated candidates are excluded from every scorer's list
# (--subgoal-candidate-filter clean; empty list -> greedy fallback,
# recorded). ONE full-panel run carries BOTH conditioned arms (bon =
# self-certainty over the eligible list, ceil = record-only oracle
# token-F1 over the eligible list) plus pass 1's 9-candidate decode —
# pass-1 bytes are (b)-identical by the pre-reg's byte-identity
# argument (checkpoint/plan/seeds/T unchanged; the filter is
# selection-side only). The banked planner-less baseline
# (5.8026/2.1431) is NOT re-run; frozen reads =
# subgoal_draws_results.py --candidate-filter clean.
# PREREQUISITES (this script refuses to run without them):
#   1. preflight adjudication GREEN — rung (b)'s apparatus, reusable
#      per the (b′) pre-reg (pass 1 is byte-identical):
#      reports/analysis__subgoal_draws_preflight_oracles.json;
#   2. the (b′) stage-1 gate OPEN — the banked-table re-adjudication
#      (reports/analysis__subgoal_draws_cleanlist_stage1.json,
#      bars a′-d PASS, 0/60 pick changes). Rung (b)'s go marker is
#      deliberately NOT consulted: its bar-(a) FAIL is why (b′)
#      exists; the (b′) gate json is the (b′) go/no-go.
# COST GATE (pre-reg, mechanized): rung (b′) ceiling 5 GPU-h total,
#   stage 1 was CPU-free, so the full-panel run's budget is 5.0.
#   draws_rate_gate.py measures the first ~200 frames; a projection
#   past the budget kills the run and relaunches on the frozen q4
#   subset (the #19 clause verbatim; switch recorded).
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a /home/ubuntu/eval__ar100k_subgoal_cleancand_arms.log) 2>&1

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_arb_rcond_100k_ddp4
CKPT=/home/ubuntu/checkpoints/bijou-checkpoints/${RUN}/step_100000
PLAN_FULL=plans/holdout_curated_v0_k4l2.json
PLAN_Q4=plans/holdout_curated_v0_k4l2_stateprobe_q4.json
STAGE1_GATE=reports/analysis__subgoal_draws_cleanlist_stage1.json
ORACLES_GREEN=reports/analysis__subgoal_draws_preflight_oracles.json
GATE_GPU_HOURS="${GATE_GPU_HOURS:-5.0}"

[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
876c39c8fe2b3cb16945a40c35ec157c907b4f7417e7dfd0b6cf46dd47355ef5  plans/holdout_curated_v0_k4l2_stateprobe_q4.json
SHAS
[ -f "$ORACLES_GREEN" ] || {
    echo "no preflight green summary $ORACLES_GREEN — oracles not adjudicated; abort"; exit 1; }
grep -q '"verdict": "GREEN"' "$ORACLES_GREEN" || {
    echo "$ORACLES_GREEN is not GREEN — abort"; exit 1; }
[ -f "$STAGE1_GATE" ] || {
    echo "no (b') stage-1 gate json $STAGE1_GATE — the banked-table"
    echo "re-adjudication has not run; no arms without it; abort"; exit 1; }
grep -q '"stage2_gate": "OPEN"' "$STAGE1_GATE" || {
    echo "$STAGE1_GATE stage2_gate is not OPEN — the (b') bars failed;"
    echo "the re-adjudication IS the rung result; abort"; exit 1; }

DATA_ARGS=(
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0
    --episodes holdout --holdout-episodes 0.1 --split-seed 0
    --fps 30 --camera-counts 1 2
)

run_arms() {  # $1 = plan path, $2 = report stem
    local plan="$1" stem="$2"
    uv run python -m bijou.eval \
        "${DATA_ARGS[@]}" \
        --sample-plan "$plan" \
        --checkpoint "$CKPT" \
        --subgoal-mode draws --subgoal-draws 8 --subgoal-temperature 1.0 \
        --subgoal-candidate-filter clean \
        --dump-subgoals "reports/${stem}_subgoals.json" \
        --dump-subgoal-candidates "reports/${stem}_candidates.json" \
        --batch-size 32 --num-workers 20 --seed 0 \
        --report-samples 0 \
        --output-json "reports/${stem}.json" \
        --dump-predictions "reports/${stem}.npz" \
        --report "reports/${stem}.html"
}

stem="eval__${RUN}__step_100000__panel_k4l2_subgoalcleandraws"
echo "=== (b') stage 2: full-panel draws-8 run, clean-list, both arms ($stem) ==="
run_arms "$PLAN_FULL" "$stem" &
EVAL_PID=$!

set +e
uv run python fontaine/scripts/draws_rate_gate.py \
    --log /home/ubuntu/eval__ar100k_subgoal_cleancand_arms.log \
    --ngpu 1 --gate-gpu-hours "$GATE_GPU_HOURS" --min-frames 200
gate_rc=$?
set -e

if [ "$gate_rc" -eq 2 ]; then
    echo "COST GATE FALLBACK — killing the full-panel run, relaunching on q4"
    kill "$EVAL_PID" 2>/dev/null || true
    wait "$EVAL_PID" 2>/dev/null || true
    sleep 10
    stem_q4="eval__${RUN}__step_100000__stateprobe_q4_subgoalcleandraws"
    echo "=== (b') stage 2 (q4 fallback): clean-list, both arms ($stem_q4) ==="
    run_arms "$PLAN_Q4" "$stem_q4"
    echo "=== q4 fallback run complete — record the switch in now.md + Discord;"
    echo "=== reads via subgoal_draws_results.py --candidate-filter clean --draws-stem reports/${stem_q4}"
    echo "=== SUBGOAL-CLEANCAND ARMS DONE (q4 fallback, rc=0) ==="
    exit 0
fi
if [ "$gate_rc" -eq 1 ]; then
    echo "rate gate INDETERMINATE — run left alive; babysit gpu_hours_max is the backstop"
fi

wait "$EVAL_PID"
echo "=== SUBGOAL-CLEANCAND ARMS DONE (full panel, rc=0) — frozen reads:"
echo "=== uv run python fontaine/scripts/subgoal_draws_results.py --candidate-filter clean ==="
