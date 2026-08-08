#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
#
# fontaine — subgoal-draws selection PREFLIGHT (#6 rung (b)).
# Pre-reg: fontaine/blog/src/posts/2026-08-08-prereg-subgoal-draws.md
# Runs, in order, everything the pre-reg gates BEFORE the stage-2 arms:
#   1. q4-subset FRESH SELF run (rung-(a) mode) — the oracle-(i)
#      comparator at matched batch composition (amendment-1 lesson:
#      banked full-panel npzs are composition-mismatched comparators);
#   2. q4-subset DRAWS-0 run (greedy candidate only, both selection
#      arms) — oracle (i): bon + narr must be bit-exact vs run 1;
#   3. q4-subset DRAWS-0 --selfsubgoal-force-empty run — oracle (ii):
#      both arms bit-exact vs the BANKED rung-(a) q4 emptyhint decode;
#   4. subgoal_draws_live_oracles.py — abort-on-red adjudication; on
#      green writes reports/analysis__subgoal_draws_preflight_oracles.json
#      (the arms launcher refuses to run without it). NO scalars.
#   5. stage-1 candidates table (selfsubgoal_stage1.py --subgoal-draws 8,
#      SAME fixed-seed 60-frame stratified sample as rung (a)) — the
#      mechanical go/no-go bars print; criterion (d) subgoal-shaped and
#      the go marker (fontaine/harness/state/subgoal_draws_stage1_go)
#      stay a human judgment, in-session.
# Cost: ~1.3 GPU-h of the pre-registered <= 6 GPU-h total.
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a /home/ubuntu/eval__ar100k_subgoal_draws_preflight.log) 2>&1

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_arb_rcond_100k_ddp4
CKPT=/home/ubuntu/checkpoints/bijou-checkpoints/${RUN}/step_100000
PLAN_Q4=plans/holdout_curated_v0_k4l2_stateprobe_q4.json
PLAN_FULL=plans/holdout_curated_v0_k4l2.json
BANKED_EMPTY=reports/eval__${RUN}__step_100000__stateprobe_q4_selfsubgoal_emptyhint.npz

[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
[ -f "$BANKED_EMPTY" ] || { echo "no banked emptyhint npz $BANKED_EMPTY — abort"; exit 1; }
sha256sum -c - <<'SHAS'
876c39c8fe2b3cb16945a40c35ec157c907b4f7417e7dfd0b6cf46dd47355ef5  plans/holdout_curated_v0_k4l2_stateprobe_q4.json
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
SHAS

DATA_ARGS=(
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0
    --episodes holdout --holdout-episodes 0.1 --split-seed 0
    --fps 30 --camera-counts 1 2
)

slf="eval__${RUN}__step_100000__stateprobe_q4_selfsubgoal"
echo "=== preflight 1/5: q4 fresh SELF run — oracle-(i) comparator ($slf) ==="
uv run python -m bijou.eval \
    "${DATA_ARGS[@]}" \
    --sample-plan "$PLAN_Q4" \
    --checkpoint "$CKPT" \
    --subgoal-mode self \
    --dump-subgoals "reports/${slf}_subgoals.json" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${slf}.json" \
    --dump-predictions "reports/${slf}.npz"

d0="eval__${RUN}__step_100000__stateprobe_q4_subgoaldraws0"
echo "=== preflight 2/5: q4 DRAWS-0 run — oracle (i) probe ($d0) ==="
uv run python -m bijou.eval \
    "${DATA_ARGS[@]}" \
    --sample-plan "$PLAN_Q4" \
    --checkpoint "$CKPT" \
    --subgoal-mode draws --subgoal-draws 0 \
    --dump-subgoal-candidates "reports/${d0}_candidates.json" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${d0}.json" \
    --dump-predictions "reports/${d0}.npz"

d0e="eval__${RUN}__step_100000__stateprobe_q4_subgoaldraws0_emptyhint"
echo "=== preflight 3/5: q4 DRAWS-0 forced-empty run — oracle (ii) probe ($d0e) ==="
uv run python -m bijou.eval \
    "${DATA_ARGS[@]}" \
    --sample-plan "$PLAN_Q4" \
    --checkpoint "$CKPT" \
    --subgoal-mode draws --subgoal-draws 0 --selfsubgoal-force-empty \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${d0e}.json" \
    --dump-predictions "reports/${d0e}.npz"

echo "=== preflight 4/5: live-oracle adjudication (abort-on-red) ==="
uv run python fontaine/scripts/subgoal_draws_live_oracles.py \
    --self-npz "reports/${slf}.npz" \
    --self-json "reports/${slf}.json" \
    --self-subgoals "reports/${slf}_subgoals.json" \
    --draws0-npz "reports/${d0}.npz" \
    --draws0-json "reports/${d0}.json" \
    --candidates "reports/${d0}_candidates.json" \
    --empty-npz "reports/${d0e}.npz" \
    --empty-json "reports/${d0e}.json" \
    --banked-empty-npz "$BANKED_EMPTY" \
    --out reports/analysis__subgoal_draws_preflight_oracles.json

echo "=== preflight 5/5: stage-1 candidates table (60 frames, 9 candidates each) ==="
uv run python fontaine/scripts/selfsubgoal_stage1.py \
    "${DATA_ARGS[@]}" \
    --checkpoint "$CKPT" \
    --sample-plan "$PLAN_FULL" \
    --num-frames 60 --seed 0 \
    --subgoal-draws 8 --subgoal-temperature 1.0 \
    --output-md reports/analysis__subgoal_draws_stage1_table.md \
    --output-json reports/analysis__subgoal_draws_stage1_table.json

echo "=== SUBGOAL-DRAWS PREFLIGHT DONE (rc=0) — live oracles GREEN; ==="
echo "=== stage-1 table awaits the go/no-go (pre-reg bars (a)-(c)    ==="
echo "=== mechanical, (d) subgoal-shaped by eyes) before the arms.   ==="
