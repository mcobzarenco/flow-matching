#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
#
# fontaine — #6 rung (c) masked-contrast selection run (mcselect).
# Pre-reg: fontaine/blog/src/posts/2026-08-09-prereg-subgoal-mcselect.md
# ONE q4 full run: per row, per ELIGIBLE banked candidate, a
# conditioned greedy decode (KL collected from the decode's own
# logits) + a teacher-forced planner-less reference forward; the
# planner-less reference decode rides the policy row. NO in-run
# sampling — candidates are injected from the sha-pinned rung-(b')
# file, so every banked comparator stays valid by construction. The
# ARGMAX lives in mcselect_results.py (frozen read), never here.
# COST: ~2-2.5 GPU-h projected (draft amendment 05:5xZ 08-09); GATE 4
# GPU-h — babysit entry mcselect_q4 armed at launch is the backstop
# (single q4 run, no downshift branch to mechanize).
# CHAIN: live oracles (abort-grade contract checks + composition-noise
# diagnostics) -> the frozen read. A red live oracle stops the chain
# BEFORE any number is read.
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a /home/ubuntu/eval__ar100k_mcselect_q4.log) 2>&1

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_arb_rcond_100k_ddp4
CKPT=/home/ubuntu/checkpoints/bijou-checkpoints/${RUN}/step_100000
PLAN_Q4=plans/holdout_curated_v0_k4l2_stateprobe_q4.json
CANDS=reports/eval__${RUN}__step_100000__stateprobe_q4_subgoalcleandraws_candidates.json
STEM=eval__${RUN}__step_100000__stateprobe_q4_subgoalmcselect

[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
# The pre-reg's frozen inputs: the q4 plan and the rung-(b') candidates
# file (its sha is also what the report must echo — the read script's
# comparator guard).
sha256sum -c - <<'SHAS'
876c39c8fe2b3cb16945a40c35ec157c907b4f7417e7dfd0b6cf46dd47355ef5  plans/holdout_curated_v0_k4l2_stateprobe_q4.json
8175624eeb787b78cbd4363c51a35d323629ca86631c71d3ffc472067801ddad  reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__stateprobe_q4_subgoalcleandraws_candidates.json
SHAS
# Instrument oracles must be green at HEAD before any launch (pre-reg
# "instrument prerequisites"): the CPU halves are pytest-pinned; the
# live-oracle selftest proves every abort branch fires.
uv run pytest tests/test_mcselect.py -q || { echo "instrument tests RED — abort"; exit 1; }
uv run python fontaine/scripts/mcselect_live_oracles.py --selftest \
    || { echo "live-oracle selftest RED — abort"; exit 1; }
uv run python fontaine/scripts/mcselect_results.py --oracle \
    || { echo "read-script oracle RED — abort"; exit 1; }

DATA_ARGS=(
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0
    --episodes holdout --holdout-episodes 0.1 --split-seed 0
    --fps 30 --camera-counts 1 2
)

echo "=== rung (c) mcselect q4 run ($STEM) ==="
uv run python -m bijou.eval \
    "${DATA_ARGS[@]}" \
    --sample-plan "$PLAN_Q4" \
    --checkpoint "$CKPT" \
    --subgoal-mode mcselect \
    --subgoal-candidates-file "$CANDS" \
    --mcselect-tau 4.0 \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${STEM}.json" \
    --dump-predictions "reports/${STEM}.npz" \
    --report "reports/${STEM}.html"

echo "=== run complete — live oracles (abort-grade before any read) ==="
uv run python fontaine/scripts/mcselect_live_oracles.py

echo "=== live oracles GREEN — frozen read ==="
uv run python fontaine/scripts/mcselect_results.py

echo "=== MCSELECT Q4 DONE (rc=0) — analysis json written; prune the"
echo "=== babysit entry + record the verdict on the queue item ==="
