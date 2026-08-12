#!/usr/bin/env bash
# Phase 2 of the 100-seed sim eval — the owner amendment 22:58Z 08-11
# ("kill the arms, let's try some other policy to see if we ever get
# more than 0 success"): same protocol as
# posts/2026-08-11-prereg-sim-policy-eval-100seeds.md (seeds 0-99,
# 30 s horizon, same metrics), new arms in the owner's priority order —
# rig fine-tune first, fast 1-NFE students before the slow teacher.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=outputs/sim/eval100
TRAIN=outputs/train
mkdir -p "$OUT"

run_arm() {
    local name=$1
    shift
    echo "=== arm $name start $(date -u +%FT%TZ) ==="
    MUJOCO_GL=egl uv run python -m sim.rollout_sim "$@" \
        --seed 0 --num-seeds 100 --replans 30 --execute-horizon 30 \
        --expert-dtype bfloat16 \
        --out-dir "$OUT/$name" --out-json "$OUT/$name.json"
    echo "=== arm $name done $(date -u +%FT%TZ) ==="
}

run_arm ftrig4k --checkpoint "$TRAIN/fontaine_flow_snapdistill_ftrig_4k_1xh100/step_004000" \
    --method euler --sample-steps 1
run_arm snap30k --checkpoint "$TRAIN/fontaine_flow_snapdistill_h1024_30k_1xh100/step_030000" \
    --method euler --sample-steps 1
run_arm teacher80k --checkpoint "$TRAIN/bijou_flow_artrunk_h1024_40k_ddp2/step_080000" \
    --method heun --sample-steps 30
echo "ALL PHASE2 ARMS DONE $(date -u +%FT%TZ)"
