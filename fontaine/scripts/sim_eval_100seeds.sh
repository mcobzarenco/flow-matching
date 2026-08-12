#!/usr/bin/env bash
# 100-seed sim policy eval — the registered run of
# posts/2026-08-11-prereg-sim-policy-eval-100seeds.md. Five arms,
# sequential (headline + metric floor first, then the ordering rungs),
# identical seed list; per-arm JSON + videos under outputs/sim/eval100/.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=outputs/sim/eval100
CKROOT=$HOME/checkpoints/er_60k/fontaine_molmo2_er_60k_ddp4
mkdir -p "$OUT"

run_arm() {
    local name=$1
    shift
    echo "=== arm $name start $(date -u +%FT%TZ) ==="
    MUJOCO_GL=egl uv run python -m sim.rollout_sim "$@" \
        --seed 0 --num-seeds 100 --replans 30 --execute-horizon 30 \
        --sample-steps 10 --expert-dtype bfloat16 \
        --out-dir "$OUT/$name" --out-json "$OUT/$name.json"
    echo "=== arm $name done $(date -u +%FT%TZ) ==="
}

run_arm er60k --checkpoint "$CKROOT/step_060000"
run_arm hold --hold
run_arm er15k --checkpoint "$CKROOT/step_015000"
run_arm er35k --checkpoint "$CKROOT/step_035000"
run_arm er55k --checkpoint "$CKROOT/step_055000"
echo "ALL ARMS DONE $(date -u +%FT%TZ)"
