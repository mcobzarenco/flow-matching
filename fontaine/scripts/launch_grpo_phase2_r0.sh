#!/usr/bin/env bash
# token-GRPO phase-2 R0 smoke (2 steps) —
# posts/2026-08-13-prereg-token-grpo-phase2-run.md (frozen 15:1xZ
# 08-13, HEAD fa739e9). All non-stated flags are the frozen defaults;
# MUJOCO_GL rides here because run_detached.sh units only inherit
# PATH/HOME. Exit code passes through (3 = tripwire stop).
set -uo pipefail
cd "$(dirname "$0")/../.."

mkdir -p outputs/sim/grpo_phase2
echo "=== grpo phase2 R0 start $(date -u +%FT%TZ) ==="
MUJOCO_GL=egl uv run python -m sim.grpo_loop \
    --checkpoint allenai/MolmoAct2-SO100_101 \
    --out-dir outputs/sim/grpo_phase2 \
    --total-steps 2 --eval-every 5 --save-every 1
rc=$?
echo "=== grpo phase2 R0 done rc=$rc $(date -u +%FT%TZ) ==="
exit $rc
