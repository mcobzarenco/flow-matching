#!/usr/bin/env bash
# token-GRPO phase-2 R1 (steps 3-17) —
# posts/2026-08-13-prereg-token-grpo-phase2-run.md (frozen 8548969):
# "R1 resumes the R0 checkpoint: --resume
# outputs/sim/grpo_phase2/step_0002.pt --total-steps 17". Launch ONLY
# on a green R0 boundary (signal + ratio gates, pace projection <= 22
# GPU-h cum). All non-stated flags are the frozen defaults; same
# alloc/memory envelope as launch 3 of R0 (d0b9a44).
set -uo pipefail
cd "$(dirname "$0")/../.."

if [ ! -f outputs/sim/grpo_phase2/step_0002.pt ]; then
    echo "FATAL: step_0002.pt missing — R0 did not complete" >&2
    exit 2
fi
echo "=== grpo phase2 R1 start $(date -u +%FT%TZ) ==="
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python -m sim.grpo_loop \
    --checkpoint allenai/MolmoAct2-SO100_101 \
    --out-dir outputs/sim/grpo_phase2 \
    --resume outputs/sim/grpo_phase2/step_0002.pt \
    --total-steps 17 --eval-every 5 --save-every 1
rc=$?
echo "=== grpo phase2 R1 done rc=$rc $(date -u +%FT%TZ) ==="
exit $rc
