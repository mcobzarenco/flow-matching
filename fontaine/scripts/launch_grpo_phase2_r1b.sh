#!/usr/bin/env bash
# token-GRPO phase-2 R1-B (steps 5-14) — the owner-approved
# (2)-then-(1) re-price (09:16Z 08-14) under the PATCHED reward:
# posts/2026-08-14-prereg-token-grpo-phase2-r1b.md. Resumes the R1-A
# tripwire-stop checkpoint into a FRESH out dir (R1-A artifacts stay
# untouched); anchor = the pristine --checkpoint step-0 policy,
# captured before the resume restore (loop contract).
set -uo pipefail
cd "$(dirname "$0")/../.."

if [ ! -f outputs/sim/grpo_phase2_a/step_0004.pt ]; then
    echo "FATAL: outputs/sim/grpo_phase2_a/step_0004.pt missing" >&2
    exit 2
fi
echo "=== grpo phase2 R1-B start $(date -u +%FT%TZ) ==="
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python -m sim.grpo_loop \
    --checkpoint allenai/MolmoAct2-SO100_101 \
    --out-dir outputs/sim/grpo_phase2_b \
    --resume outputs/sim/grpo_phase2_a/step_0004.pt \
    --total-steps 15 \
    --surface a --train-reward v2 \
    --lr 3e-7 --kl-beta 1.0 --advantage-clip 2.0 --kl-stop 0.06 \
    --eval-every 1 --save-every 1
rc=$?
echo "=== grpo phase2 R1-B done rc=$rc $(date -u +%FT%TZ) ==="
exit $rc
