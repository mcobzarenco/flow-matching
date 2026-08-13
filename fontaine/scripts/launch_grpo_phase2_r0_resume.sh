#!/usr/bin/env bash
# token-GRPO phase-2 R0 launch 4 — resume from step_0001.pt (banked by
# launch 3 before its wave-1 worker OOM, 18:57:55Z 08-13; fix:
# release_cached_vram before every wave/eval). Re-runs ONLY step 2 +
# the endpoint eval; deterministic keying makes this identical to a
# from-scratch run's remaining work (train stream 1000+8*step, baseline
# rides in the checkpoint). Pre-reg addendum 3 documents the resume.
set -uo pipefail
cd "$(dirname "$0")/../.."

if [ ! -f outputs/sim/grpo_phase2/step_0001.pt ]; then
    echo "FATAL: step_0001.pt missing — nothing to resume" >&2
    exit 2
fi
echo "=== grpo phase2 R0 resume start $(date -u +%FT%TZ) ==="
MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python -m sim.grpo_loop \
    --checkpoint allenai/MolmoAct2-SO100_101 \
    --out-dir outputs/sim/grpo_phase2 \
    --resume outputs/sim/grpo_phase2/step_0001.pt \
    --total-steps 2 --eval-every 5 --save-every 1
rc=$?
echo "=== grpo phase2 R0 resume done rc=$rc $(date -u +%FT%TZ) ==="
exit $rc
