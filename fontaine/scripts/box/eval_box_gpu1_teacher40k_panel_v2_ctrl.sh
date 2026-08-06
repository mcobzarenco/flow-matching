#!/usr/bin/env bash
# fontaine — arch-batch-1 CONTROL eval: teacher@40k on panel-v2 (BOX GPU 1).
# PRE-REG: fontaine/blog/src/posts/2026-08-06-prereg-arch-batch-1.md,
#   Amendment 1 (owner 12:05Z steer): arm 0 dropped; control :=
#   teacher's own step_040000 (bijou_flow_artrunk_h1024_40k_ddp2 —
#   completed 40k schedule, seed 0 matched to arms, eff-96).
# SPEC (amendment, verbatim): one panel-v2 endpoint eval, Heun-30,
#   draws=1, stable keying, seed 0, --dump-predictions npz, run before
#   arm A's endpoint reads open. Expectation band (was arm 0's):
#   chunk in [6.7, 7.9], first in [1.90, 2.35].
# SEMANTICS PINNED EXPLICITLY (the d9dd385 lesson): heun/30/draws-1/
#   stable are all passed, never inherited from defaults.
# EXECUTION (charter §3): box GPU 1 — arm C owns GPU 0; never
#   co-locates (CUDA_VISIBLE_DEVICES pins it; quiet-GPU guard).
#   CODE: runs from the THROWAWAY checkout ~/flow-matching-ctrl
#   (current fontaine HEAD, rsynced) — the "box code bcbf101
#   suffices" claim was WRONG (first live v2-plan consumption:
#   SamplePlan refused version 2; fixed at 59dac60). The live
#   ~/flow-matching stays bcbf101 under arm C per never-sync-under-
#   live-run; cwd shadows the installed package for the shared .venv
#   interpreter. Same code class as the arms' future endpoint evals.
#   Cleanup: remove ~/flow-matching-ctrl at the arm-C boundary sync.
# COST: heun-30 draws=1 over the 15.1k-core-frame v2 panel ~1-2 GPU-h.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
cd /home/ubuntu/flow-matching-ctrl
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

mem=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU 1 busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_flow_artrunk_h1024_40k_ddp2
CKPT=outputs/train/${RUN}/step_040000
name="eval__${RUN}__step_040000__panel_v2_ctrl_heun30_draws1_stable"
/home/ubuntu/flow-matching/.venv/bin/python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2_panel_v2.json \
    --checkpoint "$CKPT" \
    --sample-draws 1 --sample-steps 30 --sample-method heun \
    --noise-key stable \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${name}.json" \
    --dump-predictions "reports/${name}.npz" \
    --report "reports/${name}.html" \
    2>&1 | tee "/home/ubuntu/${name}.log"
echo "=== CONTROL EVAL DONE: reports/${name}.json (band: chunk [6.7,7.9], first [1.90,2.35]) ==="