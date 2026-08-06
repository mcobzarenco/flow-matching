#!/usr/bin/env bash
# fontaine — SnapFlow 1-NFE self-distillation of flow-80k (LOCAL 1×H100).
# PRE-REG (immutable): fontaine/blog/src/posts/2026-08-06-prereg-snapflow-distill.md
#   Subject: flow-80k step_080000 (local copy, verified). Recipe: the
#   teacher train_args VERBATIM except the PRE-REGISTERED deltas:
#     --steps 30000            (paper distillation budget; teacher 80000)
#     --decoder-lr 2.5e-5      (paper; teacher 1e-4)
#     --grad-clip 1.0          (paper; teacher 10.0)
#     --batch-size 24, 1×GPU   (teacher per-GPU load; effective batch
#                               differs from the teacher's — stated in
#                               the pre-reg, not hidden)
#     --init-from step_080000  (self-distill warm start; teacher used
#                               --resume of its own 40k leg)
#     --distill snapflow       (α=0.5 λ=0.1 frozen in code; implies the
#                               φ_s target-time embedding)
#   DIFF-VERIFIED 2026-08-06 (this session): flag-by-flag against the
#   teacher's recorded train_args — only the deltas above differ (see
#   the verify stage below, which re-checks at launch).
# VALIDATION GATES (block the launch):
#   (a) zero-init φ_s identity on the REAL checkpoint — PASSED
#       2026-08-06 00:5xZ, 6/6 forwards bit-exact (re-run below).
#   (b) E1-style drift gate: φ_s-extended STEP-0 checkpoint, Heun-30
#       s=t, stride-7 probe subset (2,458 frames) vs the banked flow
#       npz — frame-MAE drift < 0.05 (runs below, needs the GPU).
# GATES IN-RUN / ENDPOINT (pre-reg):
#   @10k record-only 1-NFE probe — staged separately
#     (probe_snapflow_10k_1nfe.sh): charter §3 forbids co-located GPU
#     jobs, so it runs at 10k only if a quiet GPU exists (box GPU after
#     checkpoint push, or a pause); else retroactively — it is
#     record-only, the kill line (probe chunk-MAE > teacher Heun-30
#     probe read + 3.0) fires only on a catastrophic read. In-run
#     eval_chunk_mae (s=t) is the live divergence watch either way.
#   Endpoint (30k): full panel at 1-NFE single draw — PRIMARY; adopt
#     iff chunk_mae <= 6.6232 + max(3σ_draw, 0.15), σ_draw pinned by
#     finalization amendment from draws runs 3–5 BEFORE the endpoint
#     eval is opened. Deployment headline: mean-of-N@1-NFE, N∈{5,10},
#     vs the AR anchor 5.8026.
# COST: ~30k steps × (teacher step + 2 sg expert forwards), budget
#   ~12–20 h wall — record actual. Panel@1-NFE is cheap (1 eval/draw).
# NEVER co-located with another GPU job (guard below). Launch only at
#   the quiet boundary after the draws chain + fairness probe.
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

TEACHER=outputs/train/bijou_flow_artrunk_h1024_40k_ddp2/step_080000
RUN=bijou_flow_snapdistill_h1024_30k_1xh100
STEP0="${TEACHER}_snapflow_step0"

# ---- stage 0: recipe diff-verify (teacher train_args vs this launcher,
# parsed through the REAL bijou.train.parse_args) ----
uv run python fontaine/scripts/snapflow_recipe_verify.py

# ---- stage 1: validation gate (a) — zero-init identity, real ckpt ----
uv run python fontaine/scripts/snapflow_identity_oracle.py --checkpoint "$TEACHER"

# ---- stage 2: validation gate (b) — step-0 drift vs banked flow npz ----
if [ ! -d "$STEP0" ]; then
    uv run python fontaine/scripts/materialize_snapflow_init.py --checkpoint "$TEACHER"
fi
DRIFT_NAME="eval__snapflow_step0__probe_s7_heun30"
uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2_drawsprobe_s7.json \
    --checkpoint "$STEP0" \
    --sample-steps 30 --sample-method heun \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${DRIFT_NAME}.json" \
    --dump-predictions "reports/${DRIFT_NAME}.npz" \
    2>&1 | tee "/home/ubuntu/${DRIFT_NAME}.log"
uv run python fontaine/scripts/snapflow_drift_gate.py \
    --probe "reports/${DRIFT_NAME}.npz"

# ---- stage 3: training (teacher recipe verbatim + pre-registered deltas) ----
uv run python -m bijou.train \
    --train-data \
        /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
        /home/ubuntu/datasets/mcobzarenco/so101_pick_place_v2 \
        /home/ubuntu/datasets/mcobzarenco/so101_pick_place_clean \
    --fps 30 --camera-counts 1 2 \
    --holdout-episodes 0.1 --split-seed 0 \
    --backbone google/gemma-4-e2b-it \
    --save-dir "outputs/train/${RUN}" \
    --init-from "$TEACHER" \
    --prompt-generate-bracket \
    --max-soft-tokens 140 \
    --stream-counts 4 4 7 \
    --self-attention-mode bidirectional \
    --time-conditioning adarms \
    --distill snapflow \
    --camera-kind-dropout 0.1 \
    --instruction-augment 0.5 \
    --condition-fields subgoal outcome smoothness \
    --condition-dropout 0.1 --subgoal-dropout 0.5 \
    --decoder-hidden 1024 --decoder-heads 8 \
    --decoder-intermediate 4096 --decoder-cross-heads 8 \
    --chunk-size 50 \
    --batch-size 24 \
    --steps 30000 \
    --decoder-lr 2.5e-5 \
    --warmup-steps 500 \
    --weight-decay 1e-5 \
    --grad-clip 1.0 \
    --log-every 20 --eval-every 500 --save-every 2500 \
    --num-workers 20 --prefetch-factor 4 --video-decoder-cache 4 \
    --seed 1 --eval-samples 256 --eval-seed 0 \
    --wandb-project bijou-dev --wandb-run-name "$RUN" \
    2>&1 | tee "/home/ubuntu/train_${RUN}.log"

# ---- stage 4: endpoint evals (primary + deployment headline) ----
CKPT="outputs/train/${RUN}/step_030000"
run_panel () {  # draws, tag, extra sampler args...
    local draws=$1 tag=$2; shift 2
    local name="eval__${RUN}__step_030000__panel_curated_v0_k4l2_${tag}"
    uv run python -m bijou.eval \
        --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
        --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
        --fps 30 --camera-counts 1 2 \
        --sample-plan plans/holdout_curated_v0_k4l2.json \
        --checkpoint "$CKPT" \
        --sample-draws "$draws" "$@" \
        --batch-size 32 --num-workers 20 --seed 0 \
        --report-samples 32 \
        --output-json "reports/${name}.json" \
        --report "reports/${name}.html" \
        2>&1 | tee "/home/ubuntu/${name}.log"
}
# PRIMARY: 1-NFE single draw (explicit shortcut flag — never inferred).
run_panel 1  1nfe_euler1 --sample-steps 1 --sample-method euler --target-time zero
# Deployment headline: mean-of-N at 1-NFE, N ∈ {10, 5}.
run_panel 10 1nfe_euler1_draws10 --sample-steps 1 --sample-method euler --target-time zero
run_panel 5  1nfe_euler1_draws5  --sample-steps 1 --sample-method euler --target-time zero
echo "=== SNAPFLOW DISTILL CHAIN DONE — write the results post before any adopt claim ==="
