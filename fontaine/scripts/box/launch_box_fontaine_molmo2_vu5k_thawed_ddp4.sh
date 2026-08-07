#!/usr/bin/env bash
# fontaine — #17 vision-unfreeze screen, ARM 2/2: THAWED-CONTINUE,
# box 4xDDP, 5k warm-start continuation from the 40k endpoint.
# Run name: fontaine_molmo2_ar_vu5k_thawed_ddp4.
# PRE-REG: fontaine/blog/src/posts/2026-08-07-prereg-molmo2-vision-unfreeze.md
#   (amendment-3 design, owner-agreed 18:51Z 08-07; DRAFT until the
#   finalization amendment posts — NOTHING LAUNCHES off this script
#   before that amendment + an explicit owner go + the post-attach
#   window).
# RUNS SECOND — guarded below on the frozen arm's endpoint existing
#   (pre-reg §2 frozen-first ordering: no thawed launch without the
#   control landed; §4 frozen-sanity line must not have fired).
# THE ONE FLAG: --backbone-vision-lr 6e-6 (amendment 3 — vision LR
#   tied to the text group, 6e-6 reheat peak annealing together;
#   bijou.train hard-aborts if the backbone has no vision tower, so a
#   silent no-op unfreeze cannot happen). Everything else is the
#   frozen launcher byte-identical, including --seed 1 (identical
#   batches + τ/ε streams across arms).
# MEMORY LADDER GUARD (pre-reg §3 — projected peak 70-72 GiB
#   STRADDLES the 71 gate): this launcher REFUSES to run without
#   fontaine/harness/state/vu5k_mem_ready, written by the execution
#   item's 150-step smoke FROM the endpoint checkpoint. Format
#   (KEY=VALUE lines, sourced):
#       RUNG=R0|R1|R2          # winning ladder rung
#       BACKWARD_CHUNKS=6|12   # R0=6; R1/R2=12 (microbatch 1)
#       ACT_CKPT=0|1           # R2 only: --activation-checkpointing
#       VRAM_PEAK_GIB=<smoke peak, must be <= 71.0>
#       SMOKE_UTC=<date -u of the smoke>
#   All-red ladder => NO marker => this script refuses; named ways
#   forward are in pre-reg §3 (no matched downshift exists — batch
#   semantics changes poison the contrast, §7).
# KILL LINES (pre-reg §4, judged at save boundaries): NaN/inf;
#   vision-damage = thawed probe > frozen-arm probe at the matched
#   continuation step + 2.0, x3 consecutive evals, after step 1000;
#   vram > 71 GiB; cost gate 32 GPU-h screenwide (thawed est ~13.9 at
#   a projected 2.4-2.6 s/step with the tower backward).
# SMOKE CROSS-CHECK: the trainable-param banner must count the tower
#   (~4.3e8 vision params) — quoted in the finalization amendment.
# LAUNCH (box): fontaine/scripts/run_detached.sh fontaine-vu5k-thawed \
#     bash fontaine/scripts/box/launch_box_fontaine_molmo2_vu5k_thawed_ddp4.sh
#   then uncomment + fill the prepared vu5k_thawed babysit.toml entry
#   (prune the finished frozen entry); first-poll util+rate check per
#   standing rule.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train reports

ENDPOINT=outputs/train/fontaine_molmo2_ar_40k_ddp4/step_040000
FROZEN_ENDPOINT=outputs/train/fontaine_molmo2_ar_vu5k_frozen_ddp4/step_005000
MEM_READY=fontaine/harness/state/vu5k_mem_ready
RUN_NAME=fontaine_molmo2_ar_vu5k_thawed_ddp4
STEPS=5000
ENDPOINT_STEP=$(printf "%06d" "$STEPS")

[ -d "$ENDPOINT" ] || { echo "no endpoint checkpoint $ENDPOINT — abort"; exit 1; }
[ -d "$FROZEN_ENDPOINT" ] || {
    echo "FROZEN-FIRST ORDER: no frozen-arm endpoint $FROZEN_ENDPOINT — the control must land (and pass its sanity line) before the thawed arm launches. Abort."; exit 1; }
[ -f "$MEM_READY" ] || {
    echo "NO MEMORY-LADDER RECORD: $MEM_READY missing — run the 150-step thawed smoke from the endpoint checkpoint first (pre-reg §3; execution-item finalization cell). Abort."; exit 1; }
# shellcheck disable=SC1090
. "$MEM_READY"
: "${BACKWARD_CHUNKS:?vu5k_mem_ready lacks BACKWARD_CHUNKS}"
: "${ACT_CKPT:?vu5k_mem_ready lacks ACT_CKPT}"
echo "memory ladder record: RUNG=${RUNG:-?} BACKWARD_CHUNKS=$BACKWARD_CHUNKS ACT_CKPT=$ACT_CKPT VRAM_PEAK_GIB=${VRAM_PEAK_GIB:-?} (smoke ${SMOKE_UTC:-?})"
ACT_CKPT_ARGS=()
[ "$ACT_CKPT" = "1" ] && ACT_CKPT_ARGS=(--activation-checkpointing)

sha256sum -c - <<'SHAS'
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
SHAS
for g in 0 1 2 3; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$mem" -gt 1024 ]; then echo "GPU $g busy (${mem} MiB) — abort"; exit 1; fi
done

.venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.train \
    --train-data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --fps 30 --camera-counts 1 2 \
    --holdout-episodes 0.1 --split-seed 0 \
    --decoder ar_backbone \
    --backbone allenai/Molmo2-4B \
    --max-crops 1 \
    --fast-tokenizer mcobzarenco/bijou-checkpoints/fast_tokenizer_v2 \
    --aux-fields subgoal holding progress event visible \
    --aux-dropout 0.0 --field-dropout 0.1 \
    --condition-fields subgoal outcome smoothness \
    --condition-dropout 0.1 --subgoal-dropout 0.5 \
    --instruction-augment 0.5 \
    --camera-kind-dropout 0.1 \
    --init-from "$ENDPOINT" \
    --decoder-lr 3e-5 --backbone-text-lr 6e-6 --backbone-vision-lr 6e-6 \
    --grad-clip 100 \
    --steps "$STEPS" --warmup-steps 500 --batch-size 12 \
    --zero1 --backward-chunks "$BACKWARD_CHUNKS" --chunk-grad-allreduce \
    "${ACT_CKPT_ARGS[@]}" \
    --num-workers 20 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 2500 --log-every 20 \
    --seed 1 --wandb-project fontaine \
    --wandb-run-name "$RUN_NAME" \
    --save-dir "outputs/train/$RUN_NAME" \
    2>&1 | tee "/home/ubuntu/train_${RUN_NAME}.log"

# Chained endpoint panel — the 40k launcher's eval command verbatim
# (pre-reg §2 stems); its npz is the thawed side of the §5.1 paired Δ.
.venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2.json \
    --checkpoint outputs/train/$RUN_NAME/step_$ENDPOINT_STEP \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --dump-predictions "reports/eval__${RUN_NAME}__step_${ENDPOINT_STEP}__panel_curated_v0_k4l2.npz" \
    --output-json "reports/eval__${RUN_NAME}__step_${ENDPOINT_STEP}__panel_curated_v0_k4l2.json" \
    --report "reports/eval__${RUN_NAME}__step_${ENDPOINT_STEP}__panel_curated_v0_k4l2.html" \
    2>&1 | tee "/home/ubuntu/eval_${RUN_NAME}.log"
echo "=== VU5K THAWED ARM DONE (train + endpoint panel eval) — run the §5 frozen reads ==="
