#!/usr/bin/env bash
# fontaine — #17 vision-unfreeze screen, ARM 1/2: FROZEN-CONTINUE
# (control), box 4xDDP, 5k warm-start continuation from the 40k
# endpoint. Run name: fontaine_molmo2_ar_vu5k_frozen_ddp4.
# PRE-REG: fontaine/blog/src/posts/2026-08-07-prereg-molmo2-vision-unfreeze.md
#   (amendment-3 design, owner-agreed 18:51Z 08-07; DRAFT until the
#   finalization amendment posts — NOTHING LAUNCHES off this script
#   before that amendment + an explicit owner go + the post-attach
#   window).
# SEQUENTIAL FROZEN FIRST — this arm runs before the thawed launcher;
#   its probe curve + endpoint bank are the thawed arm's kill-line
#   references (pre-reg §2/§4). The arms differ in EXACTLY ONE FLAG
#   (--backbone-vision-lr, thawed only).
# RECIPE: the 40k launcher byte-identical (data gate, collator,
#   freezing split, aux/condition/dropout flags, B12/rank 4xDDP global
#   48, ZeRO-1 + 6x2 chunked backward + --chunk-grad-allreduce) plus
#   the pinned continuation deltas (pre-reg §2, amendments 2+3):
#   --init-from step_040000 (weights-only, fresh AdamW — NOT --resume:
#   the thawed arm's extra vision param groups break
#   optimizer.load_state_dict, and fresh moments both arms make the
#   warm-restart transient common-mode), --steps 5000 --warmup-steps
#   500, LRs = 0.3x reheat of the 40k peaks (--decoder-lr 3e-5
#   --backbone-text-lr 6e-6, fresh 5k cosine to the 10% floors),
#   --seed 1 both arms (identical batches + τ/ε streams). Batch
#   semantics FROZEN — no downshift exists on this screen's ladder
#   (pre-reg §3/§7).
# E1 hard gate (train banner): 878 datasets / 38,571 episodes /
#   18,636,749 frames / dims 6/6 — any deviation aborts before step 1.
# KILL LINES (pre-reg §4, judged at save boundaries): NaN/inf; frozen
#   sanity = probe > (banked 40k endpoint probe, quoted in the
#   finalization amendment) + 2.0 x3 consecutive evals => the control
#   is broken, stop the screen, NO thawed launch; vram > 71 GiB; cost
#   gate 32 GPU-h screenwide (frozen est ~12.2 at 2.2 s/step).
# ASYNC-SAVE first-real-run validation (#18.9): check the "captured in
#   Xs" + "saved ... (async, ...)" lines at the first save boundary.
# LAUNCH (box): fontaine/scripts/run_detached.sh fontaine-vu5k-frozen \
#     bash fontaine/scripts/box/launch_box_fontaine_molmo2_vu5k_frozen_ddp4.sh
#   then uncomment + fill the prepared vu5k_frozen babysit.toml entry;
#   first-poll util+rate check per standing rule.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train reports

ENDPOINT=outputs/train/fontaine_molmo2_ar_40k_ddp4/step_040000
RUN_NAME=fontaine_molmo2_ar_vu5k_frozen_ddp4
STEPS=5000
ENDPOINT_STEP=$(printf "%06d" "$STEPS")

[ -d "$ENDPOINT" ] || { echo "no endpoint checkpoint $ENDPOINT — abort"; exit 1; }
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
    --decoder-lr 3e-5 --backbone-text-lr 6e-6 --grad-clip 100 \
    --steps "$STEPS" --warmup-steps 500 --batch-size 12 \
    --zero1 --backward-chunks 6 --chunk-grad-allreduce \
    --num-workers 20 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 2500 --log-every 20 \
    --seed 1 --wandb-project fontaine \
    --wandb-run-name "$RUN_NAME" \
    --save-dir "outputs/train/$RUN_NAME" \
    2>&1 | tee "/home/ubuntu/train_${RUN_NAME}.log"

# Chained endpoint panel — the 40k launcher's eval command verbatim
# (pre-reg §2 stems); its npz is the frozen side of the §5.1 paired Δ.
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
echo "=== VU5K FROZEN ARM DONE (train + endpoint panel eval) — launch the thawed arm next ==="
