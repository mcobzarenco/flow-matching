#!/usr/bin/env bash
# tiny10k resume after the 04:00:55Z 08-10 host-RAM OOM kill at step
# ~9,060 (unit fontaine-tiny10k, systemd oom-kill; GPU vram was fine
# at 36.6/80 — host RAM, DataLoader-worker class). Resumes the T1
# capacity rung from step_008750 (last completed async save) to the
# 10k endpoint, then the same chained panel_v2 eval @10000.
# Deltas vs launch_local_fontaine_molmo2_flow_tiny_h256_10k_1xh100.sh:
#   --resume step_008750 (full resume: weights+optim+sched+step;
#     --backbone-init-from DROPPED — mutually exclusive with --resume,
#     which loads the backbone from the checkpoint itself)
#   --seed 1 (trainer demands a fresh seed on resume; owner standing
#     seed policy agrees — eval-seed stays 0 so the probe ladder is
#     comparable)
#   --num-workers 6 (was 10) — cut host-RAM pressure for the ~940
#     remaining steps
# Everything else verbatim (b48c12, saves 1250, probe 500).
set -euo pipefail
cd /home/ubuntu/flow-matching

RUN=fontaine_molmo2_flow_tiny_h256_10k_1xh100
ENDPOINT=outputs/train/fontaine_molmo2_ar_60k_ddp4/step_060000
RESUME=outputs/train/$RUN/step_008750
PLAN_V2=plans/holdout_curated_v0_k4l2_panel_v2.json

[ -d "$RESUME" ] || { echo "no resume checkpoint $RESUME — abort"; exit 1; }
sha256sum -c - <<'SHAS'
2c98c3e14c3c73b7dec76b414112cbc3351946d9c9f4af7638a47f61d0e5b516  plans/holdout_curated_v0_k4l2_panel_v2.json
SHAS
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi

.venv/bin/torchrun --standalone --nproc-per-node=1 -m bijou.train \
    --train-data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --fps 30 --camera-counts 1 2 \
    --holdout-episodes 0.1 --split-seed 0 \
    --decoder flow --prompt-generate-bracket \
    --backbone allenai/Molmo2-4B --max-crops 1 \
    --resume "$RESUME" \
    --conditioning-streams residual \
    --self-attention-mode bidirectional --time-conditioning adarms \
    --decoder-hidden 256 --decoder-heads 4 \
    --decoder-intermediate 1024 --decoder-cross-heads 8 \
    --chunk-size 50 \
    --camera-kind-dropout 0.1 --instruction-augment 0.5 \
    --condition-fields subgoal outcome smoothness \
    --condition-dropout 0.1 --subgoal-dropout 0.5 \
    --decoder-lr 1e-4 --warmup-steps 500 --weight-decay 1e-5 \
    --grad-clip 10.0 \
    --steps 10000 --batch-size 48 \
    --backward-chunks 12 \
    --num-workers 6 --prefetch-factor 2 \
    --eval-samples 256 --eval-every 500 --log-every 20 \
    --save-every 1250 \
    --seed 1 --eval-seed 0 \
    --wandb-project fontaine --wandb-run-name "${RUN}_r8750" \
    --save-dir "outputs/train/$RUN" \
    2>&1 | tee "$HOME/train_${RUN}_r8750.log"

# ---- chained panel_v2 eval @10000 (the matched read) — verbatim ----
for STEP in 010000; do
    name="eval__${RUN}__step_${STEP}__panel_v2_heun30_draws1_stable"
    .venv/bin/torchrun --standalone --nproc-per-node=1 -m bijou.eval \
        --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
        --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
        --fps 30 --camera-counts 1 2 \
        --sample-plan "$PLAN_V2" \
        --checkpoint "outputs/train/$RUN/step_$STEP" \
        --sample-draws 1 --sample-steps 30 --sample-method heun \
        --noise-key stable \
        --batch-size 32 --num-workers 20 --seed 0 \
        --report-samples 32 \
        --output-json "reports/${name}.json" \
        --dump-predictions "reports/${name}.npz" \
        --report "reports/${name}.html" \
        2>&1 | tee "$HOME/logs/${name}.log"
done
echo "=== TINY EXPERT 10K RESUME DONE (train 8750->10000 + panel eval) ==="
