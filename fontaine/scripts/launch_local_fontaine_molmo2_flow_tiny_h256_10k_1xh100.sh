#!/usr/bin/env bash
# Tiny-expert capacity rung (T1): h256/d12 flow expert, frozen 60k
# trunk, 10k steps @ eff-48 (owner 20:08:53Z: "your original plan" —
# matched-F design; the 40k/biggest-batch amendments reverted), LOCAL
# single H100. Pre-reg:
# fontaine/blog/src/posts/2026-08-09-prereg-tiny-expert-40k.md
# (owner-approved 19:59Z 08-09; final design per owner 20:08:53Z).
# Fit ladder first (150-step rungs b48c12 -> b48c24, green = rc0 AND
# vram_alloc_peak_gib <= 74.0), then the 10k run (eff-48, saves 1250
# — F's cadence), then ONE chained single-GPU panel_v2 eval @10000
# (the matched read vs banked F@10k). Recipe is the F arm's verbatim
# (launch_box_fontaine_molmo2_attach_F_10k_ddp4.sh) except
# hidden/heads/intermediate (the contrast) and single-GPU batching
# (48x1 vs 12x4 — same eff-batch, same LR schedule).
set -euo pipefail
cd /home/ubuntu/flow-matching

RUN=fontaine_molmo2_flow_tiny_h256_10k_1xh100
ENDPOINT=outputs/train/fontaine_molmo2_ar_60k_ddp4/step_060000
PLAN_V2=plans/holdout_curated_v0_k4l2_panel_v2.json
STEPS=10000
VRAM_GATE_GIB=74.0

mkdir -p outputs/train reports "$HOME/logs"
[ -d "$ENDPOINT" ] || { echo "no trunk checkpoint $ENDPOINT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
2c98c3e14c3c73b7dec76b414112cbc3351946d9c9f4af7638a47f61d0e5b516  plans/holdout_curated_v0_k4l2_panel_v2.json
SHAS
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi

train_cmd() { # $1=steps $2=batch $3=chunks $4=save_dir $5=save_every
    .venv/bin/torchrun --standalone --nproc-per-node=1 -m bijou.train \
        --train-data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
        --fps 30 --camera-counts 1 2 \
        --holdout-episodes 0.1 --split-seed 0 \
        --decoder flow --prompt-generate-bracket \
        --backbone allenai/Molmo2-4B --max-crops 1 \
        --backbone-init-from "$ENDPOINT" \
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
        --steps "$1" --batch-size "$2" \
        --backward-chunks "$3" \
        --num-workers 10 --prefetch-factor 2 \
        --eval-samples 256 --eval-every 500 --log-every 20 \
        --save-every "$5" \
        --seed 0 --eval-seed 0 \
        --wandb-project fontaine --wandb-run-name "$(basename "$4")" \
        --save-dir "$4"
}

peak_gib() { # $1 = jsonl; max vram_alloc_peak_gib over the log
    .venv/bin/python - "$1" <<'EOF'
import json, sys
peak = 0.0
for line in open(sys.argv[1]):
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    v = d.get("vram_alloc_peak_gib")
    if isinstance(v, (int, float)):
        peak = max(peak, v)
print(f"{peak:.2f}")
EOF
}

# ---- fit ladder ----
# SKIP_LADDER=1: reuse the 2026-08-09 20:1xZ green rung (b48c12,
# 13.0 GiB vs 74 gate) instead of re-proving it on a relaunch.
WIN_BATCH=""; WIN_CHUNKS=""
if [ "${SKIP_LADDER:-0}" = 1 ]; then WIN_BATCH=48; WIN_CHUNKS=12; fi
[ -n "$WIN_BATCH" ] || for rung in "48 12" "48 24"; do
    set -- $rung; B=$1; C=$2
    DIR=outputs/train/tinyfit_b${B}c${C}
    rm -rf "$DIR"
    echo "=== FIT LADDER rung batch=$B chunks=$C ==="
    set +e
    train_cmd 150 "$B" "$C" "$DIR" 150 \
        2>&1 | tee "$HOME/logs/tinyfit_b${B}c${C}.log"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then echo "rung b${B}c${C} rc=$rc — RED"; continue; fi
    peak=$(peak_gib "$DIR/train_log.jsonl")
    ok=$(.venv/bin/python -c "print(1 if $peak <= $VRAM_GATE_GIB else 0)")
    echo "rung b${B}c${C} rc=0 vram_alloc_peak=${peak} GiB (gate $VRAM_GATE_GIB) -> $([ "$ok" = 1 ] && echo GREEN || echo RED)"
    if [ "$ok" = 1 ]; then WIN_BATCH=$B; WIN_CHUNKS=$C; break; fi
done
[ -n "$WIN_BATCH" ] || { echo "=== FIT LADDER ALL RED — no launch, owner steer ==="; exit 2; }
echo "=== FIT LADDER WINNER batch=$WIN_BATCH chunks=$WIN_CHUNKS ==="

# ---- the 10k run (matched-F) ----
train_cmd "$STEPS" "$WIN_BATCH" "$WIN_CHUNKS" "outputs/train/$RUN" 1250 \
    2>&1 | tee "$HOME/train_${RUN}.log"

# ---- chained panel_v2 eval @10000 (the matched read) ----
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
echo "=== TINY EXPERT 10K DONE (fit ladder + train + panel eval) ==="
