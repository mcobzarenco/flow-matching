#!/usr/bin/env bash
# fontaine — attach-screen ARM F (frozen trunk), box 4xDDP, 10k steps.
# PRE-REG: fontaine/blog/src/posts/2026-08-07-prereg-molmo2-attach-screen.md
#   (+ its pre-launch amendment). SEQUENTIAL F FIRST — this script runs
#   before the K launcher; the seam is the ONLY contrast between arms.
# ARM: flow expert (h1024/8h/4096/8xh, adaRMS, bidirectional, depth 12
#   structural from the pinned tap rule: 12 taps @ stride 3, layers
#   2,5,...,35) reading residual taps off the HARD-FROZEN molmo2 60k
#   endpoint trunk via --backbone-init-from. Data/topology = phase 1:
#   eff-48 (B12 x 4), same split/seed, eval 256 @ 500, save @ 1250
#   (pre-reg amendment 2: async saves landed post-pre-reg; cadence
#   halved from 2500, matched BOTH arms — every posted judgment
#   boundary stays a save boundary).
# OPENS STRICTLY AFTER (pre-reg order): endpoint saved -> chained
#   greedy endpoint eval + #19 draws arm done -> instrument oracles
#   green at HEAD -> attachment-decision owner steer window passed.
# COST GATE (mechanized, attach_rate_gate.py): first ~200 steps project
#   the BATCH total; > 70 GPU-h => 5k matched downshift for BOTH arms
#   (marker file below; the K launcher honors it). Extra term pinned
#   here: K estimate 2.6 s/step x 10k x 4 / 3600 ~= 28.9 + panel evals
#   ~4 x 2 + K drift AR panel ~3 => EXTRA_GPU_HOURS ~= 40.
# K1-STYLE KILL (babysit-surfaced, judged at save boundaries): in-run
#   probe > phase-1 curve at the matched step + 3.0 at any eval >= 5k
#   (bars: 12.6394@5000, 11.6356@7500; @10000 bar = phase-1 @10000
#   + 3.0 — see babysit.toml entry). vram_alloc_peak <= 71 GiB.
# CHAINED: panel-v2 endpoint eval (heun30/draws1/stable stems — the
#   Delta_seam read's F side), 4-GPU sharded.
# babysit.toml: uncomment the prepared attach_F entry + fill
#   started_utc AT LAUNCH; first-poll util+rate check per standing rule.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train reports "$HOME/logs"

ENDPOINT=outputs/train/fontaine_molmo2_ar_60k_ddp4/step_060000  # REPOINTED per amendment 2 (60k read IMPROVED -0.1388, 2026-08-09)
PLAN_V2=plans/holdout_curated_v0_k4l2_panel_v2.json
DOWNSHIFT_MARKER=fontaine/harness/state/attach_screen_5k_downshift
BATCH="${BATCH:-12}"
BACKWARD_CHUNKS="${BACKWARD_CHUNKS:-6}"   # same value BOTH arms (pre-reg)
EXTRA_GPU_HOURS="${EXTRA_GPU_HOURS:-40}"  # arithmetic in header comment

[ -d "$ENDPOINT" ] || { echo "no endpoint checkpoint $ENDPOINT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
2c98c3e14c3c73b7dec76b414112cbc3351946d9c9f4af7638a47f61d0e5b516  plans/holdout_curated_v0_k4l2_panel_v2.json
SHAS
for g in 0 1 2 3; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$mem" -gt 1024 ]; then echo "GPU $g busy (${mem} MiB) — abort"; exit 1; fi
done

STEPS=10000
if [ -e "$DOWNSHIFT_MARKER" ]; then
    STEPS=5000
    echo "=== 5K DOWNSHIFT MARKER PRESENT — matched 5k-screen schedule ==="
fi

run_name() { echo "fontaine_molmo2_flow_frozen_$(($1 / 1000))k_ddp4"; }

launch_train() { # $1 = steps; backgrounds torchrun, sets RUN + TRAIN_PID
    RUN=$(run_name "$1")
    echo "launching $RUN ($1 steps) — first-poll util+rate check applies"
    .venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.train \
        --train-data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
        --fps 30 --camera-counts 1 2 \
        --holdout-episodes 0.1 --split-seed 0 \
        --decoder flow --prompt-generate-bracket \
        --backbone allenai/Molmo2-4B --max-crops 1 \
        --backbone-init-from "$ENDPOINT" \
        --conditioning-streams residual \
        --self-attention-mode bidirectional --time-conditioning adarms \
        --decoder-hidden 1024 --decoder-heads 8 \
        --decoder-intermediate 4096 --decoder-cross-heads 8 \
        --chunk-size 50 \
        --camera-kind-dropout 0.1 --instruction-augment 0.5 \
        --condition-fields subgoal outcome smoothness \
        --condition-dropout 0.1 --subgoal-dropout 0.5 \
        --decoder-lr 1e-4 --warmup-steps 500 --weight-decay 1e-5 \
        --grad-clip 10.0 \
        --steps "$1" --batch-size "$BATCH" \
        --zero1 --backward-chunks "$BACKWARD_CHUNKS" --chunk-grad-allreduce \
        --num-workers 20 --prefetch-factor 4 \
        --eval-samples 256 --eval-every 500 --save-every 1250 --log-every 20 \
        --seed 0 --eval-seed 0 \
        --wandb-project fontaine --wandb-run-name "$RUN" \
        --save-dir "outputs/train/$RUN" \
        2>&1 | tee "/home/ubuntu/train_${RUN}.log" &
    TRAIN_PID=$!
}

launch_train "$STEPS"
if [ "$STEPS" -eq 10000 ]; then
    set +e
    .venv/bin/python fontaine/scripts/attach_rate_gate.py \
        --jsonl "outputs/train/$RUN/train_log.jsonl" \
        --arm-steps 10000 --ngpu 4 \
        --gate-gpu-hours 70 --extra-gpu-hours "$EXTRA_GPU_HOURS"
    gate_rc=$?
    set -e
    if [ "$gate_rc" -eq 2 ]; then
        echo "COST GATE FALLBACK — 5k matched downshift for BOTH arms (marker: $DOWNSHIFT_MARKER)"
        touch "$DOWNSHIFT_MARKER"
        pkill -TERM -f "$RUN" || true
        wait "$TRAIN_PID" || true
        for i in $(seq 1 60); do  # GPUs must drain before the relaunch
            busy=0
            for g in 0 1 2 3; do
                mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
                [ "$mem" -gt 1024 ] && busy=1
            done
            [ "$busy" -eq 0 ] && break
            [ "$i" -eq 30 ] && pkill -KILL -f "$RUN" || true
            sleep 10
        done
        STEPS=5000
        launch_train "$STEPS"
        wait "$TRAIN_PID"
    elif [ "$gate_rc" -eq 0 ]; then
        echo "COST GATE PASS — 10k arm proceeds"
        wait "$TRAIN_PID"
    else
        echo "RATE GATE INDETERMINATE (rc=$gate_rc) — run left alive; babysit registry is the backstop"
        wait "$TRAIN_PID"
    fi
else
    wait "$TRAIN_PID"
fi

# Chained panel-v2 endpoint eval — the Delta_seam read's F side
# (paired per-frame vs K from the npz dumps; arch-batch conventions).
ENDPOINT_STEP=$(printf "%06d" "$STEPS")
name="eval__${RUN}__step_${ENDPOINT_STEP}__panel_v2_heun30_draws1_stable"
.venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan "$PLAN_V2" \
    --checkpoint "outputs/train/$RUN/step_$ENDPOINT_STEP" \
    --sample-draws 1 --sample-steps 30 --sample-method heun \
    --noise-key stable \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --output-json "reports/${name}.json" \
    --dump-predictions "reports/${name}.npz" \
    --report "reports/${name}.html" \
    2>&1 | tee "$HOME/logs/${name}.log"
echo "=== ARM F DONE (train + panel-v2 eval: reports/${name}.json) — launch K next ==="
