#!/usr/bin/env bash
# fontaine — attach-screen ARM K (KI-joint), box 4xDDP, 10k steps.
# PRE-REG: fontaine/blog/src/posts/2026-08-07-prereg-molmo2-attach-screen.md
#   (+ pre-launch amendment). RUNS AFTER ARM F (sequential, F first).
# ARM: the F recipe with the SEAM as the only contrast — trunk live,
#   phase-1 CE objective continuing VERBATIM beside the flow loss at
#   fixed alpha=1 (--joint-ce), stop-grad on the expert->trunk seam
#   (--seam-stop-grad). CE branch = phase-1 flags verbatim: same aux
#   fields/dropouts, --backbone-text-lr 2e-5, --grad-clip 100 (the
#   phase-1 clip — pre-reg "CE branch verbatim" list; F carries the
#   flow lineage's 10.0, and neither clip binds at measured grad norms
#   ~5). AMENDMENT: the rider's FAST tables continue from the
#   endpoint's expert.safetensors (loaded by --backbone-init-from +
#   --joint-ce in bijou.train) — never fresh.
# HARD PREREQUISITE (pre-reg instrument item 4): #20 activation
#   checkpointing landed + an F1-style smoke memory ladder run at THIS
#   batch — phase 1 alone sat at 67.07/71 GiB with no expert riding.
#   The K_MEM_READY env guard below refuses a blind launch. If K cannot
#   fit eff-48 under 71 GiB checkpointed, BOTH arms downshift batch
#   together (matched, loudly echoed) — never K alone.
# COST GATE (attach_rate_gate.py): projected batch total > 70 GPU-h =>
#   5k matched downshift BOTH arms (marker below; if F already ran 10k,
#   its matched read re-evals from F's step_005000 checkpoint — save
#   every 2500 makes 5000 a save boundary). Extra term pinned here:
#   F actual ~14 GPU-h (replace with the measured number at launch) +
#   panel evals ~4 x 2 + drift AR panel ~3 => EXTRA_GPU_HOURS ~= 25.
# K1-STYLE KILL: probe > phase-1 curve at matched step + 3.0 at any
#   eval >= 5k (bars 12.6394@5000, 11.6356@7500, @10000 in
#   babysit.toml); vram_alloc_peak <= 71 GiB. CE-HEALTH WATCH (record,
#   not gate): loss_aux (per-token action CE) vs phase-1 loss_action at
#   the matched step — rising CE under stop-grad is the AEGIS drift
#   signal; it feeds read 4.
# CHAINED: panel-v2 eval (Delta_seam K side) + the AR-VIEW
#   materialization + greedy AR panel on the phase-1 k4l2 plan — read
#   4's trunk-drift number vs the 40k endpoint AR panel (band 0.3).
# babysit.toml: uncomment the prepared attach_K entry + fill
#   started_utc AT LAUNCH; first-poll util+rate check per standing rule.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p outputs/train reports "$HOME/logs"

: "${K_MEM_READY:?K arm refused: set K_MEM_READY=1 only after #20 activation checkpointing landed AND the smoke memory ladder passed at this BATCH (see header)}"

ENDPOINT=outputs/train/fontaine_molmo2_ar_40k_ddp4/step_040000
PLAN_V2=plans/holdout_curated_v0_k4l2_panel_v2.json
PLAN_AR=plans/holdout_curated_v0_k4l2.json
DOWNSHIFT_MARKER=fontaine/harness/state/attach_screen_5k_downshift
BATCH="${BATCH:-12}"                      # matched with F, never K alone
BACKWARD_CHUNKS="${BACKWARD_CHUNKS:-6}"   # same value BOTH arms (pre-reg)
EXTRA_GPU_HOURS="${EXTRA_GPU_HOURS:-25}"  # arithmetic in header comment

[ -d "$ENDPOINT" ] || { echo "no endpoint checkpoint $ENDPOINT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
2c98c3e14c3c73b7dec76b414112cbc3351946d9c9f4af7638a47f61d0e5b516  plans/holdout_curated_v0_k4l2_panel_v2.json
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
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

run_name() { echo "fontaine_molmo2_flow_kijoint_$(($1 / 1000))k_ddp4"; }

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
        --seam-stop-grad --joint-ce \
        --fast-tokenizer mcobzarenco/bijou-checkpoints/fast_tokenizer_v2 \
        --aux-fields subgoal holding progress event visible \
        --aux-dropout 0.0 --field-dropout 0.1 \
        --self-attention-mode bidirectional --time-conditioning adarms \
        --decoder-hidden 1024 --decoder-heads 8 \
        --decoder-intermediate 4096 --decoder-cross-heads 8 \
        --chunk-size 50 \
        --camera-kind-dropout 0.1 --instruction-augment 0.5 \
        --condition-fields subgoal outcome smoothness \
        --condition-dropout 0.1 --subgoal-dropout 0.5 \
        --decoder-lr 1e-4 --backbone-text-lr 2e-5 \
        --warmup-steps 500 --weight-decay 1e-5 \
        --grad-clip 100 \
        --steps "$1" --batch-size "$BATCH" \
        --zero1 --backward-chunks "$BACKWARD_CHUNKS" --chunk-grad-allreduce \
        --num-workers 20 --prefetch-factor 4 \
        --eval-samples 256 --eval-every 500 --save-every 2500 --log-every 20 \
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
        echo "COST GATE FALLBACK — 5k matched downshift (marker: $DOWNSHIFT_MARKER; F's matched read: re-eval F step_005000)"
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

ENDPOINT_STEP=$(printf "%06d" "$STEPS")
CKPT="outputs/train/$RUN/step_$ENDPOINT_STEP"

# Chained 1/2: panel-v2 eval — the Delta_seam read's K side.
name="eval__${RUN}__step_${ENDPOINT_STEP}__panel_v2_heun30_draws1_stable"
.venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan "$PLAN_V2" \
    --checkpoint "$CKPT" \
    --sample-draws 1 --sample-steps 30 --sample-method heun \
    --noise-key stable \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --output-json "reports/${name}.json" \
    --dump-predictions "reports/${name}.npz" \
    --report "reports/${name}.html" \
    2>&1 | tee "$HOME/logs/${name}.log"

# Chained 2/2: READ 4 — trunk-drift. Materialize the AR view of the
# joint checkpoint (rider := decoder) and greedy-eval it on the SAME
# k4l2 plan/stems family as the 40k endpoint AR number. Band: |Δ_AR|
# <= 0.3 (frozen); a K win that breaks the band escalates to owner
# steer with AEGIS projection as the named (banked, unbuilt) repair.
.venv/bin/python fontaine/scripts/materialize_joint_ar_view.py --checkpoint "$CKPT"
view_name="eval__${RUN}_ar_view__step_${ENDPOINT_STEP}__panel_curated_v0_k4l2"
.venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan "$PLAN_AR" \
    --checkpoint "${CKPT}_ar_view" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --output-json "reports/${view_name}.json" \
    --dump-predictions "reports/${view_name}.npz" \
    --report "reports/${view_name}.html" \
    2>&1 | tee "$HOME/logs/${view_name}.log"
echo "=== ARM K DONE (train + panel-v2 + AR-view drift panel: reports/${view_name}.json) ==="
