#!/usr/bin/env bash
# fontaine — molmo2 endpoint panel, ONE command once the box frees
# (ideas #19): greedy arm (only if the training launcher's chained
# endpoint eval did not land) + the draws10_t1 arm, cost-gated.
# Pre-reg: fontaine/blog/src/posts/2026-08-06-prereg-ar-sampled-draws.md
#   molmo2 arm row: "greedy + _draws10_t1, same command stems", decided
#   by the same pre-registered cost gate — measure the rate over the
#   first ~200 frames; a full-panel draws10 projecting > 24 GPU-h drops
#   BOTH arms to the frozen q4 subset (comparison rows re-pooled from
#   banked npzs); the switch is recorded, never silent.
# Greedy arm: normally produced by the chained eval inside
#   launch_box_fontaine_molmo2_ar_40k_ddp4.sh (panel_curated_v0_k4l2
#   stems). This script re-runs it with byte-same stems ONLY when that
#   report json is missing — the endpoint read survives a dead chain.
# Draws arm: --ar-temperature 1.0 --sample-draws 10 (instrument 78c9f56;
#   Molmo2KVCache snapshot/restore + T->0 greedy-recovery oracles in
#   tests/test_molmo2_ar_sampling.py), 4-GPU sharded torchrun — the
#   merge is index-sorted and world-size-invariant for AR decodes.
# Cost gate mechanized: fontaine/scripts/draws_rate_gate.py polls the
#   log (rank-0 shard counts; projection is world-invariant). FALLBACK
#   (exit 2) kills the panel run and relaunches on the q4 subset
#   (stateprobe_q4_draws10_t1 stems). INDETERMINATE (exit 1) leaves the
#   run alive — the babysit registry's 24 GPU-h gate is the backstop.
# babysit.toml: uncomment the prepared molmo2_draws10_t1 entry (bottom
#   of fontaine/harness/babysit.toml) and fill started_utc AT LAUNCH.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p "$HOME/logs" reports

RUN=fontaine_molmo2_ar_40k_ddp4
STEP=040000
CKPT=outputs/train/$RUN/step_$STEP
PLAN=plans/holdout_curated_v0_k4l2.json
PLAN_Q4=plans/holdout_curated_v0_k4l2_stateprobe_q4.json
STEM=panel_k4l2_draws10_t1
STEM_Q4=stateprobe_q4_draws10_t1

# Guards: endpoint reached, plans are the frozen ones, all 4 GPUs free.
[ -d "$CKPT" ] || { echo "no endpoint checkpoint $CKPT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
876c39c8fe2b3cb16945a40c35ec157c907b4f7417e7dfd0b6cf46dd47355ef5  plans/holdout_curated_v0_k4l2_stateprobe_q4.json
SHAS
for g in 0 1 2 3; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$mem" -gt 1024 ]; then echo "GPU $g busy (${mem} MiB) — abort"; exit 1; fi
done

# Greedy arm — byte-same stems as the chained endpoint eval; skipped
# when its report already exists (the normal case).
greedy_json=reports/eval__${RUN}__step_${STEP}__panel_curated_v0_k4l2.json
if [ -s "$greedy_json" ]; then
    echo "greedy arm banked ($greedy_json) — skipping re-run"
else
    echo "greedy arm MISSING — running the chained eval's stems"
    .venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.eval \
        --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
        --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
        --fps 30 --camera-counts 1 2 \
        --sample-plan "$PLAN" \
        --checkpoint "$CKPT" \
        --batch-size 32 --num-workers 20 --seed 0 \
        --report-samples 32 \
        --dump-predictions "reports/eval__${RUN}__step_${STEP}__panel_curated_v0_k4l2.npz" \
        --output-json "$greedy_json" \
        --report "reports/eval__${RUN}__step_${STEP}__panel_curated_v0_k4l2.html" \
        2>&1 | tee "$HOME/logs/eval__${RUN}__step_${STEP}__panel_curated_v0_k4l2.log"
fi

draws_eval() { # $1 = plan path, $2 = stem; log + reports carry the stem
    local name="eval__${RUN}__step_${STEP}__$2"
    .venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.eval \
        --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
        --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
        --fps 30 --camera-counts 1 2 \
        --sample-plan "$1" \
        --checkpoint "$CKPT" \
        --ar-temperature 1.0 --sample-draws 10 \
        --batch-size 32 --num-workers 20 --seed 0 \
        --report-samples 0 \
        --output-json "reports/${name}.json" \
        --dump-predictions "reports/${name}.npz" \
        --report "reports/${name}.html" \
        > "$HOME/logs/${name}.log" 2>&1
}

echo "draws arm: full panel, gate watching $HOME/logs/eval__${RUN}__step_${STEP}__${STEM}.log"
draws_eval "$PLAN" "$STEM" &
eval_pid=$!
set +e
.venv/bin/python fontaine/scripts/draws_rate_gate.py \
    --log "$HOME/logs/eval__${RUN}__step_${STEP}__${STEM}.log" \
    --ngpu 4 --gate-gpu-hours 24 --min-frames 200
gate_rc=$?
set -e
if [ "$gate_rc" -eq 2 ]; then
    echo "COST GATE FALLBACK — killing the full-panel run, relaunching on q4 ($STEM_Q4)"
    pkill -TERM -f "$STEM" || true
    wait "$eval_pid" || true
    for i in $(seq 1 60); do  # GPUs must drain before the relaunch
        busy=0
        for g in 0 1 2 3; do
            mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
            [ "$mem" -gt 1024 ] && busy=1
        done
        [ "$busy" -eq 0 ] && break
        [ "$i" -eq 30 ] && pkill -KILL -f "$STEM" || true
        sleep 10
    done
    draws_eval "$PLAN_Q4" "$STEM_Q4"
    echo "q4 fallback run complete — record the switch in now.md + Discord"
elif [ "$gate_rc" -eq 0 ]; then
    echo "COST GATE PASS — full panel proceeds"
    wait "$eval_pid"
else
    echo "RATE GATE INDETERMINATE (rc=$gate_rc) — run left alive; babysit gpu_hours_max is the backstop"
    wait "$eval_pid"
fi
echo "=== MOLMO2 ENDPOINT PANEL DONE (greedy banked + draws10_t1) ==="
