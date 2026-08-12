#!/usr/bin/env bash
# GRPO signal probe — posts/2026-08-12-prereg-grpo-signal-probe.md
# (frozen 20:0xZ 08-12; owner GO 13:16Z, sequence 13:36Z satisfied).
# 2 anchor passes + 5 cells, seeds 0-14, workers=8, 30 s episodes,
# v3 frames (driver default), bf16 expert. Gate <= 3.5 GPU-h;
# tripwire: first completed cell's pace projects the total.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=outputs/sim/grpo_signal_probe
TRAIN=outputs/train
mkdir -p "$OUT"

ER60K="$TRAIN/fontaine_molmo2_er_60k_ddp4/step_060000"
TEACHER80K="$TRAIN/bijou_flow_artrunk_h1024_40k_ddp2/step_080000"
FTRIG4K="$TRAIN/fontaine_flow_snapdistill_ftrig_4k_1xh100/step_004000"

run_pass() {
    local name=$1
    shift
    echo "=== probe pass $name start $(date -u +%FT%TZ) ==="
    MUJOCO_GL=egl uv run python -m sim.rollout_sim_parallel "$@" \
        --seed 0 --num-seeds 15 --workers 8 \
        --episode-seconds 30 --execute-horizon 30 \
        --expert-dtype bfloat16 \
        --out-dir "$OUT/$name" --out-json "$OUT/$name/rows.json" \
        --rows-jsonl "$OUT/$name/rows.jsonl"
    echo "=== probe pass $name done $(date -u +%FT%TZ) ==="
}

# Anchor passes (deterministic, 15 episodes each)
run_pass anchor_er60k_greedy --checkpoint "$ER60K"
run_pass anchor_teacher80k_euler10 --checkpoint "$TEACHER80K" \
    --method euler --sample-steps 10

# Cells 1-2: er60k AR sampled
run_pass cell1_er60k_t10 --checkpoint "$ER60K" --ar-temperature 1.0 --draws 8
run_pass cell2_er60k_t16 --checkpoint "$ER60K" --ar-temperature 1.6 --draws 8

# Cell 5 early (shares teacher80k weights while hot): SDE euler-10 a=0.5
run_pass cell5_teacher80k_sde05 --checkpoint "$TEACHER80K" \
    --method euler --sample-steps 10 --sde-noise-level 0.5 --draws 8

# Cells 3-4: fresh-noise ODE, draw 0 = deterministic anchor
run_pass cell3_teacher80k_heun30 --checkpoint "$TEACHER80K" \
    --method heun --sample-steps 30 --draws 9
run_pass cell4_ftrig4k_euler1 --checkpoint "$FTRIG4K" \
    --method euler --sample-steps 1 --draws 9

echo "ALL PROBE PASSES DONE $(date -u +%FT%TZ)"
