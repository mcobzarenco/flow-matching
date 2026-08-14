#!/usr/bin/env bash
# Stage 1 of the wrist-transfer screen (FINAL pre-reg
# posts/2026-08-14-prereg-wrist-transfer-screen.md §5) — sequential
# cells in ONE detached unit, launched via run_detached.sh:
#
#   entry gate: P1×W0 twice on seeds 0-9, per-seed rows bit-equal
#               (§3 pairing premise; ABORT on divergence)
#   hold floor: 25 seeds (validity gate replication under the current
#               visual config)
#   cells:      P1×W0(100), P1×W1(100), P1×W3(100), T1 top-blackout(25)
#
# All cells: ftrig4k step_004000, euler-1, draw-0 deterministic,
# episode 30 s, workers 8, frozen seeds from 0. Out-jsons land in
# outputs/sim/wrist_screen/; a DONE marker closes the unit. Reads +
# gates (sanity band, T1 gate, honesty ordering) run in the boundary
# session, never here.
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT=outputs/train/fontaine_flow_snapdistill_ftrig_4k_1xh100/step_004000
OUT=outputs/sim/wrist_screen
COMMON=(--checkpoint "$CKPT" --episode-seconds 30 --sample-steps 1
        --method euler --workers 8)
mkdir -p "$OUT"

run_cell() { # name, extra args...
    local name="$1"; shift
    echo "[stage1] CELL $name START $(date -u +%H:%M:%SZ)"
    MUJOCO_GL=egl uv run python -m sim.rollout_sim_parallel \
        "${COMMON[@]}" "$@" --out-dir "$OUT/videos_$name" \
        --out-json "$OUT/stage1_$name.json"
    echo "[stage1] CELL $name DONE $(date -u +%H:%M:%SZ)"
}

# --- stage-1 entry gate: W0 determinism, seeds 0-9 twice ---------------
run_cell det_a --seed 0 --num-seeds 10 --wrist-transform none
run_cell det_b --seed 0 --num-seeds 10 --wrist-transform none
uv run python - <<'EOF'
import json
a = json.load(open("outputs/sim/wrist_screen/stage1_det_a.json"))["episodes"]
b = json.load(open("outputs/sim/wrist_screen/stage1_det_b.json"))["episodes"]
keys = ("seed", "initial_cm", "min_cm", "final_cm", "success_tick",
        "spawn_xy", "reset_strikes", "ticks", "progress_final_cm")
for ra, rb in zip(a, b, strict=True):
    diffs = {k: (ra[k], rb[k]) for k in keys if ra[k] != rb[k]}
    if diffs:
        raise SystemExit(f"W0 DETERMINISM ABORT seed {ra['seed']}: {diffs}")
print(f"[stage1] W0 determinism gate PASS ({len(a)} seeds bit-equal)")
EOF

# --- hold floor replication -------------------------------------------
echo "[stage1] CELL hold START $(date -u +%H:%M:%SZ)"
MUJOCO_GL=egl uv run python -m sim.rollout_sim_parallel \
    --hold --seed 0 --num-seeds 25 --episode-seconds 30 --workers 8 \
    --out-dir "$OUT/videos_hold" --out-json "$OUT/stage1_hold.json"
echo "[stage1] CELL hold DONE $(date -u +%H:%M:%SZ)"

# --- the frozen stage-1 cells -----------------------------------------
run_cell w0 --seed 0 --num-seeds 100 --wrist-transform none
run_cell w1 --seed 0 --num-seeds 100 --wrist-transform blackout
run_cell w3 --seed 0 --num-seeds 100 --wrist-transform arm_blur
run_cell t1 --seed 0 --num-seeds 25 --wrist-transform none --top-transform blackout

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$OUT/STAGE1_DONE"
echo "[stage1] ALL CELLS DONE"
