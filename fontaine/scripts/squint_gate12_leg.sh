#!/usr/bin/env bash
# Squint qualification screen — GPU leg C: band pilot -> Gate-1 ->
# Gate-2 paired cells + record-only riders -> pre-registered reads
# (queue item squint-gate2-harness; frozen slots = pre-reg
# posts/2026-08-22-prereg-squint-twin-screen.md + finalization
# amendment). One detached unit; serves one arm at a time on 8145
# (owner rig server owns 8144, never touched).
#
# Flow:
#  A. adapt_onerig served once: pilot n=20 (seeds 0-19) both tasks ->
#     20-80% band verdicts banked; both tasks then completed to n=100
#     (seeds 20-99; the control runs at treatment n — pilot rows are a
#     reusable prefix of the paired 0-99 cell, merged by seed).
#  B. Gate-1 read: best task >= 20/100 or F-instrument (exit 5, screen
#     closes, Gate-2 GPU spend skipped).
#  C. Gate-2 task set = the in-band tasks. Neither in-band -> exit 4:
#     substitution ladder (Reach easier / Stack harder) is logged, a
#     substituted task needs new demos + adaptation = a new
#     pre-registered leg, never auto-run.
#  D. adapt_democlean n=100 on the Gate-2 tasks; then the unadapted
#     @3000 pair as record-only riders (sim100 worn row).
#  E. squint_screen_read.py read per task -> reports/ + the CDF panel.
#
# Launch (GPU free window only; leg B must be done):
#   systemd-run --user --unit=fontaine-squint-gate12 \
#     --setenv=HOME=/home/ubuntu \
#     --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
#     --working-directory=/home/ubuntu/flow-matching \
#     bash ~/flow-matching/fontaine/scripts/squint_gate12_leg.sh
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin

FM=/home/ubuntu/flow-matching
EVAL_DIR=$FM/outputs/squint_screen/eval
PORT="${SQUINT_EVAL_PORT:-8145}"
CKPT_ROOT=/home/ubuntu/checkpoints/finetune
ADAPT_STEP="${ADAPT_STEP:-step_000500}"
TWIN_STATS="${TWIN_STATS:-fontaine/squint_twin_demos_v1}"
RIG_STATS="${RIG_STATS:-grasp_demos_v2/merged}"
PILOT_N="${PILOT_N:-20}"
TASKS=(lift place)
READ="uv run python -m fontaine.scripts.squint_screen_read"
mkdir -p "$EVAL_DIR" "$FM/reports"

# ---- guards: leg B done, GPU free of foreign compute (8144 respected)
if systemctl --user is-active --quiet fontaine-squint-adapt; then
  echo "ABORT: leg B (fontaine-squint-adapt) still active" >&2; exit 2
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "ABORT: GPU busy (owner policy-server?) — compute apps present:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 2
fi
for arm in onerig democlean; do
  test -d "$CKPT_ROOT/grasp_sft_v2_squint_adapt_${arm}/$ADAPT_STEP" || {
    echo "ABORT: adapted ckpt missing for $arm ($ADAPT_STEP)" >&2; exit 2; }
done

cd "$FM"  # serve_start backgrounds bijou.policy_server from here

SERVER_PID=""
serve_start() {  # serve_start <checkpoint-dir> <label>
  local ckpt="$1" label="$2"
  local log="$EVAL_DIR/server_${label}.log"
  if curl -sf "http://127.0.0.1:${PORT}/spec" >/dev/null 2>&1; then
    echo "ABORT: port ${PORT} already serving" >&2; exit 2
  fi
  uv run python -m bijou.policy_server \
    --checkpoint "$ckpt" --device cuda \
    --flow-decoder-dtype bfloat16 --port "$PORT" \
    >"$log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${PORT}/spec" >/dev/null 2>&1 && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "ABORT: policy server died during load — tail:" >&2
      tail -20 "$log" >&2; exit 3
    fi
    sleep 5
  done
  curl -sf "http://127.0.0.1:${PORT}/spec" >/dev/null 2>&1 || {
    echo "ABORT: /spec never came up (15 min)" >&2; exit 3; }
  echo "=== server up: $label ($ckpt) $(date -u +%FT%TZ)"
}
serve_stop() {
  [ -n "$SERVER_PID" ] || return 0
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}
trap serve_stop EXIT

run_client() {  # run_client <arm-label> <task> <stats> <seed0> <n> <out-name>
  local arm="$1" task="$2" stats="$3" seed0="$4" n="$5" out="$6"
  echo "=== eval $arm/$task seeds $seed0..$((seed0 + n - 1)) $(date -u +%FT%TZ)"
  (cd /home/ubuntu/squint && PYTHONPATH=/home/ubuntu/squint .venv/bin/python \
    "$FM/fontaine/scripts/squint_twin_eval_client.py" \
    --task "$task" --arm-name "$arm" \
    --stats-repo-id "$stats" \
    --server "http://127.0.0.1:${PORT}" \
    --seed0 "$seed0" --num-seeds "$n" --out-name "$out")
}

# ---- phase A: adapted onerig — pilot, band, complete to n=100
serve_start "$CKPT_ROOT/grasp_sft_v2_squint_adapt_onerig/$ADAPT_STEP" adapt_onerig
declare -A BAND
for task in "${TASKS[@]}"; do
  run_client adapt_onerig "$task" "$TWIN_STATS" 0 "$PILOT_N" \
    "adapt_onerig_${task}_pilot${PILOT_N}"
  BAND[$task]=$($READ band \
    --json "$EVAL_DIR/adapt_onerig_${task}_pilot${PILOT_N}.json" \
    | tee -a "$EVAL_DIR/band_pilot.log" | tail -1)
  echo "=== band[$task] = ${BAND[$task]}"
done
for task in "${TASKS[@]}"; do
  run_client adapt_onerig "$task" "$TWIN_STATS" "$PILOT_N" $((100 - PILOT_N)) \
    "adapt_onerig_${task}_rest"
  $READ merge \
    --parts "$EVAL_DIR/adapt_onerig_${task}_pilot${PILOT_N}.json" \
            "$EVAL_DIR/adapt_onerig_${task}_rest.json" \
    --out "$EVAL_DIR/adapt_onerig_${task}.json"
done
serve_stop

# ---- phase B: Gate-1
GATE1=$($READ gate1 \
  --jsons "$EVAL_DIR/adapt_onerig_lift.json" "$EVAL_DIR/adapt_onerig_place.json" \
  | tee "$EVAL_DIR/gate1.log" | tail -1)
echo "=== Gate-1: $GATE1"
if [ "$GATE1" != "PASS" ]; then
  echo "=== F-INSTRUMENT: screen closes, no relative read attempted" >&2
  exit 5
fi

# ---- phase C: Gate-2 task set = in-band tasks
GATE2_TASKS=()
for task in "${TASKS[@]}"; do
  [ "${BAND[$task]}" = "IN_BAND" ] && GATE2_TASKS+=("$task")
done
if [ "${#GATE2_TASKS[@]}" -eq 0 ]; then
  echo "=== NO IN-BAND TASK: substitution ladder pends a new" >&2
  echo "=== pre-registered leg (new demos + adaptation); see band_pilot.log" >&2
  exit 4
fi
echo "=== Gate-2 tasks: ${GATE2_TASKS[*]}"

# ---- phase D: democlean cells + unadapted riders
serve_start "$CKPT_ROOT/grasp_sft_v2_squint_adapt_democlean/$ADAPT_STEP" adapt_democlean
for task in "${GATE2_TASKS[@]}"; do
  run_client adapt_democlean "$task" "$TWIN_STATS" 0 100 "adapt_democlean_${task}"
done
serve_stop
for arm in onerig democlean; do
  serve_start "$CKPT_ROOT/grasp_sft_v2_joint_1gpu_pdnorm_${arm}/step_003000" \
    "unadapt_${arm}"
  for task in "${GATE2_TASKS[@]}"; do
    run_client "unadapt_${arm}" "$task" "$RIG_STATS" 0 100 "unadapt_${arm}_${task}"
  done
  serve_stop
done

# ---- phase E: reads + CDF panels (no scalar summaries without them)
for task in "${GATE2_TASKS[@]}"; do
  $READ read \
    --a "$EVAL_DIR/adapt_onerig_${task}.json" \
    --b "$EVAL_DIR/adapt_democlean_${task}.json" \
    --rider-a "$EVAL_DIR/unadapt_onerig_${task}.json" \
    --rider-b "$EVAL_DIR/unadapt_democlean_${task}.json" \
    --out "$FM/reports/analysis__squint_gate2_${task}.json" \
    --charts-dir "$FM/fontaine/blog/src/img/squint"
done
echo "=== gate12 leg done $(date -u +%FT%TZ)"
