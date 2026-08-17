#!/usr/bin/env bash
# grasp_sft_v2_demosonly endpoint sim100 — 8xA100 box, seed-sharded.
# (queue item grasp-sft-v2-endpoint-boundary; pre-reg grid: flow >=
# probe band 44 => demos were the lever; flow ~5 => data not the
# lever, banked --per-dataset-flow-norm cell is next. Token leg is
# valid this time — the b779ba4 merged-table decode fix is deployed —
# read vs the >=20/100 bar.)
#
# Sharding is exact: every rollout stochastic stream derives from the
# (env seed, replan, draw) triple (sim_item docstring), so 4x25 seeds
# per leg reproduces the single-process leg bitwise. Merge with
# fontaine/scripts/merge_rollout_shards.py (tiling + config guards).
#
# AFTER the merge: rsync $OUT (jsons + video dirs) local BEFORE any
# box cleanup — the v1 merged artifacts were destroyed by a later
# session's outputs/ wipe while still awaiting their rsync (08-17).
#
# Usage (on the box):
#   bash eval_box_grasp_sft_v2_demosonly_sim100.sh smoke   # 2 seeds/leg on gpu0/gpu4, ~3 min
#   bash eval_box_grasp_sft_v2_demosonly_sim100.sh full    # flow gpu0-3, token gpu4-7
# Both legs' shards run concurrently; the script waits and reports
# per-shard exit codes (nonzero if any shard failed).
set -uo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
cd /home/ubuntu/flow-matching

MODE="${1:?usage: eval_box_grasp_sft_v2_demosonly_sim100.sh smoke|full}"
CKPT=~/checkpoints/finetune/grasp_sft_v2_demosonly_8xa100/step_003000
OUT=outputs/sim/grasp_sft/v2_endpoint
mkdir -p "$OUT"

run_shard() { # gpu leg seed n extra-args...
  local gpu="$1" leg="$2" seed="$3" n="$4"; shift 4
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl uv run python -m sim.rollout_sim \
      --checkpoint "$CKPT" "$@" \
      --seed "$seed" --num-seeds "$n" --episode-seconds 30 --execute-horizon 30 \
      --flow-decoder-dtype bfloat16 \
      --out-dir "$OUT/${leg}_s${seed}" --out-json "$OUT/${leg}_s${seed}.json" \
      > "/home/ubuntu/eval__sft_v2do_${leg}_s${seed}.log" 2>&1
}

FLOW_ARGS=(--method euler --sample-steps 10)
TOKEN_ARGS=(--serve-head ar)

pids=(); names=()
case "$MODE" in
  smoke)
    run_shard 0 flow_smoke 0 2 "${FLOW_ARGS[@]}" & pids+=($!); names+=(flow_smoke_s0)
    run_shard 4 token_smoke 0 2 "${TOKEN_ARGS[@]}" & pids+=($!); names+=(token_smoke_s0)
    ;;
  full)
    for i in 0 1 2 3; do
      run_shard "$i" flow_unseen $((i * 25)) 25 "${FLOW_ARGS[@]}" & pids+=($!); names+=("flow_unseen_s$((i * 25))")
    done
    for i in 0 1 2 3; do
      run_shard $((i + 4)) token_unseen $((i * 25)) 25 "${TOKEN_ARGS[@]}" & pids+=($!); names+=("token_unseen_s$((i * 25))")
    done
    ;;
  *)
    echo "unknown mode $MODE"; exit 2;;
esac

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "shard ${names[$i]}: OK"
  else
    echo "shard ${names[$i]}: FAILED (log /home/ubuntu/eval__sft_v2do_${names[$i]}.log)"
    rc=1
  fi
done
echo "=== sim100 $MODE done rc=$rc ==="
exit "$rc"
