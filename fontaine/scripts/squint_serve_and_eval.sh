#!/usr/bin/env bash
# Serve one checkpoint + run the twin eval client over the given tasks
# (queue item squint-gate2-harness; pre-reg
# posts/2026-08-22-prereg-squint-twin-screen.md). One arm per
# invocation; compose arms sequentially in a leg script.
#
#   bash squint_serve_and_eval.sh <checkpoint-dir> <arm-name> \
#     <stats-repo-id> <task> [<task>...]
#
# Server: bijou.policy_server, main venv, GPU, port 8145 — the owner's
# rig policy-server owns 8144 and is never touched. The server is
# started here, /spec-waited (file-existence-style wait: curl poll, no
# pgrep — the self-match incident class), and ALWAYS stopped on exit.
# Client: twin venv, physx_cpu env, raw wire protocol; rows land in
# outputs/squint_screen/eval/<arm>_<task>.json.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin

CKPT="$1"; ARM="$2"; STATS_REPO="$3"; shift 3
TASKS=("$@")
PORT="${SQUINT_EVAL_PORT:-8145}"
NUM_SEEDS="${NUM_SEEDS:-100}"
LOG=/home/ubuntu/flow-matching/outputs/squint_screen/eval/server_${ARM}.log
mkdir -p /home/ubuntu/flow-matching/outputs/squint_screen/eval

if curl -sf "http://127.0.0.1:${PORT}/spec" >/dev/null 2>&1; then
  echo "ABORT: port ${PORT} already serving — another eval leg live?" >&2
  exit 2
fi

cd /home/ubuntu/flow-matching
uv run python -m bijou.policy_server \
  --checkpoint "$CKPT" --device cuda \
  --flow-decoder-dtype bfloat16 --port "$PORT" \
  >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${PORT}/spec" >/dev/null 2>&1 && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ABORT: policy server died during load — tail:" >&2
    tail -20 "$LOG" >&2
    exit 3
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/spec" >/dev/null 2>&1 || {
  echo "ABORT: /spec never came up (15 min)" >&2; exit 3; }
echo "=== server up for $ARM ($CKPT) $(date -u +%FT%TZ)"

cd /home/ubuntu/squint
for task in "${TASKS[@]}"; do
  echo "=== eval $ARM/$task n=$NUM_SEEDS $(date -u +%FT%TZ)"
  PYTHONPATH=/home/ubuntu/squint .venv/bin/python \
    /home/ubuntu/flow-matching/fontaine/scripts/squint_twin_eval_client.py \
    --task "$task" --arm-name "$ARM" \
    --stats-repo-id "$STATS_REPO" \
    --server "http://127.0.0.1:${PORT}" \
    --num-seeds "$NUM_SEEDS"
done
echo "=== $ARM done $(date -u +%FT%TZ)"
