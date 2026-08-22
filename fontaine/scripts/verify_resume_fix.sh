#!/usr/bin/env bash
# bijou-resume-flow-state-bug — GPU verification leg (fix 665dadb7).
# Resumes the archived attempt1 step_000250 (the poisoned-resume repro
# substrate) with the FIXED code and reads the first logged flow loss:
#   healthy continuation ~0.09-0.15 at step 260  -> fix VERIFIED (exit 0)
#   fresh-init ~1.4                              -> fix REFUTED (exit 5)
# The run is killed right after the verdict row — the verdict is the
# loss value, not the checkpoint; nothing writes into the archive
# (--save-dir /tmp, --resume reads only). Fresh seed 2 per the enforced
# convention (attempt1 trained seed 0; r2/r3 rode seed 1).
# --insulate-flow/--joint-ce-weight are NOT declared: the fixed args
# REFUSE them under --resume and the cli reconstructs the recorded
# payload (ce_weight=1, insulate_flow=True) — the unit log carries the
# reconstruction print as the sub-fix receipt.
#
# Run at a GPU-free window only (needs ~62 GiB; leg B/leg C own the
# card otherwise). Timebox: 20 min, then abort + leave the item open.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
cd /home/ubuntu/flow-matching

if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "ABORT: GPU busy — compute apps present:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 2
fi

CKPT=$HOME/checkpoints/finetune/grasp_sft_v2_squint_adapt_onerig_attempt1/step_000250
OUT=/tmp/squint_resume_verify
rm -rf "$OUT"; mkdir -p "$OUT"
UNIT=fontaine-resume-verify

systemd-run --user --unit=$UNIT \
  --setenv=HOME=/home/ubuntu \
  --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
  --setenv=MALLOC_ARENA_MAX=2 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --working-directory=/home/ubuntu/flow-matching \
  .venv/bin/python -m bijou.train \
  --train-data ~/datasets/fontaine/squint_twin_demos_v1 \
  --resume "$CKPT" \
  --seed 2 \
  --image-augment 0.8 \
  --decoder-lr 5e-5 --backbone-text-lr 1e-5 \
  --steps 270 --batch-size 96 --backward-chunks 8 \
  --activation-checkpointing --offload-optim \
  --holdout-episodes 0.1 --eval-every 1000 --eval-samples 256 \
  --save-every 100000 \
  --save-dir "$OUT"

echo "verify unit launched $(date -u +%FT%TZ); polling $OUT/train_log.jsonl for the step-260 row (20 min timebox)"
t0=$(date +%s)
verdict=""
while true; do
  if [ -f "$OUT/train_log.jsonl" ]; then
    row=$(grep '"step": 260' "$OUT/train_log.jsonl" | head -1 || true)
    if [ -n "$row" ]; then echo "ROW: $row"; verdict="$row"; break; fi
  fi
  if ! systemctl --user is-active --quiet $UNIT; then
    echo "unit exited before a verdict row — inspect journalctl --user -u $UNIT" >&2
    break
  fi
  if [ $(( $(date +%s) - t0 )) -gt 1200 ]; then echo "TIMEBOX hit" >&2; break; fi
  sleep 15
done
systemctl --user stop $UNIT 2>/dev/null || true

if [ -z "$verdict" ]; then exit 3; fi
flow=$(echo "$verdict" | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['loss_action_flow'])")
echo "flow loss @260 = $flow (attempt1 continuous 0.0946@250; poisoned resume 1.4374@260)"
ok=$(echo "$flow" | .venv/bin/python -c "import sys; print(int(float(sys.stdin.read()) < 0.5))")
if [ "$ok" = "1" ]; then echo "VERDICT: continuation HEALTHY — fix VERIFIED"; exit 0; fi
echo "VERDICT: flow head still resets — fix REFUTED"; exit 5
