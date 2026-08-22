#!/usr/bin/env bash
# Squint qualification screen — GPU leg A (exec item
# squint-twin-screen-exec part (c), pre-reg
# posts/2026-08-22-prereg-squint-twin-screen.md, finalization amendment
# posted in-channel before launch). One chained unit:
#
#   1. SAC expert LiftCube  (train_squint paper recipe, DR OFF, seed 1)
#   2. SAC expert PlaceCube (identical flags)
#   3. collect lift  — expert rollouts -> dual-camera 224 re-render
#   4. collect place — same
#   5. twin episodes -> LeRobot squint_twin_demos_v1 + conversion oracle
#      (main venv)
#
# Priced ~0.5 GPU-h experts + ~0.3 collection inside the <=7 GPU-h cell
# gate. Timeouts are hard caps, not expectations (paper claims 2-9
# min/task on a 3090). Adaptation (leg B) launches separately after the
# oracle receipt is verified.
#
# Launch (announce in-channel first):
#   systemd-run --user --unit=fontaine-squint-leg-a \
#     --setenv=HOME=/home/ubuntu \
#     --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
#     --working-directory=/home/ubuntu/squint \
#     bash ~/flow-matching/fontaine/scripts/launch_squint_leg_a.sh
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
cd /home/ubuntu/squint
V=/home/ubuntu/squint/.venv/bin/python
LOG=/home/ubuntu/flow-matching/outputs/squint_screen/leg_a.log
mkdir -p /home/ubuntu/flow-matching/outputs/squint_screen
exec > >(tee -a "$LOG") 2>&1

if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "ABORT: GPU busy (owner policy-server?) — compute apps present:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 2
fi

echo "=== leg A start $(date -u +%FT%TZ)"

for task in lift place; do
  if [ "$task" = lift ]; then envid=SO101LiftCube-v1; else envid=SO101PlaceCube-v1; fi
  echo "=== expert $task ($envid) $(date -u +%FT%TZ)"
  timeout 2700 "$V" train_squint.py \
    --env-id "$envid" --exp-name "squint_expert_$task" --seed 1 \
    --no-track --no-capture-video --no-env-domain-randomization
done

for task in lift place; do
  echo "=== collect $task $(date -u +%FT%TZ)"
  timeout 2700 env PYTHONPATH=/home/ubuntu/squint "$V" \
    /home/ubuntu/flow-matching/fontaine/scripts/squint_expert_collect.py \
    --task "$task" --stage all --ckpt "runs/squint_expert_$task/ckpt.pt"
done

echo "=== convert + oracle $(date -u +%FT%TZ)"
cd /home/ubuntu/flow-matching
timeout 1800 uv run python fontaine/scripts/squint_to_lerobot.py

echo "=== leg A done $(date -u +%FT%TZ)"
