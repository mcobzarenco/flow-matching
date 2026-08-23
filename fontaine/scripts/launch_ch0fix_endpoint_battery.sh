#!/usr/bin/env bash
# ch0fix endpoint battery — GPU legs (queue: carrier-hunt-rung3-exec;
# pre-reg posts/2026-08-22-prereg-clean-ch0-affine.md, grid
# frozen:  >=20 ch0-is-carrier / <=10 suspect-list-exhausted / 11-19 ambiguous).
#
# Protocol copied EXACTLY from launch_gripfix_endpoint_battery.sh
# (the democlean/onerig protocol; paired reads vs democlean 8/100 —
# THE read — / onerig 28/100 / control 11/100 are only valid
# protocol-matched); the ONLY deltas are the run name and output paths.
#
# Leg 1: sim100 flow leg on step_003000 — the v1/v2 sim100 protocol
#   (100 unseen seeds 0-99, episode 30 s, execute-horizon 30, euler-10,
#   bfloat16 decoder), --stats-repo-id grasp_demos_v2/merged per the
#   pre-reg's worn-row rule, --clutter-appearance standins (registered
#   substrate pin; demos + the 11/100 baseline are stand-ins-era).
# Leg 2: k4l2 panel_v2 leg — protocol copied EXACTLY from
#   eval_disc1000_k4l2_panel.sh (the pairing is only valid
#   protocol-matched): same plan, euler-10 single draw stable keying,
#   chunk-size 30, batch 32 / workers 20, npz prediction dump.
#
# The script WAITS for the endpoint first (step_003000 saved + train
# procs gone) so it can be armed before the run exits — no idle gap.
#
# Launch (systemd-run so the job survives session teardown):
#   systemd-run --user --unit=fontaine-ch0fix-endpoint-battery \
#     --setenv=HOME=/home/ubuntu \
#     --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
#     --working-directory=/home/ubuntu/flow-matching \
#     bash ~/flow-matching/fontaine/scripts/launch_ch0fix_endpoint_battery.sh
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching

RUN=grasp_sft_v2_joint_1gpu_pdnorm_ch0fix
CKPT="$HOME/checkpoints/finetune/$RUN/step_003000"
OUT=outputs/sim/grasp_sft/ch0fix_endpoint
NAME="eval__${RUN}__step_003000__panel_v2_k4l2_euler10_draws1_stable"

# Wait for the endpoint: final save on disk AND the train unit exited
# (train writes step_003000 then stops; procs-gone implies the save is
# complete). Bound the wait at 90 min — if the run stalls, exit loudly
# instead of squatting. pgrep pattern is the train launcher's module
# string; this script's own name does not contain it.
for i in $(seq 1 180); do
  if [ -d "$CKPT" ] && ! pgrep -f "bijou.train.*$RUN" >/dev/null; then
    break
  fi
  sleep 30
done
if [ ! -d "$CKPT" ] || pgrep -f "bijou.train.*$RUN" >/dev/null; then
  echo "ABORT: endpoint never landed (ckpt present: $([ -d "$CKPT" ] && echo yes || echo no); train procs: $(pgrep -fc "bijou.train.*$RUN" || true))" >&2
  exit 2
fi
sleep 15  # let the exiting procs release the GPU

# Owner policy-server guard (port 8144 claims the H100 silently).
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "ABORT: GPU busy — compute apps present:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 2
fi

echo "=== endpoint battery START $(date -u +%FT%TZ) — ckpt $CKPT ==="
mkdir -p "$OUT" reports

MUJOCO_GL=egl uv run python -m sim.rollout_sim \
    --checkpoint "$CKPT" \
    --method euler --sample-steps 10 \
    --seed 0 --num-seeds 100 --episode-seconds 30 --execute-horizon 30 \
    --flow-decoder-dtype bfloat16 \
    --clutter-appearance standins \
    --stats-repo-id grasp_demos_v2/merged \
    --out-dir "$OUT/flow_unseen" --out-json "$OUT/flow_unseen.json" \
    2>&1 | tee /home/ubuntu/eval__ch0fix_endpoint_sim100.log
echo "=== leg 1 sim100 DONE $(date -u +%FT%TZ) ==="

uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan plans/holdout_curated_v0_k4l2_panel_v2.json \
    --checkpoint "$CKPT" \
    --sample-draws 1 --sample-steps 10 --sample-method euler \
    --noise-key stable \
    --chunk-size 30 \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --output-json "reports/${NAME}.json" \
    --dump-predictions "reports/${NAME}.npz" \
    --report "reports/${NAME}.html" \
    2>&1 | tee "/home/ubuntu/${NAME}.log"
echo "=== leg 2 k4l2 panel DONE $(date -u +%FT%TZ) ==="
echo "=== endpoint battery DONE $(date -u +%FT%TZ) ==="
