#!/usr/bin/env bash
# sim100 flow leg for grasp_sft_v2_demosonly_1gpu_disc/step_001000 —
# the pdnorm pre-reg's BASELINE grasp read (queue:
# disc-step1000-sim100-baseline; un-gated — eval of a banked
# checkpoint, runnable independent of the GO).
#
# Protocol = the v1/v2 sim100 flow leg verbatim
# (launch_local_grasp_sft_joint_probes.sh flow-unseen): 100 UNSEEN
# seeds 0-99 (demos were generated from seeds 1000+), episode 30 s,
# execute-horizon 30, euler-10, bfloat16 decoder. Worn-row lookup left
# at DEFAULT on purpose: the demosonly checkpoint's stats carry no rig
# key, so the lookup falls back to the merged demos-native table — the
# correct window for this checkpoint (record stats_repo_id from the
# out-json). Anchors: probe joint_corrected@2000 44/100; broken class
# ~5/100. Grid (pdnorm pre-reg): fills the demosonly-v2 grasp cell.
#
# Launch (systemd-run so the job survives session teardown):
#   systemd-run --user --unit=fontaine-disc1000-sim100 \
#     --setenv=HOME=/home/ubuntu \
#     --setenv=PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin \
#     --working-directory=/home/ubuntu/flow-matching \
#     bash ~/flow-matching/fontaine/scripts/launch_disc1000_sim100_baseline.sh
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching

# Owner policy-server guard (port 8144 claims the H100 silently).
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "ABORT: GPU busy — compute apps present:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 2
fi

CKPT=~/checkpoints/finetune/grasp_sft_v2_demosonly_1gpu_disc/step_001000
OUT=outputs/sim/grasp_sft/disc1000_baseline
mkdir -p "$OUT"

MUJOCO_GL=egl uv run python -m sim.rollout_sim \
    --checkpoint "$CKPT" \
    --method euler --sample-steps 10 \
    --seed 0 --num-seeds 100 --episode-seconds 30 --execute-horizon 30 \
    --flow-decoder-dtype bfloat16 \
    --out-dir "$OUT/flow_unseen" --out-json "$OUT/flow_unseen.json" \
    2>&1 | tee /home/ubuntu/eval__disc1000_sim100_flow_unseen.log
echo "=== disc-1000 sim100 baseline DONE $(date -u +%FT%TZ) ==="
