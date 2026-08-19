#!/usr/bin/env bash
# Route-C JOINT endpoint probes — amendment §4
# (posts/2026-08-16-amendment-grasp-sft-route-c-joint.md; both parents'
# eval protocols on the ONE joint checkpoint).
#
# Usage:
#   ./launch_local_grasp_sft_joint_probes.sh smoke            # 3-seed --serve-head ar smoke (REQUIRED first: the flag is code-reviewed but not yet GPU-fired)
#   ./launch_local_grasp_sft_joint_probes.sh flow-unseen      # euler-10, seeds 0-99      (~1.3 GPU-h)
#   ./launch_local_grasp_sft_joint_probes.sh flow-train       # euler-10, seeds 1000-1099 (~1.3 GPU-h)
#   ./launch_local_grasp_sft_joint_probes.sh token-unseen     # --serve-head ar greedy, seeds 0-99 (~1.3 GPU-h)
#   ./launch_local_grasp_sft_joint_probes.sh token-base       # base-token anchor leg on the CORRECTED BASE conversion (~1.3 GPU-h; B §3 default-run)
#
# Endpoint checkpoint: new-format VLA native (NO convert step — unlike
# stage D's two-hop). Serving seams (all verified in-code 08-16):
#  - flow legs: the recorded deployment path (euler-10 = the serving
#    metadata; --method euler --sample-steps 10 spelled anyway);
#  - token legs: --serve-head ar = dispatch-only greedy decode of the
#    trunk's discrete head on the SAME collated inputs the CE rider
#    trained on (policy name carries _arhead; grammar masking is
#    predict_ar's default decode);
#  - state/table: the checkpoint's baked table IS the corrected table
#    (init from molmoact2_base_corrected_stats_v0_vla) — rig-identity,
#    no shim, same as every B-D read.
# Reads: fontaine/scripts/grasp_sft_joint_probe_reads.py (oracle-
# tested, tests/test_joint_probe_reads.py) consumes all five jsons
# tolerant of missing legs — kept-split automatic, serve_head
# provenance guarded, A SS5 / B SS3 verdicts baked (the 29-31 clause
# overlap in A SS5 is SURFACED, the boundary post owns that call). Anchors: flow base 9 / corrupt-table-28; token R1-B
# floor context 2/20 (different protocol, record-only).
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
cd /home/ubuntu/flow-matching

MODE="${1:?usage: launch_local_grasp_sft_joint_probes.sh smoke|flow-unseen|flow-train|token-unseen|token-base}"
CKPT=~/checkpoints/finetune/fontaine_grasp_sft_joint_corrected/step_002000
BASE=~/checkpoints/converted/molmoact2_base_corrected_stats_v0_vla
OUT=outputs/sim/grasp_sft/joint_probes

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi
mkdir -p "$OUT"

case "$MODE" in
  smoke)
    # 3 seeds, token head, no videos needed — proves --serve-head ar
    # end-to-end (mount, dispatch, grammar decode, row write) before
    # any registered leg burns GPU-hours. Eyeball: policy name printed
    # must carry _arhead; rows must be well-formed.
    MUJOCO_GL=egl uv run python -m sim.rollout_sim \
        --checkpoint "$CKPT" \
        --serve-head ar \
        --seed 0 --num-seeds 3 --episode-seconds 30 --execute-horizon 30 \
        --flow-decoder-dtype bfloat16 \
        --out-dir "$OUT/smoke_arhead" --out-json "$OUT/smoke_arhead.json" \
        2>&1 | tee /home/ubuntu/eval__grasp_sft_joint_smoke_arhead.log
    ;;
  flow-unseen)
    MUJOCO_GL=egl uv run python -m sim.rollout_sim \
        --checkpoint "$CKPT" \
        --method euler --sample-steps 10 \
        --seed 0 --num-seeds 100 --episode-seconds 30 --execute-horizon 30 \
        --flow-decoder-dtype bfloat16 \
        --out-dir "$OUT/flow_unseen" --out-json "$OUT/flow_unseen.json" \
        2>&1 | tee /home/ubuntu/eval__grasp_sft_joint_flow_unseen.log
    ;;
  flow-train)
    MUJOCO_GL=egl uv run python -m sim.rollout_sim \
        --checkpoint "$CKPT" \
        --method euler --sample-steps 10 \
        --seed 1000 --num-seeds 100 --episode-seconds 30 --execute-horizon 30 \
        --flow-decoder-dtype bfloat16 \
        --out-dir "$OUT/flow_train" --out-json "$OUT/flow_train.json" \
        2>&1 | tee /home/ubuntu/eval__grasp_sft_joint_flow_train.log
    ;;
  token-unseen)
    # --clutter-appearance standins: this amendment's legs 1/2 (flow
    # 44/100 + 42/100, 08-16) ran pre-promotion; the 08-18 'patched'
    # default would change the substrate mid-registration. Same pin
    # rationale as pdnorm Amendment 1.
    MUJOCO_GL=egl uv run python -m sim.rollout_sim \
        --checkpoint "$CKPT" \
        --serve-head ar \
        --seed 0 --num-seeds 100 --episode-seconds 30 --execute-horizon 30 \
        --flow-decoder-dtype bfloat16 \
        --clutter-appearance standins \
        --out-dir "$OUT/token_unseen" --out-json "$OUT/token_unseen.json" \
        2>&1 | tee /home/ubuntu/eval__grasp_sft_joint_token_unseen.log
    ;;
  token-base)
    MUJOCO_GL=egl uv run python -m sim.rollout_sim \
        --checkpoint "$BASE" \
        --serve-head ar \
        --seed 0 --num-seeds 100 --episode-seconds 30 --execute-horizon 30 \
        --flow-decoder-dtype bfloat16 \
        --clutter-appearance standins \
        --out-dir "$OUT/token_base" --out-json "$OUT/token_base.json" \
        2>&1 | tee /home/ubuntu/eval__grasp_sft_joint_token_base.log
    ;;
  *)
    echo "unknown mode $MODE"; exit 2;;
esac
echo "=== JOINT PROBE LEG '$MODE' DONE (boundary post owns the reads) ==="
