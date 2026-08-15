#!/usr/bin/env bash
# Grasp-SFT STAGE D — sim100 eval of the stage-C endpoint(s)
# (pre-reg 2026-08-14-prereg-grasp-sft-bootstrap.md §2/§6 frozen).
# Launches ONLY after stage C lands (frozen ladder). ~1-1.5 GPU-h class.
#
# Usage:
#   ./launch_local_grasp_sft_staged_eval.sh convert <step>   # e.g. 3000
#   ./launch_local_grasp_sft_staged_eval.sh eval100 [flow]
#
# convert — two-hop conversion, the rig-r1 runbook + eval20 precedent:
#   olmo checkpoint -> HF serve dir (their tool; carries the DEMO-SET
#   recomputed norm_stats under tag so100_so101_molmoact2) -> bijou
#   checkpoint (our convert CLI; merged stats = the demo table, which
#   rollout_sim falls back to for molmoact2-lineage checkpoints — the
#   §6 seam: the endpoint is consumed through the SAME recomputed-table
#   rig-identity frame it trained in, no shim anywhere).
# eval100 — sequential rollout_sim (the only registered driver; the
#   08-12 parallel oracle FAILED and froze sequential), frozen seeds
#   0-99, v3 default frames, videos on, euler-10 for the AR-primary
#   converted checkpoint (eval20 precedent), euler-1 for the optional
#   flow arm (snapflow 1-NFE lineage, phase-2 precedent);
#   --episode-seconds 30 (the eval20 lesson: fixed --replans quietly
#   scales the time budget with chunk length; 30 s for ANY chunk size).
#   Reads: grasp_sft_staged_reads.py -> the frozen §2 decision surface
#   (>=20/100 GRPO_GO / 5-19 ITERATE_BC_ONCE / <5 F_TRANSFER).
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
cd /home/ubuntu/flow-matching

MODE="${1:?usage: launch_local_grasp_sft_staged_eval.sh convert <step> | eval100 [flow]}"
STEP="${2:-3000}"
EXP=/home/ubuntu/molmoact2/experiments
HF_DIR=~/checkpoints/molmoact2-grasp-sft-stagec-ar-step${STEP}-hf
CONVERTED=~/checkpoints/converted/molmoact2_grasp_sft_stagec_ar_step${STEP}
FLOW_CKPT=outputs/train/fontaine_grasp_sft_stagec_flow_4k/step_004000
OUT=outputs/sim/grasp_sft/stageD

if [ "$MODE" = "convert" ]; then
  (cd "$EXP" && ../.venv/bin/python -m olmo.hf_model.convert_molmoact2_to_hf \
      "checkpoints/finetune/fontaine_grasp_sft_stagec_ar/step${STEP}" "$HF_DIR")
  .venv/bin/python -m bijou.convert_molmoact2 \
      --source "$HF_DIR" \
      --out "$CONVERTED" \
      --norm-tag so100_so101_molmoact2
  echo "=== CONVERTED: $CONVERTED (verify read_checkpoint_info before eval) ==="
  exit 0
fi

[ "$MODE" = "eval100" ] || { echo "mode must be convert or eval100"; exit 2; }
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi
mkdir -p "$OUT"

if [ "${2:-}" = "flow" ]; then
  [ -d "$FLOW_CKPT" ] || { echo "flow endpoint missing ($FLOW_CKPT)"; exit 1; }
  MUJOCO_GL=egl uv run python -m sim.rollout_sim \
      --checkpoint "$FLOW_CKPT" \
      --method euler --sample-steps 1 \
      --seed 0 --num-seeds 100 --episode-seconds 30 --execute-horizon 30 \
      --expert-dtype bfloat16 \
      --out-dir "$OUT/flow" --out-json "$OUT/flow.json" \
      2>&1 | tee /home/ubuntu/eval__grasp_sft_stageD_flow.log
  uv run python fontaine/scripts/grasp_sft_staged_reads.py \
      --ar-json "$OUT/ar.json" --flow-json "$OUT/flow.json" \
      --out reports/analysis__grasp_sft_stageD_sim100.json
else
  [ -d "$CONVERTED" ] || { echo "converted checkpoint missing ($CONVERTED) — run convert first"; exit 1; }
  MUJOCO_GL=egl uv run python -m sim.rollout_sim \
      --checkpoint "$CONVERTED" \
      --method euler --sample-steps 10 \
      --seed 0 --num-seeds 100 --episode-seconds 30 --execute-horizon 30 \
      --expert-dtype bfloat16 \
      --out-dir "$OUT/ar" --out-json "$OUT/ar.json" \
      2>&1 | tee /home/ubuntu/eval__grasp_sft_stageD_ar.log
  uv run python fontaine/scripts/grasp_sft_staged_reads.py \
      --ar-json "$OUT/ar.json" \
      --out reports/analysis__grasp_sft_stageD_sim100.json
fi
echo "=== STAGE-D ARM DONE (analysis banked; boundary post owns the verdict) ==="
