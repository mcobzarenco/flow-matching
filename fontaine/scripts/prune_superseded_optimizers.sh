#!/usr/bin/env bash
# Keep-latest-optimizer pruner for a LIVE bijou.train run (democlean disk fix, 2026-08-20).
# Constraint this encodes: 6 x 44G full checkpoints (32G optimizer.pt each) do not fit
# in the free space left on /; only the newest optimizer.pt is ever a resume target,
# so every superseded one is deleted once a NEWER one is write-complete (mtime > 5 min).
# Weights (*.safetensors) are never touched. Exits when step_003000's optimizer is the
# survivor, or after ~16 h as a backstop.
set -u
CKPT_DIR="${1:-/home/ubuntu/checkpoints/finetune/grasp_sft_v2_joint_1gpu_pdnorm_democlean}"
FINAL_STEP="${2:-step_003000}"
LOG="${3:-/home/ubuntu/flow-matching/outputs/logs/$(basename "$CKPT_DIR")_ckpt_prune.log}"
mkdir -p "$(dirname "$LOG")"

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "pruner start dir=$CKPT_DIR final=$FINAL_STEP"

for _ in $(seq 1 96); do
  # step dirs that have an optimizer.pt, numerically sorted
  mapfile -t opts < <(ls -d "$CKPT_DIR"/step_*/optimizer.pt 2>/dev/null | sort -t_ -k2 -n)
  if (( ${#opts[@]} >= 2 )); then
    latest="${opts[-1]}"
    # write-complete heuristic: newest optimizer untouched for >5 min
    if [[ -z $(find "$latest" -mmin -5 2>/dev/null) ]]; then
      for f in "${opts[@]:0:${#opts[@]}-1}"; do
        sz=$(du -sh "$f" | cut -f1)
        rm -f "$f" && log "pruned $f ($sz) — superseded by $latest"
      done
    else
      log "latest $latest still fresh (<5min), holding"
    fi
  fi
  if [[ -f "$CKPT_DIR/$FINAL_STEP/optimizer.pt" ]] && \
     [[ -z $(find "$CKPT_DIR/$FINAL_STEP/optimizer.pt" -mmin -5 2>/dev/null) ]]; then
    mapfile -t opts < <(ls -d "$CKPT_DIR"/step_*/optimizer.pt 2>/dev/null)
    if (( ${#opts[@]} == 1 )); then log "final optimizer sole survivor; exit"; exit 0; fi
  fi
  sleep 600
done
log "16h backstop reached; exit"
