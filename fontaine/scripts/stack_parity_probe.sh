#!/usr/bin/env bash
# Stack-parity probe — Amendment 1's pre-registered disambiguator
# (posts/2026-08-17-prereg-sft-drift-discriminator.md, amendment §:
# "stack-parity probe of the saved ckpts on the pre-merge surface").
#
# WHAT: re-evaluate the discriminator's saved checkpoints (step 500 /
# step 1000) with the PRE-MERGE eval stack — the last fontaine commit
# before the family-norm merge (9094e60, d3dd4d0^1) — so the chunk-MAE
# lands in the same instrument units every 8x comparator's in-train
# probe reported (merged-table/checkpoint-stats surface; the train CLI
# documents the probe draw as exactly what `bijou.eval --episodes
# holdout --seed <eval-seed>` scores on the same data+split).
#
# WHY IT'S VALID: the checkpoint format did not move across the merge
# (schema_version 2; `git diff 9094e60 HEAD -- bijou/checkpoint.py
# bijou/vla.py bijou/validate_checkpoint.py` is empty) and the run was
# launched WITHOUT --per-dataset-flow-norm (metadata records
# per_dataset_flow_norm=false), so no post-merge section tag exists
# for the old loader to reject. Old-stack eval normalizes molmo-lineage
# checkpoints via the checkpoint's recorded table (MolmoNorm.CHECKPOINT
# -> metadata.stats — the pooled recomputed, source-oriented table),
# i.e. the comparator-era surface, NOT the honest per-item stats the
# post-merge in-train probe uses. That delta IS the instrument shift
# Amendment 1 wants isolated.
#
# Probe-match pins (frozen; every one mirrors the discriminator launch):
#   --data ~/datasets/fontaine/grasp_demos_v2/merged   (same local snapshot)
#   --episodes holdout --holdout-episodes 0.1 --split-seed 0
#   --num-samples 256 --seed 0        (= --eval-samples 256, eval-seed 0)
#   --chunk-size 30                   (spec.chunk_size; eval default is 50)
#   --sample-method euler --sample-steps 10  (the in-train probe's recorded
#                                             molmoact2 flow operating point)
#   --batch-size 12                   (the OOM-fixed micro size — the old
#                                      stack predates the probe-batch fix,
#                                      so the fix is applied HERE, by flag)
#
# Usage:
#   ./stack_parity_probe.sh prepare        # worktree + env + CPU dry-run
#                                          #   (metadata load on old stack;
#                                          #   no GPU, no weights)
#   ./stack_parity_probe.sh run step_000500
#   ./stack_parity_probe.sh run step_001000
#     -> reports/stack_parity/<step>.json  (~minutes each; GPU must be
#        free — runs only AFTER the discriminator reaches 1000)
#
# Comparator anchors for the read (demosonly 8x in-train probe, old
# units): 3.4623@250, 3.2397@500, 4.22@750, 5.27@1000 => their
# delta(1000-500) = +2.03; the frozen drift_min +1.0158 is half of it.
# Interpretation belongs to the verdict post per Amendment 1 — this
# script only produces numbers.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PREMERGE_SHA=9094e60
WORKTREE=/home/ubuntu/flow-matching-premerge
MAIN=/home/ubuntu/flow-matching
CKPT_ROOT="$HOME/checkpoints/finetune/grasp_sft_v2_demosonly_1gpu_disc"
OUT="$MAIN/reports/stack_parity"

MODE="${1:?usage: stack_parity_probe.sh prepare | run <step_dir>}"

if [ "$MODE" = "prepare" ]; then
  if [ ! -d "$WORKTREE" ]; then
    (cd "$MAIN" && git worktree add --detach "$WORKTREE" "$PREMERGE_SHA")
  fi
  cd "$WORKTREE"
  # Env sync (own .venv, wheels come from the shared uv cache) + import
  # check. (NOT --help: the old stack has a latent argparse bug — an
  # unescaped % in one help string breaks help FORMATTING only; running
  # the CLI never formats help.)
  uv run python -c "import bijou.eval.cli"
  # CPU dry-run: the old stack must parse the NEW save's metadata.
  uv run python - "$CKPT_ROOT/step_000500" << 'EOF'
import json, sys
from pathlib import Path
from bijou.checkpoint import VLAMetadata

ckpt = Path(sys.argv[1])
meta = VLAMetadata.from_json_dict(json.loads((ckpt / "metadata.json").read_text()))
assert meta.per_dataset_stats, "no per-dataset stats recorded"
print(
    f"DRY-RUN OK: old stack parsed {ckpt.name} — family={meta.family.value} "
    f"chunk={meta.chunk_size} step={meta.step} "
    f"per_dataset_flow_norm={meta.train_args.get('per_dataset_flow_norm')}"
)
print(f"  checkpoint table action q01[1]={meta.stats.state_dict()['action']['q01'][1]:.2f} "
      f"q99[1]={meta.stats.state_dict()['action']['q99'][1]:.2f} (source-oriented pair)")
EOF
  echo "=== PREPARE DONE: worktree @ $PREMERGE_SHA ready, metadata loads on the old stack ==="
  exit 0
fi

[ "$MODE" = "run" ] || { echo "mode must be prepare or run"; exit 2; }
STEP="${2:?run needs a step dir, e.g. step_000500}"
CKPT="$CKPT_ROOT/$STEP"
[ -d "$CKPT" ] || { echo "missing checkpoint $CKPT"; exit 1; }
[ -d "$WORKTREE" ] || { echo "worktree missing — run prepare first"; exit 1; }

# Owner policy-server guard (never preempt; port 8144 claims the H100
# silently) — and the discriminator itself must have finished.
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "ABORT: GPU busy — compute apps present:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 2
fi

mkdir -p "$OUT"
cd "$WORKTREE"
uv run python -m bijou.eval \
    --data ~/datasets/fontaine/grasp_demos_v2/merged \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --checkpoint "$CKPT" \
    --num-samples 256 --seed 0 \
    --chunk-size 30 \
    --sample-method euler --sample-steps 10 \
    --batch-size 12 --num-workers 4 \
    --output-json "$OUT/${STEP}.json" \
    2>&1 | tee "/home/ubuntu/eval__stack_parity_${STEP}.log"
echo "=== PARITY READ BANKED: $OUT/${STEP}.json (old-stack units) ==="
