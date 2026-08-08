#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
# fontaine — molmo2 60k ACCURACY-BY-FIELD panel (owner 10:08Z 08-08).
# Pre-reg: fontaine/blog/src/posts/2026-08-08-prereg-accuracy-by-field.md
#   (record-only diagnostic; ~3.5 GPU-h projected, gate 6).
# What it runs: the chained 60k endpoint eval's EXACT command
#   (flags/plan/seed byte-identical) with `_fields` output stems. No
#   new CLI flags — post-2f4d575 the narrated pass (`bijou@60000
#   +fields`: decode holding/progress/event/visible/subgoal value
#   lines, then actions) rides automatically on aux-trained
#   checkpoints, molmo2 included; its generations produce the
#   report's accuracy-by-field block vs the weak judge labels.
# Ordering (pre-reg): runs strictly AFTER the chained endpoint eval
#   lands (that eval stays narrated-arm-free — box code frozen under
#   the live run per charter — and byte-comparable to the 40k panel)
#   and AFTER refresh_ctrl.sh brings the checkout to a commit
#   carrying 2f4d575 (guarded below by grepping the fixed gate).
# Validity oracle (pre-reg read 3, mechanized below): the fields
#   run's base bijou@60000 chunk MAE must equal the chained eval's to
#   full JSON precision (same instrument, same 4-rank sharding,
#   greedy). Disagreement = instrument finding, loud abort.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ubuntu/flow-matching
mkdir -p "$HOME/logs" reports

RUN=fontaine_molmo2_ar_60k_ddp4
STEP=060000
CKPT=outputs/train/$RUN/step_$STEP
PLAN=plans/holdout_curated_v0_k4l2.json
chained_json=reports/eval__${RUN}__step_${STEP}__panel_curated_v0_k4l2.json
name="eval__${RUN}__step_${STEP}__panel_curated_v0_k4l2_fields"

# Guards: fixed code, endpoint + chained eval landed, frozen plan,
# all 4 GPUs free (also proves the training run is over).
grep -q "isinstance(decoder, ARSuffixDecoder)" bijou/eval/policies.py \
    || { echo "checkout PRE-DATES 2f4d575 (narrated-pass fix) — refresh_ctrl.sh first; abort"; exit 1; }
[ -d "$CKPT" ] || { echo "no endpoint checkpoint $CKPT — abort"; exit 1; }
[ -s "$chained_json" ] || { echo "chained endpoint eval json missing ($chained_json) — it runs first; abort"; exit 1; }
sha256sum -c - <<'SHAS'
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
SHAS
for g in 0 1 2 3; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$mem" -gt 1024 ]; then echo "GPU $g busy (${mem} MiB) — abort"; exit 1; fi
done

.venv/bin/torchrun --standalone --nproc-per-node=4 -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan "$PLAN" \
    --checkpoint "$CKPT" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 32 \
    --dump-predictions "reports/${name}.npz" \
    --output-json "reports/${name}.json" \
    --report "reports/${name}.html" \
    2>&1 | tee "$HOME/logs/${name}.log"

# Post-run: the accuracy block must exist (the point of the run) and
# the base arm must reproduce the chained eval exactly (read 3).
.venv/bin/python - "$chained_json" "reports/${name}.json" << 'EOF'
import json, sys

chained, fields = (json.load(open(p)) for p in sys.argv[1:3])
acc = {k: fields.get(k) for k in (
    "holding_accuracy", "progress_mae", "event_accuracy", "visible_accuracy",
)}
missing = [k for k, v in acc.items() if v is None]
if missing:
    sys.exit(f"ABORT: accuracy block still empty ({missing}) — "
             "narrated pass did not ride; fix regressed?")
base = {s["policy"]: s["chunk_mae"] for s in fields["summaries"]}
anchor = {s["policy"]: s["chunk_mae"] for s in chained["summaries"]}
pol = f"bijou@{60000}"
if base[pol] != anchor[pol]:
    sys.exit(f"ABORT (read 3): base {pol} chunk MAE {base[pol]!r} != "
             f"chained {anchor[pol]!r} — instrument finding, do not read")
print(f"read 3 OK: base {pol} reproduces the chained eval exactly ({base[pol]})")
narr = next(v for k, v in base.items() if k.endswith("+fields"))
print(f"accuracy-by-field: {json.dumps(acc)}")
print(f"narration delta (record-only): +fields {narr:.4f} vs base "
      f"{base[pol]:.4f} ({narr - base[pol]:+.4f}; AR-100k anchor +0.054)")
EOF
echo "=== MOLMO2 60K FIELDS PANEL DONE (rc=0) ==="
