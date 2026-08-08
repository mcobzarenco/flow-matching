#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
#
# fontaine — noise-ladder rung 2 SEATING ARM (#1, pre-reg
#   2026-08-08-prereg-noise-ladder-perdataset.md "Folded-in arm"):
#   re-run the banked random-noise mean-of-10 config WITH
#   --dump-predictions retained, enabling the paired per-frame read
#   the R3 board-row seating requires (mean-of-top-10-tickets 5.1847
#   vs mean-of-random-10, paired CI95). Independent of stages 0–2;
#   same GPU window.
# KEYING (load-bearing): the banked 5.3645/1.4242 row
#   (…panel_curated_v0_k4l2_draws10_heun30.json) is dated 2026-08-05 —
#   BEFORE --noise-key existed (no noise_key field in the json), i.e.
#   the historical INDEX-keyed path. --noise-key index reproduces that
#   path bit-for-bit at frozen corpus composition (draw 0 =
#   sample_noise(seed + index); NOISE_KEYS docstring) — the pre-reg's
#   base-equality oracle (pooled 5.3645 at 4dp) depends on it. A
#   distinct _seating stem keeps this re-run from overwriting the
#   banked report json.
# Cost: ≈ 3.0 GPU-h (the stage-3 cost) of the ≤ 4 GPU-h remaining.
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a /home/ubuntu/eval__noiseladder_seating.log) 2>&1

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_flow_artrunk_h1024_40k_ddp2
CKPT=outputs/train/${RUN}/step_080000
PLAN=plans/holdout_curated_v0_k4l2.json

[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
SHAS

name="eval__${RUN}__step_080000__panel_curated_v0_k4l2_draws10_seating_heun30"
echo "=== noise-ladder seating arm: random-noise draws-10, dumps retained ($name) ==="
uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan "$PLAN" \
    --checkpoint "$CKPT" \
    --sample-steps 30 --sample-method heun \
    --sample-draws 10 --noise-key index \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${name}.json" \
    --dump-predictions "reports/${name}.npz"

# Base-equality oracle (pre-reg, frozen): the re-run's pooled core
# chunk/first MAE must reproduce the banked 5.3645/1.4242 at 4dp —
# else the banked config was NOT reproduced and the seating read must
# not run (abort loud, never re-tolerance).
uv run python - <<'EOF'
import json
report = json.load(open(
    "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__"
    "panel_curated_v0_k4l2_draws10_seating_heun30.json",
))
rows = [s for s in report["summaries"] if s["policy"].startswith("bijou@")]
assert len(rows) == 1, [s["policy"] for s in report["summaries"]]
chunk, first = round(rows[0]["chunk_mae"], 4), round(rows[0]["first_mae"], 4)
assert chunk == 5.3645 and first == 1.4242, (
    f"base-equality FAILED: {chunk}/{first} != banked 5.3645/1.4242 — "
    "the banked config was not reproduced; seating read must not run"
)
print(f"base-equality oracle GREEN: {chunk}/{first} == banked")
EOF

echo "=== NOISE-LADDER SEATING ARM DONE (rc=0) — paired R3 seating read runs offline ==="
