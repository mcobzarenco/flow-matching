#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
#
# fontaine — #6 subgoal-swap CONTENT read (pre-reg
# 2026-08-09-prereg-subgoal-swap.md), sequential:
#   1. IDENTITY run, full k4l2 panel (~35 min): --subgoal-swap-identity
#      forces donor = self with ALL swap plumbing live; oracle (ii)
#      requires it to byte-reproduce the BANKED oracle arm npz — checked
#      abort-on-red before the swap arm may start. Never a read
#      (_swapidentity stem).
#   2. SWAP arm, full panel (~35 min): seed-0 within-dataset
#      derangement (_swapsubgoal stem); oracles (i)+(iv) checked
#      mechanically over the full dump at rc=0.
# Frozen reads (Δ_swap vs banked baseline, swap-vs-oracle, horizon
# mirror) run OFFLINE next session — this unit only banks artifacts.
# Cost: ~2.4 GPU-h total (2 × the banked oracle-arm rate) <= 3 gate.
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a /home/ubuntu/eval__ar100k_subgoal_swap_arm.log) 2>&1

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_arb_rcond_100k_ddp4
CKPT=/home/ubuntu/checkpoints/bijou-checkpoints/${RUN}/step_100000
PLAN_FULL=plans/holdout_curated_v0_k4l2.json
DATA_ROOT=/home/ubuntu/datasets/mcobzarenco/community_curated_v0
BANKED_ORACLE=reports/eval__${RUN}__step_100000__panel_k4l2_oraclesubgoal.npz

[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
[ -f "$BANKED_ORACLE" ] || { echo "no banked oracle npz — abort"; exit 1; }
sha256sum -c - <<'SHAS'
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
SHAS

idn="eval__${RUN}__step_100000__panel_k4l2_swapidentity"
swp="eval__${RUN}__step_100000__panel_k4l2_swapsubgoal"
for stem in "$idn" "$swp"; do
    if [ -e "reports/${stem}.npz" ]; then
        echo "stem collision: reports/${stem}.npz exists — abort"; exit 1
    fi
done

echo "=== live-oracle selftest (abort-on-red) ==="
uv run python fontaine/scripts/subgoal_swap_live_oracles.py --selftest

DATA_ARGS=(
    --data "$DATA_ROOT"
    --episodes holdout --holdout-episodes 0.1 --split-seed 0
    --fps 30 --camera-counts 1 2
)

echo "=== phase 1/4: IDENTITY run, full panel ($idn) ==="
uv run python -m bijou.eval \
    "${DATA_ARGS[@]}" \
    --sample-plan "$PLAN_FULL" \
    --checkpoint "$CKPT" \
    --subgoal-mode oracle \
    --subgoal-swap-seed 0 --subgoal-swap-identity \
    --dump-subgoal-swaps "reports/${idn}_swaps.json" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${idn}.json" \
    --dump-predictions "reports/${idn}.npz" \
    --report "reports/${idn}.html"

echo "=== phase 2/4: oracle (ii) — identity byte-reproduction ==="
uv run python fontaine/scripts/subgoal_swap_live_oracles.py \
    --mode identity \
    --identity-npz "reports/${idn}.npz" \
    --oracle-npz "$BANKED_ORACLE"

echo "=== phase 3/4: SWAP arm, full panel ($swp) ==="
uv run python -m bijou.eval \
    "${DATA_ARGS[@]}" \
    --sample-plan "$PLAN_FULL" \
    --checkpoint "$CKPT" \
    --subgoal-mode oracle \
    --subgoal-swap-seed 0 \
    --dump-subgoal-swaps "reports/${swp}_swaps.json" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${swp}.json" \
    --dump-predictions "reports/${swp}.npz" \
    --report "reports/${swp}.html"

echo "=== phase 4/4: oracles (i)+(iv) — mechanical dump check ==="
uv run python fontaine/scripts/subgoal_swap_live_oracles.py \
    --mode swap \
    --dump "reports/${swp}_swaps.json" \
    --data-root "$DATA_ROOT"

echo "=== SUBGOAL-SWAP ARM DONE (identity oracle green + swap + dump oracle green, rc=0) ==="
