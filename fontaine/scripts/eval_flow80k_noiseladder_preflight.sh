#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
#
# fontaine — noise-ladder rung 2 PREFLIGHT (#1, pre-reg
#   2026-08-08-prereg-noise-ladder-perdataset.md, stage-2 oracle item 5):
#   the routing byte-match that must be GREEN before the stage-2 panel
#   run. Two decodes of the committed 2-dataset ticket-2 plan (144 core
#   rows; both datasets route to ticket 2 in the committed map sha
#   15d92935…), matched composition (same plan, same batch size):
#     A. ROUTED  — --noise-tickets m64 + --noise-ticket-map (policy
#        gains _ticketmap);
#     B. PLAIN   — --noise-tickets t2-only bank (= m64[2:3] byte-
#        verified at materialization; policy gains _ticket).
#   Then noise_ladder_preflight_oracles.py adjudicates (identity +
#   pred byte-match + provenance + map coverage; abort-on-red). On
#   green it writes reports/analysis__noise_ladder_preflight_oracles.json
#   — the stage-2 launcher refuses to run without it. NO scalars.
# Cost: ~2×(corpus scan + 144-row heun-30 decode) ≈ 0.3 GPU-h of the
#   pre-registered ≤ 4 GPU-h remaining budget.
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a /home/ubuntu/eval__noiseladder_preflight.log) 2>&1

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

RUN=bijou_flow_artrunk_h1024_40k_ddp2
CKPT=outputs/train/${RUN}/step_080000
PLAN=plans/noise_ladder_preflight_t2.json
M64=plans/tickets_goldenticket_m64.npz
T2=plans/tickets_goldenticket_t2.npz
# Amendment 1 (2026-08-08): the panel-total extended enumeration —
# restriction to the committed 792 == pre-registered map 15d92935…
# exactly (the adjudicator enforces it); added datasets → 33.
MAP=plans/noise_ladder_ticketmap_panel.json

[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
f23d70abf9c3d8c90711c495452b335b9e660f92ae39a5dc8922d13082627cd3  plans/noise_ladder_preflight_t2.json
9bb13bc47a92f7cc764e81022a9a7b05dbb9ec391eb9ba8ab14d675c955cc7c0  plans/tickets_goldenticket_m64.npz
abfaf064f64299831f62cd4197721e6f206bed9b3caabd5ab9f3cfba3979a06d  plans/tickets_goldenticket_t2.npz
27858421c6293ccaf4d98405a9e8b1f2182480bc63459fea6e27d1e36e0ec6b7  plans/noise_ladder_ticketmap_panel.json
SHAS

DATA_ARGS=(
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0
    --episodes holdout --holdout-episodes 0.1 --split-seed 0
    --fps 30 --camera-counts 1 2
)

routed="eval__${RUN}__step_080000__noiseladder_preflight_t2_ticketmap_heun30"
echo "=== preflight 1/3: ROUTED decode ($routed) ==="
uv run python -m bijou.eval \
    "${DATA_ARGS[@]}" \
    --sample-plan "$PLAN" \
    --checkpoint "$CKPT" \
    --sample-steps 30 --sample-method heun \
    --sample-draws 1 --noise-tickets "$M64" --noise-ticket-map "$MAP" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${routed}.json" \
    --dump-predictions "reports/${routed}.npz"

plain="eval__${RUN}__step_080000__noiseladder_preflight_t2_ticket2_heun30"
echo "=== preflight 2/3: PLAIN ticket-2 decode ($plain) ==="
uv run python -m bijou.eval \
    "${DATA_ARGS[@]}" \
    --sample-plan "$PLAN" \
    --checkpoint "$CKPT" \
    --sample-steps 30 --sample-method heun \
    --sample-draws 1 --noise-tickets "$T2" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${plain}.json" \
    --dump-predictions "reports/${plain}.npz"

echo "=== preflight 3/3: adjudication (abort-on-red) ==="
uv run python fontaine/scripts/noise_ladder_preflight_oracles.py

echo "=== NOISE-LADDER PREFLIGHT DONE (rc=0) — stage 2 is launch-ready ==="
