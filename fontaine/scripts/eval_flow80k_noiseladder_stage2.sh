#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
#
# fontaine — noise-ladder rung 2 STAGE 2 (#1, pre-reg
#   2026-08-08-prereg-noise-ladder-perdataset.md): the ONE confirm eval —
#   full panel, deterministic single decode, per-dataset ticket routing
#   (--noise-ticket-map = the committed stage-01 map, sha 15d92935…,
#   97 qualifying datasets routed, non-qualifying → ticket 33). The
#   frozen reads (Δ_route on qualifying COMPLEMENT core rows,
#   dataset-clustered bootstrap CI95 seed 0; win table; horizon +
#   dispersion mirrors) run OFFLINE from this eval's dump — this script
#   never adjudicates.
# GATE: refuses to run without the preflight green record
#   (reports/analysis__noise_ladder_preflight_oracles.json — the routed
#   byte-match adjudication; eval_flow80k_noiseladder_preflight.sh).
# Cost: ≈ 0.9 GPU-h of the pre-registered ≤ 4 GPU-h remaining budget.
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a /home/ubuntu/eval__noiseladder_stage2.log) 2>&1

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

PREFLIGHT=reports/analysis__noise_ladder_preflight_oracles.json
[ -f "$PREFLIGHT" ] || { echo "no preflight green record $PREFLIGHT — run eval_flow80k_noiseladder_preflight.sh first; abort"; exit 1; }
uv run python - <<'EOF'
import json
record = json.load(open("reports/analysis__noise_ladder_preflight_oracles.json"))
assert record["verdict"] == "GREEN", record
assert record["ticket_map_sha256"] == (
    "15d9293553ac1a8878e0b7b0c385f03127a518d96e408bc1f496f5d8c4ec2173"
), record
print("preflight gate: GREEN, map sha pinned")
EOF

RUN=bijou_flow_artrunk_h1024_40k_ddp2
CKPT=outputs/train/${RUN}/step_080000
PLAN=plans/holdout_curated_v0_k4l2.json
M64=plans/tickets_goldenticket_m64.npz
MAP=reports/analysis__noise_ladder_stage01.json

[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
af3f85465b53d9ff783636d53923f89c6823700161da5667bf810cf17f922b1e  plans/holdout_curated_v0_k4l2.json
9bb13bc47a92f7cc764e81022a9a7b05dbb9ec391eb9ba8ab14d675c955cc7c0  plans/tickets_goldenticket_m64.npz
SHAS

name="eval__${RUN}__step_080000__panel_curated_v0_k4l2_ticketmap_heun30"
echo "=== noise-ladder stage 2: routed full panel ($name) ==="
uv run python -m bijou.eval \
    --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --sample-plan "$PLAN" \
    --checkpoint "$CKPT" \
    --sample-steps 30 --sample-method heun \
    --sample-draws 1 --noise-tickets "$M64" --noise-ticket-map "$MAP" \
    --batch-size 32 --num-workers 20 --seed 0 \
    --report-samples 0 \
    --output-json "reports/${name}.json" \
    --dump-predictions "reports/${name}.npz"

# Belt: the run's recorded map sha must equal the pre-registered one.
uv run python - <<'EOF'
import json
report = json.load(open(
    "reports/eval__bijou_flow_artrunk_h1024_40k_ddp2__step_080000__"
    "panel_curated_v0_k4l2_ticketmap_heun30.json",
))
assert report["ticket_map_sha256"] == (
    "15d9293553ac1a8878e0b7b0c385f03127a518d96e408bc1f496f5d8c4ec2173"
), report["ticket_map_sha256"]
print("stage-2 provenance: map sha matches the pre-registered map")
EOF

echo "=== NOISE-LADDER STAGE 2 DONE (rc=0) — frozen reads run offline from the npz ==="
