#!/usr/bin/env bash
# Perf pass-1 bench ladder — pre-reg:
#   fontaine/blog/src/posts/2026-08-08-prereg-molmo2-perf-pass1.md
# Sequence (single local H100, sequential, box untouched):
#   parity_A (HEAD, MATH)    50 steps log-every 1  ┐ P1 one-step bounds
#   parity_B (P1, cuDNN)     50 steps log-every 1  ┘ + 50-step overlay
#   bench_A  (HEAD)          320 steps log-every 20
#   bench_B  (P1 only, 00cdafe)   320 steps
#   bench_C  (full bundle, 22e8148) 320 steps
# AMENDMENT (recorded in-channel 2026-08-08 ~14:4xZ): batch 8 +
# backward-chunks 4, not the pre-reg's 12/6 — a single-GPU bench
# unshards the ZeRO-1 optimizer (~+11 GiB vs the 4-rank launcher's
# per-GPU footprint), so batch 12 risks OOM on the 80 GiB card; and
# chunks must divide the batch (train.py guard). 8/4 keeps the live
# per-chunk size exactly (2 samples/chunk, as 12/6); the P3
# sync-multiplier runs x4 not x6 per step (slightly conservative for
# the bundle read). Uniform across every rung: relative reads stand.
# LAUNCH: fontaine/scripts/run_detached.sh fontaine-perfpass1-bench \
#             bash fontaine/scripts/perf_pass1_bench.sh
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAIN=/home/ubuntu/flow-matching
WT=/home/ubuntu/flow-matching-perfpass1
VENV=$MAIN/.venv/bin
P1_COMMIT=00cdafe
FULL_COMMIT=22e8148
OUT=$MAIN/outputs/train

# GPU-clear guard (local card 0 only).
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort"; exit 1; fi

run_cfg () {
    local name=$1 repo=$2 steps=$3 logev=$4
    echo "=== RUN $name (repo=$repo steps=$steps) $(date -u +%H:%M:%SZ) ==="
    rm -rf "$OUT/$name"
    (
        cd "$repo"
        PYTHONPATH=$repo "$VENV/torchrun" --standalone --nproc-per-node=1 \
            -m bijou.train \
            --train-data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
                /home/ubuntu/datasets/mcobzarenco/so101_pick_place_clean \
                /home/ubuntu/datasets/mcobzarenco/so101_pick_place_v2 \
            --fps 30 --camera-counts 1 2 \
            --holdout-episodes 0.1 --split-seed 0 \
            --decoder ar_backbone \
            --backbone allenai/Molmo2-4B \
            --max-crops 1 \
            --fast-tokenizer mcobzarenco/bijou-checkpoints/fast_tokenizer_v2 \
            --aux-fields subgoal holding progress event visible \
            --aux-dropout 0.0 --field-dropout 0.1 \
            --condition-fields subgoal outcome smoothness \
            --condition-dropout 0.1 --subgoal-dropout 0.5 \
            --instruction-augment 0.5 \
            --camera-kind-dropout 0.1 \
            --decoder-lr 1e-4 --backbone-text-lr 2e-5 --grad-clip 100 \
            --steps "$steps" --warmup-steps 1000 \
            --batch-size 8 \
            --zero1 --backward-chunks 4 --chunk-grad-allreduce \
            --num-workers 20 --prefetch-factor 4 \
            --eval-samples 256 --eval-every 100000 --save-every 100000 \
            --log-every "$logev" \
            --seed 0 \
            --save-dir "$OUT/$name"
    ) > "$OUT/${name}.launch.log" 2>&1 || {
        echo "RUN $name FAILED — log tail:"
        tail -30 "$OUT/${name}.launch.log"
        exit 1
    }
    tail -3 "$OUT/${name}.launch.log"
}
mkdir -p "$OUT"

git -C "$WT" checkout -q "$P1_COMMIT"
run_cfg perfpass1_parity_A "$MAIN" 50 1
run_cfg perfpass1_parity_B "$WT"   50 1

# P1 one-step parity gate (pre-reg bounds: loss abs diff <= 1e-3,
# grad-norm rel diff <= 1e-2 at step 1) — surfaced loudly; the ladder
# continues either way and the driver judges per decision rule 4.
"$VENV/python" - << 'EOF' || true
import json
a = [json.loads(x) for x in open("/home/ubuntu/flow-matching/outputs/train/perfpass1_parity_A/train_log.jsonl")]
b = [json.loads(x) for x in open("/home/ubuntu/flow-matching/outputs/train/perfpass1_parity_B/train_log.jsonl")]
la, lb = a[0]["loss"], b[0]["loss"]
ga, gb = a[0]["grad_norm"], b[0]["grad_norm"]
dl, dg = abs(la - lb), abs(ga - gb) / max(abs(ga), 1e-9)
ok = dl <= 1e-3 and dg <= 1e-2
print(f"PARITY step1: loss {la} vs {lb} (|d|={dl:.2e} bound 1e-3), "
      f"grad_norm {ga} vs {gb} (rel {dg:.2e} bound 1e-2) -> "
      f"{'PASS' if ok else 'FAIL'}")
EOF

run_cfg perfpass1_bench_A "$MAIN" 320 20
run_cfg perfpass1_bench_B "$WT"   320 20
git -C "$WT" checkout -q perf-pass1
run_cfg perfpass1_bench_C "$WT"   320 20

"$VENV/python" - << 'EOF'
import json, statistics
def med(name):
    rows = [json.loads(x) for x in open(f"/home/ubuntu/flow-matching/outputs/train/{name}/train_log.jsonl")]
    tail = [r for r in rows if r["step"] > 80]
    return statistics.median(r["s_per_step"] for r in tail), max(
        r.get("vram_alloc_peak_gib", 0) for r in rows), rows
a, va, _ = med("perfpass1_bench_A")
b, vb, _ = med("perfpass1_bench_B")
c, vc, rows_c = med("perfpass1_bench_C")
w = max((r.get("vram_window_peak_gib", 0) for r in rows_c), default=None)
print(f"LADDER: A={a:.3f}s B={b:.3f}s C={c:.3f}s | "
      f"B vs A {(a-b)/a*+100:.1f}% | C vs A {(a-c)/a*100:.1f}% "
      f"(decision bar >=5%) | vram A={va} B={vb} C={vc} "
      f"(guard: C <= A*1.02) window_peak_C={w}")
EOF
echo "=== PERF PASS-1 LADDER DONE $(date -u +%H:%M:%SZ) ==="
