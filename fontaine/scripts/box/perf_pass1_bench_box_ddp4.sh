#!/usr/bin/env bash
# Perf pass-1 bench ladder — BOX form (pre-reg
#   fontaine/blog/src/posts/2026-08-08-prereg-molmo2-perf-pass1.md,
#   amendment ~15:0xZ 08-08: the ladder moved here from the local
#   1xH100 after the single-GPU full-recipe run proved structurally
#   OOM — unsharded AdamW states put the process at 78.2/79.18 GiB by
#   step 2 at ANY batch size (chunked backward makes activations
#   batch-invariant), and --activation-checkpointing cannot bridge it
#   at HEAD (recompute escapes the sdpa_kernel pin -> backend
#   mismatch abort; find filed on idea #20). This form is the TRUE
#   recipe — 4xDDP, zero1, batch 12, chunks 6 — so it supersedes the
#   pre-reg's "box transfer smoke": the bench IS the box read.
# Sequence: bench_A (HEAD) -> bench_B (P1 only 00cdafe) -> bench_C
#   (full bundle 22e8148), 320 steps each, ~12 min/run + loads;
#   PLUS a 50-step log-every-1 overlay pair (A vs B) for the pre-reg
#   overlay oracle (b). ~2.5-3 GPU-h total, <= 3 ceiling.
# WINDOW: opens ONLY after the 60k close + chained panel eval + the
#   fields panel have all landed (GPU-clear guard enforces); every
#   launch via the box run_detached wrapper.
# PREREQ ON THE BOX: git fetch && worktree with branch perf-pass1
#   available at /home/ubuntu/flow-matching-perfpass1 (create with:
#   git worktree add ../flow-matching-perfpass1 perf-pass1).
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

for g in 0 1 2 3; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    if [ "$mem" -gt 1024 ]; then echo "GPU $g busy (${mem} MiB) — abort"; exit 1; fi
done
[ -d "$WT" ] || { echo "worktree $WT missing — see PREREQ in header"; exit 1; }

run_cfg () {
    local name=$1 repo=$2 steps=$3 logev=$4
    echo "=== RUN $name (repo=$repo steps=$steps) $(date -u +%H:%M:%SZ) ==="
    rm -rf "$OUT/$name"
    (
        cd "$repo"
        PYTHONPATH=$repo "$VENV/torchrun" --standalone --nproc-per-node=4 \
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
            --batch-size 12 \
            --zero1 --backward-chunks 6 --chunk-grad-allreduce \
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
run_cfg perfpass1_box_overlay_A "$MAIN" 50 1
run_cfg perfpass1_box_overlay_B "$WT"   50 1
run_cfg perfpass1_box_bench_A "$MAIN" 320 20
run_cfg perfpass1_box_bench_B "$WT"   320 20
git -C "$WT" checkout -q "$FULL_COMMIT"
run_cfg perfpass1_box_bench_C "$WT"   320 20

"$VENV/python" - << 'EOF'
import json, statistics
def rows(name):
    return [json.loads(x) for x in open(
        f"/home/ubuntu/flow-matching/outputs/train/{name}/train_log.jsonl")]
def med(name):
    rs = rows(name)
    tail = [r for r in rs if r["step"] > 80]
    return statistics.median(r["s_per_step"] for r in tail), max(
        r.get("vram_alloc_peak_gib", 0) for r in rs), rs
oa, ob = rows("perfpass1_box_overlay_A"), rows("perfpass1_box_overlay_B")
band = max(abs(oa[i]["loss"] - oa[i - 1]["loss"]) for i in range(1, len(oa)))
worst = max(abs(x["loss"] - y["loss"]) for x, y in zip(oa, ob))
print(f"OVERLAY: 50-step max |lossA-lossB| = {worst:.4f} vs A's own "
      f"step-to-step band {band:.4f} -> {'PASS' if worst <= band else 'FAIL'}"
      f" | step1 loss {oa[0]['loss']} vs {ob[0]['loss']}, "
      f"grad_norm {oa[0]['grad_norm']} vs {ob[0]['grad_norm']}")
a, va, _ = med("perfpass1_box_bench_A")
b, vb, _ = med("perfpass1_box_bench_B")
c, vc, rows_c = med("perfpass1_box_bench_C")
w = max((r.get("vram_window_peak_gib", 0) for r in rows_c), default=None)
print(f"LADDER(BOX): A={a:.3f}s B={b:.3f}s C={c:.3f}s | "
      f"B vs A {(a-b)/a*100:.1f}% | C vs A {(a-c)/a*100:.1f}% "
      f"(decision bar >=5%) | vram A={va} B={vb} C={vc} "
      f"(guard: C <= A*1.02) window_peak_C={w}")
EOF
echo "=== PERF PASS-1 BOX LADDER DONE $(date -u +%H:%M:%SZ) ==="
