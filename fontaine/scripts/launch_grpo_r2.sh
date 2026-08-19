#!/usr/bin/env bash
# GRPO R2 launch kit — Amendment A3's frozen spec, one command per
# stage (posts/2026-08-15-prereg-grpo-r2-post-sft.md §9). STAGED, NOT
# ACTIVE: nothing here fires before the registered activation
# (A3.7); `parse-check` is the CPU-only oracle mode.
#
# Usage:
#   ./launch_grpo_r2.sh parse-check   # CPU: full-parse the frozen argv + emit the babysit entry template (no GPU, no load)
#   ./launch_grpo_r2.sh preflight     # GPU ~1.3 h: leg-0 sampled T=1.0 sim100 on the pinned base -> F-premise verdict json (detached unit)
#   ./launch_grpo_r2.sh launch        # GPU ~10 h: the A3.4 run — REFUSES unless the preflight verdict is PASS (BAND needs FORCE_BAND=1 after a decision post)
#
# Frozen pins (A3.4), spelled once in the ARGS array below:
#   base step_002000_v2 (schema-2 joint, corrected-table stats);
#   recipe §2+A2: v2 train / v1 eval reward, 8x8 T=1.0, clip ±2.0,
#   lr 1e-6, kl_beta 1.0, kl_stop 0.06, option-B surface;
#   re-pins: --train-seed-base 2000 (loop default 1000 collides with
#   the stage-B 1000-1099 band), --knockaway-baseline wave0 (violence
#   wire re-based at wave 0's measured rate; config default is an
#   R0-era er60k pin), --wave0-mixed-abort 0.20 (A3.3 calibration
#   gate; predicted ~0.44 at the greedy floor).
# Both launch modes go through run_detached.sh (transient systemd
# unit — survives driver teardown; the 3-incident class of 08-07).
# Standing checks before ANY local GPU launch: GPU-busy guard below +
# the owner policy-server rule (check compute-apps; never kill it).
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:?usage: launch_grpo_r2.sh parse-check|preflight|launch}"

CKPT="$HOME/checkpoints/finetune/fontaine_grasp_sft_joint_corrected/step_002000_v2"
OUT=outputs/sim/grpo_r2
PREFLIGHT_JSON="$OUT/preflight/sampled_t1_sim100.json"
VERDICT_JSON="$OUT/preflight/preflight_verdict.json"
HEARTBEAT="$OUT/loop/train.jsonl"

# The A3.4 frozen argv — ONE spelling, parsed by parse-check and fired
# by launch. --eval-every 5 / --save-every 5 are the loop defaults,
# spelled anyway (pre-reg discipline: the meta.json must not depend on
# defaults drifting).
ARGS=(
    --checkpoint "$CKPT"
    --out-dir "$OUT/loop"
    --total-steps 10
    --seeds-per-step 8 --draws 8 --temperature 1.0
    --surface b --train-reward v2
    --lr 1e-6 --kl-beta 1.0 --advantage-clip 2.0 --kl-stop 0.06
    --train-seed-base 2000
    --knockaway-baseline wave0
    --wave0-mixed-abort 0.20
    --eval-every 5 --save-every 5
)

emit_babysit_entry() {
    cat <<EOF
# --- babysit.toml entry template (append at launch, prune at completion) ---
[[run]]
name = "grpo_r2"
kind = "train-jsonl"
host = "local"
pgrep = "sim.grpo_loop"
pgrep_min = 1
gpu_indices = [0]
gpu_mem_min_mib = 8000
jsonl = "$PWD/$HEARTBEAT"
total_steps = 10
probe_key = "eval_successes"
vram_key = "vram_gib"
started_utc = "FILL-AT-LAUNCH (date -u +%FT%TZ)"
gpu_hours_per_wall_hour = 1.0
boundary = "step 10 endpoint: greedy sim100 vs 7 (PRIMARY, paired exact) + sampled T=1.0 sim100 vs the preflight floor (record-only) + flow unseen100 euler-10 vs 44 (F-regression leg); pace anchor ~0.98 GPU-h/step, budget ~14 expected / gate <=15 GPU-h (A3.5)"
anchors = [
    "token greedy 7/100 (A3.1 frozen anchor; success seeds 34,35,63,68,71,91,96)",
    "flow sibling 44/100 same trunk (A3.4 regression leg at boundary — option B moves the flow read too)",
    "preflight sampled T=1.0 count = recorded training-decode floor (verdict json in outputs/sim/grpo_r2/preflight/)",
    "wave-0 mixed_groups_frac: predicted ~0.44 at p=0.07; IN-LOOP abort <0.20 (A3.3 --wave0-mixed-abort; check the first train.jsonl row at first poll)",
    "wave-0 knockaway_frac = violence-wire baseline (--knockaway-baseline wave0 self-capture; greedy read measured 25/100 — the R0-era 10/120 pin is retired for this run)",
    "kl_stop 0.06 in-loop (single reading, no streak); anchor_k3_pre ~e-07-scale early per R1-B",
    "train-seed-base 2000 (stage-B 1000-1099 collision re-pin) — train rows must never show seeds < 2000",
]
[[run.gates]]
kind = "gpu_hours_max"
value = 15.0                      # A3.5 budget gate (~14 expected incl. boundary legs)
[[run.gates]]
kind = "vram_max_gib"
value = 75.0                      # R1-B envelope on the 80G H100
# --- end template ---
EOF
}

case "$MODE" in
  parse-check)
    echo "=== parse-check: the frozen A3.4 argv through sim.grpo_loop.parse_args ==="
    uv run python - "${ARGS[@]}" <<'PY'
import sys
from sim.grpo_loop import parse_args
args = parse_args(sys.argv[1:])
assert args.train_seed_base == 2000, args.train_seed_base
assert args.knockaway_baseline == "wave0", args.knockaway_baseline
assert args.wave0_mixed_abort == 0.20, args.wave0_mixed_abort
assert args.kl_stop == 0.06 and args.kl_beta == 1.0 and args.lr == 1e-6
assert args.surface == "b" and args.train_reward == "v2"
assert args.advantage_clip == 2.0 and args.total_steps == 10
print("parse-check GREEN:", vars(args))
PY
    echo
    emit_babysit_entry
    ;;

  preflight)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort (owner policy-server rule: check compute-apps, never kill it)"; exit 1; fi
    if [ ! -d "$CKPT" ]; then echo "FATAL: pinned base $CKPT missing" >&2; exit 2; fi
    mkdir -p "$OUT/preflight"
    # Leg 0 (A3.3): single-draw sampled T=1.0 sim100 on the pinned
    # base through the SAME probe driver as the 7/100 greedy leg —
    # same seeds, same substrate pin (--clutter-appearance standins,
    # the amendment's registration substrate), only the decode knob
    # moves. The verdict chains inside the unit.
    MUJOCO_GL=egl fontaine/scripts/run_detached.sh grpo-r2-preflight \
        bash -c "cd $PWD && MUJOCO_GL=egl uv run python -m sim.rollout_sim \
            --checkpoint '$CKPT' \
            --serve-head ar --ar-temperature 1.0 \
            --seed 0 --num-seeds 100 --episode-seconds 30 --execute-horizon 30 \
            --flow-decoder-dtype bfloat16 \
            --clutter-appearance standins \
            --out-dir '$OUT/preflight/episodes' --out-json '$PREFLIGHT_JSON' \
            2>&1 | tee '$HOME/eval__grpo_r2_preflight_sampled_t1.log' \
        && uv run python fontaine/scripts/grpo_r2_preflight_verdict.py \
            --json '$PREFLIGHT_JSON' --out '$VERDICT_JSON'"
    echo "preflight launched; verdict lands at $VERDICT_JSON"
    ;;

  launch)
    if [ ! -f "$VERDICT_JSON" ]; then
        echo "REFUSED: no preflight verdict at $VERDICT_JSON — run 'preflight' first (A3.3 leg 0 gates the launch)" >&2
        exit 1
    fi
    VERDICT=$(uv run python -c "import json,sys; print(json.load(open('$VERDICT_JSON'))['verdict'])")
    case "$VERDICT" in
      PASS) ;;
      BAND)
        if [ "${FORCE_BAND:-0}" != "1" ]; then
            echo "REFUSED: preflight verdict BAND (below the 7 anchor, not materially) — a decision post owns this; re-run with FORCE_BAND=1 after deciding + announcing" >&2
            exit 1
        fi
        echo "BAND override accepted (FORCE_BAND=1 — decision post required per charter)"
        ;;
      ABORT)
        echo "REFUSED: preflight verdict ABORT — F-premise falsified, the lane routes to iterate-once (A3.3); no override exists" >&2
        exit 1
        ;;
      *)
        echo "REFUSED: unrecognized verdict '$VERDICT' in $VERDICT_JSON" >&2
        exit 1
        ;;
    esac
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort (owner policy-server rule: check compute-apps, never kill it)"; exit 1; fi
    if [ ! -d "$CKPT" ]; then echo "FATAL: pinned base $CKPT missing" >&2; exit 2; fi
    echo "=== grpo R2 launch $(date -u +%FT%TZ): verdict $VERDICT, firing the A3.4 frozen argv ==="
    MUJOCO_GL=egl fontaine/scripts/run_detached.sh grpo-r2 \
        env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        uv run python -m sim.grpo_loop "${ARGS[@]}"
    echo
    echo "NOW (launch checklist): append the babysit entry (template below), clear"
    echo "no_live_runs_reason in fontaine/harness/babysit.toml, first babysit poll"
    echo "within ~30 min (util+rate check at first poll per the standing rule)."
    emit_babysit_entry
    ;;

  *)
    echo "unknown mode $MODE"; exit 2;;
esac
