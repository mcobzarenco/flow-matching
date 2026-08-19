#!/usr/bin/env bash
# GRPO R2 launch kit — Amendment A3's frozen spec, one command per
# stage (posts/2026-08-15-prereg-grpo-r2-post-sft.md §9). STAGED, NOT
# ACTIVE: nothing here fires before the registered activation
# (A3.7); `parse-check` is the CPU-only oracle mode.
#
# Usage:
#   ./launch_grpo_r2.sh parse-check   # CPU: full-parse the frozen argv (loop + boundary legs) + run the boundary verdict's provenance guards on the synthesized configs + emit the babysit entry template (no GPU, no load)
#   ./launch_grpo_r2.sh parity        # GPU ~0.7 h (A5 LAUNCH GATE): seeds 200-219 greedy through BOTH serving paths (loop stack --joint-frame rig vs BijouPolicy --serve-head ar) -> parity verdict json (detached unit)
#   ./launch_grpo_r2.sh preflight     # GPU ~1.3 h: leg-0 sampled T=1.0 sim100 on the pinned base -> F-premise verdict json (detached unit)
#   ./launch_grpo_r2.sh launch        # GPU ~10 h: the A3.4 run — REFUSES unless the preflight verdict is PASS (BAND needs FORCE_BAND=1 after a decision post) AND the A5 parity verdict is PASS
#   ./launch_grpo_r2.sh boundary <endpoint-dir | step_NNNN.pt>
#                                     # GPU ~3.9 h (A3.4/A3.5 endpoint): the three boundary legs sequentially as ONE detached
#                                     # unit — greedy token sim100 + sampled T=1.0 sim100 + flow unseen100 euler-10, seeds 0-99,
#                                     # anchors' exact driver — chaining grpo_r2_boundary_verdict. A loop overlay .pt is
#                                     # materialized onto the pinned base first (CPU, grpo_r2_materialize_endpoint).
#                                     # REFUSES while unit grpo-r2 is alive, without a PASS preflight verdict, or on the base dir.
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

MODE="${1:?usage: launch_grpo_r2.sh parse-check|parity|preflight|launch|boundary}"

CKPT="$HOME/checkpoints/finetune/fontaine_grasp_sft_joint_corrected/step_002000_v2"
OUT=outputs/sim/grpo_r2
PREFLIGHT_JSON="$OUT/preflight/sampled_t1_sim100.json"
VERDICT_JSON="$OUT/preflight/preflight_verdict.json"
PARITY_DISCRETE_JSON="$OUT/parity/discrete_leg.json"
PARITY_ANCHOR_JSON="$OUT/parity/anchor_leg.json"
PARITY_VERDICT_JSON="$OUT/parity/parity_verdict.json"
HEARTBEAT="$OUT/loop/train.jsonl"
BOUNDARY_DIR="$OUT/boundary"
BOUNDARY_GREEDY_JSON="$BOUNDARY_DIR/token_greedy_sim100.json"
BOUNDARY_SAMPLED_JSON="$BOUNDARY_DIR/sampled_t1_sim100.json"
BOUNDARY_FLOW_JSON="$BOUNDARY_DIR/flow_unseen100.json"
BOUNDARY_VERDICT_JSON="$BOUNDARY_DIR/boundary_verdict.json"

# The A3.4 boundary legs — the ANCHORS' exact driver + substrate pins
# (joint_probes/token_unseen.json + flow_unseen.json configs, receipts
# in-repo): sim.rollout_sim (the BijouPolicy anchor path — the A5
# parity gate is exactly what licenses serving the endpoint through
# it), seeds 0-99, 30 s episodes, horizon 30, bf16 decoder, standins
# substrate (the registration substrate; 'patched' became the driver
# default 08-18). NO --stats-repo-id: the queue item's spelled pin
# (so101_pick_place_v2) is a title drift — the anchors wore that row on
# the RETIRED step_002000 dir; the v2 re-derivation carries the
# corrected table as its merged row (per-dataset table holds only the
# demos row), so the explicit pin would be REFUSED at load and the
# default lookup (v2 row if carried, else merged) IS the lane's
# registered serving convention (the preflight PASS ran exactly this,
# worn_row "<merged-table>").
BOUNDARY_COMMON=(
    --seed 0 --num-seeds 100
    --episode-seconds 30 --execute-horizon 30
    --flow-decoder-dtype bfloat16
    --clutter-appearance standins
)
BOUNDARY_GREEDY=(--serve-head ar)                     # PRIMARY: greedy vs 7/100, paired exact
BOUNDARY_SAMPLED=(--serve-head ar --ar-temperature 1.0)  # record-only vs the preflight floor
BOUNDARY_FLOW=(--method euler --sample-steps 10)      # F-regression leg vs 44/100

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
    # Wave-0 postmortem 08-19 (A4): the loop inherited the 08-18
    # promotion's 'patched' production default while every R2 anchor +
    # the preflight ran standins — the stand-ins-era policy is inert on
    # patched (64/64 wave-0 episodes zero interaction; probe driver on
    # the SAME seeds under standins interacts 6/8). Pin the registered
    # substrate on the waves + in-loop eval.
    --clutter-appearance standins
    # A5 (serving-parity fix 08-19): the v2 base is a bijou-trained
    # v3.0-frame table — the loop's port-era v30-to-v21 shim was the
    # relaunch kill's root cause. 'rig' = identity frame; the loader
    # fingerprints the table and refuses a mismatch either way.
    --joint-frame rig
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
assert args.clutter_appearance == "standins", args.clutter_appearance
assert args.joint_frame == "rig", args.joint_frame
print("parse-check GREEN:", vars(args))
PY
    echo
    echo "=== parse-check: the boundary legs through sim.rollout_sim.parse_args + the verdict's provenance guards ==="
    # The launcher's contract is to NEVER produce a wrong-leg json: the
    # exact argv the boundary mode fires is parsed by the driver's own
    # parser, synthesized into the config dicts the driver records, and
    # pushed through grpo_r2_boundary_verdict's guard functions — the
    # legs and the verdict cannot drift apart silently.
    uv run python - \
        --checkpoint "$BOUNDARY_DIR/endpoint_stub" "${BOUNDARY_GREEDY[@]}" "${BOUNDARY_COMMON[@]}" --- \
        --checkpoint "$BOUNDARY_DIR/endpoint_stub" "${BOUNDARY_SAMPLED[@]}" "${BOUNDARY_COMMON[@]}" --- \
        --checkpoint "$BOUNDARY_DIR/endpoint_stub" "${BOUNDARY_FLOW[@]}" "${BOUNDARY_COMMON[@]}" <<'PY'
import sys

legs, current = [], []
for token in sys.argv[1:]:
    if token == "---":
        legs.append(current)
        current = []
    else:
        current.append(token)
legs.append(current)
greedy_argv, sampled_argv, flow_argv = legs

import sim.rollout_sim as rollout_sim
from fontaine.scripts.grpo_r2_boundary_verdict import (
    _guard_checkpoints,
    _guard_flow,
    _guard_greedy,
    _guard_sampled,
    _guard_seed_window,
)

def parse(argv):
    saved, sys.argv = sys.argv, ["rollout_sim.py", *argv]
    try:
        return rollout_sim.parse_args()
    finally:
        sys.argv = saved

def payload(args):
    return {
        "config": {
            "checkpoint": str(args.checkpoint),
            "seed": args.seed,
            "num_seeds": args.num_seeds,
            "serve_head": args.serve_head,
            "ar_temperature": args.ar_temperature,
            "method": args.method,
            "sample_steps": args.sample_steps,
        },
        "episodes": [
            {"seed": s} for s in range(args.seed, args.seed + args.num_seeds)
        ],
    }

parsed = [parse(argv) for argv in (greedy_argv, sampled_argv, flow_argv)]
for args in parsed:
    # The anchors' shared substrate pins, asserted leg by leg.
    assert args.clutter_appearance == "standins", args.clutter_appearance
    assert args.episode_seconds == 30.0, args.episode_seconds
    assert args.execute_horizon == 30, args.execute_horizon
    assert args.flow_decoder_dtype == "bfloat16", args.flow_decoder_dtype
    assert args.stats_repo_id is None, args.stats_repo_id
    assert args.wrist_transform == "none", args.wrist_transform
    assert args.draws == 1 and args.sde_noise_level is None
greedy, sampled, flow = (payload(args) for args in parsed)
_guard_greedy(greedy)
_guard_sampled(sampled)
_guard_flow(flow)
for leg_name, leg in (("greedy", greedy), ("sampled", sampled), ("flow", flow)):
    _guard_seed_window(leg["config"], leg["episodes"], leg_name)
_guard_checkpoints({"greedy": greedy, "sampled": sampled, "flow": flow})
print("boundary parse-check GREEN: 3 legs parse + pass the verdict's provenance guards")
PY
    echo
    emit_babysit_entry
    ;;

  parity)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort (owner policy-server rule: check compute-apps, never kill it)"; exit 1; fi
    if [ ! -d "$CKPT" ]; then echo "FATAL: pinned base $CKPT missing" >&2; exit 2; fi
    mkdir -p "$OUT/parity"
    # A5 launch gate: the in-loop eval band (seeds 200-219), greedy,
    # through BOTH serving paths on the SAME substrate pin. Leg 1 =
    # the loop's stack (rollout_sim_parallel --molmoact2-discrete,
    # grammar-masked, --joint-frame rig — exactly what waves + in-loop
    # eval serve). Leg 2 = the anchor path (BijouPolicy --serve-head
    # ar, the preflight/anchor convention). Verdict rule lives in
    # grpo_r2_parity_verdict.py; the launch mode refuses without PASS.
    MUJOCO_GL=egl fontaine/scripts/run_detached.sh grpo-r2-parity \
        bash -c "cd $PWD && MUJOCO_GL=egl uv run python -m sim.rollout_sim_parallel \
            --molmoact2-discrete '$CKPT' \
            --molmoact2-grammar-masked --joint-frame rig \
            --seed 200 --num-seeds 20 --workers 4 \
            --episode-seconds 30 --execute-horizon 30 \
            --clutter-appearance standins \
            --out-dir '$OUT/parity/discrete_episodes' \
            --out-json '$PARITY_DISCRETE_JSON' \
            2>&1 | tee '$HOME/eval__grpo_r2_parity_discrete.log' \
        && MUJOCO_GL=egl uv run python -m sim.rollout_sim \
            --checkpoint '$CKPT' \
            --serve-head ar \
            --seed 200 --num-seeds 20 --episode-seconds 30 --execute-horizon 30 \
            --flow-decoder-dtype bfloat16 \
            --clutter-appearance standins \
            --out-dir '$OUT/parity/anchor_episodes' \
            --out-json '$PARITY_ANCHOR_JSON' \
            2>&1 | tee '$HOME/eval__grpo_r2_parity_anchor.log' \
        && uv run python fontaine/scripts/grpo_r2_parity_verdict.py \
            --discrete-json '$PARITY_DISCRETE_JSON' \
            --anchor-json '$PARITY_ANCHOR_JSON' \
            --out '$PARITY_VERDICT_JSON'"
    echo "parity read launched; verdict lands at $PARITY_VERDICT_JSON"
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

  boundary)
    TARGET="${2:?usage: launch_grpo_r2.sh boundary <endpoint-ckpt-dir | loop step_NNNN.pt overlay>}"
    # The registered refusal: boundary legs read the ENDPOINT — while
    # unit grpo-r2 is alive the checkpoint is still moving and the GPU
    # is owned by the run.
    if systemctl --user is-active --quiet grpo-r2; then
        echo "REFUSED: unit grpo-r2 is still active — the boundary reads the endpoint after the run completes (never mid-run)" >&2
        exit 1
    fi
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$mem" -gt 1024 ]; then echo "GPU 0 busy (${mem} MiB) — abort (owner policy-server rule: check compute-apps, never kill it)"; exit 1; fi
    # The verdict chain re-guards this at the end; failing HERE costs
    # 0 GPU-h instead of the ~3.9 the legs burn first.
    if [ ! -f "$VERDICT_JSON" ]; then
        echo "REFUSED: no preflight verdict at $VERDICT_JSON — the boundary verdict prices the sampled leg against the preflight floor; without a PASS there is nothing to read" >&2
        exit 1
    fi
    PREFLIGHT=$(uv run python -c "import json; print(json.load(open('$VERDICT_JSON'))['verdict'])")
    if [ "$PREFLIGHT" != "PASS" ]; then
        echo "REFUSED: preflight verdict '$PREFLIGHT' — an A3.4 run only launches on PASS, so a non-PASS here means these legs are reading the wrong lane" >&2
        exit 1
    fi
    # Resolve the endpoint: a loop overlay .pt materializes onto the
    # pinned base first (CPU, atomic, validated — the loop banks
    # trainable-only saves; the anchors' serving path loads a
    # self-contained VLA dir).
    if [ -f "$TARGET" ]; then
        ENDPOINT="$BOUNDARY_DIR/endpoint_$(basename "${TARGET%.pt}")"
        uv run python -m fontaine.scripts.grpo_r2_materialize_endpoint \
            --base "$CKPT" --overlay "$TARGET" --out "$ENDPOINT"
    elif [ -d "$TARGET" ]; then
        if [ ! -f "$TARGET/metadata.json" ]; then
            echo "FATAL: $TARGET has no metadata.json — not a VLA checkpoint dir" >&2
            exit 2
        fi
        ENDPOINT="$TARGET"
    else
        echo "FATAL: $TARGET is neither a checkpoint dir nor a loop overlay .pt" >&2
        exit 2
    fi
    case "$(basename "${ENDPOINT%/}")" in
      step_002000|step_002000_v2)
        echo "REFUSED: $ENDPOINT is the pinned BASE — the boundary reads the GRPO endpoint (the verdict refuses base-minted jsons for the same reason)" >&2
        exit 1;;
    esac
    mkdir -p "$BOUNDARY_DIR"
    # ONE detached unit, legs sequential, verdict chained — the
    # boundary session launches this and reads $BOUNDARY_VERDICT_JSON.
    MUJOCO_GL=egl fontaine/scripts/run_detached.sh grpo-r2-boundary \
        bash -c "cd $PWD && MUJOCO_GL=egl uv run python -m sim.rollout_sim \
            --checkpoint '$ENDPOINT' ${BOUNDARY_GREEDY[*]} ${BOUNDARY_COMMON[*]} \
            --out-dir '$BOUNDARY_DIR/token_greedy_episodes' \
            --out-json '$BOUNDARY_GREEDY_JSON' \
            2>&1 | tee '$HOME/eval__grpo_r2_boundary_greedy.log' \
        && MUJOCO_GL=egl uv run python -m sim.rollout_sim \
            --checkpoint '$ENDPOINT' ${BOUNDARY_SAMPLED[*]} ${BOUNDARY_COMMON[*]} \
            --out-dir '$BOUNDARY_DIR/sampled_t1_episodes' \
            --out-json '$BOUNDARY_SAMPLED_JSON' \
            2>&1 | tee '$HOME/eval__grpo_r2_boundary_sampled.log' \
        && MUJOCO_GL=egl uv run python -m sim.rollout_sim \
            --checkpoint '$ENDPOINT' ${BOUNDARY_FLOW[*]} ${BOUNDARY_COMMON[*]} \
            --out-dir '$BOUNDARY_DIR/flow_unseen_episodes' \
            --out-json '$BOUNDARY_FLOW_JSON' \
            2>&1 | tee '$HOME/eval__grpo_r2_boundary_flow.log' \
        && uv run python -m fontaine.scripts.grpo_r2_boundary_verdict \
            --greedy-json '$BOUNDARY_GREEDY_JSON' \
            --sampled-json '$BOUNDARY_SAMPLED_JSON' \
            --flow-json '$BOUNDARY_FLOW_JSON' \
            --preflight-json '$VERDICT_JSON' \
            --out '$BOUNDARY_VERDICT_JSON' \
            2>&1 | tee '$HOME/eval__grpo_r2_boundary_verdict.log'"
    echo "boundary legs launched (~3.9 GPU-h, A3.5 pacing); the chained verdict lands at $BOUNDARY_VERDICT_JSON"
    ;;

  launch)
    if [ ! -f "$PARITY_VERDICT_JSON" ]; then
        echo "REFUSED: no serving-parity verdict at $PARITY_VERDICT_JSON — run 'parity' first (A5: the loop stack must agree with the anchor path before any R2 fire)" >&2
        exit 1
    fi
    PARITY=$(uv run python -c "import json; print(json.load(open('$PARITY_VERDICT_JSON'))['verdict'])")
    if [ "$PARITY" != "PASS" ]; then
        echo "REFUSED: serving-parity verdict '$PARITY' (A5) — the loop path still diverges from the anchor path; no override exists" >&2
        exit 1
    fi
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
