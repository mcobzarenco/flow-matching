#!/usr/bin/env bash
# LAUNCH VIA fontaine/scripts/run_detached.sh <unit-name> bash <this script>:
# sessions run inside the tick unit's cgroup — a bare or setsid launch
# dies at session/unit teardown (3 incidents 2026-08-07, driver guard).
# fontaine — #19 T-sensitivity rung, AR-100k, RECORD-ONLY (one command).
# Pre-reg: fontaine/blog/src/posts/2026-08-06-prereg-ar-sampled-draws.md
#   — "One pre-registered sensitivity rung — T ∈ {0.5, 0.7, 1.3} at
#   draws 10 on the frozen q4 subset (4,301 rows, the state-probe
#   artifact) — is RECORD-ONLY: quoted as a dT diagnostic, never a
#   headline, and never a license to re-pick T post hoc." Cost clause:
#   "sensitivity rung 3 × q4 draws10 ≈ 12 GPU-h worst case, run ONLY
#   if the primary lands inside the gate."
# That precondition is mechanized below, not judged: the primary
#   full-panel report must exist, carry the registered sampling
#   semantics, and its elapsed GPU-h (report mtime − the babysit
#   registry's started_utc, 1 local GPU) must be ≤ 24.0. A primary
#   that fell back to q4 lands under a different stem — this guard
#   then aborts loudly and the rung waits for owner steer.
# Stems: eval__${RUN}__step_100000__stateprobe_q4_draws10_t{0.5,0.7,1.3}
#   — the tT tag uses the same %g formatting as the policy-name suffix
#   (bijou@100000_draws10_t0.5 etc., bijou/eval/policies.py), so file
#   stem and report policy always agree.
# --dump-draws: data retention only, no registered read changes (the
#   molmo2 endpoint launcher's precedent) — makes the #19 per-draw
#   reads (dispersion vs T, ceiling per T) offline-computable from
#   this same compute (~50 MB/rung at 4,301 rows).
# Reads: re-pool through draws10_t1_results.py's q4 subset-join path
#   (a T≠1.0 delta to its T-pinned guards — a follow-up read item,
#   NOT this launcher); quoted as dT only.
# babysit.toml: uncomment the prepared ar100k_tsens_q4 entry (bottom
#   of fontaine/harness/babysit.toml) and fill started_utc AT LAUNCH;
#   repoint `log` as each rung starts.
set -euo pipefail
cd /home/ubuntu/flow-matching
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN=bijou_arb_rcond_100k_ddp4
CKPT=/home/ubuntu/checkpoints/bijou-checkpoints/${RUN}/step_100000
PLAN_Q4=plans/holdout_curated_v0_k4l2_stateprobe_q4.json
PRIMARY_JSON=reports/eval__${RUN}__step_100000__panel_k4l2_draws10_t1.json

# Guards: checkpoint present, q4 plan frozen, primary inside its gate,
# local GPU quiet.
[ -d "$CKPT" ] || { echo "no checkpoint $CKPT — abort"; exit 1; }
sha256sum -c - <<'SHAS'
876c39c8fe2b3cb16945a40c35ec157c907b4f7417e7dfd0b6cf46dd47355ef5  plans/holdout_curated_v0_k4l2_stateprobe_q4.json
SHAS

# PRIMARY GATE (pre-reg cost clause, mechanized): full-panel report
# exists + registered semantics + elapsed GPU-h ≤ 24.0.
uv run python - "$PRIMARY_JSON" <<'PYGATE'
import datetime as dt, json, pathlib, sys, tomllib

primary = pathlib.Path(sys.argv[1])
if not primary.is_file():
    sys.exit(
        f"PRIMARY GATE: {primary} missing — the full-panel primary has not "
        "landed (still running, or it fell back to the q4 stem). The rung "
        "runs ONLY on a primary inside its gate — abort."
    )
rep = json.loads(primary.read_text())
want = {"ar_temperature": 1.0, "sample_draws": 10,
        "sample_plan": "plans/holdout_curated_v0_k4l2.json",
        "core_frames": 17204, "labeled_frames": 8596}
for k, v in want.items():
    if rep.get(k) != v:
        sys.exit(f"PRIMARY GATE: {k} = {rep.get(k)!r}, registered {v!r} — abort")
pols = [s.get("policy", "") for s in rep.get("summaries", [])]
if not any(p.endswith("_draws10_t1") for p in pols):
    sys.exit(f"PRIMARY GATE: no _draws10_t1 summary in {pols} — abort")

reg = tomllib.loads(
    pathlib.Path("fontaine/harness/babysit.toml").read_text())
runs = [r for r in reg.get("run", []) if r.get("name") == "draws10_t1"]
if len(runs) != 1:
    sys.exit("PRIMARY GATE: babysit registry has no single draws10_t1 entry "
             "— cannot establish started_utc, abort")
started = runs[0]["started_utc"]
if isinstance(started, str):
    started = dt.datetime.fromisoformat(started.replace("Z", "+00:00"))
if started.tzinfo is None:
    started = started.replace(tzinfo=dt.timezone.utc)
ended = dt.datetime.fromtimestamp(primary.stat().st_mtime, dt.timezone.utc)
gpu_h = (ended - started).total_seconds() / 3600.0 \
    * float(runs[0].get("gpu_hours_per_wall_hour", 1.0))
if not 0.0 < gpu_h <= 24.0:
    sys.exit(f"PRIMARY GATE: primary took {gpu_h:.1f} GPU-h, gate 24.0 — "
             "the rung does not run (pre-reg cost clause); owner steer")
print(f"PRIMARY GATE PASS: primary landed in {gpu_h:.1f} GPU-h <= 24.0 "
      f"({started.isoformat()} -> {ended.isoformat()})")
PYGATE

mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$mem" -gt 1024 ]; then echo "GPU busy (${mem} MiB) — abort"; exit 1; fi

# Three sequential rungs; T tags match the policy suffix's %g format.
for T in 0.5 0.7 1.3; do
    name="eval__${RUN}__step_100000__stateprobe_q4_draws10_t${T}"
    if [ -s "reports/${name}.json" ]; then
        echo "rung t${T} already banked (reports/${name}.json) — skipping"
        continue
    fi
    echo "=== rung T=${T} (stem ${name}) ==="
    uv run python -m bijou.eval \
        --data /home/ubuntu/datasets/mcobzarenco/community_curated_v0 \
        --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
        --fps 30 --camera-counts 1 2 \
        --sample-plan "$PLAN_Q4" \
        --checkpoint "$CKPT" \
        --ar-temperature "$T" --sample-draws 10 \
        --batch-size 32 --num-workers 20 --seed 0 \
        --report-samples 0 \
        --output-json "reports/${name}.json" \
        --dump-predictions "reports/${name}.npz" \
        --dump-draws "reports/${name}_draws.npz" \
        --report "reports/${name}.html" \
        2>&1 | tee "/home/ubuntu/${name}.log"
    echo "=== rung T=${T} DONE ==="
done
echo "=== T-SENSITIVITY RUNG COMPLETE (3/3, record-only dT diagnostic) ==="
