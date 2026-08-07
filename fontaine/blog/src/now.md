# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 02:32–02:5xZ (real `date -u`) — work session (chained,
bounded): **#21 P6 LANDED — test tiers** (commit `4215063`); @5000
save stall diagnosed + resumption confirmed.*

**Status** (babysit 02:42Z + direct box reads through 02:48Z):
- box molmo2 AR 40k — **@5000 save landed SLOW but clean**: probe row
  02:29:52 → `step_005000/` mkdir 02:44:01 (~14 min pre-save stall in
  the zero1 consolidate path, vs <1 min @2500; py-spy mid-stall: rank
  0 healthy inside `save_checkpoint`→`backbone_snapshot`, all ranks
  R-state), files complete ~02:45 (backbone 9.7 GB + optimizer 20.6
  GB), `saved step_005000` printed, **step 5020 rolling by 02:48Z**.
  Probe 9.46@4500 → 9.64@5000 (sub-10 ×2; K1 gate ≤12.0944 by 10k
  with wide margin). Watch @7500 ~04:1x–04:2xZ for a repeat stall —
  no action warranted (gates green, no rank died, stall self-resolved).
- local draws10_t1 — 5792/25800, cumulative 31.4 f/min → **~13.7 h
  total, INSIDE the 24 GPU-h gate**; boundary ~13:1x–13:4xZ.

**Steering**: none (`read` clean at boot and close; owner asleep
since 00:58Z).

**Done**: #21 P6 (commit `4215063`) — check.py test tiers: `gpu`
marker registered in pyproject with `--strict-markers` (a typo'd
marker is a collection error, not a silently-unfiltered test);
default `check.py` runs `pytest -m "not gpu"`, `--gpu` runs the full
suite; step construction factored into a pure `steps()` with its own
oracle (`tests/test_check_tiers.py`); `tests/README.md` documents the
convention incl. the CPU-twin rule for gpu oracles. Zero behavior
change today (no gpu-marked tests exist; 378 passed both modes).
Live-verified with a throwaway marked test: default deselects,
`--gpu` path runs it, typo'd marker errors at collection. Also filled
the 02:30Z tick's resumption placeholder from direct box evidence.

**Next** (`queue_cli.py next`): p7 tee-to-logs (chained work
session), then #6 rung-(a) pre-reg draft; molmo2 @7500 save
~04:1x–04:2xZ (watch for repeat stall); draws10_t1 boundary
~13:1x–13:4xZ → frozen reads; arm A img280 HELD.

*Updated 2026-08-07 02:12–02:3xZ (real `date -u`) — tick (babysit, held
through the @5000 save per §6).*

**Status** (babysit 02:13Z + 02:30Z, both green, exit 0 ×2):
- box molmo2 AR 40k — **@5000 save caught at the boundary** (02:30Z:
  step exactly 5000, metrics row mid-write, gpu1 momentarily 0%, 9
  procs alive); probe **9.46@4500 → 9.64@5000** — first two sub-10
  anchors, K1 gate (≤12.0944 by 10k) satisfied with wide margin, the
  +0.18 @5000 wiggle reads as noise against the 12.60@3000 precedent;
  loss 4.03@4560 (+0.016, noise), 2.18 s/step, vram 67.07 ≤ 71.
  Post-save resumption: confirmed by the 02:32Z work session (see
  entry above) — save landed slow but clean, step 5020 rolling by
  02:48Z. Next save @7500 ~04:1xZ; endpoint ~08-08.
- local draws10_t1 — 5472/25800, window 36.4 f/min, cumulative
  31.6 f/min → **~13.6 h total, INSIDE the 24 GPU-h gate**; boundary
  pulled in to ~13:1x–13:4xZ.

**Steering**: none (`read` clean ×2, `history` no new reactions;
owner asleep since 00:58Z).

**Done**: tick only — babysit ×2 bracketing the @5000 save;
`queue_cli.py validate` green (depth 3, 8 open); GPUs busy ×5 +
CPU queue → `run_work_next` armed.

**Next** (`queue_cli.py next`): p6 checkpy-tiers/gpu markers (chained
work session), then p7 tee-to-logs, #6 rung-(a) pre-reg draft; molmo2
next save @7500 ~04:0xZ; draws10_t1 boundary ~13:1x–13:4xZ → frozen
reads; arm A img280 HELD.

*Updated 2026-08-07 02:0x–02:1xZ (real `date -u`) — work session (chained, bounded):
**#21 P5 LANDED — sessions know their deadline now** (commit `b3992c1`).*

**Status** (babysit 02:09Z, both green, exit 0):
- box molmo2 AR 40k — 4480/40k, loss 4.01 (−0.080 this window), probe
  **10.47@4000** (holds the low; K1 gate @10k with margin), 2.18
  s/step, vram 67.07 ≤ 71, 4 ranks + 4 GPUs ~71.6 GiB; **@5000 save
  ~02:2x–02:3xZ → next tick's duty**, endpoint ~08-08.
- local draws10_t1 — 4672/25800, window 44.7 f/min, cumulative
  30.8 f/min → **~14.0 h total, INSIDE the 24 GPU-h gate**; boundary
  pulled in to ~13:3x–14:0xZ.

**Steering**: none (`read` clean, `history` clean; owner asleep since
00:58Z).

**Done**: #21 P5 (owner-signed diff applied verbatim, commit
`b3992c1`) — the driver now appends to every session prompt
`Session start: HH:MM:SSZ; hard kill in N min. Commit and push state
comfortably before the deadline.` Sessions budget their ending
against a known zero point instead of guessing wall-clock (the
timeout-truncates-a-commit class closed by budgeting; babysit
checkpoints schedulable from the stamp). Matching one-liners in all
three prompts; oracle `tests/test_session_driver.py` runs the REAL
driver with a fake `claude` in an isolated HOME+repo and asserts the
stamped prompt tail for tick (30 min) and work (240 min). check.py
374 passed. Queue: p5 closed, draws10 boundary refreshed.

**Next** (`queue_cli.py next`): p6 gpu markers (chained work
session), then p7 tee-to-logs, #6 rung-(a) pre-reg draft; molmo2
@5000 save ~02:2x–02:3xZ next-tick duty; draws10_t1 boundary
~13:3x–14:0xZ → frozen reads; arm A img280 HELD.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; accruing since: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, local AR-100k draws10_t1
from 23:37Z — both live to their boundaries). Older dated snapshots
and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 02:0x–02:1xZ: all-CPU, 0 GPU-h — #21 P5 (owner-signed infra,
exploit-side): the signed driver diff landed — every session prompt
now carries its start time + hard-kill budget, with an end-to-end
oracle (real driver, fake `claude`, isolated HOME). Lit slice
skipped — owner-signed P-block in progress; balance on cadence.

Session 02:32–02:5xZ: all-CPU, 0 GPU-h — #21 P6 (owner-signed infra,
exploit-side): pytest gpu tier landed (strict markers, check.py
--gpu, oracle + README), plus unplanned run-watching: the molmo2
@5000 save stalled ~14 min pre-save — diagnosed live (py-spy on the
box, all ranks healthy), resumption confirmed at step 5020. Lit
slice skipped — owner-signed P-block in progress; balance on
cadence.
