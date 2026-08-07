# Now







*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-07 02:51–03:2xZ (real `date -u`) — work session (chained,
bounded): **#21 P7 LANDED — home-dir & ctrl lifecycle** (commit
`914d413`). **That closes the full owner-signed #21 batch, P1–P7.***

**Status** (babysit 03:07Z, both green):
- box molmo2 AR 40k — 5580/40k, loss 3.85 (−0.09 this window), probe
  **9.24@5500** (new low, sub-10 ×3; K1 gate ≤12.0944 by 10k with
  wide margin), 2.18 s/step, vram 67.07 ≤ 71, 9 procs / 4 ranks;
  **@7500 save ~04:1x–04:2xZ → next tick watches for a repeat of the
  @5000 slow-save**; endpoint ~08-08.
- local draws10_t1 — 6752/25800, window 41.1 f/min, cumulative 32.2
  f/min → **~13.3 h total, INSIDE the 24 GPU-h gate**; boundary
  pulled in to ~13:0x–13:2xZ → frozen reads.

**Steering**: none (`read` clean at boot, mid-session, and close;
owner asleep since 00:58Z).

**Done**: #21 P7 (commit `914d413`) — `tidy_home.py` (loose `~`
files → dated attic + manifest; never deletes, never touches dirs/
dotfiles/open files (/proc fd scan)/files <2d — the live draws10 tee
log was correctly skipped in the live dry run) and
`refresh_ctrl.sh` (box control checkout delete-and-refreshed from
`git archive HEAD`, prior snapshot renamed aside; writes
`CTRL_SOURCE_COMMIT`). Executed live: box ctrl now stamped
`fa3048eb`, old snapshot + outputs preserved at
`ctrl.prev-20260807T025826Z`; `~/logs/` created both machines;
charter §5 step 3 amended (tee targets → `~/logs/`). 7 new oracles
run the REAL scripts in isolated homes/repos; check.py 385 passed.
**Deviation, stated:** the box `~` sweep was NOT applied — every
movable file is an owner-era mainline artifact and charter
Loaned-compute makes those READ-ONLY without an explicit all-clear;
asked in-channel, queued under `owner_hold`
(`box-home-sweep`). Local sweep: legit no-op (everything <2d old).
ideas #21 marked closed.

**Next** (`queue_cli.py next`): #6 rung-(a) self-subgoal pre-reg
draft (chained work session), then #19 AR sampled-draws instrument
(new queue item — wanted before the molmo2 endpoint ~08-08); molmo2
@7500 save ~04:1x–04:2xZ (slow-save watch); draws10_t1 boundary
~13:0x–13:2xZ → frozen reads; arm A img280 + box-home-sweep HELD.

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

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; accruing since: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, local AR-100k draws10_t1
from 23:37Z — both live to their boundaries). Older dated snapshots
and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 02:32–02:5xZ: all-CPU, 0 GPU-h — #21 P6 (owner-signed infra,
exploit-side): pytest gpu tier landed (strict markers, check.py
--gpu, oracle + README), plus unplanned run-watching: the molmo2
@5000 save stalled ~14 min pre-save — diagnosed live (py-spy on the
box, all ranks healthy), resumption confirmed at step 5020. Lit
slice skipped — owner-signed P-block in progress; balance on
cadence.

Session 02:51–03:2xZ: all-CPU, 0 GPU-h — #21 P7 (owner-signed infra,
exploit-side): home-dir & ctrl lifecycle landed, closing the full
P1–P7 signed batch; box ctrl checkout stamped live
(`CTRL_SOURCE_COMMIT` = `fa3048eb`), box `~` sweep held on the
charter's Loaned-compute READ-ONLY rule (owner asked). Lit slice
TAKEN (~20 min, first since the π0.5 deep-read): LabVLA — a third
independent group ships the KI-joint stage-2 recipe (banked to #4,
feeds tomorrow's attachment decision); Hi-VLA systematic study —
explicit subgoals' gain concentrates on long horizon, self-generated
subgoals untested there (banked to #6, shapes the rung-(a)
pre-reg).
