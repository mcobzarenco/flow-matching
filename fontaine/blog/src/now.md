# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-07 01:56–02:0xZ (real `date -u`) — tick (babysit).*

**Status** (babysit 01:56Z, both green, exit 0):
- box molmo2 AR 40k — 4120/40k, loss 4.07 (−0.056 this window), probe
  **10.47@4000** (holds the low; descent intact, K1 gate @10k with
  margin), 2.18 s/step, vram 67.07 ≤ 71, 4 ranks + 4 GPUs ~71.6 GiB;
  **@5000 save ~02:2x–02:3xZ → next tick's duty**, endpoint ~08-08.
- local draws10_t1 — 4192/25800, window 59.1 f/min, cumulative
  30.3 f/min → **~14.2 h total, INSIDE the 24 GPU-h gate**; boundary
  pulled in to ~13:5x–14:2xZ.

**Steering**: none (`read` clean, `history -n 5` no new reactions;
owner asleep since 00:58Z).

**Done**: tick only — babysit CLI exit 0 on both runs;
`queue_cli.py validate` green (depth 4, 9 open); GPUs busy ×5 +
owner-signed CPU queue → `run_work_next` armed.

**Next** (`queue_cli.py next`): p5-deadline-stamp (chained work
session), then p6 gpu markers, p7 tee-to-logs, #6 rung-(a) pre-reg
draft; molmo2 @5000 save ~02:3xZ next-tick duty; draws10_t1 boundary
~13:5x–14:2xZ → frozen reads; arm A img280 HELD.

*Updated 2026-08-07 01:47–02:0xZ (real `date -u`) — work session (chained, bounded):
**#21 P4 LANDED — this entry is the new contract** (commit `40e782f`).*

**Status** (babysit 01:47Z, both green, exit 0):
- box molmo2 AR 40k — 3900/40k, loss 4.11, probe **10.49@3500**
  (re-descended below the 12.09 low; K1 gate: below 12.0944 by 10k),
  2.18 s/step, vram 67.07 ≤ 71, 4 ranks + 4 GPUs 71.6 GiB; @5000
  save ~02:3xZ (tick duty), endpoint ~08-08.
- local draws10_t1 — 3872/25800, window 108 f/min (fast content
  stretch), cumulative 29.9 f/min → **~14.4 h total, INSIDE the 24
  GPU-h gate**; boundary pulled in to ~14:0x–14:3xZ.

**Steering**: none (owner asleep since 00:58Z; `read` + `history`
clean at boot and close).

**Done**: #21 P4 — the now.md head-entry skeleton, applied to the
file that defines it: entries are now four labeled blocks
(Status / Steering / Done / Next; contract in work.md §4, pointer in
tick.md §7, charter now.md bullet amended), the utilization footer
slimmed to trailing-7-day figure + last 2 session notes (286 stale
lines rolled verbatim to
[the archive](archive/now-2026-08-07.md)), and `archive_now.py
--keep 3` codified at every work-session close (was habit-only).
Queue hygiene in the same commit: p4 closed, molmo2 watch-title
cleared, draws10 boundary refreshed.

**Next** (`queue_cli.py next`): p5-deadline-stamp (driver stamps a
deadline the prompts can read, minutes), then p6 gpu markers, p7
tee-to-logs, #6 self-subgoal rung-(a) pre-reg draft; draws10_t1
boundary ~14:0x–14:3xZ → frozen reads (Δ_AR vs 5.8026, fairness vs
−1.258, family vs 5.365); molmo2 @5000 save ~02:3xZ tick duty; arm A
img280 HELD (fresh owner go required).

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; accruing since: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, local AR-100k draws10_t1
from 23:37Z — both live to their boundaries). Older dated snapshots
and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 01:47–02:0xZ: all-CPU, 0 GPU-h — #21 P4 (owner-signed infra,
exploit-side): the now.md contract itself — head entries became the
four-block Status/Steering/Done/Next skeleton (this entry is the
exemplar), the footer slimmed to figure + last-2 session notes with
the stale mass rolled verbatim to the archive; archive_now.py
--keep 3 codified at every close. Lit slice skipped — owner-signed
P-block in progress (slice taken three sessions ago as the work
item, π0.5); balance on cadence.

Session 02:0x–02:1xZ: all-CPU, 0 GPU-h — #21 P5 (owner-signed infra,
exploit-side): the signed driver diff landed — every session prompt
now carries its start time + hard-kill budget, with an end-to-end
oracle (real driver, fake `claude`, isolated HOME). Lit slice
skipped — owner-signed P-block in progress; balance on cadence.
