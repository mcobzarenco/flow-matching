# Now








*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-19 01:39–01:4xZ (real `date -u` at write: 01:43) —
tick: **babysit + a success-count method fix — the battery has its
FIRST success (seed 29), so the running read is 1/37, not 0/x.
Successful episodes break out of the episode loop early on
`sim.success()` (`sim/rollout_sim.py:444`); the log signature of a
success is an episode whose last replan is < 29, NOT a small final
distance. Prior ticks' "success requires near-zero benchy→disk" proxy
was wrong — `sim.success()` fires within the disk radius (~5 cm)
when upright + still + released, so seed 29's early break at replan 8
(last printed distance 4.5 cm) is a placement. The 0/22 and 0/10
counts in earlier entries were numerically right (no early breaks
existed yet) but the method would have missed one.***

**Status**: `pdnorm_endpoint_battery` LIVE — babysit exit 0 at 01:40:
2 procs, GPU 12.7 GiB / 28–41% duty (6-sample; sim-rollout profile),
host RAM 192 GiB available. Progress: 38 seeds started (seed 37 in
flight) in ~57 min, window 0.7 f/min ≈ disc baseline; replans steady
~540–560 ms. Corrected running read: **1/37 completed** (seed 29;
per-seed mins otherwise 4.2 cm seed 9 / 4.5 seed 29 / 5.2 seed 30 —
the near-misses cluster just above the disk radius). Trend still
firmly early-convict vs the ≤10/100 line (exoneration needs 19 of
the remaining 63), but the frozen grid reads only at 100/100 — no
mid-run action. Grid anchor verified safe: the disc1000 11/100
baseline came from `reconstruct_sim100_from_logs.py`, which parses
the script's own end-of-run summary table (`success_tick` column) —
same criterion, no undercount there. Leg-1 rc still ~02:4x–03:0xZ,
then panel ~0.5 GPU-h. GPU-h gate 5.0, cumulative projection 0.9.
Queue green depth 2 (15 open; both gpu-gated).

**Steering**: none — read empty, inbox empty; history shows no new
reactions (all three 👍 previously recorded).

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + 6-sample GPU util standing checks, queue validate; chased
seed 29's silent early termination through `rollout_sim.py` /
`so101_sim.py` to the success-break, fixed the mid-run counting
method, and audited the baseline reconstruction against the same
bug (clean). No post (quiet interval; the 100/100 verdict post
carries the corrected count and belongs to the session holding the
read).

**Next**: the tick that catches leg-1 rc (~02:4x–03:0xZ) reads
sim100 through the frozen grid — **count successes as episodes with
last replan < 29 plus any summary-table `success_tick`, never by
final distance** — and arms `run_work_next` for the verdict battery
(paired read vs disc1000 11/100, ladder `--endpoint` restamp,
truthfit rewear, pdnormendpoint report, verdict post) with
best-save flexibility LIVE: endpoint-3000 (probe 6.17) vs **step
2000 @ 5.47**. CPU queue EMPTY → `run_work_next` NOT armed this
tick.*

*Updated 2026-08-19 01:18–01:2xZ (real `date -u` at write: 01:20) —
tick: **quiet mid-battery babysit ~20 min after the 00:58 entry — leg
1 sim100 healthy at one-third mark; 0 successes in 22 completed
episodes (per-seed min benchy→disk 4.2 cm — no placement anywhere),
early-convict trend firm but the frozen grid reads only at 100/100 —
no mid-run action.***

**Status**: `pdnorm_endpoint_battery` LIVE — babysit exit 0 at 01:19:
2 procs, GPU 12.7 GiB / 28–35% duty (6-sample; sim-rollout profile
unchanged), host RAM 192 GiB available. Progress: 23 seeds started
(seed 22 in flight) in ~35 min ≈ 0.65 ep/min — tracking the disc
baseline 0.76 net of model load; replans steady ~550 ms. Raw-log
success read: 0/22 — best any seed managed was 4.2 cm (seed 9); most
final distances 8–25 cm. Grid anchors unchanged (≥20 exonerates /
≤10 convicts / 11–19 ambiguous; baseline demosonly 11/100). Leg-1 rc
projects ~02:4x–03:0xZ (registry boundary; disc baseline ~2.2
GPU-h), then panel leg ~0.5 GPU-h. GPU-h gate 5.0, cumulative
projection 0.6. Queue green depth 2 (15 open; both gpu-gated).

**Steering**: none — read empty (not even cursor catch-up), inbox
empty; history shows no new reactions (all three 👍 previously
recorded).

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + 6-sample GPU util standing checks, queue validate, raw-log
per-seed distance sweep (23 seeds, min/final distances tabulated —
the success-count method: a success requires a placement, i.e.
near-zero benchy→disk; none present). No post (quiet interval;
verdict post belongs to the session holding the 100/100 read).

**Next**: unchanged from 00:58 — the tick that catches leg-1 rc
(~02:4x–03:1xZ) reads sim100 through the frozen grid and arms
`run_work_next` for the verdict battery (paired read vs disc1000
11/100, ladder `--endpoint` restamp, truthfit rewear, pdnormendpoint
report, verdict post) with **best-save flexibility LIVE**:
endpoint-3000 (probe 6.17) vs **step 2000 @ 5.47**. CPU queue EMPTY
→ `run_work_next` NOT armed this tick.*

*Updated 2026-08-19 00:58–01:0xZ (real `date -u` at write: 01:02) —
tick: **first babysit of the endpoint battery — the chained work
session landed the endpoint (train COMPLETE 3000/3000, probe
6.17@3000, curve closed, no retrace; posted in-channel 00:46, id
1539435263321833546) and launched the battery (unit
fontaine-pdnorm-endpoint-battery, 00:44:37Z). Leg 1 sim100 healthy
at first poll; no read licensed before 100/100.***

**Status**: `pdnorm_endpoint_battery` LIVE — babysit exit 0 at 00:58:
2 procs, GPU 12.7 GiB / 28–40% duty cycle (6-sample; sim-rollout
profile, matches the disc-baseline battery shape — not a training
input-starvation case), host RAM 192 GiB available. Rate: window
0.6/min startup-contaminated; raw-log confirm at 01:01 — 11 episodes
started in ~16.5 min (~0.7 ep/min net of model load ≈ disc baseline
0.76; replan cadence steady ~540 ms). **0 successes in the first ~10
episodes** — early-convict-ish vs the ≤10/100 line, but the frozen
grid reads only at 100/100 (≥20 exonerates the mix / ≤10 convicts /
11–19 ambiguous; baseline demosonly cell 11/100) — no mid-run
action. Leg 1 rc projects ~03:1x–03:3xZ, then panel leg ~0.5 GPU-h,
then the CPU verdict tail. GPU-h gate 5.0, projection 0.2. Queue
green depth 2 (15 open; both gpu-gated).

**Steering**: none — read surfaced only our own 00:46 endpoint post
(cursor catch-up), inbox empty; history shows no new reactions (the
three 👍 were recorded in prior ticks).

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + 6-sample GPU util standing checks, raw-log rate confirm
(direct log read, not via babysit output), queue validate. Verified
the registry pgrep (`launch_pdnorm_endpoint_battery`) can't collide
with session wait-loop cmdlines (08-18 arm-wait incident class). No
post (nothing new since the 00:46 endpoint post; the verdict post
belongs to the session holding the sim100 read).

**Next**: the tick that catches leg-1 rc (~03:1x–03:3xZ) reads sim100
through the frozen grid and arms `run_work_next` — the verdict
battery is CPU-hours (paired read vs disc1000 11/100, ladder
`--endpoint` restamp, truthfit rewear, pdnormendpoint report, verdict
post) with **best-save flexibility LIVE**: endpoint-3000 (probe 6.17)
vs **step 2000 @ 5.47**. CPU queue EMPTY now → `run_work_next` NOT
armed this tick.*

## Utilization footer

Session 2026-08-19 01:39–01:4xZ (tick; 0 GPU-h new — endpoint
battery leg 1 live since 00:44:37Z, ~0.9 GPU-h elapsed of gate 5.0):
**babysit + success-count method fix — babysit exit 0: 2 procs, GPU
12.7 GiB / 28–41% duty (sim-rollout profile), RAM 192 GiB; 38 seeds
started in ~57 min, window 0.7 f/min, replans ~540–560 ms; the
battery's FIRST success found (seed 29, early break at replan 8):
successful episodes break the loop on `sim.success()` (within disk
radius + upright + still + released), so the log signature is last
replan < 29, not a small final distance — running read corrected to
1/37; prior 0/22 counts were numerically right but the near-zero
proxy was wrong; baseline 11/100 reconstruction audited clean (it
parses the summary-table success_tick column); Discord fully quiet
(read empty, inbox empty, no new reactions)** — CPU queue empty,
`run_work_next` NOT armed; leg-1 rc ~02:4x–03:0xZ, that tick reads
sim100 through the frozen grid and arms the verdict-battery work
session (best-save flexibility live: endpoint-3000 probe 6.17 vs
step 2000 @ 5.47).

Session 2026-08-19 01:18–01:2xZ (tick; 0 GPU-h new — endpoint
battery leg 1 live since 00:44:37Z, ~0.6 GPU-h elapsed of gate 5.0):
**quiet mid-battery babysit — babysit exit 0: 2 procs, GPU 12.7
GiB / 28–35% duty (sim-rollout profile), RAM 192 GiB; 23 seeds
started in ~35 min ≈ 0.65 ep/min (disc baseline 0.76 net of load),
replans ~550 ms; raw-log per-seed sweep: 0 successes in 22 completed
episodes (min benchy→disk anywhere 4.2 cm — no placement),
early-convict trend firm but no read before 100/100 per the frozen
grid; Discord fully quiet (read empty, inbox empty, no new
reactions)** — CPU queue empty, `run_work_next` NOT armed; leg-1 rc
~02:4x–03:1xZ, that tick reads sim100 through the frozen grid and
arms the verdict-battery work session (best-save flexibility live:
endpoint-3000 probe 6.17 vs step 2000 @ 5.47).

Trailing-7-day GPU-hours on experiments / total (window 2026-08-10
00:00Z → 2026-08-17 19:45Z; rebased 08-17 from per-run prune records
+ archive session notes — receipts in
`fontaine/notes/utilization-rebase-2026-08-17.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~80.0 / ~80.2**
(incl. the discriminator at ~1.0 in-window; run COMPLETE 08-18
00:42Z at ~5.8 total — post-window ledger row landed in the 00:49
work-session note above, ~4.8 rolls into the next window), box **~250 /
~254 FINAL** (box killed by owner 08-17 ~15:xxZ; er_60k pro-rated
~147 in-window of its ~153; sim100 eval ~5 is the one estimated
figure). Older dated snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
