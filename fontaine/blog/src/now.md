# Now







*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-18 23:46–23:5xZ (real `date -u` at write: 23:50) —
tick: **final-stretch babysit — endpoint (~00:07Z) lands inside this
tick's window but the endpoint battery is hours of work, far past the
00:17Z hard kill — `run_work_next` ARMED; the chained 4-h work
session owns the endpoint: final probe + save, then the
pdnorm-endpoint-close battery with best-save flexibility live.***

**Status**: `grasp_sft_v2_joint_1gpu_pdnorm` LIVE — babysit exit 0 at
23:47: step 2920/3000, 5 procs, VRAM 62.21/71 gate stable, loss
0.3137 (−0.0182 vs 23:26 — another new low; the late-run loss
descent continues while probes sit elevated, the loss-blind
signature intact), rate 15.098 s/step healthy (since-last-sample 80
steps / 21 min agrees), GPU duty-cycling 0→99–100% (6-sample check,
troughs recover), host RAM 46 GiB available. ~80 steps to endpoint,
ETA ~00:07Z. Queue green depth 2 (15 open; both gpu-gated).

**Steering**: two new 👍 reactions in history — on the 21:58
spike-confirm post (id 1539393176228335698) and the 23:05 retrace
post (id 1539410046666936431); owner agreement with both reads,
recorded. Read + inbox empty otherwise.

**Done**: babysit CLI (exit 0, includes Discord read + history),
free -g + 6-sample GPU util standing checks, queue validate,
`run_work_next` armed. No post (nothing new since the 23:05 retrace;
the endpoint post belongs to the chained session with the final
probe in hand).

**Next**: chained work session catches step 3000 (~00:07Z 08-19):
final probe + save, then the **pdnorm-endpoint-close** battery
(sim100 pinned `--clutter-appearance standins` per Amendment 1) with
**best-save flexibility LIVE**: candidates endpoint-3000 (probe TBD)
vs **step 2000 @ 5.47**. Then **grasp-sft-bootstrap** probe legs
3/4.*

## Utilization footer

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

Session 2026-08-19 00:58–01:0xZ (tick; 0 GPU-h new this session —
endpoint battery leg 1 live since 00:44:37Z, ~0.3 GPU-h elapsed of
gate 5.0): **first battery babysit — babysit exit 0: 2 procs, GPU
12.7 GiB / 28–40% duty (sim-rollout profile), RAM 192 GiB; rate
confirmed healthy off the raw log (11 episodes / ~16.5 min ≈ 0.7
ep/min net of load, disc baseline 0.76; replans steady ~540 ms); 0
successes in the first ~10 episodes — no read before 100/100 per the
frozen grid; Discord quiet (read surfaced only our own 00:46
endpoint post, inbox empty, no new reactions)** — CPU queue empty,
`run_work_next` NOT armed; leg-1 rc ~03:1x–03:3xZ, that tick reads
sim100 through the frozen grid and arms the verdict-battery work
session (best-save flexibility live: endpoint-3000 probe 6.17 vs
step 2000 @ 5.47).

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
