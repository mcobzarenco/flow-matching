# Now








*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-21 23:34–23:4xZ (tick) — **first tick poll on the
gripfix endpoint battery: all green. Leg 1 sim100 at seed ~6/100,
~105 s/seed incl. startup → ETA ~02:2xZ 08-22 (democlean pace),
gate projection 0.2 vs 3.5 GPU-h. Discord quiet; `run_work_next`
armed for `vla-eval-design-doc` to ride the window.***

**Status**: `fontaine-gripfix-endpoint-battery` LIVE — babysit exit
0 at 23:34Z: 3 procs, gpu0 12.7 GiB / 38% (replan-bound eval, normal
for sim100), progress count 6 (bare-count = seeds done). Pace ~105
s/seed from launch 23:23:39Z — startup-inclusive, consistent with
the democlean ~90 s/seed steady pace; leg 1 ends ~01:5x–02:2xZ, then
leg 2 k4l2 panel (~30 min). Cumulative projection 0.2 GPU-h vs the
3.5 gate; cell honest total ~13.6 train + ~3 battery vs 17.

**Steering**: none — inbox empty; the one new `read` message was our
own 23:27Z launch post (BOT), history all own posts, no reactions.

**Done** (this tick): babysit poll (green, judged healthy), Discord
read + history, queue validate green (depth 2, 14 open, head
`gripfix-endpoint-close` in-flight), `run_work_next` armed 23:34Z.

**Next**: chained work session executes `vla-eval-design-doc` (CPU,
rides the battery window). Boundaries: leg 1 → leg 2 handoff
~01:5x–02:2xZ 08-22; battery end + CPU tail (paired reads vs
democlean 8/100 THE read / onerig 28 / control 11, panel guard,
truthfit rewear, frozen-grid verdict ≥20 / ≤10 / 11–19) ~02:5xZ.*

*Updated 2026-08-21 23:10–23:2xZ (tick) — **back online after a ~14 h
harness outage (usage credits exhausted 09:16Z→~22:0xZ, 13 sessions
died 429 at boot); gripfix ran unattended through all of it and
COMPLETED clean: 3000/3000, endpoint ~22:07Z, probe closed 4.88@3000.
Endpoint state verified; `gripfix-endpoint-close` chained.***

**Status**: `grasp_sft_v2_joint_pdnorm_gripfix` TRAIN COMPLETE —
3000/3000, ~13.6/17 GPU-h, wandb synced 22:08Z. Probe curve closed
smooth (5.06@2000 → 4.97 → 4.96 → 4.89 → 4.88@3000, no convicted-cell
elevation; record-only, sim100 owns the verdict). All 6 saves verified
(step_003000 full weights + optimizer; pruner kept-latest worked, unit
inactive). GPU 0 MiB, no strays (babysit's "1 proc" = the known pgrep
self-match). Disk 74G free, RAM 196G avail. Battery NOT yet run.

**Steering**: none from the owner — inbox empty; the 13 new `read`
messages were all harness BOT alerts (session exit 1, 09:17Z→22:08Z).
Root cause from the 220818Z tick log: 429 `out_of_credits`
(seven-day overage rejected, resetsAt 22:00Z 08-22 in the payload;
credits evidently topped back up — this 23:10Z session runs). Every
boundary check during the outage (step-500…3000 saves, step-1000
drift read) was missed; all were record-only/hygiene, the run never
needed intervention. Outage + completion posted in-channel 23:1xZ
(id 1540498974002258013).

**Done** (this tick): Discord read (13 alerts) + history; babysit
(exit 1 = post-completion liveness trip, judged benign); endpoint
verification (log tail + wandb footer, 6 save dirs, step_003000
full, pruner log, GPU/procs); babysit.toml gripfix entry PRUNED
with the outage post-mortem; queue validate green (depth 2, 14
open, head `gripfix-endpoint-close`); `run_work_next` armed 23:12Z.

**Next**: chained work session executes `gripfix-endpoint-close` —
sim100 battery (democlean pattern: stand-ins pin, stats-repo
grasp_demos_v2/merged) + k4l2 panel + guard vs disc-1000 npz +
paired reads vs democlean 8/100 (THE read) / onerig 28/100 /
control 11/100; verdict through the frozen grid (≥20 / ≤10 / 11–19).
GPU free; policy-server check at launch. `vla-eval-design-doc`
stays queued behind it.*

*Updated 2026-08-21 09:12–09:1xZ (tick) — **third poll on gripfix,
all green: step 140/3000, loss 0.9418 falling smoothly (2.88 @30 →
2.21 @40 → 0.94 @140), 15.811 s/step on the ~16 expectation, vram
62.24 GiB vs the 75 gate, GPU 100%. Discord quiet; exited fast.***

**Status**: `grasp_sft_v2_joint_pdnorm_gripfix` LIVE and healthy —
babysit exit 0 at 09:12Z: 6 procs, step 140, loss 0.9418, 15.811
s/step → ~12.6 h to 3000, endpoint ~22:0x–22:2xZ. RAM 90G available
(pre-first-save plateau per the democlean anchor), disk 169G free vs
~124G peak. No boundary in this window; first save + pruner-log
check at step 500 (~10:3xZ).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts (latest: the 09:02Z probe-decoupling note), no reactions.

**Done** (this tick): babysit poll (green, judged healthy — util
100% at sample, no starvation), RAM/disk check, queue validate green
(depth 2, 14 open), `run_work_next` confirmed armed (09:11Z, from
the work session's close).

**Next**: chained work session executes `vla-eval-design-doc` (CPU,
rides the GPU-busy window). Boundaries unchanged: saves every 500
(pruner-log check each, first ~10:3xZ), step-1000 drift read ~13:0xZ
(record-only), endpoint ~22:0x–22:2xZ → `gripfix-endpoint-close`.*

## Utilization footer

Session 2026-08-21 23:34–23:4xZ (tick; 0 marginal GPU-h — battery
riding): **first tick poll on the gripfix endpoint battery, all
green — leg 1 sim100 at seed ~6/100, ~105 s/seed incl. startup
(democlean-pace-consistent), ETA ~01:5x–02:2xZ 08-22, gpu0 12.7
GiB / 38% (replan-bound, normal), gate projection 0.2 vs 3.5
GPU-h. Discord fully quiet (read = own launch post only, inbox
empty, no reactions); queue green depth 2 (14 open);
run_work_next armed 23:34Z for vla-eval-design-doc (CPU, rides
the battery window). Next boundary: leg 1 → leg 2 handoff, then
CPU tail + frozen-grid verdict ~02:5xZ.**

Session 2026-08-21 23:10–23:2xZ (tick; 0 marginal GPU-h — GPU idle
post-endpoint): **outage discovered + gripfix endpoint verified —
harness was down ~14 h (429 out_of_credits, 09:16Z→22:08Z, 13
sessions died at boot; credits reset, this tick first through).
gripfix COMPLETED unattended 3000/3000 ~22:07Z (~13.6/17 GPU-h,
probe 4.88@3000 record-only, all saves verified, pruner clean, GPU
0 MiB). babysit.toml pruned with post-mortem; outage + completion
posted in-channel; run_work_next armed 23:12Z for
gripfix-endpoint-close (queue head; battery + paired reads + frozen
verdict grid). Note: ~13 h of the GPU-busy window burned with no
CPU work executed — the credit exhaustion, not idling, was the
cause; queue depth still 2.**

Trailing-7-day GPU-hours on experiments / total (window 2026-08-12
00:00Z → 2026-08-19 08:45Z; rolled 08-19 from the 08-17 rebase +
prune records + archive session notes — receipts in
`fontaine/notes/util-window-roll-2026-08-19.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~84.1 / ~85.5**
(retained 08-12→stamp ~57.5 + post-stamp ~28.1: discriminator
roll-in ~4.8, pdnorm screen-wide ~15.9 train+battery, joint-probe
legs 3+4 ~3.9 incl. leg 4 live at stamp; ops/loss ~1.4 =
discriminator attempt-1 OOM + smokes). Local-only from this roll —
the box was killed 08-17 (~106 box GPU-h fall in-window for the
record; final box history in
`fontaine/notes/utilization-rebase-2026-08-17.md`). Older dated
snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
