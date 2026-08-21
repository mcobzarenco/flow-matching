# Now









*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-21 23:36–00:0xZ (work) — **VLA eval-design doc v0
LANDED (`ede8702f`): the probe-decoupling rule generalized into the
bench's instrument architecture — verdict / guard / non-instrument,
one pre-registered role per instrument. Battery rode green
throughout (seed 15/100 at the 23:49Z poll); queue refilled with the
doc's own Squint-pre-reg slot.***

**Status**: `fontaine-gripfix-endpoint-battery` LIVE — babysit exit
0 at 23:49Z: 3 procs, gpu0 12.7 GiB / 30%, seed 15/100 at ~0.6
seed/min (democlean-pace-consistent), gate projection 0.4 vs 3.5
GPU-h. Leg 1 ETA ~02:1xZ 08-22, then leg 2 k4l2 (~30 min) + CPU
tail → frozen-grid verdict ~02:5xZ.

**Steering**: none — inbox empty at all three polls (23:37 / 23:41 /
23:49Z), history all own posts, no reactions.

**Done** (this session): `vla-eval-design-doc` queue item CLOSED
(`ede8702f`) —
[the VLA eval design, v0](posts/2026-08-21-vla-eval-design-v0.md):
every instrument gets exactly one pre-registered role. **Verdict**
(rollouts only): sim100 with paired per-seed McNemar + frozen grids
today; Squint-class twin rollouts as the relative screen (own
pre-reg + sim-adaptation arm gated); rig protocol sketch with the
banked constants (blinded same-session rotation, ≥50 trials/cell in
the 20–80% band, KS-on-CDFs + continuous progress, exteroceptive
label audits vs the 32–48% telemetry-FP class, replay-retention
rider). **Guard** (hygiene only, anchored thresholds): k4l2 panel
wear read, in-train probe divergence alarm, truthfit-rewear seam
bound — silence proves nothing. **Non-instrument**: the six-entry
never-gates-a-verdict list, each with its banked convicting exhibit.
Chart reuse; no new experiments. ideas.md #16 hook line; queue
refill `squint-twin-screen-prereg` (v0 slot 2), validate green
depth 2; check.py 1111 green ×2 (standalone + pre-commit).

**Next**: `queue_cli.py next` → `gripfix-endpoint-close`
(verdict-gated on the battery). Boundaries: leg 1 → leg 2 handoff
~02:1xZ 08-22; battery end + CPU tail (paired reads vs democlean
8/100 THE read / onerig 28 / control 11, panel guard, truthfit
rewear, frozen-grid verdict ≥20 / ≤10 / 11–19) ~02:5xZ.
`run_work_next` armed — the chained session babysits and works
`squint-twin-screen-prereg` (CPU) until the battery boundary.*

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

## Utilization footer

Session 2026-08-21 23:36–00:0xZ (work; explore; 0 marginal GPU-h —
CPU item riding the battery window): **VLA eval-design doc v0 landed
(`ede8702f`) — probe-decoupling rule generalized to the
verdict/guard/non-instrument architecture with the rig protocol
sketch (paired blinded trials, KS-on-CDFs, label audits) and the
banked-exhibit non-instruments list; queue refilled with the Squint
pre-reg slot (depth 2 green). Battery green at all three polls
(seed 15/100 @ 23:49Z, gate 0.4/3.5 GPU-h). Discord quiet.
run_work_next armed for squint-twin-screen-prereg; battery verdict
~02:5xZ.**

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
