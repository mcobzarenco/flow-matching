# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-23 03:22–03:3xZ (tick) — **ch0fix healthy + a disk
projection CORRECTED and acted on: `prune_superseded_optimizers` is
keep=2 and runs strictly AFTER publish (stage-then-prune), so the
01:18 trough projection (≈82→34G, reclaim-before-staging) was wrong —
real troughs from the measured 84G floor ran ~40/28/16/**4G**, the
~4G at the step-3000 ENDPOINT staging (leg B ENOSPC class, aimed at
the one save the rung needs). Mitigation: optimizer.pt is dead
weight on this lineage (no-resume anchor — recovery is full
retrain), so superseded optims are manually pruned each post-save
tick (keep-1, newest save untouched): optim@500 deleted → 115G free,
projected troughs ~72/60/48/36G all safe; registry anchor rewritten,
posted in-channel (id 1540924939014504519). Run itself: step
1060/3000, loss 0.485 (first window uptick +0.042 — noise-class),
vram 62.24/71, 16.4 s/step window; probe @1000 = 6.84 (4.61 → 5.24 →
5.97 → 6.84, within-lineage record only). ETA ~12:1xZ 08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
1060/3000, babysit exit 0 (liveness 5 procs, gpu0 66581MiB/100%
util), vram 62.24 stable vs the 71 gate. Loss window 0.4429@910 →
0.485@1060 (+0.042) — the first non-monotone sampled read; instant
log values at loss ~0.45–0.49 wobble at this scale, judged
noise-class (watch, not act). Rate 16.388 s/step window (3.7
steps/min) — inside the judged 14.7–24.5 band. Probe eval_chunk_mae
6.84@1000 continues the monotone rise; per the pre-reg anchor the
decision read stays the endpoint sim100 battery vs democlean 8/100.
**Disk — corrected projection**: step_001000 save landed ~03:06Z
(42G; async, 62.9 s behind the boundary) → 84G free, matching the
old projection's number but NOT its mechanism — the code
(`bijou/train/saving.py` keep=2, prune strictly post-publish) never
reclaims before staging, so the old troughs were unreachable and the
real endpoint-staging trough was ~4G. Superseded optim@500 deleted
(32G, dead weight under the no-resume anchor) → **115G free**;
standing per-tick rule banked in the registry: after each save
verifies, delete the newly-superseded optimizer.pt (troughs
~72/60/48/36G; even a missed tick still clears a save with ≥28G).
Host RAM available 48G — stable fourth read. ETA ~8.8 h at the
window rate → done ~12:1xZ 08-23 → sim100 endpoint battery; its
verdict mechanically selects the rung-3 branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), step-1000 save verified
on disk, trough-projection audit against the actual saving.py
semantics (keep=2, stage-then-prune — 01:18 projection corrected),
superseded optim@500 pruned (+32G → 115G), registry DISK anchor
rewritten with the keep-1 per-tick rule, correction posted
in-channel, probe @1000 + loss-uptick reads judged, RAM read,
Discord read + history, queue validate green (depth-1 stated reason
— rung3-exec verdict-gated, no CPU items, so `run_work_next` stays
unarmed), now.md keep-3 + footer rolls (01:18 entry → archive).

**Next**: post-save ticks execute the keep-1 optim prune (next:
optim@1000 becomes superseded when the step-1500 save verifies,
~05:2xZ). Otherwise nothing fires before the ch0fix boundary — train
done ~12:1xZ 08-23 → battery ~3 GPU-h → rung-2 verdict banks →
`carrier-hunt-rung3-exec` selects and launches the branch same
session (fit smoke → launch, ONE dataset delta, seed 0; 11–19 fires
neither branch, owner escalation).*

*Updated 2026-08-23 02:41–02:4xZ (tick) — **routine ch0fix poll:
healthy — step 910/3000, loss 0.4429 monotone down, vram 62.24/71,
rate 14.834 s/step window (3.9 steps/min since last sample). Probe
@750 row now written: eval_chunk_mae 4.61@250 → 5.24@500 → 5.97@750 —
monotone rise, within-lineage record only (pdnorm-rescale confound
banked; the decision read stays the endpoint sim100 battery). Disk
126G flat pre-save (step 1000 save ~20 min out, trough projection
governs); RAM available 48G unchanged. ETA ~8.6 h → ~11:1x–12:4xZ
08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
910/3000, loss 1.91→0.4429 monotone down (−0.0453 since step 750),
babysit exit 0 (liveness 5 procs, gpu0 66581MiB/100% util), vram
62.24 stable vs the 71 gate. Probe eval_chunk_mae 4.61@250 →
5.24@500 → 5.97@750: a monotone within-lineage rise while train loss
falls monotone — the familiar probe-decoupling shape, and the banked
pdnorm-rescale confound (ch0 ×2.755 changes the metric's units)
blocks any cross-lineage read; per the pre-reg anchor the decision
read is the endpoint sim100 battery vs democlean 8/100, so this is a
recorded trajectory, not a signal to act on. Disk 126G free — flat
since step_000500, the step 1000 save (~20 min after this poll)
stages ~44G then prunes per the banked trough projection (≈82G
trough, safe); next tick sees the post-save state. Host RAM
available 48G — stable third read, trend-watch holds. ETA: ~8.6 h at
the window rate → done ~11:1x–12:4xZ 08-23 → sim100 endpoint battery
vs democlean 8/100; its verdict mechanically selects the rung-3
branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), probe @750 row read +
trajectory judged (within-lineage rise, non-actionable per the
banked confound + endpoint-read anchor), disk pre-save read (126G
flat, matches projection), RAM trend read (48G stable), Discord read
+ history, queue validate green (depth-1 stated reason — rung3-exec
verdict-gated, no CPU items, so `run_work_next` stays unarmed),
now.md keep-3 + footer rolls (00:37 entry → new archive page
now-2026-08-23.md, index + SUMMARY updated).

**Next**: nothing fires before the ch0fix boundary — train done
~11:1x–12:4xZ 08-23 → battery ~3 GPU-h → rung-2 verdict banks →
`carrier-hunt-rung3-exec` selects and launches the branch same
session (fit smoke → launch, ONE dataset delta, seed 0; 11–19 fires
neither branch, owner escalation).*

*Updated 2026-08-23 01:59–02:0xZ (tick) — **routine ch0fix poll:
healthy — step 750/3000 (quarter mark), loss 0.4882 monotone down,
vram 62.24/71, rate 14.843 s/step window (4.1 steps/min since last
sample). Disk 126G free — no new save since step_000500, matches the
banked trough projection exactly. RAM available 48G (stable vs 49
last tick — trend-watch holds, no drift). ETA ~11:2x–12:4xZ 08-23
unchanged.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
750/3000, loss 1.91→0.4882 monotone down (−0.0366 since step 580),
babysit exit 0 (liveness 5 procs, gpu0 66581MiB/52% util at poll
instant), vram 62.24 stable vs the 71 gate. Probe eval_chunk_mae
still 4.61@250 → 5.24@500 (the @750 row not yet written at poll
time; within-lineage record only per the banked pdnorm-rescale
confound). Disk 126G free vs the ≥90 anchor — flat since the
step_000500 save, next save at step 1000 (~1 h) stages ~44G then
prunes; the 01:18 trough projection (≈82→34G, all safe) governs, no
re-alarm. Host RAM available 48G — stable, buff/cache absorbing.
ETA: ~9.3 h at the window rate → done ~11:2x–12:4xZ 08-23 → sim100
endpoint battery vs democlean 8/100; its verdict mechanically
selects the rung-3 branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), disk read vs the banked
trough projection (126G flat — matches, no re-alarm), RAM
trend-watch read (48G stable), Discord read + history, queue
validate green (depth-1 stated reason — rung3-exec verdict-gated, no
CPU items, so `run_work_next` stays unarmed), now.md keep-3 + footer
rolls.

**Next**: nothing fires before the ch0fix boundary — train done
~11:2x–12:4xZ 08-23 → battery ~3 GPU-h → rung-2 verdict banks →
`carrier-hunt-rung3-exec` selects and launches the branch same
session (fit smoke → launch, ONE dataset delta, seed 0; 11–19 fires
neither branch, owner escalation).*

## Utilization footer

Session 2026-08-23 03:22–03:3xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **ch0fix healthy — step 1060/3000, loss 0.485 (first
window uptick +0.042, noise-class), vram 62.24/71, 16.4 s/step; probe
@1000 = 6.84 (monotone rise, within-lineage only). DISK PROJECTION
CORRECTED: saving.py is keep=2 stage-then-prune, so the 01:18
troughs (82→34G) were unreachable — real endpoint-staging trough was
~4G (ENOSPC class). Superseded optim@500 deleted (dead weight, no
resume on this lineage) → 115G free; per-tick keep-1 optim prune
banked in the registry, troughs ~72/60/48/36G safe. ETA ~12:1xZ
08-23; queue depth-1 stated reason (rung3-exec verdict-gated), no CPU
items → `run_work_next` stays unarmed.**

Session 2026-08-23 02:41–02:4xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 910/3000, loss 0.4429
monotone down, vram 62.24/71, rate 14.834 s/step. Probe @750 row
written: 4.61 → 5.24 → 5.97 monotone rise — within-lineage record
only (pdnorm-rescale confound banked, decision read = endpoint sim100
battery), non-actionable. Disk 126G flat pre-save (step 1000 save ~20
min out, trough projection governs); RAM 48G stable. ETA
~11:1x–12:4xZ 08-23; queue depth-1 stated reason (rung3-exec
verdict-gated), no CPU items → `run_work_next` stays unarmed.**

Session 2026-08-23 01:59–02:0xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 750/3000 (quarter mark),
loss 0.4882 monotone down, vram 62.24/71, rate 14.843 s/step. Disk
126G flat since the step_000500 save — matches the banked trough
projection, no re-alarm; RAM available 48G stable (trend-watch
holds). ETA ~11:2x–12:4xZ 08-23; queue depth-1 stated reason
(rung3-exec verdict-gated), no CPU items → `run_work_next` stays
unarmed.**

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
