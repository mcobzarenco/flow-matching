# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-23 04:43–04:4xZ (tick) — **routine ch0fix poll:
healthy — step 1380/3000, loss 0.3907 (−0.0049 since 1220, monotone
continues), vram 62.24/71, rate 16.267 s/step window (3.9
steps/min). Probe @1250 written: 6.62 — the first DOWN move in the
series (4.61 → 5.24 → 5.97 → 6.84 → 6.62), within-lineage record
only. Disk 112G flat, RAM 48G stable sixth read. Pre-save tick: the
step-1500 save lands ~05:17Z → trough ~70G safe; the NEXT tick
verifies it and prunes optim@1000 per the keep-1 anchor. ETA ~7.3 h
→ ~12:0xZ 08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
1380/3000, babysit exit 0 (liveness 5 procs, vram 66581MiB on gpu0;
the instant util read sampled 0% but steps advance at 3.9/min and
the window rate is in-band — sampling artifact, not starvation).
Loss 0.3956@1220 → 0.3907@1380, monotone resumed-and-holding. Rate
16.267 s/step window, inside the judged 14.7–24.5 band. Probe
eval_chunk_mae @1250 = 6.62: first decrease after four rising rows
(4.61 → 5.24 → 5.97 → 6.84 → 6.62) — still within-lineage record
only per the banked pdnorm-rescale confound; decision read stays the
endpoint sim100 battery vs democlean 8/100. **Disk**: 112G free,
flat since the 04:0x read (ambient drift paused); next save step
1500 (~05:17Z at the window rate) stages 42G → trough ~70G, safe per
the keep-1 registry anchor, then the post-save tick deletes
superseded optim@1000. Host RAM available 48G — sixth stable read.
ETA ~7.3 h at the window rate → done ~12:0xZ 08-23 → sim100 endpoint
battery; its verdict mechanically selects the rung-3 branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), probe @1250 first-downtick
read judged (within-lineage, non-actionable), pre-save disk read
(112G flat, trough projection reconfirmed), RAM read (48G sixth
stable), Discord read + history, queue validate green (depth-1
stated reason — rung3-exec verdict-gated, no CPU items, so
`run_work_next` stays unarmed), now.md keep-3 + footer rolls (02:41
entry → archive).

**Next**: the ~05:2xZ tick verifies the step-1500 save (staged
~05:17Z) and executes the keep-1 prune (optim@1000 becomes
superseded). Otherwise nothing fires before the ch0fix boundary —
train done ~12:0xZ 08-23 → battery ~3 GPU-h → rung-2 verdict banks →
`carrier-hunt-rung3-exec` selects and launches the branch same
session (fit smoke → launch, ONE dataset delta, seed 0; 11–19 fires
neither branch, owner escalation).*

*Updated 2026-08-23 04:02–04:0xZ (tick) — **routine ch0fix poll:
healthy — step 1220/3000, loss 0.3956 (−0.0894 since 1060: the 03:2x
window uptick cleared as noise, monotone resumed), vram 62.24/71,
rate 14.821 s/step window (3.9 steps/min). Probe @1250 row not yet
written. First post-mitigation disk read: 112G free, checkpoint dir
verified matching the keep-1 rule (step_000500 = 13G optim-pruned,
step_001000 = 42G newest intact); next save at step 1500 (~05:1xZ)
stages 42G → trough ~70G, then that tick prunes optim@1000. RAM 48G
stable. ETA ~7.3 h → ~11:2xZ 08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
1220/3000, babysit exit 0 (liveness 5 procs, gpu0 66581MiB/100%
util), vram 62.24 stable vs the 71 gate. Loss 0.485@1060 →
0.3956@1220: the first-window uptick judged noise-class last tick is
confirmed noise — the fall resumed steeper than the wobble. Rate
14.821 s/step window, inside the judged band. Probe eval_chunk_mae
still 4.61@250 → 5.24@500 → 5.97@750 → 6.84@1000 (@1250 not yet
written; within-lineage record only, decision read = endpoint sim100
battery vs democlean 8/100). **Disk**: 112G free vs 115G at the
03:2x prune — ~3G ambient drift (logs/wandb), floors unaffected;
checkpoint dir audited and matches the banked keep-1 state exactly.
Next save step 1500 lands ~05:1xZ → trough ~70G (safe), then the
post-save tick deletes superseded optim@1000 per the registry
anchor. Host RAM available 48G — fifth stable read. ETA ~7.3 h at
the window rate → done ~11:2xZ 08-23 → sim100 endpoint battery; its
verdict mechanically selects the rung-3 branch.

**Steering**: none — inbox empty, `read` surfaced only our own 03:25
post, `history -n 5` all own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), loss-uptick resolution
read (noise confirmed, monotone resumed), first post-mitigation disk
read + checkpoint-dir audit vs the keep-1 anchor (matches: 13G/42G),
RAM read, Discord read + history, queue validate green (depth-1
stated reason — rung3-exec verdict-gated, no CPU items, so
`run_work_next` stays unarmed), now.md keep-3 + footer rolls (01:59
entry → archive).

**Next**: the ~05:1xZ tick verifies the step-1500 save and executes
the keep-1 prune (optim@1000 becomes superseded). Otherwise nothing
fires before the ch0fix boundary — train done ~11:2xZ 08-23 →
battery ~3 GPU-h → rung-2 verdict banks → `carrier-hunt-rung3-exec`
selects and launches the branch same session (fit smoke → launch,
ONE dataset delta, seed 0; 11–19 fires neither branch, owner
escalation).*

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

## Utilization footer

Session 2026-08-23 04:43–04:4xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 1380/3000, loss 0.3907
monotone, vram 62.24/71, rate 16.267 s/step. Probe @1250 = 6.62 —
first DOWN move in the series (…5.97 → 6.84 → 6.62), within-lineage
record only. Disk 112G flat, RAM 48G stable sixth read. Pre-save
tick: step-1500 save ~05:17Z → trough ~70G safe, next tick prunes
optim@1000 per the keep-1 anchor. ETA ~12:0xZ 08-23; queue depth-1
stated reason (rung3-exec verdict-gated), no CPU items →
`run_work_next` stays unarmed.**

Session 2026-08-23 04:02–04:0xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 1220/3000, loss 0.3956
(the 03:2x window uptick confirmed noise, monotone resumed), vram
62.24/71, rate 14.821 s/step. First post-mitigation disk read: 112G
free (~3G ambient drift from 115), checkpoint dir audited matching
the keep-1 anchor (step_000500 13G optim-pruned, step_001000 42G
intact); next save step 1500 ~05:1xZ → trough ~70G safe, that tick
prunes optim@1000. RAM 48G stable. ETA ~11:2xZ 08-23; queue depth-1
stated reason (rung3-exec verdict-gated), no CPU items →
`run_work_next` stays unarmed.**

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
