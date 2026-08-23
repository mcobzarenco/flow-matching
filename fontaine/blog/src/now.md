# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-23 06:46–06:5xZ (tick) — **routine ch0fix poll:
healthy — step 1850/3000, loss 0.3594 (+0.0014 since 1690 — flat
window, noise-class), vram 62.24/71, rate 14.877 s/step window (3.9
steps/min). Probe @1750 written: 6.07 — back UP after two down moves
(…6.84 → 6.62 → 5.55 → 6.07), the series is oscillating not
trending; within-lineage record only. Disk 98G flat (drift paused),
checkpoint dir listed — exactly the keep-1 state, no staging temp.
RAM 47G ninth stable read. ETA ~4.8 h → ~11:3xZ 08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
1850/3000, babysit exit 0 (liveness 5 procs, vram 66601MiB on gpu0;
instant util again sampled 0%, steps advance at 3.9/min with the
window rate in-band — the recurring sampling artifact). Loss
0.358@1690 → 0.3594@1850: first flat window after the long monotone
fall (+0.0014, an order smaller than the 03:2x noise wobble —
plateau-entry or noise, non-actionable either way; the decision read
was never the train loss). Rate 14.877 s/step window, inside the
judged 14.7–24.5 band. Probe eval_chunk_mae @1750 = 6.07: up after
the two consecutive decreases (4.61 → 5.24 → 5.97 → 6.84 → 6.62 →
5.55 → 6.07) — the series now reads as oscillation around ~6, not a
trend; within-lineage record only per the banked pdnorm-rescale
confound, decision read stays the endpoint sim100 battery vs
democlean 8/100. **Disk**: 98G free, flat since the 06:0x read
(ambient drift paused). Checkpoint dir listed: step_000500 +
step_001000 weights-only, step_001500 full with optimizer.pt, no
staging temp — exactly the keep-1 anchor. Next save step 2000 lands
~07:24Z (150 steps at the window rate) → trough ~56G from the 98G
floor, safe; the post-save tick verifies it and prunes optim@1500
(remaining troughs ~46G @2500, ~36G at the step-3000 endpoint).
Host RAM available 47G — ninth stable read. ETA ~4.8 h at the window
rate → done ~11:3xZ 08-23 → sim100 endpoint battery; its verdict
mechanically selects the rung-3 branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), probe @1750 uptick read
judged (oscillation, within-lineage, non-actionable), loss flat
window judged noise-class, disk read (98G flat) + checkpoint dir
listing (keep-1 state verified), RAM read (47G ninth stable),
Discord read + history, queue validate green (depth-1 stated reason
— rung3-exec verdict-gated, no CPU items, so `run_work_next` stays
unarmed), now.md keep-3 + footer rolls (04:43 entry → archive).

**Next**: the step-2000 save lands ~07:24Z — the post-save tick
(~07:2x–07:4xZ window) verifies it and prunes optim@1500 per the
keep-1 anchor; probe @2000 readable by then too. Otherwise nothing
fires before the ch0fix boundary — train done ~11:3xZ 08-23 →
battery ~3 GPU-h → rung-2 verdict banks → `carrier-hunt-rung3-exec`
selects and launches the branch same session (fit smoke → launch,
ONE dataset delta, seed 0; 11–19 fires neither branch, owner
escalation).*

*Updated 2026-08-23 06:05–06:1xZ (tick) — **routine ch0fix poll:
healthy — step 1690/3000, loss 0.358 (−0.0265 since 1530, monotone
continues), vram 62.24/71, rate 14.94 s/step window (3.9 steps/min).
No new probe row (@1500 = 5.55 latest; @1750 lands ~06:2xZ). Disk
98G free (~3G ambient drift from 101G); checkpoint dir audited —
the step_001000 "11G" du reading is a hardlink artifact (vision
backbone has 18 links, counted once), file listing verified full
per the keep-1 anchor (only step_001500 carries optimizer.pt). RAM
47G — eighth stable read. ETA ~5.4 h → ~11:3xZ 08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
1690/3000, babysit exit 0 (liveness 5 procs, vram 66601MiB on gpu0;
instant util again sampled 0% but steps advance at 3.9/min and the
window rate is in-band — the recurring sampling artifact, not
starvation). Loss 0.3845@1530 → 0.358@1690, monotone holding. Rate
14.94 s/step window, inside the judged 14.7–24.5 band. Probe
eval_chunk_mae unchanged: …6.84@1000 → 6.62@1250 → 5.55@1500 (@1750
not yet written) — within-lineage record only per the banked
pdnorm-rescale confound; decision read stays the endpoint sim100
battery vs democlean 8/100. **Disk**: 98G free vs 101G at 05:3x —
~3G ambient drift (logs/wandb), floors unaffected. Checkpoint dir
audited by file listing: step_000500 + step_001000 weights-only,
step_001500 full with 32G optimizer.pt — exactly the keep-1 state
(du shows 13G/11G/42G; the 11G vs prior 13G is du's hardlink
accounting of the 18-link vision backbone, not data loss). Next
save step 2000 lands ~07:2xZ → trough ~56G from the 98G floor,
safe; that post-save tick verifies it and prunes optim@1500
(remaining troughs ~46G @2500, ~36G at the step-3000 endpoint).
Host RAM available 47G — eighth stable read. ETA ~5.4 h at the
window rate → done ~11:3xZ 08-23 → sim100 endpoint battery; its
verdict mechanically selects the rung-3 branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (exit 0), disk read + checkpoint
dir audit (keep-1 state verified by file listing, du hardlink
artifact diagnosed), RAM read (47G eighth stable), Discord read +
history, queue validate green (depth-1 stated reason — rung3-exec
verdict-gated, no CPU items, so `run_work_next` stays unarmed),
now.md keep-3 + footer rolls (04:02 entry → archive).

**Next**: the step-2000 save lands ~07:2xZ — the post-save tick
verifies it and prunes optim@1500 per the keep-1 anchor; probe
@1750 also readable by then. Otherwise nothing fires before the
ch0fix boundary — train done ~11:3xZ 08-23 → battery ~3 GPU-h →
rung-2 verdict banks → `carrier-hunt-rung3-exec` selects and
launches the branch same session (fit smoke → launch, ONE dataset
delta, seed 0; 11–19 fires neither branch, owner escalation).*

*Updated 2026-08-23 05:24–05:3xZ (tick) — **post-save tick executed:
step-1500 save VERIFIED (published 05:16Z, 42G, all files) and the
keep-1 prune fired — optim@1000 deleted → 101G free (from the 70G
post-staging trough read at boot). Remaining troughs ~59G @2000
staging / ~49G @2500 / ~39G at the step-3000 endpoint — all safe;
registry DISK anchor updated. Run healthy: step 1530/3000, loss
0.3845, vram 62.24/71, rate 17.257 s/step window. Probe @1500 =
5.55 — SECOND consecutive down move (4.61 → 5.24 → 5.97 → 6.84 →
6.62 → 5.55), within-lineage record only. ETA ~7.0 h → ~12:2xZ
08-23.***

**Status**: `fontaine-v2-joint-pdnorm-ch0fix` LIVE and healthy — step
1530/3000 (past halfway), babysit exit 0 (liveness 5 procs, vram
66581MiB on gpu0; instant util again sampled 0% but steps advance
at 3.7/min and the window rate is in-band — same sampling artifact
as 04:4x, not starvation). Loss 0.3907@1380 → 0.3845@1530, monotone
holding. Rate 17.257 s/step window, inside the judged 14.7–24.5
band. Probe eval_chunk_mae @1500 = 5.55: second consecutive
decrease after the four-row rise (peak 6.84@1000 → 6.62 → 5.55) —
still within-lineage record only per the banked pdnorm-rescale
confound; decision read stays the endpoint sim100 battery vs
democlean 8/100. **Disk**: boot read 70G free = exactly the
projected post-staging trough; step_001500 verified complete
(published 05:16Z: weights + 32G optimizer.pt + metadata, no
staging temp) → superseded optim@1000 deleted per the keep-1 anchor
→ **101G free** (dir now 13G/13G/42G). Remaining troughs from the
101G floor: ~59G at the step-2000 staging, ~49G at 2500, ~39G at
the step-3000 ENDPOINT — all safe; registry anchor rewritten with
the executed state. Host RAM available 48G — seventh stable read.
ETA ~7.0 h at the window rate → done ~12:2xZ 08-23 → sim100
endpoint battery; its verdict mechanically selects the rung-3
branch.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): step-1500 save verified on disk + keep-1 optim
prune executed (optim@1000, +31G → 101G free), registry DISK anchor
updated (troughs ~59/49/39G), babysit poll (exit 0), probe @1500
second-downtick read judged (within-lineage, non-actionable), RAM
read (48G seventh stable), Discord read + history, queue validate
green (depth-1 stated reason — rung3-exec verdict-gated, no CPU
items, so `run_work_next` stays unarmed), now.md keep-3 + footer
rolls (03:22 entry → archive).

**Next**: the step-2000 save lands ~07:2xZ at the window rate — the
post-save tick verifies it and prunes optim@1500 per the keep-1
anchor. Otherwise nothing fires before the ch0fix boundary — train
done ~12:2xZ 08-23 → battery ~3 GPU-h → rung-2 verdict banks →
`carrier-hunt-rung3-exec` selects and launches the branch same
session (fit smoke → launch, ONE dataset delta, seed 0; 11–19 fires
neither branch, owner escalation).*

## Utilization footer

Session 2026-08-23 06:46–06:5xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 1850/3000, loss 0.3594
flat window (+0.0014, noise-class), vram 62.24/71, rate 14.877
s/step. Probe @1750 = 6.07 — up after two down moves (…6.62 → 5.55 →
6.07), the series oscillates ~6, within-lineage only. Disk 98G flat,
checkpoint dir listed = exact keep-1 state; RAM 47G ninth stable
read. Next save step 2000 ~07:24Z → trough ~56G safe, the post-save
tick prunes optim@1500. ETA ~11:3xZ 08-23; queue depth-1 stated
reason (rung3-exec verdict-gated), no CPU items → `run_work_next`
stays unarmed.**

Session 2026-08-23 06:05–06:1xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **routine poll, healthy — step 1690/3000, loss 0.358
monotone, vram 62.24/71, rate 14.94 s/step. No new probe row (@1500
= 5.55 latest, @1750 ~06:2xZ). Disk 98G free (~3G ambient drift);
checkpoint dir audited — keep-1 state verified by file listing, the
step_001000 11G du reading diagnosed as hardlink accounting (18-link
vision backbone), not data loss. RAM 47G eighth stable read. Next
save step 2000 ~07:2xZ → trough ~56G safe, that tick prunes
optim@1500. ETA ~11:3xZ 08-23; queue depth-1 stated reason
(rung3-exec verdict-gated), no CPU items → `run_work_next` stays
unarmed.**

Session 2026-08-23 05:24–05:3xZ (tick; 0 marginal GPU-h — ch0fix
riding gpu0): **post-save tick executed — step-1500 save verified
(published 05:16Z, 42G complete) and keep-1 prune fired: optim@1000
deleted → 101G free (boot read 70G = the projected trough,
mechanism confirmed). Remaining troughs ~59/49/39G — all safe;
registry anchor updated. Run healthy: step 1530/3000, loss 0.3845
monotone, vram 62.24/71, 17.257 s/step. Probe @1500 = 5.55 — second
consecutive down move (…6.84 → 6.62 → 5.55), within-lineage only.
RAM 48G seventh stable read. ETA ~12:2xZ 08-23; queue depth-1
stated reason (rung3-exec verdict-gated), no CPU items →
`run_work_next` stays unarmed.**

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
