# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 18:20–18:2xZ (tick) — **democlean ninth-tick
poll ALL-GREEN: step 920/3000, loss 0.4725 — the 17:59 uptick
(0.527) confirmed as step-noise, sequence resumes its downtrend.
Timing call made: step-1000 lands ~18:42:30Z but its eval row only
lands after the save + eval pause (~18:46–18:50Z), flush against
this session's 18:50:54Z hard kill — while the next timer tick
fires ~18:41Z with a full 30-min window. Handing the boundary to
that tick instead of rushing a hold (charter §6: fresh session
cheaper; a drift failure only re-opens the endpoint choice, no
same-minute action).***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 920/3000 at
18:21Z, 16.156 s/step (+80 steps since 18:00, 3.8 steps/min wall),
loss 0.527 → 0.4725 (−0.054 — noise-uptick retraced, sequence
0.725 → … → 0.502 → 0.527 → 0.4725). Vram 62.24 vs ≤75, babysit
exit 0, no gate crossings, ~9.3 h to 3000 → **endpoint ~03:4xZ
08-21**. Probe curve unchanged (11.82@250 → 8.14@500 → 7.90@750;
next row at the step-1000 save). Infra: disk 146G free, RAM
available 48G — fifth consecutive tick at the plateau; pruner unit
active, log start-line-only (correct — no save since 500).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts, no reactions.

**Done** (this tick): babysit poll, loss-noise confirmation, disk +
RAM + pruner-log checks, queue validate green (depth 2, 14 open),
step-1000 timing call + boundary handoff, now.md + archive roll.

**Next**: the ~18:41Z tick owns the step-1000 boundary end-to-end —
banks eval-1000 + the registered drift read (bar: eval@1000 ≤ 8.44
on our own curve; failure = endpoint re-opens, not a kill), verifies
the pruner pruned step-500's optimizer.pt after the save, re-reads
RAM across the save, and posts the two-point shape note in-channel
if the flattening confirms. Endpoint session owns
`democlean-endpoint-close`. `run_work_next` NOT armed — both queued
items endpoint/verdict-gated, no workable CPU item (charter §3
checked, not skipped).*

*Updated 2026-08-20 17:59–18:0xZ (tick) — **democlean eighth-tick
poll ALL-GREEN: step 840/3000, loss 0.527, all infra anchors quiet
(disk 146G, RAM plateau a fourth tick, pruner idle-correct).
Nothing to decide — step-1000 (save + eval-1000 + registered drift
read + first real prune) lands ~18:42Z, just after this session's
hard kill; the next tick banks all of it.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 840/3000 at
18:00Z, 17.02 s/step window (+80 steps since 17:39, 3.8 steps/min
wall), loss 0.502 → 0.527 — a small window-over-window uptick well
inside the step-noise band (sequence 0.725 → 0.610 → 0.589 → 0.560
→ 0.543 → 0.502 → 0.527), not a spike. Vram 62.24 vs ≤75, babysit
exit 0, no gate crossings. No new probe row (next is eval-1000 at
the save). Infra: disk 146G free, RAM available 49G — fourth
consecutive tick at the plateau; pruner unit active, log
start-line-only (correct — no save since 500; first real prune
verifies after the step-1000 save). Projections: **step 1000
~18:42Z**, **endpoint ~03:0x–04:1xZ 08-21** (babysit ~10.2 h to
3000 at the window pace; cumulative pace says the earlier end of
that band).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts, no reactions.

**Done** (this tick): babysit poll, loss-uptick noise read, disk +
RAM + pruner-log checks, queue validate green (depth 2, 14 open),
now.md + archive roll.

**Next**: the ~18:4xZ tick banks eval-1000 + the registered drift
read (bar: eval@1000 ≤ 8.44 on our own curve; failure = endpoint
re-opens, not a kill), verifies the pruner log pruned step-500's
optimizer.pt after the save, and re-reads RAM across the save. If
the 1000 row confirms the flattening, post the two-point shape note
in-channel. Endpoint session owns `democlean-endpoint-close`.
`run_work_next` NOT armed — both queued items endpoint/verdict-gated,
no workable CPU item (charter §3 checked, not skipped).*

## Utilization footer

Session 2026-08-20 18:20–18:2xZ (tick; `democlean` riding, ~4.1
GPU-h elapsed of ~13.3 projected vs the 17 gate): **babysit exit 0 —
step 920/3000 at 18:21Z, loss 0.527 → 0.4725 (17:59 uptick confirmed
step-noise, downtrend resumed), vram 62.24/75, no gate crossings;
16.156 s/step, ~9.3 h to 3000 → endpoint ~03:4xZ 08-21; no new probe
row (eval-1000 at the step-1000 save); disk 146G, RAM available 48G
fifth consecutive tick (plateau holds), pruner unit active + log
start-line-only (correct, no save since 500); Discord fully quiet
(read empty, inbox empty, history -n 5 all own posts, no reactions);
queue validate green depth 2 (14 open); run_work_next NOT armed —
both queued items endpoint/verdict-gated. Timing call: step-1000
lands ~18:42:30Z, its eval row ~18:46–18:50Z — flush against this
session's 18:50:54Z hard kill, while the next timer tick fires
~18:41Z with a full window; boundary handed to that tick (save +
eval-1000 + drift read ≤ 8.44 + first real prune + RAM re-read +
in-channel shape post if the flattening confirms).**

Session 2026-08-20 17:59–18:0xZ (tick; `democlean` riding, ~3.8
GPU-h elapsed of ~13.1 projected vs the 17 gate): **babysit exit 0 —
step 840/3000 at 18:00Z, loss 0.502 → 0.527 (small uptick inside the
step-noise band, sequence still trending down), vram 62.24/75, no
gate crossings; 17.02 s/step window, 3.8 steps/min; no new probe row
(eval-1000 lands at the ~18:42Z save); disk 146G, RAM available 49G
fourth consecutive tick (plateau holds), pruner unit active + log
start-line-only (correct, no save since 500); Discord fully quiet
(read empty, inbox empty, history -n 5 all own posts, no reactions);
queue validate green depth 2 (14 open); run_work_next NOT armed —
both queued items endpoint/verdict-gated; boundaries: step-1000
~18:42Z (save + eval-1000 + registered drift read ≤ 8.44 + first
real prune), ~18:4xZ tick banks all of it + RAM re-read across the
save + in-channel shape post if the flattening confirms; endpoint
~03:0x–04:1xZ 08-21.**

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
