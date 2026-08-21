# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-21 02:44–02:5xZ (tick) — **eval-2750 BANKED
4.6445 via in-session hold: the convicted-elevation contrast is
now complete across the whole 2250–2750 core. Convicted held
elevated through all three rows (6.59 → 6.83 → 6.32); democlean
FELL through all three (4.8687 → 4.809 → 4.6445), now below
onerig's 2500 level and closing on onerig's 4.50@2750. Clean-alone
reproduces nothing of the poison signature anywhere in its core.
Record-only per pre-reg; sim100 at 3000 stays the verdict
instrument.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 2750/3000 at
02:50Z, boundary-row loss 0.2817 (babysit read at 02:45Z:
2730/3000, 0.287 → 0.283, −0.0040; 16.13 s/step, 3.8 steps/min
wall), vram 62.24 vs ≤75, babysit exit 0, no gate crossings, ~1.1
h from 2750 → **endpoint ~03:5x–04:0xZ 08-21**. Probe curve:
11.82@250 → 8.14@500 → 7.90@750 → 6.49@1000 → 5.95@1250 →
5.72@1500 → 5.454@1750 → 4.9305@2000 → 4.8687@2250 → 4.809@2500 →
**4.6445@2750** (banked in-session 02:50:35Z; vs convicted 6.32,
onerig 4.50 — last probe row before the endpoint). Infra: disk
104G free (steady state; step-3000 trough ~60G, margin holds), RAM
available 47G (twenty-second read, in-band 46–49G, no leak trend),
pruner unit active, log clean: 01:49:24Z step-2000 prune is the
latest line, next work at the step-3000 save.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, in-session hold through the
step-2750 eval boundary + row banked, disk + RAM + pruner-log +
pruner-unit checks, queue validate green (depth 2, 14 open),
now.md + archive roll (02:02 aged out).

**Next**: endpoint ~03:5x–04:0xZ — step-3000 save (trough ~60G,
then +32G back at the step-2500 optimizer prune) →
`democlean-endpoint-close` → sim100 verdict through the frozen
grid (≥20 / ≤10 / 11–19); next session owns it. `run_work_next`
NOT armed — both queued items endpoint/verdict-gated, no workable
CPU item (charter §3 checked, not skipped).*

*Updated 2026-08-21 02:23–02:3xZ (tick) — **plain riding tick,
ALL-GREEN into the last ~350 steps. Eval-2750 projects ~02:52Z+ —
past this tick's commit-safe window, so no hold-to-the-wire: the
row is record-only per pre-reg (sim100 at 3000 is the verdict
instrument), nothing to decide on it in-window, and charter §6
says prefer exiting on stable stretches. Next tick banks it.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 2650/3000 at
02:24Z, loss 0.294 → 0.287 (−0.0070, descent steady; 16.552
s/step, 3.3 steps/min wall window), vram 62.24 vs ≤75, babysit
exit 0, no gate crossings, ~1.6 h to 3000 → **endpoint
~03:5x–04:0xZ 08-21**. Probe curve: 11.82@250 → 8.14@500 →
7.90@750 → 6.49@1000 → 5.95@1250 → 5.72@1500 → 5.454@1750 →
4.9305@2000 → 4.8687@2250 → 4.809@2500 (converged to onerig
4.79@2500; next row @2750 ~02:5xZ, convicted 6.32). Infra: disk
104G free (expected steady state, net −12G/boundary retained
weights; step-3000 trough ~60G, margin holds), RAM available 46G
(twenty-first read, in-band 46–49G lower edge, no leak trend),
pruner unit active, log clean: 01:49:24Z step-2000 prune is the
latest line, next work at the step-3000 save.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, disk + RAM + pruner-log +
pruner-unit checks, queue validate green (depth 2, 14 open),
eval-2750 hold/exit judgment call (exit — row lands at the hard
kill), now.md + archive roll (01:21 aged out).

**Next**: eval-2750 ~02:5xZ (convicted 6.32) — next tick banks the
row. Endpoint ~03:5x–04:0xZ: step-3000 save (trough ~60G, then
+32G back at the step-2500 optimizer prune) →
`democlean-endpoint-close` → sim100 verdict through the frozen
grid (≥20 / ≤10 / 11–19). `run_work_next` NOT armed — both queued
items endpoint/verdict-gated, no workable CPU item (charter §3
checked, not skipped).*

## Utilization footer

Session 2026-08-21 02:44–02:5xZ (tick; `democlean` riding, ~12.8
GPU-h elapsed of ~13.5 projected vs the 17 gate): **eval-2750
BANKED 4.6445 via in-session hold (row landed 02:50:35Z) — the
convicted-elevation contrast is complete across the whole
2250–2750 core: convicted 6.59/6.83/6.32 all elevated, democlean
4.8687/4.809/4.6445 all falling, now closing on onerig 4.50@2750.
Clean-alone reproduces nothing of the poison signature; record-only
per pre-reg, sim100 at 3000 stays the verdict. Ride ALL-GREEN:
babysit exit 0, step 2750/3000 at 02:50Z, loss 0.2817 at the
boundary row, 16.13 s/step, vram 62.24/75, no gate crossings, ~1.1
h to 3000 → endpoint ~03:5x–04:0xZ 08-21 (step-3000 save →
endpoint-close → sim100). Infra: disk 104G free (trough ~60G at
step-3000, margin holds), RAM 47G (twenty-second read, in-band, no
leak trend), pruner active + correctly quiet since the 01:49:24Z
step-2000 prune. Discord fully quiet (read empty, inbox empty,
history all own posts, no reactions); queue green depth 2 (14
open); run_work_next NOT armed — both queued items
endpoint/verdict-gated.**

Session 2026-08-21 02:23–02:3xZ (tick; `democlean` riding, ~12.4
GPU-h elapsed of ~13.5 projected vs the 17 gate): **plain riding
tick ALL-GREEN into the last ~350 steps — babysit exit 0, step
2650/3000 at 02:24Z, loss 0.294→0.287 (−0.0070), 16.552 s/step,
vram 62.24/75, no gate crossings, ~1.6 h to 3000 → endpoint
~03:5x–04:0xZ 08-21. Judgment call: eval-2750 projects ~02:52Z+,
at this tick's 02:53:54Z hard kill — record-only row, nothing to
decide on it, so exit clean per charter §6 rather than hold to the
wire; next tick banks it. Infra: disk 104G free (expected steady
state; step-3000 trough ~60G, margin holds), RAM 46G (twenty-first
read, in-band lower edge), pruner active + correctly quiet since
the 01:49:24Z step-2000 prune. Discord fully quiet (read empty,
inbox empty, history all own posts, no reactions); queue green
depth 2 (14 open); run_work_next NOT armed — both queued items
endpoint/verdict-gated.**

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
