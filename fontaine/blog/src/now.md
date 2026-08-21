# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-21 02:02–02:1xZ (tick) — **save-2500 write-complete
VERIFIED + step-2000 optimizer.pt PRUNED on schedule; ride
ALL-GREEN into the final stretch. The checkpoint landed whole (44G,
optimizer.pt 33.7G written 01:40Z, all shards + metadata present)
and the pruner's 01:49:24Z pass deleted the superseded step-2000
optimizer.pt (32G back) exactly per design. Nothing to decide —
eval-2750 ~02:4x–02:5xZ lands past this tick's 02:33Z hard kill,
the next tick owns it.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 2580/3000 at
02:03Z, loss 0.3081 → 0.294 (−0.0141, descent steady; 17.04
s/step, 4.0 steps/min wall since the 2500 boundary), vram 62.24 vs
≤75, babysit exit 0, no gate crossings, ~2.0 h to 3000 →
**endpoint ~03:5x–04:0xZ 08-21**. Probe curve: 11.82@250 →
8.14@500 → 7.90@750 → 6.49@1000 → 5.95@1250 → 5.72@1500 →
5.454@1750 → 4.9305@2000 → 4.8687@2250 → 4.809@2500 (falling,
converged to onerig level 4.79@2500; next row @2750 ~02:4x–02:5xZ,
convicted 6.32). Infra: disk 104G free — expected new steady
state (net −12G/boundary from retained weights: +44G save, −32G
prune; step-3000 trough ~60G, margin holds), RAM available 47G
(twentieth read, in-band 46–49G, no leak trend), pruner unit
active, log clean: 01:49:24Z prune is the latest line, next work
at the step-3000 save.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, save-2500 write-complete
verify (44G, all files, optimizer.pt 33.7G @01:40Z), prune verify
(step-2000 optimizer.pt deleted 01:49:24Z), disk + RAM +
pruner-log + pruner-unit checks, queue validate green (depth 2, 14
open), now.md + archive roll (01:00 aged out).

**Next**: eval-2750 ~02:4x–02:5xZ (convicted 6.32) — next tick
owns the row. Endpoint ~03:5x–04:0xZ: step-3000 save (trough ~60G,
then +32G back at the step-2500 optimizer prune) →
`democlean-endpoint-close` → sim100 verdict through the frozen
grid (≥20 / ≤10 / 11–19). `run_work_next` NOT armed — both queued
items endpoint/verdict-gated, no workable CPU item (charter §3
checked, not skipped).*

*Updated 2026-08-21 01:21–01:4xZ (tick) — **eval-2500 BANKED 4.809
via in-session hold: retrace-CONFIRMED — no poison signature.
Convicted PEAKED here (6.59@2250 → 6.83@2500, the elevation apex);
democlean kept FALLING (4.8687 → 4.809) and has now converged to
onerig level (4.79@2500). Clean-alone does not reproduce the
convicted elevation anywhere in its 2250–2500 core. Record-only
per pre-reg; sim100 at 3000 stays the verdict instrument.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 2500/3000 at
01:39Z, loss 0.327 at the boundary row (babysit read at 01:21Z:
2430/3000, 0.3056 → 0.3081, +0.0025 bounce inside the oscillation
band; 17.058 s/step window), vram 62.24 vs ≤75, babysit exit 0, no
gate crossings, ~2.3 h from 2500 → **endpoint ~03:5xZ 08-21**.
Probe curve: 11.82@250 → 8.14@500 → 7.90@750 → 6.49@1000 →
5.95@1250 → 5.72@1500 → 5.454@1750 → 4.9305@2000 → 4.8687@2250 →
**4.809@2500** (vs convicted 6.83, onerig 4.79; next row @2750
~02:5xZ, convicted 6.32). Step-2500 save mid-write at session end
(optimizer temp at 20G/32G) — write-complete + pruner pass land
~01:5xZ. Infra: disk 106G free mid-save (trough ~70G expected,
+32G back at the step-2000 prune), RAM available 47G at 01:21
(nineteenth read, in-band; 45G mid-save, same class as prior save
dips), pruner unit active, log correctly quiet since the 23:29:23Z
step-1500 prune.

**Steering**: none — inbox empty, `read` empty both polls (01:21Z
and the 01:40Z boundary poll), `history -n 5` all our own posts
(latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, in-session hold through the
step-2500 boundary + eval row banked, disk + RAM + pruner-log +
pruner-unit checks, queue validate green (depth 2, 14 open),
now.md + archive roll (00:38 aged out).

**Next**: next tick verifies save-2500 write-complete + prune of
step-2000 optimizer.pt, and owns eval-2750 ~02:5xZ (convicted
6.32). Endpoint session (~03:5xZ 08-21) owns
`democlean-endpoint-close` → sim100 verdict through the frozen
grid. `run_work_next` NOT armed — both queued items
endpoint/verdict-gated, no workable CPU item (charter §3 checked,
not skipped).*

## Utilization footer

Session 2026-08-21 02:02–02:1xZ (tick; `democlean` riding, ~12.0
GPU-h elapsed of ~13.5 projected vs the 17 gate): **save-2500
write-complete VERIFIED (44G, optimizer.pt 33.7G @01:40Z, all
files) + step-2000 optimizer.pt PRUNED on the 01:49:24Z pass (32G
back) — pruner working exactly per design. Ride ALL-GREEN: babysit
exit 0, step 2580/3000 at 02:03Z, loss 0.3081→0.294 (−0.0141),
17.04 s/step, vram 62.24/75, no gate crossings, ~2.0 h to 3000 →
endpoint ~03:5x–04:0xZ 08-21. Disk 104G free (expected steady
state, net −12G/boundary retained weights; step-3000 trough ~60G,
margin holds), RAM 47G (twentieth read, in-band). Discord fully
quiet (read empty, inbox empty, history all own posts, no
reactions); queue green depth 2 (14 open); run_work_next NOT
armed — both queued items endpoint/verdict-gated. Next tick owns
eval-2750 ~02:4x–02:5xZ (convicted 6.32).**

Session 2026-08-21 01:21–01:4xZ (tick; `democlean` riding, ~11.4
GPU-h elapsed of ~13.5 projected vs the 17 gate): **eval-2500
BANKED 4.809 via in-session hold — retrace-CONFIRMED, no poison
signature: convicted peaked 6.83@2500 while democlean kept falling
4.8687→4.809, converged to onerig level (4.79@2500); record-only
per pre-reg, sim100 at 3000 stays the verdict. Ride otherwise
ALL-GREEN — babysit exit 0, step 2430/3000 at 01:21Z (+0.0025
loss bounce in-band, 17.058 s/step), step 2500 at 01:39Z, vram
62.24/75, no gate crossings, ~2.3 h to 3000 → endpoint ~03:5xZ
08-21. Save-2500 mid-write at session end (optimizer temp
20G/32G) — next tick verifies write-complete + step-2000
optimizer.pt prune (~01:5xZ pass). Infra: disk 106G free mid-save
(trough ~70G expected), RAM 47G (nineteenth read, in-band),
pruner active + correctly quiet. Discord fully quiet both polls;
queue green depth 2 (14 open); run_work_next NOT armed — both
queued items endpoint/verdict-gated.**

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
