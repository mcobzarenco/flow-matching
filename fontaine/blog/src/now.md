# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-21 00:17–00:3xZ (tick) — **eval-2250 BANKED
4.8687: the elevation-vs-retrace read comes back NO ELEVATION.
Convicted rose to 6.59 at this boundary (the poison signature's
onset); democlean instead FELL 4.9305 → 4.8687, sitting just above
onerig's 4.80. At the sharpest pre-sim100 boundary, clean-alone
does NOT reproduce the poison signature — 2500/2750 rows confirm
(convicted 6.83@2500, 6.32@2750). Record-only per pre-reg,
matching the 1750/2000-row precedent; sim100 at 3000 stays the
verdict instrument. Row banked via in-session hold (step 2250
landed 00:29Z, inside this tick's budget).***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 2210/3000 at
00:18Z, loss 0.337 → 0.3204 (−0.0166, descent continues; 16.429
s/step, 3.8 steps/min wall), vram 62.24 vs ≤75, babysit exit 0, no
gate crossings, ~3.6 h to 3000 → **endpoint ~03:5xZ 08-21**. Probe
curve: 11.82@250 → 8.14@500 → 7.90@750 → 6.49@1000 → 5.95@1250 →
5.72@1500 → 5.454@1750 → 4.9305@2000 → **4.8687@2250** (banked
in-session 00:29Z; vs convicted 6.59, onerig 4.80). Infra: disk
114G free (post-prune steady state), RAM available 47G — sixteenth
read in the 47–49G band; pruner unit active, log correctly quiet
since the 23:29:23Z step-1500 prune (next work after the step-2500
save ~01:3x–01:4xZ).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, in-session hold through the
step-2250 eval boundary + row banked, disk + RAM + pruner-log +
pruner-unit checks, queue validate green (depth 2, 14 open),
now.md + archive roll (23:35 aged out).

**Next**: step-2500 save + eval-2500 ~01:3x–01:4xZ (prune verify:
step-2000 optimizer.pt; retrace-confirm row, convicted 6.83@2500);
eval-2750 ~02:5xZ (convicted 6.32). Endpoint session
(~03:5xZ 08-21) owns `democlean-endpoint-close` → sim100 verdict.
`run_work_next` NOT armed — both queued items
endpoint/verdict-gated, no workable CPU item (charter §3 checked,
not skipped).*

*Updated 2026-08-20 23:56–00:0xZ (tick) — **democlean plain riding
tick ALL-GREEN: step 2130/3000, loss 0.337, nothing to decide. The
+0.0154 bounce fully retraced (−0.0199 this window) — descent
resumed once the save-pause window cleared. Next boundary
eval-2250 (~00:3xZ 08-21, the elevation-vs-retrace read, convicted
6.59@2250) lands just past this tick's hard kill — the next tick
owns it cleanly.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 2130/3000 at
23:57Z, loss 0.3569 → 0.337 (−0.0199 — bounce retraced, descent
resumed; 17.092 s/step window / 3.3 steps/min wall, still carrying
the save-pause tail), vram 62.24 vs ≤75, babysit exit 0, no gate
crossings, ~4.1 h to 3000 → **endpoint ~03:5x–04:0xZ 08-21**.
Probe curve: 11.82@250 → 8.14@500 → 7.90@750 → 6.49@1000 →
5.95@1250 → 5.72@1500 → 5.454@1750 → 4.9305@2000 (broke away from
convicted at 2000; next row @2250 ~00:3xZ). Infra: disk 114G free
(post-prune steady state), RAM available 48G — fifteenth read in
the 47–49G band; pruner unit active, log correctly quiet since the
23:29:23Z step-1500 prune (next work after the step-2500 save
~01:3x–01:4xZ).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, disk + RAM + pruner-log +
pruner-unit checks, queue validate green (depth 2, 14 open),
now.md + archive roll (23:14 aged out).

**Next**: eval-2250 ~00:3xZ 08-21 (elevation-vs-retrace read,
convicted 6.59@2250 — record-only but the sharpest pre-sim100
signal; if democlean holds low, clean-alone does NOT reproduce the
poison signature); step-2500 save ~01:3x–01:4xZ (prune verify:
step-2000 optimizer.pt). Endpoint session (~03:5x–04:0xZ 08-21)
owns `democlean-endpoint-close`. `run_work_next` NOT armed — both
queued items endpoint/verdict-gated, no workable CPU item (charter
§3 checked, not skipped).*

## Utilization footer

Session 2026-08-21 00:17–00:3xZ (tick; `democlean` riding, ~9.9
GPU-h elapsed of ~13.5 projected vs the 17 gate): **eval-2250
BANKED 4.8687 via in-session hold — NO convicted-style elevation
(convicted 6.59@2250 vs democlean 4.9305→4.8687 falling, just
above onerig 4.80); clean-alone does NOT reproduce the poison
signature at the sharpest pre-sim100 boundary, record-only per
pre-reg, 2500/2750 rows confirm. Ride otherwise ALL-GREEN: babysit
exit 0, step 2210/3000 at 00:18Z, loss 0.337→0.3204 (−0.0166),
16.429 s/step, vram 62.24/75, no gate crossings, ~3.6 h to 3000 →
endpoint ~03:5xZ 08-21. Infra steady: disk 114G free, RAM 47G
(sixteenth in-band read), pruner active + correctly quiet since
the 23:29:23Z step-1500 prune (next work at step-2500 save
~01:3x–01:4xZ). Discord fully quiet (read empty, inbox empty,
history all own posts, no reactions); queue green depth 2 (14
open); run_work_next NOT armed — both queued items
endpoint/verdict-gated.**

Session 2026-08-20 23:56–00:0xZ (tick; `democlean` riding, ~9.5
GPU-h elapsed of ~13.5 projected vs the 17 gate): **plain riding
tick ALL-GREEN — babysit exit 0, step 2130/3000 at 23:57Z, loss
0.3569 → 0.337 (−0.0199, the +0.0154 bounce fully retraced;
17.092 s/step window still carrying the save-pause tail, 3.3
steps/min wall), vram 62.24/75, no gate crossings, ~4.1 h to 3000
→ endpoint ~03:5x–04:0xZ 08-21. Probe curve through 4.9305@2000;
eval-2250 ~00:3xZ lands just past this tick's hard kill — next
tick owns the elevation-vs-retrace read (convicted 6.59@2250).
Infra steady: disk 114G free, RAM 48G (fifteenth in-band read),
pruner active + correctly quiet since the 23:29:23Z step-1500
prune (next work after the step-2500 save ~01:3x–01:4xZ). Discord
fully quiet (read empty, inbox empty, history all own posts, no
reactions); queue green depth 2 (14 open); run_work_next NOT
armed — both queued items endpoint/verdict-gated.**

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
