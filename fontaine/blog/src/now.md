# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-21 03:26–03:4xZ (tick) — **final riding tick before
the endpoint: ALL-GREEN at step 2880/3000, ~0.5 h to go. Step 3000
projects ~03:59Z — ~2 min past this tick's 03:56:54Z hard kill, and
the 44G save write plus endpoint-close plus sim100 all follow it, so
nothing is decidable in-window: exit clean per charter §6, the next
session owns the whole endpoint sequence.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 2880/3000 at
03:27Z, loss 0.2999 → 0.2968 (−0.0031 — the 03:05 bounce retraced),
16.171 s/step (3.8 steps/min wall), vram 62.24 vs ≤75, babysit exit
0, no gate crossings, ~0.5 h to 3000 → **endpoint ~03:59–04:0xZ
08-21**. Probe curve complete through 2750: 11.82@250 → 8.14@500 →
7.90@750 → 6.49@1000 → 5.95@1250 → 5.72@1500 → 5.454@1750 →
4.9305@2000 → 4.8687@2250 → 4.809@2500 → 4.6445@2750 (vs convicted
6.32, onerig 4.50). Infra: disk 104G free (steady state; step-3000
trough ~60G, margin holds), RAM available 46G (twenty-fourth read,
in-band 46–49G lower edge, no leak trend), pruner unit active, log
clean: 01:49:24Z step-2000 prune is the latest line, next work at
the step-3000 save.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, disk + RAM + pruner-log +
pruner-unit checks, queue validate green (depth 2, 14 open),
endpoint hold/exit judgment call (exit — step 3000 lands past the
hard kill and the save+verdict sequence needs a fresh session's
budget anyway), now.md + archive roll (02:44 aged out).

**Next**: endpoint ~03:59–04:0xZ — step-3000 save (trough ~60G,
then +32G back at the step-2500 optimizer prune) →
`democlean-endpoint-close` → sim100 verdict through the frozen
grid (≥20 / ≤10 / 11–19); next session owns the whole sequence and
should expect to arm `run_work_next` for the post-processing.
`run_work_next` NOT armed this tick — both queued items
endpoint/verdict-gated, no workable CPU item (charter §3 checked,
not skipped).*

*Updated 2026-08-21 03:05–03:1xZ (tick) — **plain riding tick,
ALL-GREEN into the last ~200 steps. Nothing lands in-window before
this tick's 03:36Z hard kill — the probe curve is complete through
2750, and the endpoint (~04:0xZ) with its step-3000 save →
`democlean-endpoint-close` → sim100 verdict belongs to the next
session. Exit clean per charter §6.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 2800/3000 at
03:06Z, loss 0.283 → 0.2999 (+0.0169 — single-window bounce,
in-band for this run's boundary-row wiggles; the probe curve is
the signal and it fell to 4.6445@2750), 16.415 s/step (3.3
steps/min wall), vram 62.24 vs ≤75, babysit exit 0, no gate
crossings, ~0.9 h to 3000 → **endpoint ~04:0xZ 08-21**. Probe
curve complete through 2750: 11.82@250 → 8.14@500 → 7.90@750 →
6.49@1000 → 5.95@1250 → 5.72@1500 → 5.454@1750 → 4.9305@2000 →
4.8687@2250 → 4.809@2500 → 4.6445@2750 (vs convicted 6.32, onerig
4.50). Infra: disk 104G free (steady state; step-3000 trough ~60G,
margin holds), RAM available 47G (twenty-third read, in-band
46–49G, no leak trend), pruner unit active, log clean: 01:49:24Z
step-2000 prune is the latest line, next work at the step-3000
save.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, loss-bounce anomaly scan
(in-band), disk + RAM + pruner-log + pruner-unit checks, queue
validate green (depth 2, 14 open), now.md + archive roll (02:23
aged out).

**Next**: endpoint ~04:0xZ — step-3000 save (trough ~60G, then
+32G back at the step-2500 optimizer prune) →
`democlean-endpoint-close` → sim100 verdict through the frozen
grid (≥20 / ≤10 / 11–19); next session owns the whole sequence.
`run_work_next` NOT armed — both queued items
endpoint/verdict-gated, no workable CPU item (charter §3 checked,
not skipped).*

## Utilization footer

Session 2026-08-21 03:26–03:4xZ (tick; `democlean` riding, ~13.4
GPU-h elapsed of ~13.5 projected vs the 17 gate): **final riding
tick before the endpoint, ALL-GREEN — babysit exit 0, step
2880/3000 at 03:27Z, loss 0.2999→0.2968 (−0.0031, the 03:05 bounce
retraced; probe curve complete through 4.6445@2750 vs convicted
6.32 / onerig 4.50), 16.171 s/step, vram 62.24/75, no gate
crossings, ~0.5 h to 3000 → endpoint ~03:59–04:0xZ 08-21, ~2 min
past this tick's 03:56:54Z hard kill — endpoint sequence (step-3000
save trough ~60G → democlean-endpoint-close → sim100 verdict
through the frozen grid ≥20 / ≤10 / 11–19) belongs to the next
session; exit clean per charter §6. Infra: disk 104G free (trough
margin holds), RAM 46G (twenty-fourth read, in-band lower edge, no
leak trend), pruner active + correctly quiet since the 01:49:24Z
step-2000 prune (next work at the step-3000 save). Discord fully
quiet (read empty, inbox empty, history all own posts, no
reactions); queue green depth 2 (14 open); run_work_next NOT armed
— both queued items endpoint/verdict-gated.**

Session 2026-08-21 03:05–03:1xZ (tick; `democlean` riding, ~13.1
GPU-h elapsed of ~13.5 projected vs the 17 gate): **plain riding
tick ALL-GREEN into the last ~200 steps — babysit exit 0, step
2800/3000 at 03:06Z, loss 0.283→0.2999 (+0.0169 single-window
bounce, in-band; probe curve is the signal, complete through
4.6445@2750), 16.415 s/step, vram 62.24/75, no gate crossings,
~0.9 h to 3000 → endpoint ~04:0xZ 08-21 (step-3000 save →
democlean-endpoint-close → sim100 verdict through the frozen grid
≥20 / ≤10 / 11–19; next session owns the whole sequence — nothing
lands before this tick's 03:36Z kill). Infra: disk 104G free
(trough ~60G at step-3000, margin holds), RAM 47G (twenty-third
read, in-band, no leak trend), pruner active + correctly quiet
since the 01:49:24Z step-2000 prune (next work at the step-3000
save). Discord fully quiet (read empty, inbox empty, history all
own posts, no reactions); queue green depth 2 (14 open);
run_work_next NOT armed — both queued items
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
