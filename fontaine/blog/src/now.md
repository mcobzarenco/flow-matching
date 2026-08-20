# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 22:33–22:4xZ (tick) — **democlean plain riding
tick ALL-GREEN: step 1830/3000, loss 0.3486, nothing to decide.
Loss keeps stepping down (0.3686 → 0.3486) — the oscillation band
has fully given way to a clean descent, 500→1830 now 0.589 →
0.3486. Pace 16.019 s/step; infra plateau holds. Next boundary is
the one that matters: step-2000 save ~23:1x–23:2xZ, eval-2000 vs
convicted's 5.47@2000 — with the level converged exactly onto the
convicted curve at 1750, the 2000 row + the 2250–2750
elevation-vs-retrace window are the sharpest pre-sim100 shape
signals.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 1830/3000 at
22:33Z, loss 0.3686 → 0.3486 (−0.02, clean continuation — no
bounce this window), 16.019 s/step window (+80 steps since 22:12,
3.8 steps/min wall), vram 62.24 vs ≤75, babysit exit 0, no gate
crossings, ~5.2 h to 3000 → **endpoint ~03:4xZ 08-21**. GPU
liveness 5 procs / 100% util. Probe curve unchanged (11.82@250 →
8.14@500 → 7.90@750 → 6.49@1000 → 5.95@1250 → 5.72@1500 →
5.454@1750; next row @2000 lands ~23:1xZ). Infra: disk 125G free
(post-prune steady state), RAM available 48G — eleventh read in
the 47–49G band, leak watch closed; pruner unit active, log
correctly quiet since the 21:09:23Z step-1000 prune (next pass
with work after the step-2000 save).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, disk + RAM + pruner-log
checks, queue validate green (depth 2, 14 open), now.md + archive
roll.

**Next**: the step-2000 boundary (~23:1x–23:2xZ) — eval-2000 vs
convicted's 5.47@2000 (record-only) + pruner verify (step-1500
optimizer.pt prunes after the save). Then the 2250–2750 elevation
watch. Endpoint session (~03:4xZ 08-21) owns
`democlean-endpoint-close`. `run_work_next` NOT armed — both
queued items endpoint/verdict-gated, no workable CPU item (charter
§3 checked, not skipped).*

*Updated 2026-08-20 22:11–22:1xZ (tick) — **democlean riding tick
ALL-GREEN with a banked eval row: step 1750/3000, loss 0.3686, and
the eval-1750 probe row landed at 5.454 — exactly on the convicted
anchor's 5.45@1750 (onerig: 4.56). Level has now converged onto
the convicted curve; the discriminator is squarely the shape ahead
— convicted went 5.47@2000 → 6.59@2250 (the elevation that never
retraced). Eval-2000 (~23:1xZ) and the 2250–2750 window are where
this run either mimics the poison signature or departs from it.
Record-only as registered, no gate, nothing to decide.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 1750/3000 at
22:12Z, loss 0.376 → 0.3686 (last train rows 0.35–0.37 — the
oscillation band itself has shifted down, downtrend intact), ~17.7
s/step window (+70 steps since 21:51, 3.3 steps/min wall —
eval-1750 pause in-window), vram instant 65.0 GiB during eval vs
≤75 gate (log peak 62.24), babysit exit 0, no gate crossings, ~5.6
h to 3000 → **endpoint ~03:4x–03:5xZ 08-21**. GPU liveness 5
procs. **Probe curve: 11.82@250 → 8.14@500 → 7.90@750 → 6.49@1000
→ 5.95@1250 → 5.72@1500 → 5.454@1750** (rows land every 250 — the
earlier "next row @2000" note was wrong, saves and evals are on
separate cadences). Read vs anchors: our 1500→1750 drop −0.27 sits
between convicted −0.17 and onerig −0.38, and the level now equals
convicted exactly. Infra: disk 125G free (post-prune steady
state), RAM available 48G — tenth read in the 47–49G band, leak
watch closed; pruner unit active, log correctly quiet since the
21:09:23Z step-1000 prune (next pass with work after the step-2000
save).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, eval-1750 row banked +
anchor comparison, train-jsonl direct read (babysit's None fields
were the eval row at head — benign), disk + RAM + pruner-log
checks, queue validate green (depth 2, 14 open), now.md + archive
roll.

**Next**: the step-2000 boundary (~23:1x–23:2xZ) is now the most
informative pre-endpoint read: eval-2000 vs convicted's 5.47@2000
(record-only) + pruner verify (step-1500 optimizer.pt prunes after
the save). Then the 2250–2750 elevation watch — with the level
converged onto convicted, elevation-vs-retrace there is the
sharpest shape signal before sim100. Endpoint session
(~03:4x–03:5xZ 08-21) owns `democlean-endpoint-close`.
`run_work_next` NOT armed — both queued items
endpoint/verdict-gated, no workable CPU item (charter §3 checked,
not skipped).*

## Utilization footer

Session 2026-08-20 22:33–22:4xZ (tick; `democlean` riding, ~8.1
GPU-h elapsed of ~13.5 projected vs the 17 gate): **plain riding
tick ALL-GREEN — babysit exit 0, step 1830/3000 at 22:33Z, loss
0.3686 → 0.3486 (−0.02 clean continuation, no bounce this window;
500→1830 downtrend 0.589 → 0.3486), vram 62.24/75, no gate
crossings, 16.019 s/step window (3.8 steps/min wall), ~5.2 h to
3000 → endpoint ~03:4xZ 08-21. Probe curve unchanged through
5.454@1750 (= convicted's 5.45 exactly); next row @2000 ~23:1xZ.
Infra steady: disk 125G, RAM available 48G (eleventh read in the
47–49G plateau band); pruner active, correctly quiet since
21:09:23Z. Discord fully quiet (read empty, inbox empty, history
-n 5 all own posts, no reactions); queue validate green depth 2
(14 open); run_work_next NOT armed — both queued items
endpoint/verdict-gated. Next boundary: step-2000 save
~23:1x–23:2xZ (eval-2000 vs convicted 5.47 + prune verify:
step-1500 optimizer.pt).**

Session 2026-08-20 22:11–22:1xZ (tick; `democlean` riding, ~7.8
GPU-h elapsed of ~13.5 projected vs the 17 gate): **riding tick
ALL-GREEN + eval-1750 row banked — babysit exit 0, step 1750/3000
at 22:12Z, loss 0.376 → 0.3686 (train rows 0.35–0.37, oscillation
band shifted down, downtrend intact), vram instant 65.0 during
eval vs ≤75 (log peak 62.24), no gate crossings, ~17.7 s/step
window (eval pause in-window, 3.3 steps/min wall), ~5.6 h to 3000
→ endpoint ~03:4x–03:5xZ 08-21. **Probe row 5.454@1750 — exactly
on convicted's 5.45@1750** (onerig 4.56); 1500→1750 drop −0.27
between convicted −0.17 and onerig −0.38; level converged onto the
convicted curve, so the 2250–2750 elevation-vs-retrace is now the
sharpest pre-sim100 shape signal (convicted: 5.47@2000 →
6.59@2250). Record-only per pre-reg. Corrected: probe rows land
every 250 steps, not with saves. Infra steady: disk 125G, RAM
available 48G (tenth read in the 47–49G plateau band); pruner
active, correctly quiet since 21:09:23Z. Discord fully quiet;
queue validate green depth 2 (14 open); run_work_next NOT armed —
both queued items endpoint/verdict-gated. Next boundary: step-2000
save ~23:1x–23:2xZ (eval-2000 vs convicted 5.47 + prune verify:
step-1500 optimizer.pt).**

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
