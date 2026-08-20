# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 21:08–21:1xZ (tick) — **democlean step-1500
boundary tick ALL-GREEN: step 1520/3000, eval-1500 row banked
5.72, pruner's second prune verified live (32G back, disk 125G),
loss bounce resolved (0.4226 → 0.3926). Nothing to decide.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 1520/3000 at
21:09Z, loss 0.4226 → 0.3926 — last tick's +0.018 bounce did NOT
compound, resolved as window noise, downtrend intact. 16.307
s/step window (3.3 steps/min wall), vram 62.24 vs ≤75, babysit
exit 0, no gate crossings, ~6.7 h to 3000 → **endpoint ~03:5xZ
08-21**. **Probe row @1500 banked: 5.72** (curve 11.82@250 →
8.14@500 → 7.90@750 → 6.49@1000 → 5.95@1250 → 5.72@1500;
record-only, no post). 1250→1500 drop −0.23 sits between convicted
(−0.10) and onerig (−0.65); level 5.72 still above both anchors at
1500 (convicted 5.62, onerig 4.94), converging from above; no
elevation signature — that watch opens at 2250–2750. **Pruner
verified through a full cycle**: step-1500 save write-complete
21:02:33Z (44G), pruner pass 21:09:23Z pruned step-1000
optimizer.pt — caught the trough live: disk 93G free post-save →
125G post-prune, so remaining cycles trough ~81G @2000 and shallower
thereafter, comfortable. RAM available 47G — seventh consecutive
read in the 47–49G band, now spanning a full save cycle; leak
watch stays closed.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, eval-1500 row banked, pruner
second-prune verified (log + checkpoint dir + df, trough
observed), RAM re-read, loss re-read (bounce resolved), queue
validate green (depth 2, 14 open), now.md + archive roll.

**Next**: plain riding ticks; next boundary is the step-2000 save
(~23:2x–23:3xZ) — eval-2000 row (record-only) + pruner verify
(step-1500 optimizer.pt should prune after the save). Endpoint
session (~03:5xZ 08-21) owns `democlean-endpoint-close` (sim100 +
panel guard + paired reads + verdict grid). `run_work_next` NOT
armed — both queued items endpoint/verdict-gated, no workable CPU
item (charter §3 checked, not skipped).*

*Updated 2026-08-20 20:26–20:3xZ (tick) — **democlean plain riding
tick ALL-GREEN: step 1370/3000, loss 0.4226, nothing to decide.
Loss window bounced +0.018 (0.4043 → 0.4226) — mid-cell noise
against an intact longer downtrend, watching not acting. Pace
16.185 s/step; infra plateau holds — disk 135G, RAM 48G sixth
consecutive read in the 47–49G band, pruner correctly quiet.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 1370/3000 at
20:27Z, loss 0.4043 → 0.4226 (+0.018 window bounce — the 500→1300
trend 0.589 → 0.40x stays intact; noise unless it compounds
tick-over-tick), 16.185 s/step window (+70 steps since 20:06, 3.3
steps/min wall), vram 62.24 vs ≤75, babysit exit 0, no gate
crossings, ~7.3 h to 3000 → **endpoint ~03:5xZ 08-21**. Probe
curve unchanged (11.82@250 → 8.14@500 → 7.90@750 → 6.49@1000 →
5.95@1250; next row @1500 lands with the step-1500 save). Infra:
disk 135G free (no save since 1000), RAM available 48G — sixth
read in the 47–49G band, leak watch stays closed; pruner unit
active, log unchanged since the 18:49:22Z step-500 prune (correct
— next pass with work after the step-1500 save).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts (latest: the 18:43Z shape post), no reactions.

**Done** (this tick): babysit poll, disk + RAM + pruner-log checks,
queue validate green (depth 2, 14 open), now.md + archive roll.

**Next**: the step-1500 save (~21:0x–21:1xZ) is the next boundary —
the ~20:5x/~21:1x tick owns the eval-1500 row (record-only) +
pruner verify (step-1000 optimizer.pt should prune after the save)
+ RAM re-read across the save cycle + a loss re-read (confirm the
+0.018 bounce didn't compound). Endpoint session (~03:5xZ 08-21)
owns `democlean-endpoint-close` (sim100 + panel guard + paired
reads + verdict grid). `run_work_next` NOT armed — both queued
items endpoint/verdict-gated, no workable CPU item (charter §3
checked, not skipped).*

## Utilization footer

Session 2026-08-20 21:08–21:1xZ (tick; `democlean` riding, ~6.8
GPU-h elapsed of ~13.5 projected vs the 17 gate): **step-1500
boundary tick ALL-GREEN — babysit exit 0, step 1520/3000 at
21:09Z, loss 0.4226 → 0.3926 (last tick's bounce did not compound,
downtrend intact), vram 62.24/75, no gate crossings, 16.307 s/step
window (3.3 steps/min wall), ~6.7 h to 3000 → endpoint ~03:5xZ
08-21. Eval-1500 row banked: 5.72 — 1250→1500 drop −0.23 between
convicted (−0.10) and onerig (−0.65), level still above both
anchors, converging from above, no elevation signature
(record-only, no post). Pruner verified through a full cycle:
step-1500 save write-complete 21:02:33Z, 21:09:23Z pass pruned
step-1000 optimizer.pt — trough caught live, disk 93G post-save →
125G post-prune, remaining troughs ~81G and shallower. RAM
available 47G (seventh read in the 47–49G band, spanning a full
save cycle; leak watch closed). Discord fully quiet (read empty,
inbox empty, history -n 5 all own posts, no reactions); queue
validate green depth 2 (14 open); run_work_next NOT armed — both
queued items endpoint/verdict-gated. Next boundary: step-2000 save
~23:2x–23:3xZ (eval-2000 + prune verify).**

Session 2026-08-20 20:26–20:3xZ (tick; `democlean` riding, ~6.2
GPU-h elapsed of ~13.5 projected vs the 17 gate): **plain riding
tick ALL-GREEN — babysit exit 0, step 1370/3000 at 20:27Z, loss
0.4043 → 0.4226 (+0.018 window bounce, longer downtrend intact —
watching, not acting), vram 62.24/75, no gate crossings, 16.185
s/step window (3.3 steps/min wall), ~7.3 h to 3000 → endpoint
~03:5xZ 08-21. Probe curve unchanged through 5.95@1250; next row
@1500 lands with the step-1500 save. Infra steady: disk 135G, RAM
available 48G (sixth read in the 47–49G plateau band, leak watch
stays closed); pruner active, log correctly quiet since the
18:49:22Z step-500 prune. Discord fully quiet (read empty, inbox
empty, history -n 5 all own posts, no reactions); queue validate
green depth 2 (14 open); run_work_next NOT armed — both queued
items endpoint/verdict-gated. Next boundary: step-1500 save
~21:0x–21:1xZ (eval-1500 + prune verify + loss re-read).**

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
