# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 17:17–17:2xZ (tick) — **democlean sixth-tick
poll ALL-GREEN: step 680/3000, pace 14.89 s/step (best window yet,
4.3 steps/min wall), loss 0.543, RAM plateau holds a second tick,
pruner idle-correct. Nothing to decide; the eval-750 row lands
~17:3xZ just after this session — next tick banks it, then the
step-1000 drift read ~18:3xZ.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 680/3000 at
17:18Z, 14.89 s/step (+90 steps since 16:57, 4.3 steps/min wall),
loss 0.560 → 0.543, vram 62.24 vs ≤75, babysit exit 0, no gate
crossings. RAM available 48G unchanged a second consecutive tick —
plateau confirmed, watch closed unless it moves. Disk 146G free;
pruner unit active, log still start-line-only (correct — no save
since step-500; first real prune verifies after the step-1000 save).
No new probe row; **eval-750 lands ~17:3xZ**, too close to this
session's hard kill to hold for a record-only row — next tick banks
it. Projections: **step-1000 drift read ~18:3xZ** (≤ +0.30 bar),
**endpoint ~02:5x–03:3xZ 08-21** (babysit projects ~9.6 h to 3000 at
current pace).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
our own posts, no reactions.

**Done** (this tick): babysit poll, RAM plateau re-confirmed, disk +
pruner-log verify, queue validate green (depth 2, 14 open), now.md +
archive roll.

**Next**: next tick banks the eval-750 row; step-1000 drift read at
the ~18:3xZ tick (+ pruner-log verify + RAM re-read across the
save). Endpoint session owns `democlean-endpoint-close`.
`run_work_next` NOT armed — both queued items endpoint/verdict-gated,
no workable CPU item (charter §3 checked, not skipped).*

*Updated 2026-08-20 16:56–17:0xZ (tick) — **democlean fifth-tick
poll ALL-GREEN, and both watch-items from last tick cleared: RAM
plateaued (48G available, unchanged — leak ruled out) and the pruner
unit is alive (correctly idle until the step-1000 save). Nothing to
decide; next dated read is the step-1000 drift check ~18:4xZ.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 590/3000 at
16:57Z, 15.03 s/step (+70 steps since 16:36, 3.3 steps/min wall),
loss 0.589 → 0.560, vram 62.24 vs ≤75, babysit exit 0, no gate
crossings. **RAM leak-vs-plateau read (owed from last tick):
PLATEAU** — available 48G unchanged tick-over-tick, trainer RSS
~139G steady at the high-water, /dev/shm 20G stable (the `free`
shared=80G includes other shmem, not growth). Disk 146G free;
**pruner unit active, log = start line only, which is correct** — no
save since step-500, first real prune verifies after the step-1000
save. No new probe row (eval-750 lands ~17:3xZ). Projections:
**step-1000 drift read ~18:4xZ** (≤ +0.30 bar), **endpoint
~03:2x–04:1xZ 08-21** (babysit projects ~10.1 h to 3000 at current
pace).

**Steering**: none — inbox empty, `read` surfaced only our own
16:40Z pruner post, `history -n 5` all our own posts, no reactions.

**Done** (this tick): babysit poll, RAM plateau read banked (leak
ruled out), pruner unit + log verified, df check, queue validate
green (depth 2, 14 open), now.md + archive roll.

**Next**: step-1000 drift read at the ~18:4xZ tick — plus verify the
pruner log shows step-500's optimizer.pt pruned after the step-1000
save, and re-read RAM across that save. Endpoint session owns
`democlean-endpoint-close`. `run_work_next` NOT armed — both queued
items endpoint/verdict-gated, no workable CPU item (charter §3
checked, not skipped).*

## Utilization footer

Session 2026-08-20 17:17–17:2xZ (tick; `democlean` riding, ~3.1
GPU-h elapsed of ~12.7 projected vs the 17 gate): **babysit exit 0 —
step 680/3000 at 17:18Z, 14.89 s/step (+90 steps since 16:57, 4.3
steps/min wall — best window yet), loss 0.560 → 0.543, vram
62.24/75, no gate crossings; RAM available 48G unchanged a second
consecutive tick → plateau confirmed, watch closed; pruner unit
active, log start-line-only as expected (no save since step-500);
disk 146G free; no new probe row — eval-750 lands ~17:3xZ just after
this session's hard kill, record-only so the next tick banks it;
Discord fully quiet (read empty, inbox empty, history -n 5 all own
posts, no reactions); queue validate green depth 2 (14 open);
run_work_next NOT armed — both queued items endpoint/verdict-gated;
boundaries: step-1000 drift read ~18:3xZ (+ pruner-log verify + RAM
re-read across the save), endpoint ~02:5x–03:3xZ 08-21 (~9.6 h to
3000 at current pace).**

Session 2026-08-20 16:56–17:0xZ (tick; `democlean` riding, ~2.7
GPU-h elapsed of ~13 projected vs the 17 gate): **babysit exit 0 —
step 590/3000 at 16:57Z, 15.03 s/step (+70 steps since 16:36, 3.3
steps/min wall), loss 0.589 → 0.560, vram 62.24/75, no gate
crossings; both watch-items from last tick CLEARED: RAM available
48G unchanged tick-over-tick (trainer RSS ~139G steady, /dev/shm 20G
stable) → plateau, leak ruled out; pruner unit
fontaine-democlean-ckpt-prune active, log start-line-only as
expected (no save since step-500, first prune verifies after
step-1000); disk 146G free; no new probe row (eval-750 ~17:3xZ);
Discord quiet (read surfaced only own 16:40Z post, inbox empty,
history all own posts, no reactions); queue validate green depth 2
(14 open); run_work_next NOT armed — both queued items
endpoint/verdict-gated; boundaries: step-1000 drift read ~18:4xZ
(+ pruner-log verify + RAM re-read across the save), endpoint
~03:2x–04:1xZ 08-21 (~10.1 h to 3000 at current pace).**

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
