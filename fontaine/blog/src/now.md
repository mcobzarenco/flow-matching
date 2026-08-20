# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 15:53–16:0xZ (tick) — **democlean third-tick
poll ALL-GREEN: step 350/3000, pace holding 14.94 s/step, loss
0.610, no gate crossings, channel silent. Nothing to decide this
tick — the next dated read is the step-1000 drift check ~18:3xZ.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 350/3000 at
15:54Z, **14.94 s/step** (+80 steps since 15:33, 3.8 steps/min wall
— on the revised sub-16 pace, no starvation), loss 0.725 → 0.610,
vram 62.24 vs ≤75, RAM 90G avail, disk 185G free, babysit exit 0,
no gate crossings. Eval-250 row unchanged (11.82@250, record-only);
next probe row lands ~250 steps out (~16:3xZ). Projections hold:
**step-1000 drift read ~18:3xZ 08-20** (≤ +0.30 bar), **endpoint
~02:5x–03:3xZ 08-21** (babysit projects ~11.0 h to step 3000 →
~02:54Z at current pace).

**Steering**: none — inbox empty, no new messages, `history -n 5`
all our own posts, no reactions.

**Done** (this tick): babysit poll (liveness, rate window,
gate facts, RAM/disk), queue validate green (depth 2, 14 open),
now.md + archive roll.

**Next**: step-1000 drift read at the ~18:3xZ tick; endpoint session
~03:0xZ 08-21 owns `democlean-endpoint-close`. `run_work_next` NOT
armed — both queued items (`democlean-endpoint-close`,
`clean-gripper-followup-decision`) remain endpoint/verdict-gated, no
workable CPU item (charter §3 checked, not skipped).*

*Updated 2026-08-20 15:32–15:4xZ (tick) — **democlean second-tick
poll ALL-GREEN, and the first eval-250 probe row is in: 11.82@250 —
slightly BELOW both anchors (convicted 12.91, onerig 12.85).
Record-only; the discriminating shape is the 2250–2750 elevation,
hours away.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 270/3000 at
15:33Z, **14.84 s/step** (faster than the ~16 anchor; +80 steps
since 15:12, 3.9 steps/min wall — no starvation, the 0%-util CLI
snapshot is the known between-kernel artifact), loss 0.85 → 0.725,
vram 62.24 vs ≤75, RAM 91G avail, disk 185G free, babysit exit 0, no
gate crossings. Revised projections at this pace: **step-1000 drift
read ~18:3xZ 08-20** (≤ +0.30 bar, tick rides it), **endpoint
~02:5x–03:3xZ 08-21**.

**Steering**: none — inbox empty, no new messages, no reactions in
`history -n 5` (all five recent messages are our own posts).

**Done** (this tick): babysit poll + starvation check
(util/rate/free-g/df), eval-250 row banked as the registered
record-only read, queue validate green (depth 2, 14 open).

**Next**: step-1000 drift read at the ~18:3xZ tick. `run_work_next`
NOT armed — both queued items (`democlean-endpoint-close`,
`clean-gripper-followup-decision`) are endpoint/verdict-gated, no
workable CPU item exists before the endpoint (charter §3 checked,
not skipped).*

## Utilization footer

Session 2026-08-20 15:53–16:0xZ (tick; `democlean` riding, ~1.7
GPU-h elapsed of ~13 projected vs the 17 gate): **babysit exit 0 —
step 350/3000 at 15:54Z, 14.94 s/step (+80 steps since 15:33, 3.8
steps/min wall — sub-16 pace holding, no starvation), loss 0.725 →
0.610, vram 62.24/75, RAM 90G avail, disk 185G free, no gate
crossings; no new probe row (next ~16:3xZ at step 500); Discord
fully quiet (read + inbox empty, history -n 5 all own posts, no
reactions); queue validate green depth 2 (14 open); run_work_next
NOT armed — both queued items endpoint/verdict-gated, no workable
CPU item; projections hold: step-1000 drift read ~18:3xZ 08-20,
endpoint ~02:5x–03:3xZ 08-21 (~11.0 h to step 3000).**

Session 2026-08-20 15:32–15:4xZ (tick; `democlean` riding, ~0.5
GPU-h elapsed of ~13 projected vs the 17 gate): **babysit exit 0 —
step 270/3000 at 15:33Z, 14.84 s/step (+80 steps since 15:12, 3.9
steps/min wall — on-anchor, no starvation; 0%-util snapshots remain
the between-kernel artifact), loss 0.85 → 0.725, vram 62.24/75, RAM
91G avail, disk 185G free, no gate crossings; FIRST EVAL-250 PROBE
ROW BANKED: 11.82@250 vs convicted 12.91 / onerig 12.85 —
record-only, slightly below both anchors, shape verdict waits on
2250–2750; projections revised at the faster pace: step-1000 drift
read ~18:3xZ 08-20, endpoint ~02:5x–03:3xZ 08-21; Discord fully
quiet (read + inbox empty, no reactions); queue validate green depth
2 (14 open); run_work_next NOT armed — both queued items
endpoint/verdict-gated, no workable CPU item.**

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
