# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-20 16:35–16:4xZ (tick) — **democlean fourth-tick
poll: run ALL-GREEN and the eval-500 row is in (8.14@500, between
both anchors) — but the tick caught a DISK-FULL trajectory and fixed
it: the step-500 checkpoint landed at 44G (32G optimizer.pt), 6
saves = 264G vs 143G free, unpruned the step-2500 save would have
crashed the run ~01:4xZ. Keep-latest-optimizer pruner unit now
running; weights never touched.***

**Status**: `fontaine-v2-joint-pdnorm-democlean` step 520/3000 at
16:36Z, 16.88 s/step window (+80 steps since 16:15, 3.8 steps/min
wall), loss 0.604 → 0.589, vram 62.24 vs ≤75, babysit exit 0, no
gate crossings. **Eval-500 probe row banked (record-only): 8.14@500
vs convicted 8.24 / onerig 8.05** — clean-alone sits between the
anchors, nothing discriminating yet (that's the 2250–2750
elevation). Disk 143G free after the 44G step-500 save; **pruner
`fontaine-democlean-ckpt-prune` live 16:39Z**
(`scripts/prune_superseded_optimizers.sh`, 10-min loop: deletes only
superseded optimizer.pt after the newer one is write-complete →
peak need ~124G < 143G). RAM available 90G→48G across the save
(trainer RSS high-water ~139G, /dev/shm 20G) — expected to plateau,
escalate if it keeps falling toward <20G. Projections: **step-1000
drift read ~18:4xZ** (≤ +0.30 bar), **endpoint ~03:2x–04:1xZ
08-21**.

**Steering**: none — inbox empty, no new messages, `history -n 5`
all our own posts, no reactions. Intervention posted in-channel
(1540037722909974529).

**Done** (this tick): babysit poll, eval-500 row banked, disk
repricing (44G/ckpt measured, no built-in retention — the 14:2xZ
'risk cleared' used post-prune onerig footprint, ~2× off), pruner
script + systemd unit launched + log-verified, babysit.toml disk/RAM
anchors rewritten, Discord post, queue validate green (depth 2, 14
open).

**Next**: step-1000 drift read ~18:4xZ tick — also verify the pruner
log shows step-500's optimizer.pt pruned after the step-1000 save,
and re-read RAM available (leak vs plateau). Endpoint session owns
`democlean-endpoint-close`. `run_work_next` NOT armed — both queued
items endpoint/verdict-gated, no workable CPU item (charter §3
checked, not skipped).*

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

## Utilization footer

Session 2026-08-20 16:35–16:4xZ (tick; `democlean` riding, ~2.4
GPU-h elapsed of ~13.5 projected vs the 17 gate): **babysit exit 0 —
step 520/3000 at 16:36Z, 16.88 s/step window (+80 steps since 16:15,
3.8 steps/min wall), loss 0.604 → 0.589, vram 62.24/75, no gate
crossings; EVAL-500 ROW BANKED: 8.14@500 vs convicted 8.24 / onerig
8.05 (record-only, between anchors); DISK-FULL TRAJECTORY CAUGHT +
FIXED: step-500 checkpoint measured 44G (32G optimizer.pt) × 6 saves
= 264G vs 143G free, no built-in retention → unpruned step-2500 save
would crash ~01:4xZ; keep-latest-optimizer pruner unit
`fontaine-democlean-ckpt-prune` live 16:39Z (superseded optimizer.pt
only, weights untouched, peak need ~124G), babysit.toml anchors
repriced, in-channel post 1540037722909974529; RAM available
90G→48G across the save (RSS high-water, shm fine) — watch for
plateau vs leak at next tick; Discord otherwise quiet (read + inbox
empty, history all own posts); queue validate green depth 2 (14
open); run_work_next NOT armed — both queued items
endpoint/verdict-gated; boundaries: step-1000 drift read ~18:4xZ,
endpoint ~03:2x–04:1xZ 08-21.**

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
