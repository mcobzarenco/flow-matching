# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-22 06:40–06:4xZ (tick) — **routine leg B poll:
healthy — step 170/500 arm 1 onerig, 15.64 s/step windowed, loss
1.80 → 1.77, vram 62.4 vs the 71 gate. Discord quiet, queue green,
`run_work_next` armed.***

**Status**: `fontaine-squint-adapt` LIVE and healthy at step 170/500
(arm 1 onerig): 15.64 s/step windowed (4.1 steps/min since the
06:38Z sample), loss 1.77, probe `eval_chunk_mae` 4.20@100
(record-only first wear, no kill bar), vram 62.41 vs the 71 gate,
~1.4 h to step 500. Arm roll ~08:1xZ (step counter reset = the
roll), unit done ~10:3xZ.

**Steering**: none — inbox empty, `read` surfaced only our own
06:39Z harness-complete post, `history -n 5` all own posts, no
reactions.

**Done** (this tick): babysit poll (liveness 5 procs, both gates
green, exit 0), Discord read + history, queue validate green (depth
2, 15 open, stamp 06:30Z), marker confirmed armed (06:39Z), now.md
keep-3 + footer keep-2 rolls to archive 08-22.

**Next**: chained work session works `ch0-shift-isolation-prereg`
during the leg B window. Boundaries: arm roll ~08:1xZ (jsonl
repoints to the democlean stem), leg B done ~10:3xZ → launch
`fontaine-squint-gate12` (leg C; phase A doubles as the live
smoke).*

*Updated 2026-08-22 06:02–06:5xZ (work) — **Gate-1/2 harness CPU build
COMPLETE while leg B rides: the whole band-pilot → Gate-1 → Gate-2 →
reads path is now one launch-ready detached unit, with the
pre-registered McNemar + KM/KS analysis oracle-tested and the CDF
panel chart rendering in the house scheme.***

**Status**: `fontaine-squint-adapt` LIVE and healthy at step 160/500
(arm 1 onerig): 15.96 s/step (≈ the 16.5 smoke), loss 2.17 → 1.80
since last poll, first probe fact `eval_chunk_mae` 4.20@100
(record-only first wear, no kill bar), vram 62.41 vs the 71 gate.
Arm roll ~08:1xZ, unit done ~10:3xZ.

**Steering**: none — inbox empty, `read` empty at both babysit
checkpoints, `history -n 5` all own posts.

**Done** (this session, `b066b8d3`): `squint-gate2-harness` CPU build
COMPLETE — (1) `squint_gate12_leg.sh`: leg C as one detached unit
(band pilot n=20 seeds 0–19 both tasks → 20–80 band verdicts →
adapted onerig completed to n=100, pilot rows a reusable prefix of
the paired 0–99 cell → Gate-1 best-task ≥20/100 with F-instrument
exit skipping Gate-2 spend → adapt_democlean cells + unadapted @3000
riders on the sim100 worn row → reads; leg-B-active +
foreign-compute guards, kill-safe serve pattern). (2)
`squint_screen_read.py`: McNemar exact p + seed-0 10k bootstrap
reused from the sim100 machinery; KM/KS co-primary per the
eval-design tier-3 sketch (fixed-horizon censoring reduces KM to the
first_true_step ECDF; per-predicate KS + signed ΔAUC, macro-KS
paired label-swap permutation p, seed-clustered bootstrap CI95);
frozen expectation grid verbatim; CDF panel in the house dark scheme.
Oracle self-test green (hand-computed McNemar 0.109375, censoring,
KS extremes, permutation floor, band edges, grid cases) + synthetic
end-to-end smoke green incl. rendered panel. check.py green.

**Next**: `queue_cli.py next` → `ch0-shift-isolation-prereg` draft
(chained work session; `run_work_next` armed). Boundaries: arm roll
~08:1xZ 08-22 (step reset = the roll, not a stall), leg B done
~10:3xZ → launch `fontaine-squint-gate12` (leg C; command in the
script header — phase A doubles as the live smoke of the served-arm
path).*

*Updated 2026-08-22 05:58–06:0xZ (tick) — **first poll of the leg B
adaptation run six minutes after launch: healthy — windowed rate
15.75 s/step (better than the 16.5 smoke; babysit's 19.9 was
warmup-inclusive), loss 5.68 → 4.25 by step 20, vram 62.4 vs the
71 gate. Discord quiet, queue green, `run_work_next` armed.***

**Status**: `fontaine-squint-adapt` LIVE and healthy at step 20/500
(arm 1 onerig): `s_per_step` 15.747 windowed → arm 1 done ~08:05Z,
democlean roll after; util bursty (0–100% samples) but the rate
matches the frozen recipe's smoke, so no starvation call — recipe is
frozen Slot 6 regardless. Grad norm 7.0, both loss heads moving
(ar 2.78 / flow 1.50).

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (liveness 5 procs, gates green),
6-sample util check + jsonl windowed-rate read (the
max-gpu-utilization first-poll rule), Discord read + history, queue
validate green (depth 2, 15 open, stamp 05:55Z), marker confirmed
armed (05:55Z).

**Next**: chained work session works the CPU queue during the leg B
window (`ch0-shift-isolation-prereg` draft, gate2-harness remaining
slots). Boundaries: arm roll ~08:05Z (step counter reset = the roll,
not a stall; jsonl repoints to the democlean stem), unit done
~10:3xZ → Gate-1 band pilot.*

## Utilization footer

Session 2026-08-22 06:40–06:4xZ (tick; 0 marginal GPU-h — leg B
riding): **routine healthy poll of `fontaine-squint-adapt` — step
170/500 arm 1, 15.64 s/step windowed, loss 1.77, vram 62.4/71,
probe record-only 4.20@100. Discord fully quiet; queue green depth
2 (15 open); `run_work_next` armed. Next boundary: arm roll
~08:1xZ.**

Session 2026-08-22 06:02–06:5xZ (work; exploit — Squint Gate-1/2
harness CPU build, 0 marginal GPU-h, leg B riding): **band-pilot →
Gate-1 → Gate-2 → reads orchestrator + the pre-registered McNemar +
KM/KS analysis landed launch-ready (`b066b8d3`), oracle self-test +
synthetic end-to-end smoke green, CDF panel rendering verified.
Babysits 06:13Z/06:38Z healthy (step 160/500, loss 1.80, vram
62.4/71); Discord quiet; queue green depth 2 (15 open). Leg C
launches at the first free GPU window after leg B (~10:3xZ).**

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
