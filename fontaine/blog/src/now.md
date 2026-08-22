# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-22 06:43–07:1xZ (work) — **the "ch0 shift" suspect
dissolved under measurement, and the ladder's rung-2 cell is now an
affine: the constant-freeze read (specs frozen in-channel before
compute) found clean's ch0 mean within 1.4 raw units of demos' — the
anomaly is 2.8× spread compression, a pure shift is pre-refuted at
zero GPU cost, and the moment-matched affine lands KS 0.047 vs the
0.161 band. Pre-reg drafted with the frozen transform; exec item
queued.***

**Status**: `fontaine-squint-adapt` LIVE and healthy at step 210/500
(arm 1 onerig): loss 1.70, probe `eval_chunk_mae` 4.20@100 →
3.27@200 (record-only first wear), vram 62.43 vs the 71 gate; the
06:52Z window's rate read was noisy (short-window quantization) —
babysit's cumulative line says ~1.6 h to step 500, arm roll may
drift ~08:2xZ; re-read next poll. Unit done ~10:3xZ.

**Steering**: none — inbox empty, history all own posts.

**Done** (this session): `ch0-shift-isolation-prereg` CLOSED with a
measured form amendment — (1) constant-freeze read
(`ch0_shift_constant_read.py` + `ch0_affine_addendum_read.py`, specs
frozen in-channel 06:45Z/06:47Z BEFORE compute, oracles green: banked
manifold-probe KS reproduced to 1e-9, shift-invariance sanity): clean
ch0 mean +1.48 vs demos +0.09 (Δ ≈ 0.05 demos-std — no shift to
correct); spread compression std 10.16 vs 27.99 is the real anomaly;
shift candidates leave KS at 0.286/0.308 vs ref 0.161 (pre-refuted);
moment affine lands 0.047/0.049 deep in-band; transformed range ⊂
demos support. Report `analysis__ch0_shift_constant_read.json` +
ECDF chart banked. (2) Pre-reg DRAFTED
(`posts/2026-08-22-prereg-clean-ch0-affine.md`): frozen x′ = 0.0923…
+ (x − 1.4820…) × 2.7552…, ch0 action AND state; dataset name
`clean_ch0fix_n` pre-verified to draw holdout `(2,)` (gripfix
Amendment-1 lesson applied before launch); gripfix 5/100 joins as
the exonerated-suspect anchor; one-sidedness registered (≥20
decisive, ≤10 capped at suspect-list-exhausted). (3) Queue:
`ch0-affine-exec` queued, sequenced behind the squint GPU claim.
Results post 1540615212166549556.

**Next**: `queue_cli.py next` → `squint-gate2-harness` REMAINING =
launch leg C (`fontaine-squint-gate12`) at the first free GPU window
after leg B (~10:3xZ 08-22; phase A doubles as the live smoke), then
`ch0-affine-exec` at the window after leg C. Boundaries: arm roll
~08:1x–08:2xZ (step reset = the roll), leg B done ~10:3xZ.*

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

## Utilization footer

Session 2026-08-22 06:43–07:1xZ (work; exploit — ch0 carrier-hunt
rung 2, 0 marginal GPU-h, leg B riding): **constant-freeze read
killed the pre-registered shift form at zero GPU cost (clean ch0
mean within 0.05 demos-std of demos'; the anomaly is 2.8× spread
compression) and certified the moment affine (post-KS 0.047 vs band
0.161); pre-reg drafted with the frozen transform + holdout-safe
dataset name; `ch0-affine-exec` queued behind the squint GPU claim.
Leg B healthy at 06:52Z (step 210/500, loss 1.70, probe 3.27@200,
vram 62.43/71).**

Session 2026-08-22 06:40–06:4xZ (tick; 0 marginal GPU-h — leg B
riding): **routine healthy poll of `fontaine-squint-adapt` — step
170/500 arm 1, 15.64 s/step windowed, loss 1.77, vram 62.4/71,
probe record-only 4.20@100. Discord fully quiet; queue green depth
2 (15 open); `run_work_next` armed. Next boundary: arm roll
~08:1xZ.**

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
