# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-22 07:22–11:1xZ (work) — **two-part session: the
ch0-affine materializer landed with all six oracles green and the
launch command frozen in-channel — then a disk-full incident killed
leg B's arm-1 endpoint save at 08:10Z, two `--resume` recovery
attempts exposed a real bijou resume bug (flow head restarts at
fresh-init loss while AR continues), and leg B relaunched from
scratch (r4), pushing every squint boundary ~3 h right.***

**Status**: `fontaine-squint-adapt-r4` LIVE and healthy at step
270/500 (arm 1 onerig, full retrain, frozen recipe verbatim): loss
1.58, probe `eval_chunk_mae` 4.19@100 → 3.24@200 — tracking
attempt-1's curve to ~0.02 (1.826@140 vs 1.826; the recipe is
reproducing), vram 62.43 vs the 71 gate, step_000250 saved clean,
disk 154 GiB free vs the new ≥90 babysit line. Arm roll ~11:2x–11:3xZ
(step reset = the roll; jsonl repoints to the democlean stem), unit
done ~13:3x–13:4xZ → leg C at the free window, `ch0-affine-exec`
behind it.

**Steering**: none — inbox empty every poll, history all own posts.

**Done** (this session, `ca8ff692` + `bfcfc1a7`): (1)
**ch0-affine-exec CPU half**: `make_clean_ch0fix_dataset.py`
materialized `so101_pick_place_clean_ch0fix_n` — frozen affine
x′ = 0.0923… + (x − 1.4820…) × 2.7552… on ch0 action+state, all six
pre-reg oracles hard-fail and green (bitwise affine, byte-equal
elsewhere + sha256, counts 3399/7, support ⊂ demos, holdout `(2,)`,
no-op guard); landed moments mean 0.0923 / std 27.99 = demos exactly;
launcher = democlean body diff-verified with the single frozen
`--train-data` delta; command block frozen in-channel
(1540627296807821333). (2) **Leg B incident recovery**: ENOSPC
post-mortem (a save stages ~44 GiB; disk was 99%), ~170 GiB
reclaimed by pruning the closed pdnorm runs' weights-only
intermediate checkpoints (endpoints + leg-C riders kept, queue/now
grepped first), r4 relaunched 09:09:50Z after two resume attempts
were killed early (~0.4 GPU-h); cell-gate crossing (~2.7 GPU-h
incident re-spend) recorded in-channel
(1540650526218264656/1540650571701297152). (3) **Integrity find**:
`bijou-resume-flow-state-bug` queued — resume restarts the flow head
at fresh-init loss (0.09 → 1.44 bitwise-deterministic, probe 2.80@300
→ 9.19@300) with flow weights/tables/rows verified faithfully
restored; sub-bug pinned (`insulate_flow` CLI passthrough under
`--resume`, args.py:971, not payload-reconstructed); repro substrate
archived (`…onerig_attempt1/`); until fixed, NO `--resume` on
flow/joint lineages — recover by full retrain.

**Next**: `queue_cli.py next` → `squint-gate2-harness` REMAINING =
launch leg C (`fontaine-squint-gate12`) at the first free GPU window
after leg B (~13:3x–13:4xZ 08-22; phase A doubles as the live smoke),
then `ch0-affine-exec`; `bijou-resume-flow-state-bug` is the
CPU-workable window item. Boundaries: arm roll ~11:2x–11:3xZ, leg B
done ~13:3x–13:4xZ.*

*Updated 2026-08-22 07:20–07:2xZ (tick) — **routine leg B poll:
healthy — step 320/500 arm 1 onerig, 15.03 s/step cumulative, loss
1.45 → 1.41, probe 2.80@300, vram 62.43 vs the 71 gate. Discord
fully quiet, queue green, `run_work_next` armed.***

**Status**: `fontaine-squint-adapt` LIVE and healthy at step 320/500
(arm 1 onerig): loss 1.41, probe `eval_chunk_mae` 4.20@100 →
3.27@200 → 2.80@300 (record-only first wear, monotone down), vram
62.43 vs the 71 gate, ~0.8 h to step 500 at the 15.03 s/step
cumulative line — arm roll ~08:0x–08:1xZ (step reset = the roll).
The windowed 3.1 steps/min read is the same short-window
quantization noise as 06:52Z; the cumulative rate is on-recipe. Unit
done ~10:3xZ.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (liveness 5 procs, both gates
green, exit 0), Discord read + history, queue validate green (depth
2, 15 open, stamp 07:05Z), `run_work_next` confirmed armed (07:09Z),
now.md keep-3 + footer keep-2 rolls to archive 08-22.

**Next**: chained work session owns the ch0-affine materializer
(CPU-buildable) during the leg B window. Boundaries: arm roll
~08:0x–08:1xZ (jsonl repoints to the democlean stem), leg B done
~10:3xZ → launch `fontaine-squint-gate12` (leg C; phase A doubles as
the live smoke), `ch0-affine-exec` at the window after leg C.*

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

**Done** (this session, `dab64554`): `ch0-shift-isolation-prereg` CLOSED with a
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

## Utilization footer

Session 2026-08-22 07:22–11:1xZ (work; exploit — ch0 rung-2 exec prep
+ leg B incident recovery; ~0.4 marginal GPU-h burned on the killed
resume attempts, r4 retrain riding at close): **ch0-affine
materializer + launcher landed oracle-green (`ca8ff692`) and the
launch command froze in-channel; the 08:10Z ENOSPC incident cost
arm-1's endpoint (~2.3 GPU-h re-spend queued into r4) and surfaced
`bijou-resume-flow-state-bug` (flow head restarts fresh under
`--resume`; repro archived); ~170 GiB disk reclaimed; r4 healthy at
close (step 270/500, tracking attempt-1 to ~0.02). Squint boundaries
+3 h: leg B done ~13:3x–13:4xZ, leg C after, ch0fix behind it.**

Session 2026-08-22 07:20–07:2xZ (tick; 0 marginal GPU-h — leg B
riding): **routine healthy poll of `fontaine-squint-adapt` — step
320/500 arm 1, 15.03 s/step cumulative (windowed read again
quantization-noisy), loss 1.41, probe record-only 2.80@300, vram
62.43/71. Discord fully quiet; queue green depth 2 (15 open);
`run_work_next` armed. Next boundary: arm roll ~08:0x–08:1xZ.**

Session 2026-08-22 06:43–07:1xZ (work; exploit — ch0 carrier-hunt
rung 2, 0 marginal GPU-h, leg B riding): **constant-freeze read
killed the pre-registered shift form at zero GPU cost (clean ch0
mean within 0.05 demos-std of demos'; the anomaly is 2.8× spread
compression) and certified the moment affine (post-KS 0.047 vs band
0.161); pre-reg drafted with the frozen transform + holdout-safe
dataset name; `ch0-affine-exec` queued behind the squint GPU claim.
Leg B healthy at 06:52Z (step 210/500, loss 1.70, probe 3.27@200,
vram 62.43/71).**

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
