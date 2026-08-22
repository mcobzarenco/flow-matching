# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-22 10:50–10:5xZ (tick) — **routine r4 poll: healthy
— step 380/500 arm 1 onerig, 14.97 s/step cumulative, loss 1.42,
probe 2.76@300 monotone down, vram 62.43/71, disk 154G vs the ≥90
line. Discord fully quiet, queue green depth 3, `run_work_next`
armed.***

**Status**: `fontaine-squint-adapt-r4` LIVE and healthy at step
380/500 (arm 1 onerig): loss 1.42, probe `eval_chunk_mae` 4.19@100 →
3.24@200 → 2.76@300 (record-only, monotone down, tracking attempt-1's
curve), vram 62.43 vs the 71 gate, disk 154 GiB free vs the ≥90
babysit line, ~0.5 h to step 500 at 14.97 s/step cumulative — arm
roll ~11:2xZ (step reset = the roll; jsonl repoints to the democlean
stem). Unit done ~13:3x–13:4xZ.

**Steering**: none — inbox empty, `read` empty, `history -n 5` all
own posts, no reactions.

**Done** (this tick): babysit poll (liveness 5 procs, both gates
green, exit 0), Discord read + history, queue validate green (depth
3, 16 open, stamp 09:16Z), disk check 154G, `run_work_next` confirmed
armed (10:23Z), now.md keep-3 + footer keep-2 rolls to archive 08-22.

**Next**: chained work session owns `bijou-resume-flow-state-bug`
(CPU) during the r4 window and catches the arm roll ~11:2xZ. Leg B
done ~13:3x–13:4xZ → launch `fontaine-squint-gate12` (leg C; phase A
doubles as the live smoke), `ch0-affine-exec` at the window after.*

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

## Utilization footer

Session 2026-08-22 10:50–10:5xZ (tick; 0 marginal GPU-h — r4 leg B
riding): **routine healthy poll of `fontaine-squint-adapt-r4` — step
380/500 arm 1, 14.97 s/step cumulative, loss 1.42, probe record-only
2.76@300 monotone down, vram 62.43/71, disk 154G vs the ≥90 line.
Discord fully quiet; queue green depth 3 (16 open); `run_work_next`
armed. Next boundary: arm roll ~11:2xZ.**

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
