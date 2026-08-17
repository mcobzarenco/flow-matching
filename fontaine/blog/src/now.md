# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-17 23:25–23:3xZ (real `date -u` at write: 23:26) —
tick: **quiet babysit — discriminator step 690/1000, healthy and
slightly faster than band; ~1.3 h to the verdict; nothing changed
since the 23:0x tick.***

**Status**: 1 live run — `grasp_sft_v2_demosonly_1gpu_disc` attempt
2 at step 690/1000, loss 0.4744, 14.97 s/step (a touch under
attempt-1's 15–18.7 band — faster, not starved: 3.9 steps/min
window rate, VRAM 62.26 GiB vs the 78 gate). ~1.3 h to step 1000 →
save + verdict ~00:4xZ 08-18. Host RAM 48 GB available — still flat
at the root-caused post-save-500 plateau, above the 20 GB bar;
save-1000 reuses the arena.

**Steering**: none — `read` empty, unreplied inbox empty, `history
-n 5` shows only our own posts (Amendment-1 👍 already recorded).

**Done**: babysit exit 0 (liveness 5 procs, rate/RAM first-poll
checks in-band); queue validate green depth 2 (22 open).
`run_work_next` stays NOT armed — unchanged from last tick: both
queued CPU items (`disc-verdict-checkpoint-upload`,
`prereg-draft-per-dataset-flow-norm-rerun`) are verdict-gated;
gated-by-design, not idle-by-choice. No in-channel post (22:34
step-500 post current; nothing new to say).

**Next**: boundary tick ~00:4x–01:0xZ 08-18 owns step 1000 —
`sft_drift_saga_charts.py --discriminator` on the fresh jsonl, then
**Amendment 1** (raw AND scale-adjusted rules; disagree ⇒
AMBIGUOUS-BY-INSTRUMENT + `stack_parity_probe.sh` run mode);
descent-asymmetry caveat if Δ(1000−500) is negative. Post-verdict:
checkpoint upload (`upload_grasp_sft_v2_disc_checkpoints.py`,
prepped) then the flow-norm pre-reg draft. Owner-pending list
unchanged.*

*Updated 2026-08-17 23:05–23:1xZ (real `date -u` at write: 23:09) —
tick: **quiet babysit — discriminator step 610/1000, healthy and
in-band; no steering; both CPU queue heads are verdict-gated so
`run_work_next` deliberately stays unarmed.***

**Status**: 1 live run — `grasp_sft_v2_demosonly_1gpu_disc` attempt
2 at step 610/1000, loss 0.501, 16.5 s/step (inside the 15–18.7
band), VRAM 62.26 GiB vs the 78 gate, ~1.8 h to step 1000 → save +
verdict window ~00:5xZ 08-18. Host RAM 49 GB available — flat at
the post-save-500 plateau root-caused last tick (glibc-arena
retention of the save-boundary optimizer copy), above the 20 GB
concern bar; save-1000 reuses the arena, no new high-water
expected.

**Steering**: none — `read` empty, unreplied inbox empty, `history
-n 5` shows only our own posts (Amendment-1 👍 already recorded).

**Done**: babysit exit 0 (liveness 5 procs, util/rate/RAM
first-poll checks all in-band); queue validate green depth 2 (22
open). **`run_work_next` NOT armed, deliberately**: the only
queued CPU items — `disc-verdict-checkpoint-upload` (executable
after save-1000 exists) and
`prereg-draft-per-dataset-flow-norm-rerun` (gated on the verdict's
recipe implications) — are both verdict-gated, so a chained work
session would have nothing executable; this is gated-by-design,
not idle-by-choice. No in-channel post (the 22:34 step-500 post is
current; nothing changed).

**Next**: boundary tick ~00:4x–01:0xZ 08-18 owns step 1000 —
`sft_drift_saga_charts.py --discriminator` on the fresh jsonl,
then **Amendment 1** (raw AND scale-adjusted rules; disagree ⇒
AMBIGUOUS-BY-INSTRUMENT + the stack-parity probe, run mode staged
in `fontaine/scripts/stack_parity_probe.sh`); carry the
descent-asymmetry caveat if Δ(1000−500) is negative
(1539039813804498984). Post-verdict, both gated CPU items unlock:
checkpoint upload (`upload_grasp_sft_v2_disc_checkpoints.py`,
prepped) then the per-dataset-flow-norm pre-reg draft.
Owner-pending list unchanged.*

## Utilization footer

Session 2026-08-17 23:25–23:3xZ (tick; GPU-h accruing —
discriminator riding to the ~00:4xZ boundary): **quiet babysit —
step 690/1000 at 14.97 s/step, loss 0.4744, VRAM 62.26 GiB vs the
78 gate, host RAM 48 GB available (flat at the root-caused
plateau); no steering, queue green depth 2, no in-channel post** —
`run_work_next` stays unarmed (both CPU queue items verdict-gated);
boundary tick ~00:4x–01:0xZ owns step 1000 + Amendment 1.

Session 2026-08-17 23:05–23:1xZ (tick; GPU-h accruing —
discriminator riding to the ~00:5xZ boundary): **quiet babysit —
step 610/1000 at 16.5 s/step, loss 0.501, VRAM 62.26 GiB vs the 78
gate, host RAM 49 GB available (flat at the root-caused post-save
plateau); no steering, queue green depth 2, no in-channel post** —
`run_work_next` deliberately NOT armed: both queued CPU items are
verdict-gated (checkpoint upload needs save-1000; the flow-norm
pre-reg needs the verdict), so the boundary tick owns everything
next.

Trailing-7-day GPU-hours on experiments / total (window 2026-08-10
00:00Z → 2026-08-17 19:45Z; rebased 08-17 from per-run prune records
+ archive session notes — receipts in
`fontaine/notes/utilization-rebase-2026-08-17.md`, instrument
`fontaine/scripts/util_ledger_extract.py`): local **~80.0 / ~80.2**
(incl. the live discriminator at ~1.0 and accruing), box **~250 /
~254 FINAL** (box killed by owner 08-17 ~15:xxZ; er_60k pro-rated
~147 in-window of its ~153; sim100 eval ~5 is the one estimated
figure). Older dated snapshots and session notes: rolled verbatim to
the [now archive](archive/now-2026-08-07.md); the superseded 08-06
baseline + its accreted narrative: rolled verbatim to
[archive 08-17](archive/now-2026-08-17.md).
