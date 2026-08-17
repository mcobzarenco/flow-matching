# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-17 22:36–22:4xZ (real `date -u` at write: 22:41) —
tick: **quiet babysit — discriminator step 510/1000, healthy; a
host-RAM drop (91→50 GB available) investigated and cleared as a
step-change at the save-500 boundary, not a leak.***

**Status**: 1 live run — `grasp_sft_v2_demosonly_1gpu_disc` attempt
2 at step 510/1000, loss 0.5791→0.5196, window rate 3.6 steps/min
(≈16.6 s/step, inside attempt-1's 15–18.7 band; the jsonl's 22.9
s/step at 510 is inflated by the probe+save at 500), VRAM 62.26 GiB
vs the 78 gate, GPU 100%/66.6 GiB. Probe trajectory 12.51@250 →
7.57@500 as banked. **Host-RAM watch finding**: available fell
91→50 GB since the 20:30 poll — root-caused, NOT loader creep:
the save path deep-copies the CPU-offloaded optimizer state +
tensors at each boundary (`copy_to_cpu` capture + async write,
`bijou/train/cli.py:2602`), a transient double retained by glibc
arenas. Evidence: VmHWM 147.3 GB vs RSS 145.8 GB (peak ≈ current —
save-1000 reuses the arena, no new high-water), and a 66-s resample
showed RSS flat (+116 MB noise) with MemAvailable *rising*
(51.8→52.3 GB). 50 GB headroom for the remaining ~2.5 h — no
action; boundary tick should still glance at `free -g` at first
poll (concern bar: <20 GB available).

**Steering**: none — `read` empty, inbox empty, `history -n 5`
shows only our own posts (the 👍 on the Amendment-1 post was
already recorded).

**Done**: babysit exit 0 + the standing util/rate/RAM first-poll
checks (util 100%, rate in-band, RAM investigated above); queue
validate OK depth 2 (23 open); `run_work_next` confirmed armed
(GPU-busy window, `queue-box-kill-audit` is the CPU head). No
in-channel post — the 22:34 step-500 post is current; the RAM
finding is a non-event once root-caused.

**Next**: boundary tick ~00:4x–01:0xZ 08-18 owns step 1000:
`sft_drift_saga_charts.py --discriminator` on the fresh jsonl only,
then **Amendment 1** (raw AND scale-adjusted rules, disagree ⇒
AMBIGUOUS-BY-INSTRUMENT + stack-parity probe of saves 500/1000);
carry the descent-asymmetry caveat if Δ is negative
(1539039813804498984). Then `queue-box-kill-audit` (CPU head);
`prereg-draft-per-dataset-flow-norm-rerun` stays verdict-gated.
Owner-pending list unchanged (G1-miss ride 👍, augment-report
reaction, disk composite exemption, approach redesign go, v2.1
bands, ckpt-format, morning-veto items).*

## Utilization footer

Session 2026-08-17 23:05–23:1xZ (tick; GPU-h accruing —
discriminator riding to the ~00:5xZ boundary): **quiet babysit —
step 610/1000 at 16.5 s/step, loss 0.501, VRAM 62.26 GiB vs the 78
gate, host RAM 49 GB available (flat at the root-caused post-save
plateau); no steering, queue green depth 2, no in-channel post** —
`run_work_next` deliberately NOT armed: both queued CPU items are
verdict-gated (checkpoint upload needs save-1000; the flow-norm
pre-reg needs the verdict), so the boundary tick owns everything
next.

Session 2026-08-17 22:36–22:4xZ (tick; GPU-h accruing —
discriminator riding to the ~00:4xZ verdict): **quiet babysit —
step 510/1000 at ≈16.6 s/step effective, loss 0.52, VRAM 62.26 GiB
vs the 78 gate; the 91→50 GB host-RAM drop root-caused to the
save-boundary optimizer deep-copy (VmHWM≈RSS, resample flat) — no
leak, no action, boundary tick keeps the `free -g` glance** —
`run_work_next` armed; boundary tick owns the Amendment-1 verdict
read.

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
