# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-17 19:20–19:4xZ (real `date -u` at write: 19:38) —
work session: **utilization ledger REBASED — the trailing-7-day
baseline was 11 days stale; the fresh window reads local ~80 / box
~254 GPU-h.***

**Status**: 1 live run — `grasp_sft_v2_demosonly_1gpu_disc` at step
150/1000, loss 4.94→0.88, 14.9 s/step (~3.5 h to 1000 → verdict
~23:0xZ, inside this session's window), VRAM 62.24 GiB vs the 78
gate, babysit exit 0. Step-250 probe ≈19:55Z (drifting comparators
sat at 3.46 there; NO probe-kill bars — verdict at 1000 only).

**Steering**: none — `read` empty, inbox empty, nothing new in
`history -n 5`.

**Done**: queue item `utilization-ledger-rebase` CLOSED: trailing-
7-day GPU-h recomputed per-run over 08-10 00:00Z → 08-17 19:45Z —
**local ~80.0 / ~80.2** (vs the stale ~24.1/~24.4 baseline; incl.
the live discriminator at ~1.0), **box ~250 / ~254 FINAL** at the
08-17 box kill (er_60k pro-rated ~147 in-window of ~153; the box
sim100 eval ~5 is the one estimated figure). Babysit prune records
were authoritative for detached runs (tick notes log "0 new" while
units accrue — the narrative's known undercount class); receipts in
`fontaine/notes/utilization-rebase-2026-08-17.md`, instrument
`fontaine/scripts/util_ledger_extract.py` (rerunnable next rebase).
Footer baseline rewritten to the fresh stamp + standard 2-note form;
the superseded 08-06 baseline + its accreted narrative rolled
verbatim to the 08-17 archive page. Refill: `queue-box-kill-audit`
(the box kill invalidated every "box" host reference in the blocked
tail — each needs an explicit obsolete/re-platform/stays-blocked
call).

**Next**: `queue_cli.py next` = `prereg-draft-per-dataset-flow-norm-
rerun` (gated on the verdict); `queue-box-kill-audit` is the
unblocked CPU head. Discriminator boundary ~23:0xZ:
`sft_drift_saga_charts.py --discriminator` verdict → drift-saga
finalize + in-channel + un-gates the flow-norm pre-reg draft. This
session rides babysit checkpoints to the boundary; if step 1000
lands comfortably before the 23:20Z hard kill, the verdict reads
here, else the next tick owns it. Owner-pending list unchanged
(G1-miss ride 👍, augment-report reaction, disk composite exemption,
approach redesign go, v2.1 bands, ckpt-format, morning-veto items).*

*Updated 2026-08-17 19:17–19:2xZ (real `date -u` at write: 19:20) —
tick: **quiet babysit — discriminator healthy at step 100/1000, on
pace for the ~23:0xZ verdict.***

**Status**: 1 live run — `grasp_sft_v2_demosonly_1gpu_disc` at step
100/1000, loss 4.94→1.08, 15.8 s/step steady (~3.9 h to 1000), VRAM
62.24 GiB vs the 78 gate, GPU 65%/66.5 GiB mid-cycle, host RAM 91 GB
available (stable vs 92 at launch — no loader-buffer creep). babysit
exit 0. First eval probe at 250 ≈ 19:55Z — lands after this tick's
cap; the next tick reads it (drifting comparators sat at 3.46 there;
NO probe-kill bars — verdict at 1000 only).

**Steering**: none — `read` empty, inbox empty, no new reactions in
`history -n 5`.

**Done**: babysit + queue validate (OK, depth 2, 23 open) + the
standing RAM/util watch checks; `run_work_next` confirmed armed
(GPU-busy window, `utilization-ledger-rebase` is the CPU head). No
in-channel post — the 19:13 post covers current state, step-100
status adds nothing.

**Next**: chained work session takes `utilization-ledger-rebase`;
next tick reads the step-250 probe. At step 1000 (~23:0xZ):
`sft_drift_saga_charts.py --discriminator` verdict → drift-saga
finalize + in-channel + un-gates
`prereg-draft-per-dataset-flow-norm-rerun`. Owner-pending list
unchanged (G1-miss ride 👍, augment-report reaction, disk composite
exemption, approach redesign go, v2.1 bands, ckpt-format,
morning-veto items).*

*Updated 2026-08-17 19:02–19:2xZ (real `date -u` at write: 19:13) —
work session: **v1 mirror restored + a babysit-registry fix; the
discriminator is riding FAST — step-1000 verdict lands ~23:0xZ
TONIGHT, not the 7–9 h estimate.***

**Status**: 1 live run — `grasp_sft_v2_demosonly_1gpu_disc` at step
40/1000, loss 4.94→2.17, **15.1 s/step steady** (vs 25–32 box
estimate → ~4 h wall), VRAM 62.24 GiB vs the 78 gate, util cycling
100% (0% dips = offloaded-optimizer CPU phase, expected). First eval
probe at 250 ≈ 19:55Z (drifting comparators: 3.46 there); saves
500/1000; verdict read AT 1000 only.

**Steering**: none — `read` empty, inbox empty.

**Done**: (1) babysit exit-1 at boot diagnosed in minutes: the
registry's jsonl path was the BOX layout (`outputs/train/<run>/`);
the local bijou.train stack writes `~/checkpoints/finetune/<run>/` —
path fixed, babysit green (`303830d`), run never blipped. (2) Queue
item `local-dataset-mirrors-restore` DONE: audit first — NONE of the
three held gpu-local arms needs the v1 corpus (bootstrap + token-SFT
→ `grasp_sft_demos_v0`, on disk; grpo-r2 → checkpoint), mapping
recorded in their boundaries; then `fontaine-grasp-demos-v1` pulled
→ `~/datasets/fontaine/grasp_demos_v1/merged` in 1m42s, verified
EXACT vs the HF manifest (232 files, 28,099,973,012 bytes = 26.17
GiB, data/meta/videos present; disk 458 GB free). Pull = durability
redundancy — HF was the ONLY v1 copy post-box-kill. Refill:
`utilization-ledger-rebase` (footer baseline 11 days stale).
In-channel 1538989075539693651.

**Next**: `queue_cli.py next` = `utilization-ledger-rebase` (CPU,
unblocked); `run_work_next` armed. Discriminator boundary ~23:0xZ:
`sft_drift_saga_charts.py --discriminator` verdict → drift-saga
finalize + in-channel + un-gates
`prereg-draft-per-dataset-flow-norm-rerun`. Owner-pending: G1-miss
ride 👍, augment-report reaction, disk composite exemption, approach
redesign go, v2.1 bands, ckpt-format, morning-veto items.*

## Utilization footer

Session 2026-08-17 19:20–19:4xZ (work, exploit-infra; GPU-h accruing
— discriminator riding at 14.9 s/step, verdict ~23:0xZ): **utilization
ledger rebased — trailing-7-day window recomputed per-run from prune
records + archive notes (local ~80.0/~80.2, box ~250/~254 FINAL at
the box kill), footer baseline rewritten to the fresh stamp +
2-note form, receipts note + rerunnable extract instrument landed,
queue refilled `queue-box-kill-audit`** — babysit green at step 150;
session rides to the ~23:0xZ boundary.

Session 2026-08-17 19:17–19:2xZ (tick; GPU-h accruing — discriminator
riding): **quiet babysit — step 100/1000 at 15.8 s/step, loss
4.94→1.08, VRAM 62.2 GiB vs the 78 gate, host RAM stable at 91 GB
available, queue validated depth 2, no steering, no in-channel post
needed** — `run_work_next` armed; the step-250 probe (≈19:55Z) reads
at the next tick, verdict at 1000 ≈23:0xZ.

Session 2026-08-17 19:02–19:2xZ (work, exploit; GPU-h accruing —
discriminator riding at 15.1 s/step, ~4 h to verdict ~23:0xZ):
**babysit-registry jsonl path fixed (`303830d`, box layout → local
`~/checkpoints/finetune/`), v1 corpus mirror restored + verified
exact vs HF (232 files / 26.17 GiB; audit: no held arm needs it —
durability redundancy), queue refilled with
`utilization-ledger-rebase`** — `run_work_next` armed; next
executable CPU item is the utilization rebase.

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
