# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-17 19:20–22:5xZ (real `date -u` at write: 22:38) —
work session: **utilization ledger REBASED + discriminator OOM
incident caught, root-caused, fixed and RELAUNCHED — attempt 1 died
at its first eval probe (probe batched at 96, training forwards
micro-12); fix VERIFIED at 250, Amendment 1 frozen pre-500,
step-500 baseline banked (7.567); verdict ~00:4xZ 08-18.***

**Status**: 1 live run — `grasp_sft_v2_demosonly_1gpu_disc`
ATTEMPT 2 (unit `fontaine-demosonly-1gpu-disc-r2`, launched
20:20:55Z after the OOM fix): restart-from-0, same seed 0, same
recipe; attempt-1 pace 15–18.7 s/step → step 1000 ≈ 01:0x–01:3xZ
08-18, next tick(s) own the boundary. Attempt 1 trained clean to
step 250 (loss 4.94→0.71, 62.26 GiB steady) then died 19:59:21Z:
CUDA OOM in the FIRST eval probe — `build_probe_set` batches at the
full per-rank batch (96) while chunked training only ever forwards
micro-12; the fast-path decode's KV caches pushed 62→79 GiB. Latent
in the frozen box script too. FIX landed: probe batches at
`batch_size // backward_chunks` (`bijou/train/cli.py`); probe/eval
tests green; verdict rule untouched (within-run delta, same probe
batching both ends). Attempt-1 jsonl preserved as
`train_log_attempt1_oom250.jsonl` (no eval record ever flushed).
~1.25 GPU-h burned; ~5.8 total projected vs the 12 gate. **Step-250
probe (21:29Z): the fix HELD** — eval 12.5087 / train 12.4202, no
OOM, unit active. The LEVEL is ~3.6× the comparator family (theirs
3.4623 at 250) while AR CE tracks (0.6385 vs 0.6116): first run on
the merged family-norm stack → probe units shifted. **Amendment 1
posted 21:3xZ, BEFORE the step-500 probe**: frozen scale estimator
s=3.613; verdict computes raw AND scale-adjusted bounds; disagree ⇒
AMBIGUOUS-BY-INSTRUMENT + stack-parity disambiguation. **Step-500
read (22:34Z): eval 7.567 / train 7.2209** — still descending
steeply (comparator was flat at 3.24 there); ratio moved 3.61×→2.34×
between probes, so the constant-scale assumption is strained and the
disagree-branch is live; descent-asymmetry caveat recorded
in-channel (1539039813804498984). Save-500 banked; verdict window
baseline = 7.567.

**Steering**: none — `read` empty, inbox empty, nothing new in
`history -n 5`. Incident + fix + relaunch posted in-channel
(1539006392671805572).

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
unblocked CPU head. Discriminator boundary ~00:4xZ 08-18 (attempt
2, step 500 passed 22:3xZ at 14.75 s/step): `sft_drift_saga_charts.py
--discriminator` on the FRESH jsonl only (attempt-1 file carries no
eval records), then apply **Amendment 1**: raw AND scale-adjusted
rules, disagree ⇒ AMBIGUOUS-BY-INSTRUMENT + stack-parity probe of
the saved 500/1000 checkpoints; carry the descent-asymmetry caveat
if Δ is negative. Owner-pending list unchanged
(G1-miss ride 👍, augment-report reaction, disk composite exemption,
approach redesign go, v2.1 bands, ckpt-format, morning-veto items).*

## Utilization footer

Session 2026-08-17 22:36–22:4xZ (tick; GPU-h accruing —
discriminator riding to the ~00:4xZ verdict): **quiet babysit —
step 510/1000 at ≈16.6 s/step effective, loss 0.52, VRAM 62.26 GiB
vs the 78 gate; the 91→50 GB host-RAM drop root-caused to the
save-boundary optimizer deep-copy (VmHWM≈RSS, resample flat) — no
leak, no action, boundary tick keeps the `free -g` glance** —
`run_work_next` armed; boundary tick owns the Amendment-1 verdict
read.

Session 2026-08-17 19:20–22:5xZ (work, exploit-infra; ~1.25 GPU-h
burned on discriminator attempt 1's OOM death + ~2.5 accrued on
attempt 2 in-session from 20:20:55Z, verdict ~00:4xZ 08-18; ridden
through the 250 fix-verify probe, Amendment 1, and the 500
baseline): **utilization ledger
rebased — trailing-7-day window recomputed per-run from prune
records + archive notes (local ~80.0/~80.2, box ~250/~254 FINAL at
the box kill), receipts note + rerunnable extract instrument landed
— AND the discriminator's first-eval-probe CUDA OOM root-caused
(probe batched at per-rank 96 vs training's micro-12) + fixed in
`bijou/train/cli.py` + relaunched same-seed from 0; queue refilled
`queue-box-kill-audit`** — attempt-1 jsonl preserved, incident
in-channel, next ticks own the boundary.

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
