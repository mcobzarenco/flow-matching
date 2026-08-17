# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 first launch 15:01Z killed ~15:07Z by the driver
teardown (+~0.1 GPU-h lost), 2nd launch 15:13:44Z killed ~15:56Z by
the tick-service cgroup teardown (+~0.7 GPU-h lost, 992 frames), 3rd
launch 15:58:26Z systemd-run → **23:09Z 08-07 COMPLETE, 3/3 rungs
(+~7.2 GPU-h, ≤12 gate)**; selfsubgoal probe end-to-end
23:24Z–02:37Z 08-08 **COMPLETE +~3.2 GPU-h (≤ 8 gate)**;
 08-08 daytime: local rung-(b) preflight+stage1
08:49–10:15Z **+~1.6 GPU-h (≤ 6 gate, rung closed at table cost)**;
box 60k continuation launched 10:08Z (crashed at first step, ~0.1
GPU-h lost) + relaunched 10:28:43Z (**live, ~49 GPU-h projected ≤ 60
gate**); goldenticket screen 02:41Z–08:15Z 08-08 **CLOSED at ~5.55 GPU-h ≤ 6
gate** (s1 ~1.7 + s2 ~0.85 + s3 2.99); box molmo2 chain: 40k train
to ~04:0xZ, greedy ~1.7 GPU-h, draws10_t1 04:54–07:22Z **~10 GPU-h
≤ 24 gate**, microbench 07:27–07:50Z ~0.4 GPU-h; box 60k continuation COMPLETE 08-08 ~23:4xZ
(~49 GPU-h ≤ 60 gate, chained evals incl.); local subgoal-swap arms
08-09 ~02:1x–03:42Z +~1.5 GPU-h ≤ 3 gate; box K-smoke ladder 08-09
04:02–04:39Z **+~0.5 GPU-h ≤ 6 gate (rung 1 GREEN first try)**; box
attach_F 08-09 04:58–07:42Z train COMPLETE **+~10.2 GPU-h** + panel_v2
eval COMPLETE ~08:01Z (+~1.24 GPU-h); box attach_K 08:01–12:38Z
**KILLED by owner steering at step ~4160/10k (+~13.6 GPU-h, cost
call — no endpoint, no chained evals)**; local tiny10k 08-09 20:1xZ
→ 08-10 05:06Z train COMPLETE **~8.7/15 GPU-h incl. OOM replay** +
chained panel_v2 eval COMPLETE 08-10 05:45Z (+~0.6 GPU-h, **~9.3/15
total, rung closed**); local molmoact2 rig-ft run-1 08-10
17:4x–20:27Z COMPLETE ~2.7/12 GPU-h; local er35k owner-request evals
08-10 20:5x–00:41Z 08-11 ~2.2/8 GPU-h; local molmoact2 port parity
reads 08-10/11 ~0.7 GPU-h; local molmoact2_ae_ours (port item 4)
08-11 05:19–06:56Z **COMPLETE ~1.9/6 GPU-h (port total ~2.6/8)**).
Older dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).
