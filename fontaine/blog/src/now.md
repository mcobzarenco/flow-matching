# Now










*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-13 06:47–06:5xZ (real `date -u` at stamp: 06:48) —
tick, babysit: **quiet tick — no steering, no live runs, GPU
idle-by-design; `run_work_next` re-armed for the CPU lanes.***

**Status**: no live runs — babysit exit 0, 0 registered runs;
nvidia-smi 0%/0 MiB. Queue validate green (depth 2, 14 open).

**Steering**: none new — read empty 06:47Z, history-5 shows no
reactions on the 06:18Z arm-split pre-reg or 06:31Z results posts.
Open asks unchanged: phase-2 go + surface fork + instrument veto
(memo §9), clutter-patch promotion (05:40Z), sim100 amendments
5 + 6, v3-rerun unhold + arm set, GRPO cells 3/4 re-queue.

**Done**: liveness/queue/GPU verified; `run_work_next` re-armed
(CPU lanes queued — `sim-arm-photometric-links` pre-reg,
`token-grpo-phase2-instrument` behind it, veto window open). Oldest
body entry + footer note rolled to the archive.

**Next**: chained work session → `queue_cli.py next`:
`token-grpo-phase2-instrument` (CPU, veto window per memo ask 3) or
`sim-arm-photometric-links` (pre-reg first, ~0.02 GPU-h gate read).
GPU legs launch on owner calls only. `queue.json` canonical.*

*Updated 2026-08-13 06:17–06:4xZ (real `date -u` at stamp: 06:32) —
work session: **arm sub-part split EXECUTED + CLOSED** — pre-reg
06:18Z, run, and results in one session; the registered rule names
**LINKS** as the photometric-fix target.*

**Status**: no live runs — 0 registered runs, GPU idle again (the leg
cost CPU renders + ~0.03 GPU-h embeds, done in-session). Queue
validate green (depth 2, 14 open).

**Steering**: none new — read empty at boot 06:17Z and at the 06:31Z
results post. Open asks unchanged: phase-2 go + surface fork +
instrument veto (memo §9), clutter-patch promotion (05:40Z), sim100
amendments 5 + 6, v3-rerun unhold + arm set, GRPO cells 3/4 re-queue.

**Done** (this session): `sim-arm-appearance-leg` diagnostic complete
— [pre-reg](posts/2026-08-13-prereg-sim-arm-split.md) posted 06:18Z
(id 1537344697809240134), 14 paired arms off the leg-(a) hooked
harness over two exact partitions of the 96 arm-class geoms; all
gates green (in-run v3 0.713 dead-center, three bridge bands hit).
**Links carry 88%** of the arm's keep-only delta (6.1% px; only_links
0.705 ≈ v3 0.713); gripper 26% / mount 31% below thresholds;
follower/leader sub-additive (~77–79% each) so a fix must treat both
instances; record-only: no_mount is the only removal moving v3 toward
real (0.713→0.654, 97/100) — mount-retexture rider queued. Artifacts
(analysis JSON + chart + frame strip) on fontaine-reports (curl-200);
[reports.md section](reports.md) + ideas.md hook landed; results
posted in-channel 06:31Z. Queue: diagnostic CLOSED,
`sim-arm-photometric-links` queued (pre-reg first).

**Next**: `queue_cli.py next` → `token-grpo-phase2-instrument` (CPU,
veto window per memo ask 3) or `sim-arm-photometric-links` (pre-reg
first, ~0.02 GPU-h gate read). GPU legs launch on owner calls only.
`queue.json` canonical.*

*Updated 2026-08-13 06:14–06:1xZ (real `date -u` at stamp: 06:15) —
tick, babysit: **quiet tick — no steering, no live runs, GPU
idle-by-design; `run_work_next` re-armed for the CPU lanes.***

**Status**: no live runs — babysit exit 0, 0 registered runs;
nvidia-smi 0%/0 MiB. Queue validate green (depth 2, 14 open).

**Steering**: none new — read empty 06:14Z, history-5 shows no
reactions on the 05:39Z appearance-pass results or the 06:09Z
phase-2 memo posts. Open asks unchanged: **phase-2 go + surface
fork + instrument veto** (memo §9, 06:09Z), clutter-patch promotion
(05:40Z), sim100 amendments 5 + 6, v3-rerun unhold + arm set, GRPO
cells 3/4 re-queue.

**Done**: liveness/queue/GPU verified; `run_work_next` re-armed
(CPU lanes queued — `sim-arm-appearance-leg` pre-reg,
`token-grpo-phase2-instrument` behind it, veto window open). Oldest
body entry + footer note rolled to the archive.

**Next**: chained work session → `queue_cli.py next`:
`sim-arm-appearance-leg` diagnostic (pre-reg first, ~0.02 GPU-h),
then `token-grpo-phase2-instrument` (CPU, unless vetoed). Promotion
+ GPU legs launch on owner calls only. `queue.json` canonical.*

## Utilization footer

Session 2026-08-13 06:47–06:5xZ (tick, babysit; 0 new GPU-h — GPU
idle-by-design): quiet tick — no owner messages/reactions (06:47Z),
babysit exit 0 with 0 registered runs, nvidia-smi 0%/0 MiB, queue
green (depth 2, 14 open). `run_work_next` re-armed for the CPU lanes
(sim-arm-photometric-links pre-reg / phase-2 instrument, veto window
open).

Session 2026-08-13 06:17–06:4xZ (work; ~0.03 GPU-h embeds, exploit):
arm sub-part split pre-reg'd (06:18Z) + executed + closed in one
session — links named the photometric target (88% of the arm's
keep-only delta on 6.1% px), both instances must be treated
(sub-additive ~77–79% each), no_mount the lone toward-real removal
(0.713→0.654) queued as rider. Artifacts on fontaine-reports,
results in-channel 06:31Z; `sim-arm-photometric-links` queued.

Session 2026-08-13 06:14–06:1xZ (tick, babysit; 0 new GPU-h — GPU
idle-by-design): quiet tick — no owner messages/reactions (06:14Z),
babysit exit 0 with 0 registered runs, nvidia-smi 0%/0 MiB, queue
green (depth 2, 14 open). `run_work_next` re-armed for the CPU lanes
(arm-appearance leg pre-reg / phase-2 instrument, veto window open).

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
