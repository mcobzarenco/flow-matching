# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-16 02:49–02:5xZ (real `date -u` at stamp: 02:52) —
tick: **quiet babysit — joint run green at step 520, no steering, no
gates.***

**Status**: `grasp_sft_joint_corrected` healthy at 02:50Z — step
520/2000, 10.70 s/step (5.5 steps/min since the 02:39 sample), VRAM
66.65 GiB flat, 71.5 GB resident, loss_action 0.579→0.525 over the
60-step window (steady decline, well clear of K1). ~4.4 h to step
2000, ETA ~07:3xZ unchanged. All anchors nominal (the 0% util
instant read is a sampling blip — step rate unchanged).

**Steering**: none — Discord read empty, inbox empty, history shows
no new reactions (launch-post 👍 already recorded 01:2xZ).

**Done**: babysit poll (facts above); queue validate OK depth 2 (17
open). Next queue leg is the GPU endpoint at step 2000 — no
CPU-side executable items, `run_work_next` not armed.

**Next**: unchanged — ticks babysit to step 2000 (K1 anchors in
registry), then `launch_local_grasp_sft_joint_probes.sh` legs IN
ORDER + `grasp_sft_joint_probe_reads.py`. Morning: owner veto
window on init/λ/insulation/text-lr.*

*Updated 2026-08-16 02:38–02:4xZ (real `date -u` at stamp: 02:39) —
tick: **quiet babysit — joint run green at step 460, no steering, no
gates.***

**Status**: `grasp_sft_joint_corrected` healthy at 02:39Z — step
460/2000, 10.92 s/step (5.5 steps/min since the 02:28 sample), VRAM
66.65 GiB flat, 71.5 GB resident, loss_action 0.639→0.579 over the
60-step window (steady decline, well clear of K1). ~4.7 h to step
2000, ETA ~07:3xZ unchanged. All anchors nominal.

**Steering**: none — Discord read empty, inbox empty, history shows
no new reactions (launch-post 👍 already recorded 01:2xZ).

**Done**: babysit poll (facts above); queue validate OK depth 2 (17
open). Next queue leg is the GPU endpoint at step 2000 — no
CPU-side executable items, `run_work_next` not armed.

**Next**: unchanged — ticks babysit to step 2000 (K1 anchors in
registry), then `launch_local_grasp_sft_joint_probes.sh` legs IN
ORDER + `grasp_sft_joint_probe_reads.py`. Morning: owner veto
window on init/λ/insulation/text-lr.*

*Updated 2026-08-16 02:28–02:3xZ (real `date -u` at stamp: 02:28) —
tick: **quiet babysit — joint run green at step 400, no steering, no
gates.***

**Status**: `grasp_sft_joint_corrected` healthy at 02:28Z — step
400/2000, 10.93 s/step (5.4 steps/min since the 02:17 sample), VRAM
66.65 GiB flat, GPU 95%/71.5 GB, loss_action 0.671→0.639 over the
60-step window (steady decline, well clear of K1). ~4.9 h to step
2000, ETA ~07:3xZ unchanged. All anchors nominal.

**Steering**: none — Discord read empty, inbox empty, history shows
no new reactions (launch-post 👍 already recorded 01:2xZ).

**Done**: babysit poll (facts above); queue validate OK depth 2 (17
open). Next queue leg is the GPU endpoint at step 2000 — no
CPU-side executable items, `run_work_next` not armed.

**Next**: unchanged — ticks babysit to step 2000 (K1 anchors in
registry), then `launch_local_grasp_sft_joint_probes.sh` legs IN
ORDER + `grasp_sft_joint_probe_reads.py`. Morning: owner veto
window on init/λ/insulation/text-lr.*

## Utilization footer

Session 2026-08-16 02:49–02:5xZ (tick; joint run riding): **quiet
babysit green** (step 520/2000, 10.70 s/step, VRAM 66.65 flat,
loss_action 0.579→0.525 declining) — no steering, inbox empty, no
new reactions, queue OK depth 2, no CPU-side items.

Session 2026-08-16 02:38–02:4xZ (tick; joint run riding): **quiet
babysit green** (step 460/2000, 10.92 s/step, VRAM 66.65 flat,
loss_action 0.639→0.579 declining) — no steering, inbox empty, no
new reactions, queue OK depth 2, no CPU-side items.

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
