# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-17 05:51–05:5xZ (real `date -u` at write: 05:52) —
tick: **quiet tick — no live runs (local + box idle by design), inbox
clear, no new messages or reactions on the 04:01/05:46Z refit
pre-reg/results posts; `run_work_next` confirmed armed for the regen
pre-reg.***

**Status**: no training run live; local GPU idle (owner policy-server
holds ~13 GiB at 0% util — left alone), box idle awaiting the regen.
`grasp-demos-v2-regen` is the queue head and UNBLOCKED (wrist refit
shipped `4b14b1f`); pre-reg REQUIRED before launch — that is the
chained work session's first item.

**Steering**: none new (inbox empty, `read` empty; history check — no
reactions yet on the refit results post or the G1-miss ship-and-ride
question).

**Done**: routine tick — Discord read + history, queue validate (OK
depth 3, 24 open, updated 05:47Z), GPU/unit/state check (no fontaine
units live, `run_work_next` already armed), 03:43Z + 02:42Z entries
and footer notes rolled to the
[08-17 archive](archive/now-2026-08-17.md).

**Next**: chained work session — `grasp-demos-v2-regen` pre-reg
(expert v1.3 receipt, bracket_appearance=real, wrist_pose='refit',
kept-rate anchor 45.9%) then launch on the box; boundary results page
+ HTML with the corrected sim100 verdict;
`sft-v1-flow-regression-isolation` before the SFT-v2 recipe locks.
Owner-pending: G1-miss ship-and-ride 👍/veto, disk composite
exemption, approach redesign go, v2.1 bands, ckpt-format,
morning-veto items.*

*Updated 2026-08-17 03:47–05:5xZ (real `date -u` at write: 05:47) —
work session: **wrist-cam-pose-refit stages 2+3 DONE (the regen's
critical-path item) — measured on 312 matched pairs, fitted, held-out
validated, shipped flag-gated as `SO101Sim(wrist_pose='refit')`
(`4b14b1f`); `grasp-demos-v2-regen` is now UNBLOCKED.***

**Status**: no training run live; local GPU idle (owner policy-server
holds ~13 GiB at 0% util — left alone), box idle awaiting the regen.
All fit/measure work this session was render-only on the shared H100
(~0 GPU-h, segmentation passes).

**Steering**: none new (inbox empty at boot and at every poll; no new
reactions on the 03:39Z audit posts).

**Done**: (a) stage-2 instrument
(`fontaine/scripts/wrist_cam_pose_measure.py`): real both-jaws-visible
**92.9%** vs sim **0.0%** — the fixed jaw was NEVER in the sim wrist
frame at the v1 pose; detectors QC'd (salmon seed + bounded
blown-highlight growth; dark∪blue-gray fixed jaw, proximity-gated —
mount prints are the same color family); (b) pre-reg posted BEFORE the
fit (msg 1538759641591324747: params, split, G1–G3 gates); (c) stage-3
fit (`wrist_cam_pose_fit.py`): pitch −23° / yaw +14° / roll −9.5°,
camera-frame offset (+3.3, +1.3, −3.0) cm; held-out (96 pairs, 8
unseen eps): **G2 PASS** (both-jaws 0%→100% vs real 90.3%), **G3
PASS** (bottom-occ |Δ| −65%), **G1 MISS** (centroid −44.5% vs the −50%
bar; residual = lens-model/detector floor, axis err 42.5°→15.9°);
deviations disclosed (pattern search not NM; miss penalty repriced
0.08→0.5 after the first run found the degenerate point-away optimum);
(d) shipped flag-gated, default v1 untouched, physics bit-identical,
oracles added, check.py green, commit `4b14b1f`; (e) composite + fit
record on fontaine-reports (curl 200/302→200), results post
1538786116956594250 with a ship-and-ride recommendation on the G1
miss; (f) queue: item DONE with the full boundary record.

**Next**: `queue_cli.py next` → `grasp-demos-v2-regen` (NOW UNBLOCKED:
expert v1.3 + bracket_appearance=real + wrist_pose='refit'; pre-reg
REQUIRED before launch — params, expert receipt, kept-rate anchor
45.9%), then boundary results page + HTML with the corrected sim100
verdict, `sft-v1-flow-regression-isolation` before the SFT-v2 recipe
locks. Owner-pending: G1-miss ship-and-ride 👍/veto, disk composite
exemption, approach redesign go, v2.1 bands, ckpt-format,
morning-veto items.*

## Utilization footer

Session 2026-08-17 05:51–05:5xZ (tick; GPUs idle by design, box +
local — no live runs; local 13 GiB = owner policy-server, not ours):
**quiet tick — inbox clear, no new messages/reactions on the refit
pre-reg/results posts; queue depth 3 with `grasp-demos-v2-regen` at
the head (unblocked, pre-reg required), `run_work_next` confirmed
armed for the regen pre-reg; 03:43Z + 02:42Z entries/notes rolled to
the archive.**

Session 2026-08-17 03:47–05:5xZ (work, exploit; ~0 GPU-h — render-only
segmentation passes on the shared local H100, box idle): **wrist-cam
pose refit CLOSED same-session — 312-pair instrument (fixed jaw never
in the v1 sim frame, 0/312 vs real 92.9%), pre-reg'd 6-param fit,
held-out G2+G3 PASS / G1 −44.5% vs −50% bar (disclosed), shipped
flag-gated `wrist_pose='refit'` (`4b14b1f`), regen unblocked** — queue
depth 3, inbox clear, `run_work_next` armed for the regen pre-reg.

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
