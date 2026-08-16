# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-16 19:19–22:4xZ (real `date -u` at stamp: 21:55, close
addendum 22:4x) —
work session: **five owner steering threads executed live — expert
v1.3 (brackets real, retreat glide 2× slower, 1.5 cm centering) landed
measured; SFT run 1b killed on owner order and RESTARTED with
`--recompute-stats` (eval-250 14.53 → 4.05 deg, the clamping diagnosis
validated); wrist-cam pose refit scoped + queued; endpoint boundary
fully pre-staged.***

**Status**: `grasp_sft_v1_joint_8xa100_recompute` LIVE (run 2, unit
`grasp-sft-v1c`, launched 21:14:48Z after the owner's 20:51Z restart
order), step ~950/3000, **evals 4.05 @250 → 4.54 @500 → 3.74 @750 deg
vs run-1's 14.53 @250** (raw-degree metric — directly comparable; the
recomputed table opens wrist_roll to ±157° and lift to −124.8° with
the remap's descending orientation preserved; curve non-monotone at
500, recovered below start by 750 — trend watch at 1000), 100%×8,
~4 s/step, projection 10.1 vs the 40 GPU-h gate, ETA
**~00:3x–00:5xZ 08-17**. Step-500 checkpoint mirrored weights-only to
local `~/checkpoints/finetune/grasp_sft_v1_joint_recompute/step_000500`
(owner ask 21:51Z, byte-verified — they run local rollouts against
it). Local disk pruned on owner GO 22:31Z: old probe run 171→12 GB
(intermediates + optimizer dropped, step-2000 weights kept local +
Hub-banked), free space 79→239 GB; `outputs/` 486 GB audit queued.
Run 1b (remap-only table) killed at step ~1900 (~17.5 GPU-h, saves
archived `_run1_remaponly`; its evals had plateaued 13.62@1500 →
13.81@1750 with `pick_place_v2` rising — the coverage-gap signature).
Local GPU idle after sample renders.

**Steering** (5 threads, all replied + acked, inbox clear): (1) 19:41Z
camera brackets → `bracket_appearance='real'` landed (leader bracket
hidden, follower ring filled — render-only, physics oracle-tested);
(2) 19:42Z approach+retreat → retreat fold-glide landed; approach
easing measured **NO-GO as a drop-in** (placed 41.7/40.8 vs 58.3
n=120 — quasi-static arm never trips the momentum-tuned phase exits;
diagnosis banked, redesign queued owner-gated); (3) 20:24/20:26Z
centering + slower retreat → v1.3 landed (place bar 3→1.5 cm: center
distance 2.23→1.62 cm mean; retreat 5°/tick; tail 300→450 — kept 52.5
vs 54.2 with parked 98.6%, residual gap physical); (4) 20:51/20:53Z
**restart with `--recompute-stats`** → executed same-hour (main
`3a12c86` merged, 20-step smoke + per-joint receipt verified,
relaunch same seed, wandb renamed per ask); (5) 21:43Z wrist-cam
gripper mismatch → confirmed by eyeball (rig shows both jaws
symmetric; sim shows one leaning tip), matched-state instrument+fit
plan agreed, queued `wrist-cam-pose-refit`.

**Done** (commits `01ae7de`…`d15d1b9` + close, checks green every
commit): endpoint boundary pre-staged end-to-end (sharded box sim100
4×25/leg exact via triple-keyed noise + merge guards + weights-only
upload script + chart-led report generator with per-dataset MAE +
HTML report `--preset v1endpoint` — all synced to the box); babysit
ssh-transport guard (false LIVENESS FAILURE class fixed after live
false alarm); expert v1.3 + brackets + reusable sample renderer (4
sample videos posted); SFT restart executed + bookkept; queue ±2
(approach redesign, wrist-cam refit).

**Next**: `queue_cli.py next` → `grasp-sft-v1-endpoint-boundary` at
run-2 completion (~00:3x–00:5xZ 08-17: final eval + per-dataset table,
weights-only upload, sharded sim100 vs the 44/100 flow anchor + ≥20
token bar, report page + consolidated post). `run_work_next` ARMED at
close — the tick chain babysits overnight and fires the boundary.
Next work session: `wrist-cam-pose-refit`. Owner-pending unchanged:
disk composite exemption 👍, approach redesign go, v2.1 bands,
ckpt-format, morning-veto items.

*Updated 2026-08-16 19:12–19:2xZ (real `date -u` at stamp: 19:19) —
tick: **SFT healthy through eval-750 — MAE monotone 14.53 → 14.04 →
13.85, all 8 GPUs 78–100%, zero tracebacks; no steering; work
session stays chained for the endpoint boundary.***

**Status**: `grasp_sft_v1_joint_8xa100` LIVE (unit `grasp-sft-v1b`),
step 770+/3000 at 19:18Z, window 13.4 steps/min (≈3.9 s/step incl.
eval pauses), VRAM steady ~64.5 GiB/rank, cumulative projection 7.6
vs the 40 GPU-h babysit gate, 0 tracebacks. Evals monotone:
14.53@250 → 14.04@500 → **13.85@750** (train_mae 13.97). Tick held
open through the eval-750 boundary before closing (charter §6). ETA
unchanged ~21:4x–22:0xZ. Local GPU idle (owner-released).

**Steering**: none — read + inbox empty, no new reactions on the
last 5 posts (v1.1 sample videos unreacted so far; disk `realcal`
exemption still awaits the owner's 👍).

**Done**: routine babysit tick — two babysit polls (19:13, 19:18)
bracketing eval-750, remote-log eval read, queue validate (depth 1
with recorded reason, 20 open), body + footer roll to
archive/now-2026-08-16.md.

**Next**: `run_work_next` stays ARMED (armed 19:03 by the work
session close; box busy + `grasp-sft-v1-endpoint-boundary` queued
for run completion ~21:4x–22:0xZ). Owner-pending unchanged: disk
composite exemption 👍, v2.1 bands, ckpt-format, morning-veto
items.*

*Updated 2026-08-16 16:53–19:0xZ (real `date -u` at stamp: 19:02) —
work session: **grasp_sft_v1_joint LIVE on the 8×A100 box (one
crash-fix-relaunch cycle, 12-min turnaround, fix first-real-run
validated); smoother-demos v1.1 SOLVED + landed with kept% UP; disk
"does not render" root-caused to a composite-grade ceiling; sample
videos posted.***

**Status**: `grasp_sft_v1_joint_8xa100` LIVE since **18:21:14Z**
(unit `grasp-sft-v1b`; launch 1 17:49:48Z died at its FIRST eval —
the ported molmo_flow decoder returns CPU actions and the joint
family is the first through in-train validate(); fixed `2d6a2b3`
`sampled.to(device)`, ~2.5 GPU-h lost, no save existed). Relaunch:
eval-250 GREEN 14.53 → eval-500 **14.04**, step 530 loss 0.45,
3.9 s/step, VRAM 59.7/80, **first async save validated** (captured
24.9s, published 287s behind boundary), host RAM fine, 0 tracebacks.
Projection ~3.1 GPU-h so far vs the 40 babysit gate; **ETA
~21:4x–22:0xZ** (past this session's 20:53 kill — endpoint boundary
queued, babysit registry current, `run_work_next` armed). Local GPU
owner-released, idle after renders.

**Steering** (3 messages, all replied + acked, inbox clear): (1)
16:53Z "traces jumpy — smoother overall?" → executed same-session.
(2) 17:07Z four-parter: SFT GO eff-96 → launched; smoother demos →
landed; top-cam cylinder → root-caused in TWO stages (real/table lum
ratio 1.78 vs sim 0.95; then the calibration attempt found the v3
episode affine caps any foreground at ≤~1.1× plate — material-only
NO-GO, revised proposal = exempt the disk mask from the episode
affine, predicted ~1.5, flag `disk_appearance='realcal'` landed,
sign-off pending); frame leak → dataset exact (ffprobe counts match),
visualizer's inclusive endpoint. (3) 17:29Z "start ASAP" → launched.

**Done** (commits `859b249`, `0e77650`, `dbc0731`, `d40d43d`,
`2d6a2b3` + close; checks green): main merged; **smoother-expert
v1.1** — output-stage feedforward slew (10°/tick arm / 12 jaw,
`None`=legacy, 2 oracles) + tail budget 150→300; attribution harness
`smooth_expert_measure.py`, 5 configs × 120 seeds banked: **kept
54.2% vs 45.8 baseline / 48.3 anchor, parked 94.3% vs 53.5, max step
293°→10°/tick** (6°/tick NO-GO measured: main-clock starvation;
tail-150 was demoting 13/120 placed at BASELINE — a shipped-v1 yield
tax); blog post + 3 dark charts; merged-dataset boundary audit (5,000
eps exact); disk instruments (`disk_contrast_probe.py` + real
anchor); eval-device fix `2d6a2b3`; SFT launcher + babysit entry;
2 sample v1.1 videos posted in-channel; queue ±: 4 closed, 4 added.

**Next**: `queue_cli.py next` → `grasp-sft-v1-endpoint-boundary`
(fires at run completion ~21:4x–22:0xZ: final eval + per-dataset MAE
table + ckpt upload + sim100 rollout vs the 44/100 anchor + report
page). Depth 1 with reason: every other item is owner-gated (disk
composite exemption 👍, v2.1 bands, ckpt-format, morning-veto) or
gated on the run/box. `run_work_next` ARMED — box busy, boundary item
queued.*

## Utilization footer

Session 2026-08-16 19:19–22:0xZ (work, exploit; box: run-1b ride
19:19→kill ~21:07 ≈ +14 GPU-h in-window + smoke ~0.3 + run-2 live
21:14:48Z ≈ +6 to stamp, run-2 projected ~27/40 gate; local ~0.3
GPU-h renders/probes/harness ×4 n=120 runs): **five owner threads
same-session — expert v1.3 landed measured (brackets real, retreat
glide 5°/tick, centering 1.62 cm, tail 450; approach ease banked
NO-GO with mechanism), SFT restarted on order with
`--recompute-stats` (eval-250 14.53→4.05 deg), wrist-cam refit
scoped+queued, endpoint boundary fully pre-staged + re-pointed** —
queue depth 2, `run_work_next` armed, inbox clear.

Session 2026-08-16 19:12–19:2xZ (tick; box SFT riding ≈ +0.8 GPU-h
during the hold, local idle): **grasp_sft_v1_joint healthy through
eval-750 — MAE monotone 14.53→14.04→13.85, 13.4 steps/min, util
78–100%×8, projection 7.6/40 gate, 0 tracebacks** — no steering,
inbox clear, no new reactions, queue depth 1 with stated reason,
`run_work_next` stays armed for the endpoint boundary.

Session 2026-08-16 16:53–19:0xZ (work, exploit; box SFT: launch 1
+8×28 min ≈ 3.7 GPU-h incl. the eval-crash loss ~2.5, relaunch live
18:21Z ≈ +5.5 GPU-h to stamp, run projected ~26/80 gate; local ~0.2
GPU-h renders/probes): **grasp_sft_v1_joint launched, crashed at
first eval (molmo_flow CPU actions × in-train validate), fixed +
relaunched in 12 min, fix + first async save both first-real-run
validated, evals falling 14.53→14.04**; **smoother-demos v1.1 landed
same-session** (kept 54.2 vs 45.8, parked 94.3, max step 293°→10°,
n=120×5 instrumented + attribution); dataset boundaries proven exact;
top-cam disk: real-vs-sim 1.78-vs-0.95 then the composite-affine
ceiling (material NO-GO, exemption proposal pending 👍); 2 v1.1
sample videos in-channel; 3 owner messages answered live.

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
