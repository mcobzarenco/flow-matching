# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-12 06:26–07:3xZ (real `date -u` at stamp fix:
07:26; the first write carried an unchecked 07:31 stamp) — work
session: **sim-content-diversity CLOSED — registered bar MISSED
on the spread leg (top k std/mean 0.038 → 0.114 vs ≥ 0.15) while the
AUROC leg over-met: 0.773 → 0.673, k-ratio 1.02× — the top camera's
composites now sit INSIDE the real embedding spread, the best top
read this axis has produced. Default stays v2 per the registered
flip rule; the flip is a one-👍 owner ask on the results post.***

**Status**: no live jobs — registry empty, `nvidia-smi` 0% / 0 MiB
between probe reads (~0.08 GPU-h foreground total, gate 0.3). Queue
validate green (depth 2, 12 open): **sim100-v2-rerun-amendment-draft**
+ **sim-disk-position-prereg-draft** (new — fed by the measured real
disk wander, 8–29 cm × ±19 cm across A episodes);
**sim100-v1-rerun** stays owner_hold with its gate at double-GO
(v3 would make it top 0.673 + wrist 0.548 if the owner flips the
default).

**Steering**: none — Discord read empty at boot (06:27Z); owner
asleep since 01:11Z. Now pending with them: rerun spot-check
(05:01Z), double-GO rerun call, and the new v2→v3 default flip ask
(results post 07:24Z). **Owner woke 07:21Z** — good-morning reply
sent 07:25Z, conversational polling live.

**Done**: **sim-content-diversity CLOSED** (pre-reg in-channel
06:35Z, close commit this entry): 26-plate per-episode bank mined
ghost-free (inlier median vs gain-corrected global plate,
channel-MAX deviation after a channel-mean candidate let the
operator's hand smear a plate — caught by inspection); clutter
spread measured through the sim's own camera model
(displace-and-recover selfcheck 0.4/1.7 cm; mouse present 27% of A
episodes, mug-item 15%, laptop 77% as deltas, pcb static);
`render_style="v3"` ships (plate + clutter draws after every v2
draw; wrist path bit-identical to v2 — guard script GREEN, probe
wrist 0.548 reproduced). Reads: top k std/mean 0.114 (100 seeds;
0.114 at 20×5 — stable), AUROC 0.673/0.655, per-draw spread
0.5%→2.5%; between-plate variation carries most of the new spread.
Encoder-null iteration banked: composing per-episode gain onto the
foreground moved nothing — third confirmation that content moves
this encoder, light does not. Oracles 6 green, check.py green. 2
probe jsons + 2 charts on fontaine-reports; results post +
reports.md section. Queue: content-diversity done,
disk-position-prereg-draft queued (depth 2).

**Next**: `queue_cli.py next` → **sim100-v2-rerun-amendment-draft**
(or the owner's rerun/spot-check/flip call if it lands first).
`run_work_next` armed. No dated boundaries — `queue.json` canonical.*

*Updated 2026-08-12 06:23–06:3xZ (real `date -u` at write: 06:24) —
tick (babysit): **quiet tick — the wrist-periphery close (06:19Z
post) stands as latest; nothing live, no owner traffic, chained work
session armed.***

**Status**: no live jobs — registry empty (babysit exit 0),
`nvidia-smi` 0% / 0 MiB, GPU idle-by-design. Queue validate green
(depth 2, 12 open). Next items: **sim-content-diversity** +
**sim100-v2-rerun-amendment-draft**; **sim100-v1-rerun** stays
owner_hold with its gate at double-GO (top 0.773, wrist 0.548 — both
under their registered lines; the 05:01Z spot-check ask unanswered).

**Steering**: none — Discord read empty; history shows our 06:19Z
wrist-periphery close as latest, no new reactions (owner asleep
since 01:11Z).

**Done**: babysit poll exit 0; queue validate green; `run_work_next`
armed (06:23Z). Archive roll: 05:09 body entry + 05:49 footer note.

**Next**: chained work session → `queue_cli.py next` →
**sim-content-diversity** (or the owner's rerun/spot-check call if
it lands first — the gate is double-GO). No dated boundaries —
`queue.json` canonical.*

*Updated 2026-08-12 05:52–06:2xZ (real `date -u` at write: 06:17) —
work session: **sim-wrist-periphery-fix CLOSED — registered bar
SMASHED on the first candidate: wrist 5-NN AUROC 0.900 → 0.548 vs
≤ 0.786 (0.5 = can't tell sim from real; k-ratio 0.97× — sim wrist
frames now sit INSIDE the real embedding spread). One runtime pose
change: the camera moves ~10 cm forward, over the jaw base, 55°→65°
down.***

**Status**: no live jobs — registry empty, `nvidia-smi` 0% / 0 MiB
between probe reads (~0.04 GPU-h foreground total, gate 0.2). Queue
validate green (depth 2, 12 open): **sim-content-diversity** +
**sim100-v2-rerun-amendment-draft** (new, makes the rerun
launch-ready on unhold); **sim100-v1-rerun** stays owner_hold but
its gate now reads **double-GO** (top 0.773 ≤ 0.790 AND wrist
0.548 ≤ 0.786 — both cameras at/under their registered lines).

**Steering**: none — Discord read empty at boot (05:52Z) and at the
close poll (06:16Z, surfaced only our own pre-reg post); owner
asleep since 01:11Z. Rerun spot-check ask (05:01Z) still pending.

**Done**: **sim-wrist-periphery-fix CLOSED** (pre-reg posted
05:59Z, close commit this entry): `_repose_wrist_cam` re-derived —
camera from the wrist top behind the gripper (world ~(0.096,−0.004,
0.160), 55°) to over the jaw base (~(0.150,0,0.150), 65°), found in
3 encoder-free iteration rounds vs pinned A-half real starts; the
gripper-body mass filling the bottom ~40% of frame drops out,
leaving jaw tips in the bottom quarter like every real start frame.
Reads: wrist 0.548 (100 seeds; 0.550 at 20×5 — stable), centroid
0.587; guard green (top 0.773 bit-identical). Per-episode
wrist-plate axis retired. Oracles 10 green (qpos bit-identity
across styles, spawn stream vs banked v0), check.py 704 green. 2
probe jsons + REAL|old|new gallery on fontaine-reports (all curl
200). Results post + reports.md section; queue: wrist item done,
amendment-draft queued, rerun gate fact double-GO. Archive roll:
05:04 body entry + 05:09 footer note.

**Next**: `queue_cli.py next` → **sim-content-diversity** (or the
owner's rerun/spot-check call if it lands first — the gate is
double-GO). `run_work_next` armed. No dated boundaries —
`queue.json` canonical.*

## Utilization footer

Session 2026-08-12 06:26–07:3xZ (work, exploit; ~0.08 GPU-h
foreground probe/guard reads, gate 0.3): sim-content-diversity
closed — pre-reg 06:35Z, 26-plate per-episode bank + measured
clutter draws ship as render_style="v3"; **spread bar MISSED (top k
std/mean 0.038 → 0.114 vs ≥ 0.15) but AUROC 0.773 → 0.673, k-ratio
1.02× — top composites inside the real spread, best top read yet.**
Wrist guard bit-identical (0.548). Default stays v2 per the
registered flip rule → owner flip ask posted. Queue refilled:
sim-disk-position-prereg-draft (real disk wanders 8–29 cm × ±19 cm,
measured). run_work_next armed.

Session 2026-08-12 06:23–06:3xZ (tick, babysit; 0 new GPU-h — GPU
idle-by-design after the wrist-periphery close): quiet tick.
Registry empty, nvidia-smi 0%/0 MiB. Discord read empty, no new
reactions (rerun spot-check + double-GO call both pending with the
owner, asleep since 01:11Z). Queue validate green (depth 2, 12
open); run_work_next armed → sim-content-diversity (or the owner's
rerun call) chains next. Archive roll: 05:09 body entry + 05:49
footer note.

Session 2026-08-12 05:52–06:2xZ (work, exploit; ~0.04 GPU-h
foreground probe reads, gate 0.2): sim-wrist-periphery-fix closed —
pre-reg 05:59Z, one runtime wrist-cam pose change (over the jaw
base, 65° down), **registered bar SMASHED: wrist 5-NN AUROC
0.900 → 0.548 vs ≤ 0.786** (k-ratio 0.97× — sim wrist inside the
real spread; first camera to reach statistically-indistinguishable);
top guard green (0.773 bit-identical). Per-episode wrist-plate axis
retired; sim100 rerun gate now double-GO (still owner_hold on the
spot-check ask). Queue refilled: sim100-v2-rerun-amendment-draft
(depth 2). run_work_next armed.

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
