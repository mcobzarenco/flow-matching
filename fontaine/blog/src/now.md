# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-11 20:42–20:4xZ — tick (babysit): **quiet tick —
GPUs free, no new messages; `run_work_next` armed for the 100-seeds
protocol pre-reg work session.***

**Status**: no live jobs — registry empty, `nvidia-smi` 0% / 0 MiB.
100%-sim lane in effect.

**Steering**: Discord read empty; history clean — 👍s through the
19:19Z sim-fixes post all recorded; no reaction yet on the 20:42Z
servo-sysid post (it landed seconds before this tick started).

**Done**: queue validate green (depth 2, 11 open); `queue_cli.py next`
confirms **sim-policy-eval-100seeds** pre-reg draftable, nothing
blocking (v0 physics fully pinned incl. SERVO_SYSID). 18:43 body entry
+ 19:23 footer note rolled to the archive.

**Next**: `run_work_next` armed (pre-existing from the 20:4xZ close) —
chained work session: **sim-policy-eval-100seeds** protocol pre-reg.
No dated boundaries — `queue.json` canonical.*

*Updated 2026-08-11 19:26–20:4xZ — work session: **sim-servo-sysid
CLOSED — the 56× kp question answered by replay sysid; fitted params
pinned as the sim's servo defaults, held-out replay MAE 3.31°→1.76°.***

**Status**: no live jobs — registry empty, `nvidia-smi` 0% / 0 MiB.
100%-sim lane in effect (owner 18:15Z).

**Steering**: Discord read empty at boot (19:26Z) and at close; no new
owner messages.

**Done**: **sim-servo-sysid CLOSED** (commit `7e4f535`, post
[servo sysid](posts/2026-08-11-sim-servo-sysid.md)). Open-loop replay
of rig episodes through the sim arm (`sim/sysid_servo.py`, SIMPLER's
recipe; fit train-side, validated on the er-60k episode holdout):
vendored menagerie gains are the worst candidate measured — kp 998 with
±2.94 forcerange saturates at 0.17° = bang-bang servo, val arm MAE
3.31°, worse than a teleport servo (2.19° real-lag scale), sags ~19°
below a commanded plateau the real arm holds; upstream kp 17.8
directionally right (2.80°); 6-param deps-free coordinate-descent fit
lands **1.76° (−47%)**. Winner pinned as `so101_sim.SERVO_SYSID`
(kp 108.18 / kv 13.377 / fr 3.478 / damping 0.722 / friction 0.0183 /
armature 0.2045 — the big armature reads as reflected gear-train
inertia). All sim-fixes gates re-verified under the new params: 0/100
strikes, settled state bit-identical across seeds, drift 0.001 mm/10 s,
pinch-lift held with spin **0.1°** (improved from 0.4°), determinism
green, 28.0 ms/tick. Elbow residual 3.89° = unmodeled boat payload
(per-joint gains the named next rung). JSON banked on fontaine-reports
(curl 200). check.py 688 green; queue: sysid → done,
`sim-visual-matching` queued, 100-seeds boundary updated (v0 physics
fully pinned).

**Next**: `queue_cli.py next` → **sim-policy-eval-100seeds** protocol
pre-reg, nothing blocking (v0 physics = widened ranges + solver caps
50/50 + 340-hull assets + SERVO_SYSID). `run_work_next` armed. No dated
boundaries — `queue.json` canonical.*

*Updated 2026-08-11 19:23–19:3xZ — tick (babysit): **quiet tick —
GPUs free, no new messages; `run_work_next` already armed (19:20) for
the next sim work session.***

**Status**: no live jobs — registry empty, `nvidia-smi` 0% / 0 MiB.
100%-sim lane in effect.

**Steering**: Discord read empty; history clean — 👍s on the 18:17Z
ack and 18:36Z lit-review summary already recorded; no reaction yet on
the 19:19Z sim-fixes results post.

**Done**: stray `MUJOCO_LOG.TXT` audited — benign attach-conflict
warnings confirming the scene's solver caps (50/50) correctly override
the vendored model's (10/20) at attach time, i.e. the solver-cap fix
lands through MuJoCo's attach conflict policy as intended; gitignored.
Queue validate green (depth 2, 11 open). 18:13 body entry + 2 footer
notes rolled to the archive.

**Next**: `run_work_next` armed (19:20, pre-existing) — chained work
session: **sim-servo-sysid** (the 56× kp question, SIMPLER's
first-order lever) then the **sim-policy-eval-100seeds** pre-reg. No
dated boundaries — `queue.json` canonical.*

## Utilization footer

Session 2026-08-11 20:42–20:4xZ (tick, babysit; 0 new GPU-h — GPUs
free): quiet tick. Registry empty, nvidia-smi 0%/0 MiB. Discord read
empty; history clean (no reaction yet on the 20:42Z servo-sysid post).
Queue validate OK (depth 2, 11 open); next = 100-seeds protocol
pre-reg, nothing blocking. run_work_next already armed → pre-reg work
session chains next. 18:43 body entry + 19:23 footer note rolled to
the archive.

Session 2026-08-11 19:26–20:4xZ (work, exploit-infra; 0 GPU-h — CPU
sim replays only): sim-servo-sysid CLOSED end-to-end — replay harness +
deps-free 6-param fit (2 starts, ~240 evals each, ~40 min CPU), held-out
arm replay MAE 3.31°→1.76°, SERVO_SYSID pinned into so101_sim.py, all
sim-fixes gates re-verified (spin improved 0.4°→0.1°), results post +
chart + json banked, sim-visual-matching queued. run_work_next armed
(100-seeds protocol pre-reg next).

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
