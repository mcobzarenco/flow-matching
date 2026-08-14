# Now








*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-14 08:45–10:0xZ (real `date -u` at stamp: 09:55) —
work session: **texture escalation CLOSED (second refutation) + owner
GRPO steering executed end-to-end — reward patch landed and R1-B
LAUNCHED under it, all in one session.***

**Status**: **R1-B LIVE** — unit `grpo-phase2-r1b` launched 09:43:20Z
(steps 5–14 resuming R1-A's step_0004 into fresh `grpo_phase2_b`; lr
3e-7, kl_beta 1.0, **train_reward v2**). GPU 33.6 GiB / 100% (R1-A
envelope); first heartbeat 09:54Z: the duplicate step-4 eval row reads
**1.8441, 2/20, Δ −0.0239 — bit-matching the banked R1-A read**
(resume correctness confirmed live; baseline rode the checkpoint).
**Step-5 row 10:42Z — CALIBRATION PASS**: 8/8
groups kept, std 3.27 cm; decomposition earned **1.19** vs shoved
**4.98 cm** (~4:1 shove:carry — the leakage, measured);
setback_frac 0.703 vs knockaway 0.328 (excursion channel sees 2×
the endpoint stat); mechanics green (anchor_kl 0.041 < 0.06,
ratio 1.00026, 47 min/step). Knockaway streak 1/3 vs the 0.167
line — prediction on record: decays. rc ETA ~19:3xZ; ~9.6 GPU-h,
ladder cum ~14.7 of the 22 gate.

**Steering**: owner 09:16:39Z — "let's try your recommendation (2)
then (1). How is knock away currently defined? Do we actually do a
good job of defining it?" Replied in-channel 09:21Z (code-grounded
audit: endpoint-only, tripwire-only, reward-funded shoving blind spot,
no grasp channel), acked, then EXECUTED same session: option (2) is
code, option (1) is live.

**Done**: (1) **sim-arm-surface-texture-mjspec CLOSED — SECOND
REFUTATION** (`e408f9e` instrument, `92ae859` close): resumed the
orphaned WIP, fixed both red oracles (zero-clip tanh generator;
tabletop-reflection rider, mechanism confirmed), wrote the real fit
(period 32 at the plausibility bound, amplitude capped at the 0.42
no-clip headroom → lc 6.43 of real 8.36), pre-reg 09:14Z BEFORE the
read → 20×5 gates all green, **PRIMARY +3.07e-07 CI [+2.42,+3.71]e-07
(0.698→0.718)**: coherent surface-tracking bands still read MORE fake —
arm-texture direction COLD, graded arm stays the frontier; surviving
hypothesis banked (real layer contrast is RELIEF/light-transport, not
albedo). (2) **Grasp instrument + reward v2** (`5932fb6`):
`benchy_grip_contacts()` two-sided pinch predicate, per-tick grip
trace, `grasped_progress_cm`/`ungrasped_displacement_cm`/
`max_setback_cm`; `composite_reward_v2` = earned − 0.5·shoved (4 cm
shove −2.0 vs 4 cm carry +4.0, oracle-pinned); eval metric stays v1;
13 new oracles, check.py 904 green. (3) **R1-B pre-reg (posted 09:43Z
before launch) + launch** (`3c7ed82`); babysit registry entry with
the calibration bar. Queue: texture + boundary-decision + patch +
r1b-launch items closed, `grpo-r1b-boundary-reads` queued (depth 2,
15 open, validate green).

**Next**: tick chain babysits R1-B (~30-min checkpoints, poll forced
last; calibration read done, in-channel 10:43Z). At rc (~19:3xZ):
`grpo-r1b-boundary-reads` (accumulate or the ladder STOPS). Next CPU
item while GPU busy: `sim-full-optin-stack-read`.*

*Updated 2026-08-14 08:35–08:5xZ (real `date -u` at stamp: 08:42) —
tick: **owner asked for GRPO status (answered in-channel 08:37Z) +
recovered the exit-1 outage window's orphaned WIP.***

**Status**: **no live runs** — GPU 0 MiB / 0% util. Idle is by design:
launches pend `grpo-phase2-boundary-decision` (owner_hold, options
in-channel 03:1xZ, re-surfaced 08:37Z). **Harness outage window**:
every session 06:24Z–08:24Z exited 1 within ~2 s of start (work
session 06:24 + 7 ticks; alerts posted in-channel 06:35/07:40) —
signature matches a usage-cap window; this 08:35 session ran
normally, so it has cleared. Consequence: no session completed for
~2 h and the 06:24 work session died mid-item.

**Steering**: owner 08:31:17Z — "Where are we with the GRPO
experiments?" Replied in-channel 08:37Z (R1-A tripwire stop at step
5/17, held-out flat/unharmed, ~5.1 of 22 GPU-h, the three boundary
options re-surfaced with the (2)-then-(1) recommendation), inbox
acked. No follow-up by 08:4xZ; the boundary call stays open. No
reactions on earlier posts.

**Done**: orphan audit — the dead 06:24 work session left
`sim-arm-surface-texture-mjspec` WIP uncommitted (arm_texture='v2'
mjspec recompile path + albedo mean-compensation + 10 oracles).
Audited: 9/11 oracles green, 2 RED (clipping 5.4% vs <1% bar;
PLA-locality halo) — mid-calibration, NOT landed work, so no
check-skip commit; preserved as a 408-line patch at
`fontaine/harness/state/wip_arm_texture_v2_orphan_20260814T0624Z.patch`
(check-exempt path, committed `862d012`), working tree left dirty
for the chained session. Queue validate green (depth 2, 16 open).

**Next**: `run_work_next` armed — the chained work session resumes
**sim-arm-surface-texture-mjspec** from the WIP (fix the two red
oracles BEFORE any pre-reg/read; nothing was registered or read).
GPU launches wait on the owner's boundary call; if the owner
answers, that supersedes.*

*Updated 2026-08-14 06:22–06:2xZ (real `date -u` at stamp: 06:24) —
tick: **quiet tick — no live runs, no steering, GPU idle-by-design
pending the owner's R1-A boundary call.***

**Status**: **no live runs** — GPU 0 MiB / 0% util, no train procs.
Idle is by design: launches pend `grpo-phase2-boundary-decision`
(owner_hold, options in-channel 03:1xZ).

**Steering**: none — inbox empty, read empty at 06:22Z; history (last
5) shows no reactions or replies on the wrist results post (06:07Z)
or earlier asks. Still open: R1-A boundary options (03:1xZ),
arm-photometrics promotion (02:1xZ), clutter-patch promotion (05:40Z
08-13) — all three now carry the measured wrist-neutral fact.

**Done**: Discord poll + history (facts above); queue validate green
(depth 2, 16 open); confirmed `run_work_next` armed (marker present at
06:22).

**Next**: chained work session takes `queue_cli.py next` →
**sim-arm-surface-texture-mjspec** (CPU instrument + oracles; its
boundary note bars auto-running the gate read out of sequence — the
wrist read it was sequenced behind is now done, and the recompile +
physics-oracle work is CPU-side either way; `sim-full-optin-stack-read`
follows). GPU launches wait on the boundary call.*

## Utilization footer

Session 2026-08-14 08:45–10:0xZ (work; exploit; ~0.04 GPU-h spent on
the texture gate read embeds + **~9.6 GPU-h committed** by the R1-B
launch 09:43:20Z, ≤ 22-gate cum ~14.7): texture escalation closed
(second refutation, pre-reg'd read); owner GRPO steering answered
09:21Z and executed — grasp instrument + reward v2 landed (904
green), R1-B pre-reg posted then launched under it; queue reshaped
(depth 2, validate green).

Session 2026-08-14 08:35–08:5xZ (tick; 0 GPU-h decided — no live
runs, GPU idle-by-design pending the owner's boundary call): owner
GRPO-status question answered in-channel 08:37Z + inbox acked;
exit-1 outage window 06:24–08:24Z diagnosed (usage-cap signature,
cleared); the dead 06:24 work session's arm_texture v2 WIP audited
(9/11 oracles) and preserved as a check-exempt patch (`862d012`);
queue green (depth 2, 16 open); `run_work_next` armed to resume the
mjspec item.

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
