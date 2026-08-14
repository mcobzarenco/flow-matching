# Now







*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-14 05:51–06:1xZ (real `date -u` at stamp: 06:08) —
work session: **`sim-wrist-view-material-read` executed end-to-end —
WRIST-NEUTRAL: the two-flag stack's paired wrist Δknn5 CI straddles
zero; the promotion asks' wrist-side sanity is now measured, not
assumed.***

**Status**: **no live runs** — GPU idle-by-design pending the owner's
R1-A boundary call (`grpo-phase2-boundary-decision`, owner_hold,
options in-channel 03:1xZ 08-14); this session's only GPU touch was
the read's ~0.02 GPU-h embeds.

**Steering**: none — inbox empty, read empty at 05:52 / 06:07 polls
(only my own pre-reg + results posts in-channel). Asks still open:
R1-A boundary options (03:1xZ), arm-photometrics promotion (02:1xZ),
clutter-patch promotion (05:40Z 08-13) — all three now carry the
measured wrist-neutral fact.

**Done**: **sim-wrist-view-material-read CLOSED** (this commit):
pre-reg posted in-channel 05:59Z BEFORE the read (with the anchor
honesty registered: the queued 0.828 wrist anchor is ROLLOUT-frame;
the reset-pose baseline is 0.544/0.548, the gate band [0.50, 0.60]);
registered 20×5 paired read all gates green (top 0.713 dead-center,
wrist 0.561 in-band, qpos bit-equal ×100, changed-px tripwire quiet):
**PRIMARY wrist Δknn5 −1.39e-08 CI95 [−4.53, +1.73]e-08 straddles
zero (46/100) → wrist-neutral per the frozen rule**; mechanism
diagnostic: the home-pose wrist camera sees ~230 raw px of graded
surface (servo 208 / PLA 21 / mount 1); top rider replicated the
mount read's stack delta bit-for-bit (hook path ≡ production
observations — a free bit-exactness cross-check). Artifacts on
fontaine-reports curl-200 ×3 (analysis/chart/strip); results section
on the pre-reg page; reports.md + ideas.md banked; posts/index.md
drift fixed (mount + texture pre-regs added). Queue: wrist done, NEW
`sim-full-optin-stack-read` (prices the three promotions flipping
together — interactions unmeasured; depth 2, 16 open, validate
green).

**Next**: `queue_cli.py next` → **sim-arm-surface-texture-mjspec**
(the registered texture escalation, recompile path, NOT auto-run per
its boundary note — owner may reprioritize; then
`sim-full-optin-stack-read`). GPU launches wait on the owner's R1-A
boundary call. `run_work_next` armed.*

## Utilization footer

Session 2026-08-14 08:35–08:5xZ (tick; 0 GPU-h decided — no live
runs, GPU idle-by-design pending the owner's boundary call): owner
GRPO-status question answered in-channel 08:37Z + inbox acked;
exit-1 outage window 06:24–08:24Z diagnosed (usage-cap signature,
cleared); the dead 06:24 work session's arm_texture v2 WIP audited
(9/11 oracles) and preserved as a check-exempt patch (`862d012`);
queue green (depth 2, 16 open); `run_work_next` armed to resume the
mjspec item.

Session 2026-08-14 06:22–06:2xZ (tick; 0 GPU-h decided — no live
runs, GPU idle-by-design pending the owner's R1-A boundary call):
quiet poll — inbox empty, no reactions or replies, queue green (depth
2, 16 open); `run_work_next` confirmed armed for the
sim-arm-surface-texture-mjspec CPU instrument per no-idle-pauses.

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
