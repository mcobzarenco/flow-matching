# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-14 05:49–05:5xZ (real `date -u` at stamp: 05:50) —
tick: **quiet tick — no live runs, no steering, GPU idle-by-design
pending the owner's R1-A boundary call.***

**Status**: **no live runs** — GPU 0 MiB / 0% util, no train procs.
Idle is by design: launches pend `grpo-phase2-boundary-decision`
(owner_hold, options in-channel 03:1xZ).

**Steering**: none — inbox empty, read empty at 05:49Z; history (last
5) shows no new reactions or replies on the texture results post
(05:4xZ) or earlier asks. Still open: R1-A boundary options (03:1xZ),
arm-photometrics promotion (02:1xZ), clutter-patch promotion (05:40Z
08-13).

**Done**: Discord poll + history (facts above); queue validate green
(depth 2, 16 open); confirmed `run_work_next` armed (05:48 marker from
the texture session's close-out).

**Next**: chained work session picks up **sim-wrist-view-material-read**
(CPU + ~0.02 GPU-h) per no-idle-pauses; GPU launches wait on the
boundary call.*

*Updated 2026-08-14 04:46–05:5xZ (real `date -u` at stamp: 05:44) —
work session: **`sim-arm-texture-followup` executed end-to-end and
REFUTED cleanly — statistically-matched micro-texture reads MORE fake;
both registered CIs above zero. A one-session negative that kills the
composite-stage stats-matching class for texture.***

**Status**: **no live runs** — GPU idle-by-design pending the owner's
R1-A boundary call (`grpo-phase2-boundary-decision`, owner_hold,
options in-channel 03:1xZ 08-14); the gate read's embeds (~0.02 GPU-h)
were this session's only GPU touch.

**Steering**: none — inbox empty, read empty at 04:46 / 05:00 / 05:05
polls. Asks still open: R1-A boundary options (03:1xZ), arm-photometrics
promotion (02:1xZ), clutter-patch promotion (05:40Z 08-13). This
session's results post (05:4xZ) asks nothing — no promotion per the
frozen rule.

**Done**: **sim-arm-texture-followup CLOSED** (this commit): (1)
instrument — opt-in `arm_texture='v1'` composite-stage micro-texture
(deterministic static fields, private pinned RNG, zero shared-stream
draws, applied under seg masks pre-remap; 6 test oracles + init
checks, check.py 891 green); (2) fit — solve-based through the
production composite vs the mined real stats (PLA lc 8.24 vs real
8.36 dead-on; servo speckle-only, glint tail ~20% closed; two speckle
profiles rejected pre-read, recorded in the pre-reg); (3) pre-reg
posted in-channel 05:3xZ BEFORE the read with the explicit bar; (4)
registered 20×5 read, all gates green (v3_photo 0.698 dead-center):
**PRIMARY +9.33e-7 CI95 [+8.27,+10.42]e-7 ABOVE zero, 3/100, AUROC
0.698→0.751; MECHANISM +1.30e-6 [+1.22,+1.38]e-6, 0/100, 0.652→0.740
— REFUTED in the registered over-texturing direction**. The lesson
banked in ideas.md: the encoder reads spatial structure, not pooled
statistics. Artifacts on fontaine-reports (curl-200 ×5: analysis,
fit, chart, strip, zoom); reports.md section; results + disposition
in-channel 05:4xZ. Queue: item done, NEW
`sim-arm-surface-texture-mjspec` escalation (queued, NOT auto-run,
sequenced behind the wrist read).

**Next**: `queue_cli.py next` → **sim-wrist-view-material-read** (CPU
+ ~0.02 GPU-h; the wrist-side fact for the pending promotion asks);
GPU launches wait on the owner's R1-A boundary call. `run_work_next`
armed.*

## Utilization footer

Session 2026-08-14 05:51–06:1xZ (work; ~0.02 GPU-h decided — the
wrist gate read's embeds; CPU item, exploit-sim):
sim-wrist-view-material-read executed end-to-end (feasibility →
pre-reg with anchor honesty → paired 20×5 read: WRIST-NEUTRAL, CI
straddles zero — the promotion asks' wrist sanity measured); queue
depth 2 (16 open), NEW full-optin-stack item queued. `run_work_next`
armed.

Session 2026-08-14 05:49–05:5xZ (tick; 0 GPU-h decided — no live
runs, GPU idle-by-design pending the owner's R1-A boundary call):
quiet poll — inbox empty, no reactions or replies, queue green (depth
2, 16 open); `run_work_next` confirmed armed for
sim-wrist-view-material-read (CPU) per no-idle-pauses.

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
