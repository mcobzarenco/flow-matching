# Now










*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-14 10:48–11:1xZ (real `date -u` at stamp: 11:10) —
work session: **`sim-full-optin-stack-read` executed end-to-end
(pre-reg 10:54Z → read 10:58Z → results in-channel 11:00Z) — the
combined promotion is priced: clutter carries it, materials
absorbed.***

**Status**: **R1-B LIVE and healthy** — babysit exit 0 at 11:08Z: 3
procs, gpu0 33.7 GiB / 67%, step 5/15 (+0 steps since 10:46 — 47
min/step, step-6 row ~11:4xZ), held-out probe 1.84@4 → 1.89@5
(record-only vs the 1.868 banked baseline), anchor_kl 0.041 < 0.06,
rc ETA ~19:3xZ holds. Knockaway watch stands: 0.328, streak 1/3 vs
the 0.167 line.

**Steering**: none — inbox empty, no new owner messages at either
poll (10:48 boot, 11:08 babysit). No reactions on the step-5
calibration post yet.

**Done**: **`sim-full-optin-stack-read` CLOSED** (script
`sim_full_optin_stack_read.py` + chart, this commit): pre-reg posted
10:54:40Z BEFORE the read (explicit ε=0.005 bar); read 10:58Z exit 0,
ALL gates green — in-run v3 0.7127 band-center, in-run patched 0.5561
**bit-matching the banked fg-fix read**, cross-instance
qpos/draws/affine bit-equal ×100. Adjudication = the frozen MIDDLE
branch: paired stack vs v3 **−2.075e-06 CI [−2.254,−1.891]e-06**
(99/100) but stack AUROC **0.5521 > bar 0.5511** — beats clutter-alone
by only −0.0040 < ε. Materials' marginal on top of clutter
**−5.50e-08 CI [−1.44e-07,+3.37e-08]** (56/100): ~⅓ of the banked
solo effect, statistically absorbed; additivity interaction **+0.0063
(sub-additive)**. Disposition posted in-channel 11:00Z + on the three
promotion asks' queue boundaries: clutter patches carry the combined
gain (promote first/alone); material flags safe to stack but not
additive as sold; bigger-n marginal read owner-priced. check.py 904
green; queue reshaped (item closed, promotion asks annotated,
`sim-appearance-consolidated-report` queued as the closed-screen
refill — depth 2, validate green).

**Next**: `run_work_next` ARMED — the chained work session takes
`sim-appearance-consolidated-report` (CPU, banked numbers only)
alongside the run; tick chain keeps ~30-min babysit checkpoints. At
rc (~19:3xZ): `grpo-r1b-boundary-reads` — accumulate or the ladder
STOPS.*

*Updated 2026-08-14 10:45–10:5xZ (real `date -u` at stamp: 10:46) —
tick: **R1-B healthy at step 5/15, babysit green, owner 👍 on the
pre-reg recorded.***

**Status**: **R1-B LIVE and healthy** — babysit exit 0 at 10:46Z: 3
procs, gpu0 33.7 GiB / 100%, step 5/15, loss 0.058, 47 min/step (~7.8
h to step 15, rc ETA ~19:3xZ holds), anchor_kl 0.041 < 0.06 stop,
VRAM 33.89 of the 75 gate. Calibration read done last session (PASS,
posted 10:43Z); next fresh row (step 6) ~11:4xZ. Watch item stands:
knockaway 0.328, streak 1/3 vs the 0.167 line — registered
prediction is decay.

**Steering**: no new messages, inbox empty. **Reaction: 👍 on the
R1-B pre-reg post (09:43:09Z)** — owner agreement with the patched
reward + re-priced ladder, recorded per the 08-05 reaction rule. No
reactions on the step-5 calibration post yet.

**Done**: babysit poll (facts above, no gate crossing, no anomaly in
the printed trajectories); Discord read + history; queue validate
green (depth 2, 15 open); confirmed `run_work_next` armed.

**Next**: chained work session takes `sim-full-optin-stack-read`
(CPU item) alongside the run; tick chain keeps ~30-min babysit
checkpoints. At rc (~19:3xZ): `grpo-r1b-boundary-reads` — accumulate
or the ladder STOPS.*

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

## Utilization footer

Session 2026-08-14 10:48–11:1xZ (work; exploit; ~0.02 GPU-h embeds —
R1-B live within its ~9.6 GPU-h envelope): `sim-full-optin-stack-read`
executed end-to-end same session (pre-reg → read → results + chart
in-channel); combined promotion priced (clutter carries it, materials
absorbed, interaction +0.0063 sub-additive); promotion asks annotated;
babysit green at 11:08Z; `sim-appearance-consolidated-report` queued,
`run_work_next` armed for it.

Session 2026-08-14 10:45–10:5xZ (tick; 0 GPU-h decided — R1-B live
within its ~9.6 GPU-h pre-reg envelope): babysit green at step 5/15
(exit 0, all wires quiet), owner 👍 on the R1-B pre-reg recorded as
agreement, queue green (depth 2, 15 open), `run_work_next` confirmed
armed for `sim-full-optin-stack-read`.

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
