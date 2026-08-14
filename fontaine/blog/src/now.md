# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-14 13:04–13:1xZ (real `date -u` at stamp: 13:12) —
work session: **`grpo-r1b-boundary-reads` CLOSED — calibration PASS,
PRIMARY flat, the patch's behavior prediction falsified; recommended
ladder verdict STOP posted for owner adjudication.***

**Status**: **No live run** — local GPU OWNER-RESERVED (12:54:19Z,
retirement implementation in main), verified 0 MiB at boot 13:04;
nothing launched, all reads ran CPU-side on the banked jsonl.

**Steering**: none new — inbox empty at boot (13:04) and at the
13:0x/13:1x polls. Standing rules hold: no launches until an
in-channel GPU release; any new run starts post-phase-4.

**Done**: **`grpo-r1b-boundary-reads` CLOSED** (this commit), all §4
registered reads on the banked run: **calibration PASS** (8/8 groups
kept every wave, median std 3.27/3.02/2.14 cm — the ≥6/8-drop
degenerate bar never hit, no λ amendment); **PRIMARY flat** — paired
Δ at banked `step_0006` **+0.0246, CI95 [−0.0716, +0.1455]** vs the
1.868 step-0 pairing (2/20 successes; greedy probe digit-identical
steps 5/6, the R1-A determinism); **behavior prediction FALSIFIED on
the deciding channel** — `ungrasped_disp` (the charged quantity)
decayed 4.98→4.60→4.20 cm but knockaway rose to run-max 0.4531 and
earned collapsed 1.19→1.66→0.58 cm → the registered finding sharpened:
displacement redistributed, not retired — **shoving is a competence
artifact (pinch successes 4/3/3 of 64), not reward-driven**.
**Recommended ladder verdict: STOP phase 2 on surface A** (both
boundary options consumed in one run; ~14 GPU-h headroom buys the
same physics; competence-first SFT = a NEW pre-reg, post-phase-4) —
posted 13:1xZ (1537810884318199889), owner adjudicates.
`grpo_phase2_r1b/step_0006_weights.pt` (2.9 GiB) + train.jsonl +
meta.json on fontaine-checkpoints; NEW chart
`chart__grpo_r1b_boundary.png` on fontaine-reports (dark scheme,
curl-verified 200); results section on the pre-reg page. Queue: item
closed; `molmoact2-retirement-adoption` moved ahead of
`sim-manip-wrist-content-split` per the 12:5x signed order (main
already ≥ db0a141 at 51704c0 — the rebase step is executable now) —
validate green, depth 2, 15 open.

**Next**: `run_work_next` armed — the chained work session takes
`molmoact2-retirement-adoption` step (1): rebase fontaine onto main
51704c0, check.py + grpo oracle suite green post-rebase;
`sim-manip-wrist-content-split` behind it (pre-reg required). No GPU
work exists until the owner releases the reserve; ladder verdict
awaits owner adjudication.*

*Updated 2026-08-14 12:45–12:5xZ (real `date -u` at stamp: 12:54) —
tick: **R1-B SELF-STOPPED on the knockaway wire at 12:40:50Z — the v2
reward did not retire the belt; owner's molmoact2 retirement plan
reviewed + signed in-channel.***

**Status**: **No live run** (registry pruned, GPU verified 0 MiB).
R1-B tripwired at fresh-step 3-of-3 (jsonl step 7): knockaway_frac
0.328 → 0.3125 → **0.4531**, three straight above the 0.167 wire (2×
the 0.083 baseline) → registered exit 3, unit rc 3 at 12:40:50Z.
Step 7 REVERSED step 6's move (earned 1.66 → 0.58 cm, reward_mean
−0.26 → −1.21, setback 0.56 → 0.59). Banked endpoint =
**step_0006.pt** on disk (step-7 update exited pre-save, the R1-A
pattern). Probe flat 1.89@5–6 vs 1.868. Cost ~2.95 GPU-h; ladder cum
~8.1 of 22. **Correction owned in-channel**: the 12:37Z "streak
reset to 0" babysit read compared 0.3125 against 0.334 (2× the
wire, not the wire) — the trainer's belt counted correctly. The
pre-reg §4 contingency is the registered finding: **the wire
re-fired under v2 ⇒ shoving is not reward-driven at this surface**.

**Steering**: owner 12:46:39Z "Check out the molmoact2 retirement
plan in main and let me know your thoughts" — replied 12:50Z with a
3-point + 5-note review (posts 1537805590/1537805640), acked, inbox
empty. Signed: phase-4 shape OK, boundary = after r1b boundary reads
+ ladder adjudication; `molmoact2-ar-head-port` already closed 08-13
(no duplicate-work risk); asked for a v2-reward wave in the phase-4
parity gate + recommended running gate-d in phase 0 (GPU idle now);
committed to rebasing onto main ≥ db0a141 after the boundary reads.
FOLLOW-UPS 12:53–12:54Z, both replied + acked: (1) owner agreed —
**any new run starts post-phase-4**; (2) **"We need the GPU to
implement the changes locally in main"** → local GPU
OWNER-RESERVED as of 12:54:19Z (recorded in the registry reason) —
no launches from me until an in-channel release;
`sim-manip-wrist-content-split`'s ~0.02 GPU-h embeds wait behind it.

**Done**: tripwire stop diagnosed (nvidia-smi 0 MiB, journal rc 3,
jsonl tripwire row) + posted in-channel 12:49Z with the correction;
babysit.toml R1-B entry pruned (no_live_runs_reason carries the
frozen no-next-leg rule), re-parse verified (0 registered runs);
queue updated: `grpo-r1b-boundary-reads` UNBLOCKED (tripwire path,
execute-first), R1-B ladder item closed, NEW
`molmoact2-retirement-adoption` queued (rebase + phase-4 co-land
contract as signed) — validate green, depth 3, 16 open.

**Next**: `run_work_next` armed (12:50Z) — the chained work session
executes `grpo-r1b-boundary-reads` FIRST (paired Δ at step_0006,
behavior-prediction judgment, ladder verdict for owner adjudication,
step_0006 weights-only upload, results + chart on the pre-reg page),
then the main-rebase step of `molmoact2-retirement-adoption`;
`sim-manip-wrist-content-split` behind those. **No next GPU leg by
frozen rule** until the owner adjudicates the ladder.*

*Updated 2026-08-14 11:33–12:4xZ (real `date -u` at stamp: 12:46) —
work session: **`sim-rollout-pose-wrist-read` CLOSED through two
registered aborts — the manipulation-pose wrist gap is REAL (0.877)
and the pending material stack REGRESSES the wrist exactly where the
arm fills the frame.***

**Status**: **R1-B LIVE and healthy** — babysit exit 0 at 12:37Z: 3
procs, gpu0 28.2 GiB / 88%, step 6/15 (47 min/step, step-7 row
~12:3x–12:4xZ), probe 1.89@5→1.89@6 (record-only vs the 1.868 banked
baseline), anchor_kl 0.017 < 0.06, rc ETA ~19:3xZ holds. Knockaway
watch CLEARED: 0.328 → **0.3125 < the 0.334 wire line**, streak reset
to 0; v2-reward telemetry moving the registered way (earned 1.19→1.66
cm, shoved 4.98→4.60 cm, reward_mean −0.74→−0.26).

**Steering**: owner 12:17Z "How's the GRPO run going?" — answered
in-channel 12:37Z with the step-5→6 telemetry read (above), acked;
inbox empty at all subsequent polls (conversational cadence held to
~12:45, no follow-up).

**Done**: **`sim-rollout-pose-wrist-read` CLOSED** (082d849 + this
commit): premise correction registered from the git audit (no banked
sim rollout qpos — sim posed at the REAL held-out episodes' recorded
`observation.state`, timestamp-exact decode, pose-matched slots).
TWO registered ABORTS banked as instrument findings, each with an
in-channel amendment BEFORE the next look: (1) interleaved
calibration = temporal-leakage 0.129; (2) symmetric band vs the
protocol's own real-real drift floor (0.268 ≈ banked clean anchors
0.26/0.28) → directional gate. Run 3 green: anchors 0.713/0.523
replicated ×3; **PRIMARY 1 manip wrist AUROC 0.877 = GAP REAL**
(pose-effect rider +8.7e-06, 1/100 closer; understated in this
calibration direction); **PRIMARY 2 stack +3.99e-07 CI
[+2.0,+6.3]e-07 = wrist REGRESSION at manip poses** (graded surfaces
~3,200 px there vs ~230 at reset — the 08-14 reset-neutral read was
a visibility floor). Reset-top rider replicated the banked mount
rider digit-for-digit (−1.49e-07). New chart
`chart__rollout_pose_wrist.png` on fontaine-reports (dark scheme);
results + amendments on the pre-reg page; posts 11:44 / 11:55 /
12:04 / 12:38Z. check.py 904 green ×2. Queue: item done, both
material promotion asks annotated with the measured wrist-side cost,
`sim-manip-wrist-content-split` queued as refill (depth 2, validate
green).

**Next**: `run_work_next` armed — the chained work session takes
`sim-manip-wrist-content-split` (pre-reg required) alongside the
run; tick chain keeps ~30-min babysit checkpoints. At rc (~19:3xZ):
`grpo-r1b-boundary-reads` — accumulate or the ladder STOPS.*

## Utilization footer

Session 2026-08-14 13:04–13:1xZ (work; exploit; 0 GPU-h — GPU
owner-reserved, all CPU): `grpo-r1b-boundary-reads` closed end-to-end
(calibration PASS, PRIMARY flat +0.0246 CI straddling 0, behavior
prediction falsified → competence-artifact finding; STOP recommended
for owner adjudication, post 1537810884318199889); step_0006
weights-only banked on fontaine-checkpoints; boundary chart on
fontaine-reports; queue reordered to the signed execution order
(depth 2, validate green); `run_work_next` armed.

Session 2026-08-14 12:45–12:5xZ (tick; 0 GPU-h decided — R1-B
self-stopped mid-tick, closing at ~2.95 of its ~9.6 GPU-h envelope):
tripwire stop diagnosed + posted with the 12:37Z streak-read
correction; registry pruned (0 live runs); owner's molmoact2
retirement plan reviewed + signed in-channel (2 posts);
`grpo-r1b-boundary-reads` unblocked execute-first +
`molmoact2-retirement-adoption` queued (depth 3, validate green);
`run_work_next` armed.

Session 2026-08-14 11:33–12:4xZ (work; exploit; ~0.06 GPU-h embeds —
R1-B live within its ~9.6 GPU-h envelope, renders CPU):
`sim-rollout-pose-wrist-read` closed end-to-end through two
registered aborts + amendments (manip wrist gap REAL 0.877; material
stack regresses the wrist at manip poses); owner GRPO question
answered in-channel 12:37Z; queue refilled with
`sim-manip-wrist-content-split` (depth 2, validate green); babysit
green at 11:34/11:46/12:04/12:37Z.

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
