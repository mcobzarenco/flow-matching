# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-14 13:33–13:4xZ (real `date -u` at stamp: 13:41) —
work session: **`molmoact2-retirement-adoption` step (1) DONE —
fontaine rebased onto main 51704c0, all gates green, pushed.***

**Status**: **No live run** — GPU verified 0 MiB / 0% at 13:39,
OWNER-RESERVED hold (12:54:19Z) still in force; registry empty. Queue
validate green: depth 2, 15 open.

**Steering**: none — inbox empty at boot (13:33), `read` surfaced
nothing. Ladder verdict (STOP, 13:1xZ) still awaits owner
adjudication; owner presumed heads-down on the retirement phases in
main.

**Done**: **rebase step landed** (commit 3cac531 + posted in-channel
1537818089905983600): fontaine rebased onto main 51704c0 — 137
commits replayed over T1/T2; ONE conflict, exactly plan §0's
predicted surface (`model.py` `ar_predict_sampled` docstring:
`action_capture` doc kept, retired-ar_fast mention dropped; a
resolution-eaten newline caught by check.py and fixed same-session);
**check.py 858 green + grpo oracle suite 43 green** post-rebase;
pushed `--force-with-lease`, old tip tagged `pre-rebase-51704c0`.
Queue boundary updated to record step (1); steps 2–4 of the item
remain (phase 1–3 tracking, phase-4 co-land after adjudication).

**Next**: `run_work_next` armed — the chained work session writes the
`sim-manip-wrist-content-split` pre-reg (CPU-side; its ~0.02 GPU-h
embeds stay queued behind the owner reserve, so execution waits for
the in-channel GPU release). No launches until that release; ladder
adjudication pending; retirement-adoption steps 2–4 wait on owner
phase landings.*

*Updated 2026-08-14 13:30–13:3xZ (real `date -u` at stamp: 13:34) —
tick: **quiet — no steering, no live run; owner's retirement phase 0
visibly underway (tag `pre-molmoact2-retirement` pushed).***

**Status**: **No live run** — GPU verified 0 MiB / 0% at 13:30,
consistent with the OWNER-RESERVED hold (12:54:19Z); registry empty.
Queue validate green: depth 2, 15 open.

**Steering**: none — inbox empty, `read` surfaced nothing, history
shows no new reactions. Ladder verdict (STOP, posted 13:11Z) still
awaits owner adjudication; owner presumed heads-down on the
retirement implementation.

**Done**: observed the owner's phase-0 prep land on origin: annotated
tag `pre-molmoact2-retirement` → e3ec046 ("last commit where
bijou/molmoact2/ exists in full", fixture-provenance anchor per plan).
origin/main HEAD unchanged at 51704c0 — the queued rebase target
(≥ db0a141) remains satisfied; no queue edits needed. Archive rolled
--keep 3.

**Next**: `run_work_next` stays armed — the chained work session
takes `molmoact2-retirement-adoption` step (1): rebase fontaine onto
main 51704c0, check.py + grpo oracle suite green post-rebase;
`sim-manip-wrist-content-split` behind it. No launches until the
in-channel GPU release; ladder adjudication pending.*

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

## Utilization footer

Session 2026-08-14 13:33–13:4xZ (work; exploit; 0 GPU-h — GPU
owner-reserved, all CPU): `molmoact2-retirement-adoption` step (1)
landed — fontaine rebased onto main 51704c0 (137 commits, one
predicted conflict), check.py 858 + oracle suite 43 green, pushed
with the old tip tagged `pre-rebase-51704c0`; result posted
in-channel; queue validate green (depth 2, 15 open);
`run_work_next` armed for the wrist-content-split pre-reg.

Session 2026-08-14 13:30–13:3xZ (tick; 0 GPU-h — GPU owner-reserved):
quiet — no steering, no live run, queue validate green (depth 2, 15
open); owner phase-0 prep observed on origin (tag
`pre-molmoact2-retirement` → e3ec046); archive rolled --keep 3;
`run_work_next` left armed for the retirement-adoption rebase.

Session 2026-08-14 13:04–13:1xZ (work; exploit; 0 GPU-h — GPU
owner-reserved, all CPU): `grpo-r1b-boundary-reads` closed end-to-end
(calibration PASS, PRIMARY flat +0.0246 CI straddling 0, behavior
prediction falsified → competence-artifact finding; STOP recommended
for owner adjudication, post 1537810884318199889); step_0006
weights-only banked on fontaine-checkpoints; boundary chart on
fontaine-reports; queue reordered to the signed execution order
(depth 2, validate green); `run_work_next` armed.

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
