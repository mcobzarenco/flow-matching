# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-15 08:58–12:0xZ (real `date -u` at stamp: 11:56) —
work session: **owner steering morning — stage-C killed on order at
step 2040, step2000 probed: 28/100 unseen successes (14–28× the
pre-SFT anchors); the quantile class bug found+fixed; their-trainer
retired.***

**Status**: **LIVE** — `fontaine-grasp-sft-step2000-probe` train arm
(seeds 1000–1099, started ~11:51Z after the unseen arm banked, ETA
~13:5xZ, detached unit; babysit entry current). Unseen arm FINAL:
**28/100 successes**, 42 moved, mean progress +1.97 cm, 0 strikes —
vs anchors ftrig4k ~1/100, W0 2/100. Stage-C training is DEAD (owner
kill order 10:10Z at step 2040, loss 0.0246 green; checkpoints
500–2000 retained, step2000 converted).

**Steering**: heavy morning exchange (09:49–10:46Z, 9 owner messages,
all replied + acked): (1) **their `train_lerobot.py` retired** — all
training via `bijou.train` (memory `bijou-train-only`); (2) owner
caught the R2 draft head seam → §7 Amendment A1 registered
(token-SFT-before-token-GRPO route); (3) kill+probe order executed;
(4) owner caught the wrist_roll anomaly in the train256 report →
root-caused to the **lerobot quantile aggregation class bug** (q01/q99
= weighted mean of per-episode quantiles; roll box [35.5,94.4] vs true
±157° — 19% of training frames clamped out) — fixed in
`collect_demos.rewrite_quantile_stats()` + oracle, dataset corrected +
re-uploaded; the practice existed in `docs/data-curation.md` §3 and is
now enforced in code; (5) units question answered (vendored lerobot
`use_degrees=True` default = degrees everywhere).

**Done** (commits `2439869`→`4b5xxxx`-class this session): chain
results page pre-built (4 house dark charts + plain-words open, DRAFT
banner now records the re-steer); stage-C delta-upload script
oracle-tested vs rig-r1 (590/705 counts); `fontaine-sim` dataset repo
created + grasp_sft_demos_v0 uploaded (owner ask); train256 eval
report on fontaine-reports (curl-200, owner ask); step2000 probe
launched + unseen arm banked + reads script landed; R2 Amendment A1;
quantile class fix; queue boundary instructions ×2 (429-resilient).
In-session GPU ~1.8 h (probe unseen arm + train256 eval); stage-C
chain spend to kill ~0.9+4.0+2.7.

**Next**: `queue_cli.py next` → grasp-sft-bootstrap next-session remit
(on the queue item): train.json read + comparison post (~13:5xZ);
step2000 delta upload; retrain via `bijou.train` on the corrected
table = the value-unlock, **owner-gated**; probe report page.
`run_work_next` armed — the chained session owns the train-arm
boundary.*

*Updated 2026-08-15 08:45–08:5xZ (real `date -u` at stamp: 08:47) —
tick: **stage-C AR riding green at 960/3000 (loss 0.464 → 0.038);
owner status question answered in-channel; the 08:35 harness alert
diagnosed as a usage-credit 429 (resets 22:00Z).***

**Status**: **LIVE** — `fontaine-grasp-sft-stagec-ar`: 960/3000 steps
at 08:47Z, 12.3 steps/min cumulative (12.7 window), GPU 100%/38.9 GiB,
action_flow_loss 0.464 (step 20) → **0.038** (step 960) — the rig-ft r1
"materially below warm-start by ~570" reference cleared with room.
Projection ~4.1 GPU-h vs the 5.0 gate, endpoint **~11:50Z** → stage D
sim100. Babysit exit 0, no gate crossings, 500-step checkpoint banked.

**Steering**: owner 08:29Z "How's the train run going?" — answered
08:48Z with the full status (post 1538106787067068516), acked, inbox
clear. Conversational hold ran ~10 min after the reply (history-based
watch, cursor untouched); no follow-up. No new reactions in `history`.

**Done**: babysit + Discord polls, owner reply + ack, **429 diagnosis**:
the 07:04 work session died 08:35Z on `out_of_credits` (7-day overage
pool, `resetsAt` 22:00Z 08-15) — all its work was already committed
(`0bac17a`, `98524dd`); the 08:35:20Z tick died instantly on the same
429; this 08:45 tick ran normally (rolling window freed base quota).
The detached training unit is unaffected by session 429s. Queue
validate OK depth 3 (17 open). `run_work_next` re-armed. 0 GPU-h.

**Next**: sessions are credit-flaky until 22:00Z — expect possible
tick/work 429 exits; the run keeps training regardless and babysit
re-syncs at the next successful session. Stage-C endpoint ~11:50Z:
convert + stage-D sim100 per the frozen verdict surface (≥20/100 GRPO
GO / 5–19 iterate-once / <5 F-transfer); `grpo-r2-post-sft` activation
rides the stage-D read.*

*Updated 2026-08-15 07:02–07:0xZ (real `date -u` at stamp: 07:02) —
tick: **stage-B green at 274/400 kept; wall boundary (07:29:18Z)
confirmed handed to the chained work session.***

**Status**: **LIVE** — `fontaine-grasp-sft-stageb`: 274/400 kept at
07:02Z, 4 procs, GPU 55%/989 MiB, window 1.8 kept/min. Babysit exit 3 =
the known keep-rate projection (5.2 h to 400 kept vs the ≤4 gate) —
judged NOT a new anomaly: the §8-recorded 62.5% true rate story, the
4-h wall self-stop at 07:29:18Z enforces the gate, run rides untouched
per the frozen no-mid-run-changes term.

**Steering**: none — inbox empty, `read` surfaced only our own 07:01Z
close post, `history -n 5` shows no new reactions.

**Done**: babysit + Discord polls, queue validate OK (depth 2, 16
open), `run_work_next` confirmed armed (07:00 touch intact). No posts
(quiet tick, nothing owner-facing changed). 0 GPU-h.

**Next**: the wall lands ~2.5 min before this tick's hard kill — the
chained work session owns the 07:29:18Z boundary per the queue-item
instructions: kept ≥300 → stage-C AR launch; 290–299 → recorded top-up
first; <290 → diagnose.*

## Utilization footer

Session 2026-08-15 08:58–12:0xZ (work; exploit; ~1.8 GPU-h in-session
— probe unseen arm + train256 eval; stage-C accrued ~2.7 to its own
gate before the owner kill): owner-steering morning — kill+probe order
executed same-hour (28/100 unseen), quantile class bug found+fixed+
re-uploaded, their-trainer retired, R2 A1 registered, fontaine-sim
created, train256 report served. Train arm rides detached;
run_work_next armed.

Session 2026-08-15 07:02–07:0xZ (tick; 0 GPU-h): stage-B ride check —
274/400 kept green (4 procs, GPU 55%), babysit exit-3 judged as the
known §8 keep-rate projection (wall self-stop enforces the ≤4 gate, no
new anomaly), Discord/inbox empty, queue validate OK depth 2,
`run_work_next` confirmed armed — the chained work session owns the
07:29:18Z wall boundary.

Session 2026-08-15 03:37–07:1xZ (work; exploit; 0 GPU-h in-session —
stage-B collection rides detached on its own ≤4 gate): stage-C launch
prep DONE (verbatim-class launchers + oracle-tested preflight +
owner-side mixture 7fb6552) and stage-D eval prep DONE (convert+eval
launcher + frozen verdict surface, band edges oracle-tested) — the
whole remaining GPU ladder is launch-ready behind preflights; stage-B
ridden 6 polls green with the keep-rate cliff diagnosed CPU-side (true
rate 62.5% n=200, no bug, no drift; mid-ride post 04:13Z + prereg §8);
wall-tick boundary instructions on the queue item; queue validate
green depth 2; run_work_next armed for the 07:29Z wall.

Session 2026-08-15 03:27–03:3xZ (tick; 0 GPU-h in-session — stage-B
collection rides detached, counted at its boundary): owner 👍 on the
01:40Z prereg-finalization post surfaced at the history poll (explicit
go, window collapsed) → grasp-SFT **stage-B demo collection LAUNCHED
03:29:18Z** (unit `fontaine-grasp-sft-stageb`, target 400 kept, gate
≥300 / ≤4 GPU-h, resume-capable); first poll green (seed 1000 KEPT
~40 s in, GPU 50%/909 MiB); babysit entry + queue boundary record +
in-channel launch post; queue validate OK depth 2; `run_work_next`
armed for the ride + `grasp-sft-stage-c-launch-prep`.

Session 2026-08-15 01:48–03:2xZ (work; exploit; ~0.6 GPU-h — two
rendered stage-A gate reads + diagnostic/smoke episodes): grasp-SFT
stage A taken from finalized to CLOSED through a full
fail→diagnose→amend→pass cycle (gate FAIL 11/20 on held 1020–1039;
four mechanisms measured + fixed `77776fd`; A1 registered with a
one-amendment cap; fresh held 1040–1059 PASS 15/20); stage-B LeRobot
collector landed (`5b360fa`, oracles + GL smoke) — collection is
launch-ready at the next boundary; wrist-screen results page published
(`fb1e672`); three in-channel boundary posts + minutely polling
through the A1 window; queue depth 2 maintained (collector closed,
stage-C launch-prep queued).

Session 2026-08-15 01:44–01:4xZ (tick; 0 GPU-h): no-op verification —
GPU idle confirmed, Discord/inbox empty (no objection to the 01:43Z
grasp-SFT finalization yet), queue validate OK depth 2,
`run_work_next` confirmed armed for the stage-A gate-read work
session.

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
