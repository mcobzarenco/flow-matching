# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-13 14:22–14:3xZ (real `date -u` at stamp: 14:24) —
tick, babysit: **quiet tick — no live runs, no new steering; the
14:09Z owner exchange held open to the ~10-min silence handback,
`run_work_next` armed for the critical path (instrument item 4 +
phase-2 pre-reg finalization → GPU legs launch).***

**Status**: no live runs — babysit exit 0, 0 registered; nvidia-smi
0%/0 MiB (idle-by-design until the phase-2 pre-reg lands — the CPU
critical path IS the unblock). Queue validate green (depth 2, 14
open).

**Steering**: none new — read empty 14:23Z; history-5 shows no
reactions and no owner follow-up to our 14:14Z grammar-mask/seed-73
answer (~10 min silence at close → conversational window handed
back; a short in-session watcher covered the tail of the window).
The 11:07/11:18Z delegation stands.

**Done**: liveness/queue/GPU verified; `run_work_next` armed 14:23Z
so the driver chains straight into a work session for the critical
path.

**Next**: chained work session → instrument item 4 (loop harness:
rollout wave → score → z-filter → step → periodic eval + babysit
heartbeat + registry entry), then phase-2 run pre-reg finalization
(memo §5 ladder at the measured pace, §4 option B — veto window
passed). GPU legs launch on the finalized pre-reg per the
delegation. `queue.json` canonical.*

*Updated 2026-08-13 13:45–14:2xZ (real `date -u` at stamp: 14:18) —
work session: **token-GRPO instrument item 3 CLOSED on the molmoact2
surface — the RL rollout draw (masked-softmax sampling + TokenRow
capture on `predict_action_discrete`) and the replay collator landed
oracle-gated (`a268046`, check.py 849 green); the loop harness
(item 4) + run pre-reg finalization are all that stand before the
phase-2 launch.***

**Status**: no live runs — GPU idle-by-design until the phase-2
pre-reg lands (babysit registry empty, reason declared). Queue
validate green (depth 2, 14 open); `molmoact2-ar-head-port` CLOSED
(arm B read banked; the (b2) HF-parity remnant unqueued, low
priority — the behavioral gate passed both arms).

**Steering**: owner 14:09Z — explain the grammar-masked decode ("do
we do constrained decoding?") + why seed 73 flipped ("malformed
actions zero-filled?") → answered in-channel 14:14Z
(id 1537464486334832700) with the measured facts: yes, constrained
decoding over the action block under the symbol-budget mask,
identical to the reference stream wherever greedy was already legal;
seed 73 was NOT zero-filled in arm B (impossible by construction) —
the arms' distance series are bit-identical through tick ~473 then
diverge (A succeeds at 622, B ends 10.2 cm; seed 1 is the mirror
image), 47/100 seeds diverged, and at 1/100 competence the paired
delta (+0.728 cm CI95 excl. 0), not the success count, is the
registered read. Per-seed fallback attribution was not banked in
arm A — the item-3 instrument records per-predict streams, so
phase-2 rows will carry it. No reply as of 14:2xZ; the 11:07/11:18Z
delegation stands.

**Done**: token-grpo-phase2-instrument item 3 (`a268046`, retargeted
per the 10:02Z steering): `predict_action_discrete` gains
grammar-masked SAMPLING (Gumbel-max off `stable_sample_rng` keys;
sampling requires the mask — unconstrained sampling would sample the
6.8% fallback class) + per-step `ActionCaptureStep` capture, so
`token_rows_from_capture` + `TrainingRowWriter` work unchanged off
this surface; driver wiring (`--molmoact2-temperature`,
`--emit-training-rows` on the discrete path storing SHIM-APPLIED
model-unit state, `--draws` with temperature);
`bijou/molmoact2/replay.py` (row loader, bins-only grammar-mask
recompute + bit-equality guard, one-shot teacher-forced
`replay_logprobs` WITH graph, `molmoact2_grpo_loss` into the
decoder-generic surrogate). 7 CPU oracles — headline: replayed
chosen logprobs reproduce the rollout's records within the
registered 1e-5 bound, greedy AND sampled (fixture note: the tiny
trunk's real lm_head stopped below the `<action_i>` block; the
replay oracles build it widened, `build_predictor(vocab_size=156032)`).
Queue ×2 (port closed, item 3 folded).

**Next**: `queue_cli.py next` → instrument item 4 (loop harness:
rollout wave → score → z-filter → step → periodic eval + babysit
heartbeat + registry entry) then the phase-2 run pre-reg
finalization (memo §5 ladder at the measured pace, §4 option B
recommended — veto window passed unanswered). GPU legs launch on the
finalized pre-reg per the delegation. `queue.json` canonical.*

*Updated 2026-08-13 13:40–13:4xZ (real `date -u` at stamp: 13:42) —
tick, babysit: **quiet tick after the arm-B close-out — no steering,
no live runs; one repair: the 13:38Z close-out committed the arm-B
results into the pre-reg post but never pushed the Space — blog
rebuilt + pushed this tick.***

**Status**: no live runs — babysit exit 0, 0 registered (arm B
pruned at the 13:38Z close-out); nvidia-smi 0%/0 MiB. Queue validate
green (depth 3, 15 open). Standing result from the prior session:
**arm B COMPLETE 13:31Z — grammar-masked decode is a registered
improvement** (paired B−A progress_final +0.728 cm, CI95 [+0.147,
+1.325] excludes zero; knock-aways 27→13; fallbacks 0/2,996 by
construction; successes 1/100 each arm, A: seed 73 / B: seed 1) —
masked = default serving mode per the delegation.

**Steering**: none new — read empty 13:40Z; history-5 shows no
reactions yet on the 12:29Z arm-A results or 13:33Z arm-B paired-read
posts. Delegation (11:07/11:18Z: decide + keep GPU busy, no
confirmation waits) stands.

**Done**: liveness/queue/GPU verified; blog built + Space pushed so
the prereg page now serves the arm-B paired read (was committed
`d69c470` without a push); now.md brought current (arm B complete,
not live), oldest body entry + aged footer notes rolled to the
archive.

**Next**: chained work session (`run_work_next` armed 13:39Z) →
critical path: instrument items 3–4 on the molmoact2 surface
(sampling mode + TokenRow capture on `predict_action_discrete`,
replay collator, loop harness) + phase-2 run pre-reg finalization
(ladder re-priced at the measured ~0.4 s/chunk). GPU is
idle-by-design until that pre-reg lands. `queue.json` canonical.*

## Utilization footer

Session 2026-08-13 14:22–14:3xZ (tick, babysit; 0 new GPU-h — GPU
idle-by-design, CPU critical path queued): quiet tick — no owner
messages/reactions (14:23Z; the 14:09Z exchange closed at ~10 min
silence after our 14:14Z answer), babysit exit 0 with 0 registered
runs, nvidia-smi 0%/0 MiB, queue green (depth 2, 14 open).
`run_work_next` armed 14:23Z for instrument item 4 (loop harness) +
phase-2 pre-reg finalization — the pre-reg is what returns the GPU
to work.

Session 2026-08-13 13:45–14:2xZ (work; 0 new GPU-h — CPU instrument
critical path, exploit): token-GRPO instrument item 3 CLOSED
retargeted to the molmoact2 surface (`a268046`): masked-softmax
sampling + TokenRow capture on `predict_action_discrete`, driver
row-emission wiring, replay collator + GRPO glue
(`bijou/molmoact2/replay.py`), 7 CPU oracles (headline: replay
reproduces the rollout's logprobs ≤ 1e-5), check.py 849 green. Owner
14:09Z grammar-mask/seed-73 question answered in-channel with the
measured divergence facts (reply id 1537464486334832700). Queue:
ar-head-port CLOSED, item 4 (loop harness) + phase-2 pre-reg
finalization = critical path.

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
