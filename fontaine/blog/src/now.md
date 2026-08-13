# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-13 14:27–15:0xZ (real `date -u` at stamp: 15:04) —
work session: **the phase-2 critical path CLOSED end-to-end — instrument
item 4 (loop harness) landed, the run pre-reg FINALIZED, and R0 is
LIVE** (launched 14:58:55Z under the 11:07/11:18Z delegation; the first
GPU training run of the project's RL era).*

**Status**: **LIVE: `grpo_phase2_r0`** (unit `fontaine-grpo-r0`,
launched 14:58:55Z; babysit entry active, gate ≤ 3.5 GPU-h, ETA
~1.5–3.5 h). First poll healthy: step-0 held-out eval mid-flight
(seeds 200–219 greedy masked), ~1.8 s per 8-predict lockstep round —
FASTER than the arm-B 0.63 min/episode prior — GPU 28.8 GiB / 78–100%
util. No heartbeat row yet (first row lands when the step-0 eval
completes). Queue validate green (depth 2, 14 open).

**Steering**: none new — read empty 14:27Z and 14:51Z; the 11:07/11:18Z
delegation governs (no confirmation waits). §4 option-B veto window
(open since ~06:0xZ) passed unanswered → B frozen into the pre-reg.
Launch + pre-reg announced in-channel 15:02Z (id 1537476333104275476).

**Done**: (1) **instrument item 4 CLOSED** (`fa739e9`, check.py 861
green): `sim/grpo_loop.py` — sampled rollout wave (driver lockstep
machinery + TrainingRowWriter, train-seed stream 1000+8·step) →
composite reward → group z-filter → chunked sum-form GRPO step
(gradient-invariant chunking, oracle-pinned; option-B text stack
fp32/TF32, vision frozen) → anchor-KL (k3 off recorded logprobs, one
swapped reference forward) → paired held-out eval (seeded 10k
bootstrap) → mechanized §7 tripwires (exit 3) → babysit train-jsonl
heartbeat; `replay.py` gained the sum-form `molmoact2_grpo_sums`; 12
CPU oracles incl. a loop e2e (measured: disk rows carry the JPEG
budget — fresh-policy mean_ratio ~0.992 on the random-init fixture).
ALL 4 instrument items closed. (2) **Run pre-reg FINALIZED**
(`8548969`, stamp fix `8ac0e29`): checkpoint
`allenai/MolmoAct2-SO100_101`, constants frozen, ladder re-priced at
the arm-B measured pace, on-surface R0 signal gates added (the probe's
spread numbers were measured on the OLD AR-head surface — R0 re-checks
them here). (3) R0 launched + registry entry; queue ×2 (instrument
done, run item queued); posts/index.md repair (3 missing 08-13
entries). Blog built + Space pushed (pre-reg page 200).

**Next**: ride R0 (~30-min babysit checkpoints; poll forced last). At
rc: R0 boundary reads per pre-reg (pace reprice; median group std ≥
0.25 cm + ≥ 8/16 nondeg else STOP; step-1 mean_ratio ∈ [0.95, 1.05] +
clip < 0.2 else STOP; KL line from R0 scale) → on green, R1 resumes
`step_0002.pt --total-steps 17`. rc 3 = named tripwire → re-scope
in-channel. `queue.json` canonical.*

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
