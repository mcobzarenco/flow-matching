# Now







*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-12 09:2x–09:4xZ (real `date -u` at stamp fix:
09:31 — the draft guessed 09:55; my internal clock runs ~25 min
fast today, fourth catch, memory updated) —
work session, owner-steered arc 2: **GPU handed to the owner
(09:23Z, few hours) — no launches until released in-channel;
sessions switch to the owner-called research program: sim
improvement lit, SO-101-adjacent benchmark envs, GRPO-on-sim
design.***

**Status**: GPU OWNER-RESERVED (babysit no_live_runs_reason carries
the rule). Queue validate green (depth 6, 16 open): 3 research
items queued (`lit-so101-benchmark-envs` — first page landed,
`lit-sim-improvement-levers`, `grpo-on-sim-design-research`) ahead
of the amendment/disk-position drafts and `sim-parallel-rollouts`.

**Steering**: owner 09:23Z — GPU reserved + research program set
(supersedes the 08-10 lit pause for these threads). Acked 09:2xZ
with the three-lane plan. Earlier arcs this morning: v3 flip,
GPU compositor, spot20 (all closed, see the 07:2x entry).

**Done**: lit slice `0820` — **papers/so101-sim-ecosystem.md**
(update to the 08-11 census, training-in-sim angle):
lerobot-sim2real's 91.6% real cube-grasp from pure-sim RL on SO-100
= embodiment is not the blocker; named cheap bridge = port their
cube-grasp task+predicate into our sim (~1 day) to sit next to the
only published SO-100 sim2real number; ManiSkill3 throughput vs our
fidelity play (GRPO could train v0 / eval v3); GRPO deep-read
targets banked (SimpleVLA-RL for the AR head, πRL for flow-head
logprobs). ideas.md hook; queue updated. THEN slice `0821` —
**papers/grpo-for-vla-heads.md** (survey-depth mechanism map): the
flow-head logprob obstacle is solved twice in the literature
(Flow-GRPO ODE→SDE with closed-form per-step logprobs; πRL
Flow-Noise exact likelihood); SimpleVLA-RL proves token-GRPO on the
AR head from 1-demo cold start (17.3→91.7 LIBERO); our paired
seeded groups + progress_final are a ready-made reward; design memo
= the queued item's deliverable. Owner re-sequenced (09:32Z):
sim-parallel-rollouts FIRST on GPU release — encoded in queue +
registry. Both pages + ideas hooks committed (real 09:35 at this
edit).

**Next**: chained sessions continue the research program (GRPO
design memo + sim-improvement slice are the open lanes); NO GPU
work until the owner releases the box. `run_work_next` armed.
`queue.json` canonical.*

*Updated 2026-08-12 07:2x–09:1xZ (real `date -u` at stamp fix:
09:07 — the draft carried an unchecked 09:14, and the results
Discord post said 09:16Z at real 09:07; third drift today, now a
saved memory) —
work session, owner-steered arc: **spot20 CLOSED — the teacher SEES
the new sim: teacher80k paired Δ +0.97 cm [CI95 +0.16, +1.81]
toward the disk, the only CI-excludes-zero read (direction flip vs
its v0 −0.73); er60k −0.07 / snap30k +0.06 null. Plus: v2→v3
default flip (owner 07:29Z) and the GPU compositor (owner 08:12Z,
371→94 ms/tick, probe reads preserved).***

**Status**: no live jobs — spot20 units all rc=0 by 09:02Z (~1.3
GPU-h of gate 3), registry pruned, GPU free. Queue validate green
(depth 3, 13 open): **sim100-v2-rerun-amendment-draft** (retargeted
v3) + **sim-disk-position-prereg-draft** + **sim-parallel-rollouts**
(new, owner-approved 08:44Z). **sim100-v1-rerun** stays owner_hold —
both gate legs now argue GO (visuals 0.673/0.548 + behavioral
response confirmed at n=20).

**Steering**: owner active all morning — v3 flip approved 07:29Z
(done, `da96d30`); GPU compositor approved 08:12Z (done, `b99be38`,
oracle ≤2/255 + probe re-read 0.669/0.113/0.544 within noise);
spot-checks called 07:35Z incl. snapflow/teacher (done, results
posted); sim-parallel-rollouts approved 08:44Z (queued). All owner
messages replied in-channel same-session.

**Done**: spot20 end-to-end (pre-reg 07:52Z → 3 arms parallel via
run_detached → paired reads + chart + results post 09:0xZ);
`spot20_reads.py` + `spot20_chart.py`; GPU `_TorchPost` compositor
+ gpu-marked oracle; queue item sim-parallel-rollouts written with
its determinism-oracle requirement; clock-stamp corrections
(pre-reg was in-channel 06:35:39Z not 07:04Z — Discord timestamps
are authoritative; two more drifted stamps caught and owned
in-channel).

**Next**: `queue_cli.py next` → **sim100-v2-rerun-amendment-draft**
(or the owner's rerun unhold — the spot-check argues for it;
sim-parallel-rollouts first would cut the rerun to an afternoon).
`run_work_next` armed. No dated boundaries — `queue.json`
canonical.*

## Utilization footer

Session 2026-08-12 09:41–09:4xZ (tick, babysit; 0 new GPU-h — box
owner-reserved since 09:23Z): quiet tick. Registry empty, babysit
exit 0. Discord read empty; owner 👍 on the 09:27Z three-lane plan
post recorded (plan confirmed, already in motion). Queue validate
green (depth 6, 16 open); run_work_next armed → research program
continues (GRPO design memo / sim-improvement slice /
sim-parallel-rollouts CPU scaffold). Archive roll: 06:26 body entry
+ 4 footer notes (07:2x, 06:26, 06:23, 05:52).

Session 2026-08-12 09:2x–09:4xZ (work, owner-steered lit; 0 new
GPU-h — box owner-reserved 09:23Z): research program started —
slice 0820 page landed (sim ecosystem take 2: lerobot-sim2real
91.6% pure-sim-RL cube-grasp on SO-100, cube-grasp task-port named
as the ~1-day bridge; GRPO targets banked). 3 research items
queued; run_work_next armed.

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
