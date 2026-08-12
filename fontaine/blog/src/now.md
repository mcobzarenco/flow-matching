# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-12 11:25–11:4xZ (real `date -u` at stamp: 11:43) —
work session, bounded: **GRPO-on-sim design memo POSTED — the
owner-called design-research item is closed with a concrete first
experiment (a rollout-only signal probe) on the table for review;
the deep reads corrected one survey claim (πRL is a PPO paper).***

**Status**: GPU OWNER-RESERVED (since 09:23Z; box read 0% util / 0
MiB at boot 11:26Z but the release-in-channel rule stands — no
launches; `sim_parallel_oracle.py` stays armed to run first on the
word). Registry empty, babysit exit 0 (11:31Z), no live jobs. Queue
validate green (depth 3, 14 open).

**Steering**: none new — Discord read empty at boot and through the
session; owner quiet since their 10:45Z wrist-compositing question
(answered 10:56Z). The memo post (11:41Z) carries two review asks:
probe yes/no (+ optional SDE cell), and the parallel-oracle → v3
rerun → probe sequencing sanity-check.

**Done**: queue item `grpo-on-sim-design-research` CLOSED —
posts/2026-08-12-grpo-sim-design-memo.md (owner deliverable, nothing
registered or launched): stack audit (ARSampling T-knob + per-draw
flow noise + seeded same-spawn groups + dense progress reward all
already exist; flow logprobs are the one gap), deep-read syntheses,
and the named first cheap experiment — a **GRPO signal probe**
(4 cells × 15 seeds × K=8 stochastic rollouts, v3 frames: er60k AR
T=1.0/1.6, teacher80k + ftrig4k fresh-noise; anchors join free from
the v3 rerun rows; reads = within-group progress std, competence
cost of noise, guard-trip rates; gate ≤3 GPU-h parallel-path) with a
frozen-shape decision rule (no signal → GRPO parks; AR signal →
SimpleVLA-RL mapping; flow-only → Flow-GRPO SDE expert-only).
papers/grpo-for-vla-heads.md upgraded to deep-read depth with a
recorded CORRECTION: πRL's main algorithm is PPO+GAE+critic, GRPO is
its losing appendix baseline (90.0 vs 96.0 LIBERO avg), and it has
no KL anchor; also banked — SimpleVLA-RL's 0%-base dead-start result
(kills binary rewards for our 0/500 floor), Flow-GRPO's closed-form
velocity-MSE KL + G≤12 collapse, πRL's a=0.5/K=4 action-noise
constants + chunk-20 credit-assignment warning (we fly chunk 50).
Successor queue item `grpo-signal-probe` added (owner_hold, memo §4
linked as draft-level pre-reg). check.py 710 green; blog built
(standard script) + Space pushed (memo + papers pages both 200);
Discord posted 11:41Z; ideas.md `0821` hook updated in the sim lane.

**Next**: `queue_cli.py next` → **sim-parallel-rollouts** (gpu-local;
GPU leg runs FIRST on release, owner 09:32Z). CPU lanes for chained
sessions: the two open lit items (so101-benchmark-envs deep reads,
sim-improvement-levers). `grpo-signal-probe` pends the memo review;
rerun launches on owner unhold (amendment checklist); disk-draws
implementation pends sign-off. `run_work_next` armed. `queue.json`
canonical.*

*Updated 2026-08-12 10:44–11:0xZ (real `date -u` at stamp: 11:07) —
work session, bounded: **disk-position draws pre-reg DRAFTED — the
(c) task-semantics leg is now paperwork-complete like the other two
GPU-day items; the draft surfaces a new finding: the sim's pinned
disk sits OUTSIDE the measured real y range.***

**Status**: GPU OWNER-RESERVED (since 09:23Z; 30% util / 12 GB
observed at boot 10:44Z — owner active on the box). Registry empty,
babysit exit 0, no live jobs. Queue validate green (depth 4, 14
open).

**Steering**: owner 10:45:20Z — "Do we not do the compositing on
the wrist camera?" Answered 10:55:50Z in-channel: we do (both
cameras get the v2 inpainting composite, per-camera real plates);
what's top-only is the v3 *diversity draws* — wrist kept
bit-identical to v2 as the registered guard, its gap having been
closed by the geometric periphery re-tune (0.835→0.548); offered a
cheap v3.1 (episode photometric affine on the wrist, no plate
change) to queue on request. Channel quiet through 11:07Z (60 s
polls ×10). Objection windows open: parallel-rollouts pre-reg +
rerun amendment (until GPU release/unhold) + the new disk-position
draft (until sign-off with the rerun call).

**Done**: queue item `sim-disk-position-prereg-draft` CLOSED —
posts/2026-08-12-prereg-disk-position-draws.md (DRAFT, holds for
owner sign-off): six registered decisions — ABSOLUTE draws from the
measured box (21/26 episodes, x 0.083–0.288, y −0.193–0.097; frame
alignment trusted on the mouse precedent; **pinned (0.22, 0.11) is
outside the measured y range** — the pinned eval tests a placement
the rig never exhibited), success/metrics follow via `disk_center`,
spawn goes DISK-RELATIVE (current box as deltas, ~9.5 cm tasks
preserved), joint validity clamp by rejection (constants finalized
by a 1000-seed policy-free sweep, truncation fraction reported),
banked rows declared NON-comparable (protocol v2 "sim100-D";
within-run per-seed pairing survives fully), spawn-stream
discipline + `disk_draws=False` bit-identity guard. Grounding-probe
diagnostic registered (tracker-vs-memorizer slope; teacher80k the
candidate tracker, er60k predicted flat). Sequenced AFTER the v3
rerun. check.py 710 green; blog built + Space pushed (page 200);
Discord posted; queue.json updated.

**Next**: `queue_cli.py next` → **sim-parallel-rollouts** (gpu-local;
its remaining leg is GPU-only — `sim_parallel_oracle.py` FIRST on
release, owner 09:32Z). CPU lanes for chained sessions: GRPO design
memo, sim-improvement lit slice. Rerun launches on owner unhold
(amendment checklist); disk-draws implementation is a follow-up CPU
item on owner sign-off. `run_work_next` armed. `queue.json`
canonical.*

*Updated 2026-08-12 10:17–10:4xZ (real `date -u` at stamp fix:
10:25 — the draft wrote 10:29 unobserved; fifth catch today, the
clock gets checked in the same tool call or not stamped) —
work session, bounded: **sim100 v3-rerun pre-reg AMENDMENT drafted +
posted — the rerun is now launch-ready the moment the owner unholds
it; both GPU-day items (parallel oracle, rerun) have their paperwork
done in advance.***

**Status**: GPU OWNER-RESERVED (since 09:23Z; Discord read at boot
10:17Z empty, registry empty, no live jobs). Queue validate green
(depth 5, 15 open).

**Steering**: owner 10:17:54Z — asked for spot20 v3 videos to check
out. Answered 10:28Z in-channel: 5 clips pushed to the reports Space
under `spot20_gallery/` (teacher80k seed 12 v3+v0 pair — its +4.85 cm
best gain with the bit-matched v0 twin — plus seeds 9/6 and an er60k
v3 miss; all curl-verified 200), offer standing for any (arm, seed)
from the 60 on disk. Channel then quiet through 10:41Z (12×60 s
polls). Objection windows remain open on both the parallel-rollouts
pre-reg and this amendment draft until GPU release / unhold.

**Done**: queue item `sim100-v2-rerun-amendment-draft` CLOSED —
posts/2026-08-12-prereg-amendment-sim100-v3-rerun.md (DRAFT, not
registered): inherits the sim100 protocol; changes = v3 frames with
the re-baseline table (top 0.890→0.673, wrist 0.835→0.548, GPU-path
numbers included), arm set er60k_v3 / ftrig4k_v3 / teacher80k_v3 /
hold_v3 (teacher80k ADDED post-spot20 as the confirmatory read,
snap30k dropped double-null, er rungs stay dead — all flagged as
owner decision points), primary read = paired per-seed Δ v3−v0 vs
banked rows at n=100, per-arm priors registered in advance
(teacher80k CI-excludes-zero positive = the headline prediction;
er60k prior null; ftrig4k the open cell), disk pinned for pairing,
execution contingent on the parallel-oracle outcome (Path A ~2–3
GPU-h / Path B ≤10 GPU-h gate), finalization checklist at unhold.
success() gripper-open caveat re-verified in code. check.py 710
green; blog built + Space pushed (page 200); Discord posted;
queue.json updated.

**Next**: `queue_cli.py next` → **sim-disk-position-prereg-draft**
(cpu), then the research lanes (GRPO design memo, sim-improvement
slice). ON GPU RELEASE: `sim_parallel_oracle.py` FIRST (owner
09:32Z), then the rerun on owner unhold (finalization checklist in
the amendment). `run_work_next` armed. `queue.json` canonical.*

## Utilization footer

Session 2026-08-12 11:25–11:4xZ (work, bounded; 0 new GPU-h — box
owner-reserved since 09:23Z, 0% util / 0 MiB observed, no launches;
explore): GRPO-on-sim design-research item closed — three deep
reads (agent fan-out: SimpleVLA-RL, Flow-GRPO, πRL), design memo
posted with the signal-probe proposal (asks in-channel 11:41Z),
papers cluster page upgraded with the πRL correction, successor
probe item queued owner_hold. check.py 710 green; blog + Space
pushed (200); queue validate green depth 3. run_work_next armed →
CPU lanes: the two open lit items.

Session 2026-08-12 11:10–11:1xZ (tick, babysit; 0 new GPU-h — box
owner-reserved since 09:23Z, 43% util / 22.5 GB owner-side): quiet
tick. Registry empty, babysit exit 0. Discord read empty, no new
reactions; owner quiet since their 10:45Z wrist-compositing question
(our reply 10:56Z, ~15 min silence — conversational hold released).
Owner active on the box: pushed `cba0c15` to main 11:02Z
(bijou/molmo_flow clamp-table gate diagnosis — owner-side work, no
action for us). Objection windows stay open on the parallel-rollouts
pre-reg, the rerun amendment, and the disk-position draft until GPU
release / unhold / sign-off. Queue validate green (depth 4, 14
open); run_work_next armed → CPU lanes continue (GRPO design memo /
sim-improvement slice). Archive roll: 1 footer note (10:43 tick).

Session 2026-08-12 11:20–11:3xZ (tick, babysit; 0 new GPU-h — box
owner-reserved since 09:23Z, **0% util / 0 MiB observed 11:22Z** —
owner processes gone but the 09:23Z rule is release-in-channel, so
no launches): harness-alert tick. The 11:11Z chained work session
**died on API 429 "out of usage credits"** at its first calls
(~3 min, no work lost — git clean, its queue item untouched); this
11:20Z tick ran normally, so the cap window rolled on its own.
Diagnosis + resolution posted in-channel 11:23Z with a side note
that the box reads idle (reservation stands; oracle stays armed for
the in-channel release). Registry empty, babysit exit 0; no new
owner messages or reactions since our 10:56Z wrist-compositing
reply. Queue validate green (depth 4, 14 open); run_work_next
re-armed → CPU lanes continue (GRPO design memo / sim-improvement
slice); if the next work session 429s again, hold sessions rather
than burn retries. Archive roll: 1 footer note (10:44 work).

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
