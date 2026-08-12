# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-12 15:31–16:0xZ (real `date -u` at stamp: 16:02) —
work session, bounded: **the owner-prio flipped-physics rerun is
CLOSED end-to-end in ~30 min — and the answer is a clean pre-registered
null: 18/20 episodes are bit-identical across the two bracket
geometries; the MolmoAct2 knock-aways are jaw contact, not the
bracket.***

**Status**: no live jobs — `ftrig_eval20_flip_parallel` COMPLETE
(launched 15:42:03Z, all arms rc=0 by 15:59:46Z, ridden in-session;
first-poll 100% util / 20.9 GB; ~0.27/0.5 GPU-h). GPU idle again,
pending the owner's v3-rerun unhold (15:13Z ask, still open).
Registry pruned to a completion note. Queue validate green (depth 3,
13 open).

**Steering**: owner prio 15:27:11Z (re-run the 20 episodes on flipped
physics, many parallel workers) → **executed and closed this
session**; results in-channel 16:02Z. No new messages through 16:02Z
polls. Open asks: v3-rerun unhold + arm set (15:13Z), GRPO probe memo
review, disk-draws sign-off.

**Done** (commit `c68ea06` + close-out): queue item
`ftrig-eval20-flipped-parallel` CLOSED — pre-reg posted before
launch, both arms parallel workers=8 (postflip + `--no-mount-flip`
preflip, same 20 seeds, euler-10/v3, videos), paired within the
parallel path only per the failed-oracle rule. Instrument landed:
`SO101Sim(flip_camera_mount=)` toggle (CPU probe: all 3 mount geoms
mirror back, settled bracket 40.2 mm = the probe-measured pre-flip
value), parallel driver gains the merged-stats fallback +
`--no-mount-flip` + `mount_flip` in rows JSON; harness oracle 5/5,
check.py 773 green. RESULTS: paired flip effect ~null — 18/20 seeds
bit-identical (mount geoms enter dynamics only via contact; the
policy fails before reaching bracket-blocked poses), 2/20 improved
post-flip (s15 +0.81 cm, s5 +0.60 cm), 0 worsened; knock-aways 6/6
unchanged = jaw-side; the sequential-run diagnosis stands on fixed
physics. Banked incidentals: identical-config parallel runs
bit-identical at workers=8 (a launcher-flag slip became a
reproducibility datum — lockstep scheduling is exactly replayable,
GRPO-probe-relevant), and parallel-vs-sequential outcome drift
quantified (mean −0.37 cm, 11/20 seeds >0.1 cm, max 6.0 —
the 14:37Z oracle FAIL confirmed at outcome level). Rows + 40 videos
on fontaine-reports `/ftrig_eval20_flip_parallel/` (curl 200);
results appended to the pre-reg page; Discord 16:02Z.

**Next**: `queue_cli.py next` → CPU lanes: `lit-sim-improvement-levers`,
`sim-wrist-compositing`. GPU: idle until the owner answers the
v3-rerun unhold ask (15:13Z — the rerun is the re-baseline carrier
for every banked sim row post-flip); `grpo-signal-probe` owner_hold.
`queue.json` canonical.*

*Updated 2026-08-12 13:10–15:0xZ (real `date -u` at stamp: 15:02) —
work session, bounded, mid-session owner release of the GPU: **both
GPU legs ridden — parallel oracle FAIL (sequential stays registered),
ftrig MolmoAct2 first look 0/20-but-reaches; the owner's video-watching
caught a 180°-flipped wrist bracket that the probes confirm explains
~62% of the servo-replay gap. Plus: replay control-loss validator
landed (sysid passes), SDE sampler + oracles landed, branch rebased
onto latest main.***

**Status**: GPU RELEASED (owner 14:17Z "GPU is all yours"; confirmed
14:21Z). No live jobs — both GPU legs completed in-session (~0.4
GPU-h total): `sim_parallel_oracle` FAIL banked 14:37Z,
`molmoact2_ftrig_eval20` rows + 20 videos banked ~14:50Z (reports
Space, curl-verified). Registry entries pruned to completion notes.
Queue validate green (depth 3, 14 open).

**Steering** (busy day — 6 owner messages, all dispositioned):
13:16Z SDE ride-along GO + rebase ask + ftrig eval called top-prio →
all three done. 13:36Z sequencing confirmed (oracle first) + "20
episodes, rough numbers and videos" → done. 13:54/13:59Z push/rebase
nudges → branch now main+3, ahead-only (`7b793e5`→`88223b1`).
14:17/14:21Z GPU release → both legs ridden. 14:27Z "eval should
composite both cameras" → `sim-wrist-compositing` queued
(probe-gated per the SIMPLER partial-matching caution). 14:45Z
bracket-hits-table question → probe-confirmed 180° flip (numbers
below), `sim-wrist-bracket-flip` queued owner_hold → owner GO 15:01Z
("Let's do asap") → **executed + verified same session** (see Done).
No open asks.

**Done** (commits `5c64046`, `8d3227a`, `88223b1` + close-out):
(1) `sim-sysid-replay-control-loss` CLOSED — SIMPLER's offline
validator built (`sim/replay_control_loss.py` + oracles): pinned fit
L 0.083 vs floor 0.070, under SIMPLER's best anchor 0.131; finding:
joint-MAE wins don't carry to EE space (elbow lever arm 4.6 mm/°);
results post + dark chart. (2) Flow-GRPO SDE sampler
`sample_actions_sde` + 4 oracles (bit-identity at a=0, exact
logprobs) — probe cell 5 launch-ready. (3) `--draws`/`--ar-temperature`
on the sequential driver (draw-keyed identity triples, draw-0
bit-identity oracle; parallel driver deliberately untouched).
(4) Branch REBASED onto latest main per owner (merge commit dropped,
+ I001 fix main itself needed). (5) `sim-parallel-rollouts` CLOSED:
oracle FAIL at workers=2 (3/6 seeds diverge macroscopically via
batched bf16 decode; env determinism held; 1.73× throughput datum);
frozen rule applied. (6) `molmoact2-ftrig-sim-eval-20` CLOSED: 0/20,
mean −0.84 cm, 7/20 real approaches, 4 knock-aways — moves with
intent where er60k froze; videos + rows on fontaine-reports; one
integration fix (merged-stats fallback for converted checkpoints).
(7) Bracket probes: 31.9% of real-pose frames put the sim bracket
below the table; ep-21 replay grinds 22% of ticks. (8) **Bracket
flip EXECUTED on owner GO** (`_flip_camera_mount`, 180° about
mount-local x, camera view bit-unchanged): sweep 31.9%→1.4%
(bounding-conservative, center never below), strikes 0/100, oracles
7/7, replay L 0.0831→**0.0751** vs floor 0.0701 (gap −62%), arm MAE
1.88°→1.50°. Physics re-baseline boundary declared (banked rows =
pre-flip; folds into the v3 rerun). Consolidated results post
(gpu-release-results). check.py 770 green throughout.

**Next**: `queue_cli.py next` → CPU lanes: `lit-sim-improvement-levers`,
`sim-wrist-compositing`. The v3-rerun re-baseline now also carries
the bracket flip (one re-baseline, not two). `grpo-signal-probe` unblocked
for prep: finalized pre-reg is the remaining CPU step (sampler +
flags landed); GPU sequence now at owner discretion post-eval.
`run_work_next` armed. `queue.json` canonical.*

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

## Utilization footer

Session 2026-08-12 15:31–16:0xZ (work, bounded; **+~0.27 GPU-h** —
ftrig_eval20_flip_parallel, 3 arms × 5.4 min at workers=8, ridden
end-to-end; exploit, owner prio): flipped-physics rerun closed same
session as the ask (~25 min ask→numbers). Paired null banked (18/20
bit-identical, bracket innocent of the knock-aways) + two incidentals
(parallel bit-reproducibility; oracle-FAIL drift quantified at
outcomes). 1 owner prio dispositioned; v3-rerun unhold ask still open.

Session 2026-08-12 15:11–15:3xZ (tick, babysit; 0 new GPU-h — local
GPU idle since ~14:50Z, owner-released 14:17Z): post-flip tick.
Babysit exit 1 = the retained-entry footgun (the ftrig-eval registry
entry outlived its ~14:50Z completion) — pruned to a completion note,
no real failure; no live jobs. Discord read empty; history check: no
reactions or messages since our 15:10Z flip-done post. **OPEN ASK
posted 15:13Z**: v3-rerun unhold + arm-set proposal (er60k + ftrig4k
+ teacher80k + hold, v3 frames, flipped-mount physics `d5cf9fd`, same
100 seeds/metrics/gates, sequential driver ~6–9 h wall) — both
registered gates GO, amendment drafted, the GRPO probe's anchor rows
join free from its rows, and post-flip the rerun is the re-baseline
carrier for every banked sim row. Channel change-watch fired 15:27Z —
**owner steering 15:27:11Z: prio re-run of the 20 episodes on flipped
camera physics with many parallel workers.** Acknowledged + designed
in-channel 15:28Z: both arms parallel workers=8 (pre-flip + post-flip,
same 20 seeds, paired per-seed = the sanctioned within-parallel-path
read; parallel rows stay rough/exploratory per the failed oracle),
queued as `ftrig-eval20-flipped-parallel`, FIRST GPU claim — launch
rides the chained work session (tick cap). The v3-rerun unhold ask
(15:13Z) stays open underneath. Queue validate green (depth 4, 14
open); run_work_next armed → the prio GPU item, then CPU lanes (grpo
pre-reg finalize, wrist-compositing design, sim-levers lit). Archive
roll: 2 footer notes (12:17 tick, 11:25 work).

Session 2026-08-12 13:10–15:1xZ (work, bounded; **+~0.4 GPU-h** —
oracle ~0.25 + ftrig eval ~0.15, both ridden in-session after the
14:17Z owner release; exploit + instrument): replay control-loss
validator closed (sysid passes, elbow-lever finding), SDE sampler +
draws instrument landed, branch rebased onto main, parallel oracle
FAIL banked (sequential stays registered), ftrig molmoact2 0/20
first look with videos, bracket-flip finding (owner-spotted,
probe-sized at ~62% of the servo-replay gap). 6 owner messages
dispositioned; 1 ask open (re-baseline OK).

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
