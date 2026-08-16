# Now







*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-16 14:38–16:5xZ (real `date -u` at stamp: 16:43) —
work session: **v1 DATASET SHIPPED — 5,000/5,000 kept, merged, PUBLIC
on HF with card + visualizer link; SFT staged end-to-end; side-spawn
probe closed as measured NO-GO; 7 owner messages answered live.***

**Status**: no live runs — demo-gen-v1c **COMPLETE 16:32Z**: 5,000/5,000
kept (10,883 attempted = 45.9% vs the 48.3% anchor), 0 failed shards,
2h07m wall ≈ **16.9 of the 80 GPU-h gate**; boundary executed
same-session (merge → 5,000 eps / 1,506,208 frames / 26 GiB, quantile
rewrite + provenance union; dry-run then public upload; card 16:41Z).
Dataset: **https://huggingface.co/datasets/mcobzarenco/fontaine-grasp-demos-v1**.
A100 box now idle awaiting the SFT launch; home GPU owner-held
(ckpt-format). Babysit registry empty.

**Steering** (7 owner messages, all replied + acked, inbox clear): (1)
14:57Z rebase-on-main + released-ckpt training plan → main merged
(`3a38a17`, 968→975 checks), their converted ckpt validated (schema 2,
joint, 20.3 GiB). (2) 15:05Z SFT spec (rig datasets in the mix, image
aug, joint + KI, vision frozen, batch question) → full command proposed
from measured route-C numbers (eff-128 = 16×8, ~57/80 GiB per rank).
(3) 15:10Z random success video → seed 130051 re-rendered locally
(exact 242-tick/2.2 cm match to the shard log), posted. (4) 15:13Z
v2.1-vs-v3.0 joint conventions → all three datasets verified v3.0 raw
degrees, ranges overlap. (5) 15:21Z per-dataset normalization
question + eval work order → **owner was right, my claim corrected**
(molmoact2 normalization is decoder-owned q01/q99 from the ckpt) and
that check surfaced a REAL blocker: the released ckpt's table is a
different joint convention (lift 45→186 vs our −103→+29) — direct SFT
would clamp-distort; owner took the conversion-time fix 15:30Z. Eval
work order EXECUTED: `--eval-dataset-breakdown` landed (`d642f7b`).
(6) 15:22Z retreat-too-wild → queued; first pass measured + reverted
(findings in the queue item). (7) 16:18Z consolidated command → posted
16:39Z with the one blank (their stats-corrected conversion).

**Done** (commits `3a38a17`, `a8973dd`, `d642f7b`, checks 975): main
merge (ckpt schema-v2 stack); **side-spawn probe CLOSED as measured
NO-GO** (prereg §8: side rest 120/120, stock expert 0/120 but
pinch+carry works, righting 0/120 across 6 push variants — the boat
slides 6–7 cm, never rolls; tool facts banked: pad-space floor z≈0.077,
gripperframe site = jaw tip); `reset(boat_start="side")` extension +
oracles; `--eval-dataset-breakdown` (per-dataset MAE lines + counts
table, 4 oracles); **v1 dataset generated + merged + published** with
card; queue audit (spawn-v2-randomization closed as superseded);
babysit entry pruned with clock-checked stamps (one wall-clock slip
caught + corrected in-channel 16:43Z).

**Next**: `queue_cli.py next` → `expert-retreat-slew-gentle` (CPU;
first-pass findings recorded: ramped home leg collapses kept% via the
success still-bar, needs instrumented attribution). Then the SFT
pre-reg once the owner's stats-corrected conversion lands
(owner-pending, 15:30Z). Owner-pending: v2.1 band objections,
ckpt-format conversion call, morning-veto items. `run_work_next`
ARMED — box idle + CPU queue non-empty.*

*Updated 2026-08-16 14:34–14:4xZ (real `date -u` at stamp: 14:37) —
tick: **demo-gen-v1c babysit GREEN — 96/96 shards live, kept ratio
46.6% tracking the 48.3% anchor, rate ramped to 39.6 kept/min, ETA
~16:3xZ.***

**Status**: **demo-gen-v1c LIVE** on 147.224.218.164, babysit exit 0
— 3 driver procs, all 8 GPUs loaded (~9.5 GiB each, 58–97% util),
driver log 396/5000 kept of 850 attempted (**46.6%**, tracking the
measured 48.3%, well above the ~40% regression floor), rate ramped
23→39.6 kept/min since launch, driver ETA 116 min → **DONE ~16:3xZ**;
GPU-h projection 18.4 vs the 80 gate. Home GPU untouched (owner
hold). `origin/main` advanced to `cdac435` (ckpt schema-v2 flip +
`validate_checkpoint` CLI + `--family` importer) — merge stays queued
for the next work session per the 14:24Z post.

**Steering**: none — `read` empty, inbox empty, history sweep shows
no new messages or reactions since the 14:32:53Z launch post.

**Done**: routine tick — babysit + kept-ratio anomaly check straight
from the driver log (no regression, no substitution flood), queue
validate OK depth 2 (20 open), overdue footer roll executed (26
session notes + the 12:03-stamp entry → archive/now-2026-08-16.md).

**Next**: unchanged — **`run_work_next` ARMED** (14:33, by the work
session): the chained work session takes the main merge past
`57c6843` and `side-spawn-feasibility-probe` (CPU) while generation
runs; boundary ~16:3xZ: merge shards → HF upload → dataset card post
(recipe in the queue item + babysit entry). Owner-pending: v2.1 band
objections, ckpt-format conversion call, morning-veto items.*

*Updated 2026-08-16 12:36–14:3xZ (real `date -u` at stamp: 14:29) —
work session: **P1 executed end-to-end — sharded demo-gen stack
built, A100 box provisioned, spawn-v2 A′ FAILED → measured v2.1
amendment, two owner mid-flight changes folded in, 5k generation
LIVE.***

**Status**: **demo-gen-v1c LIVE on 147.224.218.164** since 14:25:12Z —
96 shards × 8 GPUs (37–83% util, ~9.3 GiB each), v2.1 + mix70 tint +
retreat tail, target 5,000 kept from seeds 10000+ (stride 2000).
First poll 14:2xZ: 96/96 shards live, 69 kept/155 attempted (44.5% ≈
the measured 48.3%), rate ramping through 23 kept/min, ETA ~2.5–3.5 h.
Babysit entry `demo_gen_v1` registered (progress-log,
`logs/driver.log`). Home GPU untouched (owner hold, ckpt-format).

**Steering** (5 messages, all replied + acked, inbox clear): (1)
13:38Z local-agent ckpt schema-v2 note (main `57c6843`) → ack'd;
merge at next session top, nothing of mine loads checkpoints now. (2)
13:46Z **retreat-to-rest tail for demos** → implemented: expert
retreats up-and-back then slews HOME; collector records the tail and
re-verifies success after it (knocked boat = miss); 48.3% kept, 86%
end parked, median 272 ticks (n=120). (3) 14:05Z **standalone public
dataset repo** → confirmed `mcobzarenco/fontaine-grasp-demos-v1`,
public at creation. (4+5) single-core question, self-resolved (was
the smoke).

**Done** (commits `05a1199`, `439704f`, `07f6de5`, check.py 952):
sharded demo-gen stack (driver w/ manifest-guarded resume + EGL/CUDA
round-robin; LeRobot shard-merge with **bit-identical oracle** —
parquet columns, decoded video pixels, stats; HF upload + card w/
dry-run); tint knob (rig_gray/wide/mix70); spawn-v2 **finalized**
(frozen 977-cell mask committed) then **A′ FAILED 19.8%/600 seeds**
→ diagnosed shoulder-lift servo *saturation* (hold probe: force frac
1.00, sag 3→20 mm over r 0.20→0.36; the reachability instrument's
torque field was wrong) → **registered v2.1 amendment** (boat r_base
[0.16,0.27], disk [0.18,0.32]; 53.8%/400 measured); found + fixed the
**phantom moved-disk collision** (midphase BVH bakes compiled
geom_pos; boat fell through — v2/v2.1 disable midphase, v1
bit-identical); A100 box provisioned (GL/EGL userspace + fabric
manager 580.178.04 aligned after apt skew, uv env, assets, HF token);
queue class `gpu-a100`; prereg §6/§7/§7.1 addenda; 3 probe reports
banked.

**Next**: `queue_cli.py next` → `side-spawn-feasibility-probe` (CPU,
owner-accepted). Boundary: generation DONE ~17–18Z → merge → upload →
card post (recipe in the queue item + babysit entry). Owner-pending:
v2.1 band objections (flagged in-channel), ckpt-format conversion
call, morning-veto items. Next session: merge main past `57c6843`.
`run_work_next` ARMED — the tick chain babysits the run and the next
work session takes the merge/upload boundary or the side-spawn probe.*

## Utilization footer

Session 2026-08-16 14:38–16:5xZ (work, exploit; box demo-gen rode to
completion ≈ +14.4 GPU-h this session's share of the 16.9 total, home
GPU owner-held): **v1 dataset SHIPPED same-session — 5,000/5,000 kept
45.9%, 0 failed shards, merged 1.5M frames, public on HF with card +
visualizer link, 16.9/80 GPU-h**; side-spawn probe executed to a
measured NO-GO (6 righting variants, 0/120 — boat slides, never
rolls) + prereg §8 report; `--eval-dataset-breakdown` landed with
oracles; main merged; released-ckpt stats-table convention mismatch
FOUND (would clamp-distort SFT) → owner took the conversion fix; 7
owner messages answered live incl. a corrected claim of mine; queue
±: +3 owner items, 3 closed (probe, demo-gen, stale spawn-v2 parent).

Session 2026-08-16 12:36–14:3xZ (work, exploit; ~0.5 GPU-h of
smokes/probes on the A100 box + **demo-gen-v1c live from 14:25Z
accruing 8 GPU-h/h**, home GPU owner-held): **P1 demo-gen executed:
stack built with oracles (952 green), box provisioned from bare
(GL/EGL + fabric-manager version skew diagnosed + fixed), spawn-v2
A′ failed honest (19.8%) → servo-saturation diagnosis → registered
v2.1 amendment (53.8% measured) + phantom-disk collision fix +
owner's retreat-tail folded in mid-flight — 5k generation running.**
5 owner messages answered, inbox clear; queue depth 2,
`run_work_next` ARMED.

Session 2026-08-16 12:09–12:3xZ (tick; home GPU owner-reserved, hold
extended): **live 7-message owner exchange all answered <2 min from
code/measured data, ending in a new 8×A100-80GB machine allocated**
— 80-GiB rec (66.65 GiB measured) taken, demo-gen sharding ordered,
**v1 dataset locked 12:21:03Z (spawn-v2 annulus + upright + 70/30
tint mix, ~5k kept)**, box `147.224.218.164` provisioned + verified
(8×A100 idle, 240 cores) and **allocated 12:26:52Z "machine is all
yours"**; side-spawn needs a righting capability (probe queued);
main fast-forwarded to fontaine `3a3daa6` by the owner; queue depth
3, `run_work_next` ARMED for the P1 execution.

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
