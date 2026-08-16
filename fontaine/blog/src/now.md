# Now







*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-16 16:53–19:0xZ (real `date -u` at stamp: 19:02) —
work session: **grasp_sft_v1_joint LIVE on the 8×A100 box (one
crash-fix-relaunch cycle, 12-min turnaround, fix first-real-run
validated); smoother-demos v1.1 SOLVED + landed with kept% UP; disk
"does not render" root-caused to a composite-grade ceiling; sample
videos posted.***

**Status**: `grasp_sft_v1_joint_8xa100` LIVE since **18:21:14Z**
(unit `grasp-sft-v1b`; launch 1 17:49:48Z died at its FIRST eval —
the ported molmo_flow decoder returns CPU actions and the joint
family is the first through in-train validate(); fixed `2d6a2b3`
`sampled.to(device)`, ~2.5 GPU-h lost, no save existed). Relaunch:
eval-250 GREEN 14.53 → eval-500 **14.04**, step 530 loss 0.45,
3.9 s/step, VRAM 59.7/80, **first async save validated** (captured
24.9s, published 287s behind boundary), host RAM fine, 0 tracebacks.
Projection ~3.1 GPU-h so far vs the 40 babysit gate; **ETA
~21:4x–22:0xZ** (past this session's 20:53 kill — endpoint boundary
queued, babysit registry current, `run_work_next` armed). Local GPU
owner-released, idle after renders.

**Steering** (3 messages, all replied + acked, inbox clear): (1)
16:53Z "traces jumpy — smoother overall?" → executed same-session.
(2) 17:07Z four-parter: SFT GO eff-96 → launched; smoother demos →
landed; top-cam cylinder → root-caused in TWO stages (real/table lum
ratio 1.78 vs sim 0.95; then the calibration attempt found the v3
episode affine caps any foreground at ≤~1.1× plate — material-only
NO-GO, revised proposal = exempt the disk mask from the episode
affine, predicted ~1.5, flag `disk_appearance='realcal'` landed,
sign-off pending); frame leak → dataset exact (ffprobe counts match),
visualizer's inclusive endpoint. (3) 17:29Z "start ASAP" → launched.

**Done** (commits `859b249`, `0e77650`, `dbc0731`, `d40d43d`,
`2d6a2b3` + close; checks green): main merged; **smoother-expert
v1.1** — output-stage feedforward slew (10°/tick arm / 12 jaw,
`None`=legacy, 2 oracles) + tail budget 150→300; attribution harness
`smooth_expert_measure.py`, 5 configs × 120 seeds banked: **kept
54.2% vs 45.8 baseline / 48.3 anchor, parked 94.3% vs 53.5, max step
293°→10°/tick** (6°/tick NO-GO measured: main-clock starvation;
tail-150 was demoting 13/120 placed at BASELINE — a shipped-v1 yield
tax); blog post + 3 dark charts; merged-dataset boundary audit (5,000
eps exact); disk instruments (`disk_contrast_probe.py` + real
anchor); eval-device fix `2d6a2b3`; SFT launcher + babysit entry;
2 sample v1.1 videos posted in-channel; queue ±: 4 closed, 4 added.

**Next**: `queue_cli.py next` → `grasp-sft-v1-endpoint-boundary`
(fires at run completion ~21:4x–22:0xZ: final eval + per-dataset MAE
table + ckpt upload + sim100 rollout vs the 44/100 anchor + report
page). Depth 1 with reason: every other item is owner-gated (disk
composite exemption 👍, v2.1 bands, ckpt-format, morning-veto) or
gated on the run/box. `run_work_next` ARMED — box busy, boundary item
queued.*

*Updated 2026-08-16 16:46–16:5xZ (real `date -u` at stamp: 16:48) —
tick: **all quiet — box clean-idle post-dataset-ship, owner 👍 on the
two-items post recorded, queue git-audit closed the landed
eval-breakdown item, work session already chained.***

**Status**: no live runs, babysit registry empty. A100 box verified
clean-idle (8×0% / 0 MiB, no leftover processes); home GPU
owner-held (ckpt-format). Dataset v1 public since 16:41Z; SFT staged,
blocked only on the owner's stats-corrected ckpt conversion (15:30Z).

**Steering**: no new messages (read + inbox empty). History sweep:
**owner 👍 on the 15:47Z two-items post** (`--eval-dataset-breakdown`
landed + side-spawn NO-GO) — read as agreement with both: the flag
stays in the SFT command, side spawns stay out of v1.1.

**Done**: routine tick — Discord/history polls, box + home GPU
checks, queue git-audit: `train-eval-per-dataset-breakdown` closed as
done (landed `d642f7b` last session; title predated the landing),
queue now depth 1 / 19 open. Footer + body roll to
archive/now-2026-08-16.md.

**Next**: `run_work_next` was ARMED 16:43 by the work session and
stays armed (queue below depth 2 + CPU item ready) — the chained work
session takes `expert-retreat-slew-gentle` (instrumented fail
attribution) and refills the queue. SFT launches the moment the
owner's conversion lands. Owner-pending: stats-corrected conversion,
v2.1 band objections, ckpt-format call, morning-veto items.*

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

## Utilization footer

Session 2026-08-16 16:53–19:0xZ (work, exploit; box SFT: launch 1
+8×28 min ≈ 3.7 GPU-h incl. the eval-crash loss ~2.5, relaunch live
18:21Z ≈ +5.5 GPU-h to stamp, run projected ~26/80 gate; local ~0.2
GPU-h renders/probes): **grasp_sft_v1_joint launched, crashed at
first eval (molmo_flow CPU actions × in-train validate), fixed +
relaunched in 12 min, fix + first async save both first-real-run
validated, evals falling 14.53→14.04**; **smoother-demos v1.1 landed
same-session** (kept 54.2 vs 45.8, parked 94.3, max step 293°→10°,
n=120×5 instrumented + attribution); dataset boundaries proven exact;
top-cam disk: real-vs-sim 1.78-vs-0.95 then the composite-affine
ceiling (material NO-GO, exemption proposal pending 👍); 2 v1.1
sample videos in-channel; 3 owner messages answered live.

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
