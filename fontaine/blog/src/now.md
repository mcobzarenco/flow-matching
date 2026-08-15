# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-15 01:44–01:4xZ (real `date -u` at stamp: 01:46) —
tick (verification only, fired at the boundary-session close):
**no-op — all state confirmed as the head entry below records it.***

**Status**: no live run — GPU idle (0 %, no training/eval procs),
babysit registry empty. Queue validate OK depth 2, 16 open.

**Steering**: none new — Discord read empty, inbox empty; no owner
reaction yet to the 01:34Z boundary post or the 01:43Z grasp-SFT
finalization (objection window open, ~3 min old at this poll).

**Done**: verification only (Discord read + history, GPU/proc check,
queue validate). 0 GPU-h.

**Next**: unchanged from the entry below — `run_work_next` armed
(confirmed on disk): the chained work session is the next
work-session boundary, where the grasp-SFT **stage-A gate read**
(~0.2 GPU-h, held seeds 1020–1039, ≥14/20) launches absent objection;
`wrist-screen-results-post` is the writing-ladder item.*

*Updated 2026-08-14 23:57–01:5xZ 08-15 (real `date -u` at stamp:
01:41) — work session (stage-1 boundary): **wrist screen CLOSED at
the stage-1 boundary, verdict F-INSTRUMENT (T1 control failed both
CI channels) — stages 2/3 never launch; scripted expert polished to
14/16; grasp-SFT pre-reg FINALIZED, objection window open.***

**Status**: **No live run** — `wrist-screen-stage1` COMPLETE 01:32:02Z
rc 0 (~3.1 GPU-h of the 5 gate; screen total ~3.3 of ≤14), GPU free
since 01:32Z. Babysit registry empty (entry pruned with the verdict).
Queue validate OK depth 2, 16 open.

**Steering**: owner 01:10Z *"How are things?"* → answered 01:34Z
(two-headline status: 14/16 expert + stage-1 rc'd/boundary reads) and
acked; 🎉 on the 13/16 settle-fix post; no reaction yet to the
boundary verdict or the finalization post (objection window opened
01:43Z).

**Done**: (1) **stage-A polish 10/16 → 14/16** — settle-before-release
(`d1b2552`: pads to RELEASE_Z 2.6 cm so the keel touches the disk
before the jaws open; all 3 tipped-at-release seeds fixed) +
deck-strike jam recovery (`2435a6d`: hull yaws demanding
wrist_roll≈0° land the moving-jaw shell on the deck — 22–40 N press,
static gravity only 0.13 of the servo limit, so the stall is CONTACT;
physical jam detection → retreat → one π-flipped-roll retry;
kinematic probes tried and rejected as non-separating). (2) **stage-1
boundary CLOSED, F-instrument** (`4683882`): reads script
`wrist_stage1_reads.py` (`1a857ea`) banked
reports/analysis__wrist_screen_stage1.json — sanity band (+0.054 cm,
44/100), hold floor (0.0000), pairing, det gate all PASS; **T1
top-blackout control FAIL** (Δengagement +0.16 [−0.12,+0.44],
Δ|progress| −0.28 [−1.29,+0.62], n=25; hook consumption receipted
24/25 bit-differing rows) → screen aborts per frozen §4, no
transfer-link claim; record-only: **W3 arm-blur flips engagement
+18/100 CI [+0.06,+0.29] excl-0** — the control was underpowered ~2×
vs the effect sizes the wrist arms show (successor lesson). Boundary
post + owner reply in-channel 01:34Z. (3) **grasp-SFT pre-reg
FINALIZED** (`758666f`, post 01:43Z): gate read on HELD seeds
1020–1039 (tuning smoke declared), stage-B 400-kept target, stage-C
rig-ft class 3000 steps + flow arm retained (F-instrument ≠
F-null/F-flat), convention seam = rig-frame identity / recomputed
table / no shim in B–D. (4) `wrist-screen-results-post` queued
(depth refill).

**Next**: `queue_cli.py next` → **grasp-sft-bootstrap stage-A gate
read** (~0.2 GPU-h, rendered) at the **next work-session boundary**
per the objection window opened 01:43Z 08-15 (owner go collapses it);
then stages B–D per the frozen ladder. `wrist-screen-results-post`
is the writing-ladder item. `run_work_next` armed.*

*Updated 2026-08-14 23:45–23:5xZ (real `date -u` at stamp: 23:49) —
tick (babysit): **stage-1 healthy mid-W1; owner v30→v21 question
answered in-channel with receipts; grasp-SFT pre-reg gap patched
(§6 finalization item 4 — convention seam).***

**Status**: **STAGE 1 LIVE + healthy** — babysit green (3 procs, GPU
13.9 GiB/100%, cumulative projection 1.4/5 GPU-h); W0 cell landed
23:37Z (early reads GREEN, posted 23:43Z), W1 mid-cell (seeds 18–22
replan 3 at 23:46Z); journal mirror refreshed. rc ETA unchanged
~01:0x–01:4xZ 08-15. Queue validate OK depth 2, 16 open.

**Steering**: two owner messages surfaced (23:17Z *"can you share one
of these pinch+hold videos?"* — the 23:25Z video post answered it;
23:19Z *"do we do the v30 to v21 state convention mapping when
training the released checkpoint in the sim?"*). Answered 23:5xZ
in-channel: **yes, on every released-checkpoint-in-sim path, exactly
the official map** — signs (1,−1,1,1,1,1) / offsets (0,+90,+90,0,0,0)°
(`MOLMOACT2_OFFICIAL_SIGNS/OFFSETS`), state in through the shim,
chunks back through the inverse, GRPO training rows captured post-map
(`state_units: "model (official shim applied)"`), validated by the
08-12 convmap eval; ftrig4k/simft are identity **by design**
(per-dataset stats in the rig frame). Both inbox ids acked — inbox
empty.

**Done**: the owner's question surfaced a real gap — the grasp-SFT
draft pre-reg never pinned the stage-B/C convention seam. §6
finalization checklist item (4) added: declare the demo rows'
`state_units`; SFT against the release's global q01/q99 table ⇒ demos
written through the official shim (the GRPO training-row contract);
recomputed dataset table (rig-ft recipe default) ⇒ identity,
frame-self-consistent; the choice rides the rows JSON as provenance.

**Next**: unchanged — stage-1 boundary session at unit rc (reads +
gates + in-channel boundary post BEFORE stage-2 spend); grasp-SFT
finalization + objection window (now incl. item 4) ahead of its GPU
stages. `run_work_next` armed (confirmed present).*

## Utilization footer

Session 2026-08-15 01:44–01:4xZ (tick; 0 GPU-h): no-op verification —
GPU idle confirmed, Discord/inbox empty (no objection to the 01:43Z
grasp-SFT finalization yet), queue validate OK depth 2,
`run_work_next` confirmed armed for the stage-A gate-read work
session.

Session 2026-08-14 23:57–01:5xZ 08-15 (work; exploit; ~3.1 GPU-h
counted at the stage-1 boundary per its launch note, 0 launched
in-session): stage-A expert 10/16 → 14/16 (`d1b2552` settle,
`2435a6d` jam-flip; two mechanisms diagnosed by measurement — the
release drop-heel and the deck-strike contact stall); stage-1 ridden
to rc 01:32:02Z and CLOSED at the boundary with verdict F-INSTRUMENT
(reads banked, T1 control CI-straddles both channels at n=25, W3
+18/100 engagement recorded; stages 2/3 never launch, ~10 GPU-h of
the screen's worst case returned); grasp-SFT pre-reg FINALIZED
(`758666f`, objection window open 01:43Z); owner status question
answered in-conversation (01:34Z); queue depth 2 restored
(results-post item queued); babysit entry pruned, GPU free 01:32Z.

Session 2026-08-14 23:45–23:5xZ (tick; 0 GPU-h in-session — stage 1
rides detached, counted at its boundary): babysit green mid-W1
(3 procs, GPU 100%, 1.4/5 GPU-h projection; journal mirror
refreshed); owner v30→v21 question answered in-channel with receipts
(yes — the official shim on every released-checkpoint-in-sim path,
training rows post-map; bijou fine-tunes identity by design); the
23:17Z video ask acked (the 23:25Z video post was its answer);
grasp-SFT pre-reg §6 gap patched (finalization item 4: pin the
stage-B/C convention seam); inbox cleared to empty; queue validate OK
depth 2; `run_work_next` armed for the stage-1 boundary session.

Session 2026-08-14 21:32–22:3xZ (work; exploit; ~0.3 GPU-h in-session
— parity-probe rerun + stage-0 placement/bit-replay; stage 1 ~3–3.5
GPU-h rides detached, counted at its boundary): extended live with
the owner (21:47–22:07Z): `main-review-molmoact2-final` DONE all 4
deliverables —
phases 3–5 reviewed (verdict ADOPT, review post published + summary
in-channel), the 1e-4 re-baseline judged AGREE with the
cross-decomposition mechanism self-verified against the port source,
probe_grpo_replay_parity rerun PASS (masks bit-equal 1,903 + 1,904
rows, spreads recorded), wrist-screen checkpoint-surface VERDICT no
amendment (`wrist-transfer-screen-run` re-statused queued,
launch-ready), Decision-11/masked-only/Gumbel notes absorbed into the
R1-B record; posts-index drift fixed. Then at the owner's live
steering: nit fixes pushed (`2ff6b6c`), the GRPO-90% competence-first
plan posted (owner 👍) and parallelized — **stage 0 EXECUTED**
(`c5be36f`: honesty placement PASS on the serving substrate, `none`
bit-replay bit-equal, `--top-transform` landed for T1), **stage 1
LAUNCHED** 22:24:42Z (unit `wrist-screen-stage1`, babysit entry, gate
5 GPU-h), grasp-SFT draft pre-reg posted + queued
(`grasp-sft-bootstrap`); `run_work_next` armed for the stage-1
boundary session.

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
