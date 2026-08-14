# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-14 21:32–22:3xZ (real `date -u` at stamp: 22:31) —
work session, extended live with the owner: **review DONE + nit fixes
pushed at the owner ask; GRPO-90% plan agreed (👍) and parallelized —
wrist-screen stage 0 EXECUTED (all oracles green), STAGE 1 LIVE
(unit `wrist-screen-stage1`), grasp-SFT draft pre-reg posted.***

**Status**: **STAGE 1 LIVE** — unit `wrist-screen-stage1` since
22:24:42Z (det gate ×2 → hold(25) → W0/W1/W3(100 each) + T1(25),
~3–3.5 GPU-h, rc ETA ~01:0x–01:4xZ 08-15); first babysit green
(4 procs, GPU 13.7 GiB/100%, gate 5 GPU-h). Queue validate OK depth 2,
16 open.

**Steering**: live exchange 21:47–22:07Z — (1) *"push fixes for the
nits to your branch"* → done `2ff6b6c`; (2) *"what should we do next
to train a policy which solves over 90% of seeds?"* → competence-first
plan posted, owner 👍; (3) *"do as much in parallel as you reasonably
can"* → stage 0+1 executed/launched THIS session + the grasp-SFT
draft pre-reg posted and queued (`grasp-sft-bootstrap`).

**Done**: `main-review-molmoact2-final` all 4 deliverables (review
post + in-channel summary, verdict ADOPT; re-baseline judgment AGREE
with the mechanism self-verified; probe rerun PASS on both banked
waves; checkpoint-surface VERDICT no amendment; Decision-11/
masked-only/Gumbel notes absorbed into the R1-B record) `58cc07f`;
nit fixes `2ff6b6c`; **stage 0 EXECUTED** `c5be36f` (honesty
placement PASS on the serving substrate — W0 0.8769 ≈ banked 0.877,
W1 1.0, W3 0.8867 CI-excl-0; `none` bit-replay PASS bit-equal;
`--top-transform` landed for T1 with oracles); **stage 1 launched**
22:24:42Z + babysit entry; grasp-SFT draft pre-reg
(posts/2026-08-14-prereg-grasp-sft-bootstrap.md) posted + queued;
**stage-A scripted expert WORKS** (`c23863d` WIP → `b564337`
breakthrough): **10/16 demo-seed successes end-to-end** (~4–5 s per
success; pan-arc traverse was the unlock — pan's vertical axis
carries no gravity load, so the lifted posture's carry height
survives the swing where IK-to-hover fought the saturated shoulder);
3 of 6 misses are ON the disk (tipped at release — the polish item);
success video in-channel; 5 CPU oracles green; seven mechanisms
diagnosed and fixed in code, the servo-saturation envelope recorded
as a finding. Stage-1 early reads GREEN (not the formal boundary):
W0 mean +0.054 cm / moved 44 per 100 — both in-band vs banked
+0.08 / 47; hold floor +0.0000; 2 W0 successes.

**Next**: stage-1 boundary session at unit rc (~01:0x–01:4xZ 08-15):
reads + gates (sanity band [−0.3,+0.5] cm / [25,70] engaged, hold
floor, T1 CI95, spawn_xy pairing, first W1/W3 deltas) + in-channel
boundary post BEFORE stage-2 spend. `grasp-sft-bootstrap` stage A
(scripted expert) is the executable CPU slice; finalization +
objection window before its GPU stages. `renderer-pbr-wrist-pilot`
stays owner-gated. `run_work_next` armed.*

*Superseded head entry from earlier this session (pre-steering,
retained verbatim below):*

*Updated 2026-08-14 21:32–21:5xZ (real `date -u` at stamp: 21:43) —
work session: **`main-review-molmoact2-final` DONE, all 4 deliverables
— review verdict ADOPT, re-baseline judgment AGREE, probe rerun PASS,
wrist screen cleared to launch (no amendment).***

**Status**: **No live run** — the parity-probe rerun (~10 min GPU)
completed and the GPU is back to 0 MiB; nothing else launched this
session. Main at `26ac1e6`, fontaine rebased on top (`64c93e6` base).
Queue validate OK: depth 1 with a stated reason (the screen ladder
generates its own follow-ons at stage boundaries), 15 open.

**Steering**: none this session (inbox empty at boot; the 21:14Z
review ask is the item executed here).

**Done**: **`main-review-molmoact2-final`** — (a) [review
post](posts/2026-08-14-molmoact2-retirement-review.md) + in-channel
summary: verdict **adopt without reservation**; the 1e-5→1e-4
re-baseline **judgment AGREE** with the mechanism self-verified (port
replay = monolithic `cat(prompt,suffix)` forward; first-class =
prefill + cached continuation — a genuine cross-decomposition, drift
in the phase-2 diagnostic's decade, ratio impact 0.01% vs the clip
band); 4 ranked nits (train.py ~4420 dead/false print after the
rider-guard raise; `hole_count` per-worker undercount; the discrete
fixture generator's missing run-at-tag note; a cosmetic from_numpy
warning). (b) **probe rerun PASS** — masks bit-equal on ALL
1,903 + 1,904 rows of R1-A/R1-B; spreads recorded (v1 med 5.68e-1 /
p90 1.29 / max 3.92; v2 med 5.52e-1 / p90 1.58 / max 8.84,
report-only per registration). (c) **VERDICT: NO AMENDMENT** —
ftrig4k/simft ride `BijouPolicy --checkpoint` (flow pathway,
untouched by the re-point); `wrist-transfer-screen-run` is
launch-ready as registered and re-statused queued. (d) Decision 11 +
masked-only decode + full-width Gumbel absorbed as a dated
post-retirement note on the R1-B record. Also: posts-index drift from
the capped 18:59Z session fixed (squint + prereg-final entries
restored).

**Next**: `queue_cli.py next` → **`wrist-transfer-screen-run`** —
stage 0 GPU tail (`none` bit-replay oracle + W1/W3 honesty placement,
~0.1 GPU-h) then stage 1 (P1 × {W0,W1,W3} + T1, ~3.3 GPU-h) under
the FINAL pre-reg, no further paperwork; hard-stop boundary posts
per §5. `run_work_next` armed. `renderer-pbr-wrist-pilot` stays
owner-gated.*

## Utilization footer

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
