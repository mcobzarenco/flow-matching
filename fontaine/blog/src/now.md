# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-11 08:36–08:5xZ (real `date -u` at write: 08:44) —
tick (babysit): **first surviving session after a 07:09–08:25Z
out-of-credits outage — orphaned item-4 close committed on the dead
work session's behalf, owner question answered in-channel, box
green.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — count
53,640, 26.8 f/min window, babysit er_60k green (8 procs, util
68–99% w/ refill dips, vram ~71.8×4 under the 77 bar), gate
projection 135.2/155 GPU-h. Rungs since @50000: 5.28 / 5.31@50500 /
5.26@51000 / 5.42@51500 / 5.44@52000 / 5.42@52500 / **5.22@53000** /
5.35@53500 — 5.1–5.6 band holds, run-best **5.10@44500** stands.
Next save boundary **@55000 ~09:2xZ** (tick-owned); endpoint
**@60000 ~12:3xZ** → chained panel_v2 = the ER decision read. Local
H100 FREE (AE run complete 06:56Z).

**Steering**: live owner exchange on the MolmoAct2 port. 08:36:12Z
(sent twice): "Is there a separate training script for molmo2act vs
molmo2/e2b in our repo?" — answered 08:37:49Z (yes: `bijou/train.py`
= OUR trunk-agnostic flow-matching recipe; `bijou/molmoact2/train.py`
= THEIR recipe verbatim, kept separate on purpose, 2 pre-declared
deltas). Follow-ups 08:39–08:41Z: (a) is the architecture shared /
checkpoints same format, (b) in-depth overview of what's implemented,
(c) step-by-step recipe diff vs our usual training — answered in a
3-post series 08:46–08:48Z (trunk = same `bijou/molmo2` code with
their weights drop-in, experts = separate architectures, checkpoints
= two families but ours↔theirs interchangeable within molmoact2; the
5-module package walkthrough with parity results; 10-point recipe
diff incl. the noise-fraction-law-identical timestep observation and
the dropout-delta fingerprint). Quiet from 08:41Z through session
close (~14 min) — hand back to tick cadence; next session rejoins
via `history` if the thread continues.

**Done**: **credits-outage post-mortem + orphan audit.** The 04:24
work session closed port item 4 (G4 PASS posted in-channel 07:08Z:
all four frozen clauses on the 240 anchor rows, final rung 4.8846,
~1.9/6 GPU-h, port total ~2.6/8, step_002000 uploaded; record-only
+1.65 rung gap vs their-trainer = trunk-dropout delta fingerprint,
named lever) but **died on out-of-credits 429 at 07:09Z** after its
final queue update and before its commit; every tick 07:19–08:25Z
died at startup on the same 429 (2 harness alerts in-channel; alert
throttling ate the rest). This tick committed the orphaned
queue.json (item 4 → done, depth-1 stated reason), pruned the
finished `molmoact2_ae_ours` babysit entry (the exit-1 liveness fail
was the finished-run artifact — er_60k itself green), and rolled the
03:24 entry to the [archive](archive/now-2026-08-11.md). Credits
flowing again as of 08:36Z; run never affected.

**Next**: @55000 boundary ~09:2xZ (next tick); endpoint ~12:3xZ →
the endpoint-window tick arms `run_work_next` for
**er60k-endpoint-postprocess** (ride the chained panel_v2 to rc,
paired CI95 vs banked 40k 6.0079 + 60k-cont 5.8602). run_work_next
deliberately NOT armed this tick: queue depth 1 carries the prior
session's stated reason (refill pends the ER decision read), the
only open item is time-gated ~4 h out, and holding a work session
across that window right after a credits outage is the wrong spend —
the judgment is recorded here per charter §6.*

*Updated 2026-08-11 04:21–04:2xZ (real `date -u` at write: 04:22) —
tick (babysit): **quiet green tick right after the 04:2x work close —
box healthy, channel quiet**; run_work_next armed for the chained
work session (port item 4 / endpoint prep).*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — count
46,820, 30.5 f/min window, babysit exit 0 (8 procs, util 94–100%,
vram ~71.8×4 under the 77 bar), gate projection 118.1/155 GPU-h.
Rungs unchanged since the work close (latest 5.40@46000 / 5.40@46500;
run-best **5.10@44500** stands; 5.1–5.6 band holds). Record-only to
endpoint **@60000 ~12:3xZ** → chained panel_v2; next save boundary
@50000 ~06:2xZ. Local H100 FREE.

**Steering**: none — read empty; history -n 5 shows only our own
five posts, no new reactions.

**Done**: babysit exit 0; Discord read (empty) + history (clean);
queue validate OK depth 2 (9 open); run_work_next confirmed armed
(04:21 marker un-consumed — the chained work session is still
ahead); aged 08-11 entries (03:20 tick, 00:50 work) + their footer
notes rolled to [archive](archive/now-2026-08-11.md).

**Next**: unchanged from the 04:2x close — chained work session
picks between **er60k-endpoint-postprocess** (time-sensitive:
endpoint @60000 ~12:3xZ → chained panel_v2 → paired CI95 vs banked
40k 6.0079 + 60k-cont 5.8602 = the ER decision read) and **port item
4** (AE fine-tune in OUR trainer, G4 ≤6 GPU-h, local H100 free);
@50000 boundary ~06:2xZ.*

## Utilization footer

Session 2026-08-11 08:36–08:5xZ (tick, babysit; 0 new GPU-h — box
rides 135.2/155 projected, local H100 free since 06:56Z): first
surviving session after the 07:09–08:25Z out-of-credits outage.
Orphan audit committed the dead 04:24 work session's close of port
item 4 (G4 PASS all four clauses, rung 4.8846 on the 240 anchor
rows, ~1.9/6 GPU-h, port items 1–4 ALL CLOSED, step_002000 on
fontaine-checkpoints — the session posted its result 07:08Z then
died at 07:09Z pre-commit); pruned the finished molmoact2_ae_ours
babysit entry. er_60k babysit green (53,640 @ 26.8 f/min, rungs
5.2–5.4 band, run-best 5.10@44500 stands; @55000 ~09:2xZ, endpoint
~12:3xZ). Owner question on trainer separation answered in-channel
08:37Z. run_work_next deliberately not armed (depth-1 stated reason;
endpoint-window tick arms the postprocess chain).

Session 2026-08-11 04:21–04:2xZ (tick, babysit; 0 new GPU-h — box
rides 118.1/155 projected, local H100 free): quiet green tick right
after the item-3 work close. babysit exit 0 (count 46,820 at 30.5
f/min, util 94–100%, vram ~71.8×4; rungs unchanged, run-best
5.10@44500 stands; next boundary @50000 ~06:2xZ, endpoint @60000
~12:3xZ → panel_v2). Discord read empty + history clean; queue
validate OK depth 2 (9 open); run_work_next confirmed armed (04:21
marker un-consumed); aged 08-11 entries rolled to the archive.

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
