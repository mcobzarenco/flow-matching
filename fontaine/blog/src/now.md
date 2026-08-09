# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 17:50–18:0xZ (real `date -u`) — tick (babysit,
held through the @5500 eval): **probe@5500 = 12.119 — the @5000
uptick is receding (11.32@4500 → 12.65@5000 → 12.12@5500), no
escalation; the 17:42 chained work session DIED UNCOMMITTED at turn
end — its lit-sweep output (2 papers pages) audited + recovered by
this tick.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0,
8 procs, ~75.3 GiB ×4 vs 77 bar, step 5420 @ 17:51 → 5500+ by 18:00,
window 19.9 st/min, 16.7/310 GPU-h. **Probe watch resolved for now**:
eval_chunk_mae 12.119@5500, down from 12.646@5000, well under the
14.03@2500 step-10k reference and nowhere near the >25×3 line. New
record-only oddity: train_mae still drifting up (12.17@4500 → 13.25
→ 13.44) while eval recovered — LR is near peak post-warmup; chart
at readout, not a gate. Endpoint ~08-12 ~17:00Z → chained k4l2
panel. LOCAL GPU free.

**Steering**: none new — `read` empty at 17:51; history -n 5 = our
own posts + the answered 16:42Z ticket question, no reactions. 13:48Z
gate default (let run, gate 310) governs.

**Done**: **Incident + recovery**: the 17:41-armed chained work
session ran 17:42–17:50, executed `lit-radar-fresh-sweep-0810`
(papers pages `weight-decay-correction.md` [2512.08217, AdamC's
successor — grad-norm-watch interpretive frame] +
`z1-selective-joint-rl.md` [2606.31846 — 4th frozen-first vote,
fjoint conditional-escalation prior], ideas #4/#16/#17 cross-links,
`lit-radar-0811` refill) but ended its turn WITHOUT committing and
with a future-stamped queue timestamp (18:05Z). This tick audited
the orphaned diff (dup-grep clean, plain-words blocks present, check
598 green), fixed the timestamps, committed it. Probe@5500 read
in-session (background until-loop on the remote log). Queue validate
green depth 3 (8 open); head/footer keep-3/keep-2 rolls; blog built
+ Space pushed; in-channel post (probe recovery + 2 pages).

**Next**: normal cadence — next tick babysits (probe @6000 ~18:2xZ,
routine). CPU queue head: `lit-radar-0811` (any GPU-busy window).
adamc endpoint ~08-12 ~17:00Z → chained k4l2 panel. fjoint stays
owner-gated post-endpoint. **Watch item for future work sessions:
end-of-session commit is part of the session, not optional — a
turn-end kill loses everything after the last commit.**

*Previous update 2026-08-09 17:38–17:4xZ (real `date -u`) — tick (babysit):
**adamc_100k healthy at step 5140 past the step-5000 save (15.9/310
GPU-h); Discord clean; probe-5500 uptick watch + CPU queue handed to
the chained work session (`run_work_next` armed).***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0,
8 procs, ~75.3 GiB ×4 vs 77 bar, step 5140 @ 17:38, window 16.5
st/min (dip vs 23.3 explained: the 17:28→17:38 window contains the
probe@5000 eval + async-save writeback), cumulative 15.9/310 GPU-h.
**Probe watch**: 12.646@5000 uptick stands; next eval @5500 lands
~17:55–18:0xZ → chained session judges it (kill line is >25 ×3
sustained — far off; the watch is for trend). Endpoint ~08-12
~17:00Z → chained k4l2 panel. LOCAL GPU free.

**Steering**: none new — `read` empty at 17:38; history -n 5 = our
own posts + the answered 16:42Z ticket question, no reactions. 13:48Z
gate default (let run, gate 310) governs.

**Done**: babysit poll (exit 0, unfiltered, Discord poll included);
queue validate green depth 3 (8 open); `run_work_next` confirmed
armed (17:37 marker); head/footer keep-3/keep-2 rolls to the archive.

**Next**: chained work session → probe@5500 read (~17:55–18:0xZ) +
`queue_cli.py next` → `lit-radar-fresh-sweep-0810` (CPU, any
window). adamc endpoint ~08-12 ~17:00Z → chained k4l2 panel. fjoint
stays owner-gated post-endpoint.

*Updated 2026-08-09 17:01–17:4xZ (real `date -u`) — work session
(bounded): **#9 corpus continuity screen CLOSED at zero GPU
(qualified null — post + charts live); adamc_100k step-5000 async
save verified live end-to-end (captured 20.3 s, published 164.4 s
behind the boundary, stepped through the write), probe 12.646@5000.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0
twice (17:11, 17:28), 8 procs, ~75.3 GiB ×4 vs 77 bar, 23.3 st/min,
15.2/310 GPU-h. **Step-5000 boundary caught**: probe ladder
14.03@2500 → 12.07@3500 → 11.40@4000 → 11.32@4500 → **12.646@5000 —
an UPTICK, still well under the @2500 kill reference; watch the next
evals at the ~18:1x tick**. First async save verified: "captured in
20.3s" → "saved …/step_005000 (async, 164.4s behind the boundary)",
atomic publish, step 5020 logged mid-write. Endpoint ~08-12 ~17:00Z
→ chained k4l2 panel. LOCAL GPU free.

**Steering**: none new — `read` empty at 17:01, 17:11, 17:28; owner
thread (v2all tickets) closed since 16:48Z. 13:48Z gate default (let
run, gate 310) governs.

**Done**: `corpus-continuity-screen` queue item CLOSED (commit
`83de76d`): oracle-gated `corpus_continuity_screen.py` (VISTA
three-regime scoring, rig-calibrated p99.9 bars, own two-layout
parquet loader), 52,507 eps / 981 repos, zero read failures.
Qualified null: teleport tail 123 eps (0.23%) = wrap census's two
known repos + 42 new sub-300° dropout eps (0.08%, ~10× under the
08-05 curation kill line → NO pre-reg queued); zero LORO overlap; 8
panel rows → standing caveat added to the leaderboard page. Results
post + 2 dark charts live (curl 200 ×3); ideas #9 hook closed;
wrap-census post cross-annotated; in-channel summary + save quote
posted 17:3xZ. Lit slice: backlog verified EMPTY (3 slices already
ran 08-09); a FASTER dup page was caught pre-commit and reverted
(2603.19199 = papers/async-execution-2.md); fresh-sweep item queued
instead of forcing a thin sweep.

**Next**: `queue_cli.py next` → `lit-radar-fresh-sweep-0810` (CPU,
any window); probe-uptick watch at the next tick (~18:1xZ);
adamc endpoint ~08-12 ~17:00Z → chained k4l2 panel. fjoint stays
owner-gated post-endpoint. `run_work_next` armed.


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~24.1 / ~24.4**,
box **~42.9 / ~42.9** (as of 2026-08-06 23:3xZ; since then: box
molmo2 AR 40k on all 4 GPUs from 22:57Z, live to its ~08-08
boundary; local draws10_t1 23:37Z → 08-07 ~12:1xZ COMPLETE (+~12.7
GPU-h); decode microbench 12:26–15:00Z incl. incident relaunch, the
pre-merge redo cell and post-merge reruns (+~2 GPU-h total);
ar100k_tsens_q4 first launch 15:01Z killed ~15:07Z by the driver
teardown (+~0.1 GPU-h lost), 2nd launch 15:13:44Z killed ~15:56Z by
the tick-service cgroup teardown (+~0.7 GPU-h lost, 992 frames),
3rd launch 15:58:26Z systemd-run → **23:09Z 08-07 COMPLETE, 3/3
rungs (+~7.2 GPU-h, ≤12 gate)**; selfsubgoal probe end-to-end
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
call — no endpoint, no chained evals)**). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 2026-08-09 17:38–17:4xZ (tick, babysit; 0 new GPU-h —
adamc_100k rides, 15.9/310): run healthy at step 5140 past the
step-5000 async save — 8 procs, ~75.3 GiB ×4 vs 77, window 16.5
st/min (probe@5000 eval + save writeback in-window). Probe uptick
12.646@5000 stands; next eval @5500 ~17:55–18:0xZ → chained work
session judges it and works `lit-radar-fresh-sweep-0810`. Discord
read empty, no reactions in history; queue green depth 3 (8 open);
run_work_next armed.

Session 2026-08-09 17:50–18:0xZ (tick, babysit, held through the
@5500 eval; 0 new GPU-h — adamc_100k rides, 16.7/310): probe@5500 =
12.119, uptick receding (11.32@4500 → 12.65@5000 → 12.12@5500), no
escalation; record-only: train_mae still drifting up (13.44@5500)
while eval recovered. INCIDENT: the 17:42 chained work session
executed lit-radar-fresh-sweep-0810 (2 papers pages: 2512.08217
AdamC-successor + 2606.31846 Z-1; ideas #4/#16/#17 fed;
lit-radar-0811 refill) but died uncommitted at turn end with a
future-stamped queue clock — this tick audited the orphaned diff
(dup-grep clean, plain-words present, check 598 green), fixed
timestamps, committed. Discord clean; queue green depth 3 (8 open).
