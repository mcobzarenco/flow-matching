# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-09 16:45–16:5xZ (real `date -u`) — tick (babysit,
conversational hold): **adamc_100k healthy step 4000 (12.4/310);
owner asked "Did you push the ticket to git?" 16:42Z — answered
16:48Z (yes: commit `ea1cbf2` on `fontaine`, in sync with origin,
sha256 `ec0484e8…` re-verified, all three ticket vectors listed +
hub mirror `d8cbfcc`).***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0,
8 procs, ~75.3 GiB ×4 vs 77 bar, step 4000 @ 16:46, window 20.5
f/min, 12.4/310 GPU-h; probe ladder unchanged (14.03@2500 = @10k
kill-bar ref). Next boundary: **step-5000 async-save line ~17:2xZ —
quote owed in-channel; falls past this tick's hard kill,
`run_work_next` armed so the chained session catches it.** LOCAL GPU
free.

**Steering**: owner question 16:42:10Z ("Did you push the ticket to
git?") — answered in-channel 16:48Z after re-verifying: npz tracked
in git at `ea1cbf2`, branch clean vs origin/fontaine, sha match;
pointed at all three vectors in `plans/` (12 / 59 / 33). No
reactions in history -n 5. Conversational hold kept with a
background history-watcher (cursor untouched) through end of tick.
13:48Z gate default (let run, gate 310) governs.

**Done**: babysit poll (exit 0, unfiltered); git/push verification +
in-channel reply; queue validate green depth 3 (8 open);
`run_work_next` confirmed armed; 15:59 head entry rolled verbatim to
the archive (keep-3), footer notes rolled (keep-2).

**Next**: chained work session → step-5000 async-save quote ~17:2xZ
+ owner-thread rejoin via `history`; CPU queue pointer
`docs-pass-followups-0809` / `corpus-continuity-screen`. adamc
endpoint ~08-12 ~17:00Z → chained k4l2 panel. fjoint stays
owner-gated post-endpoint.

*Updated 2026-08-09 16:38–16:5xZ (real `date -u`) — work session
(chained, bounded): **v2-all ticket thread CLOSED — winner ticket 12
(pooled MAE 5.265 over 32,679 frames), table + memorized-rows read
posted in-channel 16:4xZ; winner npz in-repo + on
fontaine-checkpoints.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0, 8
procs, ~75.3 GiB ×4 vs 77 bar, step 3860 @ 16:39, 11.9/310 GPU-h;
probe ladder unchanged (14.03@2500 = @10k kill-bar ref). Next
boundary: **step-5000 async-save line ~17:2xZ — quote owed in-channel,
falls to the next tick (`run_work_next` armed).** LOCAL GPU free
(`fontaine-ftrig-v2all-winner` consumed 53 s CPU, landed 16:36:58Z).

**Steering**: none new — `read` empty at 16:39 (both polls), no
reactions in history -n 5. 13:48Z gate default (let run, gate 310)
governs.

**Done**: 16:10 handoff bundle (1)–(4) executed. Winner+subsets
service verified landed; **owner table posted 16:4xZ**: winner ticket
12 5.265 · 59 (holdout winner) 5.330 rank-5 · 33 (teacher) 5.405
rank-19 · bank median 5.474. Memorized-rows read: train rows ticket 12
rank-1 (4.536) vs heldout rows rank-9 (11.808) while 59 holds rank-3
(11.722), ticket33 rank-51; Spearman train-vs-heldout rows **0.39** →
ticket choice measurably memorization-sensitive; 59 = generalization
pick, 12 = deployment-fit pick. `ticket_ftrig4k_rigv2all_winner.npz`
(sha ec0484e8) committed in-repo + uploaded to fontaine-checkpoints
`tickets/` (hub commit d8cbfcc); analysis json banked
(`reports/analysis__ftrig_ticket_selection_rigv2all.json`, subset
diagnostics appended). Queue item `owner-ticket-v2all-selection-0809`
recorded done, prereg cited via the now.md-entry route (validate green
depth 3, 8 open); blog built + Space pushed.

**Next**: `queue_cli.py next` → `docs-pass-followups-0809` /
`corpus-continuity-screen` (CPU, any GPU-busy window); step-5000 save
quote ~17:2xZ (tick, `run_work_next` armed); adamc endpoint ~08-12
~17:00Z → chained k4l2 panel. fjoint stays owner-gated post-endpoint.

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

Session 2026-08-09 16:45–16:5xZ (tick, babysit + conversational
hold; 0 new GPU-h — adamc_100k rides, 12.4/310): run healthy at step
4000 — 20.5 f/min window, ~75.3 GiB ×4 vs 77, probe ladder unchanged
(14.03@2500). Owner 16:42Z "Did you push the ticket to git?" —
re-verified (commit ea1cbf2 in sync with origin, sha256 match) and
answered in-channel 16:48Z with all three ticket vectors' paths;
conversational hold kept via background history-watcher. Queue green
depth 3 (8 open); run_work_next armed → chained session catches the
step-5000 async-save quote ~17:2xZ.

Session 2026-08-09 17:01–17:4xZ (work, bounded, explore; 0 new
GPU-h — adamc_100k rides, 15.2/310): #9 corpus continuity screen
closed at zero GPU (qualified null: tail 0.23% = the wrap census's
two known repos + 42 new sub-300° dropout eps far under the curation
kill line; instrument banked as curated_v1 intake filter; leaderboard
caveat added). Step-5000 boundary caught live: async save
end-to-end verified (20.3 s capture / 164.4 s behind-boundary atomic
publish), probe 12.646@5000 uptick flagged for the next tick. Lit
backlog verified empty → fresh-sweep item queued; FASTER dup page
caught pre-commit + reverted.
