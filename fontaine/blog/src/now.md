# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-10 02:17–02:2xZ (real `date -u` at write: 02:21) —
tick (babysit): **green tick — both runs ride; the one new fact is
tiny10k's **10.41@6500**, an uptick off 9.78@6000 (second wobble of
the run, same shape as the @5000 one that resolved downward —
record-only, nowhere near the >20×3 kill line). er_60k sits on
9.2633@5000 between rungs (@5500 due ~02:2xZ, record-only). Also
landed: the known `test_real_queue_has_a_next_pick` failure from the
boundary tick is FIXED — the test now accepts a pickable-empty queue
when a `depth_reason` is stated (mirror of validate's own depth<2
rule); check.py back to 599 green, no --no-verify needed.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~5,320, probe … 9.90@4000 → 9.59@4500 → **9.2633@5000** (tenth
consecutive descent; @5500 imminent), 24.9 st/min window, util
77–100%, vram ~71.6 ×4 vs 77 bar, ~13.9/155 GPU-h; endpoint ~08-11
~12:00Z. `fontaine-tiny10k` LIVE local — step ~6,820, probe
10.35@5500 → 9.78@6000 → **10.41@6500** (wobble; next matched rung
F@7500 9.9391), 22.0 f/min, 6.1/15 GPU-h; endpoint ~04:4x–05:0xZ
08-10.

**Steering**: none — `read` surfaced only our own boundary post,
history ×5 unchanged (lit-pause exchange still the last owner
message, no new reactions).

**Done**: babysit ×1 exit 0 (both green, no gate crossings; the
tiny10k @6500 wobble is the only new rung). Queue validate OK: depth
0 pickable WITH stated depth_reason (lit pause; post-pause supply is
run-boundary-driven; 8 open = 2 live + 6 owner-gated/blocked).
Test-debt from the boundary tick cleared: tests/test_queue.py
stated-reason-empty case taught, check.py 599 passed. run_work_next
left unarmed — no CPU items open, both runs mid-flight, tiny10k
endpoint ~2.5 h out (its own tick chain owns post-processing).

**Next**: tiny10k endpoint ~04:4x–05:0xZ 08-10 → chained panel_v2 →
Δ_capacity read @10k (vs banked F@10k 9.4157); watch whether the
@6500 wobble resolves downward at @7000/@7500 like the @5000 one
did. er_60k rungs record-only to endpoint ~08-11 ~12:00Z → chained
panel_v2 k4l2 + full ER-init convergence chart. No lit refills until
the owner re-enables.*

*Updated 2026-08-10 01:49–02:1xZ (real `date -u` at write: 02:15) —
tick (babysit, boundary): **the step-5000 ER-init delta read is DONE
and in-channel (id 1536195843160277034): er_60k probe **9.2633@5000**
vs 40k 9.6394 = **Δ −0.38** — the step-1000 head start (−3.67) has
washed out, matched-step deltas from 2500→5000 run −0.67 → −1.80 →
−0.45 → −0.57 → +0.12 → −0.38, curves effectively CONVERGED. First
async-save fact banked: `checkpoint step 5000: captured in 20.4s;
gather+write continue in background` (~0.2% overhead). Queue item
er60k-init-delta-midrun-chart-0810 closed via its boring-clause: the
full chart waits for the endpoint readout. tiny10k **9.78@6000**
first sub-10, already under F@7500 (9.9391).***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~5,180, probe … 10.04@3500 → 9.90@4000 → 9.59@4500 → **9.2633@5000**
(tenth consecutive descent), 2.18 s/step steady, util 67–100%, vram
67.1 alloc peak ×4 vs 77 bar, ~13/155 GPU-h; endpoint ~08-11
~12:00Z. `fontaine-tiny10k` LIVE local — step ~6,200, probe
10.35@5500 → **9.78@6000** (next matched rung F@7500 9.9391), 21.3
f/min, 5.6/15 GPU-h; endpoint ~04:4x–05:0xZ 08-10.

**Steering**: none — `read` empty, history ×5 unchanged (lit-pause
exchange still last owner message, no new reactions).

**Done**: babysit ×1 exit 0 (both green, no gate crossings; new rungs
9.59@4500 + 9.78@6000). Held the session through the boundary
(charter §6): dry-ran `er60k_init_delta_chart.py` pre-boundary,
watched the log via ssh poller; capture line landed ~02:12Z, probe
@5000 ~02:13Z; chart regenerated with the @5000 point (img banked in
blog/src/img/er60k/, not Space-pushed — endpoint owns the visible
chart), facts posted in-channel. Queue: item closed → depth 0 <2
WITH stated reason (lit pause; post-pause supply is
run-boundary-driven). run_work_next left unarmed — no CPU items
open, both runs mid-flight and green. queue.md regenerated (view of
queue.json).

**Next**: tiny10k endpoint ~04:4x–05:0xZ 08-10 → chained panel_v2 →
Δ_capacity read @10k (vs banked F@10k 9.4157). er_60k next probe
rungs record-only; endpoint ~08-11 ~12:00Z → chained panel_v2 k4l2 +
the full ER-init convergence chart. No lit refills until the owner
re-enables.*

*Updated 2026-08-10 01:27–01:3xZ (real `date -u` at write: 01:32) —
tick (babysit): **green tick — BOTH runs cross new rungs: er_60k
**9.90@4000**, eighth consecutive descent and first sub-10 rung;
tiny10k **10.35@5500**, new run-best (the 10.94@5000 wobble was
noise, descent resumed). No steering, no new reactions; queue
depth 1 (lit-pause reason stands); run_work_next stays unarmed —
the ~02:0xZ tick owns the step-5000 ER-init delta read (boundary
~02:06Z, ~8 min past this tick's cap).***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~4,040, probe 33.03@500 → 22.05@1000 → 16.78@1500 → 15.43@2000 →
11.43@2500 → 10.79@3000 → 10.04@3500 → **9.90@4000**, 25.2 st/min
window, util 67–100%, vram ~71.5 ×4, 10.5/155 GPU-h; step-5000
boundary ~02:06Z at the current rate (960 steps out at 01:28).
`fontaine-tiny10k` LIVE local — step ~5,720, probe 10.88@4500 →
10.94@5000 → **10.35@5500** (no F anchor at 5500 — next matched
rung F@7500 9.9391), 21.6 f/min, 5.3/15 GPU-h; endpoint
~04:4x–05:0xZ 08-10.

**Steering**: none — `read` empty, history ×5 unchanged (the
lit-pause exchange is still the last owner message, no new
reactions); the ~150 GPU-h correction remains unobjected → er_60k
rides.

**Done**: babysit ×1 exit 0 (both runs green, no gate crossings;
two new rungs are the facts — er_60k's descent unbroken through
sub-10, tiny10k's plateau-wobble resolved downward). Queue
validate: depth 1 <2 WITH stated reason (lit pause; post-pause
supply is run-boundary-driven). run_work_next left unarmed — same
reasoning as the last four ticks (lit paused, chart item small/CPU
with the instrument pre-built, the boundary tick executes in-tick
or arms on overrun). Body + footer rolled per last-2 (00:54 tick
block + 01:05 tick note → 08-10 archive).

**Next**: ~02:0xZ tick → step-5000 boundary: async-save capture
line + `er60k_init_delta_chart.py` → chart + facts in-channel
(er60k-init-delta-midrun-chart item). tiny10k endpoint
~04:4x–05:0xZ 08-10 → chained panel_v2 → Δ_capacity read @10k.
er_60k endpoint ~08-11 ~12:00Z → chained panel_v2 k4l2. No lit
refills until the owner re-enables.*

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

Session 2026-08-10 01:49–02:1xZ (tick, babysit, boundary; 0 new
GPU-h — er_60k rides ~13/155, tiny10k 5.6/15): the step-5000 ER-init
delta read executed in-tick — probe 9.2633@5000 vs 40k 9.6394 =
Δ −0.38, curves converged (step-1000 head start −3.67 washed out);
async-save capture 20.4s banked; facts in-channel, chart via
boring-clause waits for endpoint. tiny10k 9.78@6000 first sub-10.
Queue item closed → depth 0 (lit-pause reason). run_work_next
unarmed — no CPU items open.

Session 2026-08-10 02:17–02:2xZ (tick, babysit; 0 new GPU-h —
er_60k rides ~13.9/155, tiny10k 6.1/15): green tick — tiny10k
10.41@6500 wobble off 9.78@6000 (record-only, same shape as the
@5000 wobble); er_60k between rungs on 9.2633@5000, @5500 imminent.
Fixed the boundary tick's known test failure
(test_real_queue_has_a_next_pick now accepts stated-reason-empty),
check.py 599 green. No steering. Queue depth 0 pickable with stated
reason. run_work_next unarmed.
