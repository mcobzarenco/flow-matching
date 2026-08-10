# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-10 01:16–01:2xZ (real `date -u` at write: 01:19) —
tick (babysit): **green tick — er_60k probe **10.04@3500**, seventh
consecutive descent; tiny10k unchanged since 10.94@5000 (@5500
imminent at step ~5,480). No steering, no new reactions; queue
depth 1 (lit-pause reason stands); run_work_next stays unarmed —
the ~02:0x–02:1xZ tick owns the step-5000 ER-init delta read (the
boundary is ~48 min out, past this tick's cap).***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~3,760, probe 33.03@500 → 22.05@1000 → 16.78@1500 → 15.43@2000 →
11.43@2500 → 10.79@3000 → **10.04@3500**, 25.6 st/min window, util
87–100%, vram ~71.5 ×4, 9.8/155 GPU-h; step-5000 boundary ~02:05Z
at the current rate (1,240 steps out at 01:17). `fontaine-tiny10k`
LIVE local — step ~5,480, probe unchanged since **10.94@5000**
(next rung @5500 imminent; interim Δ +0.68 vs F, record-only),
21.9 f/min, 5.1/15 GPU-h; endpoint ~04:5x–05:1xZ 08-10.

**Steering**: none — `read` empty, history ×5 unchanged (the
lit-pause exchange is still the last owner message, no new
reactions); the ~150 GPU-h correction remains unobjected → er_60k
rides.

**Done**: babysit ×1 exit 0 (both runs green, no gate crossings;
er_60k's @3500 rung is the only new fact — descent unbroken).
Queue validate: depth 1 <2 WITH stated reason (lit pause;
post-pause supply is run-boundary-driven). run_work_next left
unarmed — same reasoning as 00:33/00:54/01:05 (lit paused, chart
item small/CPU with the instrument pre-built, the boundary tick
executes in-tick or arms on overrun). Body + footer rolled per
last-2 (00:33 tick block + 00:54 tick note → 08-10 archive).

**Next**: ~02:0x–02:1xZ tick → step-5000 boundary: async-save
capture line + `er60k_init_delta_chart.py` → chart + facts
in-channel (er60k-init-delta-midrun-chart item). tiny10k endpoint
~04:5x–05:1xZ 08-10 → chained panel_v2 → Δ_capacity read @10k.
er_60k endpoint ~08-11 ~12:00Z → chained panel_v2 k4l2. No lit
refills until the owner re-enables.*

*Updated 2026-08-10 01:05–01:1xZ (real `date -u` at write: 01:07) —
tick (babysit): **green tick between probe rungs — er_60k step
~3,480 at 27.4 st/min (next probe @3500 imminent), tiny10k step
~5,240; no steering, no new reactions; queue depth 1 (lit-pause
reason stands); run_work_next stays unarmed — the ~02:1xZ tick owns
the step-5000 ER-init delta read.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~3,480, probe ladder unchanged since **10.79@3000** (@3500 lands
within minutes of this write), 27.4 st/min window, util 64–95%,
vram ~71.5 ×4, 9.1/155 GPU-h; step-5000 boundary ~02:0x–02:1xZ at
the current rate (1,520 steps out at 01:06). `fontaine-tiny10k`
LIVE local — step ~5,240, probe unchanged since **10.94@5000**
(interim Δ +0.68 vs F, record-only; next rung @5500), 21.9 f/min,
4.9/15 GPU-h; endpoint ~04:5x–05:1xZ 08-10.

**Steering**: none — `read` empty, history ×5 unchanged (the
lit-pause exchange is still the last owner message, no new
reactions); the ~150 GPU-h correction remains unobjected → er_60k
rides.

**Done**: babysit ×1 exit 0 (both runs green, no gate crossings, no
new probe rungs this tick — both runs sit between eval boundaries).
Queue validate: depth 1 <2 WITH stated reason (lit pause; post-pause
supply is run-boundary-driven). run_work_next left unarmed — same
reasoning as 00:33/00:54 (lit paused, chart item small/CPU with the
instrument pre-built, the boundary tick executes in-tick or arms on
overrun). Body + footer rolled per last-2 (00:13 tick block + 00:33
tick note → 08-10 archive).

**Next**: ~02:1xZ tick → step-5000 boundary: async-save capture
line + `er60k_init_delta_chart.py` → chart + facts in-channel
(er60k-init-delta-midrun-chart item). tiny10k endpoint
~04:5x–05:1xZ 08-10 → chained panel_v2 → Δ_capacity read @10k.
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

Session 2026-08-10 01:16–01:2xZ (tick, babysit; 0 new GPU-h —
er_60k rides 9.8/155, tiny10k 5.1/15): green tick — er_60k
10.04@3500 seventh consecutive descent, 25.6 st/min, step ~3,760;
step-5000 boundary ~02:05Z stays with the next tick (past this
tick's 30-min cap). tiny10k step ~5,480, probe unchanged since
10.94@5000 (@5500 imminent). No steering, no new reactions. Queue
depth 1 (lit-pause reason stands). run_work_next left unarmed —
same reasoning as the last three ticks.

Session 2026-08-10 01:27–01:3xZ (tick, babysit; 0 new GPU-h —
er_60k rides 10.5/155, tiny10k 5.3/15): green tick, BOTH runs
cross new rungs — er_60k 9.90@4000 eighth consecutive descent and
first sub-10 (25.2 st/min, step ~4,040; step-5000 boundary ~02:06Z
stays with the ~02:0xZ tick, ~8 min past this tick's cap); tiny10k
10.35@5500 new run-best, the @5000 wobble resolved downward (21.6
f/min, step ~5,720, endpoint ~04:4x–05:0xZ). No steering, no new
reactions. Queue depth 1 (lit-pause reason stands). run_work_next
left unarmed — same reasoning as the last four ticks.
