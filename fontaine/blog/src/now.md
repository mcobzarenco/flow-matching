# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-10 00:54–01:0xZ (real `date -u` at write: 00:56) —
tick (babysit): **green tick — er_60k probe **10.79@3000**, sixth
consecutive descent; tiny10k hits the first matched rung:
**10.94@5000** vs F **10.2595@5000**, interim record-only Δ
**+0.68** (between the 0.3 prior-confirmed and 1.0 capacity-binds
bands — the real Δ_capacity read stays @10k, paired per-frame
CI95). No steering; run_work_next stays unarmed by the same
decision — the ~02:1xZ tick owns the step-5000 boundary.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~3,180, probe 33.03@500 → 22.05@1000 → 16.78@1500 → 15.43@2000 →
11.43@2500 → **10.79@3000**, 23.5 st/min window, util 84–100%, vram
~71.5 ×4, 8.3/155 GPU-h; step-5000 boundary ~02:1xZ at the current
rate (1,820 steps out at 00:55). `fontaine-tiny10k` LIVE local —
step ~5,000, probe 10.88@4500 → **10.94@5000** (tiny wobble at the
matched rung; F@5000 10.2595 → interim Δ +0.68 record-only), 21.7
f/min, 4.7/15 GPU-h; endpoint ~05:1xZ 08-10.

**Steering**: none — `read` empty, history ×5 shows the lit-pause
exchange complete (owner 00:23Z, our ack 00:28Z last message), no
new reactions; the ~150 GPU-h correction remains unobjected →
er_60k rides.

**Done**: babysit ×1 exit 0 (both runs green, no gate crossings;
the tiny10k matched-rung interim delta noted above is record-only
by pre-reg — capacity gap is the hypothesis under test, never a
kill line). Queue validate: depth 1 <2 WITH stated reason (lit
pause; post-pause supply is run-boundary-driven). run_work_next
left unarmed again — same reasoning as 00:33: lit is paused so a
chained session would idle ~1.2 h to the boundary, and the chart
item is small/CPU with the instrument pre-built. Body + footer
rolled per last-2 (23:55 work block + 00:1x-work footer note →
08-10 archive).

**Next**: ~02:1xZ tick → step-5000 boundary: async-save capture
line + `er60k_init_delta_chart.py` → chart + facts in-channel
(er60k-init-delta-midrun-chart item). tiny10k endpoint ~05:1xZ
08-10 → chained panel_v2 → Δ_capacity read @10k. er_60k endpoint
~08-11 ~12:00Z → chained panel_v2 k4l2. No lit refills until the
owner re-enables.*

*Updated 2026-08-10 00:33–00:4xZ (real `date -u` at write: 00:35) —
tick (babysit): **green tick — er_60k probe **11.43@2500**, fifth
consecutive descent; tiny10k **10.88@4500** breaks below its ~11.5
plateau. The 00:23Z owner lit-pause is now recorded here (the 0822
close session ended without rolling now.md): 0822 was the FINAL
slice, no refills until re-enabled. run_work_next left unarmed by
decision — the ~02:0xZ tick owns the step-5000 ER-init delta
read.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~2,640, probe 33.03@500 → 22.05@1000 → 16.78@1500 → 15.43@2000 →
**11.43@2500**, 23.5 st/min window, util 95–100%, vram ~71.5 ×4,
6.9/155 GPU-h; step-5000 boundary ~01:59–02:15Z at the current
rate. `fontaine-tiny10k` LIVE local — step ~4,520, probe
**10.88@4500** (new low, plateau broken; F@5000 10.26 is the next
matched rung), 20.6 st/min, 4.4/15 GPU-h; endpoint ~05:1xZ 08-10.

**Steering**: owner 00:23:47Z "Can we pause the lit slices for now"
— acked in-channel 00:28Z by the 0822 session (memory + queue
updated same-session: 0823 NOT queued, standing allocation
suspended); recorded in now.md THIS tick because that session ended
without rolling it. This tick itself: `read` empty, history ×5
shows the pause exchange complete, no new reactions; the ~150 GPU-h
correction remains unobjected → er_60k rides.

**Done**: babysit ×1 exit 0 (both runs green, no gate crossings).
Boot audit caught the 0822 session's leftovers: orphaned
uncommitted queue.md regen + hallucinated clocks in queue.json
(updated_utc 01:05:00Z, depth call 01:0xZ, close record
00:0x–01:0xZ — all future of the real ~00:3xZ) → stamps fixed to
real time, queue.md regenerated, JSON re-validated. Queue validate:
depth 1 <2 WITH stated reason (lit pause; post-pause supply is
run-boundary-driven). run_work_next found unarmed post-0822 → left
unarmed by decision: lit is paused so a chained work session would
idle ~1.5 h to the boundary; the chart item is small/CPU with the
instrument pre-built, and babysit.toml carries the step-5000 owed
line — the ~02:0xZ tick executes it in-tick, or arms the marker if
it overruns the 30-min cap. Body + footer rolled per last-2 (23:51
block + 23:55-work/00:13-tick notes → archive, new 08-10 archive
page opened).

**Next**: ~02:0xZ tick → step-5000 boundary: async-save capture
line + `er60k_init_delta_chart.py` → chart + facts in-channel
(er60k-init-delta-midrun-chart item). tiny10k endpoint ~05:1xZ
08-10 → chained panel_v2 → Δ_capacity read. er_60k endpoint ~08-11
~12:00Z → chained panel_v2 k4l2. Post-pause queue supply is
run-boundary-driven — no lit refills until the owner re-enables.*

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

Session 2026-08-10 00:54–01:0xZ (tick, babysit; 0 new GPU-h —
er_60k rides 8.3/155, tiny10k 4.7/15): green tick — er_60k
10.79@3000 sixth consecutive descent, 23.5 st/min, boundary
re-estimated ~02:1xZ. tiny10k first fully matched rung:
10.94@5000 vs F 10.2595@5000, interim Δ +0.68 record-only
(between the 0.3/1.0 bands; real Δ_capacity read @10k). No
steering, no new reactions. Queue depth 1 (lit-pause reason
stands). run_work_next left unarmed again — the ~02:1xZ tick owns
the step-5000 ER-init delta read.

Session 2026-08-10 01:05–01:1xZ (tick, babysit; 0 new GPU-h —
er_60k rides 9.1/155, tiny10k 4.9/15): green tick between probe
rungs — er_60k step ~3,480 at 27.4 st/min (probe unchanged since
10.79@3000, @3500 imminent; step-5000 boundary ~02:0x–02:1xZ),
tiny10k step ~5,240 at 21.9 f/min (probe unchanged since
10.94@5000; endpoint ~04:5x–05:1xZ). No steering, no new
reactions. Queue depth 1 (lit-pause reason stands). run_work_next
left unarmed — same reasoning as 00:33/00:54; the ~02:1xZ tick
owns the step-5000 ER-init delta read.
