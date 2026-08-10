# Now


*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-10 00:13–00:2xZ (real `date -u` at write: 00:14) —
tick (babysit): **green tick, no steering — er_60k probe
**15.43@2000**, fourth consecutive descent; both runs ride. The
0821 work session's in-session hold did NOT survive to the
boundary (turn ended; `run_work_next` was unarmed) — re-armed it,
so the chained work session owns the ~02:0xZ step-5000 read.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~2,100, probe 33.03@500 → 22.05@1000 → 16.78@1500 → **15.43@2000**,
28.2 st/min window, util 56–100%, vram ~71.5 ×4, 5.6/155 GPU-h;
step-5000 boundary ~02:0xZ. `fontaine-tiny10k` LIVE local — step
~4,080, probe 11.56@4000 (plateau forming ~11.5 vs F@5000 10.26),
18.8 f/min, 4.0/15 GPU-h; endpoint ~05:1xZ 08-10.

**Steering**: none — `read` empty, history ×5 only our own posts,
no new reactions; the ~150 GPU-h correction remains unobjected →
er_60k rides.

**Done**: babysit ×1 exit 0 (both runs green, no gate crossings).
Queue validate green depth 2 (10 open). run_work_next found
unarmed → re-armed (the 0821 session consumed it and its hold
died at turn end). Body + footer rolled per last-2 (23:27 block +
23:27/23:51 notes → archive).

**Next**: chained work session → er_60k step-5000 boundary ~02:0xZ
(async-save capture line + `er60k_init_delta_chart.py` → chart +
facts in-channel, er60k-init-delta-midrun-chart item), then
lit-radar-0822 (cpu, GPU-busy window). tiny10k endpoint ~05:1xZ
08-10 → chained panel_v2 → Δ_capacity read. er_60k endpoint ~08-11
~12:00Z → chained panel_v2 k4l2.*

*Updated 2026-08-09/10 23:55–02:xxZ (real `date -u` at write: 00:11) —
work session (bounded): **lit-radar-0821 CLOSED — 4 Papers pages
landed + wired via a 5-agent fan-out; find of the slice:
NeuralActuator's third platform IS the SO-101, everything released
(3 SO-101 checkpoints + teleop code), the rig-day force-sensing
rider is now shovel-ready. Session stays live for the er_60k
step-5000 boundary (~02:0xZ): ER-init delta chart + facts
in-channel.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~2,040, probe 33.03@500 → 22.05@1000 → 16.78@1500 → **15.43@2000**
descending, ~27 st/min, util 53–94%, vram ~71.5 ×4, 5.4/155 GPU-h;
step-5000 boundary ~02:0xZ. `fontaine-tiny10k` LIVE local — step
~4,040, probe 11.56@4000 (plateau forming ~11.5 vs F@5000 10.26),
4.0/15 GPU-h; endpoint ~05:1xZ 08-10.

**Steering**: none — `read` surfaced only my own slice post; the
~150 GPU-h correction remains unobjected → er_60k rides.

**Done**: lit-radar-0821 executed end-to-end (7951bac): 4 Papers
pages same-session per the permanent rule —
[Quality over Quantity](papers/quality-over-quantity.md) (offline
influence pole, runnable no-rollout; gains only on 40–50% injected
failures, hard top-N not weighting; cheapest #9 arm sketched),
[Curse of Precision](papers/curse-of-precision.md) (sim-only R²>0.97
fit, worst points 23–65× extrapolated; hook's "not the task"
corrected — c moved 2.35→1.00 mm by randomization alone; #16
tolerance-dial + Δc design rule; #9 clarity-filter lever),
[NeuralActuator](papers/neuralactuator.md) (SO-101 IS the third
platform: force 0.47–0.73 N MAE from load registers, no current
sensor, MIT everything; "torque-from-current" hook wrong twice; #16
rider superseded, #9 Δq_d gate stands),
[GigaWorld-1 / WMBench](papers/gigaworld-wmbench.md) (324K
"rollouts" = graded videos under replayed actions, real-ranking
never computed; Apache-2.0 release kills the 0820 "no artifact"
objection; screen ≠ certificate stands). Ideas #9/#16 pages + index
hooks fed; Radar 0821 flipped; 0822 queued (18 checked, 15
abs-verified, 12 survived; 3 dups already-read — sweep converging).
check.py 599 green ×2; Space pushed, 4 pages curl-200; slice
summary in-channel. Babysit ×2 exit 0.

**Next**: er_60k step-5000 boundary ~02:0xZ THIS SESSION →
async-save capture line + `er60k_init_delta_chart.py` → chart +
facts in-channel (er60k-init-delta-midrun-chart item). tiny10k
endpoint ~05:1xZ 08-10 → chained panel_v2 → Δ_capacity read. er_60k
endpoint ~08-11 ~12:00Z → chained panel_v2 k4l2. `queue_cli.py
next` after the boundary → lit-radar-0822 (cpu, GPU-busy window).*

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

Session 2026-08-10 00:1x–00:3xZ (work, chained; 0 new GPU-h;
note written retroactively at the 00:33 tick — the session ended
without rolling now.md): lit-radar-0822 CLOSED as the FINAL slice
before the owner pause (e887451). Steering 00:23:47Z "Can we pause
the lit slices for now" caught mid-flight (fan-out already reading
~00:20Z), acked in-channel 00:28Z; the 4 finished pages landed
quietly, no summary post, shelve-entirely offer open: Ambient
Diffusion Policy (flow-time band-mask lever, ports to rectified
flow in sigma-space, composes with QoQ), the curation-metrics pair
(detection/policy DECOUPLED 0.804→13.3%, length confound →
rank-by-length null arm, velocity census demoted to coverage-only),
PhAIL (KM/RMST/macro-KS resolves 2/3 pairs at 25–30 ep/cell, human
anchor zero statistical power). Ideas #9/#16/#15 wired; NO 0823
queued (allocation suspended, spares recorded in the closed item);
check.py 599 green.

Session 2026-08-10 00:33–00:4xZ (tick, babysit; 0 new GPU-h —
er_60k rides 6.9/155, tiny10k 4.4/15): green tick — er_60k
11.43@2500 fifth consecutive descent, 23.5 st/min; tiny10k
10.88@4500 breaks the ~11.5 plateau, next matched rung F@5000
10.26. No new steering (read empty, no new reactions); the 00:23Z
lit pause recorded in the body retroactively. Boot audit fixed the
0822 session's hallucinated clocks in queue.json (01:05Z stamps →
real ~00:35Z) and committed its orphaned queue.md regen. Queue
depth 1 with stated reason (lit pause, run-boundary-driven supply).
run_work_next LEFT unarmed by decision — the ~02:0xZ tick owns the
step-5000 ER-init delta read (in-tick, or arm-on-overrun).
