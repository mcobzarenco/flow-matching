# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-10 06:06–06:2xZ (real `date -u` at write: 06:16) —
tick (babysit): quiet single-run tick — er_60k **7.5922@11500**,
second-best of the run, holding right off the 7.54@11000 run-best:
the 7.92–7.95 plateau break is sustained, not a one-rung spike.
Matched Δ vs 40k (shared seed, box-side log) extends the
record-only table: **@11500 +0.39 (7.5922 vs 7.2014)** — full
table @9000→@11500: −0.44 / +0.53 / +0.77 / +0.80 / −0.43 /
+0.39. Wobble in both directions at the ~±0.8 rung scale;
endpoint panel (~08-11 ~12:00Z) decides. Caught the rung with a
short §6 hold (~7 min) rather than leaving it to the next tick.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~11,400 at poll, probe … 7.65@8000 → 7.95@10500 → 7.54@11000 →
**7.5922@11500**, 27.7 f/min window, vram ~71.7 ×4 vs 77 bar,
projection 29.2/155 GPU-h; endpoint ~08-11 ~12:00Z. Local GPU free
(next local launch needs a fresh pre-reg).

**Steering**: none — `read` empty, history ×5 all our own posts,
no new reactions (lit-pause exchange still the last owner
message).

**Done**: babysit ×1 exit 0 (liveness 8 procs, 4× GPU engaged,
util 96–100%, window 27.7 f/min healthy). §6 hold for the @11500
rung; watcher-pattern lesson: the first Monitor grep assumed
space-separated `step 11500` but the log is jsonl (`"step":
11500`) — pattern never matched; killed it and read the log
directly over ssh. @11500 matched-Δ leg banked record-only (no
post — in-band rung). Queue validate OK: depth 0 pickable with
stated reason (lit pause + owner-gated tail), 7 open.
run_work_next NOT armed — CPU-side queue empty, box busy, local
idle-by-design (charter §5 exit).

**Next**: er_60k rides to endpoint ~08-11 ~12:00Z → chained
panel_v2 → paired CI95 vs banked 40k (6.0079) + 60k-continuation
(5.8602) panels. Rungs record-only; @7500-class transient
recurrence upgrades to a posted fact; kill lines unchanged (NaN,
probe-vs-@2500 by 10k — PASSED, probe>25 ×3). Next local launch
owner-gated: named-not-preregistered candidates T2 depth rung +
tiny decode microbench (#16); fjoint finalize waits on owner go
(~08-12). No lit refills until re-enabled.*

*Updated 2026-08-10 05:56–06:0xZ (real `date -u` at write: 05:58) —
tick (babysit): **er_60k new run-best 7.54@11000** — first rung
under the 7.65@8000 mark since step 8000, breaking the
7.92–7.95 plateau (@9500→@10500). Matched Δ vs 40k (shared seed,
box-side curve) extends the record-only table: **@10500 +0.80
(7.95 vs 7.1514), @11000 −0.43 (7.54 vs 7.9665)** — full table
@9000→@11000: −0.44 / +0.53 / +0.77 / +0.80 / −0.43. Wobble in
both directions, both curves rung-noisy at the ~±0.8 scale;
ER-init advantage stays washed out, endpoint panel (~08-11
~12:00Z) decides. Single-run tick — tiny rung closed last
session, local GPU free.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~11,120, probe … 7.65@8000 → 7.92@9500 → 7.93@10000 → 7.95@10500 →
**7.54@11000** (run-best), 25.7 f/min window, vram ~71.7 ×4 vs 77
bar, projection 28.5/155 GPU-h; endpoint ~08-11 ~12:00Z. Local GPU
free (next local launch needs a fresh pre-reg).

**Steering**: none — `read` surfaced only our own 05:51Z results
post (cursor catch-up, no reply owed); history ×5 all our own
posts, no new reactions (lit-pause exchange still the last owner
message).

**Done**: babysit ×1 exit 0 (liveness 8 procs, 4× GPU engaged,
window 25.7 f/min vs cumulative healthy). @10500/@11000 matched-Δ
legs computed from the box-side 40k log over ssh and banked
(record-only, no post — in-band rungs, the 05:51Z results post
already carried the morning's story). Queue validate OK: depth 0
pickable with stated reason (lit pause + owner-gated tail), 7
open. run_work_next NOT armed — CPU-side queue empty, box busy,
local idle-by-design (charter §5 exit condition).

**Next**: er_60k rides to endpoint ~08-11 ~12:00Z → chained
panel_v2 → paired CI95 vs banked 40k (6.0079) + 60k-continuation
(5.8602) panels. Rungs record-only; @7500-class transient
recurrence upgrades to a posted fact; kill lines unchanged (NaN,
probe-vs-@2500 by 10k — PASSED at 7.93@10000 vs 12.5-class @2500,
probe>25 ×3). Next local launch owner-gated: named-not-preregistered
candidates T2 depth rung + tiny decode microbench (#16); fjoint
finalize waits on owner go (~08-12). No lit refills until
re-enabled.*

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
call — no endpoint, no chained evals)**; local tiny10k 08-09 20:1xZ
→ 08-10 05:06Z train COMPLETE **~8.7/15 GPU-h incl. OOM replay** +
chained panel_v2 eval COMPLETE 08-10 05:45Z (+~0.6 GPU-h, **~9.3/15
total, rung closed**)). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).


Session 2026-08-10 05:56–06:0xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides 28.5/155 projection, sole live run): quiet single-run
tick — er_60k **new run-best 7.54@11000**, breaking the 7.92–7.95
plateau; matched-Δ table vs 40k extended record-only from box logs
(@10500 +0.80, @11000 −0.43 — wobble both directions, endpoint
panel decides). No post (in-band rung). No steering (read surfaced
only our own 05:51Z post; history ×5 unchanged). Queue depth 0
pickable with stated reason (lit pause + owner-gated tail);
run_work_next NOT armed — CPU queue empty, local GPU
idle-by-design, plain §5 exit.

Session 2026-08-10 06:06–06:2xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides 29.2/155 projection, sole live run): quiet single-run
tick with a short §6 hold — er_60k **7.5922@11500** second-best,
holding off the 7.54@11000 run-best (plateau break sustained);
matched-Δ table vs 40k extended record-only (@11500 +0.39 vs
7.2014 — full table @9000→@11500 wobbles both directions, endpoint
panel decides). Watcher lesson: Monitor grep patterns must match
the jsonl log format (`"step": 11500`, not `step 11500`) — the
first watcher never fired; read the log directly over ssh instead.
No post (in-band rung). No steering (read empty, history ×5
unchanged). Queue depth 0 pickable with stated reason (lit pause +
owner-gated tail); run_work_next NOT armed — CPU queue empty,
local GPU idle-by-design, plain §5 exit.
