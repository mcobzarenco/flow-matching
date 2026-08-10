# Now

*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-10 06:54–07:1xZ (real `date -u` at write: 07:08) —
tick (babysit): **er_60k NEW RUN-BEST 6.8543@13000 — first sub-7**,
beats 7.3694@12000 by ~0.5; five straight rungs (7.54 / 7.59 /
7.37 / 7.40 / 6.85) now sit under the pre-plateau 7.65@8000 mark.
Matched Δ vs 40k (shared seed, box-side log) extends the
record-only table: **@13000 −0.24 (6.8543 vs 7.0920)** — full
table @9000→@13000: −0.44 / +0.53 / +0.77 / +0.80 / −0.43 / +0.39
/ −0.19 / −0.50 / −0.24. Three consecutive negative legs now, but
each inside the ~±0.8 rung wobble; endpoint panel (~08-11
~12:00Z) decides. Rung caught with a ~14-min §6 hold (ssh
until-loop, jsonl grep — train line first, then a second wait for
the eval line).*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
13,000 at rung, probe … 7.65@8000 → 7.95@10500 → 7.54@11000 →
7.59@11500 → 7.37@12000 → 7.40@12500 → **6.8543@13000**
(run-best, first sub-7), 26.2 f/min window, vram ~71.7 ×4 vs 77
bar, s_per_step 2.155, projection ~32.3/155 GPU-h; endpoint
~08-11 ~12:00Z. Local GPU free (next local launch needs a fresh
pre-reg).

**Steering**: none — `read` empty, history ×5 all our own posts,
no new reactions (lit-pause exchange still the last owner
message).

**Done**: babysit ×1 exit 0 (liveness 8 procs, 4× GPU engaged,
util 55–92%, window 26.2 f/min healthy). §6 hold ~14 min for the
@13000 rung; @13000 matched-Δ leg banked record-only vs the
verified ar_40k box log (baseline identity re-checked
@11000–@12500, all match banked legs; no post — in-band rung,
run-best but not a posted-fact class). Queue validate OK: depth 0
pickable with stated reason (lit pause + owner-gated tail), 7
open. run_work_next NOT armed — CPU-side queue empty, box busy,
local idle-by-design (charter §5 exit).

**Next**: er_60k rides to endpoint ~08-11 ~12:00Z → chained
panel_v2 → paired CI95 vs banked 40k (6.0079) + 60k-continuation
(5.8602) panels. Rungs record-only; @7500-class transient
recurrence upgrades to a posted fact; kill lines unchanged (NaN,
probe-vs-@2500 by 10k — PASSED, probe>25 ×3). Save boundary
@15000 is the next structural event (~08-10 ~09:0xZ) — a good
moment for a morning results post if the sub-7 trend holds. Next
local launch owner-gated: named-not-preregistered candidates T2
depth rung + tiny decode microbench (#16); fjoint finalize waits
on owner go (~08-12). No lit refills until re-enabled.*





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-10 06:35–06:5xZ (real `date -u` at write: 06:53) —
tick (babysit): er_60k **7.3977@12500** — just off the 7.3694@12000
run-best; four straight rungs (7.54 / 7.59 / 7.37 / 7.40) now sit
under the pre-plateau 7.65@8000 mark, so the break holds at depth.
Matched Δ vs 40k (shared seed, box-side log) extends the
record-only table: **@12500 −0.50 (7.3977 vs 7.8968)** — full
table @9000→@12500: −0.44 / +0.53 / +0.77 / +0.80 / −0.43 / +0.39
/ −0.19 / −0.50. Two consecutive negative legs, but still inside
the ~±0.8 rung wobble; endpoint panel (~08-11 ~12:00Z) decides.
Rung caught with a ~14-min §6 hold (ssh until-loop, jsonl-format
grep — fired first try).*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
12,500 at rung, probe … 7.65@8000 → 7.95@10500 → 7.54@11000 →
7.59@11500 → 7.37@12000 → **7.3977@12500**, 26.9 f/min window,
vram ~71.7 ×4 vs 77 bar, projection 31.0/155 GPU-h; endpoint
~08-11 ~12:00Z. Local GPU free (next local launch needs a fresh
pre-reg).

**Steering**: none — `read` empty, history ×5 all our own posts,
no new reactions (lit-pause exchange still the last owner
message).

**Done**: babysit ×1 exit 0 (liveness 8 procs, 4× GPU engaged,
util 68–99%, window 26.9 f/min healthy). §6 hold ~14 min for the
@12500 rung; @12500 matched-Δ leg banked record-only vs the
verified ar_40k box log (no post — in-band rung). Queue validate
OK: depth 0 pickable with stated reason (lit pause + owner-gated
tail), 7 open. run_work_next NOT armed — CPU-side queue empty,
box busy, local idle-by-design (charter §5 exit).

**Next**: er_60k rides to endpoint ~08-11 ~12:00Z → chained
panel_v2 → paired CI95 vs banked 40k (6.0079) + 60k-continuation
(5.8602) panels. Rungs record-only; @7500-class transient
recurrence upgrades to a posted fact; kill lines unchanged (NaN,
probe-vs-@2500 by 10k — PASSED, probe>25 ×3). Save boundary
@15000 is the next structural event (~08-10 ~09:0xZ). Next local
launch owner-gated: named-not-preregistered candidates T2 depth
rung + tiny decode microbench (#16); fjoint finalize waits on
owner go (~08-12). No lit refills until re-enabled.*

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


Session 2026-08-10 06:35–06:5xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides 31.0/155 projection, sole live run): single-run tick
with a ~14-min §6 hold — er_60k **7.3977@12500**, just off the
7.3694@12000 run-best; four straight rungs (7.54/7.59/7.37/7.40)
under the pre-plateau 7.65@8000 mark, the break holds at depth.
Matched-Δ table vs 40k extended record-only (@12500 −0.50, 7.3977
vs 7.8968 — two straight negative legs, still inside the ~±0.8
wobble; endpoint panel decides). Baseline log identity re-verified
(ar_40k @11000/@11500/@12000 values match all banked legs). No
post (in-band rung). No steering (read empty, history ×5
unchanged). Queue depth 0 pickable with stated reason (lit pause +
owner-gated tail); run_work_next NOT armed — CPU queue empty,
local GPU idle-by-design, plain §5 exit. Next structural event:
save boundary @15000 ~09:0xZ.

Session 2026-08-10 06:54–07:1xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides ~32.3/155 projection, sole live run): single-run tick
with a ~14-min §6 hold — er_60k **NEW RUN-BEST 6.8543@13000, first
sub-7** (beats 7.3694@12000 by ~0.5; five straight rungs
7.54/7.59/7.37/7.40/6.85 under the pre-plateau 7.65@8000 mark).
Matched-Δ table vs 40k extended record-only (@13000 −0.24, 6.8543
vs 7.0920 — three straight negative legs, each inside the ~±0.8
wobble; endpoint panel decides). Baseline log identity re-verified
(ar_40k @11000–@12500 values match all banked legs). Watcher note:
the jsonl grep on `"step": 13000` matches the *train* line first —
a second until-loop keyed on eval_chunk_mae caught the probe line.
No post (in-band rung; @15000 save boundary ~09:0xZ is the natural
morning-post moment if sub-7 holds). No steering (read empty,
history ×5 unchanged). Queue depth 0 pickable with stated reason
(lit pause + owner-gated tail); run_work_next NOT armed — CPU
queue empty, local GPU idle-by-design, plain §5 exit.
