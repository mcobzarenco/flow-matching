# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-10 08:08–08:3xZ (real `date -u` at write: 08:28) —
tick (babysit): **SAVE BOUNDARY @15000 caught — er_60k 6.9230,
second sub-7** (just off the 6.8543@13000 run-best); nine straight
rungs (7.54 / 7.59 / 7.37 / 7.40 / 6.85 / 7.15 / 7.37 / 7.42 /
6.92) under the pre-plateau 7.65@8000 mark. Async save green:
`captured in 21.7s; gather+write continue in background`, util back
at 100% after the pause. Matched Δ vs 40k (shared seed, box-side
log) extends the record-only table: **@15000 +0.19 (6.9230 vs
6.7311)** — full table @9000→@15000: −0.44 / +0.53 / +0.77 / +0.80
/ −0.43 / +0.39 / −0.19 / −0.50 / −0.24 / +0.17 / +0.47 / +0.73 /
+0.19. Fourth positive leg in a row but back off the +0.73 upper
edge; running mean ≈ +0.17 on the ~±0.8 wobble — endpoint panel
(~08-11 ~12:00Z) decides. **Morning results post landed
in-channel 08:2xZ** (the pre-declared post moment — sub-7.65 band
held). Rung caught with a ~13-min §6 hold (ssh until-loop keyed on
the eval_chunk_mae jsonl line — fired first try; the corrected
~08:2xZ ETA was right).*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
15,000 saved + posted, probe … 7.65@8000 → 7.95@10500 → 7.54@11000
→ 7.59@11500 → 7.37@12000 → 7.40@12500 → 6.8543@13000 →
7.1503@13500 → 7.3734@14000 → 7.4229@14500 → **6.9230@15000**,
27.0 f/min window, vram ~71.7 ×4 vs 77 bar, ~38/155 GPU-h;
endpoint ~08-11 ~12:00Z. Local GPU free (next local launch needs a
fresh pre-reg).

**Steering**: none — `read` empty, history ×5 all our own posts,
no new reactions (lit-pause exchange still the last owner
message).

**Done**: babysit exit 0 (liveness 8 procs, 4× GPU engaged, util
94–100%, window 27.0 f/min healthy). §6 hold ~13 min for the
@15000 boundary; baseline identity re-verified (ar_40k
@13000–@14500 all match banked legs) and the @15000 leg banked;
save-boundary fact captured (21.7 s async capture, util 100%
after). **Posted** the morning results post (ladder + full Δ table
+ save fact + health + endpoint plan). babysit.toml rung-state
block refreshed (@15000 boundary + @20000 ETA ~11:3xZ). Queue
validate OK: depth 0 pickable with stated reason (lit pause +
owner-gated tail), 7 open. run_work_next NOT armed — CPU-side
queue empty, box busy, local idle-by-design (charter §5 exit).

**Next**: er_60k rides to endpoint ~08-11 ~12:00Z → chained
panel_v2 → paired CI95 vs banked 40k (6.0079) + 60k-continuation
(5.8602) panels. Rungs record-only; @7500-class transient
recurrence upgrades to a posted fact; kill lines unchanged (NaN,
probe-vs-@2500 by 10k — PASSED, probe>25 ×3). Next save boundary
@20000 ~11:3xZ (rungs every ~18.5 min stay in-band unless the Δ
table breaks ±0.8). Next local launch owner-gated:
named-not-preregistered candidates T2 depth rung + tiny decode
microbench (#16); fjoint finalize waits on owner go (~08-12). No
lit refills until re-enabled.*

*Updated 2026-08-10 07:50–08:1xZ (real `date -u` at write: 08:05) —
tick (babysit): **er_60k 7.4229@14500** — fourth rung off the
6.8543@13000 run-best (first sub-7); eight straight rungs (7.54 /
7.59 / 7.37 / 7.40 / 6.85 / 7.15 / 7.37 / 7.42) now sit under the
pre-plateau 7.65@8000 mark. Matched Δ vs 40k (shared seed,
box-side log) extends the record-only table: **@14500 +0.73
(7.4229 vs 6.6921)** — full table @9000→@14500: −0.44 / +0.53 /
+0.77 / +0.80 / −0.43 / +0.39 / −0.19 / −0.50 / −0.24 / +0.17 /
+0.47 / +0.73. Third positive leg in a row, now at the upper edge
of the ~±0.8 wobble scale (the 40k baseline hit a fast patch,
6.90→6.69); endpoint panel (~08-11 ~12:00Z) decides. Rung caught
with a ~13-min §6 hold (ssh until-loop keyed on the
eval_chunk_mae jsonl line — fired first try). **ETA correction**:
at the measured ~26.6 st/min the @15000 save boundary lands
**~08:2xZ**, not ~09:0xZ as the last three notes projected — an
arithmetic slip, now fixed; the next tick catches it.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
14,500 at rung, probe … 7.65@8000 → 7.95@10500 → 7.54@11000 →
7.59@11500 → 7.37@12000 → 7.40@12500 → 6.8543@13000 →
7.1503@13500 → 7.3734@14000 → **7.4229@14500**, 26.6 f/min
window, vram ~71.7 ×4 vs 77 bar, projection ~36.9/155 GPU-h;
endpoint ~08-11 ~12:00Z. Local GPU free (next local launch needs
a fresh pre-reg).

**Steering**: none — `read` empty ×2, history ×5 all our own
posts, no new reactions (lit-pause exchange still the last owner
message).

**Done**: babysit exit 0 (liveness 8 procs, 4× GPU engaged, util
62–100%, window 26.6 f/min healthy; post-rung snapshot 62–100%
util, vram steady). §6 hold ~13 min for the @14500 rung; @14500
matched-Δ leg banked record-only vs the verified ar_40k box log
(baseline identity re-checked @13000–@14000, all match banked
legs; no post — in-band rung, the @15000 boundary is the post
moment). @15000 ETA corrected ~09:0xZ → ~08:2xZ in babysit.toml +
here. Queue validate OK: depth 0 pickable with stated reason (lit
pause + owner-gated tail), 7 open. run_work_next NOT armed —
CPU-side queue empty, box busy, local idle-by-design (charter §5
exit).

**Next**: er_60k rides to endpoint ~08-11 ~12:00Z → chained
panel_v2 → paired CI95 vs banked 40k (6.0079) + 60k-continuation
(5.8602) panels. Rungs record-only; @7500-class transient
recurrence upgrades to a posted fact; kill lines unchanged (NaN,
probe-vs-@2500 by 10k — PASSED, probe>25 ×3). **Save boundary
@15000 lands ~08:2xZ (next tick)** — the natural moment for a
morning results post if the sub-7.65 band holds. Next local
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




Session 2026-08-10 07:50–08:1xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides ~36.9/155 projection, sole live run): single-run tick
with a ~13-min §6 hold — er_60k **7.4229@14500**, fourth rung off
the 6.8543@13000 run-best (first sub-7); eight straight rungs
(7.54/7.59/7.37/7.40/6.85/7.15/7.37/7.42) under the pre-plateau
7.65@8000 mark. Matched-Δ table vs 40k extended record-only (@14500
+0.73, 7.4229 vs 6.6921 — third positive leg in a row, upper edge
of the ~±0.8 wobble; the 40k baseline hit a fast patch 6.90→6.69;
endpoint panel decides). Baseline log identity re-verified (ar_40k
@13000–@14000 values match all banked legs). Watcher: single
until-loop keyed on the eval_chunk_mae jsonl line fired first try.
ETA correction: @15000 save boundary lands ~08:2xZ at the measured
~26.6 st/min, not ~09:0xZ as the last three notes projected — next
tick catches it; that boundary stays the morning-post moment if
the sub-7.65 band holds. No post (in-band rung). No steering (read
empty ×2, history ×5 unchanged). Queue depth 0 pickable with
stated reason (lit pause + owner-gated tail); run_work_next NOT
armed — CPU queue empty, local GPU idle-by-design, plain §5 exit.

Session 2026-08-10 08:08–08:3xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides ~38/155, sole live run): single-run tick with a ~13-min
§6 hold — **SAVE BOUNDARY @15000 caught: er_60k 6.9230, second
sub-7** (just off the 6.8543@13000 run-best); nine straight rungs
(7.54/7.59/7.37/7.40/6.85/7.15/7.37/7.42/6.92) under the
pre-plateau 7.65@8000 mark. Async save green (captured 21.7 s,
util 100% after the pause). Matched-Δ table vs 40k extended
record-only (@15000 +0.19, 6.9230 vs 6.7311 — fourth positive leg
in a row but back off the +0.73 upper edge; running mean ≈ +0.17
on the ~±0.8 wobble; endpoint panel decides). Baseline log
identity re-verified (ar_40k @13000–@14500 match all banked legs).
Watcher: single until-loop keyed on the eval_chunk_mae jsonl line
fired first try — the corrected ~08:2xZ ETA was right. **Morning
results post landed in-channel 08:2xZ** (pre-declared post moment,
sub-7.65 band held). Next save boundary @20000 ~11:3xZ. No
steering (read empty, history ×5 unchanged). Queue depth 0
pickable with stated reason (lit pause + owner-gated tail);
run_work_next NOT armed — CPU queue empty, local GPU
idle-by-design, plain §5 exit.
