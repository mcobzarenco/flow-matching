# Now



*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-10 03:12–03:1xZ (real `date -u` at write: 03:14) —
tick (babysit): **the uptick-resolution watch resolves — er_60k
**8.21@6500**, straight back down past the 8.77 run-best (the @6000
wobble behaved exactly like every tiny10k wobble did). Matched delta
**−0.73** vs the 40k's 8.9431 — the largest negative matched delta
since convergence, flattered by the 40k having its own uptick at
this exact rung. tiny10k **9.59@8000**, run-best by a hair; endpoint
~04:4xZ ≈90 min out.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~6,780, probe … 8.77@5500 → 8.98@6000 → **8.21@6500** (matched
deltas −0.57, +0.12, −0.38, −0.47, +0.44, −0.73), 25.4 st/min
window, util 90–99%, vram ~71.7 ×4 vs 77 bar, ~17.5/155 GPU-h;
endpoint ~08-11 ~12:00Z. `fontaine-tiny10k` LIVE local — step
~8,020, probe 9.60@7500 → **9.59@8000** (no F anchor at 8000; probe
rungs to @9500, then the @10000 primary read vs F 9.4157), 21.7
f/min, 7.0/15 GPU-h; endpoint ~04:4xZ (≈1,980 steps left).

**Steering**: none — `read` empty, history ×5 unchanged (lit-pause
exchange still the last owner message, no new reactions).

**Done**: babysit ×1 exit 0 (both green, no gate crossings; the two
new rungs above are the facts). Pulled the 40k@6500 anchor (8.9431)
from the postmortem transcription (`AR40K` in
`adamc_postmortem_chart.py`) for the matched delta. Queue validate
OK: depth 0 pickable WITH stated depth_reason (lit pause; 8 open = 2
live + 6 owner-gated/blocked). run_work_next left unarmed — no CPU
items open; the ~04:4xZ tick chain owns tiny10k post-processing
(panel_v2 → Δ_capacity read). Body + footer rolled per last-2 (02:29
block + 02:50 note → 08-10 archive).

**Next**: tiny10k endpoint ~04:4xZ → chained panel_v2 → Δ_capacity
read @10k (vs banked F@10k 9.4157) — with tiny −0.34 under F at the
matched 7500 rung, |Δ|≤0.3 "prior confirmed" vs "tiny wins" is a
live question. er_60k rungs record-only to endpoint ~08-11 ~12:00Z →
chained panel_v2 k4l2 + full ER-init convergence chart. No lit
refills until the owner re-enables.*

*Updated 2026-08-10 03:02–03:0xZ (real `date -u` at write: 03:02) —
tick (babysit): **quiet green tick between rungs — no new probe
points on either run. er_60k rides its first-uptick reading
(**8.98@6000**, @6500 eval imminent at step ~6,500 — the
does-it-resolve-downward question is next tick's fact); tiny10k
rides the matched-rung **9.60@7500** (−0.34 under F), step ~7,780,
@8000 next, endpoint ~04:4xZ ≈1.7 h out.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~6,500, probe … 8.77@5500 → **8.98@6000** (@6500 due now; matched
deltas −0.57, +0.12, −0.38, −0.47, +0.44), 27.3 st/min window, util
97–98%, vram ~71.6 ×4 vs 77 bar, ~16.8/155 GPU-h; endpoint ~08-11
~12:00Z. `fontaine-tiny10k` LIVE local — step ~7,780, probe
**9.60@7500** (Δ −0.34 vs F matched; next @8000, then the @10000
primary read vs F 9.4157), 21.8 f/min, 6.8/15 GPU-h; endpoint
~04:4xZ (≈2,220 steps left).

**Steering**: none — `read` empty, history ×5 unchanged (lit-pause
exchange still the last owner message, no new reactions).

**Done**: babysit ×1 exit 0 (both green, no gate crossings, no new
rungs — pure between-rungs tick). Queue validate OK: depth 0
pickable WITH stated depth_reason (lit pause; 8 open = 2 live + 6
owner-gated/blocked). run_work_next left unarmed — no CPU items
open, both runs mid-flight; the ~04:4xZ tick chain owns tiny10k
post-processing (panel_v2 → Δ_capacity read). Body + footer rolled
per last-2 (02:17 block + 02:29 note → 08-10 archive).

**Next**: tiny10k endpoint ~04:4xZ → chained panel_v2 → Δ_capacity
read @10k (vs banked F@10k 9.4157) — with tiny already −0.34 under F
at 7500, |Δ|≤0.3 "prior confirmed" vs "tiny wins" is a live
question. er_60k @6500 next tick: does the @6000 uptick resolve
downward like every tiny10k wobble did? Rungs record-only to
endpoint ~08-11 ~12:00Z → chained panel_v2 k4l2 + full ER-init
convergence chart. No lit refills until the owner re-enables.*

*Updated 2026-08-10 02:50–02:5xZ (real `date -u` at write: 02:52) —
tick (babysit): **green tick, one new rung each — and both are
story-rungs. er_60k **8.98@6000**: the run's FIRST uptick after
eleven consecutive descents (record-only, nowhere near any kill
line); vs the 40k's 8.5413 at the matched step → Δ +0.44, the first
clearly-positive matched delta — the converged-oscillating-around-
zero story strengthens. tiny10k **9.60@7500** — THE matched rung:
F@7500 is 9.9391, so tiny sits **−0.34 under F** at the
step-and-batch-matched point. Capacity is not visibly binding at
h256 yet; the primary Δ_capacity read @10k (~2 h out) decides.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~6,200, probe … 9.26@5000 → 8.77@5500 → **8.98@6000** (first
wobble; matched deltas now −0.57, +0.12, −0.38, −0.47, +0.44), 25.5
st/min window, util 83–100%, vram ~71.7 ×4 vs 77 bar, ~16.1/155
GPU-h; endpoint ~08-11 ~12:00Z. `fontaine-tiny10k` LIVE local —
step ~7,540, probe 9.73@7000 → **9.60@7500** (Δ −0.34 vs F matched;
next and final probe rungs → @10000 primary read vs F 9.4157), 21.9
f/min, 6.7/15 GPU-h; endpoint ~04:4xZ (≈2,460 steps left).

**Steering**: none — `read` empty, history ×5 unchanged (lit-pause
exchange still the last owner message, no new reactions).

**Done**: babysit ×1 exit 0 (both green, no gate crossings; the two
story-rungs above are the facts). Queue validate OK: depth 0
pickable WITH stated depth_reason (lit pause; 8 open = 2 live + 6
owner-gated/blocked). run_work_next left unarmed — no CPU items
open, both runs mid-flight; the ~04:4xZ tick chain owns tiny10k
post-processing (panel_v2 → Δ_capacity read). Body + footer rolled
per last-2 (01:49 block + 02:17 note → 08-10 archive).

**Next**: tiny10k endpoint ~04:4xZ → chained panel_v2 → Δ_capacity
read @10k (vs banked F@10k 9.4157) — with tiny already −0.34 under
F at 7500, |Δ|≤0.3 "prior confirmed" vs "tiny wins" is now a live
question. er_60k rungs record-only to endpoint ~08-11 ~12:00Z
(watch whether the @6000 uptick resolves downward like every
tiny10k wobble did) → chained panel_v2 k4l2 + full ER-init
convergence chart. No lit refills until the owner re-enables.*

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

Session 2026-08-10 03:02–03:0xZ (tick, babysit; 0 new GPU-h —
er_60k rides ~16.8/155, tiny10k 6.8/15): quiet green tick between
rungs — no new probe points on either run; er_60k step ~6,500 on
8.98@6000 (@6500 eval imminent — uptick-resolution watch), tiny10k
step ~7,780 on 9.60@7500, endpoint ~04:4xZ. No steering. Queue depth
0 pickable with stated reason (lit pause). run_work_next unarmed —
the ~04:4xZ tick chain owns tiny10k post-processing.

Session 2026-08-10 03:12–03:1xZ (tick, babysit; 0 new GPU-h —
er_60k rides ~17.5/155, tiny10k 7.0/15): green tick, both watch
questions resolve — er_60k 8.21@6500, the @6000 uptick resolved
downward into a new run-best (Δ −0.73 vs 40k's 8.9431 matched, the
largest negative delta since convergence, flattered by the 40k's own
uptick at this rung; record-only); tiny10k 9.59@8000 run-best by a
hair, endpoint ~04:4xZ ≈90 min out. No steering. Queue depth 0
pickable with stated reason (lit pause). run_work_next unarmed —
the ~04:4xZ tick chain owns tiny10k post-processing.
