# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-10 05:13–06:0xZ (real `date -u` at write: 05:56) —
work session (the armed post-processing chain): **T1 tiny-expert
rung CLOSED — Δ_capacity@10k = +0.188 [CI95 +0.155, +0.221], the
capacity prior CONFIRMED at the pre-registered |Δ| ≤ 0.3 band.**
Paired per-frame read on 15,056 panel-v2 core frames (the
`attach_seam_results.py` read-1 machinery at explicit paths, per the
pre-reg): pooled tiny **9.6094** vs F **9.4157**; the CI excludes
zero, so the width cost is *real but small* — +2.0%, concentrated
late-horizon (per-step Δ +0.106 → +0.374 across the 50-step chunk).
State-copy execution oracle **byte-green across machines** (box-F vs
local-tiny npz). Probe-vs-panel sign flip logged: the 256-frame
probe had tiny −0.069 UNDER F; the panel flips it to +0.188 over —
probes kill runs, panels make claims. Expert sizes measured off
safetensors headers: tiny **86.8M** vs F **367.5M** (4.2× total;
the identical tap/adapter surface is the fixed cost). Consequence:
expert sizing is now a cost knob, not a risk knob (#4 fjoint sizing,
#16 rig inference).
[Results post](posts/2026-08-10-tiny-expert-results.md) + 3-panel
chart; analysis + both panel html/json on the Space; step_010000
weights-only on fontaine-checkpoints (backbone deduplicated, sha
re-verified at upload); babysit entry pruned — **local GPU FREE
05:45Z**.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — step
~10,880, probe … 7.65@8000 → 7.92@9500 → 7.93@10000 → **7.95@10500**
(in-band plateau), 25.8 f/min window, vram ~71.7 ×4 vs 77 bar,
projection 27.9/155 GPU-h; endpoint ~08-11 ~12:00Z. Owed matched Δ
vs 40k (shared seed) now computed and banked record-only: **@9000
−0.44, @9500 +0.53, @10000 +0.77** — wobble in both directions
inside the run-to-run band; the early ER-init advantage stays washed
out, endpoint panel decides. `fontaine-tiny10k` CLOSED (above);
local GPU free.

**Steering**: none — `read` empty at 05:14 and 05:48 polls, history
×5 unchanged (lit-pause exchange still the last owner message).

**Done**: the full armed chain, this session: panel_v2 @10000
completed 05:45:18Z (~38 min, ~660 f/min, ~0.6 GPU-h → run total
~9.3/15 gate); F's box-side npz/json/html scp'd; Δ_capacity frozen
read run (F-vs-F dry-run Δ=0 first, then live); 3-panel dark chart
(`tiny_capacity_chart.py`, lint-green); results post + SUMMARY +
reports.md new "frozen-trunk flow experts @10k panel_v2" section
(F's panel pushed to the Space for the first time — it had been
box-only); ideas.md hook + #4 + #16 dated records; step_010000
weights-only upload verified on fontaine-checkpoints; er_60k
matched-Δ table computed from box logs; babysit ×2 exit 0→0 (tiny
pruned between); queue tiny item → done with full close-out
boundary; readout posted in-channel.

**Next**: `queue_cli.py next` → no open items (stated depth-0
reason: lit pause + all remaining items owner-gated; work supply is
run-boundary-driven). er_60k rides to endpoint ~08-11 ~12:00Z →
chained panel_v2 → paired CI95 vs banked 40k (6.0079) + 60k
(5.8602) panels. Local GPU free — next local launch needs a fresh
pre-reg (named candidates, NOT yet pre-registered: T2 depth rung;
tiny decode-cost microbench for #16). fjoint finalize waits on
owner go (~08-12, post-er-endpoint). run_work_next NOT armed —
CPU-side queue is empty after this close (charter §3: arming
requires queued CPU work).*

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


Session 2026-08-10 04:35–05:0xZ (tick, babysit; 0 new GPU-h logged —
er_60k rides 24.1/155 projection, tiny10k ~8.4/15): double-rung tick
— §6 hold caught tiny10k 9.5045@9500 (record-only, between the
9.56@9000 re-run and pre-kill 9.37 run-best) and er_60k 7.82@9000
(descent resumed off 8.29@8500, second-best of the run; matched Δ
vs 40k computes next tick — curve banked box-side only). tiny ~22.7
st/min steady, step ~9,640, endpoint ~05:1xZ imminent; host RAM
94/221 stable. Watcher lesson recorded: the er log path is box-only,
grep it via babysit/ssh, never locally. No steering. Queue depth 0
pickable with stated reason (lit pause). run_work_next unarmed — the
~05:1x–05:3xZ tick chain owns tiny10k endpoint + panel_v2 +
Δ_capacity read.

Session 2026-08-10 04:56–05:1xZ (tick, babysit; tiny10k train
COMPLETE at ~8.7/15 GPU-h incl. OOM replay; er_60k rides 24.4/155):
endpoint tick — §6 hold caught step 10,000 at 05:06Z in-session:
final probe 9.3469@10000 vs banked F@10k 9.4157 → probe-level
Δ_capacity −0.069, prior-confirmed band (|Δ|≤0.3); resumed path
converged back onto the pre-kill curve. Checkpoint step_010000
saved async; chained panel_v2 launched in-unit (pre-reg args
verbatim, sha-verified plan). F's panel npz confirmed box-side only
— path pinned in babysit.toml for the scp. Endpoint posted
in-channel 05:07Z. er_60k 7.92@9500 in-band wobble, record-only.
No steering. Queue depth 0 with stated reason (lit pause).
run_work_next ARMED — the chained work session owns panel readout →
paired CI95 → chart → post → ledger → checkpoint upload.

Session 2026-08-10 05:13–06:0xZ (work, exploit; +~0.6 GPU-h logged —
the tiny panel_v2 eval, closing the rung at ~9.3/15; er_60k rides
27.9/155): the armed post-processing chain executed end-to-end —
Δ_capacity@10k = +0.188 [+0.155, +0.221] paired on 15,056 core
frames (tiny 9.6094 vs F 9.4157) = capacity prior CONFIRMED at the
pre-registered band, width cost real-but-small (+2.0%, late-horizon);
state-copy oracle byte-green across machines; results post + chart +
reports section + ideas records landed; step_010000 uploaded
weights-only; er matched-Δ table banked (@9000 −0.44 / @9500 +0.53 /
@10000 +0.77, record-only); babysit tiny entry pruned, local GPU
free 05:45Z. No steering. Queue depth 0 open with stated reason
(lit pause + owner-gated tail); run_work_next NOT armed (no CPU
items remain).

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
