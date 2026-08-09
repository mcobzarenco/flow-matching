# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 12:12–12:2xZ (real `date -u`) — tick (babysit):
**attach_K healthy past the run's midpoint approach — probe margin
~1.4 held, all quiet; queue armed for the next lit slice.***

**Status**: attach_K healthy at the 12:13Z poll — step 3800/10k,
loss 3.10, 3.78 s/step (13.1 steps/min window; endpoint ~18:3xZ
holds), vram 59.07 ≤ 71, liveness 7 procs / 4 GPUs. Probe
**11.2033@3500** (best); first kill-bar 12.6394 binds ≥5k (~13:2xZ)
with ~1.4 margin. CE aux flat. Local GPU free.

**Steering**: none — `read` clean; `history` shows nothing from the
owner after the answered 11:43:03Z loss_action question and no new
reactions on our 11:48Z answer or the 12:12Z session post.

**Done**: babysit poll (exit 0, facts above — trajectories nominal,
no anomaly beyond the CLI facts: loss stepping down 3.21 → 3.10,
probe monotone-improving since 2500); queue validate green (depth 2,
8 open); `run_work_next` armed (chained work session takes
`lit-radar-hooks-0809b` — QDepth-VLA + fresh sweep; banked radar
backlog is empty).

**Next**: 5k kill-bar binds ~13:2xZ (probe must be < 12.6394 —
currently 11.20; next tick catches the crossing); endpoint ~18:3xZ
→ chained panel_v2 + AR-view drift panel → **Δ_seam frozen read
(runbook staged, pre-audited)** → stage-2 decision.

*Updated 2026-08-09 11:56–12:1xZ (real `date -u`) — work session
(bounded, chained via `run_work_next`): **the async-execution radar
cluster cleared and then some — FIVE papers read, THREE papers pages
landed same session (async II cluster + Spatial Forcing + RDT2); #22's
arm menu re-ranked around FASTER's "the delay is a scheduling
artifact" result, and RDT2 files a production-scale F-shape vote
hours before tonight's Δ_seam read.***

**Status**: attach_K healthy at the 11:57Z + 12:08Z polls — step
3740/10k, loss 3.21, 3.74 s/step (endpoint ~18:3xZ holds), vram
59.07 ≤ 71, liveness 7 procs / 4 GPUs. Probe **11.2033@3500** (new
best); first kill-bar 12.6394 binds ≥5k (~13:2xZ) with ~1.4 margin.
CE aux flat. Local GPU free.

**Steering**: none — `read` clean at boot and at the 12:08Z babysit;
no new owner messages after the answered 11:43:03Z loss_action
question, no new reactions.

**Done**: **`lit-radar-async-exec` EXECUTED, both ride-along clauses
fired** (the cluster closed early, so Spatial Forcing AND RDT2 rode
per the item's own text): (1)
[async execution II](papers/async-execution-2.md) — FASTER
2603.19199 (TTFA theory + horizon-aware schedule: first action in 1
flow step of N, streams while the tail refines, 1.29–3.09×; tiles
across our draws-major batch, so the 18-tick mean-of-10 staleness
may be a scheduling artifact), ABPolicy 2602.23901 (B-spline
control-point flow + continuity refitting; jerk instruments banked),
DEFLECT 2605.19294 (stale-vs-fresh FM-DPO where RTC/BID measure ≤5%
at d≥5; carried at its restart-corrected +1.6–2.3 pp, not the +6.4
headline) → **#22 arm order: measure naive-switch → HAS-on-decode →
PAINT → A2C2 → TT-RTC/DEFLECT**; d≈18 untested by anyone stays
loud. (2) [Spatial Forcing](papers/spatial-forcing.md) 2510.12276 —
teacher×depth interact (VGGT works at LLM-24, collapsed at encoder
in VEGA); the 3.8× is a **fewer-steps** lever (≈50k vs 150k iters,
+25.8 pp at 5% data), a new column in the throughput accounting;
teacher overhead unreported. (3)
[RDT2](papers/rdt2-umi-scaling.md) 2602.03310 — 10k h robot-free
UMI data, zero-shot cross-embodiment; recipe = AR-first +
**frozen-trunk** flow expert + 1-step distill, no joint stage —
F-pole ledger context for tonight's decision (frozen read
untouched); #16 β≈0.23 data exponent; #5 RVQ ~⅓ tokens of FAST;
#12 second production 1-NFE point. Ideas #4/#5/#11/#12/#16/#17/#22
records + index hooks updated; papers index/SUMMARY rows. Queue:
item closed, refill `lit-radar-hooks-0809b` (QDepth-VLA + fresh
sweep — the banked radar backlog is now EMPTY), validate green
depth 2.

**Next**: 5k kill-bar binds ~13:2xZ (probe must be < 12.6394 —
currently 11.20); endpoint ~18:3xZ → chained panel_v2 + AR-view
drift panel → **Δ_seam frozen read (runbook staged, pre-audited)**
→ stage-2 decision. `queue_cli.py next` → `lit-radar-hooks-0809b`
(any GPU-busy window).

*Updated 2026-08-09 11:49–12:0xZ (real `date -u`) — tick (babysit):
**attach_K healthy at mid-run — probe margin ~1.0 held going into
the 5k bar window; plus a future-dated queue stamp and 89 broken
archive links caught and fixed.***

**Status**: attach_K healthy at the 11:50Z poll — step 3460/10k,
loss 3.18, 3.799 s/step (15.7 steps/min window; endpoint ~18:3xZ
holds), vram 59.07 ≤ 71, liveness 7 procs / 4 GPUs. Probe
11.6124@3000 (best); first kill-bar 12.6394 binds ≥5k (~13:2xZ)
with ~1.0 margin. CE aux flat. Local GPU free.

**Steering**: none new — `read` surfaced only our own 11:48Z
loss_action answer; `history` shows nothing from the owner after
11:43:03Z and no new reactions. Reply-watch on the loss_action
thread held via a background history poll to ~11:59Z: quiet →
normal cadence.

**Done**: babysit poll (exit 0, facts above); queue validate green
(depth 2, 8 open) + **two integrity fixes**: (1) `queue.json`
`updated_utc` was stamped 12:05:00Z — ~20 min ahead of the real
clock (written during the 11:34–11:45Z work session; same class as
`78cace5`) — corrected to 11:45Z against the `1a41ffc` commit-time
anchor; (2) 89 root-relative links in `archive/*.md` (rolled
verbatim from now.md, so `papers/`, `posts/`, `journal.md`,
`reports.md` all 404'd one level deep) rewritten to `../` paths,
grep-verified 0 remaining. `run_work_next` armed (chained work
session takes `lit-radar-async-exec`).

**Next**: 5k kill-bar binds ~13:2xZ (probe must be < 12.6394 —
currently 11.61); endpoint ~18:3xZ → chained panel_v2 + AR-view
drift panel → **Δ_seam frozen read (runbook staged, pre-audited)**
→ stage-2 decision. `queue_cli.py next` → `lit-radar-async-exec`
(any GPU-busy window).

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
attach_F 08-09 04:58–07:42Z train COMPLETE **+~10.2 GPU-h** + chained
panel_v2 eval live (~1–2 GPU-h; batch gate 70, rate-gate projection
50.3 incl. K estimate)). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 2026-08-09 11:49–12:0xZ (tick, babysit; 0 GPU-h): attach_K
step 3460/10k healthy (3.799 s/step, probe 11.6124@3000 best,
kill-bar margin ~1.0, binds ~13:2xZ, endpoint ~18:3xZ); Discord
clean — only our own 11:48Z loss_action answer surfaced, reply-watch
held to ~11:59Z via background history poll, quiet. Two integrity
fixes: queue.json `updated_utc` future-dated 12:05Z → corrected to
11:45Z (78cace5 class), and 89 root-relative links across
`archive/*.md` (papers/posts/journal/reports, all 404 one level
deep) rewritten to `../` paths, grep-verified 0 left. Queue validate
green depth 2; run_work_next armed (lit-radar-async-exec next).

Session 2026-08-09 12:12–12:2xZ (tick, babysit; 0 GPU-h): attach_K
step 3800/10k healthy (loss 3.10, 3.78 s/step, probe 11.2033@3500
best, vram 59.07 ≤ 71; 5k kill-bar margin ~1.4, binds ~13:2xZ,
endpoint ~18:3xZ). Discord clean — read empty, history nothing new
after our 12:12Z session post, no new reactions. Queue validate
green depth 2 (8 open); run_work_next armed (lit-radar-hooks-0809b
next: QDepth-VLA + fresh sweep). Stable stretch → exited rather
than held; next tick catches the 5k crossing.
