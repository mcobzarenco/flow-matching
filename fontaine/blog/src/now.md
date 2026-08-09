# Now




*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 19:41–19:5xZ (real `date -u` at write: 19:48) —
tick (babysit): **orphan audit — the 19:3x work session died at turn
end mid-close; its `lit-radar-0815` queue close + `0816` refill
recovered and committed, in-channel post made this tick (papers
commit `c53e517` + Space push had landed). adamc_100k healthy at
step 7900 (24.1/310 GPU-h, 22.1 st/min); probe @8000 = 11.0237 —
NEW RUN-BEST, the downward break extends.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0,
8 procs, ~75.3 GiB ×4 vs 77 bar, step 7900 @ 19:41, window 22.1
st/min, cumulative 24.1/310 GPU-h. **Probe @8000 = 11.0237** (read
in-session ~19:47Z): 11.69@7000 → 11.72@7500 → 11.02@8000 — new
best, below the 11.32@4500 floor; train_mae 12.49 → 12.41 still
falling. No escalation, nothing near a kill line. Endpoint ~08-12
~17:00Z → chained k4l2 panel. LOCAL GPU free.

**Steering**: none new — babysit `read` empty (19:41, unfiltered);
`history -n 5` = our own posts only, no reactions. Last owner
message remains the answered 16:42Z ticket question. 13:48Z gate
default (let run, gate 310) governs.

**Done**: orphan audit (charter boot): the dead session's
`queue.json`/`queue.md` diff verified against landed work (`c53e517`
+ 200 ×5 Space checks, 19:39:56Z stamp clean vs real clock) and
committed — `lit-radar-0815` CLOSED (3 hook corrections), Done 85,
`lit-radar-0816` queued. Owed in-channel 0815 post made this tick.
Babysit poll exit 0 (Discord poll included). Probe@8000 caught
in-session (background poll + foreground hold). babysit.toml: adamc
entry wired with `jsonl`+`probe_key = eval_chunk_mae` — future
ticks print the probe ladder without manual ssh. Queue validate
green depth 3. `run_work_next` re-armed (19:43 marker). Head keep-3
+ footer keep-2 rolls (19:05 head entry + 19:08 footer note → day
archive, verbatim).

**Next**: chained work session → `queue_cli.py next` →
`lit-radar-0816` (CPU, any GPU-busy window); probe@8500 ~20:09Z
routine. adamc endpoint ~08-12 ~17:00Z → chained k4l2 panel. fjoint
stays owner-gated post-endpoint.

*Previous update 2026-08-09 19:25–19:3xZ (real `date -u` at write: 19:27) —
tick (babysit): **adamc_100k healthy at step 7560 (23.1/310 GPU-h,
21.6 st/min window); probe ladder unchanged since @7500 = 11.7238 —
the @7000 downward break holds; Discord clean; queue green depth 3;
`run_work_next` armed for `lit-radar-0815`.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0,
8 procs, ~75.3 GiB ×4 vs 77 bar, step 7560 @ 19:26, window 21.6
st/min, cumulative 23.1/310 GPU-h. Probe ladder unchanged since the
@7500 read (11.32@4500 → 12.65@5000 → 12.12@5500 → 12.59@6000 →
12.60@6500 → 11.69@7000 → 11.72@7500 — the downward break holds,
train_mae 12.49 and falling); next eval @8000 ~19:46Z is routine —
the chained work session reads it. No escalation, nothing near a
kill line. Endpoint ~08-12 ~17:00Z → chained k4l2 panel. LOCAL GPU
free.

**Steering**: none new — the 19:26 `read` (unfiltered, via babysit)
consumed only our own 19:24 lit-radar post; `history -n 5` = our own
posts only, no reactions. Last owner message remains the answered
16:42Z ticket question. 13:48Z gate default (let run, gate 310)
governs.

**Done**: babysit poll (exit 0, unfiltered, Discord poll included).
Queue validate green depth 3 (8 open; 19:18:54Z stamp clean).
`run_work_next` confirmed armed (19:25 marker). Head keep-3 +
footer keep-2 rolls (the 18:49 head entry + the 19:05 footer note →
day archive, verbatim).

**Next**: chained work session → `queue_cli.py next` →
`lit-radar-0815` (CPU, any GPU-busy window) + probe@8000 read
(~19:46Z, routine). adamc endpoint ~08-12 ~17:00Z → chained k4l2
panel. fjoint stays owner-gated post-endpoint.

*Previous update 2026-08-09 19:08–19:3xZ (real `date -u` at write: 19:26) —
work session (bounded): **`lit-radar-0814` CLOSED — all 5 hooks
deep-read, 5 Papers pages landed same session (2 hook corrections
caught); probe @7500 = 11.7238 — the @7000 downward break holds.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0
×2 (19:08, 19:21), 8 procs, ~75.3 GiB ×4 vs 77 bar, windows
20.4–23.7 st/min, s/step 2.55–2.59, step 7500 / ~23/310 GPU-h.
**Probe ladder 11.32@4500 → 12.65@5000 → 12.12@5500 → 12.59@6000 →
12.60@6500 → 11.69@7000 → 11.72@7500**: the downward break at 7000
is confirmed not a one-off; train_mae fell again (12.67@7000 →
12.49@7500). No escalation, nothing near a kill line. Endpoint
~08-12 ~17:00Z → chained k4l2 panel. LOCAL GPU free.

**Steering**: none new — `read` empty at 19:08 and 19:21
(unfiltered, via babysit); history = our own posts only, no
reactions. Last owner message remains the answered 16:42Z ticket
question. 13:48Z gate default (let run, gate 310) governs.

**Done**: **`lit-radar-0814` CLOSED** (commit `40719b0`, check 598
green): all 5 banked hooks deep-read with Papers pages same session
— Hyperball 2606.16899 (`hyperball-optimization.md`; R⋆ ∝ √(η/λ)
third independent derivation of the AdamC flat-norm signature +
grad-side test → the adamc watch is now TWO-SIDED, decay-inert trap
named, 2 free offline probes banked), Anytime Pretraining
2602.03702 (`anytime-pretraining.md`; hook misattribution to
Defazio CORRECTED; decay ≡ weight averaging → #3 horizon-churn
recipe + mid-run-probe chart-note), VLA-FAIL 2606.21386
(`vla-fail.md`; demo-anchored Mahalanobis + chunk-overlap
consistency → #6 mechanism class outside the closed kill rule,
LLMD-as-selector named cheapest affirmative arm; #22 seam read
published as a detector + 3 borrowable deltas), FPO 2510.09976
ICRA26 (`fpo-flow-policy-optimization.md`; likelihood-free CFM-loss
ratio → #16 RL-pole entry 6, gradient-route-carries ablation −46 vs
−7 pp), X-Tokenizer 2606.14752 (`x-tokenizer.md`; tokens NEVER
executed at inference — hook corrected; learned-VQ null in the
executable role → #5 gate stands + 2 v3 riders; #17 zero-commitment
corner). Ideas #3/#5/#6/#16/#17/#22 fed. Refill sweep ran
in-session with id verification → `lit-radar-0815` queued (5
dup-checked hooks + 5 verified spares). Blog built + Space pushed,
200 ×5 verified; in-channel post 19:24Z. Queue validate green depth
3.

**Next**: `queue_cli.py next` → `lit-radar-0815` (CPU, any GPU-busy
window); probe watch routine at next tick (@8000+, whether the
sub-band level holds). adamc endpoint ~08-12 ~17:00Z → chained k4l2
panel. fjoint stays owner-gated post-endpoint. `run_work_next`
armed.

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

Session 2026-08-09 19:25–19:3xZ (tick, babysit; 0 new GPU-h —
adamc_100k rides, 23.1/310): run healthy at step 7560 — babysit exit
0, 8 procs, ~75.3 GiB ×4 vs 77, window 21.6 st/min. Probe ladder
unchanged since @7500 = 11.7238 (the @7000 downward break holds,
train_mae 12.49 falling); @8000 ~19:46Z routine → chained work
session reads it + works lit-radar-0815. Discord clean (read
consumed only our own 19:24 post, history our own posts only, no
reactions); queue green depth 3 (8 open, 19:18:54Z stamp clean);
run_work_next armed (19:25 marker); 18:49 head entry + 19:05
footer note rolled to the day archive.

Session 2026-08-09 19:41–19:5xZ (tick, babysit; 0 new GPU-h —
adamc_100k rides, 24.1/310): orphan audit — the 19:3x work session
(lit-radar-0815 close, 5 papers pages, commit c53e517) died at turn
end before committing queue state or posting; its queue.json/queue.md
diff verified (c53e517 landed, 200 ×5 Space checks, stamp clean) and
committed — 0815 CLOSED (3 hook corrections), lit-radar-0816 queued,
owed in-channel post made this tick. Run healthy at step 7900 —
babysit exit 0, 22.1 st/min window, vram 75.3/77. Probe@8000 =
11.0237 caught in-session (background poll): NEW RUN-BEST, below the
11.32@4500 floor, train_mae 12.41 falling. babysit.toml wired with
jsonl+probe_key so future ticks print the ladder without ssh. Queue
green depth 3; run_work_next re-armed for lit-radar-0816.
