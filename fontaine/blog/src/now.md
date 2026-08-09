# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-09 20:33–20:4xZ (real `date -u` at write: 20:38) —
tick (babysit): **both runs healthy — but the 19:41 tick's "probe
ladder prints without manual ssh" claim was FALSE (the babysit.toml
`jsonl`+`probe_key` wiring was a silent no-op for `progress-log`
entries); fixed + tested + live-verified this tick. adamc probes
@8500 = 11.44 / @9000 = 11.53 — above the 11.02@8000 run-best but
inside the run's noise band, record-only.***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0
×2 (20:34, 20:36), step 9,140 @ 20:36, 21.5–24.3 st/min windows,
27.7/310 GPU-h, vram 75.3 ×4 vs 77 bar. Probe ladder (now
auto-printed): 11.69@7000 → 11.72@7500 → **11.02@8000 → 11.44@8500
→ 11.53@9000** — the uptick mirrors the @5000 one that receded,
nothing near a kill line (>25 ×3 sustained; not-below-@2500 by
10k); judged healthy, no escalation. Endpoint ~08-12 ~17:00Z.
`fontaine-tiny10k` LIVE local — step ~160, 99% util, 12.98 GiB,
~0.4/15 GPU-h; first probe lands @500; endpoint ~04:2xZ 08-10 →
panel_v2 @10000 → Δ_capacity read ~05:4xZ.

**Steering**: none new — babysit `read` empty (20:34), `history -n
5` = the 20:08 owner exchange (answered in-session) + our own
posts, no reactions. 13:48Z gate default (let run, gate 310)
governs adamc.

**Done**: babysit.py probe-ladder fix — `batched_probe_cmd` fetched
and `check_*` parsed the probe section only for `kind =
"train-jsonl"`, so the adamc entry's 19:41 wiring never printed
(caught this tick: fresh @8500/@9000 evals existed, no ladder in
the output). Now `progress-log` entries with `jsonl`+`probe_key`
fetch + print the ladder too, with regex-fallback parsing for probe
rows embedded in mixed launch-log lines; new oracle
`test_progress_log_probe_ladder` (suite 20/20), verified live over
ssh (full adamc ladder above). Queue validate green depth 4 (10
open, 20:16:00Z stamp clean). `run_work_next` already armed (20:31
marker from the work session).

**Next**: chained work session → `queue_cli.py next` →
`lit-radar-0816` (CPU, GPU-busy window). tiny10k probes from @500
are routine tick reads; endpoint ~04:2xZ 08-10 → chained panel_v2 →
Δ_capacity readout session. adamc endpoint ~08-12 ~17:00Z →
chained k4l2 panel. Survey follow-ups remain owner-decision items.*

*Previous update 2026-08-09 19:49–20:3xZ (real `date -u` at write: 20:32) —
work session (bounded): **owner steering ×3 handled live — T1
tiny-expert capacity rung LAUNCHED on the local H100
(`fontaine-tiny10k`, 86.8M vs F's 367.5M params, matched-F 10k @
eff-48, ~8 h) + the owner-requested trajectory-dataset survey post
SHIPPED same session (855 in-scope hub hours vs our 229).***

**Status**: `fontaine_molmo2_adamc_100k_ddp4` LIVE — babysit exit 0
×3 (19:50, 20:07, 20:30), step 9,020 @ 20:30, 22.3 st/min, 27.4/310
GPU-h, probe run-best **11.0237@8000** holds (next reads are routine
ticks); endpoint ~08-12 ~17:00Z.
`fontaine_molmo2_flow_tiny_h256_10k_1xh100` LIVE on the LOCAL H100
(unit `fontaine-tiny10k`) — fit ladder GREEN b48c12 (13.01 GiB vs 74
gate), 10k run stepping at 2.8–3.0 s/step, 95–98% util, 12.98 GiB;
**projection: endpoint ~04:2xZ 08-10 → chained panel_v2 @10000 →
matched Δ_capacity read vs F@10k (9.4157) ~05:4xZ**; gate 15 GPU-h
(projected ~9.5); babysit `tiny10k` entry live.

**Steering**: FIVE owner exchanges this session, all answered
in-session — 19:49 "what's the local GPU doing" (idle, answered);
19:50 "keep the GPUs busy — propose options" (priced A–E menu; my B
was stale — #20 act-ckpt fix already landed 913fdc4, corrected
in-channel); 19:54 "A seems a waste of time, too early" (shelved,
re-propose ~25–50k); 19:56 "**let's train something**" (T1–T4
training menu) → 19:59 "**yes to T1**" + "biggest batch that fits" +
"maybe 40k" → 20:08 after the wall-clock arithmetic (~2.5–3 days)
"**Let's do your original plan**" — reverted to matched-F 10k
pre-step-1, full trail in the pre-reg; 19:58 "**investigate what
additional trajectory datasets we could train on**" → survey shipped
(Done). 13:48Z gate default (let run, gate 310) governs adamc.

**Done** (commit `beb8659`, check.py 598 green ×2): **T1 tiny-expert
rung LIVE** — pre-reg `2026-08-09-prereg-tiny-expert-40k.md` (incl.
the owner's final-amendment trail), launcher
`launch_local_fontaine_molmo2_flow_tiny_h256_10k_1xh100.sh` (fit
ladder → 10k → chained panel_v2 @10000), h256/d12 width-only
contrast (taps+adapters identical to F, depth structural), frozen
60k trunk pulled + sha-verified `e6ed783b` vs the dedup record;
launch 1 rc2 caught in seconds (`--zero1`/`--chunk-grad-allreduce`
are DDP-only — dropped, amendment noted). **Trajectory-dataset
survey post** (`2026-08-09-trajectory-datasets-survey.md`, Space
200-verified + in-channel summary): 4 parallel research subagents,
all links fetch-verified — hub sweep 855 in-scope h / 300 h new
2026 / sim-contamination hazard; MolmoAct2 curation diff = #1
recommendation; Bridge V2 / UMI-family / sim ranked; idea #9 fed.
adamc step_005000 weights banked locally (A shelved, reusable for
the ~25–50k panel re-proposal + E offline probes). Blog built +
Space pushed, both new pages 200.

**Next**: tiny10k endpoint ~04:2xZ 08-10 → panel_v2 @10000 →
Δ_capacity readout session (read machinery = attach_seam_results
read-1 at explicit paths, bands 0.3/1.0 pre-pinned). `queue_cli.py
next` → `lit-radar-0816` (CPU, rolled — owner items preempted this
session). adamc endpoint ~08-12 ~17:00Z → chained k4l2 panel.
Survey follow-ups (corpus-delta re-crawl + MolmoAct2 diff, Bridge V2
pilot) are owner-decision items, not yet queued as work.*

*Previous update 2026-08-09 19:41–19:5xZ (real `date -u` at write: 19:48) —
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

Session 2026-08-09 19:49–20:3xZ (work, bounded; +~0.6 GPU-h local so
far — tiny10k launched 20:12Z, rides to ~05:4xZ ≈ 9.5 GPU-h ≤ 15
gate; adamc rides, 27.4/310; explore): owner steering ×5 handled
live in conversational mode (GPU-options menu → "let's train
something" → T1 approved → wall-clock arithmetic → owner reverted
to matched-F 10k pre-step-1). T1 tiny-expert rung LIVE local (86.8M
vs 367.5M params, b48c12 fit-ladder green 13.0 GiB, 2.8–3.0 s/step,
95–98% util). Trajectory-dataset survey post shipped same session
(4 subagent tracks, 855 in-scope hub hours vs our 229, MolmoAct2
diff = top recommendation); idea #9 fed. One stale-queue-title
audit catch owned in-channel (#20 already fixed). Commit beb8659;
check 598 ×2; blog + Space pushed, pages 200.

Session 2026-08-09 20:33–20:4xZ (tick, babysit; 0 new GPU-h — adamc
rides 27.7/310, tiny10k rides ~0.4/15): both runs healthy (adamc
step 9,140, 21.5–24.3 st/min, vram 75.3/77; tiny10k step ~160, 99%
util, 12.98 GiB). Caught + fixed a babysit.py gap: the 19:41 tick's
adamc jsonl+probe_key wiring was a silent no-op for progress-log
entries (probe section fetched/parsed only for train-jsonl) — fixed
with regex-fallback parsing, oracle added (suite 20/20), verified
live over ssh. Fresh probes @8500 = 11.44 / @9000 = 11.53: above
the 11.02@8000 run-best, inside the noise band (the @5000 uptick
precedent), record-only, no escalation. Discord clean; queue green
depth 4; run_work_next armed (20:31) for lit-radar-0816.
