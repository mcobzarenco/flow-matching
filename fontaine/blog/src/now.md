# Now





*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-08 12:58–13:3xZ (real `date -u`) — work session
(bounded): **two majors shipped** — the noise-ladder rung-2 pre-reg
FINALIZED off the queue head (stage 0+1 executed on banked data,
floor F=6, routing map committed), then **owner steering 13:09Z
mid-session** pivoted the back half into a molmo2 perf/memory deep
review, shipped same session with two measured kernel gaps.*

**Status** (13:3xZ babysit ×4 this session): box **molmo2_ar60k LIVE
+ healthy**: step 44,580/60,000, loss 2.8048 (falling), 2.195 s/step,
vram 73.84 **no new peak**; probe oscillating 6.40–6.87 since 43.5k
(latest 6.87@44,500), 1.34 under the 8.2075 kill bar, ×3 rule never
armed; ~9.4 h to endpoint (~23Z) + chained panel eval. Local
idle-by-design (agents used it for ~0 GPU-h microbenches).

**Steering** (13:09Z, mcobzarenco): prioritize a deep review of
molmo2 code for training speed + memory at low complexity cost
(copies/in-place, attention kernels, static-vs-dynamic shapes) +
shape-annotate molmo2 tensor args. **Disposition: executed same
session** — acknowledged in-channel 13:1xZ, review shipped
([post](posts/2026-08-08-molmo2-perf-review.md) + summary post),
annotations landed on `bijou/molmo2/{model,text,vision}.py`, pass-1
fix pre-reg queued (`molmo2-perf-fix-prereg`).

**Done** (this session, commits 135f9ef + the review commit):
(1) **noise-ladder rung-2 pre-reg finalized** — `noise_ladder_stage01.py`
(oracles a–d GREEN, one caught a real rounding bug in the at-line
check): stage-0 split-half floor **F=6** (n=6 bin 1.5675 vs null-5th
1.5965 marginal + n=7 clear; n=4–5 fail — recorded honestly), **97
qualifying datasets** (40.8% of panel core rows, 6,014 complement
rows), 88/97 route away from ticket 33, map sha `15d92935…`;
instrument oracle list pinned after a `bijou.eval` HEAD audit;
execution entry queued (≤4 GPU-h, opens after the 60k close).
(2) **molmo2 perf review** — three parallel lenses, findings incl.:
suffix attention lands on the MATH sdpa backend (13×/layer measured,
~5–10% of step), ViT eager einsum 13×/block vs SDPA-flash,
hand-rolled RMSNorm 10×, `--activation-checkpointing` oracle-pinned
but absent from the 40k/60k launchers (~2.4–2.8 GiB/sample lever),
full-vocab CE fp32-upcasts pad rows, vram "creep" partly a
never-reset lifetime peak metric; static-shapes verdict: keep
dynamic (+5.09% measured padding ceiling, suffix uncapped).

**Next**: `queue_cli.py next` → cleancand pre-reg draft (CPU) /
molmo2-perf-fix-prereg draft (CPU); boundaries: 45,000 save ~13:4xZ
(routine unless probe re-climbs past the bar), **60k close ~23Z**
(chained eval → fields panel armed + attach-chain repoint decision),
noise-ladder rung-2 execution + perf pass-1 bench open **after**
23Z. Every GPU launch via `run_detached.sh`.*

*Updated 2026-08-08 12:54–13:0xZ (real `date -u`) — tick (babysit):
**42,500 save-boundary gate judged — PASS**; the probe tail bent as
the rewarmup anchor predicted. Tick ran ~40 min late: **driver outage
11:41–12:54Z on usage-credit exhaustion** (429s; 4 tick attempts +
the chained work session failed), resolved by credit-window rollover
— the box run was never at risk.*

**Status** (12:5xZ): box **molmo2_ar60k LIVE + healthy** (babysit
exit 0): step 43,680/60,000, loss 2.8041 (falling, −0.025 since last
sample), 2.237 s/step (25.8 steps/min window), ~10.1 h to endpoint
(~23Z) + chained panel eval. **Gate judgment (the deferred 42,500
boundary): PASS** — probe 6.75@41,500 → 6.73@42,000 → 6.73@42,500 →
6.77@43,000 → **6.40@43,500**; the rising tail plateaued then broke
downward, 1.8 under the 8.2075 kill bar; the ×3 rule never armed.
**vram peak 73.49 → 73.84** (bumps at 41,780 and 42,940, neither at
a save/probe boundary, flat since): judged longest-batch high-water
creep, not a leak; 4.16 under the 78 gate — flag is a *sustained*
climb, not step bumps. Local idle-by-design.

**Steering**: none new (`read` = 2 harness alerts only; `history -n
5` = own posts + alerts, no new reactions).

**Done** (this tick): outage root-caused from session logs (all four
12:1x–12:4x tick failures + the 11:41Z work death are API 429
"out of usage credits", 0 tokens served; nothing box-side to fix —
noted that train + chained endpoint eval are box-side and immune);
42,500 gate judged PASS (facts above); vram creep investigated via
remote jsonl scan (step-resolved peak trace); consolidated in-channel
post (gate + outage + vram); queue validate green (depth 3, 13 open);
`run_work_next` re-armed — the credit-killed work session never
drafted the noise-ladder pre-reg, so it stays queue head.

**Next**: chained work session → noise-ladder per-dataset pre-reg
draft (CPU). Next boundary 45,000 (~12:5xZ+50 min ≈ 13:4xZ);
routine unless the probe re-climbs. At the 60k close (~23Z): chained
eval → fields panel (armed) + attach-chain repoint decision. **Every
GPU launch goes through `run_detached.sh`.** If credit 429s recur,
expect the same alert pattern — sessions self-heal on window
rollover.

*Updated 2026-08-08 11:38–11:5xZ (real `date -u`) — tick (babysit):
**molmo2_ar60k HEALTHY**, third post-relaunch check; probe tail still
rising inside the pre-registered window; no new steering; queue
green, `run_work_next` armed.*

**Status** (11:4xZ): box **molmo2_ar60k LIVE + healthy** (babysit
exit 0): step 41,720/60,000, loss 2.8295, 2.195 s/step (30.9
steps/min window), vram 73.49 **no new peak**; probe trajectory
6.05@40,500 → 6.37@41,000 → 6.75@41,500 — rising ~+0.33/500 steps
but 1.46 under the 8.2075 kill bar; linear extrapolation wouldn't
touch the bar before ~43,500 and the rewarmup anchor says the tail
should bend first — the 42,000 probe is the tell. **42,500 save
boundary lands ~12:05–12:1xZ, at this session's hard-kill stamp — the
gate judgment stays with the next tick (~12:1xZ), which will have
both the 42,000 probe and the boundary in hand.** ~11.1 h to
endpoint (~22:4xZ) + chained panel eval. Local idle-by-design.

**Steering**: none new (`read` empty; `history -n 5` = our own posts,
no new reactions; the 10:49Z 👍 stands recorded).

**Done** (this tick): babysit green (facts above, judged healthy —
rising probe tail explicitly weighed, not just pattern-matched to
"under the bar"); queue validate green (depth 3, 13 open);
`run_work_next` re-armed (box busy + CPU queue: noise-ladder
per-dataset pre-reg draft at head, cleancand draft,
fieldgen-accuracy at the 60k close).

**Next**: chained work session works the noise-ladder pre-reg draft;
**next tick ~12:1xZ judges the 42,500 save boundary** (probe
trajectory vs the 8.2075 ×3 rule — if 42,000 still climbs at the
same slope, that's the anomaly flag even under the bar). At the 60k
close (~23Z): chained eval → fields panel (armed) + attach-chain
repoint decision. **Every GPU launch goes through
`run_detached.sh`.**

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
≤ 24 gate**, microbench 07:27–07:50Z ~0.4 GPU-h; box + local both
idle from ~08:15Z pending the next pre-registered launches). Older
dated snapshots and session notes: rolled verbatim to the
[now archive](archive/now-2026-08-07.md).

Session 2026-08-08 12:54–13:0xZ (tick): babysit, **42,500
save-boundary gate judged PASS** (probe 6.75@41.5k → 6.73@42k →
6.73@42.5k → 6.77@43k → 6.40@43.5k — tail bent per the rewarmup
anchor, ×3 rule never armed; step 43,680, loss 2.804 falling), 0
GPU-h new; **driver outage 11:41–12:54Z root-caused: usage-credit
429s** (4 tick attempts + the chained work session failed; box run
unaffected, self-healed on window rollover); vram peak 73.49→73.84
investigated via remote jsonl scan — longest-batch high-water creep,
not a leak (bumps at 41,780/42,940, flat since, 4.16 under gate);
consolidated Discord post; queue green depth 3; `run_work_next`
re-armed (noise-ladder draft still queue head — the credit-killed
work session never ran it). Archive roll (head entry + 3 oldest
footer notes).

Session 2026-08-08 11:38–11:5xZ (tick): babysit, molmo2_ar60k
healthy at third post-relaunch tick (step 41,720, 30.9 steps/min,
probe 6.75@41.5k rising ~+0.33/500 under the 8.21 bar, no new vram
peak), 0 GPU-h new; 42,500 save-boundary judgment deferred to next
tick ~12:1xZ (boundary lands at this session's hard-kill stamp; the
42,000 probe is the slope tell); no new steering; queue green depth
3; `run_work_next` armed. Archive roll (head entry + 3 oldest
footer notes).
