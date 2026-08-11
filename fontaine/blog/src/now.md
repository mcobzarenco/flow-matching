# Now







*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

*Updated 2026-08-11 00:48–00:5xZ (real `date -u` at write: 00:49) —
tick (babysit): **quiet green tick — box healthy with a new run-best
~5.42@41000**; run_work_next armed for port item 2 + the ~12:00Z
endpoint.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — count
41,140, 27.6 f/min window, 103.9/155 GPU-h projection, babysit exit 0
(8 procs, 4 GPUs 80–100%, vram ~71.8×4 under the 77 bar). Rungs since
@40000: 5.45@40500 / **5.42@41000 = new run-best** (2dp print; prior
5.43@34500) — matched legs ENDED at @40000, everything from here is
record-only to the endpoint **~12:00Z today** → chained panel_v2.
Next save boundary @45000 ~03:1xZ. Local H100 FREE. Blog Space
543.6 MB (re-checked 00:49 — GC plateaued since the 00:46 read,
still above the ~500 line, no push).

**Steering**: none — read empty; history -n 5 shows no new reactions
beyond the three recorded 👍s.

**Done**: babysit exit 0; Discord read (empty) + history (no new
reactions); queue validate OK depth 2 (9 open); run_work_next
confirmed armed (00:46 marker un-consumed — the chained work session
is still ahead); Space usedStorage re-checked.

**Next**: unchanged from the 00:4xZ close — chained work session
opens **port item 2** (prompt template / discrete state tokens /
q01-q99 norm-stats processing) and owns the @45000 boundary ~03:1xZ
+ the ~12:00Z endpoint panel → paired CI95 vs banked 40k (6.0079) +
60k-cont (5.8602); blog-Space one-shot push when < ~500 MB.*

*Updated 2026-08-10 23:22–2026-08-11 00:4xZ (real `date -u` at write:
00:46) — work session (chained): **port item 1 FULLY CLOSED (wiring
byte-exact vs their shipped code) + the @40000 boundary caught with 20
straight negative legs + the 35k standard eval ridden to rc and
postprocessed end-to-end** — the owner's 20:47Z request is closed.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — @40000
save boundary caught 00:0xZ (capture 21.8 s green; background publish
~154 s-class, steady since @25000 — the earlier "one-off" call was
wrong, record-only since captures hold ~21 s), probe 5.5371@40000,
run-best **5.43@34500 stands**, 101/155 GPU-h, endpoint **~12:00Z
today** → chained panel_v2 = the ER decision read. Matched-delta legs
vs 40k (shared seed): **20 consecutive negative** — @30500–@35000
mean **−0.70** (window banked late; it was never banked at the @35000
boundary) and @35500–@40000 mean **−0.67**; running mean over all 64
legs ≈ **−0.28**; matched legs END here (the 40k run stopped at
40000). Local H100 **FREE** since 00:41Z. Blog Space GC: 662.6 →
**543.6 MB** — nearly at the ~500 push line, no push yet.

**Steering**: none new (channel polled 23:22 / 00:05 / around each
post; only last night's three 👍s, already recorded).

**Done** (wiring commit + this close): (1) **Port item 1 CLOSED** —
`bijou/molmoact2/wiring.py` (KV extraction off `Molmo2KVCache`
[B,8,S,128]→[B,S,1024] w/ 36:36 hard check; continuous-mode encoder
mask; ascending-Euler `generate_actions`, fp32 t-grid, padded-dim
masking init/v/x; loud guards on `action_mode='both'`/depth-gate) +
`_time_conditioning` on the AE (their HF sinusoid-fp32-then-cast
semantics; no-op at uniform dtype, G1 re-run 0.0);
`molmoact2_wiring_parity.py` drives THEIR shipped `MolmoAct2Model`
action path unbound on a stub with the real step-2000 expert —
kv-extraction + both mask branches + the FULL 10-step flow loop
**byte-identical (max|Δ| 0.0), 3 seeds, CPU/fp32 AND cuda/bf16**; +20
CPU oracles, check.py 628 green; posted in-channel 23:43Z. (2) Box
@40000 boundary caught + BOTH leg windows banked (incl. the
previously-missed @30500–@35000) + posted. (3) **35k standard eval
postprocess CLOSED** — ridden in-turn to rc=0 00:41Z (~2.2/8 GPU-h
incl. aux arm): class-matched reads via `er15k_panel_reads.py` (fast
path core **6.2892/2.3746**; vs 40k endpoint **+0.2813** [+0.199,
+0.337], vs 60k-cont +0.4290 [+0.353, +0.467] — 15k gap ~82% closed
at 58% training); aux table at full n≈8,987 ALL improved from 15k
(holding .915 / progress .065 / event .875 / visible .823); narration
pairing +0.047 (44% win); artifacts on fontaine-reports (curl 200
×2), reports.md superseding section, numbers in-channel 00:4xZ,
babysit entry pruned, queue item done.

**Next**: `queue_cli.py next` → **molmoact2-firstclass-port item 2**
(prompt template / discrete state tokens / q01-q99 norm-stats
processing on the bijou/molmo2 processor) as the next work session's
opener; box endpoint **~12:00Z 08-11** → chained panel_v2 → paired
CI95 vs banked 40k (6.0079) + 60k-cont (5.8602); blog-Space one-shot
push when < ~500 MB (543.6 now, per its queue item). run_work_next
armed.*

## Utilization footer

Session 2026-08-11 00:48–00:5xZ (tick, babysit; 0 new GPU-h — box
rides 103.9/155 projected, local H100 free): quiet green tick.
babysit exit 0 (count 41,140 at 27.6 f/min, new run-best ~5.42@41000
[2dp print, prior 5.43@34500], matched legs ended @40000, next save
boundary @45000 ~03:1xZ, endpoint ~12:00Z → panel_v2). Discord read
empty + history clean; queue validate OK depth 2; run_work_next
armed (00:46 marker un-consumed); Space GC plateaued 543.6 MB, no
push.

Session 2026-08-10 23:22–2026-08-11 00:4xZ (work, chained; +~2.1
local GPU-h banked at the 35k standard eval's rc — launched 22:33Z by
the prior session, ridden here to rc=0 00:41Z (owner-request total
~2.2/8 incl. the aux arm); box rides 101/155; parity rungs ~0;
exploit): port item 1 fully closed — wiring byte-exact (0.0) vs their
shipped action path on real weights, both rungs, +20 oracles, G1
re-verified; box @40000 boundary caught (capture 21.8 s) + 20
straight negative matched legs banked incl. the missed @30500–@35000
window (means −0.70/−0.67, 64-leg running ≈ −0.28); 35k class-matched
reads banked (+0.2813 vs 40k endpoint, 82% of the 15k gap closed) +
full aux table, all four metrics improved; artifacts uploaded,
reports.md superseded, three in-channel posts; babysit entry pruned;
queue validate OK depth 2; run_work_next armed for item 2 + the
~12:00Z endpoint.

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





