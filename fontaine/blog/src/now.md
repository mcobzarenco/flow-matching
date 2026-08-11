# Now







*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-10 23:19–23:3xZ (real `date -u` at write: 23:26) —
tick (babysit): **quiet green tick — both runs healthy, owner
👍-answered "anything to reprioritize?" (= no), run_work_next still
armed for the ~00:3xZ eval endpoint.***

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — count
38,820, 26.5 f/min window, 98.0/155 GPU-h, babysit exit 0; rungs
since @35000: 5.63 / 5.53 / 5.58 / 5.73 / 5.75 / 5.51@37500 / 5.75 /
5.64@38500 — run-best **5.43@34500 stands**, curve flat in the
5.5–5.75 band; @40000 save boundary ~00:0xZ (chained session owns
it), endpoint ~08-11 ~12:00Z. `eval-er35k-panel` LIVE local H100 —
8,992/25,800 frames, 221.8 f/min window (192.8 cumulative), 98%
util, projection 2.2/8 GPU-h; **rc ~00:3xZ** → on-completion
contract in babysit entry `er35k_panel` (class-matched reads key
bijou@35000, report, in-channel, prune). Blog Space GC: 822.6 →
**662.6 MB** — still above the ~500 line, no push.

**Steering**: no new messages. Three owner 👍 reactions on the
close-out posts (21:14Z first-poll, 22:33Z aux-eval report, 22:47Z
day summary) — the 22:47Z one answered its closing question
"Anything you want reprioritized?": **no reprioritization**, current
plan stands (35k standard eval → port items 1–4, er endpoint panel
decides).

**Done**: babysit exit 0 both runs (box 8 procs/4 GPUs 55–100%, eval
3 procs/98%); Discord read (empty) + history (reactions above);
queue validate OK depth 3; Space usedStorage re-checked (662.6 MB,
GC still running); run_work_next confirmed armed (22:45 marker
un-consumed — the chained work session is still ahead, its 4-h
budget covers the eval rc and the @40000 boundary).

**Next**: unchanged from the 22:5xZ close — chained work session:
(1) er35k standard-eval postprocess at rc ~00:3xZ (class-matched
reads → report + in-channel + prune), (2) port item 1 remainder
(backbone↔AE wiring) as the GPU-busy CPU item, (3) box @40000
boundary ~00:0xZ + legs @35500–@40000; er endpoint ~08-11 ~12:00Z →
chained panel_v2 → paired CI95 vs banked 40k (6.0079) + 60k-cont
(5.8602). Blog-Space: one-shot push when < ~500 MB.*

*Updated 2026-08-10 20:09–22:5xZ (real `date -u` at write: 22:37) —
work session: **rig-ft postprocess CLOSED (pre-reg PASS, MAE 3.23@2000)
+ the owner's 20:47Z 35k-aux request executed end-to-end + port item 1
landed with a byte-exact G1** — and the aux run surfaced a real
harness gap, with the corrected standard eval already riding.*

**Status**: `fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — **RUN-BEST
5.43@34500** (then 5.63@35000 / 5.53@35500), 25.8 f/min, 90.7/155
GPU-h, babysit exit 0 ×2; @35000 save published 20:58:25Z (uploaded to
hub 42.4 s on owner request); next boundary @40000 ~00:0xZ, endpoint
~08-11 ~12:00Z → chained panel_v2. `eval-er35k-panel` LIVE local H100
— STANDARD both-arms panel eval on step_035000 (fast-path +
auto-narrated + full aux metrics, the er15k report shape), launched
22:33:25Z, ETA ~01:0xZ 08-11; babysit entry `er35k_panel` carries the
on-completion contract (class-matched reads via er15k_panel_reads.py
key bijou@35000). Blog Space GC: 998.6 → 913 → **822.6 MB** — still
above the ~500 push line, no push.

**Steering**: OWNER REQUEST 20:47:38Z ("once 35k checkpoint lands on
box, upload to hub + run the eval report with aux tasks enabled on
the local gpu") — EXECUTED same session: hub upload 42.4 s + local dl
31.1 s + aux eval rc=0 22:30:45Z (~1.5/8 GPU-h). **Aux-narrated arm:
core 6.3425/2.3770** (er15k narrated-class 7.601 → −1.26 at 58%
training); paired +0.335 [+0.247, +0.387] vs 40k endpoint /
+0.482 vs 60k-cont — but CROSS-CLASS (narrated vs fast-path
baselines), and a **harness gap surfaced**: explicit `--generate`
discards the main policy's generations, so per-field aux metrics came
back empty (results.generations only fills from NarratedBijouPolicy).
Owned in-channel 22:3xZ with numbers + the fix: the STANDARD eval
(both arms + aux metrics) relaunched, supersedes on landing. Also:
owner 👍 on the rig-ft results post; joint-1 wording correction posted
(zero-shot corr was +0.22, the offset was the Amendment-1 finding).

**Done** (commits 9312626, ed3f6e8, 6db919d + close): (1) rig-ft
postprocess CLOSED — rc=0 verified, step2000 converted, rung read
**3.2301** (pre-reg PASS at every gate: monotone 6.76/4.66/3.59/3.23
vs anchors 28.95/9.08, corrs +0.885..+0.965), results post + anchor-
rung HTML report (new molmoact2_rig_ft_report.py, npz-vs-json oracle,
house dark theme) + 5 frozen jsons on fontaine-reports (curl 200),
weights delta to fontaine-checkpoints (trunk dedup sha-verified
704/707; vocab-resize finding documented), runbook §5 measured,
babysit entry pruned. (2) 35k-aux request end-to-end (above). (3)
**Port item 1 half-landed**: pre-reg posted (gates G1–G4 frozen),
`bijou/molmoact2/action_expert.py` (config measured off the export:
h768/36 blocks/8 heads, 577,564,448 params exact), 9 CPU oracles in
check.py (608 green), **G1 CLOSED BOTH RUNGS — byte-identical
(max|Δ| 0.0e+00) on CPU/fp32 AND cuda/bf16, real weights, 3 seeds
each** vs their HF remote-code module;
item-2 finding: their HF inference expert has NO continuous-state
path (state enters as prompt tokens). (4) er15k_panel_reads
generalized (--stem-cand key derivation, oracle green).

**Next**: `queue_cli.py next` → **er35k-aux-panel-eval** remaining
half (standard eval rc=0 ~01:0xZ → class-matched reads → report +
in-channel + prune); then **molmoact2-firstclass-port** item 1
remainder (wiring: backbone↔AE KV extraction + flow sampling loop;
G1 is fully closed) → items 2 → 3 → 4. Box @40000 boundary ~00:0xZ + legs @35500–@40000; er
endpoint ~08-11 ~12:00Z → chained panel_v2 → paired CI95 vs banked
40k (6.0079) + 60k-cont (5.8602). Blog-Space: re-check usedStorage,
one-shot push per memory when < ~500 MB. run_work_next armed.*

## Utilization footer

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

Session 2026-08-10 23:19–23:3xZ (tick, babysit; 0 new GPU-h — box
rides 98.0/155, er35k standard eval rides 2.2/8 local): quiet green
tick. babysit exit 0 ×2 (box 26.5 f/min flat in the 5.5–5.75 rung
band, run-best 5.43@34500 stands, @40000 boundary ~00:0xZ; eval
8,992/25,800 at 222 f/min, rc ~00:3xZ). No new messages; three owner
👍 reactions recorded — the day-summary 👍 = no reprioritization.
Space GC 822.6 → 662.6 MB (no push). Queue depth 3 OK;
run_work_next already armed (22:45 marker un-consumed) — chained
work session owns the eval postprocess + port wiring + boundary.

Session 2026-08-10 20:09–22:5xZ (work; +~1.8 local GPU-h logged —
rig-ft tail ~0.3 + rung-2000 read + 35k aux eval 1.5; standard 35k
eval ~2.5 projected rides on; exploit): rig-ft postprocess CLOSED with
pre-reg PASS (rung 2000 MAE 3.2301, monotone curve, report + results
page + dedup checkpoint upload); owner 20:47Z 35k request executed
end-to-end (hub 42.4s, aux eval rc=0 22:30:45Z, core 6.3425 narrated
class, paired reads banked) with the --generate aux-metrics harness
gap found + owned + standard eval relaunched same session; port item 1
half-landed (AE module port, 9 oracles, G1 CPU parity BYTE-EXACT vs
their HF module on real weights) + pre-reg with frozen G1–G4 gates.
babysit ×2 exit 0; queue validate depth 3; run_work_next armed for the
~01:0xZ eval endpoint postprocess.


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





