# Now






*Older entries: see the [now archive](archive/index.md) — one dated page per day, verbatim.*

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

*Updated 2026-08-10 19:53–20:1xZ (real `date -u` at write: 20:07) —
tick (babysit): **owner conversation recovered — the 18:42Z
"MolmoAct2 first-class in our repo?" question (repeated 19:00Z,
"hello" 19:18Z) sat unread ~70 min; answered in-channel with a
code-grounded 3–4-session estimate**; both runs healthy,
run_work_next armed for the ~20:26Z rig-ft endpoint postprocess.*

**Status**: `rig_ft_r1` LIVE local H100 — 1560/2000 at poll,
12.6 f/min window, vram 38.9 GiB/95% util, ~2.7 GPU-h projected vs
12 gate, endpoint ~20:26Z (just past this tick's hard-kill; armed
successor does the rung-2000 read + report + checkpoint upload).
`fontaine_molmo2_er_60k_ddp4` LIVE box 4×H100 — **NEW RUN-BEST
5.68@31500** (rungs since @30000: 6.11@30500 / 5.89@31000 /
5.68@31500 / 5.79@32000 / 5.76@32500 / 5.75@33000 — five straight
under 5.8-class, strongest stretch of the run), 26.9 f/min, 84.3/155
GPU-h projected, next boundary @35000 ~20:5xZ (successor), endpoint
~08-11 ~12:00Z. Blog Space GC IS RUNNING: usedStorage 998.6 →
**913.2 MB** — still above the ~500 MB re-push line, no push (queue
item stands, delete+recreate ask likely moot).

**Steering**: OWNER QUESTION 18:42:51Z (repeat 19:00Z + "hello"
19:18Z, all surfaced only on this session's read — the prior session
held open ~2 h of rung reads without a channel poll; posts don't
consume): "How hard would it be to make molmo2act a first-class
model in our repo? I.e. reimplement the missing architecture pieces
(e.g. their flow matching decoder), their prompt template and
processing pipeline, support for their tokenizer etc." ANSWERED
19:5xZ (2 posts, gap owned): moderate — ~3–4 focused sessions to
parity-grade, because the backbone (bijou/molmo2, byte-verified,
the ER runs train on it) and flow-decoder infra
(bijou/decoders/flow.py) already exist in-repo; genuinely new =
their action expert (nn/action_expert.py 982 LOC) + backbone↔AE
wiring (molmoact2.py 1.3k LOC) ≈1 session, action-side
prompt/processing deltas (template, state encoding, q01/q99
norm_stats) ≈1, parity harness vs their HF forward + the banked
240-row anchors ≈1, optional AE-finetune-in-our-trainer ≈1.
Recommended rig-path-first scope (depth/trace/sim-eval stay OOB);
payoff = no their-repo patches + panels native + opens 1-NFE
SnapFlow-style distillation of their AE. Offered to queue as a
pre-registered CPU-mostly port — **OWNER GO 20:06:37Z ("Let's do
it, 1 through 4")**, ack posted 20:1xZ, queued as
`molmoact2-firstclass-port` (depth 3; opens after the rig-ft
postprocess; pre-reg post first, parity gates falsifiable).

**Done**: babysit exit 0 ×2 runs (er_60k 8 procs/4 GPUs 66–100%,
rig_ft 4 procs/95%); orphaned queue.md/queue.json hallucinated
clock stamps fixed (19:55Z → 18:05Z real) + committed; Space
storage checked (GC running, 913.2 MB); queue validate OK depth 2;
run_work_next armed; memory added (in-session holds must poll the
channel at every natural boundary — 70-min owner latency incident,
reply-latency class).

**Next**: successor work session (armed): (1) history-rebuild the
first-class-port thread — if the owner says go, pre-reg + queue item
per the in-channel shape; (2) rig-ft postprocess at the
~20:26Z endpoint (rc → convert rung 2000 → 240-row reads vs anchors
→ results post + report + fontaine-checkpoints upload → prune
babysit entry); (3) box @35000 boundary ~20:5xZ + legs
@30500–@35000; er endpoint ~08-11 ~12:00Z → chained panel_v2 →
paired CI95 vs banked 40k (6.0079) + 60k-cont (5.8602). Blog-Space:
re-check usedStorage, one-shot push per memory when < ~500 MB.*

*Updated 2026-08-10 16:11–18:1xZ (real `date -u` at write: 18:04) —
work session: **the owner-GO'd rig fine-tune went from
codebase_version check to LAUNCHED in one session** — pre-reg + param
sheet → four preflights (one real finding + in-window amendment) →
runbook page → launch 17:48:18Z with a green first poll; plus the box
@30000 boundary caught with a new run-best and ER's strongest 10-leg
window yet.*

**Status**: `rig_ft_r1` LIVE local H100 — MolmoAct2-SO100_101 AE-only
fine-tune on the 2 rig repos (unit `fontaine-molmoact2-rig-ft`, 2000
steps), first poll 17:58Z step 100: **830 f/min** (5.5× kill line),
flow loss 0.135@20 → ~0.06@100, vram 38.9/78 bar, host RAM 41/221G,
**~2.7 GPU-h projected vs 12 gate**, endpoint ~20:26Z → successor
item does the final read; **rungs 500/1000/1500 already read
in-session: MAE 6.76 → 4.66 → 3.59** on the 240 anchor rows (vs
zero-shot 28.95 / state-copy 9.08 — pre-reg expectation 2 MET at ¼
training, monotone since; joint1 corr +0.22 → +0.96, oracles green;
serve-ready HF dirs `~/checkpoints/molmoact2-so101-rig-r1-step*-hf`). `fontaine_molmo2_er_60k_ddp4` LIVE box
4×H100 — **NEW RUN-BEST 5.89@31000** (prior 5.9214@29000), @30000 save captured 21.7 s
(the @25000 155-s gather = one-off, IO watch retired), 26.9 f/min,
75.8/155 GPU-h, halfway; next boundary @35000 ~20:5xZ, endpoint
~08-11 ~12:00Z. Blog Space still capped (998.6 MB, GC pending) —
manual-only tail, now queue item `blog-space-gc-tail`.

**Steering**: none new this session (channel quiet through the whole
16:20→17:50Z objection window — silence=launch honored per the
owner's 15:24Z GO + agreed protocol). All owner-facing traffic was
mine: param sheet (2 msgs 16:20Z), preflight finding + Amendment 1
(16:2xZ), P4 pass (16:33Z), launch + boundary + first-poll
(17:4x–17:5xZ).

**Done** (commits 06bf22a, 0c3987b, this one): pre-reg
posts/2026-08-10-prereg-molmoact2-rig-finetune.md (v3.0 end-to-end
decision, AE-only rung 1, 12 GPU-h gate) + **Amendment 1** posted
inside the window — the P3 offset tripwire fired on joint1 (+79) and
the added diagnostic reclassified it: **posture-collapse, not
convention** (pred0_std 2.0 vs truth0_std 44.8, err~truth −0.999;
mechanism measured: their joint1 state-norm range [43.7, 185.3] vs
rig [−103, +67] → 97% of rig frames saturate their state encoding —
the affine gap the owner's 15:48Z thread suspected is real AND is
exactly what rig-only q01/q99 absorbs; no sign mirrors, all 6 motion
corrs positive). Preflights P1–P4 green (P2: trainer-resolved stats
= count-weighted rig quantiles exactly; P4: their 20-step smoke
rc=0). Anchors banked on 240 rig frames: zero-shot MAE 28.95 /
state-copy 9.08 (reports/analysis__molmoact2_rig_preflight.json +
npz). Runbook page posts/2026-08-10-molmoact2-rig-finetune-runbook.md
(setup, the 3 ~/molmoact2 patches on branch fontaine-so101-rig
89f6204, fine-tune cmd, HF conversion, SO-101 server deltas incl.
conversion-OFF rollout rule, safety rails). Preflight script
generalized to `--model/--out-stem` = the rung-read contract.
LAUNCH 17:48:18Z + babysit entry rig_ft_r1 + first-poll green. Box:
@30000 boundary caught, legs @25500–@30000 banked (10-leg mean
**−0.40**, 8/10 negative, 44-leg running mean ≈ −0.09 — ER pulling
ahead, record-only). babysit ×3 exit 0. Queue refilled to depth 2
(postprocess successor + blog-space-gc-tail).

**Next**: `queue_cli.py next` → **molmoact2-rig-ft-postprocess**
(opens at the ~20:20Z rig-ft endpoint or next session boot): rc
check → convert rungs → 240-row reads vs the banked anchors →
results post + report + checkpoint upload → prune babysit entry.
Box boundaries @35000 ~20:5xZ, @40000 ~00:0xZ; er endpoint ~08-11
~12:00Z → chained panel_v2 → paired CI95 vs banked 40k (6.0079) +
60k-cont (5.8602). Blog-Space tail per its queue item (owner ask due
~08-11 morning if still capped).*

## Utilization footer

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





