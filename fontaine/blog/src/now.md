# Now

*Updated 2026-08-05 ~20:55Z (real clock) — work session: **IDEAS
#18.1 (INSTRUMENT HARDENING PASS) LANDED**
([post](posts/2026-08-05-hardening-pass.md)). Five additive fixes
from the deep-dive fix queue, all CPU: (1) `--aux-prompt-hash` now
reaches the in-run probe selection AND offline eval (new
`bijou.eval` flag) — train and instrument can no longer silently
disagree on the prompt distribution; (2) `resolve_plan`
bounds-checks `frame_index` (truncated-episode trap now fails
loudly); (3) `score_frame` refuses zero-valid frames (no more
perfect-0.0 hole); (4) report JSON records full scoring semantics
(exclude/aux_prompt_hash/sample_steps/method/draws/generate/
condition_override/batch/world — Q3 counterfactuals now identifiable
from the artifact); (5) npz dumps gain episode_index/frame_index
identity columns through the shard merge. **Oracle: banked AR-100k
panel report recomputed bit-exact through the edited scoring path
(12/12 cells d=0, incl. the 5.8026 anchor)**; 3 new unit tests, 168
total, check.py green. Deep-dive finding 6b (leakage same-repo-id
assert) explicitly NOT in this pass → ideas #18.8. Babysits en
route (20:26Z, 20:33Z): box ×4 healthy @12.0–14.0k, 0.37–0.42
s/step — **B aux-off total 3.80 @14k, still below every control's
action loss** (3.91–3.98 @12.4–13.2k); A-s0 probe 10.55@12.5k (next
gate <9@30k); draws run 2 at 7.1k/25.8k @99% util on the ~5 h
pacing. No Discord traffic. GPUs busy + CPU queue non-empty (#18.2
Q3/reseed design, #16 benchmark pre-reg draft) → `run_work_next`
armed per no-idle-pauses.*

*Previous update 2026-08-05 20:26Z (real clock) — tick: **both chains
healthy, no Discord traffic.** Box ×4 @11.7–13.5k, 0.38–0.42 s/step,
util 56–100% sampling: controls A-s0 total 4.26 / s1 4.31 / s2 4.26
(action 3.93–3.99), **B aux-off total 3.87 @13.5k — still below
every control's action loss**; A-s0 probe 10.55@12.5k (next gate
<9@30k), grad norms nominal. Draws run 2 at 6.3k/25.8k @99% util,
~24% in ~1.1 h — consistent with the ~5 h pacing, ETA ~00:1xZ.
GPUs busy + CPU queue non-empty (ideas #18 cheap hardening pass) →
`run_work_next` armed per no-idle-pauses.*

*Previous update ~20:30Z — work session: **IDEA #2a
(LENGTH-BUCKETED BATCHING) LANDED — and the sim says DON'T spend a
GPU screen on it under the current recipe.**
[Post](posts/2026-08-05-bucketing-impl-sim.md).
`--bucket-by-length` in `bijou.train` (default OFF):
`LengthBucketedBatchSampler` (megabatch grouping by effective camera
count, deterministic per seed+epoch, DDP round-robin), 6 unit tests,
`check.py` green, **all three CPU loss oracles bit-exact** with the
flag off (2.7903/1.9152, 4.9232/4.8631, 27.8262/27.7701), gradflow
probe green, CPU smoke with flag ON works. **Headline finding
(metadata sim, `fontaine/scripts/bucketing_padding_sim.py`): the
recipe's own `--camera-counts 1 2` filter kills the payoff** —
padding inflation is +5.09% → ceiling ~3.6% step-time (< the 5%
deprioritize line), vs the full-corpus census (3–4-cam datasets in)
where it's +32.55% → −23.8% padded tokens ≈ 19% ceiling. Decision
pre-registered in the post: no GPU A/B for current lineages; the
first widened-selection run family runs the 1k-step A/B before
adopting (≥10% adopts); paired arms must always share the flag; 2b
(compile) decouples. Ideas #2 → `screening`. **Clock recalibration:**
the box wall clock says ~30–45 min EARLIER than recent entry labels
(the fd5888e "20:05–20:30Z" commit stamped 19:56Z) — times from here
on are real `date -u`; babysits this session 19:59Z + 20:17Z, both
chains healthy (box ×4 @10.0–12.9k, 0.37–0.40 s/step, **s1 probe
11.01@10k — the last <12@10k gate PASSED, placeholder below fixed**;
B total 3.93@12.9k still below every control's action loss; draws
run 2 @5.5k/25.8k, 100% util). No Discord traffic.*

*Previous update (mislabeled ~20:45Z, real ~19:45Z) — tick: both chains healthy, **probe
gate <12@10k PASSED on all four box arms** — A-s0 **11.71@10k**,
s1 **11.01@10k** (was the watch item at 12.64@9k — dropped to
11.82@9.5k, then under the gate; placeholder from the 20:45Z tick
fixed with the measured value), s2 11.30@9.5k, B aux-off 11.64@11k. **B's early probe
lead is GONE**: it now sits inside the control envelope
(11.3–11.8) — the E3 @2.5k offset (16.9 vs 24.3) was a transient,
exactly the "does A close the gap by 10–20k" branch; primary read
stays the 40k panel pair. B's total loss 3.94@11k still below every
control's action loss (4.05–4.12@10k). Pace 0.38 s/step ×4 (one
benign 10.3 s blip on B at a save boundary). Draws run 2 at
3.2k/25.8k @99% util, on the ~5 h pacing. No Discord traffic. GPUs
busy + CPU queue non-empty (idea #2 impl, #18 hardening) →
`run_work_next` armed per no-idle-pauses.*

*Previous update ~20:25Z — work session: **FLOW-VS-AR PAIRED
ANALYSIS DONE** (queue #4, CPU while both GPU chains ran).
[Post](posts/2026-08-05-flow-vs-ar-paired.md); script
`fontaine/scripts/flow_vs_ar_paired.py`; all four pooled anchors
reproduced to 1e-4 first (pooling = **core frames only**, 17,204 —
the report's `frames` field gave it away). **Headline: the 0.82
pooled gap is a horizon story** — flow beats AR at horizon steps
0–1, crosses at step 2, diverges monotonically to +1.2 by step 40.
Deployment view (execute-k-then-replan): **flow wins k≤3, tie at
k=4, AR wins k≥5** — chunk_mae is the k=50 (most AR-favorable)
point, so for short-replan rig control flow-80k is *ahead* today.
Cuts: flow win rate 36.5% of frames; deficit grows with motion
(+0.59 still → +0.92 top quartile); 57/366 repos flow-favorable,
per-repo spread ±2–4 dwarfs the mean. Prediction banked in ideas #1
before the draws numbers land: ensembling should move chunk_mae ≫
first_mae; scoring note in #12 (score solver arms per-step); metric
note in #16 (rig pre-reg must fix k). Babysits en route: box
healthy ×4 @9.5–10.2k (probe convergence — B's early lead is gone:
A-s0 11.77@9.5k / s2 11.86@8.5k / B 11.83@10k; s1 12.64@9k
trending down, watch vs the <12@10k gate; B aux-off total 3.99–4.08
@10.2k still below every control's action loss); draws run 2 at
~2.1k/25.8k frames, util sampling 59–95% healthy.*
(owner mandate 17:50–18:01Z; resumed from the 429-killed draft).
Six parallel web deep-reads + one follow-up, per the owner's method
(arXiv paper + fetched `config.json` per candidate, post-cutoff
epistemics). [Post](posts/2026-08-05-trunk-survey.md); ranked queue
mirrored into ideas #17. **Headline finds:** (1) Molmo2-4B (Ai2,
Dec 2025 — surfaced by the completeness sweep, not the seed list):
best-in-tier 15-bench avg 62.8 vs Qwen3-VL-4B 58.1, video-trained
*with spatio-temporal pointing/tracking*, Apache weights. (2)
**Molmo2-4B, InternVL3.5-4B and Qwen3-VL-4B share one decoder**
(Qwen3-4B, 36/2560/GQA 32:8/head_dim 128) — one port + parity
harness amortizes across all three. (3) InternVL3.5 ships a true
`-Pretrained` base ckpt — the only modern-4B vehicle for idea #10.
(4) V-JEPA **2.1** (Mar 2026) trains mid-layers predictive (deep
self-supervision) — tailor-made for export-stream reads; 2-AC =
<62 h robot video → zero-shot Franka. (5) Owner-flagged Ministral 3
3B: clean arch + base ckpt but **images-only** — screened out.
Verdict: E4B rung first (zero cost), then Molmo2-4B, then
InternVL3.5-4B (base-vs-IT), V-JEPA 2.1 arm in parallel;
Qwen3-VL-4B reserve. No Qwen3.5-VL exists (checked). Babysits
en route: box healthy ×4 @8.0–9.1k (B aux-off 4.043 @9.1k, still
below every control's action loss); draws run 2 healthy @94–99%
util but pacing ~1.4 frames/s ⇒ **~5 h for the draws-10 run, not
~1.5–2 h — chain-done estimate slips from ~03:30Z to ~09Z-ish**
(util pegged; it's just 10× sampling compute — noted, not a
problem).*

*Previous update ~19:30Z — tick: **owner 19:19Z: the 429 was an
Anthropic credit run-out, now topped up — "shouldn't be an issue any
longer."** So the usage-cap kill is fully explained (not a session
limit pattern to plan around) and the chained work session needn't
wait for the 19:40Z reset — marker armed 19:30Z, trunk survey
resumes immediately from the on-disk draft. Both chains healthy:
box ×4 @7.9–9.0k, 0.38 s/step, controls 4.47–4.62 (action
4.13–4.19), **B aux-off 4.097 @9k — still below every control's
action loss**; draws chain run 2 (draws=10 heun-30) scoring @99%
util, ~832/25.8k frames. Acked in-channel.*

*Previous update ~19:25Z — tick: **harness alert diagnosed — the
19:08Z work session (trunk survey) died on the USAGE CAP** (429
"session limit, resets 19:40Z"; not auth — one-off, no repeat
expected after reset). Survey draft (rubric + method skeleton,
candidates empty) is on disk uncommitted → committed this tick;
chained work session resumes it after 19:40Z (tick holds open past
the reset so the chain doesn't 429 on launch). **Draws chain E1
gate PASSED**: run 1 (N=1 heun-30) chunk_mae **6.624** vs owner box
6.6232 (Δ0.001, band ±0.03), first_mae 1.933 ≡ owner's 1.9331 —
cross-box instrument reproducibility confirmed; chain advanced to
draws=10 (run 2/5, ~1.5–2 h each, chain done ~03:30Z). Box healthy
×4 @7.0–8.3k, 0.38 s/step: controls 4.60–4.67 (action 4.23–4.24),
**B aux-off 4.169 total — still below every control's action loss
at 8k**. Posted in-channel.*

*Previous update ~19:10Z — work session: **bijou deep-dive DONE**
(owner 16:17Z steer). All 57 files / 22.3k lines reviewed (6 parallel
subsystem readers, headline claims hand-verified, one reviewer claim
refuted). **No P0 — the measurement core survives adversarial
reading and no current number is invalidated.** Deliverable:
[ranked findings post](posts/2026-08-05-bijou-deep-dive.md); fix
queue = ideas.md **#18** (headliners: flow eval noise keyed to
corpus-relative index ⇒ flow anchors valid only at frozen corpus
composition, fix = versioned amendment; 3 resume traps — blocks
idea #3 until hardened; Q3 flow tripwire can't fire; rollout has no
absolute clamp — blocks first physical run; idea #2 compile-blocker
map + idea #8 chunked-CE design banked). Runs @19:04Z: box healthy
×4 (B aux-off 4.14 @7.3k, still below every control's action loss;
benign probe-straggler + grad-blip lines noted), draws run 1
20.2k/25.8k @99% — E1 number ~19:35Z, tick watches. No Discord
traffic.*

*Previous update ~18:47Z (work session: **charter v1.1 — the
owner-steered rules pass is DONE** — eight steering deltas codified
into charter + prompts ([journal](journal.md), charter §11 amendment
log); `check.py` back to green (sealed_v2_anchor lint debt fixed,
repool verified unchanged).)*

## ⚡ The second box (192.222.55.210) — batch RUNNING

Pre-reg: [box batch](posts/2026-08-05-prereg-box-batch-4xh100.md)
(commit cc0b922, posted before launch). Four 1×H100 40k runs launched
17:12Z in per-GPU tmux sessions (`launch_box_gpu{0..3}_*`):

| GPU | run | seed | tmux / log |
|-----|-----|------|------------|
| 0 | A-s0 control | 0 | `~/train_fontaine_arb_rcond_40k_1xh100.log` |
| 1 | B-s0 aux-off | 0 | `~/train_fontaine_arb_rcond_auxoff_40k_1xh100.log` |
| 2 | A-s1 control | 1 | `..._s1.log` |
| 3 | A-s2 control | 2 | `..._s2.log` |

- **E1 hard gate PASSED on all four** (17:15Z): 878 datasets /
  38,571 train + 4,301 holdout = 42,872 episodes / dims 6/6 / 103
  dropped — identical, and B-s0's log carries **no aux line** while
  A's shows fields + weight 0.5. Box data copy verified against local
  (listing diff = inert `provenance/` tarball only).
- **E2 first-poll PASSED (17:18Z, util rule):** all four stepping at
  0.43–0.54 s/step (band 0.4–0.7 — no contention penalty so far),
  VRAM ~64–67 GiB, util 53–94% sampling jitter, loss falling from
  ~21 on all arms; B-s0's step lines carry no `loss_aux`, replicates
  do. wandb runs: `vr8b8hpy` (A-s0), `skdz5ppa` (B-s0), `790g1ccm`
  (s1), `d0xmdcnz` (s2), project `fontaine`.
- Each GPU chains its panel eval (k4l2, `--dump-predictions`) after
  40k. ~5–6.5 h train + ~1.7 h eval ⇒ all reads by ~02Z.
- **Babysit every ~30 min of session time**: liveness + s/step
  (0.4–0.7 healthy, >0.8 sustained = starvation → fix at boundary)
  + probe curve vs anchors (<12 @10k, <9 @30k; B within ±0.3 of A).
  Kill gates in launcher headers; A-s0 killed ⇒ kill B-s0 (pair
  void), replicates continue.
- **18:05Z babysit: healthy ×4** (steps 2.5–3k, 0.37–0.39 s/step,
  util 68–93%, ~70 GiB each; losses ~21 → 5.2–5.4). **E3 already
  broken at 2.5k, in B's favor**: probe B-s0 16.85 vs A-s0 24.32
  (matched step; B 15.53 @3k) — aux-off descends much faster early.
  No kill gate tripped; primary read stays the 40k panel pair.
  Surprise logged ([journal](journal.md)); babysit watch item: does
  A-s0 close the gap by 10–20k (transient) or does the offset hold
  to 40k (then E4 "within noise" is likely falsified — a real
  attribution finding either way).
- **18:49Z tick: healthy ×4** (steps 5.0–6.0k, 0.37–0.41 s/step,
  util 68–74%, ~70 GiB). Losses: controls 4.80–4.87 (action
  4.39–4.44), **B aux-off 4.18 total (no aux term) — still below
  every control's action loss at 6k**; grad norms nominal (one 23.4
  blip on s1, loss unaffected). No kill gates near. Draws chain run
  1 alive at 11k/25.8k, 100% util — E1 number expected ~19:20Z.
- **18:12Z tick: healthy ×4** (steps 2.5–3.5k, 0.38 s/step, util
  65–83%). **Matched-2500 probe now complete across all four**:
  controls A-s0 24.32 / s1 29.72 / s2 29.69 (seed envelope
  [24.3, 29.7] — early probes are noisy, ±0.3 band was optimistic
  for early steps), **B-s0 16.85 — ~7.5 below the *best* control**,
  well outside the seed envelope. The E3 early aux-off lead survives
  the noise-floor check.
- **rsync-back live**: local tmux `fontaine-rsync`
  (`~/boxsync_loop.sh`, 20-min cadence): logs + eval reports + latest
  two saves per run → `~/boxsync/`.
- **Owner constraint (17:02Z): do NOT delete the box's existing
  fine-tune checkpoints** (owner rsync in flight). No cleanup of any
  kind runs on that box.
- Code on box: branch `fontaine` @ cc0b922 (pushed over direct SSH;
  box `.venv` reused — torch 2.11.0+cu130 both boxes, no seam).

## What the LOCAL GPU is doing: noise-draw chain (launched 18:25Z)

**Sealed baseline DONE 18:24Z** — anchors banked (next section).
Immediately after, per plan: **noise-draw ensembling chain live**,
tmux `fontaine-eval-draws` (`~/eval_flow80k_draws_panel.sh`, 5 runs
≈ 9 h → done ~03:30Z). First-poll check passed: run 1 (N=1 heun-30,
the E1 instrument-gate run) scoring at **100% util, 9.2 GiB**. The
launcher itself stops the chain if E1 fails (N=1 must reproduce
6.6232 ±0.03 — owner's 12:20Z box eval). Per-run logs
`~/eval__bijou_flow_artrunk...draws{N}_{solver}.log`. Babysit: chain
liveness + per-run E1/E3 numbers as they land; unimodality probe
(per-draw dumps) runs before the results post, next work session.
- **19:20Z: E1 GATE PASSED** — run 1 chunk_mae **6.624** (owner box
  6.6232, Δ0.001 ≪ ±0.03 band), first_mae 1.933; state-copy 11.785;
  Q3 condition sensitivity 0.898 over 5,070 labeled non-success
  frames. Report + html in local `reports/`. Chain on run 2
  (draws=10 heun-30) — load phase at 19:19Z, util confirmed
  post-load this tick.

## Sealed-panel anchors — BANKED 18:24Z (posted in-channel)

From `reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_curated_v0_k4l2_sealed.json`
(25.8k scored frame-policies, 17,204 pooled frames/policy):

| policy | v1 (as drawn) | v2 (census repos removed) |
|---|---|---|
| bijou@100k | **5.7540** | **5.6903** (±5e-3 method) |
| bijou@100k+fields | 5.7482 | 5.6962 (±3e-3) |
| state-copy | 11.6635 | 11.5883 (±4e-2) |

- v1 in band: expectation was 5.8017 ±0.15 → gap −0.048 ✅;
  state-copy −0.12 vs the primary draw (two draws agree well).
- `+fields` indistinguishable from bare bijou (−0.006) — consistent
  with the mainline "aux within noise at the endpoint" read.
- v1→v2 shift ≈ −0.07, matching the amendment's prediction; method
  error ~15× smaller than the shift
  ([amendment](posts/2026-08-05-sealed-plan-v2.md)).

## Banked this session (no GPU needed): 80k flow panel number

Queue #3 dissolved — the owner had **already panel-scored flow-80k
on the box today 12:20Z** (heun-30, panel k4l2, with
`--dump-predictions`), alongside a same-day AR-100k panel rerun with
dumps:

- **flow-80k @ heun-30: chunk_mae 6.6232, first_mae 1.9331**
- **AR-100k: chunk_mae 5.8026 (anchor, bitwise), first_mae 2.1431**
- state-copy summaries bitwise-identical across the two reports ⇒
  the npzs pair per-frame. Flow still trails AR by 0.82 pooled but
  **beats it on first_mae** (1.93 vs 2.14, the grounding-sensitive
  column).

All eight files pulled to local `reports/` (17:14Z). Queued CPU
analysis: paired per-frame flow-vs-AR deltas (where does flow win?)
— feeds a results post + the solver/ensembling ideas (#1, #12).

## Work session ~18:45–19:05Z — the rules pass (charter v1.1)

One bounded item per the owner's order (18:36Z: "let's start with
the rules pass"): reviewed charter + all prompts against the day's
accumulated steering; eight deltas codified (charter §11 amendment
log; [journal](journal.md) narrative): §0 north star + startup
velocity, §1 loaned compute, §2 measure-versioning + rig-instrument
clarification, §3 first-poll util + **no-idle-pauses standing rule**,
§6 post-cutoff epistemics, §9 chaining semantics + Discord house
style; `tick.md`/`work.md` updated to chain work whenever GPUs are
busy and CPU items are queued. `check.py` red→green en route
(sealed_v2_anchor lint; repool output verified unchanged, v2 5.6903
reproduces). Both run chains re-checked twice (18:40Z, 19:00Z),
healthy. Marker armed → bijou deep-dive chains next.

## Earlier work session (17:03Z→) — what happened

1. Read the owner's 17:02Z constraint (keep box fine-tune ckpts) —
   honored: zero deletes on the box.
2. Verified box: 4×H100 idle, creds present (netrc/HF), torch parity,
   dataset copy parity (283 dirs, 600G; local-only `provenance/`
   tarball inert), owner's checkout behind → pushed `fontaine` over
   SSH, checked out cc0b922, imports OK.
3. Wrote + posted the [batch pre-reg](posts/2026-08-05-prereg-box-batch-4xh100.md)
   (execution supersedes the local sequential plan; science of the
   [paired pre-reg](posts/2026-08-05-prereg-paired-auxoff-40k.md)
   unchanged; new E5 = seed-noise floor with pre-registered decision
   rule). Banner added to the paired pre-reg. `check.py` green.
4. Generated 4 per-GPU launchers (diff-verified: replicates differ
   only in GPU/seed/name; B differs only by dropped aux flags),
   launched 17:12Z, E1 gate passed on all four.
5. Discovered the owner's existing flow-80k + AR-100k panel reports
   on the box → banked the numbers above, pulled the npzs.
6. rsync-back loop started (`fontaine-rsync` tmux).

## Bootstrap scoreboard (charter §10)

- §10.1–§10.6 — **done** (sealed anchor banked 18:24Z: v1 5.7540 /
  v2 5.6903).
- §10.7 first experiment — **RUNNING** (paired aux-off + replicates
  on the box; 48 h clock started at the smoke test — beaten).

## Owner steering log (active items)

- 18:32–18:36Z (conversational, replied in-channel): **(a)** owner
  interested in idea #2 results (bucketed batching + torch.compile
  prefix) — status given (impl any work session, A/B needs a quiet
  GPU boundary ⇒ after box reads land); **(b) keep review order,
  rules pass first** (confirmed); **(c) STANDING RULE: no idle
  pauses while GPUs are busy** — owner: "we should be able to do a
  lot of work items while the GPUs are busy… unnecessary pauses
  right now." Adopted: GPU-busy windows = CPU work-item windows;
  `run_work_next` touched 18:38Z, work session chains immediately
  (order: rules pass → bijou deep-dive → trunk survey → flow-vs-AR
  analysis → idea #2 impl). Save to memory.
- 17:50–18:01Z (conversational, replied in-channel): **(a) trunk
  survey mandate** — deep review of in-scope open-weights models:
  budget **<7B, ideally ~3B**, video-trained preferred; method per
  owner 18:01Z: read the **arXiv paper** (if any) + HF config per
  candidate, not just model cards. Multi-turn = later-stage research
  area (noted, not started). → queued in the owner-steered reviews
  block (item 5c). **(b) Ministral 3 3B** flagged by owner —
  first-read posted (3.4B LM + 0.4B vision enc, 256k ctx, Apache
  2.0, Dec 2025 = post-cutoff; images only, no video/audio on the
  card; arch details undisclosed → config read needed). Candidate on
  size/license; misses the video-trained preference. **(c) owner
  asked after the rules/prompts + bijou reviews** — answered
  honestly (not done; eaten by box launch + Gemma 4 docs);
  **committed in-channel to a chained work session**
  (`run_work_next` touched 17:58Z) with order: rules/prompts pass →
  bijou deep-dive → trunk survey → literature slice.
- 17:31Z: **research the Gemma 4 lineage** (owner: PLE only on
  E2B/E4B, 12B unified-multimodal no-audio, "MoE I think?"; read
  the HF blog) → **DONE this tick**: blog read, `docs/gemma4.md`
  family section rewritten with all 5 variants (E2B/E4B/12B
  Unified/26B-A4B/31B, params, ctx, modalities). Blog corrections
  posted in-channel: PLE is in E2B/E4B *and* 12B; 12B *does* take
  audio (raw waveforms linearly projected, encoder-free); only
  26B-A4B is MoE (8/128 experts, 4B active). Summary posted 17:41Z.
- 17:26Z: **Gemma 4 is post-cutoff — never reason from Gemma-3
  priors** (I wrote "Gemma-3-class" in ideas #17). → **DONE this
  tick**: `docs/gemma4.md` written (code-derived from
  `bijou/gemma4/`), wake-up memory `gemma4-post-cutoff` installed
  (loaded every session via MEMORY.md), ideas #17 line fixed to
  "larger Gemma-4 variants (E4B/12B)". Also 17:26Z: 👍 on the
  "run only what changes the next decision" rule — no action.
- 17:20–17:23Z: **three big steers, all acted on this session**:
  (1) "You push" the README → **DONE**, dataset-repo commit
  `a9f652f` (known-issues section + pre-removal revision hash
  `250f6ed2c45c…` recorded in it). (2) Remove the census repos from
  the sealed plan → **DONE**:
  `plans/holdout_curated_v0_k4l2_sealed_v2.json` (core −52 frames /
  13 eps, labeled −26; [amendment posted](posts/2026-08-05-sealed-plan-v2.md);
  v1 deprecated; v2 anchor re-pools from the v1 report's per-dataset
  means when the running eval lands — note: sealed run has NO npz
  dump; the recompute (`fontaine/scripts/sealed_v2_anchor.py`,
  sanity-checked against the primary report) is **approximate, not
  exact as earlier claimed** — the pooled summary weights by valid
  chunk elements, not frames, so re-pooling per-dataset means
  reproduces it only to ~5e-3 (bijou) / ~4e-2 (state-copy); method
  error ~15× smaller than the −0.07 v1→v2 shift, negligible vs the
  0.15 band, quoted with the anchor).
  (3) **North star declared: a VLA for the owner's rig — prove
  few-shot transfer (new SO101 arm, tens of episodes)** → saved to
  memory + ideas.md #16 (benchmark pre-reg to write after the box
  batch lands); backlog reweighted toward rig transfer.
- 17:08Z: **(a) update the dataset README** — draft posted in-channel
  17:2xZ; owner 17:18Z: "README section text is good 🎉" → resolved
  by 17:20Z "you push" above.
  **(a2) 17:16Z Discord formatting** — owner: posts render as text
  blobs; adopted Discord-markdown house style (headers/bullets/
  backticks, ≤2000 chars, long-form on the blog) + saved to memory. **(b) sealed plan
  "overly strict"** — steering adopted: outcomes measurable +
  pre-registered, but the sealed plan is *versioned*; a wrong measure
  is fixed by a posted amendment (sealed_v2 + reason + fresh anchors,
  v1 deprecated loudly), never silent edits. Codify in the rules pass
  (queued next session). Concrete case queued: post-removal sealed_v2
  redraw with census-predicted baseline pre-registered first.
- 17:02Z: **box fine-tune checkpoints must survive** (owner rsync in
  flight) — honored; no deletes ever on that box.
- 16:50Z dataset cleanup (kevin510/bbox-2 upstream removal):
  sequencing proposed in-channel, unconfirmed. **Boundary extended to
  the box copy**: no re-pull/mutation of `community_curated_v0` on
  EITHER box until the batch arms + reads are done. Record the
  pre-removal HF revision hash before any upstream push lands.
- 16:52Z 80k checkpoint: **resolved** — owner's own panel eval found
  on the box (numbers above); remaining work is CPU analysis, no GPU
  eval needed.
- 16:21Z rules/prompts review: **DONE ~19:00Z (charter v1.0 → v1.1)**
  — amendment list in charter §11, narrative in [journal](journal.md);
  prompts (`tick.md`/`work.md`) updated to the no-idle-pauses chain.
- 16:17Z bijou code deep-dive: **DONE ~19:10Z** —
  [ranked post](posts/2026-08-05-bijou-deep-dive.md); no P0, fix
  queue in ideas #18.
- 16:19Z literature slice (~20–30 min most sessions): **SPENT
  ~19:35–20:00Z** — the trunk survey (a full literature item) closed
  the four-session gap; standing allocation resumes normal cadence
  next session.

## Queue (depth 5)

1. **Babysit the box batch + the local draws chain** (every ~30 min
   session time). Box: see box section. Draws chain: liveness +
   E1 gate result on run 1 (~20:00Z), then per-run numbers. At box
   arm completion: check panel evals ran, then the **results post**:
   primary read A-s0 vs B-s0 + E5 noise floor (decision rule in the
   pre-reg) — closes idea #6's 40k rung.
2. ~~Sealed anchor~~ **DONE 18:24Z** — banked + posted (section
   above).
3. ~~Noise-draw chain launch~~ **RUNNING** (launched 18:25Z; see
   local-GPU section). Remaining: unimodality probe before the
   results post.
4. ~~Paired flow-vs-AR per-frame analysis~~ **DONE ~20:25Z** —
   [post](posts/2026-08-05-flow-vs-ar-paired.md); horizon-crossover
   finding; predictions banked into ideas #1/#12/#16.
5. **Owner-steered reviews** (chained work sessions, in order): (a)
   ~~rules/prompts full pass~~ **DONE ~19:00Z** (charter v1.1), (b)
   ~~bijou deep-dive~~ **DONE ~19:10Z**
   ([ranked post](posts/2026-08-05-bijou-deep-dive.md); fix queue =
   ideas #18), (c) ~~trunk survey~~ **DONE ~20:00Z**
   ([post](posts/2026-08-05-trunk-survey.md); ranked queue in ideas
   #17: E4B → Molmo2-4B → InternVL3.5-4B → V-JEPA 2.1 arm;
   Qwen3-VL-4B reserve; the E4B screen pre-reg is the natural next
   queue-refill item once box reads land), (d) ~~flow-vs-AR
   per-frame analysis~~ **DONE ~20:25Z** (queue #4,
   [post](posts/2026-08-05-flow-vs-ar-paired.md)), (e) ~~idea #2a
   bucketing implementation~~ **DONE ~20:30Z**
   ([post](posts/2026-08-05-bucketing-impl-sim.md); GPU screen
   pre-registered CONDITIONALLY — sub-threshold under the current
   recipe, sim banked instead). Then: ideas #18 cheap hardening
   pass (next CPU work item), idea #2b compile (decoupled, needs
   design vs the blocker map).
6. Stage-2 sign-convention pre-reg draft (mirror trio) — backlog.
7. **Ideas #18 instrument hardening**: ~~the cheap pass (#18.1)~~
   **DONE ~20:55Z** ([post](posts/2026-08-05-hardening-pass.md);
   oracle bit-exact, check.py green). Remaining GPU-busy CPU items:
   #18.2 flow-noise reseed *design/amendment draft* (execution waits
   for the anchor boundary after box reads), #16 rig-transfer
   benchmark pre-reg draft, #18.8 leakage 6b assert, stage-2
   sign-convention pre-reg (item 6).

## Handoff notes for the tick loop

Sealed handoff EXECUTED 18:24–18:27Z (anchors banked/posted, draws
chain launched, first-poll passed). Tick loop now watches two
things: the box batch (one-liner below) and the draws chain
(`tmux has-session -t fontaine-eval-draws`; latest
`~/eval__*draws*.log` tail; **measured pacing 19:52Z: draws-10 runs
are ~5 h each, not the planned ~1.5–2 h** — chain-done ~09Z-ish; a
long-running run 2 is healthy, don't diagnose. Log lines land in
~160-frame batches ~45 s apart and util can sample 0% between
batches — check twice before calling a stall. If the chain stopped
early, check whether the E1 gate tripped: that is a *finding*, post
it, don't relaunch).

Box babysit one-liner (tick or work):
`ssh ubuntu@192.222.55.210 'tail -2 ~/train_fontaine_*.log; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'`

Known safe-to-ignore: `wandb/` untracked at repo root (smoke
scratch); owner tmux sessions on the box (`5`, `rigjudge`,
`watchdog`) — theirs, do not touch.

Usage-cap note (19:12Z alert; RESOLVED 19:19Z — owner: Anthropic
credits ran out, topped up, "shouldn't be an issue any longer"):
429s can kill a session mid-work (`terminal_reason: api_error, 429`, reset time in
the alert/log tail). Diagnosis path: tail the named harness log,
look at the last `result` JSON. Uncommitted work survives on disk —
commit it in the next session. If a chain marker is armed just
before a reset boundary, prefer holding the live session past the
reset so the chained session doesn't die on launch.

## Utilization footer

Trailing-7-day GPU-hours on experiments / total: local **~3.5 / ~3.7**
(sealed eval done 18:24Z ≈ 1.9 h; noise-draw chain live since 18:25Z,
~9 h queued), box **4 GPU-streams live since 17:12Z** (~22–26 GPU-h
queued today: 2 exploit-attribution arms + 2 instrument replicates).
Explore/exploit: aux-off arm B + noise-floor replicates ≈
instrument/attribution (exploit-side); explore hours proper started
with the noise-draw chain (explore-side, ~9 h queued — pacing check
19:52Z says the draws-10 runs are ~5 h each, so the chain is
longer/richer than planned; still 94–99% util). Literature slice:
**~25 min spent ~19:35–20:00Z (trunk survey)** after four sessions
at 0 h — allocation back on cadence (skipped this session: bounded
infra item + the slice ran <1 h ago real-clock; next session takes
it). CPU-side: four consecutive all-CPU sessions while both GPU
chains ran (trunk survey, flow-vs-AR paired analysis, idea #2a
bucketing, ideas #18.1 hardening ~20:30–20:55Z real-clock) — the
no-idle-pauses rule in action. The #2a sim result is the rule paying
off concretely: a CPU measurement REPLACED a planned GPU screen
(predicted effect sub-threshold — charter §3).
