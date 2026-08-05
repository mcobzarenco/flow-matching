# Now

*Updated 2026-08-05 ~21:51–22:2xZ (real `date -u`) — work session: **THE
MODE-AVERAGING FAIRNESS INSTRUMENT IS EXECUTION-READY — the owner's
21:49Z three pre-declared reads now have a data path**
([Amendment 1](posts/2026-08-05-draws-fairness-amendment.md) on the
noise-draw pre-reg). **Instrument finding en route: the pre-declared
"draws-10 per-draw dumps" could not have existed** — the draws chain
passes no dump flag, and `--dump-predictions` stores the
*post-average* prediction; per-draw chunks died inside
`predict_with_text`. Landed: (1) **`bijou.eval --dump-draws`** —
pre-average `[frames, draws, chunk, dim]` npz + full identity columns
(#18.1 conventions) + standalone scoring-semantics scalars, loud
constraints (needs `--checkpoint` + `--sample-draws > 1`), threaded
through the DDP shard merge; `collapse_draws` factored pure +
unit-tested (dump averages back byte-identical to the prediction —
the mean is still taken once on the full stack). (2) **Probe frozen**:
stride-7 core subset plan (2,458 frames / 792 repos, deterministic
builder) + launcher `~/eval_flow80k_drawsprobe_dump.sh` (draws=10
heun-30, ~30 min 1×GPU, GPU-quiet guard, auto-runs the analysis;
E1-style gate: draw-0 frame-MAE drift vs the banked flow npz < 0.05).
(3) **`fontaine/scripts/draws_fairness.py`** — the three reads with
the report's exact valid-element pooling; joins probe rows to the
banked AR/flow npzs on concat `index` with hard row-agreement asserts.
**Oracles: banked AR-100k panel rebuilt through the edited scoring
path 12/12 cells d=0 (incl. 5.802585); degenerate draws=1 validation
reproduces 6.6232 EXACTLY on reads 1+2 with all-zero dispersion.**
check.py green (184 tests, +5). Launch: first quiet local-GPU
boundary after the draws chain (~06–09Z), before the results post.
Babysits 21:52/22:1xZ: box ×4 healthy @20.3–23k, 0.39–0.42 s/step
(one benign 5.6 s save blip on B; B aux-off total 3.487@23k, still
at/below control action losses 3.63–3.68); draws run 2 @17.2k/25.8k
99% util (ETA ~23:50Z, then runs 3–5). **OWNER STEERING 21:52–21:58Z
(replied 22:2xZ, monitor polling 30 s): (a) E4B SCREEN PICKED** —
AR-100k on the freed 4×H100, matched parameters with the E2B
AR-100k (recipe verified: `--batch-size 12`/GPU DDP4 = effective 48
— owner remembered 10; grad-accum fallback to effective 48 if E4B
OOMs), gates = the MAE curve over time vs the banked E2B curve +
mid-run panel evals with pre-registered bands. **The E4B pre-reg is
the next CPU work item.** (b) Image-embedding budget = follow-on
ablation arm on the winning trunk (banked in #17, pairs with #11
grounding). (c) Owner measured FAST round-trip ≈ error-free
(+attachment) — quantization not the AR binding limit, banked in
#8; fits the paired late-horizon read. GPUs busy + CPU queue
non-empty (**E4B screen pre-reg next**, stage-2 sign pre-reg, lit
slice two sessions overdue) → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-05 21:47–21:5xZ (real `date -u`) — tick: **both chains
healthy; PROBE GATE <9@30k EFFECTIVELY MET EARLY ON TWO CONTROLS —
s1 8.991@18k, s2 8.982@18k, the first sub-9 probes of the batch**
(A-s0 9.15@18.5k → 9.31@19k noisy bounce; B aux-off 9.58@20.5k — B
is now the *trailing* probe arm despite being ~2k steps ahead,
further strengthening the B-early-lead-was-transient read). Box ×4
@18.2–20.7k, 0.377–0.399 s/step, util 58–62%, ~71–74 GiB, grad norms
nominal: A-s0 total 3.956@19.5k (action 3.642), s1 4.16@18.5k
(action 3.816, one noisy line off 3.906/3.652), s2 4.01@18.2k
(action 3.728), B aux-off total 3.665@20.66k — the action-loss
margin keeps oscillating around zero at line noise (B 3.665 vs
A-s0's action 3.642). Draws run 2 @14.75k/25.8k, 99% util. **LIVE
EXCHANGE: owner 21:48:14Z** (landed seconds after the cursor read)
— challenge on the flow-vs-AR crossover: k≤3 @ 30 fps ≈ 100 ms, not
a realistic replan horizon (inference would need <100 ms). Replied
21:5xZ agreeing with the arithmetic and the thrust: deployable
regime is k≥5 where AR wins today; draws-10 is attribution, not a
deployable config (N draws multiply decode cost); flow's residual
case = first_mae grounding edge + (if draws close the gap) SnapFlow
1-NFE distill + small N; otherwise attribution screens run on the
AR recipe. **Steering applied: weight the AR-side arm in the
limit-attribution plan.** Owner 21:49Z follow-up: **is MAE unfair
to flow — mode-averaging-forgiving?** Replied: yes it's the right
worry and it's measurable tonight on CPU from the draws-10 per-draw
dumps — three pre-declared reads for the results post: (1)
mean-of-draws MAE (ensembling ≈ manufacturing the mode-averaged
predictor; closes gap ⇒ deficit was punished dispersion), (2)
best-of-N MAE (oracle mode-match bound on
'sampled-a-different-valid-mode'), (3) dispersion-conditioned
deficit (the queued unimodality probe — deficit concentrating on
high-disagreement frames = the unfair-penalty signature).
Circumstantial fingerprint already present: flow wins horizon 0–1,
deficit grows with horizon + motion quartile. Honest limit stated:
MAE can't settle actual performance, and the owner's comm-MAE→rig
bridge was built on AR checkpoints — if flow is being punished for
multimodality, the comm holdout needs a distributional column
(best-of-N / energy distance) before it can rank flow arms. Monitor
polling the channel at 30 s while the exchange is live. GPUs busy + CPU queue non-empty (stage-2
pre-reg, lit slice due, E4B screen launcher after box reads) →
`run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 ~21:5xZ (real `date -u`) — work session: **IDEAS #16
INSTRUMENTS LANDED — the rig benchmark is execution-ready up to its
two slots** ([Amendment 1](posts/2026-08-05-prereg-rig-fewshot-benchmark.md)).
Plan frozen (`plans/rig_fewshot_v0_k4l2.json`: 12 holdout eps — v2
{1,2,3,6,11,15,20,24,25,30,41} + clean {2}, 48 core + 24 labeled,
draws through `build_plan` itself); **mechanism amendment posted
before any model number**: the draft's bespoke SeedSequence holdout
draw could not feed the leakage checker (its self-check demands the
codebase-native split — #18.8's anti-drift assert working as
designed), so the holdout is the native split at 0.212/seed 16 =
exactly the pre-registered 11+1 counts. Subsets materialized +
verified (`~/datasets/rig_fewshot_v0/`: n10 6,223 / n25 15,881 / n45
29,107 frames; videos hardlinked → bit-identical pixels, verified on
shifted mid-file decode both cameras; judgments episode-remapped;
stats recomputed, oracle worst |Δ| 1.2e-4 vs both shipped
stats.json). **Leakage certs ×3 PASSED** (first production consumers
of the #18.8 provenance path; doctored-provenance negative control
FAILS loud). **Wrap census CLEAN on both rig repos** (hygiene gate 1
done). Remaining before launch: launcher gen + finalization
amendment (slots 1–2) after tonight's box reads. Babysit ~21:50Z:
box ×4 healthy @17.4–19.8k, 0.38–0.40 s/step (one benign 10.0 s
save-boundary blip on s1), **B aux-off total 3.58–3.60 @19.8k — back
below every control's action loss (3.64–3.79)** after the 21:24Z
margin-zero read; draws run 2 @13.6k/25.8k @100% util. No Discord
traffic. GPUs busy + CPU queue non-empty (launcher gen, stage-2 sign
pre-reg, lit slice due next session) → `run_work_next` armed per
no-idle-pauses.*

*Previous update 2026-08-05 21:24Z (real `date -u`) — tick: **both chains
healthy, no Discord traffic** (only our own #16 pre-reg post). Box ×4
@16.2–18.5k, 0.374–0.382 s/step, util 57–99%, ~71–74 GiB, grad norms
nominal: controls A-s0 total 4.04 @17.3k / s1 4.09 @16.5k / s2 4.08
@16.2k (action 3.64–3.80), B aux-off total 3.686 @18.5k — **the
aux-off action-loss margin has closed to ~zero at line noise** (A-s0's
last line action 3.642 sits below B's 3.686 total; per-line noise
~0.1). Probes now one interleaved band 9.2–10.2: **s1 9.216@16k — new
best across all arms**, A-s0 10.21@17k (noisy bounce off its
9.4472@16.5k), s2 9.93@16k, B 9.80@18k (off its 9.59@17.5k) — the
B-early-lead-was-transient read is now strongly supported; probe
noise between consecutive evals is ±0.5–0.8, so the <9@30k gate is
the next real checkpoint. Draws run 2 @12.4k/25.8k @100% util on the
~5 h pacing. GPUs busy + CPU queue non-empty (#16 follow-on
instruments: subset materializer + plan builder; stage-2 sign
pre-reg; lit slice due) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 ~21:3xZ (real `date -u`) — work session: **IDEAS #16
PRE-REG DRAFT POSTED — the north-star benchmark design is frozen**
([post](posts/2026-08-05-prereg-rig-fewshot-benchmark.md)). Few-shot
rig-transfer v0: sample-efficiency curve MAE(N), N ∈ {0,10,25,45},
over the 57 owner rig episodes — 12-ep fixed holdout
(SeedSequence(16)), nested train subsets as **materialized derived
corpora with the #18.8 leakage gate** (the first consumer of that
work); owner `run_ft_rig.sh` protocol constants, 1×H100 B10,
best-checkpoint-at-200 selection; co-primary chunk_mae + first-4
pooled MAE (k fixed per the flow-vs-AR crossover); 3·σ_ft decision
rule with σ_ft from N25 seed replicates + an honest degrade rule if
σ_ft > 0.5. **Key design find: flow-80k is contaminated as a
few-shot subject** (rig data in its pretrain mix per the owner's
`run_ft_rig_flow.sh` header) — eligibility gate pre-registered;
rcond-100k and all four box arms qualify. Two slots (init selection,
E5 noise scale) fill by finalization amendment after tonight's box
reads; execution ≈ one evening on 1 GPU at the first quiet boundary.
check.py green (179). Babysit 21:16Z: box ×4 healthy @15.5–17.8k,
0.376–0.387 s/step — **A-s0 probe 9.4472@16.5k, first control under
9.5 and now below B's 9.59@17.5k** (the B-early-lead-was-transient
read strengthens); B aux-off 3.689@17.8k still at/below every
control's action loss (controls 3.77–3.91); draws run 2 @11.5k/25.8k
@100% util. No Discord traffic. GPUs busy + CPU queue non-empty
(#16 follow-on instruments: subset materializer + plan builder;
stage-2 sign pre-reg) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 21:14Z (real `date -u`) — tick: **both chains
healthy, no Discord traffic; s1 watch item RESOLVED** (log had
advanced 15500→15720 — it was the probe/save boundary as suspected,
not a stall). Box ×4 @15.5–17.5k, 0.377–0.387 s/step, util 63–99%,
grad norms nominal: controls A-s0 total 4.12 @16.5k / s1 4.11
@15.7k / s2 4.23 @15.5k (action 3.73–3.91), **B aux-off total
3.685 @17.5k — still at/below every control's action loss** (margin
~0.05 vs s1's 3.73, continuing to narrow). Probe: **B 9.59@17.5k —
B's first sub-10 probe** (joins s2's 9.92@14.5k), trending toward
the <9@30k gate. Draws run 2 @11.4k/25.8k @100% util on the ~5 h
pacing. GPUs busy + CPU queue non-empty (#16 rig benchmark pre-reg
draft, stage-2 sign pre-reg) → `run_work_next` already armed per
no-idle-pauses.*

*Previous update 2026-08-05 21:12Z (real `date -u`) — work session: **IDEAS
#18.8 LANDED (leakage identity branch verified, not assumed) + the
standing literature slice taken.** #18.8
([journal](journal.md)): `bijou.eval.leakage`'s same-repo-id branch
now asserts episode-count equality vs the panel copy AND compares
per-episode length fingerprints (jsonl v2 / parquet v3; asymmetric
metadata fatal; same-dir shortcut) — a filtered-and-renumbered
corpus keeping its repo id can no longer certify a false PASS.
Mismatch = SystemExit demanding `source_provenance.json`. +4 tests
(179 green), `check.py` green; full-corpus identity cert re-run
PASSED (5267 radioactive / 47240 checked, 4.1 s); mutated-count
production copy fails loud. **Unblocks derived-corpus training
(#9, #13 repair).** Literature slice (~20 min, banked in ideas +
journal): **SnapFlow** (2604.05656) — self-distill flow VLAs to
1-NFE, no teacher, ~12 h/1 GPU, π0.5 1-step ≈ 10-step teacher,
SmolVLA-validated ⇒ ideas #12's distillation leg is now an
in-budget arm; **LoRA-π0** (2607.10172) — r=32 saturation, frozen
vision encoder degrades (external support for #11's
grounding-bottleneck read) ⇒ ft-protocol arm for #16. Babysits
21:04/21:10/21:11Z: box ×4 healthy @15.2–17.5k, 0.38–0.40 s/step
(**B aux-off total 3.69 @17.5k — still at/below every control's
action loss (3.69–3.79), margin narrowing**; watch item: s1 log
paused @15500 across two polls ~90 s apart, util fine — likely the
15.5k probe/save boundary, next tick verifies advance); draws run 2
@11.1k/25.8k on the ~5 h pacing. No Discord traffic. GPUs busy +
CPU queue non-empty (#16 rig benchmark pre-reg draft, stage-2 sign
pre-reg) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 21:05Z (real `date -u`) — tick: **both chains
healthy, no Discord traffic** (the one new message was our own #18.2
post). Box ×4 @14.5–16.7k, 0.37–0.39 s/step (one benign 10.2 s
probe/save-boundary blip on s2): controls A-s0 total 4.22 @15.5k /
s1 4.23 @15k / s2 4.00 @14.5k (action 3.83–3.90), **B aux-off total
3.66 @16.7k — still below every control's action loss.** Probe:
**s2 9.92@14.5k — first arm under 10**, trending well toward the
<9@30k gate. Draws run 2 at 10.3k/25.8k @99% util on the ~5 h
pacing. GPUs busy + CPU queue non-empty (#16 rig benchmark pre-reg
draft, #18.8 leakage assert, stage-2 sign pre-reg, literature slice
due) → `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 ~21:25Z-labeled (really ~20:55Z) — work session: **IDEAS
#18.2 (FLOW-NOISE STABLE-TRIPLE RESEED) LANDED BEHIND A FLAG, BREAK
PRE-REGISTERED** ([amendment](posts/2026-08-05-noise-reseed-prereg.md)).
The deep-dive's top finding fixed at the design level: `bijou.eval`
gains `--noise-key {index,stable}` — `stable` keys each frame's flow
noise to blake2b(repo_id, episode, frame) through a numpy SeedSequence
(128-bit keying, draw number as entropy: no torch 32-bit manual_seed
trap, no DRAW_SEED_STRIDE, no birthday collisions), making flow
numbers corpus-composition-invariant. Default stays `index`
(byte-identical to every banked anchor) until the pre-registered flip:
first anchor boundary after the box reads, one flow-80k panel re-bank,
decision band `6.6232 ± 3·max(0.045, empirical σ_draw from tonight's
draws chain)`, state-copy/AR bitwise-identity as hard controls.
Report JSON + banner now record `noise_key`; SmolVLA path threaded;
Q3 forced pass verified to share noise under both keyings. **Oracle:
AR-100k panel recomputed bit-exact through the edited path (12/12
cells d=0 incl. the 5.8026 anchor)**; 7 new unit tests (175 green),
`check.py` green. Babysits en route (~20:45Z, ~21:20Z): box ×4
healthy @14.5–16.5k, 0.38–0.40 s/step — **B aux-off total 3.80
@16.5k, still below every control's action loss** (3.83–3.88
@14.5–15.5k); draws run 2 at ~10k/25.8k @99% util on the ~5 h pacing.
No Discord traffic. GPUs busy + CPU queue non-empty (#16 rig
benchmark pre-reg draft, #18.8 leakage assert, stage-2 sign pre-reg)
→ `run_work_next` armed per no-idle-pauses.*

*Previous update 2026-08-05 20:38Z (real `date -u` — NB the previous entry's
"~20:55Z real clock" label was stamped ~20 min ahead of reality;
clock-label skepticism stays warranted) — tick: **both chains
healthy, no Discord traffic.** Box ×4 @12.5–14.5k, 0.37–0.42 s/step
(one benign 11.3 s save-boundary blip on s2), util 64–94%, ~71–74
GiB: controls A-s0 total 4.26 @13.5k / s1 4.28 @13k / s2 4.36
@12.5k (action 3.91–4.00), **B aux-off total 3.81 @14.5k — still
below every control's action loss.** Probes converged into one band:
A-s0 10.55@12.5k (10.99@13k), s1 10.69@12.5k, s2 10.18@12.5k, B
10.95@14k — B inside the control envelope, all trending toward the
<9@30k gate. Draws run 2 at 7.5k/25.8k @97% util on the ~5 h pacing.
GPUs busy + CPU queue non-empty (#18.2 reseed design, #16 rig
benchmark pre-reg draft, #18.8 leakage assert) → `run_work_next`
armed per no-idle-pauses.*

*Previous update 2026-08-05 ~20:55Z-labeled (really ~20:35Z) — work session: **IDEAS
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

- **21:43Z (conversational, replied 21:45Z, exchange live): MAJOR
  REWEIGHT — #16 rig-benchmark execution PARKED, short-term focus =
  comm-holdout MAE + limit attribution.** Owner: rig datasets
  small/noisy, 12-ep holdout high-variance; a better rig dataset
  comes later; "lower MAE on the comm holdout always translated to
  good fine-tunes on my rig." Attribution questions to attack:
  bigger trunk / bigger image embeddings / video-trained trunk /
  is flow even needed vs pure AR — these map to ideas #17 (E4B →
  Molmo2 → InternVL3.5, V-JEPA 2.1), #11 (grounding; owner's
  failure anecdote is gripper *placement*, i.e. grounding), #12/#1
  (flow-vs-AR + ensembling). Aux anecdote banked: 4k ft on AR-100k
  produced sensible subgoals for a fully-OOD instruction (USB-C
  cable / terrarium) — the language-generalization north-star
  behavior exists already. Proposed in-channel: E4B trunk-swap
  screen as the next pre-reg after box reads (or grounding arms —
  awaiting owner pick). #16 instruments stay banked
  (corpus-agnostic, minutes to re-run on the future dataset).

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
8. **Ideas #16 rig benchmark**: ~~pre-reg draft~~ **POSTED ~21:3xZ**;
   ~~subset materializer + plan builder + leakage certs + wrap
   census~~ **LANDED + CERTIFIED ~21:5xZ** (Amendment 1 on the post;
   n10/n25/n45 under `~/datasets/rig_fewshot_v0/`). **EXECUTION
   PARKED per owner 21:43Z** (instruments banked, corpus-agnostic);
   launcher gen + finalization deferred until the better rig dataset.
9. **NEW (owner 21:43Z): comm-MAE limit-attribution front** — next
   pre-reg after the box reads: E4B trunk-swap screen proposed
   in-channel (alternative: #11 grounding arms — owner pick
   pending). The freed 4×H100 at ~02Z goes here, not to rig ft.
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
it). CPU-side: five consecutive all-CPU sessions while both GPU
chains ran (trunk survey, flow-vs-AR paired analysis, idea #2a
bucketing, ideas #18.1 hardening, ideas #18.2 reseed-behind-flag
~20:45–21:25Z real-clock) — the no-idle-pauses rule in action. The
#2a sim result is the rule paying off concretely: a CPU measurement
REPLACED a planned GPU screen (predicted effect sub-threshold —
charter §3). #18.2 keeps the pattern: the instrument break is fully
implemented + pre-registered on CPU; the flip costs one token + one
eval at a boundary we already visit. Sixth consecutive all-CPU
session (#18.8 leakage identity assert ~21:05–21:12Z) continues it.
Literature slice: **~20 min taken this session (~21:10Z real-clock,
SnapFlow + LoRA-π0 — both banked into ideas #12/#16 with numbers)**
— standing allocation back on cadence. Seventh consecutive all-CPU
session (~21:16–21:3xZ): the #16 rig-benchmark pre-reg draft — the
north-star instrument is now designed and posted before the box
reads that fill its slots land (skipped lit slice this session: ran
<30 min ago real-clock; next session takes it). Eighth consecutive
all-CPU session (~21:30–21:5xZ): the #16 instruments — plan frozen,
subsets materialized + leakage-certified, wrap census clean; the
benchmark can now execute the moment the box reads fill its slots,
instead of losing a session to prep at the quiet boundary (skipped
lit slice again: ran ~45 min ago real-clock; next session takes it).
Ninth consecutive all-CPU session (~21:51–22:2xZ): the draws-fairness
instrument — the owner's live 21:49Z challenge went from
in-channel pre-declaration to execution-ready (dump path + frozen
probe + validated reads) before the data it needs finishes
computing; the probe itself costs ~30 GPU-min instead of a ~5 h
full-panel repeat (skipped lit slice: owner-steered item took the
session; the slice is now two sessions overdue — next session MUST
take it).
