# 6. Aux attribution arms — `confirmed` (aux HELPS actions; results 2026-08-06)

*Tag: `aux-subgoals` · idea #6 · [index](../ideas.md)*

**ANSWERED 2026-08-06 04:2xZ
([results post](../posts/2026-08-06-box-batch-results.md)): the
pre-registered decision rule fired REAL — aux-off costs +0.462 panel
chunk MAE (CI [0.387, 0.537], 7.5× the 0.061 replicate threshold,
leave-one-repo-out coherent). The mainline "within noise" expectation
is falsified: aux supervision shapes the action representation.**
Arms: A-s0/s1/s2 7.7966/7.8052/7.7355, B 8.2989; σ_seed(chunk) 0.038
→ E4B adopt band = 0.15 (floor binds). Twist held up: B's first_mae
3.5009 BEATS aux-on (3.94–4.11) with cond-sensitivity 1.13 vs
1.86–2.00 and predictions 8% closer to state-copy — consistent with
the #11 state-shortcut mechanism; the state-reliance probe is the
falsification instrument (all four npzs now banked). Standing rule:
**aux stays ON in every future recipe; an aux-off arm needs a new
pre-reg citing this result.**

The still-owed paired aux-on vs aux-off arms (does aux supervision
shape the representation, separate from "does narrating help" — the
100k run answered only the latter). Pre-registered mainline
expectation: within probe noise (±0.3). Promoted to arm B of the
paired 40k run after the wrap census killed unwrap-at-load:
[pre-reg](../posts/2026-08-05-prereg-paired-auxoff-40k.md). Primary read:
paired per-frame panel chunk_mae A@40k vs B@40k. **Executing on the
4×H100 box since 17:12Z** (parallel arms + 2 control seed replicates
for the noise floor, with a pre-registered decision rule:
[box batch pre-reg](../posts/2026-08-05-prereg-box-batch-4xh100.md)).
2026-08-06 01:3xZ: all four arms trained (A-s0 formal probe 7.0882@40k,
B 7.702@40k; s1/s2 at their boundary), panel evals chaining; **results
instrument `fontaine/scripts/box_batch_results.py` landed + oracled
before the data** — frozen decision rule, mechanical headline-column
matching vs report JSONs, σ_seed → the E4B adopt band and rig slot 2;
anchors/degenerate/synthetic-inflation oracles all passed. When the
four npz+report pairs land: one command produces the results-post
numbers and both finalization amendments.

- **External replication + a new rung-(a) probe (deep read
  2026-08-07, [post](../posts/2026-08-07-pi05-deep-read.md)):** our
  +0.462 aux-off cost is the same result class as π0.5's Fig. 13 —
  "Implicit HL" (subtask data in training, no runtime decoding) is
  their second-best config, i.e. semantic co-training shapes the
  action representation. Their further increment we have NEVER
  tested: **explicit runtime hierarchy** — decode a subtask first,
  condition actions on it. We own the seam: the `[subgoal|…]`
  conditioning slot (heavily dropped out; planner-less default
  well-trained). **Probe (zero training, quiet-GPU window):** have
  the AR model generate its own subgoal per panel frame, feed it
  back through `[subgoal|…]`, score panel vs no-hint baseline
  5.8026. Validity check first — eyes on a table of self-generated
  subgoals before any scalar (the never-generated-subgoal scar).
  Owner anchor in favor: the 21:43Z steer notes aux subgoals
  generalize strikingly OOD.
- **Lit slice 2026-08-07 03:2xZ — external support + two design
  constraints for rung (a):**
  [Hi-VLA systematic study](https://arxiv.org/html/2606.10267v1)
  (2606.10267) benchmarks hierarchy design and finds explicit
  language subgoals beat flat VLA **largest on long horizon** (flat
  25.30% → naive hierarchy 40.56% → best 67.08%; short-horizon gap
  near zero) — so the probe's per-step decomposition should expect
  the gain concentrated in LATE-horizon chunk_mae, mirroring the #1
  banked-prediction pattern. Two carried constraints: (i) their
  planner/controller are separate models — SELF-generated subgoals
  (our probe) are untested there, so ours is a genuine increment,
  not a replication; (ii) subgoal refresh granularity mattered a lot
  (4–8 s best; model-predicted horizons WORST) — our panel probe
  conditions per-frame, sidestepping refresh policy, but any later
  rollout arm must pre-register the refresh rule. Their hardest-task
  failure mode ("VLMs tend to ignore image inputs as task becomes
  harder") is the #11 state-dominant-bias story from the hierarchy
  side.
- **Rung (a) PRE-REGISTERED 2026-08-07 ~03:5xZ
  ([pre-reg](../posts/2026-08-07-prereg-selfsubgoal-probe.md)):** four
  arms on AR-100k (banked planner-less 5.8026 / oracle-truth
  `[subgoal|…]` / self-generated fed back through the slot /
  narrated-subgoal-only), validity table gated go/no-go BEFORE any
  scalar, frozen Δ-reads + horizon decomposition, ≤ 8 GPU-h with the
  q4-subset fallback. Execution at the first quiet local-GPU
  window ≥ the draws10_t1 boundary + its frozen reads.
- **Instrument LANDED 2026-08-07 ~04:3xZ (oracle-gated, this
  commit):** `bijou.eval --subgoal-mode {oracle,self}` — oracle mode
  renders per-frame TRUE labels through the trained slot (label-less
  frames decode the baseline context); self mode is the two-pass
  loop sharing one model load (pass 1 planner-less
  `[generate|subgoal actions]` = the `_narrsubgoal` arm free, pass 2
  feeds the text back through `[subgoal|…]` on the fast path =
  `_selfsubgoal`). `--dump-subgoals` retains per-frame generations
  (identity triple → text); `--selfsubgoal-force-empty` is the live
  oracle-(i) no-hint-limit run (`_emptyhint`, never a self-arm
  read); report JSON records the mode. Stage-1 validity table:
  `fontaine/scripts/selfsubgoal_stage1.py` (60 stratified frames,
  generation-only — NO scalars before the gate). The four
  pre-registered oracles' CPU halves are pinned in
  `tests/test_selfsubgoal.py` (prompt-byte equality of the no-hint
  limit and label-less oracle frames; one shared rendering path;
  pass 2's request set excludes subgoal); the real-checkpoint halves
  run pre-launch per the pre-reg. No semantic deviation from the
  pre-reg → no amendment needed.
- **Lit slice 2026-08-07 ~04:0xZ — two escalation anchors (radar
  only, no design change to rung (a)):** (i)
  [CAC-VLA](https://arxiv.org/html/2607.04816v1) (2607.04816)
  conditions the action head on VLM-predicted latent actions with a
  LEARNED GATE modulating conditioning strength — and trains on
  ground-truth-encoded conditioning while inferring on
  self-predicted, exactly the truth-vs-self asymmetry our
  Δ_oracle/Δ_self split diagnoses; if rung (a) lands in the
  "Δ_oracle < 0 but Δ_self ≥ 0" cell (generation quality is the
  gap), a gated-strength variant is a named escalation candidate
  (needs its own pre-reg). (ii) π0.7 (via the
  [NVIDIA WAM post](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/))
  escalates explicit-HL beyond text: HL policy emits subtask
  instructions, a BAGEL-based world model renders them as subgoal
  IMAGES, the action expert conditions on obs+subgoal-image —
  reported "necessary for some dataset-bias-breaking tasks where
  no-subgoal variants fail", and subgoal images reportedly speed
  training by making action prediction near-inverse-dynamics. Our
  text-slot probe is the cheap first rung of exactly this ladder.
- **Lit slice 2026-08-07 ~20:0xZ — two same-day releases (both
  announced Fri 08-07, read hours old; page:
  [subgoal-sourcing](../papers/subgoal-sourcing-post-training.md))
  land two fresh directional priors on rung (a) BEFORE its read
  (no design change; the pre-reg is frozen):** (i) HiRoC
  (2608.05999) shows subgoal-source misalignment is a
  cold-start-scale effect — an executor trained to condition on
  task instructions collapses on planner-generated subgoals until a
  dedicated SFT alignment stage retrains it on (obs, subgoal,
  chunk) triples; RL does not recover it. Prior for the probe:
  Δ_self ≤ Δ_oracle; and if the probe lands "oracle helps, self
  doesn't," HiRoC's alignment-SFT joins CAC-VLA's gate on the
  named-escalation list (cheaper: an SFT recipe, no new
  architecture). No oracle-vs-planner ablation in the paper — our
  Δ_oracle/Δ_self split measures the decomposition they skipped.
  (ii) VLA-Talker (2608.05738) at matched evidence: generate+
  supervise text 81.5 / inject+supervise 89.7 / inject+action-only
  97.4 on LIBERO — supervised language regeneration of available
  evidence costs 15.9 pts + 4.6× latency. TENSION with our aux-on
  +0.462 result, resolved (our synthesis, flagged as such): harm
  mechanism = token imbalance + copy-work; our aux fields are
  sparse predictions of latent task structure, not verbose copies —
  "supervise sparse structure prediction, never verbose evidence
  regeneration." Their result also predicts the narrated arm
  (injected, never supervised) is safe-to-helpful.
- **Lit slice 2026-08-08 ~01:1xZ (read while the stage-2 arms
  decoded; page:
  [runtime-plan-verification](../papers/runtime-plan-verification.md))
  — the escalation ladder above rung (a), priced before the
  readout:** three published shapes of the runtime loop any
  escalation would enter. (i) SV-VLA (2604.02965): chunked
  macro-plan + a 17×-cheaper trained verifier carrying the plan
  intent, replan on L1 discrepancy > τ — the ablation that matters:
  verification WITHOUT a recovery path crashes 90.9%→15.5%, so any
  #6 refresh policy must budget the re-decode, not just the gate;
  threshold sensitivity (τ 0.1/0.2/0.4 → 83.1/90.9/77.4) is their
  named open problem. (ii) Do What You Say (2510.16281): embodied
  CoT faithfulness — text right, actions wrong — is the
  execution-side noise source our Δ_oracle/Δ_self split does NOT
  price; their sample-and-align fix needs outcome simulation we
  don't have, but alignment scoring over our existing draws
  machinery is the cheap fragment. (iii) VINE (2512.03913):
  test-time compute scales in candidate-SUBGOAL width (K=1→5:
  28.9→44.4% unseen, peak K=4) — "sample N subgoals, condition on
  best" is implementable depth-1 with the #1 batched-draws
  instrument; their load-bearing failure-aware value function needs
  failure-labeled demos we don't have. All rollout-granularity; no
  change to the frozen reads or E5 — these price the escalation
  pre-regs if Δ_self earns one.
- **2026-08-08 ~02:5xZ — RUNG (a) READ OUT
  ([results post](../posts/2026-08-08-selfsubgoal-results.md), reads
  by `selfsubgoal_results.py`, execution oracles green):** the
  trained `[subgoal|…]` slot is ALIVE — **Δ_oracle −0.290
  [−0.331, −0.225]** on 25,788 labeled panel rows (≈ the whole
  panel), concentrated 6× late-horizon (last-10 −0.480 vs first-10
  −0.081; E3 confirmed) — but the closed loop returns
  **Δ_self −0.018 [−0.052, +0.026]**, a statistical zero at ~3×
  decode cost (E2 point-wise only; E5's falsifier does not fire by
  the letter, the deployment claim is dead anyway). **Channel read
  (the probe's only significant self-text number): narr − self
  +0.043 [+0.023, +0.064]** — identical text, suffix voice loses to
  the condition slot; with stage-1's ~10/60 phase-offset rows this
  locates the bottleneck in single-frame PHASE ESTIMATION, not the
  channel and not language quality. Decode-noise floor context
  −0.0008 ± 0.016 quoted per amendment 1. Cost ~3.2 GPU-h ≤ 8 gate.
  **Disposition: rung (a) closed — "don't deploy"; ceiling −0.29
  banked as the escalation prize. Next rung, own pre-reg required:
  subgoal-DRAWS selection (decode N candidates, condition on
  best-scored — the [runtime-plan-verification
  slice](../papers/runtime-plan-verification.md)'s VINE width
  scaling, depth-1 via the #1 batched-draws machinery); heavier
  siblings: planner-side SFT (HiRoC direction), rollout refresh
  policies (SV-VLA shape, #16-gated).**
- **2026-08-08 ~03:2xZ — RUNG (b) PRE-REGISTERED
  ([pre-reg](../posts/2026-08-08-prereg-subgoal-draws.md)),
  scorer cell settled by a targeted lit check first
  ([Self-Certainty page](../papers/self-certainty.md),
  2502.18581 NeurIPS 2025):** subgoal-DRAWS selection — pass 1
  decodes 9 candidates (greedy + 8 sampled T=1, draws10_t1 seeding
  verbatim), the frozen verifier-free scorer is **self-certainty**
  (mean KL from uniform of the candidate's own decode
  distributions; zero extra forwards, no oracle access; the
  published best reward-free selector on open-ended text — chosen
  over likelihood/medoid, which stay record-only alternates from
  the same dumps). Two conditioned arms: **bon** (scorer pick,
  deployment-honest primary) and **ceil** (token-F1-vs-true-label
  pick, record-only) — the ceiling bounds EVERY scorer at this
  width, so a failed falsifier still adjudicates no-diversity
  (family closes) vs no-scorer (MG-Select-style masked-contrast or
  planner SFT earn a look; masked reference is OOD for us without
  trained image dropout, so it's an escalation, not the primary).
  Head-to-head read: paired per-frame (bon − self) vs rung (a)'s
  banked self npz; falsified unless CI95 entirely below 0.
  Stage-1 candidates table gates stage 2 (diversity ≥ 2 unique
  strings on ≥ 50% of frames, else the rung closes at table cost).
  Gate ≤ 6 GPU-h, q4 fallback; venue local behind the goldenticket
  R1 chain; instrument (subgoal draws + SC dump + two modes) lands
  oracle-gated before launch, draws-0 limit must reproduce the
  rung-(a) self arm bit-exact at matched composition.
- **2026-08-08 ~03:5xZ — RUNG (b) INSTRUMENT LANDED, oracle-green
  (CPU work session inside the goldenticket/molmo2 GPU-busy
  window):** `bijou.eval --subgoal-mode draws` — pass 1 decodes the
  greedy subgoal plus `--subgoal-draws` sampled candidates
  (`--subgoal-temperature`, draws10_t1 stable frame-keying
  verbatim) off ONE shared prefill via the new
  `ARSuffixDecoder.decode_value_line` (text-only value decode,
  per-step chosen/mean log-probs over the masked value softmax —
  the exact sufficient statistics for self-certainty, recomputable
  offline); a model-level assert pins candidate 0 byte-equal to the
  full pass's parsed subgoal. Pass 2 runs BOTH selection arms in
  one invocation (`_bonsubgoal` = frozen SC argmax, structurally
  label-blind; `_ceilsubgoal` = token-F1 vs true label, label-less
  rows render no hint). `--dump-subgoal-candidates` writes the
  machine-readable table (stats + LIVE picks + record-only
  likelihood/medoid alternates). Scorers pure in
  `bijou/eval/subgoal_scoring.py` (ties → lowest index, greedy
  first); read script
  `fontaine/scripts/subgoal_draws_results.py` mechanizes the frozen
  reads (Δ_bon + paired bon−self vs the banked rung-(a) self npz,
  Δ_ceil + ceil−self no-diversity/no-scorer adjudication,
  agreement records, horizon, first_mae mirrors) with an `--oracle`
  selftest: exact planted deltas, degenerate CI [0,0] + falsifier,
  11 abort branches — all green. 22 new tests
  (`tests/test_subgoal_draws.py`) incl. the REAL decode-loop
  oracle-i half on the tiny fixture model; `check.py` 489 green.
  Remaining before launch (execution item's preflight, GPU): draws-0
  bit-exact vs the banked self arm at matched composition +
  forced-empty = plain path.

**2026-08-08 ~10:2xZ — rung (b) CLOSED AT TABLE COST**
([close post](../posts/2026-08-08-subgoal-draws-stage1-close.md)):
preflight live oracles ALL GREEN (draws-0 bit-exact vs a fresh
matched-composition q4 self run; forced-empty bit-exact vs the
banked emptyhint), then stage-1 bar (a) FAILED — **11.5% of T=1.0
sampled draws derail into budget-truncated multilingual gibberish**
(55/480; greedy clean 60/60; 0.885⁸ binomial arithmetic reproduces
the 20/60 row rate). Bars (b)/(c) passed (97% diverse, 4.8% top
pooled string); clean candidates are subgoal-shaped with real
adjacent-phase alternatives; SC pick ≠ greedy 39/60; **SC never
picks a truncated candidate** (0/60, median rank 9/9). Δ_bon/Δ_ceil
stay unmeasured. Escalation queued
(`idea6-subgoal-draws-cleancand-prereg-draft`): truncation-robust
candidate list, own pre-reg required. Cost ~1.6 of 6 GPU-h.

**2026-08-08 ~14:1xZ — lit
([observation aliasing](../papers/observation-aliasing.md),
2605.14712 + 2605.14598):** the subgoal channel's external
validation shape is now pinned — frame-conditioned 9% →
intent-conditioned 45.8% on a benchmark built of aliased states,
plus DSSP's theorem that only extra conditioning can move the
reactive loss floor there. For the owner's meta-report: mine
aliased frames by NN-retrieval divergence (close in embedding,
divergent in ground-truth continuation) and test whether OUR
subgoal-conditioning delta concentrates on them; concentration =
disambiguation (the published mechanism), no concentration = the
channel is a style/dataset prior — either sharpens the report.

**2026-08-08 ~15:5xZ — frame-mining read EXECUTED
([post](../posts/2026-08-08-framemining-aliased-frames.md)): the
concentration test is a clean NULL.** Flagged (top-decile alias
score) − rest Δ_oracle = −0.003 [CI95 −0.205, +0.176], Spearman
ρ = −0.01 over 14,064 qualifying frames — the oracle-subgoal gain is
FLAT across the aliasing spectrum, except the least-aliased decile
gets almost nothing (−0.04, post-hoc observation). The subgoal slot
behaves as a uniform prior/guidance signal, not a disambiguator of
aliased observations; escalations (#6 rungs) should sell generation
quality and broad gain, not aliasing rescue. Instrument validated
independently: alias score ↔ baseline per-frame MAE ρ = 0.41,
flagged frames +29% baseline error (the DSSP floor, real on our
corpus, caveat: state-copy error elevated too — "ambiguous" partly
conflated with "dynamic").

**2026-08-08 ~17:5xZ — RUNG (b′) PRE-REGISTERED
([pre-reg](../posts/2026-08-08-prereg-subgoal-draws-cleanlist.md)),
the stage-1 close's named escalation:** clean-list subgoal-draws —
rung (b) inherited verbatim except budget-truncated candidates are
EXCLUDED from every scorer's eligible list (empty list → greedy
fallback, recorded); nucleus/lower-T rejected with reasons banked
(distribution change re-buys stage 1; dT monotonicity says lower T
trades away the diversity that gives width its value). Priors
verified on the banked stage-1 table BEFORE freezing: the filter
changes **0/60 SC picks and 0/60 ceiling picks** (structural not
behavioral — 40/60 rows carry ≥ 1 truncated candidate), filtered
bars all clear (60/60 rows keep ≥ 1 eligible sampled draw, 57/60
diverse, top pooled string 5.4%). Stage 1 is therefore CPU-free
(banked-table re-adjudication, byte-identity argument); stage 2 =
the two conditioned arms exactly as rung (b) froze them, Δ_bon /
Δ_ceil finally measured; falsifier inherited verbatim. Ceiling
≤ 5 GPU-h, local, post-close window behind the noise-ladder rung-2
obligations. Instrument delta small (`SelectedSubgoalPolicy._pick`
+ 4 new oracles incl. banked-table pick-invariance as a regression
fixture and a planted filter-binds world); execution item queued
(`idea6-subgoal-draws-cleancand-execution`).

**2026-08-08 ~16:2xZ — lit
([conditioning shortcuts](../papers/conditioning-shortcuts.md),
2602.24143 + 2605.20856):** the concentration null now has its
external family — "robust skills, brittle grounding" documents
conditioning channels consumed as coarse priors (region-prior
picking: compositional holdout 44%→0%; 10k→100k demos buys nothing),
DISC names the mechanism (task-state entanglement) and shows
structural decoupling fixes it. Missing cell for our slot named: a
**subgoal-swap sensitivity read** (wrong-episode subgoal at fixed
frame vs true-subgoal pass) would close the presence(−0.29) /
channel(+0.043) / CONTENT triangle for ~1 panel pass — meta-report
open-questions candidate, own pre-reg if it graduates.

**2026-08-09 ~00:2xZ — RUNG (b′) READ OUT, E6 FALSIFIED →
NO-SCORER
([results](../posts/2026-08-09-subgoal-draws-cleanlist-results.md)):**
run landed 23:52Z 08-08 on the pre-registered q4 fallback (4,301
rows, rate gate fired at launch); reads ran after the subset-join
path landed in `subgoal_draws_results.py` (draws10/energy join
convention, q4-shaped slice fixture in the oracle). Head-to-head
(bon − self) **+0.210 [+0.113, +0.312]** — entirely above 0, and
Δ_bon vs bare baseline **+0.142 [+0.027, +0.260]**: the SC pick
*anti-selects* (E3 failed too — bon below both self and ceiling).
Ceiling ALIVE: Δ_ceil **−0.250 [−0.353, −0.148]**, ceil − self
−0.181 CI clear, late-horizon −0.464 last-10% (the rung-(a) slot
signature). Filter did its structural job (eligible 8.06/9 mean, 0
fallback rows, 97.7% rows ≥ 2 unique texts) — width is not the
constraint, the scorer is. Alternates agree with SC ~40%, with the
oracle ~45%: nothing in the free family tracks the ceiling. All
execution oracles green (picks byte-match offline recompute,
state-copy byte-match on joined rows). ~1.4 GPU-h ≤ 5 gate.
**Selection family closed on scorer-free tricks; named next rungs
(each its own pre-reg): learned verifier (RoVer shape,
chunk-as-unit), fields-probe ranker, or distillation from the
4,298 dumped picked-vs-oracle pairs.**

- **Fields-panel input banked 2026-08-09 00:49Z
  ([results](../posts/2026-08-09-molmo2-fields-panel-results.md))**:
  Molmo2@60k narrated-field accuracies vs AR-100k — holding 0.897
  (0.807), progress MAE 0.059 (0.062), event 0.880 (0.878),
  **visible slot-set 0.819 vs 0.319** (+0.50 on the strictest
  metric, with *more* frames parsed 8,981 vs 8,260). Narration
  still costs at decode (paired +0.083, cost concentrated on
  failure-labeled frames +0.50): the aux head stays a training-time
  asset, not a decode-time one. Relevance here: any learned scorer
  rung (fields-probe ranker especially) gets a far better scene
  reader on the Molmo2 trunk than the AR-100k numbers implied.

- **Lit 2026-08-09, two escalation-map inputs**: (1)
  [VLAFlow](../papers/vla-training-objectives.md) independently
  replicates aux-is-load-bearing (verbalized-action co-training +3.5
  LIBERO-Plus) and names a NEW aux family we haven't tried —
  **future-latent alignment** (frozen V-JEPA-2 tower, predict the
  +8-frame latent; their single biggest control-transfer lever,
  +8.6 WidowX). Hook shared with #17; needs its own pre-reg + tower
  choice. (2) [Guided Action Flow](../papers/qguided-flow-critic.md)
  adds a third learned-scorer shape to the NO-SCORER escalation map:
  continuous **gradient guidance** from a chunk critic (MLP over
  obs + chunk + frozen-VLM task embedding, success-to-go labels,
  ensemble-disagreement gate) — flow-side only, sidesteps fixed-K
  width; weak-label success-to-go over banked episodes is the
  no-new-GPU label route; held-out evidence thin (+2.5 pts / 40
  episodes, no best-of-N baseline — our banked ceilings are the
  missing comparison).

- **Subgoal-swap content read PRE-REGISTERED 2026-08-09
  ([pre-reg](../posts/2026-08-09-prereg-subgoal-swap.md))**: the
  §6.1 triangle-closer — oracle arm re-run with an episode-level
  derangement of segment labels (format-valid, content-wrong);
  frozen 3-row table adjudicates whether learned-scorer
  escalations are even coherent before any of them earns a
  pre-reg. Instrument delta (`--subgoal-swap-seed` + 4 oracles) is
  the prerequisite; ~1.2 GPU-h ≤ 3, local, any quiet window.

- **Subgoal-swap READ OUT 2026-08-09 03:5xZ
  ([results](../posts/2026-08-09-subgoal-swap-results.md)) — MIXED,
  record-only, and the triangle is closed**: wrong-but-plausible
  words still help (Δ_swap **−0.113** [−0.161, −0.060]) but truth
  beats them clearly (paired swap−oracle **+0.166** [+0.127,
  +0.205]) — the −0.290 slot value decomposes ~40% format/prior
  floor + ~60% content margin. Late-horizon dive reproduced in both
  arms (oracle −0.480, swap −0.175 last-10 — NOT flat, so the
  format floor compounds too). Scorer escalations stay coherent
  (content IS consumed) but any scorer rung must now be costed
  against the free any-plausible-words floor, and its prize is the
  ~0.17 content margin. Caveat recorded: 8.4% of swapped rows drew
  a textually-true donor label (bias runs against the content
  reading, which won anyway).

- **Rung (c) pre-reg DRAFT landed 2026-08-09 05:3xZ
  ([draft](../posts/2026-08-09-prereg-subgoal-mcselect.md)) + read
  script pre-data (mcselect_results.py, oracle-gated, check.py
  559)**: the scorer-side escalation the (b′) routing licensed —
  masked-contrast (MG-Select form) selection, zero training:
  KL(conditioned ‖ masked reference, τ=4) over teacher-forced action
  tokens, our 50% subgoal-dropout training supplying the well-trained
  masked path. Re-ranks the EXACT banked (b′) width (4,301 q4 rows ×
  candidates, sha-pinned) so ceiling (−0.250) / SC-anti-select
  (+0.142) / floor (−0.113 free words) all stand as comparators.
  Falsifier = E6 mirror (mc − self CI95 < 0); an anti-select read is
  a second strike and closes the zero-training scorer family.
  ≤ 4 GPU-h local. Remaining before launch: producer instrument
  (candidates-file injection + in-model KL) + finalization stamp.
  Candidate 2 (TOPReward history probe) escalates only on a
  phase-specific failure.

- **Rung (c) READ OUT 2026-08-09
  ([results](../posts/2026-08-09-mcselect-results.md), run 09:12:36Z
  → 10:20Z, ~1.1 GPU-h ≤ 4 gate)**: masked-contrast (MG-Select form,
  τ=4, KL from the decode's own logits vs the tempered planner-less
  reference) **ANTI-SELECTS** — primary (mc − self) **+0.31317 CI95
  [+0.19962, +0.42894]**, the harder strike vs SC's +0.210; Δ vs
  bare +0.245 (worse than no subgoal); capture fraction **−1.73**;
  late-horizon signature **+0.385** (the ceiling's −0.464 slot,
  inverted — max-KL candidates are disruptive, not phase-right);
  oracle-pick agreement 14.4% ≈ chance at width 9 while 66% of
  picks differ from greedy (decidedly not inert). Execution oracles
  green; pred_masked composition-flip count 1207/4301 reproduced
  the rung-(a) amendment-1 figure exactly. **Kill rule executed:
  the zero-training scorer family CLOSES for this trunk** —
  [RoVer](../papers/rover-learned-verifier.md) /
  [Q-guided](../papers/qguided-flow-critic.md) shapes now need
  their own affirmative case; candidate 2 (TOPReward history probe)
  does not auto-open (its trigger was flat-late-horizon, the
  observed failure is active anti-selection). The (b′) ceiling
  stands (−0.250 vs bare) — the gap is a scorer gap, twice
  measured. Free follow-up queued: record-only KL-vs-quality
  post-mortem on the banked [N,C] KL + [N,C,S,D] error dump.

- **Post-mortem map READ 2026-08-09 same-day (record-only, NOT
  pre-registered, no decision rides on it;
  [addendum](../posts/2026-08-09-mcselect-results.md) with charts;
  `mcselect_postmortem.py`, oracle-gated, raw sidecar npz banked)**:
  the closed family's failure decomposed on the banked dump. (1)
  **KL is rank-noise**: per-row Spearman(KL, err) **+0.012 CI95
  [−0.005, +0.029]** (frac-positive 0.503), oracle-best (frame-error
  best eligible) sits UNIFORMLY on the KL axis — mean normalized
  rank 0.498 vs 0.5 null, mild excess at BOTH extremes (top1 17.4%,
  bottom1 16.8%, null 12.6%) ⇒ argmin-KL would fail too; the
  +0.313 harm is magnitude-driven (row-centered value-level Pearson
  +0.126 vs rank-level ~0 — winner's curse on the far tail with
  heavy-tailed damage; MC's pick is oracle-best MORE often than
  SC's, 25.6% vs 23.6%, while losing harder on MAE). (2) **SC was
  the better axis all along**: −0.030 [−0.046, −0.014], right-signed,
  oracle-best at SC-top 30.1% vs 12.6% null — real signal ~6× too
  weak to survive an argmax over width ~8. (3) **Axes mutually
  uncorrelated** (Spearman(KL, SC) +0.032) — the family failed twice
  INDEPENDENTLY; "family" was the right closure unit. Calibration
  number for any learned-verifier case: the ceiling (−0.250 vs
  bare) is real and zero-training rank signal toward it tops out at
  |rho| ≈ 0.03 — a verifier must argue for ~an order of magnitude
  more before its GPU-hours are priced. 153 constant-KL rows
  excluded from rho reads; eligible width 4–9, median 8.
