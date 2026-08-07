# Ideas

The backlog. Every entry: hypothesis, expected effect, cost, cheapest
falsification. Seeded 2026-08-05 from charter §8 (which distills the
mainline ledger, `docs/architecture.md` §7–8); ordering ≈ expected
information × cheapness. Status tags: `queued` / `screening` /
`running` / `confirmed` / `falsified` / `parked`.

## 1. Inference-time noise-draw ensembling — `queued`, natural first

- **Hypothesis:** mean-of-N noise draws through the flow expert cuts
  panel MAE substantially in the unconstrained class (mainline
  measured 5.30°→2.88° on motion frames for a ft'd model,
  mean-of-10); the stage-2 flow-on-AR-trunk lineage (6.623 panel)
  should benefit similarly.
- **Expected effect:** large on the flow lineage's panel number;
  unconstrained-class only (charter §2) until distilled.
- **Cost:** ~20 lines eval-side + one eval burst per N in {1,5,10}.
  Zero training.
- **Falsification:** paired panel eval, same checkpoint, draws
  stated. Check unimodality of draws first (averaging multi-modal
  draws is wrong): per-frame draw spread on a few hundred panel
  frames. If mean-of-10 does not beat single-draw beyond the panel's
  pairing noise, kill.
- **Open sub-question:** an AR-family analogue (temperature/nucleus
  chunk ensembles, chunk-level medians) — separate screen.
- **Instrument note (2026-08-05):** upstream already ships
  `bijou.eval --sample-draws N` (a16e65a) — verify its semantics
  (independent draws? mean-of-N in action space?) before writing any
  code; the pre-reg's eval-side work may reduce to a flag.
- **Paired-analysis prediction (2026-08-05 ~20:15Z, before the
  draws-10 numbers land):** flow's deficit vs AR is a monotone
  horizon-divergence (crossover at step 2; +1.2 by step 40 —
  [post](posts/2026-08-05-flow-vs-ar-paired.md)). If per-draw
  spread grows along the horizon, mean-of-N should close the
  *late-horizon* deficit preferentially: chunk_mae moves a lot,
  first_mae barely. Score the draws-10 run per-step, not just
  pooled.
- **Fairness reads pre-registered (2026-08-05 ~22:1xZ,
  [Amendment 1](posts/2026-08-05-draws-fairness-amendment.md)),
  from the owner's 21:49Z is-MAE-unfair-to-flow challenge:**
  instrument finding — per-draw chunks never left the process
  (`--dump-predictions` stores the post-average), so
  `bijou.eval --dump-draws` landed (tests + bit-exact scoring
  oracle), a 2,458-frame stride-7 probe plan + launcher
  (`~/eval_flow80k_drawsprobe_dump.sh`, ~30 min 1×GPU) is frozen,
  and `fontaine/scripts/draws_fairness.py` computes the three
  pre-declared reads (mean-of-draws / best-of-N / dispersion-
  conditioned deficit; degenerate draws=1 validation reproduces
  6.6232 exactly). Launch at the first quiet local-GPU boundary
  after the draws chain.
- **FIRST DRAWS-10 PANEL READ (2026-08-05 23:31Z, run 2 of the
  chain, full 25.8k-frame panel): chunk_mae 5.365 / first_mae 1.424**
  vs single-draw 6.6232/1.9331 (−19%/−26%) — **mean-of-10 flow beats
  the AR-100k anchor (5.8026/2.1431) on both columns** (unconstrained
  class: 10× NFE; charter §2 caveat until distilled — #12 SnapFlow
  leg). Banked-prediction check: "chunk_mae moves a lot" ✅;
  "first_mae barely" ❌ — first_mae moved 26%, so the gain is NOT
  purely late-horizon; per-step decomposition promoted to a required
  read in the results post (after runs 3–5 + the fairness probe).
- **Read 4 pre-declared (2026-08-05 ~22:5xZ,
  [Amendment 2](posts/2026-08-05-draws-fairness-amendment2.md), from
  the lit slice):** the **energy score** (RMS-normalized, valid-
  element mask, N=10 vs AR's degenerate N=1) — a strictly proper
  scoring rule where neither mode-averaging nor scatter wins for
  free; the principled middle between MAE and the best-of-N oracle
  bound, and the candidate distributional column for ranking flow
  arms on the comm holdout. Source: Energy Policy
  ([2510.12483](https://arxiv.org/abs/2510.12483)) trains on it;
  we take the metric, computable on CPU from the same
  `--dump-draws` npz. `read4_energy_score` in `draws_fairness.py` +
  degenerate draws=1 validation must land BEFORE the probe npz is
  opened (next CPU work item alongside the E4B launch checklist).
- **FAIRNESS READS IN (2026-08-06 ~07:4xZ,
  [results](posts/2026-08-06-draws-fairness-results.md)) — the
  unfair-penalty signature FIRED on every declared criterion:** E1
  gate passed (draw-0 drift 0.0145 < 0.05); dispersion-quartile
  deficit monotone 0.23→0.60→0.87→1.42 (Spearman +0.13, q4 = 6.2×
  q1); best-of-10 3.8597 is 2.01 BELOW AR's paired 5.8680;
  **energy score (read 4): flow 5.9308 vs AR 8.7696 — flow wins the
  proper score while losing single-draw MAE**. Honest residual:
  deficit positive even in the tight quartile (+0.23), win rate
  < 0.5 everywhere — partly artifact, not wholly; ES is now the
  candidate distributional column (owner decision to adopt).
  **σ_draw direct = 0.02367 SUPERSEDES the 0.0159 pin** (floors
  0.045/0.05 still bind → both live bands numerically unchanged;
  `sigma_draw_direct.py`, cross-estimator inside the χ²₉ band).
- **Golden-ticket noise search (lit slice 2026-08-06,
  [2603.15757](https://arxiv.org/html/2603.15757v1) "You've Got a
  Golden Ticket"):** a *single searched noise vector* (Monte Carlo
  over candidate tickets, weights frozen, inference-only) improved
  38/43 tasks across diffusion/flow policies incl. SmolVLA-LIBERO,
  with gains *growing at fewer solver steps*. Their search needs env
  rollouts; **our panel gives the offline criterion they lack** —
  score M candidate tickets by probe-subset MAE via
  `sample_actions(noise=...)` (the hook already exists), then
  validate the winner on the full panel. Caveats to carry: their
  LIBERO-Spatial cell *regressed* (−3%), tickets showed limited
  cross-task universality, and a fixed ticket makes the policy
  deterministic. Pairs with #12's 1-NFE distill (fewer-steps trend)
  and with mean-of-N (ticket vs mean-of-10 vs both). Cheap eval-side
  screen; needs its own pre-reg before any number is read.

## 2. Throughput: bucketed batching + torch.compile on the frozen prefix — `screening` (2a landed 2026-08-05; GPU A/B conditional)

- **Hypothesis:** length-bucketed batching + `torch.compile` of the
  prefix encode (79% of step time) buys ≥20% step-time on 1×H100 —
  compounding interest on every later run.
- **2a LANDED (2026-08-05, [post](posts/2026-08-05-bucketing-impl-sim.md)):**
  `--bucket-by-length` (default OFF) — `LengthBucketedBatchSampler`,
  camera-count keys, oracle-gated (3 CPU oracles bit-exact, gradflow
  green, 6 unit tests). **Sim finding: under the current recipe
  (`--camera-counts 1 2`) padding inflation is only +5.09% → ceiling
  ~3.6% step-time — below the <5% deprioritize line ⇒ NO GPU screen
  for current lineages.** Full-corpus census (3–4-cam datasets in):
  +32.55% → −23.8% padded tokens, ~19% ceiling. Conditional pre-reg
  in the post: first widened-selection run family runs the 1k-step
  A/B before adopting; paired arms must share the flag.
- **Cost remaining:** 2b (compile) — real implementation vs the
  blocker map below; decoupled from bucketing under narrow census
  (shape variance is text-jitter ⇒ pad-to-fixed-length).
- **Falsification (2b):** measured s/step and samples/s on identical
  configs, before/after, on THIS box. If <10% combined, bank the
  numbers and deprioritize.
- **Implementation notes (deep-dive 2026-08-05):** compile blockers
  on the prefix path: `pooled[valid_mask]` dynamic shape
  (vision.py:606), host syncs + `masked_scatter` (masks.py:132,
  model.py:196-204), KVCache `torch.cat` mutation, dense additive
  masks. No prefix attention takes the flash path today (sliding =
  always-masked, global head_dim 512 > fused cap). Bucketing is a
  compile prerequisite. Bonus levers: skip K/V writes for
  non-exported layers when `retain_cache=False`; cache frozen-run
  probe prefix encodes (bit-identical across evals).

## 3. Longer training on the best recipe — `queued`

- **Hypothesis:** rcond-100k was still improving at 100k
  (75k→100k bought 0.05–0.3); an extension banks a cheap win.
- **Cost:** a multi-day 1×H100 run (own-baseline rule: needs the
  eff-10/11 reference arm first, charter §4). Resume traps: fresh
  `--seed`, `--steps` = new TOTAL, cosine re-heat semantics.
- **Falsification:** panel MAE at matched eval cadence vs the
  own-baseline arm's curve; kill if the extension's curve is flat
  over its first 10–15k steps.

## 4. Stage-2 follow-ups (flow expert on AR trunk) — `queued`

Inherited questions from mainline §8.11 (banked: 6.57 in-run / 6.62
panel @80k, 2.2× smaller expert): more/deeper export streams (AR
adaptation lives in all 35 layers; the expert reads {4,9,14} —
untested headroom), expert width h512/h1536 on the better features, a
second-generation AR trunk re-measured through the stage-2 lens.
Cost: one screen-rung run per arm. Falsification: paired screens at
matched steps.

- **Deep read 2026-08-07 00:2x–00:5xZ
  ([post](posts/2026-08-07-pi05-deep-read.md)): π0.5
  (arXiv:2504.16054) + Knowledge Insulation (arXiv:2505.23705) are
  the production version of this recipe, and the two dials where
  theirs differs from ours are now named external arms:**
  1. **All-layer reads** — their expert attends per-layer to the
     full backbone KV stack; ours cross-attends to 3 exported
     streams ({4,9,14} of 35). Production-scale evidence for the
     deep end of the already-flagged export-streams headroom.
  2. **Trunk kept adapting under stop-grad** — KI's backbone
     continues CE-on-FAST *during* expert training, with
     stop-gradient on the expert→backbone attention seam
     (expert queries attend to `sg(K_b)`, `sg(V_b)`); naive joint
     training collapses language following (~75%→~5-10%) and is
     7.5× slower to converge; with stop-grad the CE/flow loss
     balance needs no tuning (α=1 vs π0.5's tuned α=10). Our
     sequential freeze is "extreme KI" — KI's frozen-backbone-0%
     result does NOT indict it (their backbone was action-naive,
     ours is action-pretrained), but a joint arm (trunk CE
     continuing + stop-grad seam) vs the frozen baseline is a
     screen-rung question with a banked anchor.
  Natural venue: the Molmo2 trunk endpoint (~08-08), where stage-2
  attachment becomes a live decision. π0.5 post-training also keeps
  the discrete head alive beside the flow head — our decoder kinds
  share the seam, so a both-heads arm is config, not surgery.

## 5. FAST tokenizer v3 — `queued`

- **Hypothesis:** refitting on curated-v0's exact quantiles removes
  the ~1.94%-of-chunks clip rate; small but real MAE effect on
  clipped chunks.
- **Cost:** CPU-only fit (~32 min measured for v2); token metrics
  RESET (never cross tokenizer versions) — coordinate with run seams.
- **Falsification:** paired arms (same seed/data/arch, only the
  artifact differs — the v1-vs-v2 precedent); recon error + clip rate
  in the fit report before any training touches it.
- **Lit radar (2026-08-06 20:5xZ):** FASTer (arXiv:2512.04952)
  replaces DCT+BPE with a learned VQ tokenizer ("FASTerVQ" — action
  chunks encoded as single-channel images, global spatio-temporal
  dependencies) + block-wise AR decoding; claims better token
  utilization/reconstruction and SOTA-beating speed+success vs
  FAST-style AR VLAs. If v3's quantile refit leaves clip/recon
  headroom on curated-v0, a learned-VQ arm is the natural rung after
  it (same paired-arm falsification; token metrics reset applies
  either way).
- **Deep read 2026-08-07 ([post](posts/2026-08-07-pi05-deep-read.md)):
  KI (arXiv:2505.23705) measured FAST vs naive per-dim binning as
  the backbone's discrete training signal: ~95% vs ~85% table-
  bussing success — external support for the token-quality premise
  behind the v3 refit.

## 6. Aux attribution arms — `confirmed` (aux HELPS actions; results 2026-08-06)

**ANSWERED 2026-08-06 04:2xZ
([results post](posts/2026-08-06-box-batch-results.md)): the
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
[pre-reg](posts/2026-08-05-prereg-paired-auxoff-40k.md). Primary read:
paired per-frame panel chunk_mae A@40k vs B@40k. **Executing on the
4×H100 box since 17:12Z** (parallel arms + 2 control seed replicates
for the noise floor, with a pre-registered decision rule:
[box batch pre-reg](posts/2026-08-05-prereg-box-batch-4xh100.md)).
2026-08-06 01:3xZ: all four arms trained (A-s0 formal probe 7.0882@40k,
B 7.702@40k; s1/s2 at their boundary), panel evals chaining; **results
instrument `fontaine/scripts/box_batch_results.py` landed + oracled
before the data** — frozen decision rule, mechanical headline-column
matching vs report JSONs, σ_seed → the E4B adopt band and rig slot 2;
anchors/degenerate/synthetic-inflation oracles all passed. When the
four npz+report pairs land: one command produces the results-post
numbers and both finalization amendments.

- **External replication + a new rung-(a) probe (deep read
  2026-08-07, [post](posts/2026-08-07-pi05-deep-read.md)):** our
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

## 7. Stream-schedule re-test — `queued`

0-0-16 vs 4-4-8 vs shallow-heavy (8-4-4) at scale: the acuity probe
(shallow stream carries sharpest position) and streams0016's rig hint
pull opposite directions — measure. Config-diff cheap per arm; enters
at the short-run screen rung.

## 8. Shortlist/output-vocab head for ar_backbone — `queued`

The 262k-vocab CE softmax is the VRAM headroom eater; a shortlist
head raises feasible batch on 1×H100 (mainline queued it as the
structural fix after the B12 OOM). Cost: real code + an equivalence
check (loss oracle moves → loud re-baseline). Payoff multiplies every
future ar_backbone run on this box. **Design concretized (deep-dive
2026-08-05):** chunked/fused linear-CE (logsumexp vs `lm_head.weight`
+ the 1026-row patch; elementwise softcap fuses) — never materialize
the `[B·S, 262k]` fp32 logits (~1 GiB at B10,
`ar_backbone.py:743-748`). Decode-side: action-phase argmax over
block columns only is exact (grammar mask + monotone softcap).

- **Owner measurement (2026-08-05 21:52Z, in-channel + html
  attachment): FAST round-trip error is barely measurable** —
  quantization is not the binding limit of the AR approach, and AR
  "definitely trains faster". Consistent with the paired analysis
  (AR wins the late horizon — a codec-bound model wouldn't): the
  limit is upstream of the codec, in trunk/grounding. Strengthens
  the AR-side weighting of the attribution front (owner steer
  21:48Z).

## 9. Data levers — `screening` (state-DROPOUT arm PRE-REGISTERED 2026-08-06 ~08:1xZ)

`--trim-leading-idle` (~6.7% of frames), state-noise augmentation,
judge-score-weighted sampling (never yet run). Each is a cheap paired
arm at the screen rung. Any derived corpus ships with the leakage
check (charter §2) before training touches it.

- **Lit slice 2026-08-06 ~02:5xZ — state-noise sharpened to state
  DROPOUT:** the shortcut-learning literature's standard lever is
  random state *masking*, not noise —
  [Adapt Your Body](https://arxiv.org/abs/2506.23944) masks
  proprioception to zeros with p=0.8 and reports it effective
  against proprioception-shortcut overfitting;
  [2509.18644](https://arxiv.org/abs/2509.18644) goes further
  (state-FREE policy, relative EE actions, vision-only) and reports
  better spatial generalization. If the #11 reliance probe shows
  heavy state reliance, the paired arm here is `--state-dropout p`
  (train-time masking, eval unchanged) — config-only surface, screen
  rung. **PROMOTED 2026-08-06 06:1xZ: the #11 probe came back
  SUPPORTED (D = +0.702 [0.498, 0.916],
  [results](posts/2026-08-06-state-probe-results.md)) — the branch
  rule fires and state-dropout is owed its own pre-reg** (design
  notes: p per 2506.23944's 0.8 vs a lower screen value is the one
  free parameter; the probe's masked-eval instrument doubles as the
  reliance readout for the trained arm; GAP-style phase-guided
  gradient scaling stays the follow-on if dropout helps but
  plateaus). **PRE-REGISTERED 2026-08-06 ~08:1xZ
  ([pre-reg](posts/2026-08-06-prereg-state-dropout-40k.md)): arm C =
  A-s0 recipe + `--state-dropout 0.8`, seed 0, 40k, box GPU 0 —
  paired vs A-s0's banked npz, band 0.15; chained masked-subset
  reliance eval; `--state-dropout` landed with the pre-reg (shared
  `mask_state_item` primitive with the eval probe, p=0 bitwise-inert,
  oracles green).**
- **Results instrument banked BEFORE the data (2026-08-06 ~09:0xZ,
  box-batch pattern): `fontaine/scripts/statedrop_results.py`** — all
  three frozen reads + the E3 probe gate + the verdict assembly
  (adopt-default / hardening-lever / mechanism-inert-kill / p=0.3
  branch / falsified) encoded and oracled against the banked A-s0
  panel npz: anchors 7.7966/3.9422 + state-copy 11.7848/2.6202 +
  subset state-copy first 2.4316 all reproduce through its pooling;
  degenerate zeros; synthetic COSTS/HELPS/inert/strong known-effect
  cases; misaligned-index abort. 4 CPU tests under `check.py` (244
  green). Arm C's ~12:3x–12:4xZ boundary read is now
  zero-improvisation: defaults point at the chained eval's output
  names; pass `--probe-final` from the train log's last in-run probe.
  before citing numbers):** the masking lever keeps accumulating
  neighbors: [ThinkProprio, 2602.06575](https://arxiv.org/abs/2602.06575)
  goes the OPPOSITE direction (proprioception as text tokens fused at
  the prompt input rather than late conditioning — relevant to our
  soft-state-token placement question, #11 discussion);
  [Cloak, 2606.22836](https://arxiv.org/pdf/2606.22836) masks the
  END-EFFECTOR VISUALLY for zero-shot cross-embodiment — a different
  masking axis (vision-side, not state-side) that would matter for
  the rig-transfer north star if arm C's mechanism reads clean. (Skim-depth, same pass:
  [2602.09722](https://arxiv.org/abs/2602.09722) "Rethinking VLA
  scaling" — pooling heterogeneous robot data induces negative
  transfer; selective mixture + regularization beat full pooling.
  Directionally supports judge-score-weighted sampling and the
  census's fork findings; re-read before citing numbers. The
  [data-engine survey](https://arxiv.org/abs/2604.23001) frames
  dedup/contamination checks as THE underexamined bottleneck — our
  #18.7 census is exactly this; no new action.)

## 10. E2B base-vs-IT swap — `queued`

Pre-registered mainline prediction ±0.2 MAE; backbone-swap arm, tests
whether instruct tuning matters at our instruction distribution.
Verify the -pt checkpoint ships the vision tower first.

## 11. Visual grounding arms — `queued`, the open front

Re-anchor probe: error is frame-dependent level mis-estimation;
acuity probe: the text stack's use of visual tokens is the
bottleneck. Arms: trunk shaping, schedules, vision-side aux tasks —
chartered on the community panel; `first_mae` is the
grounding-sensitive column (2.143 vs copy 2.620 — headroom).
High-variance; counts toward the ≥20% exploration budget.

- **Lit slice 2026-08-06 ~02:5xZ — mechanism story named: state-
  dominant bias.** [ReViP](https://arxiv.org/abs/2601.16667)
  diagnoses "false completion" in VLAs as modality imbalance —
  policies over-rely on internal state progression and under-use
  visual evidence (their fix: a progress-aware observer that FiLM-
  modulates the vision/proprioception coupling; +26% over π0 on
  their perturbation suite; abstract-depth read). The causal-
  confusion line ([2506.23944](https://arxiv.org/abs/2506.23944),
  [2509.18644](https://arxiv.org/abs/2509.18644)) says the same:
  proprioception is the shortcut, vision is what generalizes. This
  is a candidate mechanism for BOTH our standing grounding gap
  (first_mae barely ahead of state-copy) AND B's pending aux-off
  flag (first_mae 3.5009 WORSE than copy 2.6202 @40k — consistent
  with aux-off models leaning harder on the state shortcut; paired
  per-frame reads pending ~04Z decide nothing until then).
- **State-reliance probe rung (a) — PRE-REGISTERED 2026-08-06
  ~03:1xZ, instrument landed**
  ([pre-reg](posts/2026-08-06-prereg-state-reliance-probe.md)):
  `bijou.eval --mask-state` substitutes the dataset state mean (soft
  state token collates to exactly zero; `_state-masked` name suffix;
  report/npz record it; parse guards; `tests/test_mask_state.py`).
  Frozen subset `plans/holdout_curated_v0_k4l2_stateprobe_q4.json`
  (every 4th core row, 4,301 frames — intact side pools from banked
  npzs, zero intact evals). Primary read D = Δ_first(B) −
  Δ_first(A-s0), supported iff CI excludes 0 and D ≥ 0.05. 4 masked
  runs ≈ 1.7 GPU-h, blocked on A-s0's ~04Z npz; first quiet GPU
  window. Supported ⇒ #9 state-dropout gets its own pre-reg;
  ReViP-style modulation stays the heavier arm behind it.
- **Rung (a) RESULT 2026-08-06 06:1xZ — SUPPORTED**
  ([results](posts/2026-08-06-state-probe-results.md), instrument
  `fontaine/scripts/state_probe_results.py`, report
  `reports/analysis__state_probe_q4.json`): **D = Δ_first(B) −
  Δ_first(A-s0) = +0.702, CI95 [0.498, 0.916]** — 14× the 0.05
  threshold; chunk secondary agrees (+0.389 [0.106, 0.674]). B's
  better intact first_mae is bought with heavier state reliance.
  All three banked expectations came true (Δ_chunk +15.3–16.4 on
  every arm; no masked arm beats intact state-copy first; D > 0).
  Absolute Δs stay descriptive (OOD masking — masked levels ~2×
  worse than state-copy). Branch rule fired: #9 state-DROPOUT
  promoted to its own pre-reg. The grounding gap keeps re-anchor +
  acuity live for the residual intact-state gap.
- **ARCHITECTURE BATCH #1 PRE-REGISTERED 2026-08-06 ~12:2xZ (owner
  steering 11:44Z: multi-GPU run on fundamental architecture
  changes)** ([pre-reg](posts/2026-08-06-prereg-arch-batch-1.md)):
  paired arms on the stage-2 family, DDP3 on box GPUs 1–3, panel-v2 +
  stable keying, 40k eff-96 — **arm A `--max-soft-tokens 280`**
  (2× visual tokens/camera, the acuity lever; Amendment 2, owner
  12:59Z: 480p sources make 560's marginal tokens the most
  interpolated — 560 demoted to a follow-on rung contingent on a
  positive 280 read) and **arm B full-residual conditioning**
  (res0..res14 hidden-state streams with learned K/V projections
  replace kv4/9/14; ~23.6M params; impl + 5 oracles landed 12:2xZ)
  vs **control := teacher@40k** (Amendment 1; arm 0 dropped).
  Adopt-lever iff paired Δchunk ≤ −0.15 CI-excl-0; grounding read
  Δfirst ≤ −0.10. Both-null branch promotes the Molmo2-4B trunk
  swap. **Results instrument `arch_batch_results.py` banked before
  any data 13:4xZ (5th oracle-before-data application): 5 oracles
  green incl. v2 anchors + K1 gate vs the teacher's banked probe
  curve.** Explore class.
- **Lit slice 2026-08-06 ~13:3xZ — independent support for the
  early-layers story (arm B context, banked before its data):**
  [SmolVLA](https://learnopencv.com/smolvla-lerobot-vision-language-action-model/)
  conditions its action expert on features from ~L/2 of the VLM
  (not the last layer), and
  [FLOWER](https://arxiv.org/html/2509.04996v1) prunes up to 50% of
  the deep LLM layers outright and reallocates the capacity to the
  diffusion head — both consistent with our acuity probe (position
  info sharpest at the vision-tower output, degrading through the
  LM stack). Read for arm B: if full-residual res0..res14 nulls,
  the cheap follow-on is an EARLY-ONLY schedule (res0..res7, or
  vision-tower output as a direct stream) rather than more layers —
  the literature's winning configs concentrate conditioning at or
  below mid-stack. Also
  [SCALE](https://arxiv.org/pdf/2602.04208) (self-uncertainty
  conditioned adaptive looking) as arm-A-adjacent: token budget
  spent adaptively rather than uniformly; parked unless arm A
  reads positive. Trunk-swap caveat from the
  [ICLR 2026 VLA survey](https://mbreuss.github.io/blog_post_iclr_26_vla.html):
  VLM4VLA finds downstream VLA performance has NO correlation with
  the VLM's standard-benchmark scores — the Molmo2-4B port's case
  must rest on its vision-tower/grounding properties (pointing-
  pretrained, our acuity story), not on benchmark superiority;
  frame the port plan's success criteria accordingly.
  (Abstract-depth reads.)
- **Lit radar 2026-08-06 ~03:2xZ — the mechanism gets a training-
  dynamics CAUSE: [GAP](https://arxiv.org/abs/2602.12032) (ICLR
  2026)** shows proprioception dominates because it offers *faster
  loss reduction early in training*, suppressing visual learning
  specifically during motion-transition phases (target
  localization); their fix adaptively shrinks proprioceptive
  gradients during those phases (phase detection via proprio state
  estimation; sim+real, single+dual arm, works on VLAs).
  Consequences for us: (1) if the state-reliance probe supports the
  mechanism, the #9 train-time arm has TWO candidate levers —
  state DROPOUT (input-side, cheap, the current pick) vs GAP-style
  phase-guided gradient scaling (optimizer-side, no input
  corruption); dropout stays first (simpler, matches
  [2506.23944]'s p=0.8 masking evidence), GAP banked as the
  follow-on if dropout helps but plateaus. (2) GAP predicts the
  grounding gap concentrates in motion-transition frames —
  testable for free in the probe's npz by conditioning Δ_first on
  progress-within-episode; noted for the probe's discussion
  section, not its frozen reads. (Abstract-depth read.)

## 12. Solver/Heun-gap work — `screening` (distillation leg PRE-REGISTERED 2026-08-06)

The h1536 adaRMS Heun-gap collapse did NOT transfer to h1024-on-AR-trunk
(measured −0.28 at 10→30, first_mae −0.46): sampler quality is back on
the table for the best flow lineage. Arms: step-count sweeps, solver
variants, consistency/distillation toward 1–2-step deployment decodes
(the distillation leg pairs with idea 1).

- **Scoring note (2026-08-05 paired analysis):** flow's deficit is
  ~all late-horizon (crossover step 2, monotone to +1.2 @40 —
  [post](posts/2026-08-05-flow-vs-ar-paired.md)). Score solver arms
  per-step: a solver fixing only late-horizon costs nothing at
  first_mae; pooled-only scoring would misread it.
- **Literature (2026-08-05 slice): SnapFlow (arXiv:2604.05656) is
  the distillation-leg recipe to try first.** Plug-and-play
  SELF-distillation for flow-matching VLAs — no external teacher:
  mixes standard flow-matching samples with consistency samples
  whose targets are two-step Euler shortcut velocities from the
  model's own marginal predictions; zero-init target-time embedding
  switches velocity-estimation vs one-step modes in one network.
  Claimed: ~12 h on ONE GPU, no arch changes; π0.5-3B 1-NFE matches
  the 10-step teacher (98.75% vs 97.75% LIBERO, 274→83 ms); **tested
  on SmolVLA-500M too** (−8.3% MSE, 3.56× e2e) — the closest
  external analogue to our trunk+flow-expert protocol. Cheapest
  falsification here: distill flow-80k, score the panel at 1-NFE vs
  Heun-30 (band: within the σ_draw noise floor of 6.6232). Also
  pairs with #1 (a distilled 1-step model makes mean-of-N nearly
  free).
- **Literature (2026-08-06 slice, ~10:0xZ): the one-step fallback/
  follow-on menu, banked while the SnapFlow run climbs to its 10k
  probe.** If the @30k endpoint misses its band (or to extend a hit),
  three distinct objective families now have external evidence:
  (1) **One-Step Flow Policy** ([2603 era, self-distillation
  w/o pretrained teacher](https://zhaoyang97.github.io/daily-arxiv/2026-03-12/one-step-flow-policy/)) —
  self-consistency loss + self-guided regularization + warm start,
  71.6% avg on 56 sim manipulation tasks at 1-NFE; nearest
  competitor recipe to SnapFlow's. (2) **MeanFlow-based one-step
  VLA** ([arXiv:2603.01469](https://arxiv.org/abs/2603.01469)) —
  average-velocity (MeanFlow) objective, claims to *eliminate the
  consistency constraint* entirely (the constraint whose s=t
  divergence we are currently watching drift); 8.7× vs SmolVLA.
  A MeanFlow arm would be a genuinely different objective, not a
  SnapFlow re-tune — the right shape for a paired follow-up if
  consistency-style distillation is what misses. (3) **"Let It Be
  Simple"** ([arXiv:2606.05737](https://arxiv.org/abs/2606.05737)) —
  claims VLA is image-to-text-like (strong conditioning), so
  HIGH-NOISE TRAINING alone yields one-step decoding (95.6%
  LIBERO-Long, no distillation stage at all); their "irreducible
  velocity loss" framing + the ablation note (weakening the
  condition erases the one-step gain) ties directly to our
  conditioning stack (#11/Q3). Cheapest local probe of (3): score
  the TEACHER at 1-NFE (zero training — we may already have this
  number from the @10k/endpoint probe protocol runs) and read how
  much of the gap distillation actually closed vs what high-noise
  fine-tuning would have to. No new launch implied; feeds the
  SnapFlow results post's discussion + the next pre-reg if the
  endpoint branch fires.
- **PRE-REGISTERED (2026-08-06 ~00:3xZ,
  [pre-reg](posts/2026-08-06-prereg-snapflow-distill.md)):** SnapFlow
  self-distill of flow-80k — full recipe deep-read and frozen
  (α=0.5/λ=0.1 mix, sg two-step-Euler shortcut targets, zero-init
  φ_s target-time embedding, 30k steps LR 2.5e-5 cosine, trunk
  frozen, ~12–20 h 1×H100). Primary: full panel at 1-NFE vs 6.6232
  (+max(3σ_draw, 0.15) band, σ_draw by finalization amendment from
  draws runs 3–5); deployment headline: mean-of-10@1-NFE vs the AR
  anchor 5.8026 at ~one-Heun-5-draw cost. Fills the local-GPU queue
  slot after the draws chain + fairness probe; pre-launch impl
  checklist (φ_s, `--distill snapflow`, 1-NFE eval switch, oracles)
  = CPU work items.
- **IMPLEMENTATION COMPLETE (2026-08-06 ~00:3x–01:0xZ session): all
  five pre-launch checklist items landed; the launch path is
  zero-CPU.** φ_s config-flagged + checkpoint-compat (sanctioned
  additive warm start in the --init-from guard/loader);
  `--distill snapflow` (α/λ frozen in code, mean- and sum-form so
  chunked backward stays available); `bijou.eval --target-time
  {t,zero}` loud 1-NFE switch recorded through report/npz/banner; 10
  new oracles (validation gate (a) also RUN on the real checkpoint:
  6/6 forwards bit-exact, PASSED); launcher staged + recipe
  diff-verified through the real parse_args (50 teacher fields
  verbatim, 11 pre-registered deltas), chains gates (a)+(b) then
  training then the endpoint 1-NFE panels (1/5/10 draws). Gate (b)
  drift eval + @10k probe script wait on GPU only. check.py 201.
- **σ_draw FINALIZED (2026-08-06 ~05:5xZ,
  [amendment](posts/2026-08-06-sigma-draw-finalization.md)):**
  σ_draw = 0.0159 from the chain's pooled mean-of-N curve
  (gaussian-bias family, held-out N=5 error 0.087%; CPU-only,
  oracled) — 3σ = 0.048 < 0.15, **the floor binds: endpoint
  adopt-signal iff 1-NFE chunk_mae ≤ 6.7732.** Verdict
  family-independent (even the a-priori-max pure-noise reading gives
  0.040 < 0.045). Fairness-probe direct measurement supersedes if
  larger. The launch's last CPU-side blocker is closed — SnapFlow
  waits only on a quiet local GPU. **Direct measurement IN
  (2026-08-06 ~07:4xZ): σ_draw = 0.02367 supersedes the pin; 3σ =
  0.071 < 0.15 → the floor still binds, adopt band ≤ 6.7732
  UNCHANGED.** SnapFlow queues behind the #18.2 flip eval
  (~09:2xZ boundary) on the local GPU.
- **RESULTS INSTRUMENT BANKED BEFORE THE DATA (2026-08-06 ~09:3xZ,
  `4d48120`):** `fontaine/scripts/snapflow_results.py` — every frozen
  read (probe kill 9.6755, adopt ≤ 6.7732, falsify > 7.1232, edge
  ≤ 1.9831, deploy ≤ 5.8026, per-step horizon read + v2 column),
  oracles (a)–(e) green on banked data only, strict semantics guards
  (1-NFE/euler/target_time/draws/index-keying/full-panel all
  asserted). **Gap caught by banking early: the chained stage-4
  endpoint evals dump no npz** → per-step read had no data source;
  addendum `eval_snapdistill_endpoint_1nfe_npz.sh` staged (noise-key
  pinned `index` explicitly). Standing hold: the `--noise-key`
  default stays `index` until the chain's endpoint evals run at 30k.
  **Hold RELEASED — default flipped to `stable` 2026-08-06 ~15:5xZ
  (#18.2 follow-on, see item 2 in the deep-dive list).**
- **Pointer reads closed (lit slice 2026-08-06):** OFP
  ([2603.12480](https://arxiv.org/abs/2603.12480)) — *from-scratch*
  one-step self-distillation (self-consistency + self-guided
  regularization + warm-start from temporal action correlations);
  π0.5 one-step beats the 10-step teacher on RoboTwin 2.0 — the
  reserve recipe if SnapFlow misses expectation 2, and its
  warm-start trick touches #1's noise structure. GoldenStart
  (2603.14245) — Q-guided VAE priors + entropy control for *online
  RL* distillation — screened out (needs Q-functions/rollouts; not
  our offline setting). Golden Ticket noise search banked in #1.

## 13. Sign-convention detection & repair (owner hypothesis) — `screening`

- **Hypothesis:** a small set of community repos encodes joint angles
  with flipped sign conventions (esp. wrist_roll on mirrored wrist-cam
  mounts); training on them injects contradictory supervision.
- **Status:** stage 1 (CPU screen over the panel npz) done 2026-08-05
  — 9 candidate (repo, dim) cells, three pathologies separated by
  per-frame classification; cleanest mirror lead
  kantine/domotic_dishTidyUp_anomaly wrist_flex (median frame corr
  −0.75). Instrument: `probes/probe_sign_convention_stage1.py`;
  [results post](posts/2026-08-05-sign-convention-stage1.md).
- **Stage 2 PRE-REGISTERED (2026-08-05 ~23:3xZ,
  [pre-reg](posts/2026-08-05-prereg-sign-stage2.md)):** optical-flow
  cross-check on the three mirror-signature cells (dishTidyUp_anomaly
  wrist_flex, groceriesSorting_expert wrist_roll, aractingi
  shoulder_lift), CPU-only (~20–40 min spare cores). Frozen: Farneback
  params, isolated-motion pair selection (|v_d| ≥ 0.5°/frame, 2×
  dominance), ego-cam identification rule (cams are unlabeled),
  15-repo so100 reference population with an 80% sign-consistency
  validity gate, MIRRORED/NORMAL/INCONCLUSIVE bootstrap rules,
  synthetic-flip hard validation gate before candidate cells open,
  Dongkkka + kevin510 as specificity controls, and the stream-
  consistency read (calibration-mirror vs action-only flip).
  Feasibility verified pre-post: all repos local, torchcodec decodes
  the AV1 videos, state+action parquet intact. Execution = a later
  work session; if ≥1 MIRRORED, the repair arm (flip-corrected
  derived corpus through #18.8 certs + paired screen) gets its own
  pre-reg.
- **Stage 2 EXECUTED 2026-08-05 ~23:5xZ — the escalation branch fired**
  ([results post](posts/2026-08-05-sign-stage2-results.md), probe
  `probes/probe_sign_convention_stage2.py`): 3 of 4 reference
  populations FAILED the 80% sign-consistency gate (wrist_roll 9/15,
  wrist_flex 10/15, shoulder_lift 9/15; only shoulder_pan valid at
  13/15) ⇒ hard gate failed, **candidate cells never opened, no
  verdicts**. The t_x oracle PASSED end-to-end (mass 1.000 both
  directions) — the mechanism works where the population premise
  holds. Diagnosis: image-plane statistic signs follow *camera
  mounting* (cams sign-disagree in 11/15 shoulder_lift refs; ego-cam
  rule NO-MARGIN on ~half; ω underpowered off-wrist-cam) — not
  evidence that joint conventions vary corpus-wide. The three stage-1
  mirror cells remain unresolved leads; repair arm neither eligible
  nor dead. **Next (owner steer wanted): stage-2b amendment
  conditioning reference populations on `meta/camera_kinds.json`**
  (the 2026-08-02 VLM cam-labeling pass: wrist/front/side/top) — t_y
  from front cams, ω from wrist cams, label-gated ego rule; reuses
  the 38-repo flow cache, so it is cheap.

## 14. ±180° wraparound census & shortest-arc error — `confirmed`/banked (measured 2026-08-05)

- **Hypothesis:** truth chunks wrapping the ±180° boundary inject
  ~360°-scale discontinuities into BOTH raw-degree training targets
  and MAE; a wrap census may explain a measurable slice of panel MAE.
- **Measured** ([write-up](posts/2026-08-05-wrap-census.md),
  instrument `probes/probe_wrap_census.py`, anchors in-probe): panel —
  16/17,204 wrap frames (0.093%, under the 0.1% gate) carrying 0.0720
  of the 5.8026 pooled chunk_mae (1.24%; shortest-arc re-score
  5.7498). Corpus — 81/42,872 episodes (0.19%) across 23 repos;
  kevin510 systemically corrupted (40/40 eps), willnorris/bbox-2 a
  separate state-stream glitch. wrist_roll dominates (204 action
  jumps), matching the SO101 calibration story (lerobot#1255, PR#777,
  fixed in 0.6.0).
- **Consequences:** unwrap-at-load training arm killed (0.19% cannot
  move a 40k pair); shortest-arc metric proposal → owner sign-off
  (moves every anchor); kevin510 + willnorris/bbox-2 flagged for any
  future curated-v1 exclusion list.

## 16. Few-shot rig-transfer benchmark — `parked` for execution (owner 2026-08-05 21:43Z), instruments banked; **the north star** (owner 2026-08-05 17:20–17:23Z)

- **OWNER STEER 2026-08-05 21:43Z — execution PARKED, priorities
  reweighted:** the rig datasets are small/noisy and a 12-ep fixed
  holdout is high-variance ("really depends on which episodes you
  choose"); owner will collect a better rig dataset later. **Short
  term: improve MAE on the comm holdout and/or attribute where the
  limit comes from** (bigger trunk? bigger image embeddings?
  video-trained trunk? is the flow expert needed at all vs pure AR?).
  Empirical anchor from the owner: lower comm-holdout MAE has always
  translated to good rig fine-tunes; current failure mode is gripper
  *placement* accuracy (grounding), motion is fine; aux tasks
  generalize strikingly (4k ft on AR-100k produced sensible subgoals
  for a fully-OOD instruction — "coiled USB-C cable", "glass pot").
  The instruments below stay banked: they are corpus-agnostic and
  re-run on the future dataset in minutes.

- **External anchor for the premise (deep read 2026-08-07,
  [post](posts/2026-08-07-pi05-deep-read.md)):** π0.5's Fig. 8 is
  the diversity-buys-transfer bet measured at production scale —
  held-out-home performance scales monotonically with training
  locations (3→104), and at 104 locations MATCHES a control trained
  on the test homes; 97.6% of their phase-1 examples are not the
  target embodiment. Evidence, not proof (their scale: ~400 h,
  ~100 homes) — but the north-star premise now has a citable
  production-scale precedent.

- **Pre-reg draft posted 2026-08-05 ~21:2xZ**
  ([post](posts/2026-08-05-prereg-rig-fewshot-benchmark.md)): design
  frozen — 12-ep holdout (SeedSequence(16)) + nested N ∈ {10,25,45}
  materialized derived corpora (leakage-checked, the #18.8 consumer);
  owner `run_ft_rig.sh` protocol constants; best-checkpoint-at-200
  selection; co-primary chunk_mae + first-4 pooled MAE; 3·σ_ft
  decision rule (σ_ft from N25 seed replicates); **eligibility gate:
  init pretrain corpus must certifiably exclude the rig repos —
  flow-80k is contaminated (rig data in its pretrain mix), rcond/box
  arms qualify.** Two slots (init selection rule + E5 noise scale)
  fill by finalization amendment after the box reads; execution at
  the first quiet GPU boundary after.
- **Instruments LANDED 2026-08-05 ~21:5xZ (Amendment 1 on the
  pre-reg):** plan frozen (`plans/rig_fewshot_v0_k4l2.json`, 12 eps /
  48 core + 24 labeled; holdout = native split 0.212/seed 16 → v2
  {1,2,3,6,11,15,20,24,25,30,41} + clean {2} — mechanism amendment:
  the draft's bespoke SeedSequence draw could not feed the leakage
  checker); subsets materialized + verified
  (`~/datasets/rig_fewshot_v0/`, n10 6,223 / n25 15,881 / n45 29,107
  frames, videos hardlinked bit-identical, judgments remapped, stats
  recomputed w/ oracle worst |Δ| 1.2e-4); **leakage certs PASSED ×3**
  (#18.8 provenance path, doctored-provenance negative control fails
  loud); loader smoke bit-exact incl. shifted mid-file video decode;
  **wrap census CLEAN on both rig repos** (hygiene gate 1). Remaining
  before launch: launcher gen + finalization amendment (slots 1–2)
  after the box reads.

- **Goal statement (owner):** "build a VLA for my rig… prove transfer
  so you can fine-tune a task on a new SO101 arm with tens of
  examples." Community-panel MAE is the proxy; **the
  sample-efficiency curve is the product metric.**
- **Design sketch (pre-reg to write after the box batch lands):**
  fine-tune the best lineage on N ∈ {10, 25, 50} episodes of a
  held-out rig task; measure panel-style MAE on that task's holdout
  (and eventually rollout success) vs N. Protocol precedent: the
  owner's ft-rig lineage (4–5k-step fine-tunes, `run_ft_rig*.sh` on
  the second box, both AR and flow variants).
- **Dependencies:** tonight's aux-off answer + seed-noise floor pick
  the trunk and set the minimum detectable effect for paired ft
  comparisons; sign/calibration hygiene (ideas #13, #14) bites
  hardest on a new arm — keep them warm.
- **Falsification:** the curve itself — if MAE at N=50 is no better
  than zero-shot, transfer is not proven and the pretraining recipe
  (not the ft protocol) is the suspect.
- Reweights the whole list: rig-transfer relevance now outranks
  community-panel micro-optimization at equal cost.
- **Metric note (2026-08-05 paired analysis):** the pre-reg must fix
  the deployment replan interval k and quote first-k pooled MAE next
  to chunk_mae — the flow-vs-AR ranking *flips* at k≤3 vs k≥5
  ([post](posts/2026-08-05-flow-vs-ar-paired.md)); chunk_mae alone
  is the most AR-favorable point on that axis.
- **Literature (2026-08-05 slice): the ft-protocol arm should
  include LoRA-r32 + full vision-encoder ft** (arXiv:2607.10172,
  π0 on UR5e precision assembly): LoRA saturates at r=32 with no
  significant FFT advantage; **freezing or LoRA-restricting the
  vision encoder significantly degrades** (independent external
  support for our grounding-bottleneck reads, idea #11); static
  peak VRAM 36.2→10.8 GiB — on 1×H100 that headroom converts
  directly to batch for the few-shot fine-tunes.

## 17. New trunks / new architectures — standing owner mandate (2026-08-05 17:24Z)

"The current repo should just be a starting point for what I've
tried." A ranked exploration front, fed by the literature slice;
every candidate enters at the screen rung with a pre-reg and counts
toward the exploration budget.

- **OWNER PICK (2026-08-05 21:57Z): E4B screen confirmed as the next
  pre-reg** — AR-100k on the freed 4×H100, **matched parameters with
  the E2B AR-100k run** (verified from the recipe: `--batch-size 12`
  /GPU on DDP4 = effective 48, decoder-lr 1e-4, backbone-text-lr
  2e-5, grad-clip 100, seed 0, same aux/condition/dropout flags; if
  E4B OOMs at 12/GPU, grad-accumulate to the same effective 48 —
  stated in the pre-reg, batch semantics never change
  mid-comparison). **Gates = the MAE curve over time, not the
  endpoint**: matched-cadence in-run probes vs the banked E2B curve
  + mid-run panel evals (~25k/50k) with pre-registered bands so a
  losing rung dies early. **Owner 21:58Z: image (embedding) budget
  is the follow-on ablation arm** on the winning trunk — one
  variable per rung, trunk first, then image token budget (pairs
  with #11's grounding read). **Pre-reg POSTED 2026-08-05 ~22:4xZ**
  ([post](posts/2026-08-05-prereg-e4b-screen.md)): verbatim mainline
  recipe + `--backbone google/gemma-4-e4b-it` (AR path verified
  fully config-driven), eff-48 with a pre-registered
  chunked-backward fallback ladder (bijou.train has no grad-accum
  today — impl + oracles is a pre-launch CPU item if the memory
  smoke says B12 doesn't fit; E2B B12 peaked 77.5 GiB, E4B text
  ~2.2× params), probe-curve gates @30k/50k (±0.5 floor) + mid-run
  panels @25k/50k on the local GPU, endpoint adopt rule bound to
  tonight's E5 σ_seed via finalization amendment. Launch blocked on:
  box free + e4b checkpoint download (not in box cache) + parity
  spot-check + memory smoke + amendment.
- **NO-LAUNCH (2026-08-06 ~05:4xZ)** — the pre-registered memory
  ladder exhausted: all four rungs (B12 direct / 2×6 / 3×4 / 4×3
  chunked backward) OOM'd on 80 GB before completing one optimizer
  step; Adam state (~31.8 GiB) never even allocated ⇒ steady-state
  needs ~≥110 GiB/rank under the matched recipe
  ([finding post](posts/2026-08-06-e4b-no-launch.md), Amendment 2 of
  the pre-reg). **Feasibility negative, not a scale answer** — the
  probe/panel gates never ran; the attribution question stays open.
  Follow-on = owner decision (options posted): ZeRO-1 re-entry as a
  NEW pre-reg vs redirect to Molmo2-4B (rank 2) / #11 grounding
  arms. E4B's zero-port-cost premise is dead; Molmo2-4B competes on
  closer-to-even terms now.
- **Owner steering 2026-08-06 11:44Z — multi-GPU architecture run
  requested** (new trunk / full residuals / bigger images, "really
  just examples") → **architecture batch #1 pre-registered ~12:2xZ**
  ([pre-reg](posts/2026-08-06-prereg-arch-batch-1.md), filed under
  #11 — the owner's examples are the grounding front): bigger-images
  + full-residual arms first (same trunk, clean attribution); the
  trunk swap (Molmo2-4B) is its own follow-on pre-reg, promoted to
  next-in-line if both arms null. E4B ZeRO-1 re-entry queued behind
  the architecture run (owner 11:44Z: E4B paused).
- **Arm B implementation LANDED 2026-08-06 ~12:4xZ** (the F1 critical
  path: the smoke needs BOTH configs, so arm A could not launch
  before this existed): `--conditioning-streams residual` in
  bijou.train — encoder exports raw post-layer hidden states
  (res0..res14), the flow expert projects them through learned
  per-layer adapters (RMSNorm + K/V proj + k_norm/v_norm + RoPE,
  mirroring `TextAttention.project_kv` exactly, so the streams are
  contract-identical to K/V exports and the blocks are untouched).
  Adapters live DECODER-side and attach OUTSIDE the no-grad prefix
  encode — trainable under the frozen trunk. Real-config count
  23.62M params ✓ (pre-reg said ≈23.6M). All five pre-launch oracles
  green as CPU tests (`tests/test_residual_streams.py`, 11 tests):
  stream contract + padding-orientation invariance, trunk
  bitwise-frozen through an optimizer step, grads reach all
  adapters, checkpoint round-trip with no flags, K/V path untouched
  (state-dict keys + banked loss oracles). `check.py` 285 green.
  Still owed at the arm-C-boundary code sync: SnapFlow stage-0
  re-verify on the box + F1 two-config smoke.
- **Molmo2-4B port plan POSTED 2026-08-06 ~14:1xZ** (owner-promoted
  12:03Z, background work independent of the batch verdict;
  [plan](posts/2026-08-06-molmo2-port-plan.md), distilled primary-
  source doc `docs/molmo2.md` per §6 post-cutoff rule). Key design
  calls: **residual-only conditioning** (arm B's path — learned
  adapters keep the expert contract at kv1×512 regardless of
  Qwen3's GQA 32:8; no KVCache/layer-type/`project_kv` port
  needed) and **15-of-36-layer mount** (fractional depth 0.417 vs
  E2B's 15/35 = 0.429 — expert depth and res0..res14 schedule
  carry over unchanged). Five WPs: WP0 seam refactor (the
  `docs/plan.md` encoder ABC, oracle-guarded, lands alone) → WP1
  Qwen3 decoder port + HF parity (shared with InternVL3.5/Qwen3-VL
  — one port, three trunks) → WP2 SigLIP tower/connector → WP3
  ChatML collator (turn-close probe + state-slot splice re-proved)
  → WP4–5 exports/schema/audit. Phase 1 = flow on the **raw frozen
  prefix** (no AR port, no vocab surgery; the AR-adaptation −2.7
  confound ships with any claim). Mounted footprint ~2.3B ≈
  4.7 GiB bf16; est. 4–6 CPU work sessions, GPU only for parity
  bursts + the memory smoke. First run gets its own pre-reg after
  the §4 oracle suite is green.
- **External prior (lit slice 2026-08-05 ~22:5xZ):**
  [2606.31382](https://arxiv.org/pdf/2606.31382) (VLM-to-VLA
  parameter redundancy) reports **bigger VLM backbones do NOT
  consistently improve action performance after adaptation** (their
  ablations; skim-depth read via a fast-model summary — re-read
  before citing numbers). Direction: strengthens the E4B screen's
  kill branch as a live outcome, not a formality — and raises the
  prior on #11 (grounding/adaptation quality, not trunk scale, as
  the binding limit). The screen runs regardless: our recipe, our
  corpus, pre-registered either way.

**Ranked 2026-08-05 by the [trunk survey](posts/2026-08-05-trunk-survey.md)**
(paper + fetched-config deep-reads, owner method): **1. Gemma 4 E4B**
(zero-cost in-family rung) → **2. Molmo2-4B** (best-in-tier quality,
video+spatio-temporal grounding, Apache) → **3. InternVL3.5-4B**
(same Qwen3-4B decoder as Molmo2 — one port serves both; only modern
4B with a true base ckpt ⇒ the idea #10 vehicle) → **4. V-JEPA 2.1
ViT-L augmentation arm** (the dynamics bet; 2-AC's <62h-robot-video
→ zero-shot Franka is the strongest external evidence for the
north-star thesis) → **5. Qwen3-VL-4B in reserve** (most
reimplementation surface, no base ckpt). Screened out: Ministral 3
3B (no video), SmolVLM2 (older gen; but SmolVLA = external
validation of our trunk+flow-expert protocol), Cosmos-Reason1-7B
(second round iff E4B says scale is the lever), all MoEs (budget +
export-stream semantics). Original slate below, kept for scope:

- **Trunk swaps at reachable scale:** E2B → stronger open VLM
  families (Qwen-VL, larger Gemma-4 variants — E4B/12B) through the
  existing stage-2 trunk-swap protocol; also the queued base-vs-IT
  swap (idea #10) as
  the cheapest member of this family.
- **Video/dynamics-pretrained encoders** (V-JEPA-style) vs
  image-language pretraining — the grounding probes (idea #11) say
  the visual stack is the bottleneck; dynamics-predictive
  pretraining is the structurally different bet.
- **Tokenizer-free continuous action heads end-to-end** — remove
  FAST; the flow expert reads the trunk at full depth rather than
  export streams {4,9,14} (subsumes the idea #4 stream question).
- **Small world-model / latent-dynamics trunk** trained on the
  community corpus, policy as readout — cross-embodiment by
  construction; the speculative end.
- **Consistency-distilled 1–2-step deployment decoders** (pairs with
  ideas #1 and #12) — the deployment-latency leg of the rig goal.

## 18. Instrument & infra hardening — `screening` (items 1+8 done 2026-08-05, 2 flag-landed, 3+4+7 done 2026-08-06)

The [bijou deep-dive](posts/2026-08-05-bijou-deep-dive.md)'s fix
queue, in leverage order (details + file:line in the post):

1. ~~Hardening pass~~ **DONE 2026-08-05 ~20:55Z**
   ([post](posts/2026-08-05-hardening-pass.md)): aux-prompt-hash →
   probe/eval selection (`bijou.eval --aux-prompt-hash` new flag);
   `resolve_plan` bounds assert; `score_frame` n_valid assert;
   report JSON records full scoring semantics
   (exclude/aux_prompt_hash/sample_steps/method/draws/generate/
   condition_override/batch/world); npz gains episode_index/
   frame_index identity columns. Oracle: banked AR-100k panel
   recomputed bit-exact (12/12 cells, d=0) through the edited
   scoring path; 3 new unit tests; check.py green. NOT included:
   deep-dive finding 6b — now item 8 below.
2. Flow-noise stable-triple seeding — **implemented behind
   `--noise-key` 2026-08-05 ~21:20Z, break pre-registered**
   ([amendment](posts/2026-08-05-noise-reseed-prereg.md)): `stable`
   keys noise to blake2b(repo_id, episode, frame) via numpy
   SeedSequence (128-bit, no torch 32-bit trap, no draw stride);
   default stays `index` (byte-identical, oracle 12/12 d=0) until the
   flip executes at the first anchor boundary after the box reads —
   one flow-80k panel re-bank, decision band pre-registered off the
   draws chain's empirical σ_draw. Until then flow anchors remain
   valid only at frozen corpus composition. **Band FINALIZED
   2026-08-06 ~05:5xZ
   ([amendment](posts/2026-08-06-sigma-draw-finalization.md)):
   σ_draw = 0.0159 < 0.045 → floor binds, re-bank band
   [6.4882, 6.7582]; the flip eval is eligible now (box reads
   posted) and queues behind the probe work on the local GPU.**
   **FLIP EVAL LAUNCHED 2026-08-06 ~07:41Z** (tmux `stablekeyrebank`,
   `~/eval_flow80k_stablekey_rebank.sh`) after the fairness probe's
   direct σ_draw = 0.02367 kept the floors (`reopen_floors: false`
   asserted in-launcher); band [6.4882, 6.7582] + bitwise
   state-copy/AR controls read at the ~09:2xZ boundary.
   **DONE — ADOPTED 2026-08-06 08:3xZ
   ([results](posts/2026-08-06-stablekey-rebank-results.md)): controls
   bitwise ✓, stable-key chunk 6.5997 INSIDE the band (Δ −0.0242 ≈
   1σ_draw), first 1.9355. `stable` is now the quoted keying for all
   new flow numbers; ledger anchor re-banked. The #18.2 chain is
   closed.** **DEFAULT FLIPPED 2026-08-06 ~15:5xZ:** the code default
   (`bijou.eval` CLI + `BijouPolicy`/`SmolVLAEvalPolicy` ctors) is now
   `stable` — the hold expired when the SnapFlow chain's index-keyed
   stage-4 endpoint evals + npz addendum completed (15:10Z). `index`
   retained permanently behind an explicit flag for historical
   reproduction; new default-pin regression test; check.py 295 green.
   Arm A/B launchers written at the box boundary inherit `stable`, as
   the arch-batch pre-reg requires.
3. ~~Q3 tripwire noise fix~~ **DONE 2026-08-06 ~02:4xZ** (deep-dive
   finding 3, closed before the SnapFlow distill launch — the next
   conditioned flow run): `FlowDecoder.predict_chunk` now returns the
   noise it integrated (`BijouPrediction.noise`; the fallback draw
   moved from `sample_actions` into `predict_chunk` — same randn
   call, proven bit-exact incl. generator consumption against a
   pre-edit banked reference), `validate()` captures it per rich row,
   and the Q3 override decode reuses each row's scalar-pass noise —
   |Δ| is now purely the conditioning effect (was floored at sampling
   variance for a conditioning-blind flow model, the exact state the
   alarm exists to catch). AR path byte-unchanged (noise None,
   greedy). Eval/panel paths untouched structurally: eval always
   supplies per-item noise explicitly. 3 new tests
   (`tests/test_condition_tripwire.py`); `check.py` 215 green.
   Semantics note: `condition_sensitivity` for flow runs is not
   comparable to mainline's historical values (which carried the
   variance floor).
4. ~~Resume hardening~~ **DONE 2026-08-06 ~01:1xZ** (deep-dive
   finding 2, all three traps): (a) fresh-seed-on-resume is now
   ENFORCED — `--resume` with the checkpoint's recorded
   `train_args.seed` dies loud at startup (before data/model build;
   the epoch-0 restart replays the same batches + τ/ε draws), with
   `--allow-same-seed-resume` as the explicit reproduction-only
   escape hatch and a warn-not-die path for pre-recording
   checkpoints; (b) live-backbone resume prints a WARNING that fp32
   masters restart snapped to the bf16 grid (the "lossless
   continuation" comment corrected — lossless only in the
   frozen-backbone regime); (c) the resume hyperparameter note now
   covers EVERY optimizer param group via CLI-intent capture at
   group construction (was group 0 only — a changed
   `--backbone-*-lr` on resume was silently ignored), reading
   `initial_lr` so schedule-decayed lr can't fake a mismatch. 11 new
   tests (`tests/test_resume_guards.py`), live oracle on the real
   flow-80k checkpoint: same-seed refused / fresh-seed proceeds;
   `snapflow_recipe_verify` extended (new field at inert default,
   stage 0 re-run green, 51 verbatim). **Unblocks idea #3, and lands
   before the E4B 100k launch opens its crash+resume risk window.**
5. ~~Rig-rollout safety gate~~ **DONE 2026-08-06 ~09:5xZ** (deep-dive
   findings 8+9, the first-physical-run blocker, closed while #16
   execution is parked so the gate exists before it is ever needed):
   new lerobot-free `bijou/rollout_safety.py` + wiring in
   `bijou.rollout`. (a) **Clamp mandatory** — `--max-relative-target`
   (positive, finite) required before the arm moves, `--unclamped` is
   the explicit opt-out, clamp+unclamped together die as
   contradictory; gate runs before the (slow) policy load and in
   `--check` mode too. (b) **First-obs envelope** — after connect, the
   first observation must lie inside per-joint bounds from the rig
   stats (q01..q99 widened by half-band, 15° absolute floor;
   mean±3σ fallback for quantile-less checkpoint tables; stats
   dim ≠ 6 joints dies as wrong-embodiment). Catches wrong
   `--stats-repo-id`, ticks-vs-degrees (~10³ ticks flags every
   joint), uncalibrated arms; `--skip-envelope-check` for deliberate
   unusual starts; per-joint table printed every run, envelope shown
   in `--check`. (c) **Camera kinds mirror training** — with
   `--stats-dataset`, kinds resolve through training's own path
   (`annotation_stamp` + `camera_kinds_of`: stamped+hash-matched
   file, else "unknown" — never the name heuristic, which stays only
   for the no-dataset case); `--camera-kind NAME=KIND` explicit
   override, validated against the vocabulary (deep-dive's wild case
   "front-named cam judged top" covered by test). 22 new CPU tests
   (`tests/test_rollout_safety.py`); `--check` exercised end-to-end
   on the real flow-80k checkpoint (CPU); `check.py` 274 green.
6. ~~Parity extension~~ **DONE 2026-08-06 ~03:4xZ** (deep-dive
   finding 7): `verify_parity` gains (a) a default-on **padded
   2-sample × 2-image batch check** — mixed-length prompts through
   the processor's `padding=True` path (natively LEFT-padded on
   transformers 5.14 — measured, not assumed), HF attention mask +
   per-sample logical position_ids (the `encode_tensors`
   convention, passed to HF too since its forward defaults to
   arange), gated per sample at its last REAL position against HF
   on the same padded batch AND against HF's unpadded per-sample
   forwards; **both padding orientations** run (the native side +
   the per-row roll to the other — the ar_backbone prompt path
   collates left, the token-identity of each row vs its solo
   tokenization is asserted). (b) **`--require-bitwise`** —
   escalates every same-shape HF comparison from tolerance to
   bitwise (the measured eager/H100 contract, previously printed
   but never enforced) and refuses near-tie token forks;
   cross-shape (padded-row vs unpadded) comparisons stay
   tolerance-only, labeled as such. Validated on the real E2B (CPU
   eager): full harness PASS — **ours-vs-HF-padded BITWISE on all
   real positions in both orientations**; solo cross-checks ≤0.44
   (GEMM-shape fp noise, tol 2.0). **Falsification oracle taught a
   scope lesson worth recording:** an arange-doctored run passes
   WITHIN TOL in any orientation, because in a single forward
   positions enter only through RoPE, which is *relative* — arange
   vs logical is a per-sample constant shift, visible only as
   fp-rotation noise (~0.6). So this check pins mask + padding +
   multi-image semantics vs HF; the position CHAIN (where the
   convention genuinely bites — cached continuation) stays pinned
   by `tests/test_backbone_continuation.py`. Oracle (corrected to
   a genuine corruption): real=PASS, zero-positions=FAIL,
   mask-dropped=FAIL. Remaining honest gap: state-token splice and
   15-layer truncation/`kv_stop_layer` have no HF counterpart
   (bijou self-consistency tests cover them).
7. ~~Duplicate-content census over curated_v0~~ **DONE 2026-08-06
   ~02:0xZ** ([results
   post](posts/2026-08-06-dup-census-results.md)): the corpus is
   heavily forked — 6,935 of 52,507 episodes (2.67M frames) in 3,348
   cross-repo BYTE-EXACT clusters (action+state streams identical;
   quantized tier adds nothing). **The split is breached: 524 holdout
   episodes across 79 repos have byte-exact twins in train — 2,096 of
   17,204 core panel rows (12.2%) score on leaked episodes**, all via
   the cross-repo fork channel the repo-id dedup can't see. Anchor
   impact (validated partition, anchors reproduce exactly): leaked
   frames score ~1.3–1.6 better than clean on BOTH banked models —
   **clean-core anchors: AR-100k 5.9761/2.1695, flow-80k
   6.8137/1.9714** (published 5.8026/2.1431, 6.6232/1.9331 are
   ~0.17–0.19 optimistic in level; content-difficulty confound stated
   honestly). Paired within-corpus deltas (box batch, E4B, draws
   chain) UNAFFECTED — every model shares the same train corpus and
   the same leaked frames. Instruments:
   `fontaine/scripts/dup_content_census.py` (+`--oracle` 7-case
   suite, split mirror proven on all 878 plan repos, collision guard),
   `dup_census_anchor_impact.py` (join content-checked vs raw
   parquet). Exclusion list frozen in `~/dup_census_report.json`.
   **Panel-v2 amendment PROPOSED 2026-08-06 ~02:3xZ, awaiting owner
   steer** ([amendment](posts/2026-08-06-panel-v2-amendment.md),
   instrument `fontaine/scripts/panel_v2.py`): v2 = v1 minus the 524
   leaked episodes minus the 3 wrap-census corrupt repos, strict
   row-subset (core 17,204→15,056, labeled 8,596→7,522) so every
   banked npz re-pools exactly with zero re-evals. v2 anchors
   derived + oracle-gated: **AR-100k 5.8894/2.1396, flow-80k
   6.7151/1.9453, state-copy 11.7639/2.5851** (frozen plan
   `plans/holdout_curated_v0_k4l2_panel_v2.json`, embeds exclusions).
   Transition proposal: in-flight pre-registered reads finish on v1;
   v2 for every new pre-reg on approval; bundle the #18.2 noise-key
   flip (+ optionally #14 shortest-arc) at the same re-bank boundary
   so the flow anchor re-banks once. Until steer, results posts quote
   full-panel (anchor convention) with the v2 column alongside.
8. ~~Leakage checker same-repo-id count/content assert~~ **DONE
   2026-08-05 ~21:20Z** (deep-dive finding 6b): the identity branch
   now VERIFIES the claim — episode-count assert plus per-episode
   length fingerprint (`meta/episodes.jsonl` v2 or `meta/episodes/`
   parquet v3; asymmetric metadata is fatal; same-directory shortcut
   for the literal identity case). Mismatch ⇒ SystemExit demanding
   `meta/source_provenance.json`, symmetric with the provenance
   branch's count assert. 4 new tests (179 green); full-corpus
   identity certification re-run PASSED with the new code
   (radioactive 5267 / checked 47240, 4.1 s); a mutated-count copy of
   `therarelab/so100_pick_place_2` fails loud in production. Unblocks
   derived-corpus training (ideas #9, #13 repair arm).

## 20. Activation checkpointing for live-trunk training — `queued` (measured need 2026-08-06)

The Molmo2 AR smoke measured the wall: fp32 masters + DDP grad
buckets + Adam on a 3.7B trainable set ≈ 63 GiB static on an 80 GiB
card, and at ~2.4 GiB/sample of saved activations (820 image tokens ×
36 layers × 9,728-wide MLP) only ~2-sample chunks fit. Chunked
backward works (gradient-exact) but 6 passes/step taxes throughput.
torch.utils.checkpoint over the decoder blocks would cut saved
activations to ~1 layer's worth for ~30% recompute — the standard
trade at this scale. Scope: the Molmo2 transformer first (uniform
blocks make it trivial), Gemma later if a live-trunk E4B+ run recurs.
Gate: keystone oracle (checkpointed ≡ plain forward/backward, loss
bit-close) + a measured chunk-size ladder re-run.

## 21. Agentic-loop & infrastructure deep review — `delivered`, awaiting owner sign-off (review published 2026-08-07 00:3xZ)

**Status 2026-08-07:** the main deliverable is published —
[the review post](posts/2026-08-07-agentic-loop-review.md) with 7
prioritized proposals (P1 babysit CLI, P2 queue-as-data, P3
pre-commit hook, P4 now.md skeleton, P5 deadline stamp, P6 gpu test
markers, P7 home-dir/ctrl hygiene) + inline prompt/driver diffs.
Applied as class fixes (no sign-off needed): `archive_now.py`
(2026-08-06), `discord.py post --body-file` (2026-08-07). Everything
else is **blocked on owner review** — first owner reply
re-prioritizes; P3+P5+P6+P7 fit one work session, P1 and P2 one
each.

Owner steering, verbatim scope: "a deep review of your charter focus
on optimising the way you work and your local infrastructure … The
overall exercise is to improve your core agentic loop." A bounded
work session (CPU-only, GPU-independent) producing a written review +
concrete proposals for owner sign-off, covering: (1) tooling gaps —
what would raise throughput (e.g. a single `babysit` CLI that bundles
box/local liveness + curve-vs-anchor checks + Discord poll; a
Discord-post helper that takes a file argument so shell quoting can
never garble a message again — bitten 23:38Z); (2) code debt worth
burning (stale tmux sessions, ~-level launcher/log sprawl vs
`fontaine/scripts/`, the `flow-matching-ctrl` checkout lifecycle);
(3) testing infra — check.py wall-time now 22 s at 351 tests, fine,
but no smoke-tier separation for GPU-oracle runs; (4) the wake-up
framework itself — tick/work-session prompts, the `run_work_next`
chaining contract, lock handling across boundaries (the 08-06 class
fix), and whether queue state should live in a machine-readable file
instead of prose inside now.md; (5) now.md hygiene — 3,700 lines /
~109k tokens; sessions only ever read the head entry so it does not
bloat context per se, but head entries have grown into mega-paragraphs
and the file needs an archive policy (e.g. keep last N entries, roll
the rest to dated archive pages). Deliverable: a blog post with
prioritized proposals + the charter/prompt diffs, nothing applied
without owner review. Cost: 0 GPU-h.

## 19. AR sampled-draws eval (mean-of-samples) — `queued` (owner ask 2026-08-06 19:15Z)

Greedy decode is the AR family's single-draw voice; the flow family's
deployment read is mean-of-10 draws. The owner's fairness point: when
we quote flow mean-of-N, the AR models should get temperature-sampled
draws-N + mean-of-samples too. Instrument work, mirrors the flow draws
machinery: sample N chunk decodes per frame at a pre-registered
temperature (grammar mask unchanged — sampling within legal ids),
mean the decoded chunks, report draws1/drawsN like the flow panels.
Open design points for the pre-reg: temperature (fit on a probe set,
never the panel), whether aux value lines stay greedy (they should —
only the action block samples), and the fairness caveat that flow
draws are i.i.d. noise draws while AR draws share the prompt prefill
(cheaper per draw with the KV cache). First consumer: the
`fontaine_molmo2_ar_40k_ddp4` endpoint vs the A-s0 anchor —
BOTH sides get the same instrument or neither.

## 15. Literature-sourced arms — standing

The arXiv radar (VLA/robot learning, flow matching, action
tokenization, data curation) feeds this list; every borrowed idea
cites its source in the pre-registration; every "novel" idea gets a
search first. Local canon: π0, π0.5, SmolVLA, FAST
(arXiv:2501.09747).

- **π0.5 canon deep-read DONE (2026-08-07,
  [post](posts/2026-08-07-pi05-deep-read.md))** — π0.5
  (arXiv:2504.16054) + KI (arXiv:2505.23705), read against the live
  stage-2/Molmo2 question. Findings banked into #4 (two named
  attachment arms), #5 (FAST-vs-naive ablation), #6 (Implicit-HL
  replication + self-subgoal rung-(a) probe), #16 (Fig. 8 external
  anchor). Convention flag: π0.5's τ=1 is DATA; ours is NOISE.

- **IVRA (arXiv:2601.16207, lit slice 2026-08-06 16:2xZ)** —
  training-free, inference-side: VLAs flatten patches to 1D and lose
  2D spatial cues; IVRA injects vision-encoder patch-affinity signals
  into ONE LM layer ("where instance-level features reside"), no
  retraining, +4.2% on VIMA low-data / consistent LIBERO gains across
  LLaRA/OpenVLA/FLOWER. Fits #11's diagnosis exactly (acuity probe:
  position info sharpest at tower output, degraded through LM layers).
  Our analogue: bias trunk attention over soft tokens with
  tower-output affinities at eval — rung (a), zero training, panel
  first_mae is the readout. Cheapest falsification: single-layer
  injection on the flow teacher, panel-v2 first_mae vs banked
  2.0720 ctrl. Worth a probe if arm A's img280 read leaves grounding
  headroom on the table (interacts: more tokens vs better-used
  tokens are the same front, opposite ends).
