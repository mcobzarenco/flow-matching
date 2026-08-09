# Ideas

The backlog, one page per idea (sidebar, or the index below). Every
idea page carries: hypothesis, expected effect, cost, cheapest
falsification, and the dated record of everything that has happened
to it since. Seeded 2026-08-05 from charter §8 (which distills the
mainline ledger, `docs/architecture.md` §7–8). Status tags: `queued`
/ `screening` / `running` / `confirmed` / `falsified` / `parked`.

This page is the **index**: what is hot right now vs what is on ice.
It is updated whenever an idea moves (the per-idea page is the
record; the line here is the hook). *Index last updated 2026-08-07.*

## Hot — actively pursued

- **`ar-draws` [#19 AR sampled-draws eval](ideas/19-ar-sampled-draws.md)** —
  `screening`. The AR side of the draws fairness programme.
  draws10_t1 read out 2026-08-07: all three pre-registered
  expectations met (Δ_AR −0.145, ~9× smaller than the flow gain —
  the mean-collapse shape). T-sensitivity dT table banked 23:09Z
  same day (record-only): monotone in T — 6.50/6.57/6.78/7.18 at
  T=0.5/0.7/1.0/1.3 on the q4 rows; the T=1.3 asymmetry prior
  confirmed, primary stays T=1.0. **Molmo2 arm READ OUT 08-08
  07:2xZ, all expectations met**: Δ_AR −0.154 [CI −0.195, −0.113] —
  the mean-collapse shape replicated on a second AR trunk (AR-100k
  was −0.145); draws10_t1 5.8492/1.9736 → leaderboard row 9, no
  overtake of the flow band; execution oracles byte-green. Decode
  microbench rows measured same window (box H100): greedy
  143.8/678.1, draws10 1191.2/6291.3 ms — row 8's mtime caveat
  retired. Lit (08-08,
  [steering III](papers/noise-space-steering-3.md)): SDN
  (2606.14084) — a judge-free per-step noise selector whose
  smoothness-only ablation carries most of its +18 pp real-robot
  gain; jerk-pick is a pure function of our banked draw stacks →
  record-only ceiling-ladder read EXECUTED same session: flow
  fresh-noise NULL on every diagnostic (agreement at the 10% null,
  −2.3% of the oracle gap; ODE draws uniformly smooth), AR
  real-but-small and T-monotone (5.6/7.5/20.9% of the gap at
  T=0.5/0.7/1.3, Spearman +0.36); never approaches mean-of-N —
  family decodes stand. Molmo2 stack half at the #19 landing.
- **`seam-screen` [#4 Stage-2 attachment seam](ideas/04-stage2-attachment.md)** —
  `decided` 2026-08-09
  ([memo](posts/2026-08-09-molmo2-stage2-attachment-decision.md)):
  **the frozen default stands** — sequential hard-freeze is the
  attachment recipe for the Molmo2 trunk class; KI-joint
  closed-unmeasured (owner cost-killed K at ~4160; F panel 9.4157
  beat state-copy by 2.35, 8 matched probes show no K edge, measured
  4.11× step cost, production frozen-first votes). Δ_seam@3750
  rescue read priced ~2.5 GPU-h (own pre-reg). Open remnants:
  arm 1 depth-of-reads, F-then-joint rung —
  **[pre-reg DRAFT posted 2026-08-09](posts/2026-08-09-prereg-fjoint-rung.md)**:
  J (unfrozen, no stop-grad, CE rider, from the banked F@10k
  expert) vs F2 (frozen continuation control), matched +5k,
  committed ~32 GPU-h ceiling 35, conditional 10k extension;
  **instrument LANDED 08-09 15:0xZ** (composite materializer +
  `--joint-unfrozen-seam` escape + AR-view compat, 12 oracles) —
  finalizes on owner go alone, venue ~08-12 post-adamc-endpoint.
  New 2026-08-09 ([LP-FT](papers/lpft-two-phase-schedules.md), owner
  a(t)/b(t) steering): the rung's THIRD same-shape citation and the
  first with a matched frozen control AND a mechanism theorem —
  feature distortion is front-loaded while the head is uninformed;
  align the head on frozen features first, then unfreeze (+1 ID /
  +10 OOD vs constant-schedule FT). Maps: expert=head, taps=features,
  F=LP phase; silent on F-vs-K itself (K's stop-grad blocks the
  distortion channel); prices the rung's ordering + the
  compute-Pareto case for cheap-a=0-steps-first. Draft note: joint
  phase should start from the CONVERGED F expert, not a fixed step
  count. New 2026-08-09 later session
  ([RDT2](papers/rdt2-umi-scaling.md), 2602.03310): a production
  vote for the F shape hours before the Δ_seam read — RDT2's 7B
  recipe is AR-first (protects VLM knowledge, ablated) + flow expert
  on a FROZEN backbone + 1-step distill, **no joint stage**; ledger
  context only, the frozen read is untouched. New 2026-08-09 fresh
  sweep ([Z-1](papers/z1-selective-joint-rl.md), 2606.31846): a
  FOURTH same-shape vote with a sharper second half — production
  GRPO on a flow VLA keeps the trunk frozen by default and unfreezes
  it per-task only on measured diagnostics (SFT success, early
  expert-only progress, rollout failure modes); joint as
  *conditional escalation*, not a scheduled phase — exactly the
  fjoint rung's conditional-extension clause (evidence thin: one
  task, no final-number decomposition). New 2026-08-09 lit `0815`
  ([Decoupled Action Expert](papers/decoupled-action-expert.md),
  2511.12101): the seam question's *capacity axis* measured — a 5M
  MLP denoiser pretrained on observation-free kinematics data, then
  frozen with only the conditioning pathway retrained, matches a
  244M U-Net (LIBERO 84.7 vs 79.3; 84.2 with the freeze), so the F
  arm is not expert-capacity-starved and a J-beats-F2 read should be
  read as trunk-representation adaptation, not expert relief; their
  conditioning ablation — cross-attention conditioning collapses
  under backbone freezing (76.4→5.9) while modulation survives —
  makes the banked F@10k expert task/trunk-entangled capital, not
  portable. Framing caveats loud: testbed is Diffusion Policy (no
  VLM anywhere), and the freeze direction is inverted vs our seam —
  capacity datum only, silent on frozen-vs-joint.
- **`new-trunks` [#17 New trunks / architectures](ideas/17-new-trunks.md)** —
  standing owner mandate. **Molmo2-4B AR 40k ENDPOINT READ OUT
  2026-08-08 ([results](posts/2026-08-08-molmo2-endpoint-results.md)):
  BEATS — 6.0079/2.1871 vs the E2B own-topology control 7.7966/3.9422,
  paired −1.717 [CI −1.80, −1.63] on 17,204 core frames; frozen
  decision executes, Molmo2 is the phase-2 flow-trunk candidate (the
  #4 attach screen holds this AR-adapted prefix frozen). At 40k it
  sits 0.21 behind AR-100k's greedy at 2.5× fewer steps. Endpoint
  probe 6.2075@40000 = the vu5k amendment's frozen-sanity bar input.** New 2026-08-07
  (owner question): the SigLIP-unfreeze evidence got its own slice
  ([vision-encoder-freeze](papers/vision-encoder-freeze.md) — both
  poles are real: unfreeze wins adaptation regimes, freeze/anchor
  wins OOD retention; our rung sits in the adaptation regime; recipe
  prior = full-FT vision at low LR, never LoRA-on-SigLIP);
  vision-unfreeze pre-reg
  [DRAFT](posts/2026-08-07-prereg-molmo2-vision-unfreeze.md) now at
  the owner-agreed amendment-3 design (warm-start two-arm vu5k:
  frozen-continue vs thawed-continue from the 40k endpoint, 5k
  steps, 0.3× LR reheat, vision LR 6e-6 tied to text; superseded
  10k-from-scratch form recorded in its §8); finalization PREP
  landed 08-07 (`485194b`: byte-audit clean, both arm launchers +
  prepared babysit entries) — execution is launch-only after the
  150-step memory smoke + amendment post + owner go, window
  post-attach-screen. New 2026-08-09
  ([VLM4VLA](papers/vlm4vla-trunk-ablation.md), 9-trunk sweep):
  strong external prior FOR the thawed arm — frozen vision encoder
  loses 1–3 pts uniformly across 9 trunks × 3 sims (name in the
  finalization amendment); ALSO a trunk-shopping warning: VQA
  capability→control correlation collapses off-Calvin (r≈−0.36
  Simpler, −0.19 Libero) — trunk swaps get priced by panel screens
  only, never benchmark cards. Not compute-matched; suggestive, not
  priced. New 2026-08-09 (radar hooks cleared,
  [VEGA](papers/vega-encoder-grounding.md) +
  [HyperVLA](papers/hypervla-hypernetwork-inference.md)): VEGA =
  the *third pole* between freeze and thaw — a 3D-aware-teacher
  alignment aux at the encoder output (projector discarded at
  inference) substitutes for unfreezing when the missing ingredient
  is spatial structure (frozen-FiT3D ≈ unfrozen-FiT3D probe); banked
  as vu5k interpretation lever + named cheap escalation if thawed
  wins. HyperVLA = the inference-efficiency pole for the trunk
  ledger (0.1M generated policy/episode, 4 ms/step, sim-only,
  2024-baseline caveats loud) + the generated-update normalization
  design rule; radar-only, changes no queued arm. New 2026-08-09
  later session ([Spatial Forcing](papers/spatial-forcing.md),
  VEGA's baseline examined): the aux pole gains a second recipe —
  LLM-layer-24 alignment where VGGT works as teacher (vs collapsing
  at encoder depth: teacher×depth interact) — and its real headline
  is convergence, ~3× fewer iterations to matched LIBERO success +
  25.8 pp at 5% data; a fewer-steps lever for the throughput
  accounting, teacher overhead unreported. SF may fit single-tower
  Molmo2 better than the VEGA recipe; escalation order unchanged
  (VEGA won head-to-head). New 2026-08-09: LAFP
  ([page](papers/lafp-latent-flow-policy.md), 2606.10517) fills the
  latent-action-from-video pole of the family map (LAOM +
  flow-in-latent beats BC on multimodal envs; Procgen only,
  skim-to-place) — the documented recipe if unlabeled-video
  pretraining ever enters via the RDT2/VISTA data premise. New
  2026-08-09 fresh sweep
  ([weight-decay correction](papers/weight-decay-correction.md),
  2512.08217, AdamC's direct successor): the live adamc_100k
  grad-norm watch gets its interpretive frame — expect flat
  grad/weight norms through decay but ~nil final-loss effect
  (AdamC 76.98 vs AdamW 76.92 ViT-S/16); our head-exclusion
  partition validated in two papers; our 10%-of-peak LR floor sits
  on the paper's recommended side (λ ∝ γ avoids terminal weight-norm
  suppression); caveat banked — AdamC may not reach steady state in
  a 100k window, so slow weight-norm drift ≠ falsification. New
  2026-08-09 lit `0812b` ([DFM-VLA](papers/dfm-vla.md) +
  [OneWM-VLA](papers/onewm-vla-one-token.md)): head-axis map
  completes to four quadrants — DFM-VLA (discrete tokens +
  whole-sequence refinement, LIBERO-Plus 77.8 vs π0.5 75.7) pairs
  with HiFlow to show *commitment, not discreteness*, is the
  expensive property, measured from both directions; and the
  predictive-supervision pole gains its cheap self-anchored variant
  (one pooled token/frame jointly denoised with actions, 14.7M LoRA,
  monotone bandwidth sweep, unsupervised-scaffold-worse-than-nothing
  ablation) — the plausible aux-rider entry for a trained trunk. New
  2026-08-09 lit `0813` ([Muon-SW](papers/muon-sw.md), 2607.23777 +
  [AsyncVLA](papers/asyncvla.md), 2511.14148): the adamc watch's
  *weight-norm* frame lands — corrected decay's norm target is
  LR-independent, so expect plateau-then-flat (peak-then-decline =
  the uncorrected signature); λ ∝ η now has three independent
  derivations; alignment-cosine probe banked as a free second
  opinion. And the commitment axis gets its within-model
  intervention: re-noising low-confidence tokens + regenerating with
  trusted context is worth ~5× extra denoise compute, 2/3 of it
  with a coin-flip selector — revisability itself carries the
  effect (must be trained in: bolt-on collapses 70.8 → 7.3). New
  2026-08-09 lit `0814`
  ([Hyperball](papers/hyperball-optimization.md), 2606.16899 +
  [Anytime](papers/anytime-pretraining.md), 2602.03702): the adamc
  watch goes **two-sided** — R⋆ ∝ √(η/λ) gives plateau-then-flat a
  third independent derivation AND a grad-norm-side test (corrected
  group grads should stay flat through decay; climb ∝ 1/√η with
  sagging norms = uncorrected), two free offline probes banked
  (‖∇L‖·‖W‖ constancy, stable rank), and the decay-inert trap named
  (λ=1e-5 on a pretrained init may never reach equilibrium); plus a
  chart-note — mid-run probe reads understate decayed-model quality
  (cosine's endpoint is largely implicit averaging). And
  [X-Tokenizer](papers/x-tokenizer.md) adds the commitment axis's
  zero-test-time-commitment corner (tokens as pure training signal,
  flow head executes). New 2026-08-09 lit `0815`
  ([Weight-norm criticality](papers/weight-norm-criticality.md),
  2607.21005 + [Weibull weight-scale](papers/weibull-weight-scale.md),
  2606.19367): the adamc watch gains its *failure-side* frame and a
  measured disambiguator — decay+normalization can drag
  scale-invariant norms toward a derived floor c⋆ = √(ηρ/2) where
  sharpness (∝ 1/‖u‖²) spikes the loss (MLP blocks carry the
  blow-up); the named failure signature is joint: per-group norm
  decline + that group's grad climb + co-timed train spikes, all
  three series already recorded. The decay-inert trap flips valence
  — spikes need λ ~0.01–1, so λ=1e-5 sits in the safe corner — and
  stops being unmeasurable: the three-force decomposition
  (alignment ≈88–94% of the norm-force budget, spline-recovered at
  92–94% accuracy from sparse *weights-only* checkpoints; decay
  exact from our known λ_t·η_t schedule; injection ~4% residual,
  not recoverable) turns per-matrix |F_decay|/|F_align| across the
  ~20 banked 5k saves into a number — ratio ≪1 = decay inert, →O(1)
  into the cosine tail = the AdamC balance is real. Bonus: the
  Muon-SW alignment-cosine probe becomes computable weights-only.
  Two offline probes join the endpoint list (distance-to-criticality
  margin; force chronicle).
- **`aux-subgoals` [#6 Aux attribution](ideas/06-aux-attribution.md)** —
  `confirmed` (aux HELPS actions, +0.462 cost when off).
  **CONSOLIDATED REPORT 2026-08-09
  ([Conditioning on words](posts/2026-08-09-fieldcond-subgoal-report.md),
  owner ask 13:21Z 08-08): the whole thread — aux attribution,
  rung (a), fields tables both trunks, mined ambiguous frames,
  (b)/(b′) ladder + priced escalations — on one chart-led page.** **Rung (a)
  self-subgoal probe READ OUT 2026-08-08
  ([results](posts/2026-08-08-selfsubgoal-results.md)): the slot is
  ALIVE — Δ_oracle −0.290 [CI −0.331, −0.225], 6× late-horizon,
  twice the AR draws-10 gain — but self-generated subgoals recover
  almost none of it (Δ_self −0.018, CI spans 0; no deployment win at
  3× decode cost). Channel read significant: same text via suffix
  is +0.043 worse than the slot — generation quality (phase
  estimation), not the channel, is the bottleneck. Escalations
  (subgoal-draws selection first) each need a new pre-reg.**
  **Rung (b) PRE-REGISTERED 2026-08-08
  ([pre-reg](posts/2026-08-08-prereg-subgoal-draws.md)): sample 9
  subgoal candidates (greedy + 8 at T=1), condition on the
  self-certainty pick (frozen scorer,
  [Self-Certainty page](papers/self-certainty.md), 2502.18581) —
  plus a record-only oracle-similarity CEILING arm that bounds
  every scorer at this width and adjudicates no-diversity vs
  no-scorer if the falsifier fires; gate ≤ 6 GPU-h; execution
  queued behind the goldenticket R1 chain. Instrument LANDED
  oracle-green 08-08 03:5xZ (draws mode + SC-sufficient stats dump
  + both selection arms + read script w/ 11 abort branches;
  check.py 489) — only the GPU-side preflight oracles remain
  before launch.** New 2026-08-07 (radar, both papers
  announced same day —
  [subgoal-sourcing page](papers/subgoal-sourcing-post-training.md)):
  two fresh directional priors before the read — HiRoC's
  subgoal-source cold start (Δ_self ≤ Δ_oracle expected; its
  alignment-SFT is a new named escalation) and VLA-Talker's
  inject-vs-supervise 15.9-pt gap (predicts the narrated arm is
  safe; tension with our aux-on result recorded + resolved on the
  page). New 2026-08-08 (slice while the arms decoded —
  [runtime-plan-verification page](papers/runtime-plan-verification.md)):
  the escalation ladder above rung (a) priced with published
  numbers — SV-VLA's cheap-gate-heavy-replan (verification without
  recovery crashes 90.9%→15.5%), VINE's subgoal-draws width
  scaling (peak at K=4), Do-What-You-Say's faithfulness gap (the
  execution-side noise our Δ_oracle/Δ_self split doesn't price).
  New 2026-08-08 ~04:1xZ (targeted slice BEFORE the rung-(b) read —
  [progress-from-logits page](papers/progress-from-logits.md),
  TOPReward 2602.19313 + ProgVLA 2605.28231): escalation routing
  pre-mapped for a no-scorer verdict — (1) masked-contrast
  prerequisite VERIFIED MET (MG-Select masks text/state, never
  frames; our subgoal-masked reference = the planner-less path,
  trained at 50% dropout — correction banked on the
  [self-certainty page](papers/self-certainty.md)); (2)
  history-conditioned planning is evidence-backed (TOPReward: phase
  zero-shot recoverable from a video prefix via one completion
  logit, incl. on Molmo2-8B; single frames are the measured rung-(a)
  bottleneck). Lane (a) verdict: nothing published beats SC
  label-free on open-ended text — the frozen scorer cell stands.
  New 2026-08-08 ~19:0xZ (slice while rung-2 stage-2 decoded —
  [RoVer page](papers/rover-learned-verifier.md), 2510.10975): the
  "scorer is the gap" escalation now has a priced recipe — a 0.2B
  PRM (40M trainable) trained offline from demos alone via
  anchor-centered preference pairs, no environment or success
  labels; its stated chunk–step mismatch limitation (gains go
  unstable on chunked policies — all of ours are) is pre-registered
  ammunition: any learned-verifier arm must score the chunk as the
  unit. ELASTIC (same slice) independently names verifier noise as
  THE parallel-scaling bottleneck — the Δ_ceil/Δ_bon split is
  built to price exactly that.
  **Rung (b′) READ OUT 2026-08-09
  ([results](posts/2026-08-09-subgoal-draws-cleanlist-results.md)):
  E6 FALSIFIED, adjudication NO-SCORER — SC pick is +0.210
  [+0.113, +0.312] WORSE than greedy self head-to-head (and +0.142
  worse than the bare baseline: it anti-selects), while the oracle
  ceiling is alive at −0.250 [−0.353, −0.148], concentrated
  late-horizon (−0.464 last-10%). Width is fine (eligible 8.06/9,
  0 fallback rows); the scorer is the whole gap. SC is dead as the
  picker; scorer-side escalations (learned verifier per RoVer,
  probe ranker, or distilling the 4,298 dumped oracle-pick pairs)
  each need their own pre-reg.** New 2026-08-09 (slice same session
  as the verdict —
  [label-free selection signals](papers/label-free-selection-signals.md),
  uPRM 2605.10158 + SDN 2606.14084): the scorer-rung design
  constraint is now published twice over — label-free signals work
  when they score the candidate SET jointly (uPRM batch-joint
  first-error inference beats supervised PRMs; SDN's kNN density
  contrast vs masked decode), and per-candidate confidence is
  exactly the shape that failed here; a subgoal-scorer variant of
  masked-contrast is sketched on the page (one masked decode per
  frame + the K conditioned decodes any selection arm already pays;
  the planner-less path is the masked side). Audit catch recorded
  on the page: SDN's jerk half was ALREADY executed 08-08
  (jerkpick: flow null / AR 8% of the oracle gap, banked) — the
  physics-side selector is priced and is not the answer alone; the
  scorer rung choice is now RoVer-style supervised (4,298 in-domain
  pairs available) vs set-joint label-free. **Rung (c) READ OUT
  2026-08-09 ([results](posts/2026-08-09-mcselect-results.md)):
  masked-contrast ANTI-SELECTS — (mc − self) +0.313 [CI +0.200,
  +0.429], worse than SC's +0.210; capture fraction −1.73,
  late-horizon +0.385 (the ceiling's slot, inverted), oracle
  agreement chance-level at 66% active picks. Informativeness is
  anti-correlated with quality: max-KL candidates are disruptive,
  not phase-right. Second strike ⇒ the pre-registered kill rule
  executed: the ZERO-TRAINING SCORER FAMILY IS CLOSED for this
  trunk — learned-verifier shapes need their own affirmative case.
  The ceiling itself stays alive (−0.250 vs bare). POST-MORTEM MAP
  READ 2026-08-09 (record-only, banked dump,
  [addendum](posts/2026-08-09-mcselect-results.md)): KL is
  rank-NOISE, not a reversed compass — per-row Spearman(KL, err)
  +0.012 [−0.005, +0.029], oracle-best uniform on the axis (0.498
  vs 0.5); the +0.313 harm is magnitude-driven (winner's curse on
  a noisy axis, value-level rho +0.126). SC was the better axis
  all along (−0.030, CI < 0, oracle-best at its top 30% vs 12.6%
  null) but ~6× too weak for an argmax. Axes mutually uncorrelated
  (+0.032) — two independent failures. Calibration bar for any
  learned verifier: beat |rho| ≈ 0.03 by ~an order of magnitude.**
  New 2026-08-09 lit `0812b`
  ([VLA-Corrector](papers/vla-corrector.md), 2607.01804): a
  *drift-monitor* verifier that escapes the closed candidate-scorer
  family on both axes (trained 40M from demos alone; judges temporal
  drift, not candidates) — two design constraints banked for any
  learned-verifier case: predict residuals not states, and keep the
  judge decoupled from the policy (+14.8 pp external vs internal
  head). Closed-loop only; parked on #16. New 2026-08-09 lit `0813`
  (three angles on the verifier ledger):
  [AsyncVLA](papers/asyncvla.md) — dense per-token error labels beat
  trajectory-outcome labels 70.8 vs 64.6, and its
  relative-confidence blind spot is our anti-selection failure class;
  [silent-failures](papers/silent-failure-observability.md) —
  modality > capacity, final-state exteroception carries the
  precision signal (proprio's 0.97 is a noiseless-sim artifact);
  [StreamVLA](papers/streamvla.md) — completion-anchored gating
  sidesteps the measured mid-execution phase bottleneck (τ-sweep
  flat 0.5→never-skip); refresh rule: event-triggered ≈
  always-reason at half latency ≫ fixed schedule. New 2026-08-09 lit
  `0814` ([VLA-FAIL](papers/vla-fail.md), 2606.21386): a verifier
  mechanism class the kill rule doesn't cover — last-layer
  Mahalanobis against *demo* statistics (zero training, fixed
  prior-noise feature pass, ~2 ms vs 32-sample baselines) is
  demo-anchored density, not policy self-report;
  **LLMD-as-selector** is the cheapest named affirmative-case arm
  (retro-computable on banked dumps once a feature-dump hook
  exists, own pre-reg required); caveat carried — its stated blind
  spot, confident coherent failure, is plausibly our ceiling's
  class. New 2026-08-09 lit `0815`
  ([Foresight](papers/foresight-failure-detection.md), 2606.23085 —
  hook corrected loudly): NOT a current-phase affirmative case — it
  trains on success *and failure* rollouts ("task-level labels"
  means label granularity, not a demos-only diet), so it enters the
  ledger as the **rig-phase supervised endpoint** (teleop attempts +
  worked/didn't tags = its full diet; LLMD keeps the cheapest-arm
  slot). Banked anyway: 0.78 balanced accuracy at an 8,557-step
  horizon (+0.14 over best baseline) from a 2-layer head on frozen
  *action-conditioned world-model* latents — third echo that
  decoupled features beat policy internals, sequence head mandatory
  (MLP near chance on real robots), an outcome-labels-suffice
  counterpoint to AsyncVLA's dense-labels result; the time-varying
  conformal band (δ_t = μ_t + q̂σ_t, calibrated on successes only,
  anytime FPR ≤ α) is a borrowable no-failure-data upgrade for the
  VLA-FAIL recipe; cross-policy transfer is asymmetric (π₀.₅→ACT
  0.94, ACT→π₀.₅ 0.56) — failure logs age across policy
  generations.
- **`noise-draws` [#1 Noise-draw ensembling](ideas/01-noise-draw-ensembling.md)** —
  flow mean-of-10 banked (5.365); batched draws merged 2026-08-07.
  **GOLDEN-TICKET SCREEN R1+R2 READ OUT 2026-08-08
  ([results](posts/2026-08-08-goldenticket-results.md)): tickets are
  REAL — R1 CONFIRM (sd 0.823 vs null line 0.0785, 12× the null;
  winner ticket 33), R2 REAL on 14,746 complement rows (paired
  −0.924 [CI −0.985, −0.866] vs stable-key; LARGER than the
  selection-biased probe delta): one fixed sha-pinned noise vector =
  5.6468/1.8963 core-pooled, ~75% of the mean-of-10 gain at 1/10th
  the draws — leaderboard row 7. Effect DIRECTIONAL, not norm
  (ticket-33 norm rank 29/64, corr(norm,score) −0.05) — the
  LAFM/DSRL structured-noise premise showing up unprompted.
  **SCREEN CLOSED 08-08 08:2xZ — R3 INTERESTING, 9× the band**:
  mean-of-top-10 **5.1847/1.3831** vs banked mean-of-10
  5.3645/1.4242 (Δ −0.180, record-only; best chunk AND first
  numbers measured on this panel — row-seating needs the paired
  follow-up now folded into the queued noise-ladder pre-reg). R4a:
  ticket 33 argmin in 4.4% of 792 probe datasets (top-10
  containment 29.8% ≈ 2× null; median cell 2 frames —
  selection-noise caveat). R4b: winner gain monotone in draw
  dispersion (−0.35 → −1.44 by quartile). Screen total ~5.55/6
  GPU-h. The
  **Rung-2 pre-reg FINALIZED 08-08 13:2xZ
  ([pre-reg](posts/2026-08-08-prereg-noise-ladder-perdataset.md)):
  stage 0 found a thin floor F=6 on banked data (split-half regret
  n=6 bin 2% under the permutation null; n=4–5 fail — the
  median-2-frame caveat was right), 97 qualifying datasets = 40.8%
  of panel core rows, 88/97 route away from ticket 33 (map sha
  15d92935…).** **Stage-2 READ OUT 08-08 19:4xZ — FALSIFIED
  ([results](posts/2026-08-08-noiseladder-rung2-results.md)):
  Δ_route +0.129 [CI95 +0.060, +0.205] entirely ABOVE zero on the
  6,014 held-out complement core rows (win table 34W/54L, sign
  p 0.042) — the in-sample −0.60 probe delta INVERTED out-of-sample;
  per-dataset argmin on ~6–20-frame cells memorizes its cell even
  past the F=6 permutation floor. Ticket 33 itself re-confirmed
  (routed-vs-stablekey −0.756; board row stays global t33). Rung
  CLOSED; measured prior inherited by every ladder rung above:
  specialization must prove held-out-row transfer AT SELECTION
  TIME. Record-only lead: routing wins chunk steps ~1–8, loses
  ~15+ — a chunk-position noise policy is a different, cheaper
  axis (no arm without its own pre-reg). Seating arm independent,
  in flight.**
  noise-structure ladder (per-dataset tickets → LAFM priors → DSRL
  state-conditioned) has met its entry condition — each rung needs
  its own pre-reg.** Lit
  (08-07, [LAFM page](papers/latent-action-priors.md)): the
  noise-structure ladder above the screen is now mapped — searched
  ticket → per-dataset tickets → LAFM's learned mode-prior library
  (2606.23420, training-time; +10.4 LIBERO-90 over FM at 110M) →
  state-conditioned noise (DSRL + 2026 kin, now read —
  [noise-space-steering page](papers/noise-space-steering.md)):
  DSRL's dual-critic RL-on-noise, LP-DS's off-manifold drift
  diagnosis (trust-region clause banked for any CEM escalation;
  ‖ε‖ ≈ √300 shell), FRS's reverse-ODE noise recovery + 10-demo
  DSBC distillation (also a #16 rig lever). R4 per-dataset argmin
  disagreement would be LAFM's "fragmented action space" showing up
  in our data; the whole ladder stays gated on stage-1 R1/R2. Both
  banked hooks closed same day
  ([part II](papers/noise-space-steering-2.md) — PAINT + UniSteer):
  a probeable prefix-locality property of our teacher noted
  (record-only), the per-step fixed-point inversion primitive is
  the numbers-backed default, no gate change. Lit (08-08,
  [steering III](papers/noise-space-steering-3.md)): the
  per-dataset-tickets pre-reg inherits three published priors —
  2603.11642's variance decomposition (noise main effect 1.4%,
  context×noise interaction 39.4%, best-shared-noise optimal in
  3.1% of contexts: per-dataset search aims at the interaction
  term); the channel exists because our Heun decode is path-intact
  (DDIM 0.96 vs DDPM 0.11 direction→jerk correlation — any sampler
  change re-tests the ladder); and chunk-boundary artifact is a
  named panel-blind unknown of ticket 33 (rollout-gated jerk read
  banked). Lit (08-08 ~19:0xZ,
  [ELASTIC page](papers/elastic-adaptive-compute.md), 2606.31132):
  a rung-3 candidate named — **dispersion-gated draw allocation**
  (full draws budget only where the banked R4b dispersion quartile
  says ensembling pays, 1 draw elsewhere); ELASTIC learns this
  allocation with per-task online RL and matches best-of-10 at 34%
  lower latency — our version is a zero-training offline re-read of
  banked dumps, gated on the rung-2 verdicts. Directed candidate
  expansion (RoVer, same slice) lands in noise space for us —
  prior art alongside LAFM/DSRL, not a new rung.

## Standing

- **`rig-benchmark` [#16 Few-shot rig-transfer benchmark](ideas/16-rig-transfer-benchmark.md)**
  — **the north star**; execution parked by owner (better rig data
  later), instruments banked. Short-term proxy: comm-holdout MAE +
  attribution. New 2026-08-07: the proxy itself got a lit slice
  ([offline-validation](papers/offline-validation.md) — raw MSE
  measured at ρ −0.61 vs rollout success, sign flips exist);
  critical-frame re-pooling rung **executed same-day — every
  published ranking holds on the critical pool, separation widens**
  ([results](posts/2026-08-07-prereg-critical-frame-repooling.md)).
  Rig-time menu now four deep
  ([noise-steering II](papers/noise-space-steering-2.md)): ticket →
  DSBC (10 demos) → UniSteer (teleop corrections→noise, 20%→90% in
  ~66 min, OOD-robust where DSRL isn't) → DSRL (rewards). New
  2026-08-09 ([Hy-Embodied stack](papers/hy-embodied-stack.md),
  2606.14409): the **weight-space pole** of the post-SFT menu banked
  — FlowPRO preference RL (flow loss as implicit reward,
  intervention-and-rollback pairs, +6–12 pts over DAgger, retention
  UNMEASURED vs FlowDAgger's 0.88-vs-−0.94) + the H=50
  Bézier chunk-stitch deployment lever. New 2026-08-09 later session
  ([async II](papers/async-execution-2.md) +
  [RDT2](papers/rdt2-umi-scaling.md)): TTFA accounting
  (E[react] = Δt_infer + ½·Δt_exec — the chosen execution horizon
  will dominate decode latency on the rig) + ABPolicy's jerk
  instruments banked as bench design inputs; and the north-star
  premise gains its hours-scale existence proof — 10k h of
  robot-free UMI data transfers zero-shot cross-embodiment with a
  predictable data exponent (β≈0.23), though the zero-shot column
  needs a UMI-twin gripper the SO101 rig lacks. New 2026-08-09 fresh
  sweep ([Z-1](papers/z1-selective-joint-rl.md)): the post-SFT
  menu's RL pole gains a data-efficiency datum — +13.2 pts over SFT
  on 24 RoboCasa tasks from 1,199 public demos and sparse success
  rewards only (flow-SDE log-probs + task-wise GRPO); sim-only, zero
  compute accounting. New 2026-08-09 lit `0812b`
  ([π-StepNFT](papers/pi-stepnft.md), 2603.02083): RL-pole entry 4
  and the pole's first measured IND-vs-OOD trade — critic-free
  step-wise contrastive on flow-SDE transitions matches PPO IND but
  beats it +11.1 pp OOD (semantic 49.1 vs 25.4); for the
  few-demo/shifted rig regime the trade favors critic-free. Pole
  stays sim-first (8×H100, co-located rollouts, success flags). New
  2026-08-09 lit `0813` ([SA-VLA](papers/sa-vla.md), 2602.00743 +
  [silent-failures](papers/silent-failure-observability.md),
  2606.03134): RL-pole entry 5 measures the first *negative* sign —
  sparse-reward PPO lands below no-RL (77.5 vs 81.0 OOD); published
  gains are protective machinery (dense privileged rewards, frozen
  spatial injection, learned exploration noise); and a bench
  constraint for the north star — telemetry success flags run
  32–48% false-positive in clean sim, so binary-success RL and any
  rig bench need an exteroceptive label audit (final-frame check is
  the cheapest sufficient form). New 2026-08-09 lit `0814`
  ([FPO](papers/fpo-flow-policy-optimization.md), 2510.09976, ICRA
  2026): RL-pole entry 6 fills the missing gradient route —
  likelihood-free PPO ratio from the CFM-loss *change* (no SDE, no
  BPTT); ALOHA ~40%→65%+ own-baseline sparse-reward sim; its
  ablation says the gradient route carries the method (−46 pp) while
  the critic ensemble is seasoning (−7 pp); third frozen-trunk vote;
  env/compute cost unreported, zero retention measurement. New
  2026-08-09 lit `0815` ([RedFlow](papers/redflow.md), 2607.27782):
  RL-pole entry 7, the first fully *offline* + real-robot entry —
  failed deployment rollouts become action-level corrective
  supervision (progress-model advantage + context clustering, then
  attraction/suppression/redirection targets on the flow endpoint):
  real-world 56.7→74.7 avg across three AgileX tasks from 100–200
  rollouts + binary outcomes, no envs, no teleop, no critic; matches
  PPO/GRPO/DDPO on LIBERO-Spatial at ~10× fewer samples. Re-prices
  the pole (parallel-env infra is no longer the universal entry fee)
  and bridges to the intervention levers — corrections without a
  human. Sharpest ablation repeats the protective-structure pattern:
  knowing which failures NOT to correct carries −11.5 avg alone.
  Caveats: retention unmeasured (FlowDAgger critique stands),
  deliberately weakened base policy, progress model unvalidated on
  rig scenes.
- **`lit-arms` [#15 Literature-sourced arms](ideas/15-literature-arms.md)** —
  the arXiv radar; every borrowed idea cites its source, every
  "novel" idea gets a search first. Feeds the Papers section.
- **`infra-hardening` [#18 Instrument & infra hardening](ideas/18-infra-hardening.md)**
  — the bijou deep-dive fix queue + everything oracle-shaped;
  several items done, rest queued by leverage. New 2026-08-07: item
  9 async checkpoint saves LANDED (owner HIGH; byte-identical
  oracle, ~14% wall-time payoff at the attach screen) + its
  [checkpointing-systems lit page](papers/checkpointing-systems.md).
  New 2026-08-08: **owner-steered molmo2 perf/memory deep review
  SHIPPED** ([review](posts/2026-08-08-molmo2-perf-review.md)) —
  suffix attention on the MATH backend (13×/layer measured), ViT
  eager einsum (13×/block), act-ckpt absent from live launchers;
  S-bundle queued (`molmo2-perf-fix-prereg`, ~8–15% step expected).
  **2026-08-09: the box ladder FALSIFIED the bundle**
  ([results](posts/2026-08-09-perfpass1-box-results.md)) — C −7.3%
  (a regression; P1 cuDNN alone −10.8%; the local microbench did not
  transfer to 4×DDP). Frozen <5% branch: nothing perf-claiming
  lands; P2 + bitwise items split to a hygiene item; P1 dead.
  Same-session lit ([loss + mask page](papers/memory-efficient-loss-attention.md)):
  CCE (2411.09009) banked as the CE escalation ladder
  (valid-row → two-segment lse → CCE; entry = wanting
  backward-chunks < 6 or batch > 12), FlexAttention banked as the
  dense-mask successor gated on compile (#2b) or long prefixes.

## On ice — queued or parked, each with its named trigger

- **`throughput-compile` [#2 Throughput: bucketing + compile](ideas/02-throughput-bucketing-compile.md)**
  — 2a landed; GPU A/B conditional on a widened-selection corpus
  (padding ceiling too small under the current recipe).
- **`longer-training` [#3 Longer training](ideas/03-longer-training.md)** — needs the
  own-baseline reference arm first. New 2026-08-09 lit `0814`
  ([Anytime Pretraining](papers/anytime-pretraining.md),
  2602.03702): the horizon-churn fix published — constant-LR trunk +
  branch decays from banked saves matches per-horizon-tuned cosine;
  our 40k→60k→100k restart-from-the-floor lineage is the paper's
  motivating pathology; mid-run checkpoint-averaging "endpoint
  preview" priced as a CPU read (own pre-reg needed). Hook
  correction on the page: not a Defazio paper.
- **`tokenizer-v3` [#5 FAST tokenizer v3](ideas/05-fast-tokenizer-v3.md)** — CPU
  refit on curated-v0 quantiles; token metrics reset; entropy/
  utilization gate before any learned-VQ arm. New 2026-08-09
  ([DFM-VLA](papers/dfm-vla.md)): MAAT's metric-aligned embeddings
  (+4.4 pp for a refinement decoder) = the first measured
  order-preservation datum; "ablate embedding metric structure"
  banked as a free rider on any v3 refit. New 2026-08-09 lit `0814`
  ([X-Tokenizer](papers/x-tokenizer.md), 2606.14752): a clean
  external null for learned-VQ in the *executable* role — RVQ-no-aux
  loses to FAST on control (69.1 vs ~73.0) and the full tokenizer
  reconstructs 17% worse; its wins are auxiliary-supervision only
  (tokens never executed, needs a frozen 7B teacher). Gate stands;
  two v3 riders banked (quantile normalization confirmed;
  WER-under-noise probe — FAST's BPE re-segmentation blows up 3× at
  σ=0.008).
- **`stream-schedule` [#7 Stream-schedule re-test](ideas/07-stream-schedule.md)** —
  enters at the short-run screen rung.
- **`vocab-head` [#8 Shortlist/output-vocab head](ideas/08-shortlist-vocab-head.md)**
  — VRAM lever for ar_backbone; design concretized, unbuilt.
- **`data-levers` [#9 Data levers](ideas/09-data-levers.md)** — state-dropout arm
  C answered "adopt nothing"; p=0.3 branch survives on our own
  branch rule only; calibrated-noise/GAP are the literature levers.
  New 2026-08-09: VISTA
  ([page](papers/vista-umi-validation.md), 2606.04708) — physics
  validation of human-collected data (continuity/collision/fidelity
  scores predict deployment: 65% vs 0% OSR at matched grasp); banked
  hook: the embodiment-agnostic *continuity screen* is a zero-GPU
  read on our own corpus (per-tick displacement thresholds — a
  kinematic-corruption dimension orthogonal to the VLM judge).
  **Hook CLOSED 2026-08-09
  ([screen results](posts/2026-08-09-corpus-continuity-screen.md)):
  qualified null — tail 0.23%, dominated by the wrap census's two
  known repos; 42 new dropout episodes far under the curation
  effect-size line; instrument banked as a curated_v1 intake
  filter.**
- **`base-vs-it` [#10 E2B base-vs-IT swap](ideas/10-e2b-base-vs-it.md)** —
  backbone-swap arm, pre-registered prediction ±0.2.
- **`visual-grounding` [#11 Visual grounding arms](ideas/11-visual-grounding.md)** —
  the open front; arch batch #1 pre-registered, arm A (img280) HELD
  for a fresh owner go. New 2026-08-09 lit `0812b`
  ([HiF-VLA](papers/hif-vla.md), CVPR26): codec motion vectors
  (~free from stored video) + decode-stage AdaLN banked as the
  cheapest history-arm representation, strictly behind the
  aliasing-census entry condition. New 2026-08-09 lit `0813`
  ([SA-VLA](papers/sa-vla.md)): aux-family fourth integration mode —
  frozen VGGT-token injection via gated cross-attention (read-only,
  erosion-proof under RL; +2.25 zero-shot, viewpoint-loaded); the
  family axis is now *when the geometry is allowed to change*.
- **`one-step` [#12 Solver/Heun-gap + 1-NFE distill](ideas/12-solver-heun-gap.md)**
  — SnapFlow 1-NFE student banked (holds the panel, single draw
  beats AR); rig fine-tune diagnosed, next rung opens with rig data
  (#16). New 2026-08-09: FAFM
  ([page](papers/frequency-aware-flow-matching.md), 2606.20135) —
  flow matching over DCT coefficients (M≈K/3: 17×6 target instead of
  50×6, smooth by construction, +λ‖v̇−ξ̇‖² = a weighted H¹ loss);
  banked as a representation option for future distill rungs (our
  within-chunk smoothness is already clean per SDN, so the live half
  is the smaller target, not the smoothing). Also fed #9
  (mixed-frequency data becomes well-posed — their Prop 1 +
  94%→0% π₀ collapse demo) and #16 (LDLJ jerk metric).
- **`sign-convention` [#13 Sign-convention repair](ideas/13-sign-convention.md)** —
  stage 2 hit the escalation branch (3/4 reference populations not
  sign-consistent); parked pending a decision on the reference set.
- **`async-staleness` [#22 Async staleness bridging](ideas/22-async-staleness.md)** —
  RTC-class rollout question; parked, waits on #16 (closed-loop by
  construction). New 2026-08-07: PAINT (2606.19774,
  [noise-steering II page](papers/noise-space-steering-2.md)) —
  training-free initial-noise selection matches RTC on a chunk-50
  π₀ with no gradients. New 2026-08-09
  ([async execution II](papers/async-execution-2.md)): FASTER's
  horizon-aware schedule attacks the delay itself (first action in
  1 flow step of N, TTFA 1.3–3×) and tiles across batched draws —
  the 18-tick mean-of-10 staleness may be a scheduling artifact;
  DEFLECT measures RTC/BID at ≤5% for d≥5 and fixes it with
  stale-vs-fresh FM-DPO (restart-corrected net +1.6–2.3 pp). **Arm
  order re-banked: measure naive-switch → HAS-on-decode → SEAM →
  PAINT → A2C2 → TT-RTC/DEFLECT.** New 2026-08-09: **the free
  boundary read EXECUTED — NOT a null**
  ([results](posts/2026-08-09-boundary-incompat-results.md)): seam
  disagreement ≈ 1.1–1.3× model error, boundary jump 11–14× per-step
  motion, and the dt→0 split shows fresh noise carries a ~3.3-unit
  mode term that a shared noise ticket deletes entirely (2.07 vs
  6.04, below even greedy AR). The direction is confirmed, with a
  measured target; still parked on #16 for any fix. New 2026-08-09
  lit `0812b` ([VLA-Corrector](papers/vla-corrector.md)):
  event-triggered truncation datum — a 40M drift monitor cutting
  stale chunks is +11.65 of +15.65 pp before any steering; *when to
  cut* dominates *how to steer*. Menu adjacency, closed-loop,
  parked on #16. New 2026-08-09 lit `0813`: two placements, menu
  unchanged — [AsyncVLA](papers/asyncvla.md) is NOT async execution
  despite the name (all correction pre-execution; filed so the
  title isn't re-banked), and [StreamVLA](papers/streamvla.md)'s
  gate re-reasons but never cuts the chunk (complements the
  truncation axis; event-triggered refresh economics datum banked).
  New 2026-08-09 lit `0814` ([VLA-FAIL](papers/vla-fail.md),
  2606.21386): our seam read published as a detector — ACC compares
  the previous chunk's unexecuted suffix vs the new chunk's prefix
  over the receding-horizon overlap; three deltas banked (velocity
  normalization, EMA α=0.9, position-dims-only) + the
  conformal-band recipe; and the cross-read — they fix prior noise
  for features but NOT for ACC's actions, so our shared noise
  ticket would tighten their own detector (the ~3.3-unit
  fresh-noise mode term is their undecomposed noise floor).

## Answered — banked results

- **`wrap-census` [#14 ±180° wraparound census](ideas/14-wraparound-census.md)** —
  measured: 1.24% of pooled panel MAE; under the gate, banked.
- **`activation-ckpt` [#20 Activation checkpointing](ideas/20-activation-checkpointing.md)**
  — landed + oracle-gated; the GPU ladder lives on as #4's K smoke
  item.
- **`loop-review` [#21 Agentic-loop deep review](ideas/21-agentic-loop-review.md)**
  — CLOSED: P1–P7 all landed, owner-signed.

