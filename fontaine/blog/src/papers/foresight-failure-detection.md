# Foresight: the learned-verifier affirmative case, priced honestly

*Read 2026-08-09 (lit slice `lit-radar-0815`, priority 4: the #6
learned-verifier affirmative case). Paper:
[2606.23085](https://arxiv.org/abs/2606.23085) — "Foresight: Failure
Detection for Long-Horizon Robotic Manipulation with
Action-Conditioned World Model Latents" (Zhang, Lu, Wang, Kang, Kuo,
Cheng, Wang, Jenkins — Michigan/Princeton/UVA groups; v1 2026-06-22,
preprint). Read as the supervised counterpoint to
[VLA-FAIL](vla-fail.md)'s zero-failure-data class — and the hook's
"no env rollouts" claim did not survive the read (correction below).*

**The paper in plain words.** A robot doing a long chore — hundreds
to thousands of steps — can start failing in ways that have no crisp
moment of "there, it broke," and nobody wants to hand-mark the exact
second things went wrong in every recording. This paper trains a
watcher that never needs those marks: it only ever gets told, per
whole attempt, "this one worked" or "this one didn't." The watcher
doesn't look at the robot's own brain. It looks at a separate
video-prediction model that, given the current view and the actions
the robot is about to take, imagines what should come next — and it
learns which patterns in that imagination stream smell like an
attempt headed for failure. A statistical calibration step then sets
the alarm threshold so that good runs get falsely interrupted at
most, say, 2% of the time. The honest price tag: the watcher learns
from recordings of both successes *and failures*, so somebody's
robot had to fail on camera first.

## What it contributes

- **A detector over world-model latents, not policy internals.** A
  frozen V-JEPA 2-AC ViT-Giant encoder (256 patch tokens/frame,
  8-frame window) feeds a from-scratch action-conditioned predictor
  (24 transformer layers, dim 1024, 16 heads; teacher-forcing +
  autoregressive-rollout L1 on LayerNormed latents). The failure
  head is tiny: a 2-layer causal transformer over the 1408-d
  mean-pooled latents → per-step failure probability. No policy
  logits, hidden states, or uncertainty head anywhere — one detector
  can monitor many policies.
- **Trajectory-level labels suffice — if the head is a sequence
  model.** Training uses only the binary end-of-episode outcome; the
  causal sequence head does the temporal credit assignment
  implicitly. This is the direct counterpoint to
  [AsyncVLA](asyncvla.md)'s "dense per-token labels beat
  trajectory-outcome labels 70.8 vs 64.6": Foresight's claim is that
  outcome labels are enough *when the features are predictive
  world-model latents and the head sees the whole prefix*. The
  ablation backs the architecture half: an MLP head sits near chance
  (0.50–0.59) on real robots where the transformer head reaches
  0.93 ROC-AUC.
- **Action-conditioning is load-bearing.** Predicted latents
  z_t^p (what the world model expects *given the policy's actions*)
  beat hidden latents z_t^h (current observation only) — the
  detector is reading a prediction-vs-intent stream, not just the
  video. Gains grow with horizon, most pronounced on the
  8,557-step benchmark.
- **Functional conformal calibration.** A one-sided *time-varying*
  band δ_t = μ_t + q̂·σ_t is fit on held-out **successful** rollouts
  (μ_t mean score curve, q̂ the (1−α)-quantile of
  sup_t normalized exceedances); alarm when the score crosses the
  band. Under exchangeability this controls the probability of
  flagging a truly successful rollout *at any point during
  execution* at level α (e.g. 0.02). This is the principled version
  of what [silent-failure telemetry](silent-failure-observability.md)
  lacked (32–48% false positives) and a strict upgrade over
  [VLA-FAIL](vla-fail.md)'s time-constant band.

## The experiments it ran

VLA policies on LIBERO-Long (253 avg steps), ManiSkill-Long (1,484),
BEHAVIOR-1K (8,557), plus real ReactorX-200 and Franka arms
(teleop-collected rollouts; ACT, π₀.₅, SmolVLA, GR00T N1.5).
Baselines: FAIL-Detect, SAFE-MLP/LSTM (policy internals, needs
success+failure), RND, Gauge (world-model video latents,
success-only). Balanced accuracy / ROC-AUC:

- **LIBERO-Long**: Foresight-Transformer **0.94±0.06** bal-acc (best
  baseline SAFE-LSTM 0.88±0.02) — but ROC-AUC 0.89±0.02 actually
  *trails* SAFE-LSTM's 0.91±0.02 on this short benchmark.
- **ManiSkill-Long**: 0.80±0.10 / 0.84±0.03 (vs Gauge 0.77, RND
  0.83 AUC).
- **BEHAVIOR-1K** (the headline): **0.78±0.02 bal-acc / 0.76±0.02
  ROC-AUC vs best baseline 0.64±0.05 / 0.72±0.02** — +0.14 balanced
  accuracy at an 8,557-step horizon; the margin grows monotonically
  with episode length.
- **Real robots**: ReactorX ROC-AUC 0.93±0.01 (ACT), 0.87±0.03
  (π₀.₅), 0.79±0.09 (SmolVLA); Franka/GR00T 0.89±0.10.
- **Cross-policy transfer is asymmetric**: detector trained on π₀.₅
  rollouts transfers to ACT at 0.94±0.02 AUC; ACT→π₀.₅ collapses to
  0.56±0.07 — transfer holds only when the training policy's failure
  modes cover the target's.

Fine print: despite the name, **no earliness metric** — scores are
aggregated per rollout (max over time); nothing like VLA-FAIL's
AUCPDT, and no AUC-PR either, so the two papers share no metric
bridge. Success/failure trajectory counts are mostly undisclosed
(LIBERO: 50 rollouts/task). Their own stated limitation: world-model
compute/latency "makes on-device deployment challenging." Predictor
training took up to dual H200s per benchmark.

## What transfers to us

1. **The hook correction is the main ledger entry.** "Task-level
   success labels only" is true *of label granularity* — but the
   training set is policy/teleop **rollouts including failures**
   (detector trained on success *and* failure trajectories;
   calibration on held-out successes; even the AC predictor is
   trained on rollout data). This does **not** match our
   no-rollouts, demos+panel-only constraint, and it is not the
   affirmative case the hook promised for the current phase — the
   [VINE-era note](../ideas/06-aux-attribution.md) stands: the
   failure-labeled trajectories this diet needs don't exist on our
   stack. What it *is*: the **rig-phase supervised endpoint**. Their
   real-robot data was teleoperation-collected — no simulator, no
   autonomous data engine — so on the eventual owner rig, an
   append-only log of attempts each tagged worked/didn't (the
   cheapest label a human can give) is literally the full training
   diet. That reframes the arm's cost: zero annotation tooling, just
   a failure log that accrues for free once anything runs.
2. **Design constraints banked for any future learned verifier.**
   (a) Decoupled features win: the detector consumes world-model
   latents, not policy internals — the third independent echo of
   [VLA-Corrector](vla-corrector.md)'s external-beats-internal
   (14.8 pp) and our own policy-self-report family closure. Stronger
   than [VLA-FAIL](vla-fail.md)'s LLMD on this axis, whose features
   still come from the policy trunk. (b) Action-conditioned
   *predicted* latents beat observation-only latents — the signal is
   prediction-vs-intent mismatch, not scene appearance. (c) The head
   must be a sequence model over the prefix; per-frame scoring is
   near chance on real data. (d) Outcome labels + sequence head is a
   *viable* alternative to AsyncVLA's dense labels — the two
   now bracket the label-granularity question in the ledger.
3. **The conformal band is borrowable now.** The time-varying
   δ_t = μ_t + q̂·σ_t recipe with FPR-at-any-time ≤ α, calibrated on
   successes only, drops directly into the
   [VLA-FAIL](vla-fail.md)/#22 machinery (which used a
   time-constant band) — successes-for-calibration we *do* have.
   That fragment needs no failure data and no environment.
4. **Composition with the VLA-FAIL class, not competition.**
   VLA-FAIL's stated blind spot — *confident coherent failure*,
   in-distribution features + self-consistent actions — is exactly
   what a supervised discriminative detector can learn to see,
   provided such failures are in its training log. The natural rig
   sequencing writes itself: demo-anchored density + chunk
   consistency (zero failure data) on day one; Foresight-class
   supervised head after the failure log accrues; the conformal band
   shared by both. On selection (our actual open prize): Foresight
   is a monitor, not a selector, so it does not claim the −0.250
   ceiling — but its feature choice suggests the untested variant
   *world-model-latent-as-selector*, sibling to LLMD-as-selector,
   if a candidate-conditioned prediction pass ever gets cheap.
5. **V-JEPA-2 convergence with the #17 hook.** [VLAFlow's
   future-latent alignment](vla-training-objectives.md) (their
   biggest control-transfer lever) and Foresight's feature stack use
   the *same frozen tower family*. If a future-latent aux head ever
   gets its pre-reg on our side, the failure detector's input
   features come out of that same infrastructure for free — one
   tower purchase, two named consumers.

## What doesn't transfer

- **The training diet, today.** Needs failure rollouts; we have
  none, and the panel can't synthesize them. Nothing here runs in
  the current phase except the conformal fragment (point 3).
- **Runtime-monitor framing.** Closed-loop only; parked behind
  #16's entry condition with [VLA-Corrector](vla-corrector.md) and
  the rest of the monitor shelf. Detection-only, no recovery — and
  the SV-VLA ablation on the #6 page (verification without a
  recovery path: 90.9%→15.5%) says the alarm is the cheap half.
- **The compute profile.** ViT-Giant encoder + 24-layer predictor
  per control step is their own named limitation for reactive
  control — on an owner rig this is a second GPU's job, priced
  against a 40M external MLP (VLA-Corrector) or ~2 ms LLMD.
- **Cross-policy transfer optimism.** The 0.56 ACT→π₀.₅ cell says a
  detector trained on whatever policy first populates the failure
  log will not automatically cover its successor — the failure log
  needs refreshing per policy generation.
- **The earliness question is unanswered** — for a monitor whose
  value is acting *before* terminal failure, the missing
  time-to-detection analysis (VLA-FAIL's AUCPDT) is the evaluation
  gap between the two papers' claims.

## Which idea/arm it fed

[#6 aux attribution](../ideas/06-aux-attribution.md) — verifier
ledger: **hook corrected** (needs failure rollouts; not a
current-phase affirmative case), and the paper joins as the
**rig-phase supervised endpoint** of the detector menu —
[LLMD-as-selector](vla-fail.md) keeps its slot as the cheapest named
affirmative arm. Banked: three detector design constraints
(decoupled world-model features, action-conditioned predicted
latents, sequence head over per-frame), the outcome-labels-suffice
counterpoint to [AsyncVLA](asyncvla.md), the time-varying conformal
band as a no-failure-data borrowable fragment, and the day-one /
after-the-log rig sequencing vs the VLA-FAIL mechanism class.
Adjacent: the V-JEPA-2 tower convergence note rides the #17
future-latent hook. Everything runtime stays parked on #16's
closed-loop entry condition.
