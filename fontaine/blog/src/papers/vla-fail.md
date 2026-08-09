# VLA-FAIL: noticing failure without ever having seen one

*Read 2026-08-09 (lit slice `lit-radar-0814`, priority 3). Paper:
[2606.21386](https://arxiv.org/abs/2606.21386) — "VLA-FAIL:
Efficient Task Failure Detection for Finetuned Vision-Language-Action
Models" (Seligmann, Gospodinov, Dincer, Neumann — KIT ALR; v1
2026-06-19, preprint). Banked because its two signals are (a) a new
mechanism class for the #6 verifier ledger and (b) our #22
boundary-disagreement read published as a detector.*

**The paper in plain words.** How can a robot notice it's failing at
runtime if it was never shown a failure? Watch two cheap things. One:
do the network's internal features on the current observation look
statistically unlike anything in the training demos? (A Mahalanobis
distance, fit once over the demo set — no labels, no rollouts.) Two:
does the plan the policy just made agree with the tail of the plan it
made a moment ago? (Receding-horizon control re-plans before the old
chunk finishes, so successive chunks overlap in wall-clock time —
disagreement on the overlap is free to measure.) Either signal
crossing a threshold — calibrated on ~20 successful rollouts —
flags failure. The two catch different failure styles, cost ~2 ms
per step against baselines that need 32 extra action samples, and
the paper adds a metric (AUCPDT) that scores not just whether you
detect but how *early*.

## What it contributes

- **LLMD** — last-layer Mahalanobis distance. Features are the
  action expert's tokens just before the final linear projection
  (not the VLM's last layer), with one clever twist for flow heads:
  the feature pass runs with a **single fixed prior-noise draw,
  sampled once and reused forever** — a shared-noise ticket for the
  feature pass, eliminating sampling stochasticity from the score.
  Per-chunk-position mean/covariance are fit on the finetuning demos
  in one gradient-free pass; the score is the max over positions of
  squared Mahalanobis distance (token-wise stats matter: going
  global degrades AUCPDT 0.19 → 0.24 on their worst case).
- **ACC** — action-chunk consistency. Compare the *unexecuted
  suffix* of the previous chunk against the new chunk's prefix over
  all overlapping timesteps: per-dimension MAE,
  **velocity-normalized** (divide by the dimension's motion range
  within the chunk, clamped below), EMA-smoothed with α=0.9,
  position dims only. Their own framing: "a velocity-normalized
  single-sample estimator of STAC" — that beats STAC on most
  real-world tasks at 1/32 the samples.
- **Calibration**: OR over the two scores, thresholds from a
  time-constant conformal band at 0.05 on ~20 *successful* rollouts.
  Plus **AUCPDT**: per failed episode, the normalized time of first
  detection (1 if missed), integrated over the precision–earliness
  Pareto front.

## The experiments it ran

Two flow-matching VLAs — π₀.₅ (3.6B) and X-VLA (0.9B) — on 6 real
tabletop tasks (~80 rollouts each, 3 seeds) and LIBERO-Plus in sim
(single seed). Baselines all need 32 action samples: ACE (chunk
entropy), STAC (distributional overlap divergence), Diff
(diffusion-loss pseudo-label). VLA-FAIL wins most real-world AUC-PR
cells (Kitchen 1.00, Stack T 0.96) and most earliness (PDT) columns,
but loses several PR cells to STAC in sim (Spatial 0.94 vs 0.99,
Goal 0.94 vs 1.00) and to Diff on one real task (Blocks 0.81 vs
0.93) — the honest headline is *comparable at 1/32 the cost and
earlier*, not dominance. Velocity normalization is worth 0.28 vs
0.38 AUCPDT on its ablation; detection degrades as the
receding-horizon overlap shrinks. Undisclosed: chunk/overlap sizes
H and R, covariance regularizer, v_min. No learned-detector or
logpZO baseline actually run; no AUROC/TPR@FPR anywhere, so no
bridge to the SAFE/FAIL-Detect numbers.

## What transfers to us

1. **#6 — a mechanism class our kill rule doesn't cover.** Our
   closed zero-training family (self-certainty, masked-contrast KL)
   was *policy self-report*: the model grading its own outputs. LLMD
   is a **demo-anchored density score** — external statistics, no
   self-report anywhere. It's a runtime monitor, not a candidate
   selector, but the mechanism ports: *LLMD-as-selector* (pick the
   candidate whose features sit least far from the demo
   distribution) is a genuinely new, cheap affirmative-case
   candidate — computable retroactively on banked draw dumps once a
   feature-dump hook exists. Against the banked verifier
   constraints: decoupled from the policy's probabilities (yes),
   though its features still come from the policy trunk
   (VLA-Corrector's warning applies); chunk-as-unit (partial —
   per-position stats, max-aggregated); no labels at all (stronger
   than the dense-labels rule needs).
2. **#22 — our seam read, with three borrowable deltas.** ACC is
   the boundary-disagreement quantity we measured on our own stack,
   plus: velocity normalization (scale-free across slow/fast phases
   — cheaper than our model-error normalizer), EMA before
   thresholding (our jump numbers are instantaneous), position-dims
   restriction. The conformal-band-on-20-successes recipe and
   rank-transform+min fusion are borrowable wholesale.
3. **The fresh-noise interaction is the sharpest cross-read.** ACC
   compares *sampled* chunks, so our measured ~3.3-unit fresh-noise
   mode term sits inside their signal as an irreducible noise floor
   — which they never decompose. They apply the fixed-noise trick
   to LLMD's features but **not** to ACC's actions; our shared
   noise ticket would shrink ACC's null distribution and make the
   detector strictly more sensitive. That improvement falls straight
   out of our #22 boundary-incompat read — banked as a note on the
   idea page.
4. **Population-level confirmation of the boundary read**: detection
   quality degrades as overlap shrinks, single-action overlap
   retains "some" signal. No per-boundary jump statistics though —
   our 1.1–1.3× and 11–14× ratios have no counterpart.

## What doesn't transfer

- **ACC needs receding-horizon overlap** — their own stated
  limitation ("does not directly apply to fully open-loop chunk
  execution"). Until a receding-horizon deployment exists on our
  side, ACC is offline-analysis machinery only.
- **Both tested policies are flow heads.** The "equally applies to
  discrete VLAs" claim for LLMD carries zero evidence; ACC on AR
  heads is not even discussed.
- **Its blind spot is exactly our open gap.** Verbatim: it "can miss
  failures that are consistent in features and actions, such as
  confidently stopping or ignoring a language instruction."
  Confident coherent failure is plausibly what our alive oracle
  ceiling (−0.250) contains — a detector family closed under
  "looks in-distribution and self-consistent" cannot see it.
- Calibration fragility: one outlier in the 20-rollout calibration
  set "can significantly raise the threshold"; no size ablation.

## Which idea/arm it fed

[#6 aux attribution](../ideas/06-aux-attribution.md) — the verifier
ledger gains the *demo-anchored density* mechanism class, outside
the closed self-report family; LLMD-as-selector named as the
cheapest affirmative-case arm (needs a feature-dump hook + its own
pre-reg). [#22 async staleness](../ideas/22-async-staleness.md) —
our boundary read's machinery published as a detector; three
normalization/smoothing deltas banked, plus the observation that a
shared noise ticket (which they only apply to features) would
tighten their own detector. Menu unchanged; everything stays parked
on #16's closed-loop entry condition.
