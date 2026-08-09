# Weight-norm criticality: where decay + normalization actually breaks training

*Read 2026-08-09 (lit slice `lit-radar-0815`, priority 1:
adamc-watch adjacency). Paper:
[2607.21005](https://arxiv.org/abs/2607.21005) — "Weight-norm
Criticality: A Mechanism for Loss Spikes Induced by the Normalization
and Weight Decay" (Li, Zhou, Xu — Zhi-Qin John Xu's group, SJTU; v1
2026-07-23, preprint). Fifth paper in our corrected-decay reading
after AdamC, [Chou](weight-decay-correction.md),
[Muon-SW](muon-sw.md) and [Hyperball](hyperball-optimization.md) —
and the first from the *failure* side: where the others ask how to
schedule decay, this one shows what happens when decay wins.*

**The paper in plain words.** Modern networks contain layers whose
output doesn't change if you scale their weights up or down — a
normalization step right after them erases the scale. Weight decay,
the small force that shrinks weights every step, therefore gets no
pushback from the loss on those layers: it can drag their size toward
zero without the network's predictions changing at all. But the
*landscape* changes. The smaller those weights get, the more sharply
curved the loss surface around them becomes — halve the size,
quadruple the curvature — until the optimizer's step size is suddenly
too big for the terrain and the loss spikes. The paper works out the
exact size floor below which this happens, shows spikes lining up in
time with weights crossing that floor, and localizes the blow-up in
transformers to the MLP blocks. The moral: weight decay can't be made
arbitrarily strong, and loss spikes late in training may be decay
quietly winning a fight nobody was watching.

## What it contributes

- **A second criticality axis.** Instability analysis usually tracks
  *learning-rate* criticality (Edge of Stability, `η·λ_max(H) ≤ 2`).
  This paper adds **weight-norm criticality**: for scale-invariant
  parameters `u` (weights feeding a norm layer), the Hessian block
  obeys `H_uu(αu, v) = α⁻²·H_uu(u, v)` (Theorem 5.1), so
  `λ_max ≥ α⁻²·λ_max(H_uu)` — curvature explodes **quadratically**
  as `‖u‖` shrinks. This is the same scale-invariance lemma behind
  [Hyperball's](hyperball-optimization.md) grad law
  (`‖∇L‖ ∝ 1/‖u‖`), taken one derivative higher: grads scale as
  `1/‖u‖`, sharpness as `1/‖u‖²`.
- **A concrete critical norm.** Combining the two criticalities:
  instability once `‖u‖ < c* = √(ηρ/2)` with
  `ρ := ‖u‖²·λ_max(H_uu)` (Prop 5.2); a sharper spike boundary
  `c*_spike = √(ηρ_grad/2)` using gradient-direction curvature
  `λ_grad = gᵀH_uu g/‖g‖²` (Prop 5.3). Since decay shrinks `‖u‖`
  monotonically on scale-invariant blocks (the loss can't resist),
  large-λ training *must* eventually cross the floor.
- **Localization in transformers.** During spikes, the top Hessian
  eigenvector concentrates in the **MLP modules** (the
  scale-invariant blocks under prenorm); disabling weight decay on
  the MLPs alone reduces spike frequency and — at 187M scale —
  yields consistently *lower* training loss.

## The experiments it ran

- **187M LLaMA-style transformer** (16L, hidden 1280), 100B tokens,
  1 epoch, AdamW, linear warmup to η=1e-3 then decay to 1e-4, grad
  clip 1.0, **λ ∈ {0, 0.5, 1}**: spike frequency rises visibly with
  λ (Fig. 1a — trajectories only, no spike counts reported). Rerun
  with MLP decay disabled: fewer/smaller spikes, lower loss (Fig. 9).
- **ResNet-50 / CIFAR-100** (SGD, η=0.003): same spike-vs-λ pattern.
- **MNIST 4-layer FNN** (h=512, SGD, η=0.003,
  λ ∈ {0, 0.001, 0.01, 0.03}) with Norm ∈ {BN, LN, none}: spikes
  appear **only when normalization is present** (Fig. 2); the
  no-norm control is clean at the same η, λ.
- **Controlled synthetic regression** (3-layer FNN, full-batch GD):
  PCA of trajectories shows decay steering iterates into the
  zero-norm high-curvature region; Hessian scaling validated against
  the α⁻² law on both the toy net and ResNet-50 (Fig. 5).
- **Toy 44-layer single-head transformer** ("predict the token after
  the anchor 3", d_model=400, AdamW): MLP share of the top
  eigenvector grows monotonically and jumps at spikes; killing MLP
  decay stabilizes it (Fig. 8).
- **Boundary validation** (Fig. 7): loss spikes align in time with
  `‖u‖` excursions below `c*_spike` — though via a filtering
  protocol (sub-boundary intervals <200 iters discarded, gaps <30
  iters merged), so the boundary is a good post-hoc aligner, softer
  as a forward predictor.
- **The clinching ablation** (Fig. 12): same net, decay applied
  *only* to the non-scale-invariant output layer — no pronounced
  spikes even at large η and λ, and λ_max stops growing with λ.

Fine print: single seeds, no error bars, spike results are visual;
the LLM sweep skips every practical λ between 0 and 0.5; no
interaction with LR schedules or scheduled decay is studied (AdamC /
λ ∝ η is never mentioned); no comparison against other spike
mitigations.

## What transfers to us

The `adamc_100k` watch gets its **missing failure direction** — until
now every frame said what *healthy* looks like; this paper says what
the dangerous corner looks like:

1. **New named failure mode: criticality approach.** A sustained
   per-group weight-norm *decline* is not merely the cosmetic
   "uncorrected signature" of the [Muon-SW frame](muon-sw.md) — it
   has a hard floor where training destabilizes. The full chart
   pattern to watch for: group norm trending down, that group's grad
   norm climbing (the `1/‖u‖` law), and train-loss spikes co-timed
   with the deepest norm excursions. All three series are already
   recorded; only the joint read is new.
2. **The decay-inert trap flips valence.** The
   [Hyperball page](hyperball-optimization.md) named the trap "flat
   norms at λ=1e-5 could mean decay inert, not correction working."
   This paper says decay-inert is the *safe* corner: spikes at 187M
   needed λ ∈ {0.5, 1}, and even the toy nets needed 0.01–0.03 —
   four-plus orders of magnitude above our λ=1e-5, which AdamC then
   scales *down* with the cosine. If our run ever spikes, weight-norm
   criticality is the mechanism to rule out *last*, not first —
   unless the norm chart shows an actual collapse.
3. **Corrected decay is incidentally spike-protective** (our
   synthesis, not the paper's — it never touches schedules).
   Hyperball's equilibrium `R⋆ ∝ √(η/λ)` says constant-λ runs ride
   their norms *down* during LR decay, i.e. toward this paper's
   floor; λ_t ∝ η_t pins `η/λ` and holds the norms — and hence the
   distance to criticality — flat. A fourth independent reason the
   correction we run is the right sign.
4. **Free offline probe on banked checkpoints: a distance-to-
   criticality margin.** `ρ_grad = ‖u‖²·(gᵀHg/‖g‖²)` needs one
   minibatch gradient plus one Hessian-vector product per group;
   then `c*_spike = √(η_t·ρ_grad/2)` vs the measured group norm
   gives a numeric safety margin for the endpoint readout. Cheap,
   record-only, and it directly quantifies point 2 instead of
   hand-waving it.
5. **If spikes ever appear, look at the MLP-group norms first.** The
   paper localizes spike curvature to the MLP blocks; our per-group
   recording (corrected matrices / head / no-decay) is exactly the
   diagnostic granularity it says you need. The unfrozen ViT encoder
   has LN-fed weights too — same theory, same tiny λ, same read.

## What doesn't transfer

- **The λ regime.** Every spike in the paper lives at decay values
  we will never run (0.5–1 at LLM scale, 0.01–0.03 on toys). Nothing
  quantitative is measured at λ ≤ 0.1, let alone 1e-5; the transfer
  is the *mechanism and the probe*, not any number.
- **From-scratch vs pretrained init.** Their norms start small and
  get ground down over a full pretraining run; our Molmo2 trunk
  starts at settled pretrained norms and 100k steps at λ̂ ≤ 1e-5
  moves them approximately nothing (the relaxation-timescale
  argument on the Hyperball page). We are structurally far from the
  boundary.
- **"Don't decay scale-invariant layers" as a prescription** cuts
  against the entire corrected-decay family we run, which wants
  decay *on* the hidden matrices, scheduled — and the paper's
  evidence for selective decay is exclusively from the extreme-λ
  regime. It never prices selective decay at sane λ, and never
  tests whether λ ∝ η alone dissolves the problem (its most
  conspicuous missing baseline, same gap as Hyperball's).
- **Rigor**: single seeds, no error bars, visual spike counts, and
  the boundary-validation filtering protocol means `c*_spike` is
  established as an explainer, not yet a predictor.

## Which idea/arm it fed

The `adamc-100k-live` endpoint readout — the watch gains its
**failure-side frame**: the criticality-approach pattern (norm
decline + grad climb + co-timed spikes) is the new named failure
mode, the decay-inert trap is re-tagged as the *safe* corner at our
λ, and a distance-to-criticality margin (`c*_spike` via one HVP per
group on a banked checkpoint) joins the offline-probe list beside
Hyperball's grad·norm constancy and Muon-SW's alignment cosine. No
new arm; no change to the live run. Cross-refs:
[Chou](weight-decay-correction.md), [Muon-SW](muon-sw.md),
[Hyperball](hyperball-optimization.md), the run
[parameter sheet](../posts/2026-08-09-prereg-molmo2-adamc-100k.md).
