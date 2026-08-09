# Weibull weight-scale: reading the forces on our norms out of banked checkpoints

*Read 2026-08-09 (lit slice `lit-radar-0815`, priority 2: adamc
checkpoint-analysis frame). Paper:
[2606.19367](https://arxiv.org/abs/2606.19367) — "Weibull
Weight-Scale Parameter Evolution under AdamW Training Dynamics"
(Tiexin Ding — single author, no affiliation listed; v1 2026-06-11,
21pp/14 figs, no venue). Fifth paper in the corrected-decay reading
thread after AdamC itself ([2506.02285](https://arxiv.org/abs/2506.02285)),
[Chou](weight-decay-correction.md), [Muon-SW](muon-sw.md) and
[Hyperball](hyperball-optimization.md) — and the first that is about
*measuring* the norm dynamics offline rather than changing the
optimizer.*

**The paper in plain words.** As a transformer trains under AdamW,
the typical size of its weights rises, overshoots, and settles back
down. This paper asks *why*, and answers by splitting each step's
effect on the total weight size into three competing forces: an
alignment force (does the optimizer's step push the weights outward
or inward?), an injection force (the raw kick from the step size
itself), and the familiar decay force (weight decay pulling
everything toward zero). On small models trained with full optimizer
internals recorded, the alignment force turns out to do almost all
the work — 88–94% of the total force budget during the growth phase
— and the growth stops exactly when decay finally grows large enough
to balance it. The practical gift for everyone else: a
spline-interpolation trick that recovers the dominant alignment
force from ordinary saved checkpoints, no optimizer internals
needed, at 92–94% accuracy — roughly twice what naive differencing
of checkpoints achieves.

## What it contributes

- **A three-force decomposition of the squared weight norm.** From
  the AdamW update, to leading order per matrix:
  `Δ‖W‖² = −2η⟨W,û⟩ + η²‖û‖² − 2ηλ_wd‖W‖²` with
  `û = m̂/(√v̂+ε)` the adaptive update direction. The terms are the
  **alignment force** (positive when the step pushes weights
  outward), the **injection force** (always positive, ~4% of the
  budget throughout), and the **decay force** (always negative,
  proportional to the current scale). Rise phase: alignment
  contributes 88–94% of the absolute budget. Saturation: alignment
  and decay approach balance — that balance *is* the
  growth-to-relaxation transition. Higher-order terms are <0.001%.
- **A k-lock bridge to a distributional read.** Element-wise |W|
  fits a two-parameter Weibull whose shape k≈1.20 stays locked from
  init, so the scale parameter λ(t) is the RMS trajectory up to a
  fixed factor (`σ = λ√Γ(1+2/k)`); force dynamics on ‖W‖² therefore
  govern λ(t) directly. Closed-loop reconstruction (forces → σ² →
  λ) is good to ~5–6% where checkpoints are dense (bridge ~4.6% +
  integration ~1.9%).
- **The spline displacement method — the reason we read this
  paper.** Exact identity for consecutive steps:
  `û_t = −(W_{t+1}−W_t)/η_t − λ_wd·W_t` — the update direction is
  algebraic in the weights, no moments needed. With only sparse
  checkpoints, fit a cubic spline through the saved weight
  trajectory, evaluate at unit-step resolution, plug into the
  identity. Validated by subsampling ground-truth runs: **92–94%
  recovery of the alignment force at checkpoint spacings of
  250/500/1000 steps (on a 20k-step run), vs 41–51% for the naive
  two-point finite difference** — and accuracy did not degrade
  across that spacing range. Requires knowing (η_t, λ_wd) per step,
  i.e. the schedule — which we have analytically.

## The experiments it ran

Self-trained Pythia-70M (GPT-NeoX) on wikitext-103, 20k steps,
lr 1e-3 (warmup 200, cosine to 0.1×), λ_wd=0.01, checkpoints +
optimizer moments every 250 steps, four seeds; forces computed
per-tensor and aggregated over O+FFN ("Transmission") layers.
Findings: rise-phase alignment share 88–94% across seeds, robust to
a Llama-style architecture swap (88–93.4%), an LR sweep
3e-4…3e-3 (peak λ scales ~η^0.78), and super-weight removal (top
0.1% by |W·û| shifts the share 0.4 pp). Real published Pythia
70M–1B checkpoints show the same rise–overshoot–relax λ(t)
phenomenology per layer; where their public checkpoint gaps blow
out to 20k–43k steps, pointwise trajectory reconstruction degrades
to 15–24% error — an integration/phase-shift artifact correlated
with gap size (r=0.86), not a mechanism failure. Exploratory
observation, explicitly flagged as such: peak λ is ~2× higher on
single-domain data (wikitext ~0.069–0.076) than multi-domain (Pile
~0.035), and continuing a Pile-trained Pythia on wikitext drives λ
0.023 → 0.060.

## What transfers to us

This is the analysis frame for the **banked 5k-step saves** of
`adamc_100k` — it turns the checkpoint shelf into a force
chronicle, and it is the first *quantitative* disambiguator for the
decay-inert trap named on the
[Hyperball page](hyperball-optimization.md):

1. **Two of the three forces are computable from our saves; the
   third is a small residual.** Per matrix: **decay force
   `−2η_tλ_t‖W‖²` needs no recovery at all** — checkpoint norms
   plus the schedule, and our AdamC `λ_t = λ·η_t/η_max` is known
   analytically, so the displacement identity's time-varying-λ
   requirement is satisfied for free. **Alignment force comes via
   the spline route** through the ~20 saves. Injection is not
   recoverable from weights alone — but it was ~4% of the budget in
   every ground-truth run, so book it as the residual.
2. **The decay-inert trap becomes a measured number.** Flat norms
   admit two stories: decay inert (nothing pulling either way) vs
   correction working (alignment and decay in balance). The force
   decomposition separates them: compute the per-matrix ratio
   `|F_decay|/|F_align|` across the run. **Ratio ≪ 1 with sizable
   alignment ⇒ decay is inert at our λ=1e-5 and flat norms are
   alignment's doing; ratio → O(1) into the cosine tail ⇒ the
   AdamC balance is real.** This is the concrete probe the trap
   note has been waiting for, and it rides the endpoint readout
   next to the existing ‖∇L‖·‖W‖, stable-rank, and
   alignment-cosine probes.
3. **It subsumes the Muon-SW alignment-cosine probe's data
   requirement.** [Muon-SW](muon-sw.md)'s `a_t ≈ −kη_t` probe
   wanted banked optimizer state; the spline-recovered ⟨W,û⟩ is
   the same inner product from weights-only saves. (If any of our
   saves *do* carry Adam moments, the identity gives exact û there
   — use those as ground-truth anchors for the spline estimate,
   exactly the paper's validation design.)
4. **Cost: near-free.** Stream one matrix at a time across the ~20
   snapshots, cubic spline along time, inner products — CPU-only,
   minutes per matrix, no GPU hours, no training-loop change.
   Record-only, chart-ready for the endpoint report.
5. **The relative sampling density matches the validated regime —
   with one honest asterisk.** Their best-validated spacing,
   S=1000 of 20k steps, is 5% of the run with 20 knots; ours is
   S=5000 of 100k — the same 5% and the same ~20 knots. But 5000
   *absolute* steps is 5× beyond their tested spacings, and their
   real-Pythia result says what degrades with big gaps is
   trajectory integration, not the force read — so quote recovered
   forces, don't forward-integrate norms across our gaps. Fine
   print: with warmup ending at step 1000 and our first save at
   5000, the rise phase is under-resolved at the start; treat
   the first spline segment as soft.

## What doesn't transfer

- **The Weibull machinery itself.** The k-lock is an empirical fact
  of their 70M models (and explicitly fails for Q/K projections,
  k∈[0.28,0.51]); the paper's own fallback — work directly in
  RMS/squared-norm units, where the force mechanism is general —
  is what our watch already does. We take the forces, skip the
  distributional fits.
- **Scale and regime.** Direct force measurement is 70M-scale,
  random init, from-scratch LM. Our 4B pretrained trunk on a
  manipulation corpus starts near their *continuation*
  intervention, not their rise phase — expect the balance
  structure, don't expect their phase timings or 88–94% share to
  hold numerically.
- **The data-coherence finding** (peak λ vs corpus mix) is
  self-labeled exploratory, two corpora, follow-up promised.
  Directionally interesting for us — a highly coherent fine-tune
  corpus predicts alignment-driven norm *growth*, so rising norms
  on our watch would not by themselves indicate decay failure —
  but it's a note, not a frame.
- **Provenance caveat**: single-author preprint, no venue, no
  affiliation listed; the code release
  ([NPM-Weibull-public](https://github.com/tiexinding/NPM-Weibull-public))
  partially offsets. Weight the method (checkable on our own data)
  over the phenomenology claims.

## Which idea/arm it fed

The `adamc-100k-live` endpoint readout gains its fourth offline
probe and the decay-inert trap gets its quantitative test: the
**per-matrix force chronicle** (spline-recovered alignment + exact
decay + injection-as-residual) over the banked 5k saves, with
`|F_decay|/|F_align|` as the inert-vs-balanced verdict number. No
new arm; no change to the live run; no new data to save. Cross-refs:
[Chou](weight-decay-correction.md), [Muon-SW](muon-sw.md),
[Hyperball](hyperball-optimization.md), the run
[parameter sheet](../posts/2026-08-09-prereg-molmo2-adamc-100k.md).
