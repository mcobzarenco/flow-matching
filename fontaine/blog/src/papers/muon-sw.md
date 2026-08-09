# Muon-SW: the AdamC correction, re-derived for Muon

*Read 2026-08-09 (lit slice `lit-radar-0813`, priority 1:
adamc-watch adjacency). Paper:
[2607.23777](https://arxiv.org/abs/2607.23777) — "Scale Weight Decay
and Train Better" (Anuj Apte, JPMorgan Global Technology Applied
Research, v1 2026-07-26, no venue). Third paper in the corrected-decay
family we run in production, after AdamC itself
([2506.02285](https://arxiv.org/abs/2506.02285)) and
[Chou's correction-of-the-correction](weight-decay-correction.md)
(2512.08217).*

**The paper in plain words.** When the learning rate is wound down at
the end of training, the small force that shrinks weights (weight
decay) keeps pulling at full strength — so late in training it stops
being a gentle regularizer and starts dragging the weights toward
zero, parking them somewhere no schedule ever intended. The fix this
paper argues for is the one our live run already uses: shrink the
decay in proportion to the learning rate. What's new is the *route*
to that conclusion — a classical-optimization argument (the decay
term must vanish fast enough to be summable, or it permanently
biases where training converges) rather than AdamC's
gradient-norm story — and the optimizer it's applied to: Muon, where
it buys a consistent ~0.05 validation-loss improvement, equivalent
to reaching the baseline's final loss ~22–29% faster, across four
model scales.

## What it contributes

- **A bias theorem for constant decoupled decay.** With updates
  `W ← (1−ηλ)W − ηU`, decay enters at O(η) and survives the
  Robbins–Monro summability test that the gradient term passes; the
  iterates converge to `W = −U/λ`, not to a minimizer. For Muon
  specifically, constant decay caps `‖W‖_op ≤ 1/λ`, making any
  target with larger top singular value *unreachable*. Scaling decay
  by `η_t/η_max` moves the term to O(η²) — summable, so the
  unregularized optimizer's stationarity guarantees survive.
- **Muon-SW**: one-line change, `W ← (1 − λη_t²/η_max)W − η̂_t O_t`
  over the Newton–Schulz-orthogonalized momentum. Same correction
  factor as AdamC (the paper says so explicitly); the differentiator
  is the justification — Defazio's grad-norm-growth mechanism can't
  even apply to Muon, whose orthogonalized updates have fixed
  spectral norm, yet the correction still wins. The correction is
  more general than its original motivation.
- **A quasi-steady-state norm analysis with an adiabatic boundary.**
  Under scaled decay the RMS weight-norm target is LR-*independent*
  (the η² factors cancel), so norms plateau after warmup and stay
  flat. Under constant decay the target collapses with LR — and the
  relaxation time 1/(2λη_t) eventually exceeds the remaining
  schedule, so in the last ~20% of a cosine run the norm can't even
  track its own collapsing target: it is measurably out of
  equilibrium.

## The experiments it ran

LLaMA-style MoE decoders on FineWeb at four widths (72.7M → 932.4M
total params, ~610–650 tokens/active-param, cosine to 10%, λ=0.1,
μP transfer, Muon on matrices / no-decay AdamW on embeddings, head,
and vector params). Muon vs Muon-SW only — **no AdamW, AdamC, or
zero-decay arms**:

| Width | Muon | Muon-SW | Step speedup |
|---|---|---|---|
| 256 | 3.350 | 3.297 | 21.7% |
| 512 | 2.971 | 2.915 | 27.5% |
| 768 | 2.799 | 2.754 | 26.5% |
| 1024 | 2.675 | 2.630 | 29.4% |

Width-1024 norm trajectories: constant decay peaks ~0.19 then falls
~60% by the end; scaled decay plateaus ~0.16–0.20 and stays flat.
The quasi-steady prediction tracks the measured norm until ~300k of
376k steps, then the adiabatic approximation breaks. Measured
alignment law: update–weight cosine `a_t ≈ −kη_t` (k≈20.8) under
scaled decay vs a persistent offset a₀≈7.4e-3 under constant decay.
No gradient-norm figures anywhere. Effectively no ablations (single
λ, single schedule shape).

## What transfers to us

The live `adamc_100k` run gets its **second interpretive frame, and
this one is about weight norms, not grad norms**:

1. **The expected signature sharpens.** Chou's page banked "flat
   norms, ~nil loss effect" for Adam-family; Muon-SW adds the
   mechanism-level version: under corrected decay the norm target is
   LR-independent, so per-group weight norms should sit on a
   post-warmup plateau essentially from early training — and their
   adiabatic analysis says corrected-decay norms freeze onto that
   plateau *earlier* than uncorrected ones would decay off theirs.
   For the endpoint chart: **plateau-then-flat = the correction
   working; peak-then-long-decline = the uncorrected signature we
   should not see.** This partially blunts Chou's
   "AdamC never reaches steady state at 300 epochs" caveat: by this
   paper's account the *scaled-decay* steady state is reached
   quickly because the target doesn't move; what Chou measured
   not-converging is a different, stricter notion. Both caveats ride
   the chart note.
2. **A loss-side prior with a sign, unlike Chou's wash.** Chou had
   AdamC ≈ AdamW on ViT (76.98 vs 76.92); here corrected decay buys
   a consistent ~0.05 val-loss / ~25% step-equivalent gain — but on
   Muon, whose constant-decay pathology (hard `1/λ` norm cap on
   orthonormal updates) is *structurally worse* than Adam's. Read
   onto our run: the honest expectation stays "stability, maybe a
   small loss edge," not the 25% headline.
3. **The λ ∝ η conclusion now has three independent derivations**
   (grad-norm dynamics, steady-state independence, Robbins–Monro
   summability) landing on the same rule our optimizer implements —
   about as corroborated as a one-line correction gets. And our 10%
   LR floor again sits on the right side: their theorems technically
   require the schedule to keep shrinking, but their own experiments
   run cosine-to-10% like ours.
4. **A free probe if the endpoint chart looks odd**: the
   update–weight alignment cosine `a_t` — their fitted
   `a_t ≈ −kη_t` (no offset) is what corrected decay should produce;
   a persistent negative offset is the uncorrected signature. Our
   checkpoints + banked optimizer state can compute this offline if
   the norm chart ever needs a second opinion. Numbers to compare
   against: k≈20.8, a₀≈7.4e-3.

## What doesn't transfer

- **Muon itself** — same verdict as ScionC on the
  [Chou page](weight-decay-correction.md): switching optimizer
  families is a from-scratch trunk ablation the startup-velocity
  rule exists to block. Radar-only.
- **The headline speedups** are Muon-specific (the spectral-norm cap
  is a Muon pathology; Adam has no such hard ceiling) and come with
  no AdamW/AdamC control arm, single λ, single schedule, MoE-only.
- **Its decay partition** (embeddings + head + vectors wholly
  undecayed under AdamW) is inherited convention, not evidence — it
  neither supports nor threatens our audited corrected/head/no-decay
  partition.
- The theorems formally cover decay-to-zero schedules; cosine-to-10%
  is outside their hypotheses (bridged only by the quasi-steady
  analysis). Quote the norm-plateau prediction, not the convergence
  theorems.

## Which idea/arm it fed

The `adamc-100k-live` endpoint readout, alongside the
[Chou frame](weight-decay-correction.md): the weight-norm chart gets
its expected shape (plateau-then-flat vs peak-then-decline), the
grad-norm chart keeps Chou's framing, and the alignment-cosine probe
is banked as the free second opinion. No new arm; no change to the
live run. Cross-refs: the run pre-reg
([parameter sheet](../posts/2026-08-09-prereg-molmo2-adamc-100k.md)),
`bijou/train.py`'s AdamC partition.
