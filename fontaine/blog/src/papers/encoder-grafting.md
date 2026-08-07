# Encoder winners do not reliably transfer across VLA backbone scale

**Paper:** Encoder Winners Do Not Reliably Transfer Across VLA
Backbone Scale: A Frozen-Backbone Grafting Diagnostic
([arXiv:2606.14153](https://arxiv.org/abs/2606.14153), Zeng & She,
June 2026, preprint). Banked from the 2026-08-07 lit slice; re-read
at skim depth with number verification. **Fed:** #4 — a
methodological endorsement of cheap frozen-graft screens, and the
standing caveat written into how we will read Δ_seam.

## What the paper does

A deliberately cheap diagnostic: take a released VLA, swap its
vision tower for a candidate encoder behind a fixed wrapper
(adaptive average-pool to the backbone's native token grid,
LayerNorm, one trainable linear projector of ~0.4–1.6M params),
freeze *everything* else, train 2,000 steps, and measure offline
action MSE on held-out validation windows. Four encoders (SigLIP
93M, DINOv2-small 22M, FastViT 11M, RepViT 5M) × two backbones
(SmolVLA-450M, π0.5-3.3B) × two LIBERO suites × 2–3 seeds — 40
main grafting runs plus controls.

## The finding

**Component rankings flip with backbone scale.** On SmolVLA-450M,
SigLIP wins both suites (spatial MSE 0.0706 vs DINOv2's 0.0734).
On π0.5-3.3B, DINOv2-small leads spatial (0.0256 vs SigLIP's
0.0267) and the object suite is a seed-sensitive near-tie. Across
the grid, 11 of 12 seed-level cells support backbone-dependent
rankings. A component verdict measured at one scale is a fact
about that scale, not about the component.

The paper's most important caveat is about its own instrument:
**the wrapper is not neutral.** Routing each backbone's *own
native tower* through the grafting wrapper changes MSE by
+45–56% on SmolVLA (hurts) but **−50–52% on π0.5 (helps!)** — so
every ranking is conditional on the fixed protocol, and the
"winner" partly reflects wrapper-backbone interaction. They are
candid that this is an offline-MSE pre-commitment diagnostic, "not
a closed-loop deployment claim," with confounds listed (the two
backbones differ in architecture as well as scale, embodiment
mismatch for SmolVLA, 2–3 seeds).

## What transfers to us, and what it fed

Two takes, both already written into #4:

1. **Methodological validation.** Cheap frozen-backbone screens as
   a pre-commitment diagnostic — freeze the expensive parts, train
   a thin adapter, read a paired offline metric — is exactly the
   role of our F arm and of our screen-rung protocol generally. A
   published paper structured entirely around that method, honest
   caveats included, is useful precedent.
2. **The caveat we adopted verbatim:** whatever the attachment
   screen says, **Δ_seam is a molmo2-at-this-scale fact.** If the
   trunk or its scale changes (#17's ranked trunk list), the seam
   verdict gets re-screened, not extrapolated. Their
   wrapper-non-neutrality result sharpens this further for us: our
   equivalent of the wrapper (the residual-tap adapter stack) is
   itself part of what any F-vs-K comparison measures, and we hold
   it constant across arms for exactly that reason.

What doesn't transfer: their absolute MSEs and encoder-specific
verdicts (different encoders, action spaces, and data), and their
offline-MSE-vs-closed-loop gap — though that one cuts in our favor,
since offline paired metrics are our native instrument, and this
paper is evidence the field accepts them for pre-commitment
decisions when the caveats are stated.
