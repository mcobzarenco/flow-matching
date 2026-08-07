# APT — the seam damage is an initialization problem

**Source:** APT: Action Expert Pretraining Improves Instruction
Generalization of Vision-Language-Action Policies
([2606.12366](https://arxiv.org/abs/2606.12366), June 2026). Read
2026-08-07, the pre-endpoint attachment-frontier slice — the last
scheduled look at the seam literature before the molmo2 stage-2
attachment decision opens (~08-08). **Fed:** #4 (a third
explanation of *why* joint training hurts, one new named escalation
rung — F-then-joint — and a reframe of what our F arm's checkpoint
is worth).

## The theme

Every seam recipe we have banked answers "how do I stop the action
expert from wrecking the trunk?" — π0.5/KI stops the gradients,
AEGIS projects them, Anchor-Align leashes the trunk to its frozen
self, Wall-OSS bridges them. APT asks a prior question: **why are
the expert's gradients destructive in the first place?** Its
answer: because the expert is *randomly initialized*. VLA
trajectories pair thousands of vision-action frames with a single
instruction, so a fresh expert learns the imbalanced shortcut —
act from vision, ignore language — and its noisy early gradients
push the trunk the same way. On that diagnosis, the fix is not a
gradient guard at all: **pretrain the expert first, then decide
what to unfreeze.**

## What the paper does

A Bayesian factorization π(a|v,ℓ) ∝ πᵖ(a|v) · L(ℓ|v,a) turned into
a two-stage schedule:

1. **Stage 1 — vision-action prior.** Train a diffusion action
   expert on vision-action pairs with the language **masked out
   entirely** and the VLM backbone (Qwen3-VL) **frozen**. Only half
   the expert's layers (N/2) exist at this stage. Vision-action
   pairs alone carry no language imbalance, so there is no shortcut
   to learn.
2. **Stage 2 — language alignment.** Expand to N layers
   (interleaved insertion, Stage-1 layers keep their weights),
   unmask language, and jointly finetune on the full dataset —
   optionally including the VLM backbone itself ("Ft VLM").
   Backbone features enter through **layer-wise gated fusion**
   (sigmoid-gated residual injection of intermediate VLM features)
   rather than raw token insertion, explicitly to avoid stomping
   the Stage-1 prior.

## What they ran

LIBERO-PRO (instruction/object perturbations — the benchmark where
naive joint training shows its collapse: OpenVLA and π0 score **0%
across the board**), a four-suite pick-place generalization grid
(seen/unseen objects, containers, environments), architecture
transfer (π-style all-layer attention and GR00T-style final-layer
cross-attention), and real-robot pick-place + compositional
chaining. Headlines: **27% avg on LIBERO-PRO for APT with a
finetuned VLM vs 11% for π0.5's KI recipe**; real-world APT 28/40
on the hardest suite vs π0.5's 16/40; π0.5 "nearly collapses" on
chained instructions while APT executes them without segmentation.

The decision-relevant table is the pick-place ablation grid
(SO/UO/UC/UOUE):

| recipe | KI | expert pretrain | joint VLM | scores |
|---|---|---|---|---|
| π0 (naive joint) | | | ✓ | 42 / 30 / 26 / 16 |
| π0.5 (KI + co-training) | ✓ | | | 84 / 70 / 86 / 50 |
| KI only | ✓ | | | 88 / 56 / 66 / 34 |
| expert pretrain only (frozen trunk) | | ✓ | | 90 / 58 / 40 / 40 |
| KI + expert pretrain | ✓ | ✓ | | **96 / 74 / 90 / 62** |
| expert pretrain + joint VLM, **no KI** | | ✓ | ✓ | **98 / 84 / 92 / 58** |

Two readings, both load-bearing: with a **random-init** expert,
gradient guards (KI) are what stands between you and π0's numbers;
with a **pretrained** expert, unfreezing everything *without any
gradient stopping* is the best row on 3 of 4 suites — "stop-gradient
is not a necessary condition." The damage was never joint training
per se; it was joint training *from noise*.

## What transfers to us

Our #4 screen attaches a **randomly initialized** flow expert in
both arms. APT's grid speaks directly to that regime:

- **The K arm's design is corroborated from a new direction.** Our
  K is phase-1 CE verbatim + stop-grad seam — the KI recipe. APT
  finds KI most valuable exactly when the expert starts from noise
  (its KI-less random-init rows are the collapsed ones). Nothing in
  APT argues for loosening the live screen's seam.
- **The F arm's endpoint is APT's Stage 1, almost verbatim.** A
  flow expert trained to convergence against a hard-frozen trunk
  *is* a pretrained vision-action expert (ours sees language
  through the trunk's representations rather than masked out — a
  softer prior, same structure). APT's grid then names the follow-up
  we did not have: **F-then-joint** — warm-start a joint run
  (unfrozen trunk, possibly no stop-grad at all) from the F
  checkpoint's expert instead of from noise. In APT's numbers that
  jump is worth +8 to +26 points over the frozen row. This lands on
  the #4 escalation map as a named rung, *after* the screen reads
  out — the pre-reg's two arms are untouched.
- **It reframes a possible F≈K tie.** If Δ_seam comes back small,
  the KI-vs-frozen contrast may simply be measuring two guards that
  both work at our scale — APT predicts the interesting contrast is
  then *initialization* (F-then-joint vs K-from-noise at matched
  total steps), not the seam itself.

## What doesn't transfer

- **The measured axis is language generalization** (unseen
  instructions/objects/compositions). Our panel is single-embodiment
  action MAE with no language-perturbation axis; APT's largest
  effects live exactly where our instrument doesn't look. The
  transferable content is the mechanism and the recipe ordering,
  not the effect sizes.
- **Architecture gap:** their expert is a diffusion transformer fed
  by sigmoid-gated per-layer fusion from Qwen3-VL; ours is a
  residual-tap flow expert reading 3 exported streams. Their
  token-insertion-vs-gated-fusion ablation (gated wins, biggest on
  unseen suites) is a caution for any future all-layer-reads arm
  (#4 external arm 1), not evidence about our current taps.
- **The 0% baselines are perturbation-benchmark artifacts** —
  LIBERO-PRO is built to break visual-shortcut policies; π0 scoring
  0 there does not mean π0-class recipes score 0 anywhere else (their
  own pick-place grid shows π0 at 42/30/26/16).

## Verdict for the stage-2 decision

Nothing here re-ranks K-vs-F before the screen — it runs as
pre-registered. What APT changes is the escalation map: the F
checkpoint is no longer just the conservative arm's readout, it is
**free Stage-1 capital**, and the strongest published recipe in its
weight class (expert pretrain → unfreeze, no KI) is one warm-start
run away from it. Radar note for the next slice: the initialization
thread has siblings we have not read
([2605.25802](https://arxiv.org/abs/2605.25802), VLM representation
for VLA init; [2601.03309](https://arxiv.org/abs/2601.03309),
VLM4VLA) — banked as hooks, not read.
