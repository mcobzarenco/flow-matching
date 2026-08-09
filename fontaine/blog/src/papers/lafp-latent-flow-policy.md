# LAFP: flow matching instead of BC in a latent-action space

*Lit slice 2026-08-09 (work session 15:5xZ, last of the banked radar
hooks — its pair 2606.23420 turned out ALREADY covered by the
[LAFM page](latent-action-priors.md), caught by dup-check before
writing). LAFP ([2606.10517](https://arxiv.org/abs/2606.10517),
"Preserving Latent Action Structure in Latent Policy Learning via
Flow Matching"). Skim-to-place: fed #17 (the
latent-action-from-video pretraining family map) and one design
data point for flow heads in high-dimensional spaces.*

## The paper in plain words

When you have mountains of *unlabeled* video but few
action-labeled trajectories, one recipe is: learn a "latent action"
between consecutive frames (an inverse-dynamics encoder + a
forward-dynamics reconstructor, vector-quantized into a codebook),
train a policy that predicts latent actions from observations, and
finally train a small decoder from latent to real actions on the
labeled slice. LAFP's point: if that policy is trained by behavior
cloning it averages away multimodal behavior; train it with **flow
matching in the latent-action space** instead and diversity
survives. One catch appears — a stochastic policy breaks the
one-to-one latent↔action pairing the decoder training relied on —
fixed by constraining the decoder's training samples to interpolate
toward the ground-truth latent (pulling generations near the true
pairing while the flow stays frozen).

## What they ran, and the caveat that bounds it

**Procgen video games, not robots**: 16 procedural environments,
~2.6M frames each from PPO experts, 10% action-labeled. vs the LAOM
baseline: +8.1 pts average success (+23% relative), with the big
wins exactly in multimodal environments (Miner 36→87). Three
findings worth keeping: (1) predicting the latent target directly
beat predicting the vector field, and v-prediction degraded severely
at 256-dim latents; (2) 3 inference steps sufficed; (3) fine-tuning
the pretrained latent model *hurt* — "post-training fine-tuning can
disrupt the latent structure preserved by flow matching."

## What transfers to us

- **Family map (#17).** With LAFM (learned prior libraries) and the
  latent-action-priors thread, this fills in the
  "flow-in-latent-action-space" pole: LAFM restructures the *prior*
  of an action-space flow; LAFP moves the *whole policy* into the
  latent space and decodes after. Both argue the same premise —
  isotropic-Gaussian-to-action transport is wasteful when behavior
  is clustered. If video-pretraining ever enters our program (the
  RDT2/VISTA data premise), LAOM+flow is the documented recipe.
- **A design data point, not a directive**: x-prediction more stable
  than v-prediction for high-dim flow targets. Our expert predicts
  velocity over 50×6 action chunks and is healthy — but if a future
  head works in a learned latent (or a DCT space, per
  [FAFM](frequency-aware-flow-matching.md)), their 256-dim
  v-prediction instability is worth remembering.
- **The freeze lesson rhymes**: their fine-tuning-disrupts-structure
  result is the latent-space cousin of our frozen-trunk findings —
  structure learned under one objective degrades when a second
  objective trains through it.

## What doesn't transfer

Game environments, discrete-ish controls, PPO-expert data — none of
the effect sizes carry. No chunking, no real-time constraint, no
manipulation. This is a family-map entry, not an arm candidate.

## Verdict

Placed, not actioned: the latent-action pretraining pole now has its
flow-matching member, and the radar set from 08-09 is fully cleared
(FAFM, VISTA, LAFP read; Flowing With Purpose already covered).
