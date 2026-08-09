# Decoupled action expert: the denoiser was never the hard part

*Read 2026-08-09 (lit slice `lit-radar-0815`, priority 3: the fjoint
seam question). Paper:
[2511.12101](https://arxiv.org/abs/2511.12101) — "Decoupled Action
Expert: Confining Task Knowledge to the Conditioning Pathway" (Zhou,
Lin, Fu, Li, Zhou, Wu — Australian Institute for Machine Learning,
University of Adelaide; v1 2025-11-15, v2 2026-03-14, preprint,
cs.RO). Read against the [attachment
decision](../posts/2026-08-09-molmo2-stage2-attachment-decision.md)
and the [fjoint pre-reg
draft](../posts/2026-08-09-prereg-fjoint-rung.md): the capacity half
of the seam question — how much has to live in the expert vs arrive
through conditioning — finally has a controlled measurement, though
on Diffusion Policy, not a VLA.*

**The paper in plain words.** A robot policy that generates arm
motions by denoising has two jobs bundled together: knowing *what*
the task wants (from camera images and instructions) and knowing
*how* to shape a physically plausible 16-step arm trajectory. This
paper argues the second job is tiny — a policy emits only 16×10=160
numbers per prediction, nothing like an image — and shows you can
split the jobs cleanly: pretrain a small generic "motion" network on
kinematics data that contains **no images and no tasks at all**
(just joint-angles-to-gripper-pose sequences), freeze it forever,
and teach each new task only to the side-channel that modulates it.
The frozen generic core plus a retrained side-channel nearly matches
full training, and a 5M-parameter MLP core matches — on their
benchmarks, beats — the standard 244M U-Net. The catch is *how* the
side-channel connects: modulation-style conditioning (scale/shift,
as in AdaLN/FiLM) survives the freeze; cross-attention conditioning
collapses to near zero, because task knowledge soaks into the
frozen weights themselves.

## What it contributes

- **A three-way factorization with the seam made explicit.** Policy
  = observation encoders (ResNet-18/50) → conditioning network
  (produces per-layer FiLM γ/β) → action backbone (the denoiser).
  Decoupled recipe: Stage 1 trains {conditioning, backbone} on
  observation-free forward-kinematics data (joint positions →
  end-effector pose sequences, extractable from any trajectory
  dataset "at negligible cost"); Stage 2 freezes the backbone and
  trains fresh encoders + conditioning for the downstream task.
- **DP-MLP: the 5M backbone.** Replaces Diffusion Policy's 244M
  CNN U-Net with L residual FiLM-MLP blocks
  (`h ← LN(h + W₂(γ⊙GELU(W₁h) + β))`), a ~51× parameter cut. Two
  separable claims ride on it: (i) *capacity* — DP-MLP beats DP-C
  even under normal end-to-end training; (ii) *decoupling* — the
  frozen-backbone version keeps nearly all of it.
- **The conditioning-mechanism law.** On one matched 8-layer
  transformer, freezing the backbone costs: cross-attention −41.5
  pts, prefix tuning −36.8, additive −11.7, FiLM −7.7, adaRMSNorm
  −3.8, AdaLN +0.3, AdaLN-Zero +0.5. Token/attention routes embed
  condition-specific projections *inside* the backbone weights and
  collapse when frozen; modulation routes keep them outside and
  don't. (They note π0.5 ships adaRMSNorm, GR00T-N1 AdaLN — the
  production experts are already on the decouplable side.)
- **The pretraining signal barely matters; pretraining does.** A
  random frozen backbone scores 0.0. Unconditional, self-conditioned
  and joint-position-conditioned Stage-1 variants land within 1.6
  pts of each other (62.2–63.8 MimicGen avg): the backbone learns
  generic trajectory structure, not task content.

## The experiments it ran

MimicGen (8 tasks, 1000 demos each) and LIBERO (4 suites, 50
demos/task, DistilBERT language via the conditioning pathway), 3
seeds, max success rate. Headlines: **LIBERO avg — DP-MLP normal
84.7 vs DP-C normal 79.3** (+10.9 on Long), and **DP-MLP decoupled
84.2** (−0.5 from its own normal); MimicGen — DP-MLP normal 65.9 vs
DP-C 63.6, decoupled 61.2 (−4.7, driven by Coffee/Stack3/Square).
DP-T (cross-attention Diffusion Policy) under decoupling: **76.4 →
5.9** LIBERO — the mechanism law in benchmark form. Cross-embodiment
Stage 1: pretraining the backbone on 76k external DROID Franka
trajectories *beats* in-distribution FK pretraining (63.8/78.3 vs
62.2/76.8) and essentially closes the gap to normal training. Fine
print: simulation only; Diffusion Policy only ("directly validating
the decoupled recipe on full VLA systems remains important future
work"); flow matching never tested; no inference-speedup or
minimum-size sweep; decoupled DP-C still trails normal by 1.4–2.5
pts before the DROID rescue.

## What transfers to us

The seam question ("how much capacity/task knowledge must live in
the expert vs arrive through the conditioning pathway") gets its
sharpest *capacity-axis* datum, and it points one way: **the
denoising function class is tiny.** Concretely for #4:

1. **The F arm is not capacity-starved, so the fjoint unfreeze
   shouldn't be justified on capacity grounds.** Our h1024×12 expert
   sits two orders of magnitude above the 5M floor this paper
   measures for the pure denoising job; whatever the frozen-trunk
   configuration lacks, it is not room in the expert. Sharpened
   prior for the [fjoint rung](../posts/2026-08-09-prereg-fjoint-rung.md):
   if J beats F2, read it as the trunk's *representations* adapting
   (task-relevant features the frozen taps don't surface), not as
   gradient relief for an overworked expert — and if J≈F2, this
   paper is the null's mechanism (task knowledge was already
   arriving fine through a trainable conditioning path).
2. **Expert sizing (inherited h512/h1536 arm) gets a direction.**
   The capacity result says smaller-with-good-conditioning is the
   live end of that dial, not larger — consistent with the ~11:1
   trunk:expert ratios on the
   [Hy-Embodied page](hy-embodied-stack.md), and cheap to check
   because sizing rungs were already priced as screens.
3. **Our expert checkpoint is task-entangled capital, not a generic
   head.** Our expert conditions by cross-attention to residual
   taps — exactly the mechanism whose weights absorb
   condition-specific structure in their ablation. The F@10k expert
   is trunk-specific and task-specific; it warm-starts the fjoint
   rung (same trunk, byte-checked by `materialize_fjoint_init.py`)
   but should not be expected to survive a trunk swap (#17) the way
   a modulation-conditioned expert might. If a reusable-expert
   ambition ever appears, the seam mechanism — not the size — is
   what to change first.
4. **Observation-free Stage 1 is the cheapest expert-init yet
   filed.** [APT](apt-expert-pretraining.md) pretrained the expert
   on vision-action pairs with language masked; this paper gets a
   working prior from kinematics alone, no images, transferring
   *across embodiments* (DROID→MimicGen/LIBERO). Radar for any
   future fresh-expert attach: a near-free FK pretraining pass is
   now a published alternative to random init — the same
   random-init damage regime APT diagnosed, attacked from below.

## What doesn't transfer

- **The freeze is on the opposite side of the seam.** They freeze
  the *action backbone* and retrain the *conditioning*; our F arm
  freezes the conditioning source (the trunk) and trains the
  expert. The result constrains the capacity split — it is not
  evidence for or against the frozen-trunk-vs-joint contrast the
  [decision memo](../posts/2026-08-09-molmo2-stage2-attachment-decision.md)
  settled, and it does not re-rank F vs K.
- **Diffusion Policy, not a VLA; DDPM-style noise prediction, not
  flow matching; ResNet/DistilBERT conditioning, not a 4B VLM
  trunk.** The authors chose DP precisely to strip the VLM out as a
  confound; the price is that nothing here measures what happens
  when the conditioning pathway is a frozen language model's
  residual streams.
- **Sim-only, and the headline is partly a capacity story.** DP-MLP
  *normal* already beats DP-C — some of "5M matches 244M" is
  "244M was oversized for these benchmarks," a MimicGen/LIBERO
  fact, not a manipulation law.
- **The cross-attention collapse is a frozen-backbone fact.** Our
  expert's cross-attention taps are fine while the expert trains
  (both our arms train it); the −41.5 only bites if we ever freeze
  the expert and expect conditioning-side retraining to steer it.

## Which idea/arm it fed

Idea #4 (`seam-screen`, decided — fjoint rung open): a capacity
prior for the rung's interpretation (J-wins ⇒ representation
adaptation, not expert relief; J≈F2 ⇒ conditioning was sufficient),
a direction for the inherited expert-width dial (down, not up), and
the task-entanglement caveat on treating the F@10k expert as
reusable capital across trunks. No gate, bar, or design change to
the pre-registered rung. Cross-refs:
[APT](apt-expert-pretraining.md) (expert init from above),
[ActionX](actionx-rl-expert-pretraining.md) (the rung's shape),
[seam-debate](seam-debate.md) /
[π0.5-KI](pi05-knowledge-insulation.md) (the gradient side of the
same seam), [VLAFlow](vla-training-objectives.md) (recipe bake-off
the capacity axis was missing from).
