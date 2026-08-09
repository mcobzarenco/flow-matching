# RLDT: RL on flow policies as density transport — the cleanest-gradients entry in the RL pole

*Read 2026-08-09 (lit slice `lit-radar-0811`, priority 2: the
post-SFT menu, #16). Paper:
[2606.08602](https://arxiv.org/abs/2606.08602) — "Reinforcement
Learning for Flow-Matching Policies with Density Transport" (Lei,
Daniilidis, Loquercio;
[project page](https://rpfey.github.io/rldt/)).*

**The paper in plain words.** Once you have a robot policy trained by
imitation, the obvious next step is reinforcement learning: let the
robot practice and push its behavior toward what actually earns
reward. For flow-matching policies this is awkward — the action is
produced by many small denoising steps, and standard RL wants a
probability for the final action, which the multi-step process
doesn't hand you. Most existing fixes either approximate that
probability (biased) or backpropagate through the whole denoising
chain (unstable, and in practice the early steps stop learning).
This paper sidesteps both: it treats improvement as literally
*moving the cloud of candidate actions* toward high-reward regions —
a transport problem, the thing flow matching is natively good at. A
kernel-based update (Stein variational gradient descent) computes,
from a handful of sampled actions, which direction each should move —
an attraction toward higher critic value plus a repulsion keeping the
samples spread out — and the velocity field is trained to follow that
direction at every denoising step. A trick they call expected-target
estimation converts any intermediate noisy action into its expected
final action in one step, so every denoising depth gets a clean,
equally-sized gradient without backprop-through-time. The policy
stays an ordinary flow model — same sampler, same steps — just with
its density steadily transported toward reward.

## What it contributes

- **The transport framing**: max-entropy RL's optimal policy
  ∝ exp(Q/κ) is approximated by SVGD particles (K=8 per state, RBF
  kernel), giving a per-sample transport direction φ* = attraction
  (kernel-weighted Q-gradient) + repulsion (kernel gradient, prevents
  collapse). The actor loss aligns v_θ with φ* at a random denoising
  time, on the *expected-target* point a† = a_τ + (1−τ)·v(a_τ,τ) —
  the one-step endpoint estimate, so the direction is always
  evaluated on the action manifold. No policy log-likelihoods
  anywhere.
- **Well-conditioned gradients across denoising depth** (their Fig.
  4): uniform per-step gradient contribution through training, vs a
  backprop-through-time baseline where early steps contribute
  <0.001. This is the paper's sharpest empirical differentiator.
- **Regularization**: a consistency term (straight paths) + a Fisher
  divergence to the pretrained velocity field (stay near the SFT
  policy). Double-Q critic, standard TD.

## The experiments it ran

- Gym locomotion (dense), FurnitureBench (sparse long-horizon, 1,000
  parallel envs, ~500k steps, 48 GPU-h), Robomimic vision-based
  Square/Transport (64 envs, ~12.8k steps, 30 GPU-h). Policies are
  small (MLPs to ViT+MLP), always warm-started from BC on demos —
  no VLA-scale model anywhere.
- Vs DPPO / ReinFlow / FPO++ / QAM: ~2× on HalfCheetah, Robomimic
  Square ~90% vs DPPO ~60%, FurnitureBench Lamp ~70% vs ~30%.
  Ablations: RBF kernel needed on sparse reward (delta-kernel
  variant destabilizes — the repulsion term is doing exploration,
  not just diversity); insensitive to particle count K=4–16 and
  temperature.

## What transfers to us

1. **A third, mechanistically distinct entry for the #16 RL pole.**
   The post-SFT menu now holds: preference RL from intervention
   pairs (FlowPRO, off-policy-ish, human-labeled), GRPO on flow-SDE
   rollouts (Z-1, on-policy, diagnostic-gated), and now
   SVGD-transport (RLDT, on-policy, critic-based, no likelihoods).
   RLDT's differentiators — unbiased w.r.t. the flow structure,
   per-depth gradient conditioning, multimodality preserved by
   construction (repulsion) — make it the theoretically cleanest of
   the three, and the only one whose update is *native* to flow
   matching rather than adapted from LLM RL.
2. **Expected-target estimation is a pattern we already trust.** Its
   a† = a_τ + (1−τ)v is exactly the 1-NFE endpoint estimate our
   ForesightFlow read benchmarked (Kendall τ 0.80–0.86 vs full
   integration) — a second independent use of "the one-step preview
   is good enough to steer by," here for gradients rather than
   ranking.
3. **The requirements table is the real payload for planning:**
   parallel simulated rollouts (64–1,000 envs), a trained critic,
   ~30–48 GPU-h per task at *small* policy scale. That prices the
   RL pole honestly for the rig bench: without a simulator of the
   owner rig or massive parallel hardware practice, every method in
   this family is sim-first. Sample counts (~12.8k steps for
   vision-based Robomimic) are not the blocker; the parallel-env
   infrastructure is.

## What doesn't transfer

- **No VLA-scale evidence.** Largest policy is a ViT+MLP; nothing
  says the SVGD particle geometry or the Fisher constraint behave at
  4B-trunk + flow-expert scale, and K=8 forward passes per gradient
  step is real money there.
- **Multimodality preservation is asserted-by-mechanism, not
  measured** — no mode-coverage metric anywhere; the repulsion
  term's benefit shows up only indirectly (sparse-reward
  stability).
- **Online-only.** Nothing here helps the offline setting we
  actually occupy today; this is a bank-for-later entry, alive iff
  the #16 rig bench grows a practice loop (sim or hardware).

## Which idea/arm it fed

**#16 (rig transfer benchmark)**: RL-pole roster entry #3 with an
honest infrastructure price; no arm, no pre-reg — the pole stays
parked until a rollout loop exists. Cross-refs: the
[Hy-Embodied stack page](hy-embodied-stack.md) (FlowPRO), the
[Z-1 page](z1-selective-joint-rl.md) (GRPO, diagnostic-gated), the
[ForesightFlow page](foresightflow-self-scored-bestofk.md) (the same
1-NFE endpoint estimate, used for selection instead of gradients).
