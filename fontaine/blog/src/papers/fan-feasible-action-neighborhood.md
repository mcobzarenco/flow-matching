# FAN: a feasible-action-neighborhood prior — physical tolerance as a token-head regularizer

*Read 2026-08-09 (lit slice `lit-radar-0811`, priority 2: the
post-SFT menu, #16). Paper:
[2604.01570](https://arxiv.org/abs/2604.01570) — "Feasible Action
Neighborhood prior for VLA finetuning" (Niu et al., CVPR 2026,
cs.RO).*

**The paper in plain words.** When a robot picks something up, there
is never exactly one correct motor command — a millimeter left or
right, a touch faster or slower, and the grasp still works. But a
VLA model with a discrete-token action head is trained as if exactly
one token were right and every neighbor equally wrong. This paper
adds a nudge to fine-tuning that encodes the physical reality:
wherever the model currently puts its most confident action, the
predicted distribution should look like a smooth bump *around* that
action rather than a spike on it or scattered mass far away. The
nudge is a KL penalty toward a Gaussian centered on the model's own
argmax — self-referential, not copied from the demonstration — with
a width that either tracks the model's own uncertainty (during
supervised fine-tuning) or is a fixed dial (during RL fine-tuning).
The claimed effect is better sample-efficiency and, mainly, better
robustness out of distribution: a smoothly-covered neighborhood
degrades gracefully when the scene shifts, where a spike misses
outright.

## What it contributes

- **The regularizer**: ℒ_FAN = KL(π(·|s) ‖ N(μ(s), Σ(s))) with
  μ(s) = argmax π — the prior chases the policy's own peak.
  SFT: Σ adaptive (the policy's own variance); RFT/PPO: Σ = σ²I
  fixed (σ ≈ 0.2–0.3). Weight α 0.01–0.05, tuned per benchmark.
  Discrete tokenized heads only (OpenVLA 7-DoF bins, OpenVLA-OFT
  8-step chunks), LoRA r=32.
- **Numbers**: ManiSkill SFT 78.1→89.8% ID, 58.1→63.3% OOD; PPO
  +1.5 ID / +6.2 OOD; LIBERO-Spatial +2.5 (OpenVLA) / +3.6 (OFT).
  RFT reaches 90% success in ~⅓ the training steps of baseline PPO.
  Real JAKA arm: biggest win on the high-perturbation task (1/30 →
  7/30). The consistent pattern: modest ID gains, the OOD/perturbed
  column is where it earns its keep.
- Explicitly framed as *not* entropy maximization — the mass is
  concentrated in a physically-motivated neighborhood, not spread
  everywhere.

## What transfers to us

1. **A cheap-SFT-lever entry for the #16 menu** — notable as the
   only entry in the post-SFT roster that needs *no rollouts, no
   critic, no preference labels*: one extra loss term at SFT time.
   For any future rig fine-tune of the AR trunk, it is the lowest-
   infrastructure candidate on the list, and its claimed strength
   (perturbation robustness) is exactly the rig-transfer failure
   mode.
2. **A #19 adjacency worth one sentence on the record.** Our
   sampled-draws programme measured the AR head's distribution
   shape from the *decode* side (mean-collapse: sampling loses
   −0.145/−0.154 to greedy, monotone in T). FAN shapes the same
   object from the *training* side — deliberately unimodal-smooth
   around the peak. A FAN-trained head plausibly *widens* the
   greedy-vs-sampled gap (more mass adjacent to the mode), which
   would strengthen, not threaten, our family-decode default. No
   read needed; noted on the #19 ledger as an external prior.
3. **The self-referential prior is the interesting design choice** —
   centering on the model's argmax rather than the demo action
   makes it a smoothing operator, not extra supervision; it cannot
   inject new information, only redistribute confidence. That is
   why the honest read of their table is "regularization with a
   physical story," and why the OOD column benefits most.

## What doesn't transfer

- **Head mismatch for the flow side.** Defined on discrete token
  distributions; our flow expert has no per-token distribution to
  smooth (and FAFM already occupies the smooth-the-continuous-head
  slot). Only the AR trunk qualifies.
- **α is benchmark-tuned** (0.01–0.05, no transfer rule given), and
  the authors show no failure modes — bimodal states (two valid
  grasps) are exactly where a forced-unimodal prior should hurt,
  and they don't test it. Our corpus is multi-repo, multi-scene;
  unimodality per state is a stronger assumption for us than for
  their single-task cells.
- **7B-LoRA regime**, 150-demo real tasks; same small-data caveat
  as everything in this family.

## Which idea/arm it fed

**#16 (rig transfer benchmark)**: menu entry — the zero-
infrastructure SFT lever, priced at one loss term + one tuned
weight; candidate for any future rig-side AR fine-tune pre-reg, not
an arm today. **#19 (AR sampled draws)**: external-prior note — a
training-side push toward exactly the unimodal-around-the-mode shape
our decode reads measured. Cross-refs:
[decode temperature](decode-temperature.md),
[action tokenization](action-tokenization.md), the RL-pole entries
([RLDT](rldt-density-transport-rl.md),
[Z-1](z1-selective-joint-rl.md), [FlowPRO
in the Hy-Embodied stack](hy-embodied-stack.md)).
