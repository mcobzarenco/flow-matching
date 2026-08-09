# SA-VLA: keeping RL from eroding what pretraining knew

*Read 2026-08-09 (lit slice `lit-radar-0813`, priority 4: #16 RL pole
× #11 spatial-aux crossover). Paper:
[2602.00743](https://arxiv.org/abs/2602.00743) — "SA-VLA:
Spatially-Aware Flow-Matching for Vision-Language-Action
Reinforcement Learning" (Pan, Wan, Yu et al. — A*STAR CFAR + Wuhan +
NUS + NTU, v1 2026-01-31, preprint, no venue).*

**The paper in plain words.** Fine-tuning a robot policy with
reinforcement learning is supposed to make it better at the task, but
when the only feedback is "you succeeded / you failed," the policy
tends to find shortcuts — it latches onto surface visual cues and
quietly forgets the 3D understanding it inherited from pretraining,
so it breaks the moment the camera moves. This paper's fix comes in
three parts: feed the policy frozen 3D features from a pretrained
geometry model (so the spatial knowledge can't be trained away),
replace the all-or-nothing reward with dense progress rewards
computed from geometric distances (did the gripper get closer to the
object? did the object get closer to its goal?), and make the
exploration noise a learned, state-dependent quantity so that the RL
algorithm's bookkeeping actually accounts for it. The most honest
number in the paper cuts the other way, though: with sparse rewards,
RL fine-tuning made the policy *worse* than not doing RL at all
(77.5% vs 81.0%) — the machinery is mostly recovering ground that
naive RL loses.

## What it contributes

- **Frozen geometric feature injection, explicitly *not* an aux
  loss.** VGGT (a pretrained multi-view geometry transformer)
  tokens enter the visual stream through gated unidirectional
  cross-attention (visual queries only; a learnable tanh gate;
  global scene tokens bypass attention). The tokens are frozen and
  gradient-free during RL — read-only geometry the optimizer cannot
  erode. Their appendix argues the alternative (co-trained
  reconstruction/depth losses) injects competing gradients that
  destabilize RL — a design claim worth remembering independent of
  this paper.
- **Dense geometric progress rewards**: episodes decomposed online
  into Reach/Place/Leave by stability heuristics; reward = signed
  per-step *change* in normalized eef–object / object–goal distance
  (λ=0.3), reference distances re-anchored at each phase switch.
  Requires privileged object/goal poses — simulator-only by
  construction.
- **SCAN exploration**: exploration noise σ(x,t) predicted from the
  fused embedding with an annealed isotropic floor
  (α(t)·√(t/(1−t)), annealed over 80 of 100 PPO steps). The
  taxonomy behind it is the reusable part: external SDE noise is
  invisible to PPO's likelihood ratio (the surrogate can't adapt
  variance); a learned noise head makes each denoising transition a
  Gaussian whose mean *and* variance enter the ratio; SCAN anneals
  from the first to the second, buying early coverage and end-state
  PPO-consistency.
- RL itself is the heavy pole: actor-critic PPO + GAE over
  ReinFlow-style flow transitions, 64 parallel envs, batch 1024,
  ~154 GPU-h on 4×H800 for 100 update steps.

## The experiments it ran

Sim only — LIBERO SFT, evaluated exclusively on the LIBERO-Plus
spatial-perturbation subset (camera + init-state shifts), π0.5 base,
2 seeds, internal ablations only (no external RL-VLA baselines run):

| Variant | SR |
|---|---|
| π0.5 SFT, no RL, no fusion | 81.00 |
| + spatial fusion, no RL (zero-shot) | 83.25 |
| RL sparse reward + fusion | **77.50** |
| RL dense reward + fusion (learned noise) | 83.00 |
| Full SA-VLA (+ SCAN) | 83.75 |

Zero-shot, fusion helps viewpoint shift (+3.83) far more than
init-state shift (+0.52).

## What transfers to us

- **The #16 RL-pole ledger gets its cautionary row.** This is the
  first entry where the *sign* of naive RL is measured negative:
  sparse-reward PPO with a critic lands 3.5 points *below* no-RL,
  and the full three-mechanism apparatus nets +2.75 over the SFT
  baseline — of which +2.25 was available zero-shot from the frozen
  fusion alone. Reward density is the biggest lever (+5.5
  sparse→dense), exploration parameterization the smallest (+0.75).
  Beside [π-StepNFT](pi-stepnft.md)'s IND-vs-OOD trade and
  [Z-1](z1-selective-joint-rl.md)'s diagnostic gating, the pole's
  emerging shape is: the RL update itself is the risk, and most
  published gains are protective machinery against it.
- **The noise-parameterization taxonomy** (external = ratio-blind,
  learned = consistent, annealed = both) is a design pattern any
  future flow-RL arm should inherit — it's the same
  likelihood-tractability trick as Z-1's flow-SDE conversion, now
  with an explicit account of *why* the variance must live inside
  the policy.
- **For #11/#17: inject frozen geometry, don't co-train it, when RL
  is in the loop.** This is a third position distinct from our
  banked aux poles (VEGA/Spatial-Forcing co-train an alignment loss
  at SFT; QDepth co-trains a generative head): under RL, frozen
  read-only features are the erosion-proof form. Notably their
  zero-shot fusion gain (+2.25, viewpoint-loaded) is itself a small
  SFT-side datum for the spatial-aux family — from injection alone,
  no loss.

## What doesn't transfer

- **The rewards.** Ground-truth object/goal/eef poses and scripted
  phase logic — privileged sim state with no stated path to real
  hardware. The whole dense-reward lever is sim-first, like the
  rest of the RL pole (this paper: 64 parallel envs).
- **The headline as an RL win.** SFT+fusion zero-shot (83.25) vs
  full RL pipeline (83.75) is +0.5 under not-directly-comparable
  protocols, 2 seeds, no CIs; IND performance is never reported, so
  the "RL erodes spatial bias" narrative rests on one sparse-RL
  datum plus citations. Evidence class: suggestive, self-ablated
  only.
- The clean spatial-fusion × RL interaction cell (RL without fusion)
  is missing from the ablation table — the one comparison the title
  promises isn't isolated.

## Which idea/arm it fed

[#16](../ideas/16-rig-transfer-benchmark.md) — RL-pole entry 5:
first measured *negative* sign for naive sparse RL (77.5 vs 81.0
no-RL); protective-machinery framing; noise-taxonomy design pattern
banked for any flow-RL pre-reg. [#11](../ideas/11-visual-grounding.md)
/ [#17](../ideas/17-new-trunks.md) — the aux family gains a fourth
integration mode: frozen feature injection (erosion-proof under RL;
+2.25 zero-shot, viewpoint-loaded), beside VEGA / Spatial Forcing /
QDepth's trained-loss recipes. Cross-refs:
[Z-1](z1-selective-joint-rl.md), [π-StepNFT](pi-stepnft.md),
[RLDT](rldt-density-transport-rl.md),
[VEGA](vega-encoder-grounding.md),
[Spatial Forcing](spatial-forcing.md), [QDepth-VLA](qdepth-vla.md).
