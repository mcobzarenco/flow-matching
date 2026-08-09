# π-StepNFT: RL on a flow policy, one denoising step at a time

*Read 2026-08-09 (lit slice `lit-radar-0812b`, priority 2: the #16
RL pole, beside RLDT / Z-1 / FlowPRO). Paper:
[2603.02083](https://arxiv.org/abs/2603.02083) — "π-StepNFT: Wider
Space Needs Finer Steps in Online RL for Flow-based VLAs" (cs.RO).*

**The paper in plain words.** After you train a robot policy by
imitation, you can improve it further with reinforcement learning:
let it try tasks, and push it toward what worked. For flow-matching
policies this is awkward — the action comes out of an iterative
denoising process, so there's no simple probability to increase, and
the usual fixes bolt on a value network (a critic) or backpropagate
through the whole denoising chain. This paper's move: make the
denoising *stochastic* so the policy explores a wider space
("wider space"), and then supervise each individual denoising step
toward or away from the transition it actually took, depending only
on whether the episode succeeded ("finer steps"). No critic, no
likelihood, one forward pass per update. It roughly matches PPO-based
methods on in-distribution benchmarks and clearly beats them
out-of-distribution — the claim is that critic-free step-level
updates don't overfit the way value networks do.

## What it contributes

- **The algorithm**: roll out with a K-step flow-SDE sampler (noise
  injected per step = the exploration), record single transitions
  (x_t → x_{t−}) with the episode's binary outcome r ∈ {0,1};
  construct two mirrored velocity candidates v⁺/v⁻ around the
  rollout policy and train a contrastive ranking objective that
  pulls the observed transition's branch for successes and pushes it
  for failures. The affine velocity↔transition-mean relation carries
  gradients without differentiating through the solver.
- **Numbers (few-shot SFT + RL)**: LIBERO π0 57.6 → 90.5 avg
  (PPO-based πRL reaches 96.0 — value methods still win IND);
  π0.5 77.1 → 94.0 (πRL 97.9). ManiSkill OOD is the headline:
  π0 50.4% vs πRL's 39.3% (+11.1), with semantic-shift nearly
  doubled (49.1 vs 25.4); π0.5 59.5 vs 49.3.
- **Ablations**: terminal-x₀ supervision is unstable (needs
  conservative EMA), step-wise targets train aggressively and
  stably; ranking beats weighted-MSE on binary signals; sparse
  binary labels stay competitive with dense value estimates.
- **Infra price**: 8×H100, RLinf co-located sim rollouts
  (LIBERO/ManiSkill), sim-only, binary success signal required.

## What transfers to us

1. **RL-pole entry 4, and the pole now has an internal axis.** The
   #16 post-SFT menu's RL pole holds RLDT (SVGD density transport),
   Z-1 (GRPO over flow-SDE log-probs), FlowPRO (preference pairs),
   and now StepNFT (critic-free step-contrastive). The new structure:
   *value-based methods buy peak in-distribution success; critic-free
   methods buy OOD retention* (−5.5 IND / +11.1 OOD vs PPO on π0).
   For a rig-transfer setting — few demos, distribution shift
   guaranteed — that trade favors the critic-free end, and StepNFT
   is the first to measure it head-to-head.
2. **The SDE-exploration premise touches our noise-ticket findings.**
   Their "wider space" is per-step noise injection — the same
   channel our ticket screen proved is *directional and
   context-interacting* (steering III: interaction 39.4% vs noise
   main effect 1.4%). StepNFT explores it blindly and lets binary
   outcomes sort it out; a ticket-informed exploration prior is an
   obvious unpublished hybrid, noted on the ladder, no arm.
3. **Step-level supervision echoes TCFM** from the training side:
   both argue the flow's *intermediate* states, not just endpoints,
   are where useful signal lives.

## What doesn't transfer

- **Sim-only, env-in-the-loop**: needs thousands of parallel
  rollouts with success flags — the same infra price as RLDT, and
  the pole stays sim-first/parked on #16's rig data.
- **Long-horizon weakness**: LIBERO-Long 86.7 vs PPO's 90.2 — their
  own numbers say sparse binary credit assignment degrades exactly
  where episodes are long; a rig curriculum would sit there.
- **K is small by latency necessity** (short denoising paths for
  real-time rollout) — our 1-NFE SnapFlow student has *no*
  intermediate steps to supervise; StepNFT-style RL would apply to
  the teacher, not the deployed student.

## Which idea/arm it fed

**#16 (rig-transfer benchmark)**: post-SFT menu RL-pole entry 4,
with the pole's first measured IND-vs-OOD trade (critic-free +11.1
OOD over PPO, −5.5 IND); menu ordering unchanged, everything still
gated on rig data. **#1 (noise ensembling)**: footnote on the
ladder — RL-through-noise-space is DSRL's premise arriving via
per-step SDE noise; ticket-informed exploration named, not queued.
Cross-refs: [RLDT](rldt-density-transport-rl.md) and
[Z-1](z1-selective-joint-rl.md) (the pole's other entries),
[Hy-Embodied stack](hy-embodied-stack.md) (FlowPRO, the
weight-space/preference pole),
[noise-space steering III](noise-space-steering-3.md) (why blind
noise exploration is leaving structure on the table).
