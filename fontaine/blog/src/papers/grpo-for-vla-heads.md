# GRPO for our two heads: the mechanism map (Flow-GRPO · πRL · SimpleVLA-RL)

*Lit slice `0821`, 2026-08-12 (owner-called 09:23Z: "investigate how
we could implement GRPO to train jointly the AR objective and
flow-matching (or maybe just one) directly on the sim — just
research at this point"). **Survey-depth page** — abstracts + docs;
the deep reads (algorithms sections, hyperparameters, compute) are
the queued `grpo-on-sim-design-research` item, which owes the owner
a design memo naming a first cheap experiment. Cluster:
[Flow-GRPO](https://arxiv.org/abs/2505.05470) (NeurIPS 2025) ·
[πRL](https://arxiv.org/abs/2510.25889) ·
[SimpleVLA-RL](https://arxiv.org/abs/2509.09674).*

## Plain words

GRPO is the training trick behind recent reasoning-model successes:
generate a *group* of attempts at the same problem, score them,
and push the model toward the attempts that scored above the
group's average — no learned value function needed, just relative
comparison. To use it on a robot policy you need two things: a
score (our simulator's progress/success measures, free and
automatic) and the ability to say *how probable was the action
sequence I just tried* so you can nudge probabilities. For our AR
head, actions are tokens — probability is what the model already
outputs. For our flow-matching head there's a genuine obstacle: the
flow decoder is a deterministic process (same noise in, same action
out) with no tractable probability of its output. The 2025 papers
resolved exactly this, two ways, and both are compatible with our
stack on paper.

## The three mechanisms

1. **AR head — token-level GRPO
   ([SimpleVLA-RL](https://arxiv.org/abs/2509.09674))**: sample K
   rollouts per task by *sampling action tokens* instead of greedy
   decode; reward = outcome 0/1 from the env; GRPO = group-relative
   advantage + PPO clipping on token logprobs (veRL infra). On
   LIBERO with OpenVLA-OFT: 17.3 → 91.7 from a SINGLE demo per task
   as cold-start SFT — the headline is data efficiency, exactly our
   regime (26 A-half episodes). This maps onto our AR objective with
   no algorithmic invention: our sim already provides seeded resets,
   `progress_final_cm` (denser than 0/1 — likely necessary given our
   0/500 success floor) and deterministic paired groups.
2. **Flow head, route A — ODE→SDE
   ([Flow-GRPO](https://arxiv.org/abs/2505.05470))**: convert the
   deterministic flow ODE into an SDE that provably preserves the
   per-timestep marginals; the SDE's per-step Gaussian transitions
   give closed-form logprobs and real sampling diversity → GRPO
   ratios/KL work as usual. Plus "denoising reduction": train on
   few SDE steps, infer with the full schedule. πRL's **Flow-SDE**
   is this idea embedded in a two-layer MDP (denoise-step ×
   env-step) for π0/π0.5-class VLAs.
3. **Flow head, route B — learnable noise
   ([πRL](https://arxiv.org/abs/2510.25889) Flow-Noise)**: make the
   denoising chain a discrete-time MDP with a *learnable noise
   network* so the action log-likelihood is exact. More invasive
   (adds a network) but no marginal-preservation caveats.

## What transfers to us / what to check in the deep read

- **Rewards we already own**: seeded sim, paired groups (K
  appearance/noise draws at the SAME spawn = a natural GRPO group
  with the spawn held fixed), `progress_final_cm` as a shaped
  reward, strike gates as constraint terms. Nobody hands us this —
  it is the payoff of the eval-first sim work.
- **Throughput is the real constraint**: GRPO wants thousands of
  rollouts. At ~10 ticks/s/env (v3, GPU compositor) a 900-tick
  episode is ~90 s — `sim-parallel-rollouts` (owner-sequenced
  first GPU item) is a prerequisite, and training could run v0
  frames / eval v3 (the ecosystem page's split).
- **Joint vs single head**: SimpleVLA-RL suggests starting with the
  AR head alone (mechanism proven, infra light); Flow-GRPO/πRL say
  the flow head is feasible second. "Jointly" (owner's ask) has no
  published recipe we found at survey depth — the design memo
  should treat it as the open question (shared trunk, two
  policy-gradient terms, one reward).
- **Deep-read checklist** (queued item): πRL's group construction +
  KL anchor for flow; Flow-GRPO's SDE noise scale vs our heun-10
  decode; SimpleVLA-RL's exploration temperature and how it avoids
  reward hacking on dense rewards; all three papers' GPU budgets
  scaled to 1×H100.

## Fed into

`grpo-on-sim-design-research` (queue): this page is its mechanism
baseline; the memo owes the owner a concrete first experiment
(candidate shape: AR head, er60k init, 20 spawn-seeds × K=8
token-sampled rollouts/group, reward = progress_final + success
bonus, GRPO with KL-to-SFT anchor, v0 frames — priced AFTER
sim-parallel-rollouts lands). `ideas.md` hook `0821`.

Sources:
[Flow-GRPO](https://arxiv.org/abs/2505.05470) ·
[Flow-GRPO OpenReview](https://openreview.net/forum?id=oCBKGw5HNf) ·
[πRL](https://arxiv.org/abs/2510.25889) ·
[SimpleVLA-RL](https://arxiv.org/abs/2509.09674) ·
[RLinf-VLA](https://arxiv.org/abs/2510.06710)
