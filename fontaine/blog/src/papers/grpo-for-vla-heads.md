# GRPO for our two heads: Flow-GRPO · πRL · SimpleVLA-RL (deep read)

*Lit slice `0821`, 2026-08-12 (owner-called 09:23Z: "investigate how
we could implement GRPO to train jointly the AR objective and
flow-matching (or maybe just one) directly on the sim — just
research at this point"). First landed 09:3xZ at survey depth;
**upgraded to deep-read depth 11:3xZ** the same day by the
`grpo-on-sim-design-research` item, whose deliverable is the
[design memo](../posts/2026-08-12-grpo-sim-design-memo.md). One
survey-stage claim is corrected below (πRL's algorithm). Cluster:
[Flow-GRPO](https://arxiv.org/abs/2505.05470) (NeurIPS 2025) ·
[πRL](https://arxiv.org/abs/2510.25889) ·
[SimpleVLA-RL](https://arxiv.org/abs/2509.09674).*

## Plain words

GRPO is the training trick behind recent reasoning-model successes:
generate a *group* of attempts at the same problem, score them, and
push the model toward the attempts that scored above the group's
average — no learned value function, just relative comparison. For
a robot policy you need a score (our simulator's progress measures,
free and automatic), genuinely different attempts (stochastic
decoding), and the probability of each attempt (so the push has a
direction). Our AR head has probabilities natively — actions are
tokens. Our flow head is deterministic given its noise and has no
tractable probability — the 2025 papers close exactly that gap. The
deep read adds the fine print the abstracts hide: the group sizes
that collapse, the noise levels that break rollouts, the clipping
constants that differ by four orders of magnitude between domains,
and one surprise — the paper we filed as the flow-GRPO-for-robots
existence proof actually trains with PPO and shows GRPO *losing*.

## SimpleVLA-RL (2509.09674) — the AR blueprint, now with constants

**Contribution**: outcome-only RL on an autoregressive VLA
(OpenVLA-OFT, veRL infra), aimed at the data-starved regime.

**The recipe** (§3): rollouts sample action tokens at **temperature
1.6** (raised from 1.0 as an explicit exploration knob; greedy at
eval), **G=8** rollouts per task, advantage = group z-score
broadcast to every action token, **clip-higher [0.8, 1.28]** (from
DAPO), **KL penalty removed entirely** (no reference policy kept),
token-level loss aggregation, lr 5e-6, 64 tasks × 8 rollouts = 512
trajectories per step on 8×A800. **Dynamic sampling**: groups where
all 8 succeed or all 8 fail are discarded — a degenerate group has
no gradient. Reward strictly binary 0/1, no shaping.

**Experiments**: from one-demo-per-task SFT, LIBERO-Long 17.3→91.7
(suite avg 48.9→96.9); from full-data SFT 91.0→99.1, past π0
(94.2). RoboTwin 1.0/2.0: +30 pts over SFT, past π0 both times.
Real transfer: sim-only training deployed on AgileX arms, 17.5→38.5
avg over 4 tasks. Emergent "pushcut" behavior — the policy finds
non-demonstrated shortcuts that genuinely succeed (framed as
discovery, but it is one detector-audit short of reward hacking).

**The catch for us** (§6.2): from a 0%-success base the binary
reward is all-zero and RL never starts; 100-demo SFT (~1% success)
barely moves (1.2→4.3). **Every published number starts from a
policy that already succeeds sometimes.** Our sim arms are 0/500 —
binary success cannot be our reward; the dense `progress_final_cm`
substitute and whether it carries within-group variance is exactly
what the design memo's signal probe measures.

## Flow-GRPO (2505.05470) — the ODE→SDE mechanism, exact form

**Contribution**: makes GRPO possible on flow-matching generators
by converting the deterministic ODE into a marginal-preserving SDE
whose per-step transitions are Gaussian — closed-form logprobs and
real sampling diversity (image models: SD3.5-M, FLUX).

**The mechanism** (§4, App. A): with rectified-flow velocity v and
noise schedule σ_t = a·√(t/(1−t)), the Euler–Maruyama step is
x_{t+Δt} = x_t + [v + (σ_t²/2t)(x_t + (1−t)v)]Δt + σ_t√Δt·ε — an
isotropic Gaussian with known mean/variance, so per-step logprobs
(and PPO-style ratios) are exact. Two practical gifts: the **KL to
the frozen reference is closed-form in velocity space** — a
weighted MSE between current and reference velocity, no reference
logprob pass — and **denoising reduction**: train on 10 SDE steps,
infer with 40, >4× wall-clock, no reward loss (T=5 stopped
helping). For an action head already decoding in ~10 steps the
train/infer gap mostly vanishes.

**Constants and their bracketing** (§5.3, App. B, repo): noise
a=0.7 for image latents (a=0.1 explores too little; a>1 wrecks
samples, zero reward, dead training). **G=24, and the ablation
matters: G=12 and G=6 collapsed** (noisy advantage estimates). Clip
ε ≈ 1e-4 (not the LLM 0.2 — per-pixel Gaussian density ratios over
a huge latent explode; our 50×6 chunks sit between the regimes, so
ε must be re-found, likely well above 1e-4 and below 0.2). β_KL =
0.04/0.01 by task; LoRA lr 3e-4; 24×A800, ~1152 rollouts per epoch.

**Reward hacking** (§5.3): without KL, off-task quality collapses
(aesthetic scores drop) or — subtler — **diversity collapses**:
different seeds converge to nearly identical outputs while the
reward metric stays green. With KL they match the no-KL peak reward
at the cost of longer training. The robot-policy analogue of
diversity collapse is a mode-collapsed policy; our early-warning
channel is guard-trip telemetry plus per-group action spread.

**Caveat**: single-generation MDP — one image, one reward. A robot
episode chains many chunk generations, so cross-chunk credit
assignment is outside their formulation (πRL's two-layer MDP is the
published answer).

## πRL (2510.25889) — CORRECTED: a PPO paper with a GRPO appendix

**The survey-stage claim on this page ("πRL's group construction +
KL anchor for flow" as a deep-read target) was wrong on both
counts.** πRL's main algorithm is **PPO with GAE and a learned
critic**; GRPO appears once, in App. F.1, and **loses** (LIBERO avg,
Flow-SDE: π0 90.0 GRPO vs 96.0 PPO; π0.5 91.5 vs 97.9), with no
group-construction details published. And there is **no KL anchor
anywhere** — KL to the SFT policy is only monitored; a runaway on
LIBERO-Long is tamed with cosine LR annealing, not a penalty.

**Contribution**: two ways to give a flow-based VLA (π0/π0.5) usable
action likelihoods, trained online in parallel simulators (RLinf).

- **Flow-Noise** (§4.1): the denoising chain becomes a discrete
  MDP with a *learnable noise network* (per-dimension σ, trained
  jointly, **discarded at inference** → deterministic ODE deploy);
  the joint chain likelihood is exact. Log-var clamped ([0.08,
  0.16] π0), entropy bonus 0.005.
- **Flow-SDE** (§4.2): Flow-GRPO's conversion embedded in a
  **two-layer MDP** (denoise-step × env-step, reward only at the
  boundary), plus a **hybrid sampler**: per env step, ONE randomly
  chosen denoising step is stochastic SDE, the rest stay ODE —
  same final success, 2× wall-clock.

**Constants for actions** (Tables 11–13): noise **a=0.5** (0.3 on
two suites; ablation: a=0.2 barely refines — high clip fraction,
unstable; a=0.8 degrades rollout fidelity), **K=4–5 denoising
steps** during RL (K=1 collapses rollouts: severe discretization
error), deterministic ODE at eval, clip ε=0.2, actor lr ~5e-6,
γ=0.99/λ=0.95, 64–320 parallel envs, 8×H100. **Action-chunk
ablation aimed at us**: chunk 5→10→20 gives RL eval 94.5/95.5/89.2
— big chunks blur credit assignment and cap the RL ceiling. We fly
chunk 50 (execute 30/replan).

**Trunk policy**: SFT tunes all 3.3B; **RL freezes the VLM trunk
and trains only the ~300M expert** (+critic). Their VLM-LoRA
ablation found no benefit. This is the fourth frozen-trunk vote in
our RL-pole reading (Z-1, RDT2, LWD).

**Experiments**: π0.5 from **40 demos total** (one per subtask):
LIBERO avg 77.1→98.3, past full-data SFT (96.9). ManiSkill 4,352
pick-place combos: ID 40.1→90.9. Real2Sim2Real: 20-demo SFT at 0%
→ RL 40% real-world. OOD: gains transfer to visual/execution
shifts, NOT to novel task objectives.

## What transfers to us / what doesn't

- **Transfers**: seeded same-spawn groups (cleaner than anything in
  these papers); dense `progress_final_cm` as the group score
  (GRPO only needs within-group *ranking*); `ARSampling`
  temperature (the T=1.6 knob exists in our stack today, as does
  per-draw flow noise); the SDE sampler as ~30 lines beside
  `sample_actions`; the velocity-MSE KL; a≈0.5/K≈4 as flow
  starting points; frozen-trunk discipline; dynamic-sampling
  filtering; clip-higher for the AR head.
- **Doesn't (or needs work)**: binary success rewards (our 0/500
  floor — the SimpleVLA-RL dead-start result is the sharpest fact
  in the cluster); their group sizes may not survive our reward
  noise (images needed G=24; binary-reward VLAs used G=8; dense
  rewards land somewhere between — probe question); Flow-GRPO's
  clip ε and image noise scale; πRL's critic + 64–320-env fleets
  (we have one H100 and a parallel-rollouts scaffold); chunk-50
  credit assignment is flagged risky by the one ablation that
  tested it; "joint AR+flow GRPO" exists in no paper — the memo
  files it as phase 3 on the merged molmo_flow model.

## Fed into

The [GRPO-on-sim design memo](../posts/2026-08-12-grpo-sim-design-memo.md)
(this item's deliverable): stack audit, the within-group-variance
crux, and the proposed first experiment — a rollout-only **signal
probe** (4 cells × 15 seeds × K=8 draws, v3 frames, ≤3 GPU-h on the
parallel path) measuring whether group-relative advantage has any
signal at our competence floor before GRPO infra is built.
`ideas.md` hook `0821`.

Sources:
[Flow-GRPO](https://arxiv.org/abs/2505.05470) ·
[Flow-GRPO code](https://github.com/yifan123/flow_grpo) ·
[πRL](https://arxiv.org/abs/2510.25889) ·
[SimpleVLA-RL](https://arxiv.org/abs/2509.09674) ·
[SimpleVLA-RL code](https://github.com/PRIME-RL/SimpleVLA-RL) ·
[RLinf-VLA](https://arxiv.org/abs/2510.06710)
