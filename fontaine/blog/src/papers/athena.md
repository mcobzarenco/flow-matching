# ATHENA: influence-function curation finally runs at π-0 scale — and it still needs the rollouts we don't have

*Read 2026-08-09 (lit slice `lit-radar-0818`, priority 1). Paper:
[2606.16208](https://arxiv.org/abs/2606.16208) — "ATHENA: Accelerated
Multi-Task Heterogeneous Influence Functions for Robot Data Curation"
(Tao Xu, Jiaxin Wang, Runhao Zhang, Jiayi Guan, Xianchao Zeng, Weixi
Song, Xinyu Zhou, Zhetao Chen, Guang Chen, Yong-Lu Li; arXiv cs.RO,
submitted 2026-06-15; Shanghai Innovation Institute et al.; CC BY 4.0).*

**The paper in plain words.** When you train a robot on hundreds of
demonstrations, some of them help and some quietly hurt. "Influence
functions" are the principled way to find out which is which: for each
demo, estimate how the robot's success would change if that demo were
removed from training. The catch is that the estimate involves the
gradients of every training example through the whole model, plus an
inverse-Hessian — hopeless at billions of parameters. ATHENA's
contribution is making that arithmetic cheap enough to run on π-0, a
3.3B-parameter vision-language-action model: it never materializes the
full gradient (it exploits the fact that a linear layer's gradient is
an outer product of two small vectors), and it replaces the Hessian
inverse with a low-rank approximation. Result: what would have cost
~8,000 GPU-hours costs ~26. They then keep only the most helpful half
of the data, retrain, and match — sometimes beat — training on
everything. The fine print: "helpful" is defined against success and
failure rollouts of an already-trained policy, so you need a working
evaluation loop before you can curate anything, and the corpora they
curate are small (9.3 hours in sim, 6.9 hours real).

## What it contributes

- **Kronecker-structured gradient projection.** For a linear layer,
  the per-example weight gradient is delta_i (x_i)^T — an outer
  product. ATHENA projects the two factors separately instead of the
  materialized D_l-dim gradient, cutting per-layer projection cost
  from O(D_l · P) to O(sqrt(D_l · P)). No per-example full-gradient
  storage at 3.3B params.
- **Random Truncated Approximation (RTA) for the Hessian.** Rank-r
  SVD of projected gradients replaces dense inversion:
  psi = phi_te^T (Sigma_r² + lambda I)^−1 phi_tr, dropping leading
  cost from O(N P² + P³) to O(N P r). Values of P, r, lambda are
  **not stated in the text we could extract** — a reproducibility gap
  on top of the missing code.
- **Square-flow attribution surrogate** for flow-matching policies:
  f = E_{t,eps} ||v_theta(x_t, s, t)||², a scalar on the learned
  velocity field. This sidesteps backprop through ODE integration —
  directly relevant to any flow-matching action expert, including
  ours.
- **Rollout-anchored influence.** Demonstration influence aggregates
  action-level influence over m evaluation rollouts weighted by
  binary return R(tau) in {1, −1}. This is the CUPID recipe (which
  the paper credits, noting CUPID topped out at 24M-param,
  single-task policies) scaled up. It is NOT offline curation.
- **Multitask Influence Interaction (MII).** Rank-normalized product
  of a demo's influence on its own task and on all other tasks, so
  greedy selection doesn't starve low-signal tasks across the
  50-task joint training mix.
- **Speedup: 313.4× at K=50 tasks** (8,054.6 → 25.7 GPU-hours,
  560.5K timesteps; range 235.5×–405.6× across K=5..50). Baseline is
  their own unaccelerated dense-influence implementation, not a
  prior accelerated method — read the multiplier accordingly.
  Hardware named only as "140 GB memory" GPUs (H200-class?), count
  unstated.

## The experiments it ran

- **RoboTwin 2.0 sim, 50 bimanual tasks**, 2,500 demos, 9.34 h at
  16.67 Hz; retention ratios rho in {0.90, 0.75, 0.50, 0.25, 0.10};
  clean + randomized evaluators. At rho=0.50 ATHENA matches full-data
  clean (43.36% vs 43.42%) and beats it randomized (17.30% vs
  15.44%); average 30.33% vs 29.43% (+0.90pp — the abstract's
  "cumulative 45.0-point improvement" is this times 50 tasks). At
  rho=0.10 it still hits 44.70% clean / 17.72% randomized —
  half-to-90%-off the data with no loss, which says as much about
  redundancy in RoboTwin's generated demos as about the method.
- **Real robot: AgileX Cobot Magic (ALOHA-style), 6 tasks**, 720
  demos (120/task), 6.90 h at 25 Hz, 25 trials/task. ATHENA at 66.7%
  data: **68.0%** avg success vs Joint-100% full-data **60.0%**,
  Random-66.7% **50.0%**, Oracle (demo-length heuristic) **47.3%**,
  Single-task-100% **46.7%**. Note the heuristic Oracle landing
  below random. 150 total trials, so ~±5pp noise on these means.
- **Cross-model transfer (Table 2):** subsets curated with π-0
  gradients, retrained on π-0.5: 50.66% avg at rho=0.50, 41.34% at
  rho=0.10. We could not extract the matching π-0.5 full-data
  baseline, so the size of the transfer win is unverified.
- Baselines: Random, Oracle (length), TAROT (optimal transport), TSS
  (temporal surprise), Distillation (prototype deviation). Per-rho
  numbers for these live in Figure 3, which we could not read
  numerically — flagged rather than guessed.
- **Not verified / not stated:** rollout count m per task; whether
  real-task influence used real or sim rollouts; P, r, lambda; GPU
  model and count; RoboTwin demo generation pipeline (RoboTwin 2.0
  demos are tool-generated, but the paper doesn't discuss demo
  quality variance).
- **Release audit: FAILED.** Project page's Code button links to
  `./` (itself). No repo found by search. No release commitment
  anywhere in the paper. Another deep read where the banked
  artifact story doesn't survive contact.

## What transfers to us

- The two accelerations are architecture-agnostic and the
  square-flow surrogate is literally built for flow-matching action
  heads like our 367M expert. Nothing in the math blocks a
  Molmo2-4B trunk; π-0 at 3.3B is our size class.
- The cross-model transfer result licenses the cheap version we'd
  actually want: score the 229h corpus with a small proxy policy,
  apply the selection to the big run.
- The real-robot direction of the effect (+8pp over full data at
  two-thirds data; heuristic quality gating BELOW random) is the
  strongest evidence yet that our planned heuristic gates on
  community_curated_v0 need an influence-shaped sanity check.
- Cost feel: 25.7 GPU-h for 560.5K timesteps. Naive linear scaling
  to our ~25M timesteps is ~1,100 GPU-h; chunk-level scoring and
  subsampling plausibly cut an order of magnitude. Our estimate,
  not the paper's — but it is not obviously infeasible on 1–4 GPUs
  over days.

## What does NOT transfer

- **The performance signal.** Influence is anchored to closed-loop
  rollout returns R in {1,−1}. We have no rollout eval — sim or
  real. Frozen-panel chunk MAE could stand in as a pseudo-return,
  but that variant is unvalidated by this paper and inherits every
  panel-vs-rollout gap we already worry about.
- **The corpus regime.** They curate 2,500 scripted sim demos /
  720 in-house real demos with uniform collection. Our 229h is
  heterogeneous community teleop across rigs and operators — the
  redundancy structure that lets rho=0.10 match full data in
  RoboTwin may simply not exist in our data.
- **The implementation.** No code, no P/r/lambda, no rollout counts.
  Reimplementation from equations is the only path.

## Which idea it feeds

- **Idea #9 (data levers).** Concrete next step it suggests: do NOT
  invest further in demo-length or heuristic quality gates as the
  primary lever (their Oracle < Random on real tasks); instead,
  park an "offline-ATHENA" design note — Kronecker-projected
  per-chunk gradients on the 367M expert + panel-MAE pseudo-return
  — as the principled curation candidate, gated on us first having
  any rollout (or trusted-proxy) success signal. It also kills the
  hope of just cloning their pipeline: no code, and the method is
  rollout-anchored by construction.
