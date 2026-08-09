# Learning While Deploying: the RL pole's first fleet-scale entry, and a critic built for messy fleet data

*Read 2026-08-09 (lit slice `lit-radar-0816`, priority 2: the #16
RL-pole roster). Paper:
[2605.00416](https://arxiv.org/abs/2605.00416) — "Learning While
Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot
Policies" (Wang et al., AgiBot; v2 2026-06-03).*

**The paper in plain words.** Most robot policies are trained once
and then frozen when deployed. This paper keeps sixteen real
two-armed robots *learning while they work*: each robot streams its
experience back to a central learner, which updates the policy and
pushes a new version to the whole fleet every fifty training steps.
Humans step in only to correct a rollout going wrong, and those
corrections become ordinary training data. The learning recipe has
two pieces: a value function that models the *distribution* of
outcomes rather than a single average — so a rare success in messy
fleet data isn't averaged into oblivion — and a policy update that
steers a flow-matching action generator directly with the critic's
gradient, no likelihoods needed. After four wall-clock hours
(~60 robot-hours) of deployment learning, average task score rises
from 0.88 to 0.95.

## What it contributes

- **The fleet system**: centralized learner, asynchronous
  transition streaming, policy broadcast every 50 steps; offline RL
  pretraining then online continuation on mixed replay. During
  online updates **the VLM backbone stays frozen — only the flow
  action expert trains** (critics train fully): our architecture
  shape, running RL in production.
- **DIVL (their novel piece)**: instead of IQL's scalar expectile
  value, a categorical distribution over dataset action-values
  trained by NLL against the target critic, with the implicit max
  recovered by quantile extraction. Provably equivalent to
  expectile regression at the optimum — the payoff is the
  *representation*, which keeps rare-but-reproducible successes
  visible in heterogeneous fleet data. Plus an adaptive quantile
  driven by the value distribution's entropy (diffuse → act
  conservative), and n-step chunk-level backups for sparse rewards.
- **QAM policy extraction — adopted, not invented** (our hook
  over-credited it; it is Li & Levine's method, used here): the
  critic's gradient at the denoised action becomes, via backward
  adjoint dynamics, a per-noise-level regression target on the
  velocity field. Flow-native — no discretization, no likelihood,
  no backprop through the ODE — now demonstrated at VLA scale on
  real hardware.

## The experiments it ran

16 AgiBot G1 dual-arm robots (30 Hz joint control, π0.5-style
PaliGemma + 300M flow expert), 8 tasks: 4 grocery-restocking
(binary success) + 4 long-horizon 3–5 min tasks (Gongfu tea, juice,
cocktail, shoebox; human-rubric scores with partial credit).
Averages: SFT 0.76 → offline RL 0.88 → **online 0.95** (RECAP and
HG-DAgger baselines both 0.85); short-horizon 0.99, long-horizon
0.91. The ablation that matters: DIVL vs plain expectile regression
is a wash on short-horizon but **+9.7/+16.7 points
(offline/online) on long-horizon** — the distributional critic
carries the gain exactly where credit assignment is hard. Online
stage: 4 h wall-clock ≈ 60 robot-hours. Honesty flags: the 0.95
mixes binary success with human-scored rubrics (trial counts
unreported), robots were pooled per-task rather than one generalist
deployment, and intervention rates / reset mechanics / human cost
go unreported.

## What transfers to us

- **RL-pole roster entry 8 — a new infrastructure tier.** RedFlow
  (entry 7) showed offline RL from 100–200 deployment rollouts;
  LWD sits one rung up: offline-RL pretrain → continuous online
  improvement, real hardware, frozen trunk + flow-expert-only
  updates. The pole's cost axis finally gets a real number at the
  high end: ~60 robot-hours bought +7 points over an already-tuned
  offline policy.
- **The borrowable-today piece is the offline column, and it's not
  small**: LWD-Offline alone beats SFT 0.88 vs 0.76, before any
  online loop — and DIVL-vs-expectile is +9.7 of that on
  long-horizon. DIVL + QAM are both offline-runnable on a frozen
  trunk. The honest prerequisite: their offline buffer contains
  failures and play data with terminal binary labels — on our
  success-only corpus every trajectory has r=1 and the advantage
  signal collapses. The path runs through the owner rig's future
  early-policy rollouts (bank failures as they happen) or post-hoc
  failure labeling, not through more teleop successes.
- The frozen-backbone-in-production detail is a sixth production
  vote for the F shape in the #4 ledger: even mid-RL, with every
  incentive to adapt, the trunk stays frozen.

## What doesn't transfer

- The .88→.95 online jump and everything fleet: broadcast loops,
  intervention streams, human evaluators scoring rubrics — none of
  it exists without deployed robots and supervising humans.
- The 95% headline is a mixed, human-judged metric with unreported
  trial counts; treat it as "large, real, imprecisely priced."
- No safety modeling, single-instruction tasks, 4 h online horizon —
  drift and forgetting over long continual deployment untested.

## Which idea/arm it fed

#16 (`rig-benchmark`) — RL-pole entry 8, the fleet-scale
offline-to-online tier; DIVL banked as the offline-critic candidate
for the rig-data era (with the failure-data prerequisite stated),
QAM as the flow-native extraction primitive at VLA scale. #4
(`seam-screen`) — one more production frozen-trunk vote in the
ledger. No gate changes; the pole stays gated on rig data, but its
cheapest real-hardware recipe is now better specified.
