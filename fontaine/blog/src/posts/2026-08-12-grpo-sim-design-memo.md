# Design memo: GRPO on the sim for our two heads — and the probe that should run first

*2026-08-12 11:3xZ. The deliverable of the owner-called design-research
item (09:23Z: "investigate how we could implement GRPO to train
jointly the AR objective and flow-matching (or maybe just one)
directly on the sim — just research at this point"). Deep reads done
this session on [SimpleVLA-RL](https://arxiv.org/abs/2509.09674),
[Flow-GRPO](https://arxiv.org/abs/2505.05470) and
[πRL](https://arxiv.org/abs/2510.25889); the cluster page
[GRPO for our two heads](../papers/grpo-for-vla-heads.md) is updated
to deep-read depth alongside this memo (including one correction:
πRL's main algorithm is PPO with a critic, not GRPO). **This memo is
for owner review — nothing here is registered or launched.**

## Plain words

GRPO trains a policy by trying the same task several times, scoring
each attempt, and pushing the model toward its own above-average
attempts. To use it we need three things: a way to make the policy
try *different* things on the same task (stochastic decoding), a
score that *differs* across those attempts (a reward with
within-group variance), and the probability of each attempt so the
push has a direction (logprobs). Our AR head has all three on paper;
our flow head is missing the third until we add a known trick. But
the deep reads surfaced a sharper problem: every published success
starts from a policy that already succeeds sometimes. Ours succeed
never (0/500 in the sim study) — our substitute is the sim's dense
centimeter-level progress score. Whether that score actually varies
within a group of retries, at a noise level the policies can
tolerate, is an empirical question nobody can answer from papers —
so the memo's recommendation is a cheap measurement (a "signal
probe", rollouts only, no training, no new RL code) before any GRPO
implementation work is committed.

## 1. What the deep reads changed (vs the survey page)

**SimpleVLA-RL** (token-level GRPO on OpenVLA-OFT, the AR
blueprint) — the recipe is now concrete: rollout temperature **1.6**
(greedy at eval), group size **G=8**, group z-score advantage
broadcast to all action tokens, **clip-higher [0.8, 1.28]** (DAPO),
**KL penalty removed**, token-level loss, lr 5e-6, and *dynamic
sampling*: groups where all 8 rollouts succeed or all fail are
discarded (no gradient in a degenerate group). Reward is strictly
binary success. The catch that matters most to us (their §6.2): from
a 0%-success base the reward is all-zero and RL never starts; even a
weak base (100-demo SFT, ~1% success) barely moves. Their headline
(LIBERO-Long 17.3→91.7 from one demo per task) starts from 17%.

**Flow-GRPO** (image flows, the ODE→SDE source) — the mechanism is
exact and small: replace the deterministic ODE step with the
marginal-preserving SDE step; the per-step transition becomes an
isotropic Gaussian whose logprob is closed-form. Noise schedule σ_t
= a·√(t/(1−t)) with **a=0.7** for image latents, and their ablation
brackets it: a=0.1 explores too little, a>1 destroys sample quality
and zeroes the reward. Two transferable bonuses: the **KL anchor to
the frozen reference is closed-form in velocity space** (a weighted
velocity-MSE — no reference logprob pass needed), and *denoising
reduction* (train on 10 SDE steps, infer with 40) — which for us is
a non-issue since we already decode in 10. Two warnings: their group
size is **G=24, and G=12/6 collapsed training** (noisy advantages);
and their clip ε is ~1e-4, orders below the LLM 0.2, because
high-dimensional Gaussian density ratios explode — our 50×6 action
chunks sit between the two regimes and ε must be re-found. Without
KL they get reward hacking (quality/diversity collapse); with KL
they match peak reward but train longer.

**πRL** (π0/π0.5 in parallel sims) — **correction: this is a PPO
paper, not a GRPO paper.** PPO+GAE with a learned critic is the main
algorithm; GRPO appears once in an appendix and *loses* (LIBERO avg
90.0 vs 96.0 on π0, 91.5 vs 97.9 on π0.5), with no group details
published. There is **no KL anchor anywhere** — KL to the SFT policy
is only monitored, tamed by LR annealing. What survives for us:
their Flow-SDE noise scale for *actions* is **a=0.5** (0.3 on some
suites; their a-ablation: 0.2 too little to refine, 0.8 hurts
rollout fidelity), **K=4** denoising steps during RL with
deterministic ODE at eval, the VLM **trunk stays frozen** during RL
(only the ~300M action expert + critic trains — memory and RL4VLA
precedent), a *hybrid* sampler (one random SDE step per env step,
rest ODE) that halves wall-clock, and an action-chunk ablation
pointing straight at us: **chunk 20 already blurs credit assignment
and lowers the RL ceiling** (chunk 5–10 preferred) — we fly chunk
50, executing 30 per replan. Their few-demo result is real (π0.5
from 40 demos: 77.1→98.3 LIBERO avg) but every run leans on a
critic, 64–320 parallel envs and 8×H100.

**Net effect on the design**: (a) the AR recipe is fully specified
by SimpleVLA-RL minus its binary reward, which our 0-success floor
rules out; (b) the flow path is implementable at known cost
(ODE→SDE sampler + per-step Gaussian logprobs + velocity-MSE KL),
with a≈0.5 and K≈4–10 as starting points; (c) "GRPO vs PPO" is a
real fork for the flow head — the one head-to-head we have says PPO
wins, but it buys that with a critic and env fleets we don't have;
GRPO stays the right *first* harness on cost grounds, with the πRL
number filed as the reason to expect headroom; (d) nobody trains
the trunk during RL.

## 2. What we already own (stack audit)

- **Stochastic decoding, AR**: `ARSampling` — grammar-masked
  temperature sampling on the action block via per-row CPU-RNG
  Gumbel-max, already plumbed as `BijouPolicy(ar_temperature=…)`;
  sampled ids are device-invariant and batch-composition-invariant.
  Exact token logprobs are a softmax away. er_60k *is* an
  `ar_backbone` checkpoint (FAST tokens, vocab 1026, chunk 50).
- **Stochastic decoding, flow**: fresh noise per draw is native
  (`sample_draws`, seeded generators, stable noise keying).
  That is real action diversity but with **no logprob** — exactly
  the gap Flow-GRPO's SDE closes. Our Heun-10 decode ≈ their
  training-side step counts; an SDE sampler is ~30 lines next to
  `sample_actions` (Euler–Maruyama form, Flow-GRPO Eq. 9).
- **Reward**: the sim hands us `progress_final_cm` (initial − final
  boat→disk distance), `best-point`, a full per-tick distance trace,
  plus guard channels (`reset_strikes`, `final_upright`,
  `final_z_mm`). Dense, automatic, already pre-registered metrics
  in the sim100 protocol.
- **Groups**: seeded resets make "K stochastic rollouts at the SAME
  spawn" exact — a cleaner group than anything in the papers.
- **Throughput**: `sim-parallel-rollouts` (owner-sequenced first GPU
  item) is the enabling infra — GRPO at 450-tick episodes without
  batched envs is not viable (πRL runs 64–320 parallel envs).
- **Training step**: the AR action-block CE is a standard masked
  cross-entropy — a GRPO step is that CE, advantage-weighted with a
  clipped importance ratio. Bounded new code, but new code
  (rollout→batch plumbing, old-policy logprob capture at rollout).

## 3. The crux the papers can't answer

GRPO's gradient is proportional to the **within-group reward
spread**. Published recipes get spread from binary success at
17–77% base rates. We are at 0/500 success; our spread must come
from `progress_final_cm` under stochastic decoding. Two failure
modes, both plausible from the sim100/spot20 data:

1. **No spread**: er60k (AR) at greedy engages the boat in 4/100
   episodes and its progress distribution is a spike at ~0. If
   temperature 1.6 just adds tremor to reach-over-the-table
   trajectories, every group z-score divides ~0 by ~0.
2. **Spread at the cost of competence**: if the noise needed to
   create spread (T=1.6, or SDE a=0.5) destroys what little
   directed behavior exists (teacher80k's v3 engagement, ftrig4k's
   toward-bias), the on-policy data GRPO would train on is flailing
   — group-relative selection among garbage optimizes garbage.
   πRL's own table shows rollout success degrading with a.

There is also a **reward-hacking shape specific to us**: progress =
initial − final distance is maximized by *any* displacement toward
the disk, including teacher80k's signature knock-the-boat-flying.
sim100 already measured that behavior at scale (56/100 contact, 38
away, worst seed −12.4 cm — but sign flips under noise). A dense-
reward GRPO without guards would plausibly learn to swat. Guards exist
(`reset_strikes`, `final_upright`, `final_z_mm`, the away-tail) and
would enter the reward as registered penalties/gates, but the probe
should first measure how often sampled rollouts trip them.

## 4. The first cheap experiment: the GRPO signal probe

**Rollouts only. No training, no RL code, no new math beyond an
optional 30-line SDE sampler. Answers: does group-relative
advantage have signal here, and what does stochasticity cost us?**

Design (to be pre-registered properly before launch; numbers here
are the proposal):

- **Cells** (each: 15 seeds × K=8 stochastic rollouts, v3 frames,
  sim100 conventions — 15 replans × 30 ticks, disk pinned):
  1. `er60k` AR, T=1.0;
  2. `er60k` AR, T=1.6 (SimpleVLA-RL's setting);
  3. `teacher80k` flow, fresh ODE noise per draw (the stochasticity
     we already own);
  4. `ftrig4k` flow, fresh ODE noise per draw (the only arm with
     toward > away in sim100).
  Optional cell 5 if the owner wants the SDE question priced in the
  same pass: `teacher80k` SDE at a=0.5 (the πRL action value) —
  requires the small sampler + its bit-identity-at-a=0 oracle.
- **Deterministic anchors come free**: the v3 rerun (already
  amendment-drafted, same seeds, same spawn stream) provides each
  arm's greedy/keyed-noise baseline per seed — the probe joins
  against those rows instead of re-running them.
- **Instrument delta**: `rollout_sim` grows `--ar-temperature` and
  `--flow-draws K` (both thin flags over existing `BijouPolicy`
  knobs) + per-draw RNG keying by (seed, replan, draw). Oracle:
  draw 0 at T→greedy / keyed noise reproduces the sequential rows
  bit-for-bit.
- **Primary read, per cell**: the distribution over seeds of the
  within-group std of `progress_final_cm` (and best-point), and the
  fraction of groups whose ranking is non-degenerate — i.e. would
  survive SimpleVLA-RL's dynamic-sampling filter. Candidate signal
  bar (final number at pre-reg): median group std ≥ ~0.25 cm, a
  quarter of teacher80k's spot20 paired effect.
- **Secondary reads**: competence cost = mean progress of sampled
  rollouts vs the paired deterministic anchor (does T=1.6 / fresh
  noise lose the v3 engagement?); guard-trip rates under sampling
  (strikes, knock-offs, final_upright) = the hacking-risk price;
  AR-cell action-token entropy per replan (how much room the
  temperature actually opens).
- **Cost**: 4 cells × 120 = 480 episodes ≈ one v3 sim100 arm-set;
  **gate ≤ 3 GPU-h on the parallel path** (Path A validated), ≤ 8
  sequential fallback. Sequenced strictly AFTER
  `sim_parallel_oracle.py` and the v3 rerun (GPU-day order stands;
  the probe reuses the rerun's rows as anchors, so the order is
  also logically forced).
- **Decision rule the probe buys**: no cell clears the signal bar →
  GRPO-on-sim parks (the policies are too far from the task for
  outcome-driven RL; the sim axis keeps improving via visuals/task
  semantics instead). AR cell clears it without losing its anchor
  competence → phase 2 is the SimpleVLA-RL mapping (cheapest
  infra). Only flow cells clear it → phase 2 is Flow-GRPO SDE on
  the engaging arm, expert-only, and the AR head waits. Both clear
  → AR first (infra), flow second, joint last.

## 5. Phase 2 sketch (priced, NOT proposed for launch yet)

If (and only if) the probe shows signal — the shape of the first
actual training run, so the owner can see where this is heading:

- **AR path**: token-GRPO on er60k. Per step: 16 spawn-seeds × K=8
  = 128 episodes (~12–15 min at 8 workers), reward =
  `progress_final_cm` group-z-scored with registered guard
  penalties, clip-higher [0.8, 1.28], token-level loss, zero-var
  groups dropped, lr ~5e-6 (decoder currently trains at 1e-4 —
  RL wants the small end), KL off but the anchor implemented
  (Flow-GRPO's lesson: hacking shows up as diversity collapse; our
  guard-trip telemetry is the early warning). ~40 steps ≈ **10–15
  GPU-h gate** on 1×H100. New infra: rollout logprob capture,
  advantage-weighted clipped CE step, rollout→batch plumbing.
- **Flow path**: SDE-GRPO on teacher80k (or ftrig4k), expert-only
  per πRL, a≈0.5 and K per the probe/small sweep, velocity-MSE KL
  available closed-form. Same group geometry. Add πRL's hybrid
  sampler only if wall-clock hurts. Flag: the one published
  head-to-head says PPO+critic beats GRPO here — if the flow path
  becomes the main line, budget a critic-head experiment as the
  follow-up fork.
- **"Jointly"** (the owner's original word): no published recipe
  exists; πRL explicitly freezes the trunk, and the whole RL-pole
  reading (Z-1, RDT2, LWD) votes frozen-trunk. The honest
  formulation of joint = one reward, two ratio terms (token ratio
  for AR, per-step Gaussian ratio for flow) sharing a trunk — which
  only becomes *possible* on the merged molmo_flow model (owner
  lane, §8.13: one trunk, AR head + flow expert). Recommendation:
  treat joint as phase 3 on the merged model, evidence-gated by the
  single-head phases.

## 6. What this memo asks of the owner

1. A read of §4 — is the signal probe the right first spend, and
   should optional cell 5 (SDE) ride along?
2. The sequencing sanity-check: parallel-oracle → v3 rerun → probe,
   all inside the standing GPU-day plan.
3. Nothing else — no launches; the probe gets its own pre-reg (with
   final thresholds) if approved.

Sources: [SimpleVLA-RL](https://arxiv.org/abs/2509.09674) ·
[Flow-GRPO](https://arxiv.org/abs/2505.05470) ·
[πRL](https://arxiv.org/abs/2510.25889) ·
[cluster page (deep-read)](../papers/grpo-for-vla-heads.md) ·
[sim100 results](2026-08-12-sim100-results.md) ·
[spot20 v3](2026-08-12-sim-spot20-v3-results.md) ·
[parallel-rollouts pre-reg](2026-08-12-prereg-sim-parallel-rollouts.md)
