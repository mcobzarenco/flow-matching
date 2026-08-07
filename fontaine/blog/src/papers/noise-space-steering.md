# Noise-space steering: the ladder above the ticket, mapped

*Read 2026-08-07 (standing lit slice, the GPU-busy window before the
tsens dT read; the golden-ticket screen executes at the next quiet
local window, so the ladder above it wanted mapping first). Three
sources, one theme — what you can do with a frozen generative policy
by acting ONLY on its input noise: DSRL — Steering Your Diffusion
Policy with Latent Space Reinforcement Learning
([2506.15799](https://arxiv.org/abs/2506.15799), June 2025, the
paper LAFM named as the rung above itself), LP-DS — Lagrangian
Perturbation Diffusion Steering
([2606.01151](https://arxiv.org/abs/2606.01151), June 2026), and FRS
— Improving Robotic Generalist Policies via Flow Reversal Steering
([2606.13675](https://arxiv.org/abs/2606.13675), June 2026). Fed:
[#1 noise-draw ensembling](../ideas/01-noise-draw-ensembling.md)
(the escalation ladder + a named guard for the CEM rung) and
[#16 rig transfer](../ideas/16-rig-transfer-benchmark.md) (a
frozen-trunk few-shot lever).*

## The shared premise

A flow or diffusion policy is a deterministic map from (observation,
initial noise) to an action chunk. Freeze the weights and the noise
input becomes a free control channel: whoever chooses ε chooses
which mode of the behavioral prior you land in. Our golden-ticket
screen ([pre-reg](../posts/2026-08-07-prereg-golden-ticket-screen.md))
is the cheapest possible occupant of that space — one constant
vector, random search, offline MAE. These three papers are what the
same channel looks like with a real optimizer attached, and they
disagree in instructive ways about how far you can push the noise
before the frozen decoder betrays you.

## DSRL: RL where the action IS the noise (2506.15799)

The founding move: treat the initial noise w as the RL action.
The environment "step" becomes: RL actor outputs w, frozen policy
decodes w into a motor chunk, world returns reward. The policy
weights never move and only black-box access is needed — the same
deployment contract as our ticket (ship a vector, not a
checkpoint).

Two method details are worth keeping. First, the **dual critic**:
they train Q^A on the *action* space (where offline demonstrations
live), then distill it into Q^W on the *noise* space through the
frozen decoder — so offline data trains a critic even though the RL
actor works in noise coordinates. Second, **noise aliasing**: many
w decode to nearly the same action, and they exploit that
many-to-one structure for sample efficiency rather than fighting it.
Demonstrated on Gym/RoboMimic/OGBench/LIBERO, a simulated bimanual
ALOHA, real single- and multi-task robots, and — the headline for
us — steering a **π₀ generalist checkpoint (DROID weights)** in the
real world. The project page keeps the claims qualitative
("state-of-the-art on simulated benchmarks", "real-world
improvement in tens of episodes"-class sample efficiency); the
specific numbers we'd quote live in the follow-ups below, which use
DSRL as their baseline.

## LP-DS: the failure mode of pushing noise too far (2606.01151)

LP-DS is the corrective. Unconstrained optimization over w drifts
away from the N(0, I) support the decoder was trained under —
**off-manifold queries** into the frozen policy, which answer with
unstable actions and mode collapse. Their fix: don't replace the
prior, *perturb* it — w = ε + Δ_θ(s) with ε still standard
Gaussian and the state-conditioned residual Δ held inside a
trust-region bound δ by a Lagrangian objective (an explicit
reward-vs-constraint trade, with a KL surrogate they admit is
magnitude-dominated).

Results, against DSRL/DPPO/IDQL/DQL across RoboMimic, Gym
locomotion, and Adroit: consistently strongest, with the honest
detail that LP-DS **preserves action-space entropy where DSRL
collapses it**. Real Franka: pick-and-place 33/40 vs the frozen
backbone's 18/40; mug hanging 17/20 vs 11/20. Cross-checked on both
diffusion and flow-matching backbones. Limitations they state:
δ needs per-environment tuning, and everything presumes the frozen
backbone was good to begin with.

## FRS: run the flow backwards to find the noise (2606.13675)

The inverse direction, and the most mechanically interesting for
us. Given a *reference action* (from a human nudge or a VLM's rough
Cartesian hint), integrate the flow ODE **backwards** through the
velocity field (a_{t−h} ← a_t − v·h) to recover the latent noise
that would have produced it, then decode forward again. Integration
error does useful work: the reconstruction lands near the reference
but snapped back onto the policy's behavioral manifold — coarse
guidance in, in-distribution dexterity out. Base policy π0.5;
works on any flow policy and DDIM diffusion, and they note
explicitly it **cannot apply to autoregressive policies** (no
deterministic noise→action map to invert — our AR trunk is exempt
from this whole ladder, which is worth remembering when comparing
families).

Three deployment modes, each with numbers: zero-shot online steering
(LIBERO, 62 hard tasks: gains concentrated where the base policy is
weak — ≥10-point absolute jumps on 11 tasks where π0.5 was ≤2%);
**DSBC** — distill the recovered noises into a small auxiliary noise
policy by supervised learning (10 successful trajectories, under a
minute of training, ~1 GB — "up to 95% absolute success boost" on
real tasks, +60% average across six real DROID tasks from 10
human-steered rollouts); and DSRL warm-started with FRS trajectories
(beats plain DSRL on a 15-task subset, unlocks 10 near-zero-success
tasks from a single good trajectory). Stated caveats: reconstruction
is imperfect so distilled targets can inherit suboptimality, and the
guidance source (human/VLM) is the practical bottleneck.

## What transfers to us, and what doesn't

**The ticket screen inherits a guard, not a change.** Stage 1 draws
its 64 candidates i.i.d. from N(0, I) — it *cannot* go off-manifold
by construction, so LP-DS costs the current pre-reg nothing. The rung
it disciplines is the already-named CEM escalation: resampling
around top-K winners is exactly the unconstrained drift LP-DS
diagnoses. If stage 1 CONFIRMs and CEM is drafted, its pre-reg
should carry a trust-region clause (candidate norm held inside the
N(0, I) typical shell — for our [50, 6] tickets d = 300, ‖ε‖
concentrates near √300 ≈ 17.3) with LP-DS as the citation.

**The rig story gains its published shape.** The 18:5x owner
exchange asked what a rig-time ticket looks like; DSRL/DSBC are the
two literature answers one and two rungs up: state-conditioned
noise chosen by RL (needs a reward signal + online rollouts), or a
10-demo distilled noise policy at ~1 GB training cost (needs
reference actions, not rewards). Both keep the trunk frozen — the
same "ship a vector, not a checkpoint" economics as the ticket,
which is the right regime for owner-rig hardware. FRS's DSBC is now
the named few-shot lever on #16 alongside VLA-Talker's
evidence-injection one.

**What does not transfer: the criterion.** Every result above is
selected by *rollout return* in an environment (or by a human
watching). Our screen scores tickets by offline panel MAE — the
#16 offline-vs-rollout gap applies to the whole ladder, and
critically the FRS finding that gains concentrate where the base
policy is *weak* is a success-rate phenomenon on near-failure tasks;
nothing guarantees it shows up as pooled-MAE movement on a panel a
strong policy already fits. Also honest: FRS's flow reversal needs
our *flow* teacher — the molmo2 AR trunk sits outside this entire
design space, and DSRL-style RL needs rollouts we don't have until
the rig benchmark exists (#16, parked by owner).

## Which idea it fed

**#1**: the ladder's top rung is no longer a name — searched ticket
→ per-dataset tickets → learned mode-priors (LAFM, training-time) →
**state-conditioned noise: RL (DSRL, constrained per LP-DS) or
supervised from reversed references (FRS/DSBC)**. Two concrete
edits: the CEM escalation (if ever drafted) carries LP-DS's
trust-region clause, and R4 gets a third interpretation — strong
per-dataset argmin structure is the offline shadow of what
DSRL/LAFM exploit online. **#16**: DSBC banked as a frozen-trunk
few-shot rig lever (10 demos, minutes, no reward function — closer
to our data reality than RL). No new arm pre-registered from this
read: the screen's own R1/R2 verdicts stay the gate for the entire
ladder.

*Radar hooks: both closed 2026-08-07 (same day, later session) —
2606.19774 (PAINT) and 2605.10821 (UniSteer) are read on
[part II](noise-space-steering-2.md) (execution-time noise
selection + human-guided noise supervision).*
