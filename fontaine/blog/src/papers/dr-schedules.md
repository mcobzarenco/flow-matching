# Scheduling the randomization: DR as a curriculum with a success constraint

**Papers:** DORAEMON
([2311.01885](https://arxiv.org/abs/2311.01885)) · AutoDR / OpenAI
Rubik's-cube ([1910.07113](https://arxiv.org/abs/1910.07113)) ·
curriculum-coefficient DR in the embodiment-scaling-laws study
([2505.05753](https://arxiv.org/abs/2505.05753)) · frame: the
randomized-sims review ([2111.00956](https://arxiv.org/abs/2111.00956)).
**Read:** 2026-08-12, sim lit lane (`lit-sim-improvement-levers`,
owner-called 09:23Z). **Fed:** the training-in-sim arm that the
live GRPO signal probe may open — this page is the design citation
for *how wide to randomize* if we ever optimize a policy inside the
sim; complements [sim-as-eval](sim-as-eval.md) (which is about
*evaluating* in sim, where randomization is the enemy and matching
is the point).

**The problem in plain words.** Domain randomization is the standard
insurance policy for training robots in simulation: instead of one
simulated world, train across thousands of slightly different ones —
heavier boxes, slipperier tables, weaker motors — so the real world
looks like just one more variant. But *how much* variety, and *when*?
Too little and the policy memorizes the simulator; too much from
step one and it never learns anything (or learns a timid,
lowest-common-denominator shuffle). The papers here converge on the
same answer from three directions: **randomization width should be a
schedule, not a setting — start narrow, widen as fast as the policy
can tolerate, and let a measured success rate be the throttle.**
This matters to us conditionally: today our simulator is an
*evaluation mirror* (where we deliberately randomize nothing and
match everything), but the GRPO signal probe running as this page is
written is asking whether we will soon *train* in it. The moment the
answer is yes, DR width becomes a live design choice.

## AutoDR: the original throttle

OpenAI's Rubik's-cube system (1910.07113) introduced Automatic
Domain Randomization: every randomized parameter starts at a single
calibrated value (zero width), and the *boundaries* of each uniform
range are pushed outward whenever performance sampled at the current
boundary clears a threshold, pulled inward when it falls below a
lower one. The policy is always training at the edge of what it can
survive — a curriculum the policy writes for itself. It produced the
famous result (in-hand cube rotation transferring to a real
Shadow Hand), but the mechanism has two structural limits: ranges
are uniform by construction, and only one dimension's boundary is
probed and moved at a time, which gets slow as the parameter count
grows.

## DORAEMON: the same idea as a constrained optimization

DORAEMON (2311.01885) recasts the schedule cleanly: maximize the
**entropy** of the sampling distribution over dynamics parameters,
subject to the policy's in-distribution success probability staying
above a floor — maxφ ℋ(νφ) s.t. 𝒢(θ,φ) ≥ α — with a KL trust region
between successive distributions, importance sampling to reuse
rollouts across the update, and a backup problem that backtracks
when the constraint is violated. Beta distributions per parameter,
all dimensions updated jointly — both AutoDR limits removed.

The measured-transfer receipt is a Franka Panda pushing a 10 cm box
with a deliberately shifted, unknown center of mass, 17 randomized
dynamics parameters (mass, friction, joint damping, CoM, ...):
**60% real success vs AutoDR's 26.7%** (sim: 66.6% vs 30.5%), with
the trained policy exhibiting an emergent information-seeking nudge
— touch the box lightly to reveal the CoM, then push. The ablations
are as useful as the headline: the success floor α is the whole
game — α = 0.9 yields overly conservative policies that generalize
poorly, α = 0.5 is their sweet spot (the policy must be *allowed to
fail* on a good fraction of sampled worlds or the distribution never
widens); removing the backup mechanism silently breaks the
constraint; and near maximum entropy, performance degrades and
recovers slowly, with a stated failure mode of collapsing back to an
"easy" region and forgetting previously-mastered dynamics.

## The cheap version: one scalar curriculum

The embodiment-scaling-laws study (2505.05753, locomotion) needs DR
at scale without machinery: a single curriculum coefficient
c ∈ [0, 1] multiplies *all* randomization ranges, starts at 0, and
moves ±0.01 per episode on a simple performance test (completed
without falling, low tracking error → widen; otherwise narrow).
One line of logic, most of the benefit of ADR-style scheduling. The
review (2111.00956) frames all of these as the same object — a
curriculum over domain parameters — and adds the field's standing
warning that static, hand-guessed wide ranges are the configuration
most likely to produce the timid lowest-common-denominator policy.

## What transfers, what doesn't

**Transfers — the shape of the recipe, the day we train in sim.** If
the GRPO probe finds signal and a sim-RL arm launches, the DR
question arrives immediately (our sysid'd dynamics are one point
estimate; GRPO will exploit any inaccuracy it can find — reward
hacking against sim physics is the known failure mode). The recipe
this slice fixes: **start at the sysid'd center with zero width**
(our matched sim, as-is), widen dynamics parameters on a schedule
throttled by measured success, and use the scalar-coefficient
version first — one knob multiplying conservative ranges around the
sysid values (friction, boat mass/CoM, joint damping — the
parameters our own sysid actually fit, hence the ones whose
uncertainty is quantified). DORAEMON's α ≈ 0.5 finding translates
directly: the throttle metric must be one the policy can pass ~half
the time. At our 9/100 competence floor, binary success is too rare
to throttle on — progress_final_cm against a threshold is the
throttle until success rate leaves the floor. Fed to `ideas.md` as
the `dr-schedule-for-sim-rl` hook, explicitly conditional on the
probe's decision rule firing.

**Transfers — the eval/train firewall.** Randomization width > 0 is
a *training* device only. Eval rows stay at the matched center,
always — a policy scored under randomized dynamics is being scored
in a different casino every episode, and comparability across
checkpoints (our whole anchor discipline) dies. The same firewall
the compositing slice found for shadows
([composite-shadows](composite-shadows.md)): randomize in training,
match in eval.

**Doesn't transfer (yet).** DORAEMON's full machinery — Beta
distributions, importance-sampled constraint estimates, backup
problem — is sized for 17-dimensional dynamics spaces and
from-scratch RL; our conditional arm is GRPO fine-tuning of an
already-competent policy over a handful of sysid'd parameters, where
the one-scalar curriculum almost certainly suffices and adds no new
failure modes. Visual randomization schedules are likewise not our
lever: our visual channel is matched by construction (composites),
and un-matching it for training would first require the
GreenAug-style split argued on the shadows page. And none of this
touches the current live question — the probe itself runs on the
matched sim and stays that way by pre-registration.
