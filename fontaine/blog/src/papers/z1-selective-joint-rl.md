# Z-1: a production RL recipe that unfreezes the trunk only when diagnostics say so

*Read 2026-08-09 (lit slice `lit-radar-fresh-sweep-0810`, priority 2:
the fjoint sequencing call). Paper:
[2606.31846](https://arxiv.org/abs/2606.31846) — "Z-1: Efficient
Reinforcement Learning for Vision-Language-Action Models" (Cao, Chen,
Li, Wang, Peng, Li).*

**The paper in plain words.** After a robot policy has been trained
by imitating demonstrations, it still makes mistakes the
demonstrations never showed it how to recover from. Reinforcement
learning — letting the policy try tasks, rewarding successes — is the
standard fix, but it is expensive and unstable on big
vision-language-action models. Z-1 is a bag of engineering tricks
that makes it cheap enough to work: rollouts share their common
prefix instead of being recomputed, trajectories branch like a tree,
rewards gently decay to favor faster completions — and, most
interesting to us, the big vision-language brain stays **frozen** by
default while only the small action module trains, with the full
model unfrozen *only when diagnostics say the small module alone is
stalling*. On 24 simulated kitchen tasks it turns a 67% imitation
policy into an 81% one.

## What it contributes

- **An efficiency-first GRPO recipe for flow-based VLAs**: the
  deterministic flow decode is converted to a stochastic Markov chain
  (Gaussian noise injected into intermediate flow transitions) so
  per-action log-probabilities — and hence a clipped
  importance-ratio objective — are well-defined. Group-relative
  advantages, group size 8, sparse binary success reward with a
  0.998 success-aware decay that rewards finishing sooner.
- **Selective joint training (Sel-JT)**: default is action-expert-only
  on a frozen PaliGemma backbone; the backbone (including the
  vision-language prefix encoder) joins the trainable set per-task
  when three diagnostics warrant it — SFT success level, early
  expert-only GRPO progress, and failure modes seen in training
  rollouts. The chosen configuration is fixed before final
  evaluation.
- Built on π0.5 with only 1,199 public RoboCasa demonstrations — no
  private data.

## The experiments it ran

24 RoboCasa tasks, average success rate: GR00T 49.7 → GR00T N1.5
59.7 → X-WAM 79.2 → **Z-1 RL 80.6** (from its own SFT at 67.4, a
+13.2-point RL gain; wins 55 of 77 categories vs X-WAM). Largest
category gains where SFT was weakest: sink/faucet 63.2 → 94.3,
drawer 83.4 → 96.1. The Sel-JT ablation is thin: on one task
(TurnOnStove) joint training tracks above the expert-only baseline
in success rate and below it in policy loss throughout training, but
no final-number decomposition separates Sel-JT's contribution from
the other components. Simulation-only; no model-size, wall-clock, or
compute figures anywhere in the paper.

## What transfers to us

- **The fjoint sequencing call gets a fourth vote, with a new
  shape.** The published F-then-joint family
  ([LP-FT](lpft-two-phase-schedules.md),
  [APT](apt-expert-pretraining.md),
  [ActionX](actionx-rl-expert-pretraining.md)) says: converge the
  expert on a frozen trunk first, then consider unfreezing. Z-1's
  production stance sharpens the *second* half: joint training is
  not a scheduled phase but a **conditional escalation, triggered by
  measured stalls of the expert-only configuration**. That is
  exactly the shape of our fjoint rung's conditional-extension
  clause — and a caution against making the joint phase
  unconditional. Its trigger menu (baseline success level, early
  progress slope, rollout failure modes) is heuristic, but it is a
  deployed answer to "when is F alone not enough?"
- **For the #16 post-SFT menu**: the RL pole gains a
  data-efficiency datum — +13.2 points over SFT from 1,199 demos and
  sparse success rewards only, no reward engineering. The flow-SDE
  log-probability construction is the enabling primitive and is
  policy-agnostic for flow experts like ours.

## What doesn't transfer

- **Simulation-only** (RoboCasa kitchens); recovery behaviors and the
  success-decay calibration may not survive contact-rich real-rig
  noise, and the paper reports its own weakness on long-horizon
  transport and stove tasks (56.2 vs X-WAM's 80.0).
- **No compute accounting at all** — the efficiency claims are
  relative to its own ablations, unpriceable against our budget.
- The Sel-JT evidence is one task without final numbers; it
  seeds a design prior for the fjoint rung, not a measured ranking
  between frozen-only and joint.

## Which idea/arm it fed

#4 (`seam-screen`) — the fjoint rung's ledger: conditional-escalation
prior for the joint phase, banked before the owner go/no-go.
#16 (`rig-benchmark`) — post-SFT menu, RL pole: GRPO-on-flow recipe
+ demo-count datum. No gate changes; the fjoint pre-reg's frozen
reads are untouched.
