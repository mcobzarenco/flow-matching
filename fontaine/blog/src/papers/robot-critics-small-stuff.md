# Robot Critics that Sweat the Small Stuff: the trained pole of the selector family

*Lit slice 2026-08-09 (work session 14:1xZ, skim-to-place per the
queue item — read, placed, parked). Robot Critics
([2606.21572](https://arxiv.org/abs/2606.21572)). Fed #19/#6 (the
selector ledger's trained pole, priced) — sibling of
[RoVer](rover-learned-verifier.md) and the
[Q-guided flow critic](qguided-flow-critic.md).*

## The paper in plain words

If a robot policy proposes several candidate actions, something has
to judge which one to execute. Off-the-shelf vision-language models
are bad judges of the *small* things that decide manipulation — a
gripper a centimeter off, a block not quite seated. This paper
fine-tunes a VLM into a critic using pairs of success and failure
snapshots from the policy's own rollouts, so it learns exactly those
small visual differences, and adds an action-conditioned video
model that imagines the outcome of each candidate before the critic
judges it. Executing the critic's pick improves success by ~11
points on their real-robot tasks and ~6 in simulation.

## Placement, and why it parks

This is the **trained-critic pole** of the selection family we have
now measured from the free end: our #6/#19 arc showed the
*label-free* scorer family anti-selects or nulls on our stacks
(masked-contrast rung (c), jerk-pick/SDN), and the
[selection-ceiling reads](../ideas/19-ar-sampled-draws.md) bound
what ANY selector could buy — real but small for AR draws, ~null
for flow fresh-noise. Robot Critics is evidence the trained pole
works where the free pole fails — consistent with RoVer, and with
its price tag stated plainly:

- **Supervision we don't have**: pairwise success/failure labels
  from *policy rollouts* — closed-loop artifacts. Our stack is
  offline; rollout labels arrive only with the #16 rig bench.
- **A second model we don't have**: the action-conditioned video
  predictor doing the imagining. That is a bigger build than the
  selector it serves.
- **A payoff our ceiling reads cap**: +11% real-world is against
  policies with selection headroom. Our measured oracle gaps say
  the headroom on our decodes is small (AR) to absent (flow) —
  the ceiling read is exactly the number that says whether this
  machinery could ever pay before building any of it.

## Verdict

Placed and parked: the trained-critic pole stays priced-not-built
until (a) #16 exists and produces rollout labels, and (b) a ceiling
read on the deployment-relevant decode shows headroom worth a
trained judge. No idea page changes rank; the selector ledger gains
one more data point that *learning* the judge is what makes judging
work.
