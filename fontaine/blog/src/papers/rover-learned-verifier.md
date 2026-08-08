# RoVer — a 0.2B learned verifier, and the chunk-step catch

> **The paper in plain words.** Same best-of-N setting: the policy
> proposes several candidate actions, and something has to judge
> which one to execute. Prior work used a big vision-language model
> as the judge (7B parameters — nearly a second policy's worth of
> compute). RoVer shows a tiny judge works: a 0.2B reward model with
> only 40M trained weights, taught *entirely from demonstration
> recordings* — no robot trials, no success/failure labels. The
> trick for training data: take an expert action from the demos,
> nudge it with noise, and you know for free that candidates closer
> to the expert are "better" — infinite ranked pairs from data we
> already have. At runtime it scores all candidates (plus suggests a
> direction to nudge them), and one such judge lifts success rates
> across three very different robot policies without retraining any
> of them. Their own caveat matters most to us: the judge scores one
> action at a time, and it gets shaky on policies that emit whole
> action *chunks* — which every policy of ours does.

*Lit slice 2026-08-08 ~19:0xZ (same sitting as the
[ELASTIC page](elastic-adaptive-compute.md)). Source: "RoVer: Robot
Reward Model as Test-Time Verifier for Vision-Language-Action
Models" ([2510.10975](https://arxiv.org/abs/2510.10975)). Why this
paper now: the #6 rung-(b′) run about to launch prices the
self-certainty scorer against an oracle ceiling — and if the verdict
is "the scorer is the gap", the published escalation is a *learned*
verifier. RoVer is the cheapest credible recipe for one: 0.2B
parameters, 40M trained, supervised entirely from demonstrations —
no environment, no success labels. Cross-references:
[test-time selection](test-time-selection.md),
[self-certainty](self-certainty.md),
[progress-from-logits](progress-from-logits.md).*

## The contribution

A process reward model (PRM) for best-of-N action selection that is
small enough to be an accessory rather than a second policy: GR-1
initialization, frozen CLIP-text + MAE-vision encoders, 40M
trainable parameters. Given (observation, language, candidate
action) it outputs a scalar reward **and a 6D improvement
direction** in action space. The training signal needs only expert
demos: sample an anchor action by Gaussian-perturbing the expert
one, take the anchor→expert vector as the ground-truth direction,
then construct better/worse action pairs on opposite sides of the
orthogonal hyperplane — Bradley–Terry preference loss on the pairs
plus a cosine loss on the direction. Two deployment tricks: the
predicted direction turns N policy proposals into N+M candidates by
sampling *along the predicted improvement direction* rather than
isotropically, and perception features are computed once per control
step and cached across all candidates — per-candidate cost settles
at ~6 ms, ~7× cheaper at width 1000 than naive re-encoding.

## The experiments

CALVIN ABC→D with three deliberately different frozen base policies
— GR-1 (autoregressive), Dita (diffusion transformer), MoDE
(mixture-of-experts denoiser, chunked output) — plus a real dual-arm
Dobot. One verifier, no per-policy retraining: GR-1 average chain
length 3.19→3.33 (SR@5 +17.4% relative), Dita 3.61→3.84, MoDE
4.01→4.12 (weakest gain — see below). Real robot: Diffusion Policy
72.9%→88.6% average success, with the gains concentrated in
unseen-object/position conditions. Directed expansion beats
isotropic Gaussian expansion at matched candidate budgets. The PRM
trained on 20% of the CALVIN split. No head-to-head with RoboMonkey
(they note its verifier rides a 7B backbone, ~18× larger).

## What transfers, what doesn't

**Transfers.** Three things. (1) The **anchor-centered pair
construction** is fully offline and label-free in exactly our sense
— we could mint preference pairs from our training corpus by
perturbing expert action chunks, no rollouts, no success labels.
As an escalation for a "scorer is the gap" verdict on #6 it is the
counterpart recipe: where self-certainty is verifier-free scoring of
*text* candidates, RoVer is a trained scorer for *action*
candidates; both live at the cheap end of the
[test-time-selection](test-time-selection.md) menu. (2) The
**perception cache** is our shared-prefill trick, independently
converged on — corroboration that amortizing the encoder across
candidates is where the width budget hides. (3) The stated
limitation is the load-bearing part for us: **chunk–step mismatch**.
Their PRM scores single steps, MoDE emits chunks, and MoDE is where
gains go unstable — and *every* policy of ours emits chunks. Any
learned-verifier arm we pre-register has to score the chunk as the
unit (or aggregate per-step scores with the aggregation frozen in
the pre-reg), not inherit the per-step default.

**Doesn't transfer.** The 6D direction head assumes a flat
end-effector action space; our action space is joint-space chunks
through a flow head, so "expand along the improvement direction"
would need to act in noise space to stay on-manifold — at which
point it stops being RoVer and becomes the DSRL/noise-steering lane
of #1 (the ladder already owns that). Expert-proximity supervision
is also a proxy the authors flag themselves: it scores "close to the
demo", not "will succeed", which collides with multimodal demos —
the same mode-collapse worry our draws-diversity bars exist to
watch.

## What it fed

- **#6 aux attribution / subgoal draws** — the escalation map gains
  a priced rung: if rung (b′) reads "scorer is the gap", a
  RoVer-style 40M-trainable PRM (chunk-scored, offline
  anchor-centered pairs from our own demos) is the published recipe
  BEFORE any environment-labeled verifier; the chunk–step caveat is
  pre-registered ammunition.
- **#19 AR sampled draws** — the verifier flavor column on the
  best-of-10 ceiling gate gets a concrete small-model entry
  (RoboMonkey's 7B verifier was the expensive one).
- **#1 noise-draw ensembling** — directed (non-isotropic) candidate
  expansion lands in noise space for us; noted on the ladder as
  prior art pointing the same way as LAFM/DSRL, not a new rung.
