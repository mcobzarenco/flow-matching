# Label-free selection signals — scoring candidates without a trained judge

> **The papers in plain words.** Tonight our experiment showed that
> when the policy writes 8 candidate "what to do next" notes to
> itself, the set *contains* genuinely better notes than the one it
> would write greedily — but its own confidence score picks the
> wrong ones, reliably enough to make things worse. So: how do you
> pick a winner from a set of candidates when you have no trained
> judge and no ground-truth labels? These two papers give two very
> different answers. **uPRM** (language-model reasoning) says: don't
> score candidates one at a time — score the whole *batch* jointly,
> asking "where does each trajectory first go wrong?" using only the
> model's own next-token probabilities, and the joint structure
> recovers most of what a supervised judge knows. **SDN** (robot
> diffusion policies) says: don't use probabilities at all — test
> each candidate for *sensitivity to the thing that matters* (mask
> the target object out of the image; a trustworthy candidate should
> change, a hallucinating one won't) and for *physical smoothness*
> (jerk), both computable from the candidates themselves. Both beat
> naive best-of-N selection in their home domains without a single
> label.

*Lit slice 2026-08-09 ~01:1xZ, the same session as the
[rung-(b′) NO-SCORER verdict](../posts/2026-08-09-subgoal-draws-cleanlist-results.md)
that makes scorer design THE open question on #6 — this slice
targeted "selection signals that need no labels" before any scorer
rung gets drafted. Sources: "Unsupervised Process Reward Models"
([2605.10158](https://arxiv.org/abs/2605.10158)) and
"Self-Improving VLA Policies: Selected Diffusion Noise for
Spurious-Robust Action Smoothing"
([2606.14084](https://arxiv.org/abs/2606.14084)). Cross-references:
[RoVer](rover-learned-verifier.md) (the supervised-from-demos
alternative), [Self-Certainty](self-certainty.md) (the scorer that
just failed), [test-time selection](test-time-selection.md),
[sampling beyond selection](sampling-beyond-selection.md) (the
golden-ticket family SDN's stage-0 belongs to).*

## uPRM: the judge is a joint inference, not a per-candidate score

**Contribution.** Process reward models normally need step-level
human labels ("this is where the reasoning went wrong"). uPRM
replaces them with a scoring function over the LLM's own next-token
probabilities that **jointly assesses candidate first-error
positions across a batch of trajectories** — the location of the
first bad step is inferred from how probability mass behaves across
the whole set, not from any single trajectory's confidence.

**Experiments.** Three settings: (1) first-error detection on
ProcessBench — up to **+15% absolute** over LLM-as-a-Judge; (2)
best-of-N verification at test time — comparable to *supervised*
PRMs, up to +6.9% over majority voting; (3) as an RL reward — more
robust policy optimization than a supervised PRM trained on
ground-truth labels.

**What transfers.** The structural lesson lands squarely on
tonight's falsification: self-certainty scored each subgoal
candidate *independently* by its own mean logprob, and anti-selected
(fluent-but-wrong beats awkward-but-right on per-candidate
confidence). uPRM's claim is that the label-free signal only works
when it is **comparative across the candidate set** — the batch
jointly constrains where quality lives. Any next scorer rung on #6
should score the 8 candidates *as a set* (contrasts, relative
probability structure), not as 8 independent floats. It also
suggests the weak-judge labels aren't strictly necessary for a
scorer — relevant because our oracle-pick distillation data (4,298
pairs) inherits the weak judge's noise.

**What doesn't.** Domain: LLM math/reasoning with verifiable steps;
"first erroneous step" has no clean analogue in a 9-candidate
subgoal set (candidates are alternatives, not sequential steps).
The mechanics need re-derivation, not reuse — this is a
design-principle transfer, not a recipe.

## SDN revisited: its set-level half is the part we haven't used

**Correction first (the audit catch).** SDN already has a full page
— [noise-space steering III](noise-space-steering-3.md), read
2026-08-08 — and its cheap half was **already executed on our banked
stacks the same day** (`analysis__jerkpick_selector.json`): min-jerk
draw-picking is a clean NULL on the flow family (ODE draws are
uniformly smooth) and small-but-real on the AR family (oracle-gap
recovered 8.0% on the molmo2 stack, Spearman +0.55, agreement 12.1%
vs 10% null) — never approaching mean-of-N on either family. This
page does not re-bank any of that; what it adds is the *other*
stage, re-read through tonight's verdict.

**Stage 1 is a set-joint scorer, and that's the transferable part.**
SDN's grounding filter never scores a candidate in isolation: it
decodes the candidate set twice (real observation vs target-masked
observation) and scores by **kNN-density contrast between the two
sets** — a candidate is trustworthy if it sits far from how the
policy behaves when it *cannot see what matters*. That is the same
structural move as uPRM's batch-joint inference, in a different
modality: the label-free signal lives in the *relation between
decodes*, not in any single decode's confidence. MG-Select masked
text/state per-candidate; SDN shows the masked contrast working as
a set-level density, which is stronger.

**The #6 sketch this licenses.** A subgoal-scorer variant would
ask: does conditioning the action head on candidate *i* move the
action distribution away from the *masked-slot* (planner-less)
decode, toward a mode the other candidates corroborate? Candidates
whose conditioning does nothing are noise; the planner-less path
needed for the masked side already exists in our instrument
(50%-dropout training, the masked-contrast prerequisite verified on
the [progress-from-logits](progress-from-logits.md) page). Cost
shape: one extra masked decode per frame plus the K conditioned
decodes we already pay for in any selection arm — no new model, no
labels.

## Where this leaves the scorer question

The falsified SC scorer was per-candidate, probability-based, and
label-free. The escalation map, with tonight's and the banked
evidence in place: (a) **physics-side, scorer-free** — already
priced: jerk-pick recovers ~8% of the oracle gap on the AR family
(banked 08-08), nowhere near the −0.25 ceiling; not the answer
alone. (b) **supervised-from-demos**
([RoVer](rover-learned-verifier.md) — chunk-unit caveat
pre-registered as ammunition), now with 4,298 in-domain
picked-vs-oracle pairs dumped by the rung-(b′) run itself. (c)
**label-free but set-joint** (uPRM's principle; SDN's density
contrast; the masked-conditioning sketch above). The design
constraint for (b)/(c) is the same: **score the set, not the
candidate**. Any escalation still needs its own pre-reg per the
rung-(b′) close.
