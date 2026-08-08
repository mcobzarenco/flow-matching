# Conditioning channels that don't condition: shortcut reliance in instructed policies

*2026-08-08 lit slice. Two papers: **Robust Skills, Brittle Grounding**
([2602.24143](https://arxiv.org/abs/2602.24143)) and **DISC**
([2605.20856](https://arxiv.org/abs/2605.20856)). Read because this
afternoon's frame-mining read
([post](../posts/2026-08-08-framemining-aliased-frames.md)) left the
meta-report with a fact in need of a frame: the oracle-subgoal gain is
FLAT across the aliasing spectrum — the conditioning channel helps
everywhere equally rather than where the image underdetermines the
action. The literature has a name for conditioning channels that don't
do the conditioning they advertise, and diagnostics for proving it.
Fed: the `fieldcond-subgoal-meta-report` interpretation section,
[#6](../ideas/06-aux-attribution.md),
[#17](../ideas/17-new-trunks.md).*

## Robust Skills, Brittle Grounding (2602.24143)

**Contribution.** A controlled diagnostic that *decomposes* what a
single benchmark success rate conflates: can the policy execute the
motion (reach, grasp-anything) vs did it do the *instructed* thing
(instruction-conditioned success). Multi-object picking with a ladder
of placement variability, up to 100k scripted demonstrations (10M
frames), testing SmolVLA and π₀.₅.

**Experiments.** The decomposition is the story. SmolVLA at small
jitter: 90% success, ~100% reach. Widen placement and execution
*survives while selection dies*: large jitter 41% success but still
100% reach and 50% grasp-anything; full workspace randomization 2%
success. With a single object present (no selection needed) full
randomization recovers to 15% — the failure is grounding, not motor
skill. The compositional cell is brutal: hold out object–region
*pairings* (every object and every region seen individually in
training) and success goes **44% → 0%** at unchanged spatial
difficulty. And the "more data" escape hatch is closed: 10k → 100k
demos under full randomization moves SmolVLA 2% → 1% and π₀.₅ 4% → 6%.
The policies read the instruction as a *coarse region prior*, not a
per-scene referent.

**Honest limits.** Two policies, one synthetic picking domain,
scripted demos; the abstract-level conclusion ("benchmarks measure
manipulation, not following") is argued from this one family.

## DISC (2605.20856)

**Contribution.** Names the mechanism — **task-state entanglement** —
and removes it structurally. When instruction and observation share
network parameters, and scenes correlate with tasks in the data (they
always do), the net learns scene→action shortcuts that bypass language
entirely: their demos show Octo approaching the microwave when told to
fetch a white bowl, π₀.₅ skipping instructed stove-activation because
the scene context matches a related task. DISC's fix: a two-stage
hypernetwork maps the instruction to the *parameters* of a compact
observation-only policy (coarse generation + learned iterative
refinement, all feed-forward). The instruction cannot leak through the
observation pathway because there is no shared pathway.

**Experiments.** LIBERO-90 from scratch 94.3% vs the best entangled
baseline 86.6%, with the margin growing with task complexity (+0.9
easy → +8.4 long-horizon); beats pretrained π₀ (91.6%) without
pretraining. The diagnostic that matters for us: a real-robot
**combinatorial benchmark** — 9 tasks as 3 objects × 3 containers in
an *identical visual scene*, so behavior variation can only come from
language. DISC 86.4%, Octo 78.5% (failing by coarse scene→action
association), plain hypernetwork 18.5%. Paraphrase-robustness (50
rephrasings per task) holds at 85.4%. Limits: heavier training, a
compact target policy that costs fine placement precision on one task
(53.3% vs 96.7%), and under-specified instructions surface uncertainty
instead of being silently resolved by the scene — which the authors
frame, fairly, as a feature.

## What transfers to us, and what doesn't

**1. The meta-report gets its interpretive frame (direct feed).** Our
concentration null — Δ_oracle flat across aliasing, ~zero only on
fully-determined frames — is the *offline signature* of what these
papers measure online: a conditioning channel consumed as a **coarse
prior** rather than a per-frame referent. Brittle Grounding's
decomposition (execution vs instructed-selection) is structurally our
decomposition (pooled gain vs aliased-frame gain); their region-prior
finding is our "uniform style/phase prior" reading. The report can now
say: the −0.29 oracle-subgoal gain behaves like the coarse-prior
regime the diagnostic literature documents, and the aliased-frame
error floor (+29%) is exactly the part a prior cannot fix — DSSP says
only extra *information* moves it, and these papers say the
information must arrive through a channel the policy can't shortcut
around.

**2. A cheap offline leakage read exists for our stack (candidate,
not queued).** Their strongest diagnostics are counterfactual
conditioning at fixed scene (DISC's combinatorial bench; the held-out
pairings). Our panel apparatus already does forced-outcome
counterfactuals (Q3); the subgoal analog — score frames under a
*wrong-episode* subgoal and measure prediction sensitivity vs the
true-subgoal pass — would quantify how much the slot's content (vs its
mere presence) moves the action. Rung (a) already hinted: the same
text moved through the suffix channel scored +0.043 *worse* than the
slot, and dropped-vs-present is −0.29 — presence is worth far more
than placement. A swap read closes the triangle (presence vs content
vs channel) for ~1 panel pass; it belongs in the meta-report's
open-questions, gated on its own pre-reg if it graduates.

**3. Entanglement is an argument in the #4/#17 design space, not a
recipe.** DISC's hypernetwork is far from our decoder-stack reality
(and its precision regression on fine placement is disqualifying for
a manipulation trunk). What transfers is the principle now backed by
numbers: conditioning delivered through a *separate structural path*
grounds better than conditioning mixed into shared tokens — worth one
line when the attach-screen seam variants are debated, not a new arm.

**4. The scaling null is a prior worth banking.** 10k → 100k demos
buying ~nothing under randomization is the cleanest published "more
data doesn't fix grounding" cell — relevant whenever a fieldgen or
subgoal shortfall tempts a bigger-data response.

**Doesn't transfer:** all success rates (sim/lab picking vs our
offline teleop-corpus MAE); DISC's from-scratch training regime (we
warm-start VLM trunks precisely for the semantics entanglement
erodes); their instruction-selection failure mode in its pure form —
our panel conditions on *episode-true* fields, so we measure prior-vs-
referent, never wrong-object grasping.

## Where it fed

- `fieldcond-subgoal-meta-report`: the interpretation section's
  external frame — the flat gain = coarse-prior consumption, the
  documented failure family; plus the presence/content/channel
  triangle as the open-questions candidate.
- [#6 aux attribution](../ideas/06-aux-attribution.md): dated hook —
  the subgoal-swap sensitivity read named as the missing cell after
  today's concentration null.
- [#17 new trunks](../ideas/17-new-trunks.md): structural-decoupling
  prior for future conditioning-path debates (one line, no arm).
