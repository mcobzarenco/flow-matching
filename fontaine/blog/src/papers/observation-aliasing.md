# Observation aliasing: when the frame alone can't tell you what to do

*2026-08-08 lit slice. Two papers: **IntentVLA + AliasBench**
([2605.14712](https://arxiv.org/abs/2605.14712)) and **DSSP**
([2605.14598](https://arxiv.org/abs/2605.14598)). Read because the
owner's 13:21Z steering asked the meta-report on field/subgoal
conditioning to showcase frames "where the right thing to do might be
ambiguous just from the image — am I at the beginning of the episode
or the end?" That is not an anecdote; it has a name (observation
aliasing), a formal result, a benchmark built to isolate it, and — the
part we can use tomorrow morning — an automatic diagnostic for finding
those frames in a corpus. Fed: the `fieldcond-subgoal-meta-report`
frame-mining protocol, [#6](../ideas/06-aux-attribution.md),
[#11](../ideas/11-visual-grounding.md).*

## The problem, stated once

A frame-conditioned policy computes `action = f(image, instruction)`.
If two moments in a task produce near-identical images but demand
different continuations — bread held mid-air on the way *to* the pan
vs *from* it; a phone in transit whose destination depends on where it
came from — then no function of the current image can do better than
split the difference. The demonstrations are not noisy; the *map* is
one-to-many. Every policy we train (single-timestep prefix, 1–2
cameras) lives on the reactive side of this line.

**DSSP makes the folklore formal** (its Prop 4.2): in a POMDP with a
non-injective observation function, if the conditional mutual
information I(action; history | current observation) is positive, the
minimum achievable imitation loss of a history-conditioned policy is
*strictly* below the reactive policy's minimum. Not "helps in
practice" — a floor gap. The reactive policy's residual error on
aliased states is irreducible by more data, more epochs, or a bigger
trunk; only extra conditioning information moves the floor.

## IntentVLA / AliasBench (2605.14712)

**Contribution.** A history-conditioned VLA: a frozen VGGT-1B encodes
the last K frames into a handful of tokens (one camera token, four
register tokens), gated cross-attention fuses them into the
Qwen3-VL context, and a DiT flow-matching head generates chunks
conditioned on the fused "intent" representation. No intent labels
anywhere — the latent is shaped only by the chunk-prediction loss.
And, the part that outlives the architecture: **AliasBench**, 12
RoboTwin2 tasks in four families engineered so the current frame is
insufficient — back-and-forth (same local state recurs in different
phases — the owner's exact example), crossing-path (transit states
with hidden origin), bimanual handoffs (direction invisible at the
symmetric midpoint), multi-goal (the disambiguating cue flashed
earlier and is gone).

**Their aliasing diagnostic — the transferable instrument.** To prove
the tasks are genuinely aliased they retrieve, for each frame in an
annotated ambiguity window, its top-5 nearest neighbors in VLM
embedding space and check (a) ~50% of neighbors belong to a
*different* continuation, and (b) the cosine-distance gap between
same-intent and different-intent neighbors is tiny (<3e-3). Visually
indistinguishable, behaviorally divergent — measured, not asserted.

**Experiments.** On AliasBench with matched training data: frame-only
baseline 9.0%; naive history (concatenate +4/+8/+16 frames) tops out
at 28.1% and then OOMs; MemoryVLA 14.9% (its similarity-based memory
consolidation *averages intent-distinct neighbors* — similarity is
exactly the wrong key under aliasing, a lovely negative result);
IntentVLA 45.8%. Gains concentrate where history genuinely
disambiguates (crossing-path 74.7%) and stay modest where fine
geometry dominates (bimanual 17.0%). On non-aliased benchmarks the
method is merely competitive (SimplerEnv +7.6 pts, LIBERO-Long
+5.4) — and on one fine-grained task history *hurts* (Put Spoon
83.0% → 70.8%): over-attending to history dilutes current-frame
detail. They also measure inter-chunk consistency across replanning
boundaries (ICC-L2 down 17.6% mean, 21.7% at p90) — aliasing shows
up as *intent switching* between adjacent chunks, not just as worse
single predictions.

**Honest limits.** Sim benchmark, tasks hand-designed to alias,
ambiguity windows manually annotated (the NN diagnostic validates,
it does not discover); short-horizon only — sparse far-past cues are
explicitly out of scope.

## DSSP (2605.14598)

**Contribution.** The theorem above, plus an existence proof that
full-history conditioning is cheap if the encoder is right: a causal
Mamba/SSM compresses the *entire* observation stream into one context
token (kept honest by an auxiliary next-state-prediction loss), which
prefixes a diffusion action head alongside the N most recent states.
Linear time in history length; 44M params.

**Experiments.** RoboTwin 2.0 (50 bimanual tasks): 62.3% vs DP3's
55.2%, with the margin concentrated on long-horizon tasks (+21.4%
relative) — exactly where task-progress aliasing accumulates.
Ablations: removing the history encoder costs ~10% relative; Mamba
full-history beats a Transformer on the same history at half the
latency and 40% of the memory. Robustness: at observation noise
σ=0.15 DSSP holds 20.8% vs DP3's 3.2% — history is also a filter.
Admitted limit: history fixes *disambiguation*, not grasping — their
residual failures are local manipulation errors.

## What transfers to us, and what doesn't

**1. The meta-report gets a mining protocol instead of anecdotes
(direct feed, CPU-only).** The owner asked for frames where the image
alone underdetermines the action. AliasBench's diagnostic, run in
reverse, *finds* them: embed panel/episode frames (any frozen vision
tower we have on disk qualifies), retrieve nearest neighbors, and
flag frames whose close neighbors carry **divergent ground-truth
continuations** (large action-chunk distance despite small embedding
distance). Rank by divergence-over-distance and the top of the list
is precisely "start-vs-end indistinguishable" and "goal not visible
from the parked position". Then the meta-report's key chart writes
itself: **the subgoal-conditioning delta (conditioned vs
subgoal-dropped prediction, per frame) should concentrate on the
flagged frames** — IntentVLA's 9% → 45.8% says conditioning earns its
keep *on aliased states specifically*, and DSSP's theorem says
nothing else could have closed that gap. If our delta does NOT
concentrate there, the subgoal channel is doing something other than
disambiguation (style prior, dataset fingerprint) — either answer
sharpens the report. This slots into the queued
`fieldcond-subgoal-meta-report` frame-mining stage, which was already
scheduled to start in a CPU window before the fields panel lands.

**2. Our conditioning fields are intent tokens we get for free.**
IntentVLA spends a 1B encoder inferring a latent intent from history;
our corpus *ships* the intent as text — `subgoal`, `outcome`,
`progress` — and we already train with `--condition-fields` +
dropout. Their frame-only-vs-intent gap is the published ceiling for
what that channel is worth on aliased states; our
`--subgoal-dropout 0.5` is the knob their Put-Spoon regression
argues for (the policy must survive conditioning absence and not
over-rely). The aux-field probes (#6) and the fields panel are our
instruments on this exact channel — this cluster gives them their
external baseline story.

**3. A history arm is NOT the cheap next step for us.** Their naive
+4-frames baseline (the only version compatible with our
architecture today) bought 19 points of the 37-point gap at prefix
cost we can't pay — ~410 image tokens per extra frame per camera on
molmo2, on a 2.2 s step we are currently trying to *shrink*. The
literature's verdict is consistent: naive frame stacking is the
worst point on the curve (cost of history, gains of neither a
learned intent latent nor an SSM compressor). If an aliasing census
(below) ever shows a large aliased fraction in our corpus, the
entry-level arm is a *compact learned context* (IntentVLA-style few
tokens, or DSSP-style single token), pre-registered as its own
screen — not stacked frames.

**4. A free falsifiable census, someday.** The same NN mining, run
corpus-wide, yields "what fraction of our frames are aliased?" — a
number that decides whether history/memory work (idea #17/#22
adjacencies) is worth any GPU at all on this corpus. Banked as an
entry condition, not a queue item: it rides the meta-report's
mining code for free.

**Doesn't transfer:** RoboTwin2 success rates (sim, engineered
tasks, matched-data protocol — our teleop corpus has unknown and
probably much lower aliasing density); VGGT specifics; DSSP's
point-cloud observation space; both papers' rollout-based metrics
(our panel is offline MAE — though their ICC-L2 inter-chunk
consistency metric has an offline cousin: prediction divergence
between adjacent-frame conditioning windows, cheap on banked draws).

## Where it fed

- `fieldcond-subgoal-meta-report` (owner 13:21Z): frame-mining is
  now specified — NN-retrieval divergence mining + the
  delta-concentration chart as the report's central claim; queue
  item amended this session.
- [#6 aux attribution](../ideas/06-aux-attribution.md): dated hook —
  the conditioning-delta-on-aliased-frames read is the external
  validation shape for the subgoal channel.
- [#11 visual grounding](../ideas/11-visual-grounding.md): the
  aliasing census is a named entry condition for any
  history/memory escalation.
