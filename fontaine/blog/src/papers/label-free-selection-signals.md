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

## SDN: masked-contrast + jerk — two signals, zero labels, zero training

**Contribution.** Treat initial diffusion noise as a controllable
input: per step, draw N=12 candidates, then filter in two stages.
**Stage 1 (visual grounding):** decode candidates twice — from the
real observation and from an *object-masked* observation (target
zeroed out); score each real candidate by kNN-density distance to
the masked set minus distance to the real set. Candidates that look
the same whether or not the object is visible are hallucinating
their confidence — drop them. **Stage 2 (kinematics):** of the
survivors, execute the minimum-jerk candidate (RMS of third-order
finite differences over the chunk). Training-free, no external VLM.

**Experiments.** π0 and GR00T N1.6 on SIMPLER (+5.1–8.7% average
success over nine tasks), GR00T N1.5 on a real ALOHA
(+18.3 pts absolute, 48.3% vs 30.0%, jerk −5.2%); beats CFG / WNG /
ACG and Policy Contrastive Decoding while running 1.53× faster than
PCD (zero-masking instead of inpainting; 245 ms/step on an
RTX 4070). Stated limitations: masking only catches object-level
hallucination (not lighting/texture/distractor failures), and the
fixed N=12 doesn't adapt to task difficulty — they name adaptive
candidate allocation as future work, which is exactly the
ELASTIC-flavored rung-3 candidate already on #1's board.

**What transfers — three concrete hooks.**
1. **A jerk selector is testable tonight for free.** We hold banked
   `--dump-draws` npzs (10 draws/frame, molmo2 endpoint + AR-100k
   q4): rank draws by chunk JerkRMS offline and read the selection
   delta against the banked best-of-10 ceiling — a CPU-only,
   record-only read that prices a zero-cost selector before anyone
   builds a learned one. (#19/#1.)
2. **Masked-contrast is now precedented for *frames*, not just
   text/state.** MG-Select masked text/state; SDN masks the visual
   target and uses a *set-level density contrast* rather than a
   per-candidate score — the same "score the set jointly" shape uPRM
   argues for. A subgoal-scorer variant would ask: does conditioning
   on this candidate *change* the action distribution vs a masked
   subgoal slot? Candidates whose conditioning does nothing are
   noise; candidates that move the actions toward the
   subgoal-consistent mode are live. (#6 scorer rung.)
3. **Per-step noise selection is the dynamic cousin of golden
   tickets** — our tickets are searched once and frozen; SDN
   re-selects every step from intrinsic signals. If ticket gains and
   SDN-style gains stack is an open empirical question; their
   fixed-N limitation and our dispersion-gated-allocation rung-3
   candidate are the same idea from two directions. (#1.)

**What doesn't.** Their policies are flow/diffusion with cheap
parallel candidate decode; our AR family pays serially for width
(the #19 cost story), so per-step N=12 is a flow-family-only
pattern for us. Success-rate benchmarks ≠ our MAE panel; magnitudes
don't map. And stage 1 needs target-object masks we don't have —
our nearest usable analogue is the subgoal-slot masking already in
the instrument (the planner-less path), not image masking.

## Where this leaves the scorer question

The falsified SC scorer was per-candidate, probability-based, and
label-free. The three published escalation shapes are now: (a)
**supervised-from-demos** ([RoVer](rover-learned-verifier.md) — the
chunk-unit caveat pre-registered as ammunition); (b) **label-free
but set-joint** (uPRM's principle, SDN's density contrast); (c)
**physics-side, scorer-free** (jerk — testable on banked dumps
before any pre-reg). The cheap discriminating read is (c); the
design constraint for (a)/(b) is "score the set, not the
candidate". Any escalation still needs its own pre-reg per the
rung-(b′) close.
