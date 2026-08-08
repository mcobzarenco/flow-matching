# Self-Certainty: scoring open-ended generations without a judge

*Lit slice 2026-08-08 ~03:1xZ, read to settle the one design cell the
#6 rung-(b) pre-reg could not freeze from what we had banked: if you
sample N candidate subgoal texts, what picks the winner — with no
trained verifier, no reward model, and no oracle? Source:
"Scalable Best-of-N Selection for Large Language Models via
Self-Certainty" ([2502.18581](https://arxiv.org/abs/2502.18581),
NeurIPS 2025, UC Berkeley). Cross-references:
[test-time selection for VLAs](test-time-selection.md) (MG-Select —
the action-side sibling already banked for #19) and
[runtime plan verification](runtime-plan-verification.md) (the
escalation shapes that made a scorer necessary at all).*

## Why this paper, right now

Rung (a) of the self-subgoal probe
([results](../posts/2026-08-08-selfsubgoal-results.md)) left a very
specific hole. The trained `[subgoal|…]` slot transmits −0.29 chunk
MAE when fed TRUE segment labels, but the model's own greedy subgoal
recovers almost none of it (−0.018, CI spans zero), and stage 1
located the failure in single-frame *phase estimation* — the model
plans a valid step of the task, just often the wrong one. The
cheapest published attack is candidate-subgoal *selection* (VINE's
width scaling, banked on the
[runtime-plan-verification page](runtime-plan-verification.md)):
decode N subgoals, condition on the best. But VINE's selector is a
value function trained on failure-labeled data we don't have, and
Do-What-You-Say's is outcome simulation we also don't have. The
rung-(b) pre-reg needs a scorer that exists *now*, is frozen before
data, and provably can't leak the oracle. This paper is the current
best published answer to exactly that question, at the text level
where our candidates live.

## The contribution

Best-of-N needs a ranking signal. Reward models work but cost a
second large model per call; majority voting (self-consistency)
needs answers that can be *counted* — it collapses on open-ended
generation, where no two samples are string-identical. The paper's
metric, **self-certainty**, is computed from the token
distributions the model already produced while generating:

> self-certainty = −(1/nV) Σᵢ Σⱼ log(V · p(j | x, y<ᵢ))

— the mean KL divergence of each step's next-token distribution
from the *uniform* distribution over the vocabulary, averaged over
the n generated tokens. A peaked distribution scores high, a flat
one low. Length-normalized by construction, zero extra forward
passes (the numbers fall out of the sampling pass), and defined for
any output — including one-line imperative subgoal clauses.

For selection they use it two ways: pick the argmax directly, or
combine with voting via **Borda ranks** (rank candidates by
self-certainty, weight vote r by (N−r+1)^p) when answers are
countable. The Borda hybrid is their headline on math tasks; pure
argmax is the form that survives open-ended text.

## The experiments

Llama-3.1-8B-Instruct and Qwen-2.5 models, N up to 64. On
convergent-answer tasks (GSM8K, MATH, LiveBench-Math) the Borda
hybrid edges plain self-consistency (e.g. 63.85% vs 63.40% on MATH
at N=64) and closes much of the gap to a process reward model
without any reward-model calls. The result that matters for us is
the open-ended column: on LiveCodeBench code generation —
where majority voting has nothing to count — self-certainty beats
both greedy decoding and universal self-consistency (the
LLM-judges-its-own-samples workaround), and keeps improving as N
grows while USC degrades on smaller models. Scaling with N is
clean and monotone across benchmarks, tracking reward-model
trajectories at zero marginal cost.

Honest caveats from their own ablations: on tasks with countable
convergent answers, plain majority voting at equal N is still
competitive or better; the Borda exponent p is tuned per sample
size; and everything sits well below oracle selection — the signal
is real but far from saturating the candidate set.

## What transfers, what doesn't

**Transfers.** The metric itself, verbatim: our pass-1 subgoal
decode is a short open-ended text generation from an AR head, and
self-certainty needs only the per-step distributions we can dump
during sampling. It is length-normalized (our candidates vary from
four words to a clause), judge-free (nothing to train), and
oracle-clean (no access to the true label anywhere in the
computation). Its published edge over likelihood-style and
USC-style baselines is specifically on open-ended text — our case,
not the math-answer case.

**Doesn't.** Their accuracy correlation is measured on reasoning
benchmarks where confidence tracks correctness of a *derivation*.
Our failure mode is different: a phase-offset subgoal is a fluent,
high-probability string about the wrong moment of the episode.
Whether distributional confidence discriminates *phase* from a
single frame is exactly the open question — which is why rung (b)
pairs the frozen scorer with a selection-*ceiling* arm (oracle-pick,
record-only) that bounds what any scorer could extract from the same
candidates. If ceiling ≫ scorer, the scorer is the gap and
heavier signals earn a look; if ceiling ≈ self-greedy, no scorer
can save selection at this width and the family closes cheaply.

**The named heavier sibling.** MG-Select
([test-time-selection page](test-time-selection.md)) is the
condition-masked contrastive version of the same idea — score by KL
between the conditional and a condition-*masked* reference
distribution, i.e. "how much did the conditioning inform this
token". That contrast would directly penalize generically-frequent
subgoal strings (stage 1's most common string, `retract the arm to
the home pose`, is exactly the kind of prior-heavy candidate PMI
punishes). **Correction (verified against the paper, lit slice
2026-08-08 ~04:1xZ):** an earlier note here said the masked
reference was off-distribution for us; that read the prerequisite
too broadly. MG-Select's masking variants are *text*, *state*, and
*text&state* — it never masks frames — and its prerequisite is
condition-dropout training (their joint recipe drops each condition
at 10%; the ablation says bare masking still gains — 17.0→22.6 on
RoboCasa PnP-100 — but dropout training nearly doubles it, to 31.0).
For scoring candidate *subgoals*, the natural reference is the
**subgoal-masked** forward — which for us is the planner-less path,
trained at 50% dropout and literally the deployment default. The
prerequisite is *met*, not missing; only a frame-masked reference
(which the paper never uses) would be off-distribution. It stays a
named escalation (teacher-force pass-2 actions under each candidate
prompt + one masked reference — N+1 teacher-forced forwards, no
decode loop — score by the contrast, reference tempered τ=4), gated
on rung (b) showing a scorer gap worth attacking.

## What this fed

- **#6 rung (b)**: the pre-reg
  ([posted this session](../posts/2026-08-08-prereg-subgoal-draws.md))
  freezes self-certainty (argmax form, formula above) as the primary
  selection scorer, with length-normalized mean logprob and medoid
  token-F1 as record-only alternates computed offline from the same
  retained dumps — and the oracle-similarity ceiling arm as the
  scorer-independent bound.
- **#19**: nothing new required — but the per-draw logit retention
  the draws instrument already carries is exactly what
  self-certainty-style scoring of *action* draws would need, and the
  MG-Select flavor there stays the cheapest trained-dropout read on
  this list.
