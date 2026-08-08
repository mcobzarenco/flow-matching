# Pre-registration: subgoal-draws selection (#6 rung (b))

*2026-08-08 ~03:2xZ. Immutable once posted. Idea
[#6](../ideas.md), escalation rung (b) above the
[rung-(a) self-subgoal probe](2026-08-08-selfsubgoal-results.md),
whose own closing clause requires this post: every escalation needs
a new pre-reg citing that result. Design anchors: VINE's
candidate-width scaling and the selection shapes on the
[runtime-plan-verification page](../papers/runtime-plan-verification.md);
the scorer cell settled by the
[Self-Certainty read](../papers/self-certainty.md) (2502.18581),
done this session before anything here was frozen. Zero training.
The instrument does NOT exist yet: it lands oracle-gated before
launch; any forced semantic deviation gets an amendment posted
before launch (the rung-(a) precedent, twice used).*

## Question

Rung (a) banked a live conditioning channel (Δ_oracle = −0.290
[−0.331, −0.225] through the trained `[subgoal|…]` slot) that the
model's own greedy subgoal fails to exploit (Δ_self = −0.018
[−0.052, +0.026]), and located the bottleneck in single-frame phase
estimation — valid plans about the wrong step. Rung (b) asks the
cheapest published follow-up: **if the model samples N candidate
subgoals and a frozen, verifier-free scorer picks one, does
conditioning on the pick beat conditioning on the greedy subgoal?**
And — measured in the same run — **how much could ANY scorer get
from these candidates?** The second question is the rung's real
payload: it decides whether the selection family lives or dies at
this width, independent of any particular scorer's cleverness.

## Candidate set (frozen)

Per panel frame, pass 1 decodes **9 candidates**: the greedy
subgoal (candidate 0, identical in expectation to rung (a)'s
pass 1) plus **8 sampled at temperature 1.0** under the identical
planner-less prompt (`[generate|subgoal actions]` context, text
line only), the draws10_t1 seeding conventions verbatim (seed 0,
per-frame stable seeding). Width rationale, banked before data:
VINE's expansion-width scaling peaks at K=4 and MG-Select's action
draws saturate at N=4–8; 8 sampled draws sits at the top of that
band and pass-1 text decodes share one prefill, so marginal width
is nearly free relative to the conditioned decode. Candidates are
used as decoded — no re-phrasing, no filtering beyond exact-string
dedup for the scorer's candidate list. All 9 candidate strings,
their per-token distributions' summary stats, all scorer values,
and the pick are retained machine-readable per frame (identity
triple → record), the rung-(a) dump convention.

## Scorer (frozen): self-certainty, argmax form

The primary scorer is **self-certainty**
([2502.18581](https://arxiv.org/abs/2502.18581)): for candidate y
of length n, the mean KL divergence of each generation step's
next-token distribution from uniform,

> SC(y) = −(1/nV) Σᵢ Σⱼ log(V · p(j | x, y<ᵢ)),

computed from the distributions of the pass that produced the
candidate (greedy candidate scored from its own teacher-forced
pass-1 distributions), length-normalized by construction, no extra
forward passes, no access to the true label anywhere. Selection =
argmax over the 9 candidates; exact ties break toward the
lowest-index candidate (greedy first). Why this signal and not
likelihood or consensus, banked now: it is the published best
reward-free selector on *open-ended* text, where majority voting
has nothing to count; mean logprob and medoid similarity are
retained as **record-only alternates** computed offline from the
same dumps (agreement rates and hypothetical picks recorded; their
conditioned deltas are NOT measured — a future pre-reg may promote
one, no post-hoc promotion here). The known risk is also banked: a
phase-offset subgoal is a fluent high-confidence string, so
self-certainty may not discriminate phase — that is what the
ceiling arm prices.

## Design — two stages, gated in order

**Stage 1 — candidates table (eyes before any scalar).** For the
SAME fixed-seed 60-frame stratified sample as rung (a): all 9
candidates per frame, scorer values, the self-certainty pick, the
true segment label. Read row-by-row and commented in the results
post. **Pre-registered go/no-go for stage 2:** (a) sampled
candidates non-empty and non-truncated on ≥ 90% of rows; (b)
**diversity exists** — ≥ 2 unique candidate strings on ≥ 50% of
frames (if sampling at T=1 collapses onto the greedy string,
selection is vacuous at this width and the rung CLOSES here, at
table cost — that is a result, not a failure); (c) no single string
> 50% of all sampled candidates pooled across frames
(cross-frame collapse check); (d) candidates are subgoal-shaped
(the rung-(a) bar), commented where failing. Fail on any → stage 2
does not run; the table is the rung-(b) result.

**Stage 2 — two conditioned arms**, identical rows via the shared
plan `plans/holdout_curated_v0_k4l2.json`, AR-100k
(`bijou_arb_rcond_100k_ddp4/step_100000`), seed 0, full k4l2
panel — the same plan, order and batch size as the banked baseline
(the amendment-1 composition lesson, inherited as a design
constraint):

| arm | conditioning text | selector | role |
|---|---|---|---|
| **bon-subgoal** | self-certainty pick of the 9 | frozen scorer, no oracle access | primary, deployment-honest |
| **ceil-subgoal** | candidate maximizing token-F1 vs the TRUE segment label | oracle similarity, record-only | scorer-independent bound at this width |

Both arms decode actions on the deployment fast path
(`[generate|actions]`, pass 2 never requests subgoal generation —
the trained anti-copy constraint, inherited). Ceiling-arm selector,
frozen exactly: token-level F1 between candidate and true label,
lowercase, whitespace tokenization; ties break toward the
lowest-index candidate. Label-less frames (12 of 25,800) render no
subgoal in the ceil arm (the rung-(a) oracle-arm convention) and
the self-certainty pick in the bon arm (deployment-honest: a rig
has no labels). The ceil arm's oracle-informed text must never
appear in a deployment-named row — policy names carry the modes
(`_bonsubgoal`, `_ceilsubgoal`), reports carry the scorer id.

## Frozen reads

Paired per-row, seeded bootstrap 95% CI (seed 0, 10,000
resamples), the rung-(a) assembly conventions; banked baseline
5.8026 / 2.1431 re-pooled from its npz inside the execution
oracles, never re-run. The rung-(a) self arm npz
(`…_panel_k4l2_selfsubgoal`) is a second frozen comparator — same
plan, same composition, paired per-frame is valid with the
decode-noise floor (−0.0008, per-frame CI ±0.016) quoted beside
any small delta.

1. **Primary: Δ_bon = chunk_mae(bon) − 5.8026**, all core frames,
   with the **head-to-head paired read (bon − self) per-frame vs
   the banked rung-(a) self npz** quoted beside it — the rung's
   pass/fail number.
2. **Bound: Δ_ceil = chunk_mae(ceil) − 5.8026** on the labeled
   subset (≈ the panel). Interpretive frame, fixed now:
   Δ_ceil bounds every scorer at this width. Δ_ceil ≈ Δ_self
   (paired ceil − self CI including 0) ⇒ the candidate sets do not
   contain usefully better-phase texts ⇒ the selection family
   closes at N=9/T=1 regardless of scorer; Δ_ceil clearly below
   both ⇒ the gap between Δ_ceil and Δ_bon prices the scorer, and
   the gap between Δ_ceil and Δ_oracle (−0.290) prices what wider
   generation would still be missing.
3. **Scorer-agreement records**: fraction of frames where the
   self-certainty pick ≠ greedy string; agreement of the
   record-only alternates (mean logprob, medoid token-F1) with the
   primary pick and with the ceil pick; per-frame unique-candidate
   count distribution.
4. **Horizon decomposition**: per-step-in-horizon MAE curves for
   both arms from the dumped npz, the rung-(a) conventions — the
   slot's gain is 6× late-horizon, so any recovered fraction
   should be too.
5. first_mae mirrors of 1–2.
6. Execution oracles (abort): state-copy / state-copy-norm rows
   byte-match banked values; anchor re-pool exact; modes and
   scorer ids in every name and report; the label-less-row decode
   count recorded descriptively (amendment-1 form).

## Numbered expectations (banked before data)

1. **Stage-1 diversity exists**: ≥ 2 unique candidates on most
   frames, and on frames matching stage-1's phase-offset pattern
   the true phase appears among the 9 in a nontrivial fraction —
   confidence medium.
2. **Δ_ceil < Δ_self with CI clear of zero on the paired
   (ceil − self) read** — the width contains better-phase texts —
   confidence medium.
3. **Δ_bon lands between Δ_self and Δ_ceil, closer to Δ_self**
   (confidence medium-low; this is the rung's genuinely open
   number — whether distributional confidence discriminates phase
   from a single frame).
4. Gains, where present, concentrate late-horizon; first_mae moves
   little — confidence medium.
5. The self-certainty pick differs from the greedy string on
   ≥ 20% of frames (record-only) — confidence low.
6. **Falsified if the paired (bon − self) CI95 does not lie
   entirely below zero**: verifier-free selection at this width
   buys nothing over greedy self-conditioning. The Δ_ceil read
   then adjudicates *why* (no diversity vs no scorer), and that
   adjudication routes the escalation: scorer-side (the
   MG-Select-style masked-contrast signal named on the
   [Self-Certainty page](../papers/self-certainty.md), or trained
   scorers) vs planner-side (HiRoC-direction SFT) vs close-the-
   family. Any escalation needs its own pre-reg citing this
   result. No prompt fishing, no post-hoc scorer promotion, no
   candidate re-phrasing.

## Cost & scheduling

Pass 1 with 9 candidates shares one prefill per frame (text lines
are ~10–20 tokens; marginal candidate cost is small against the
prefill); the two conditioned arms are each one greedy panel decode
(banked rate 0.081 s/frame ≈ 35–50 min each); stage 1 is minutes.
Projected ~2.5–3.5 GPU-h. Pre-registered ceiling **≤ 6 GPU-h
total**; if a first-200-frame rate measurement projects past it,
all arms drop to the frozen q4 subset (4,301 rows, the #19 clause
verbatim) and the switch is recorded. Venue: local GPU, first
quiet window after the #1 golden-ticket screen's R1 chain resolves
(that pre-reg's obligations come first); never co-located with a
training run's eval chain. First-poll util+rate check per standing
rule.

## Instrument (to land, oracle-gated, before launch)

`bijou.eval`'s selfsubgoal mode gains sampled pass-1 draws
(`--subgoal-draws 8 --subgoal-temperature 1.0` spelling is
implementation's), per-candidate distribution stats sufficient to
compute SC exactly, the two selection modes, and the machine-
readable candidate dump. **Oracles (abort-on-red before launch):**
(i) the draws-0 limit (greedy candidate only, bon mode) reproduces
the rung-(a) self arm decode bit-exact at matched composition;
(ii) forced-empty reproduces the plain path (inherited, matched
composition per amendment 1); (iii) SC and token-F1 scorers pass
exact-arithmetic fixtures incl. single-candidate and tie cases;
(iv) pass 2's generate list excludes subgoal (inherited); (v)
provenance separation — `_ceilsubgoal` never reachable from a
deployment-named entry point; (vi) conditioned prompt bytes match
the training collator's rendering (inherited).
