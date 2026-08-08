# Pre-registration: clean-list subgoal-draws selection (#6 rung (b′))

*2026-08-08 ~17:5xZ. Immutable once posted. Idea
[#6](../ideas/06-aux-attribution.md), the escalation named by the
[rung-(b) stage-1 close](2026-08-08-subgoal-draws-stage1-close.md),
whose closing clause requires this post. The
[rung-(b) pre-reg](2026-08-08-prereg-subgoal-draws.md) is inherited
**verbatim except where this post explicitly amends it** — one
design change, one re-scoped stage-1 bar, fresh naming and cost
clauses. Zero training. The instrument delta is small and lands
oracle-gated before launch; any forced semantic deviation gets an
amendment posted before launch (the rung-(a) precedent).*

## Question

Rung (b) closed at table cost: its stage-1 bar (a) — every sampled
candidate clean on ≥ 90% of rows — failed at 20/60, because **11.5%
of T=1.0 sampled draws derail into budget-truncated multilingual
gibberish** (55/480; greedy clean 60/60; 0.885⁸ binomial arithmetic
reproduces the row rate). Everything else passed: diversity is real
(97% of rows ≥ 2 unique strings), clean candidates are
subgoal-shaped with genuine adjacent-phase alternatives, and the
frozen scorer already refuses the gibberish (self-certainty picks a
truncated candidate on 0/60 rows, median rank 9/9). Δ_bon and
Δ_ceil — the rung's actual payload — were never measured.

Rung (b′) asks the same question with the failure mode structurally
removed: **exclude budget-truncated candidates from every scorer's
candidate list**, re-adjudicate the stage-1 bars on the filtered
list, and — if they pass — run the two conditioned arms exactly as
rung (b) froze them.

## The one design change (frozen): the eligible-candidate list

Pass 1 is **unchanged**: per frame, 9 candidates — greedy
(candidate 0) plus 8 sampled at T=1.0, draws10_t1 seeding verbatim,
decoded and dumped exactly as rung (b) specified (candidates are
still recorded as decoded, unfiltered, with their `truncated`
flags — the flag is already a first-class field on every dumped
candidate).

Selection-side, every scorer — the primary self-certainty pick, the
token-F1 ceiling pick, and the record-only alternates (mean
logprob, medoid) — operates on the **eligible list**:

> eligible = all candidates with `truncated == false`;
> if that list is empty, eligible = [candidate 0] (greedy as
> decoded — the fallback row is recorded and counted).

Ties still break toward the lowest *original* index (greedy first);
picks are recorded as original-list indices. Nothing else about
selection changes: no re-phrasing, no dedup changes, no new scorer.

**Rejected alternative, reasons banked now:** moving the sampler to
nucleus/lower-T was the other named fix. Rejected because (i) it
changes the sampling distribution, so every banked stage-1 number
(11.5% derailment, 97% diversity, 65% pick≠greedy) stops
transferring and stage 1 must be re-bought on GPU; (ii) the #19 dT
table says sampled behavior is monotone in T — lower T trades away
exactly the diversity that gives width its value; (iii) the
exclusion filter preserves pass-1 byte-identity with the banked
stage-1 table, which makes stage 1 free (below). Nucleus/lower-T
remains available only to a future pre-reg.

## Written priors (computed from the banked stage-1 table, this session, before this post froze)

The stage-1 pass-1 decode is deterministic given checkpoint, plan,
frames, seeds and temperature — all unchanged — so the banked
60-row table
(`reports/analysis__subgoal_draws_stage1_table.json`) IS rung
(b′)'s stage-1 data. Re-scoring it through the frozen filter and
the committed scorer functions (`bijou.eval.subgoal_scoring`):

- **The filter is structural, not behavioral, on every observed
  pick:** exclusion changes the self-certainty pick on **0/60**
  rows and the token-F1 ceiling pick on **0/60** rows (both
  scorers land on truncated candidates 0/60 times unfiltered).
  40/60 rows carry ≥ 1 truncated candidate, so the filter binds
  on the *list* two rows in three while changing no observed pick
  — exactly the "structural version of what SC already does" the
  close post projected.
- **Filtered bars:** ≥ 1 eligible sampled candidate on **60/60**
  rows (no row lost all 8 draws; binomial expectation of an
  all-8-truncated row at 11.5% is ~3×10⁻⁸); ≥ 2 unique eligible
  strings on **57/60** rows (95%); top pooled eligible sampled
  string 23/425 (**5.4%**).
- Carried from the close post: SC pick ≠ greedy on 39/60 rows
  (65%); every inspected clean candidate subgoal-shaped and
  phase-relevant.

## Stage 1 (CPU, zero GPU): mechanical re-adjudication on the banked table

The execution lands a small re-adjudication script that recomputes
the numbers above from the banked table json through the committed
filter + scorer code, and gates stage 2 on the **re-scoped bars**:

| bar | line | banked prior |
|---|---|---|
| (a′) rows with ≥ 1 eligible **sampled** candidate | ≥ 90% | 60/60 |
| (b′) rows with ≥ 2 unique eligible strings | ≥ 50% | 57/60 |
| (c′) top pooled eligible sampled string | ≤ 50% | 5.4% |
| (d) eligible candidates subgoal-shaped (eyes, commented) | pass | pass (close post) |

Bar (a)'s original "non-truncated" clause is discharged by
construction on the filtered list; its surviving content is
non-emptiness, which is what (a′) scores. Fail on any bar → stage 2
does not run and the table is the rung-(b′) result (inherited
rule). Since the priors are computed from banked data, a failed bar
here would mean instrument breakage, not new evidence — the script
aborts loudly rather than adjudicating in that case.

## Stage 2 — inherited verbatim

Everything from the rung-(b) pre-reg's stage-2 section applies
unchanged except the candidate list and the names: same two
conditioned arms (**bon** = self-certainty pick over the eligible
list, deployment-honest primary; **ceil** = token-F1-vs-true-label
pick over the eligible list, record-only bound), same plan
(`plans/holdout_curated_v0_k4l2.json`), same checkpoint
(`bijou_arb_rcond_100k_ddp4/step_100000`), seed 0, full k4l2 panel,
same order and batch size as the banked baseline; label-less frames
render no subgoal in ceil and the SC pick in bon; both arms decode
actions on the deployment fast path with pass 2's generate list
excluding subgoal.

**Naming (fresh):** policy names and stems must be distinct from
rung (b)'s — they carry the filter (spelling is implementation's,
e.g. `_boncleansubgoal` / `_ceilcleansubgoal`), and the report
records the candidate-filter rule alongside the scorer id. The
ceil arm's oracle-informed text must never appear in a
deployment-named row (inherited oracle v).

## Frozen reads — inherited verbatim

Reads 1–6 of the rung-(b) pre-reg apply word-for-word with "the 9
candidates" read as "the eligible list": primary **Δ_bon =
chunk_mae(bon) − 5.8026** with the paired per-frame (bon − self)
read vs the banked rung-(a) self npz quoted beside it; **Δ_ceil**
on the labeled subset with the same interpretive frame (Δ_ceil ≈
Δ_self ⇒ the family closes at this width regardless of scorer;
Δ_ceil clearly below both ⇒ the Δ_ceil-to-Δ_bon gap prices the
scorer and the Δ_ceil-to-−0.290 gap prices wider generation);
scorer-agreement records (now also: per-row eligible-list size
distribution and fallback-row count); horizon decomposition;
first_mae mirrors; execution oracles (state-copy byte-match, anchor
re-pool exact, modes + scorer + filter ids in every name and
report). Seeded bootstrap CI95 (seed 0, 10,000), decode-noise floor
(−0.0008, ±0.016) quoted beside any small delta.

## Numbered expectations (banked before data)

1. Stage-1 bars (a′)–(d) pass on the banked table via the
   committed script — confidence high (they are computed above;
   failure = instrument breakage).
2. **Δ_ceil < Δ_self with CI clear of zero on the paired
   (ceil − self) read** — the width contains better-phase texts —
   confidence medium (inherited; stage-1 eyes saw real
   adjacent-phase alternatives).
3. **Δ_bon lands between Δ_self and Δ_ceil, closer to Δ_self**
   (inherited, confidence medium-low — whether distributional
   confidence discriminates phase from a single frame is still the
   rung's open number; stage 1 says SC discriminates *gibberish*,
   which is not phase).
4. Gains, where present, concentrate late-horizon; first_mae moves
   little — confidence medium (inherited).
5. The SC pick differs from the greedy string on ≥ 40% of frames
   — confidence medium (upgraded from rung (b)'s ≥ 20% low: the
   stage-1 measurement was 65%, and the filter changed no pick).
6. **Falsified if the paired (bon − self) CI95 does not lie
   entirely below zero** (inherited verbatim). The Δ_ceil read
   then adjudicates no-diversity vs no-scorer, and that
   adjudication routes the escalation exactly as rung (b) froze
   it: scorer-side (masked-contrast MG-Select form / TOPReward
   history-conditioning, pre-mapped on the escalation queue item)
   vs planner-side (HiRoC-direction SFT) vs close-the-family. Any
   escalation needs its own pre-reg citing this result. No prompt
   fishing, no post-hoc scorer promotion, no candidate
   re-phrasing, no filter-rule tuning.

## Cost & scheduling

Stage 1 is CPU-only (banked data). Stage 2: full-panel pass 1 (9
candidates off one shared prefill) + two conditioned greedy panel
decodes (banked rate 0.081 s/frame ≈ 35–50 min each) — projected
**~2.5–3.5 GPU-h**. Pre-registered ceiling **≤ 5 GPU-h** (tighter
than rung (b)'s 6: its stage-1 spend is already banked); if a
first-200-frame rate measurement projects past it, all arms drop to
the frozen q4 subset (4,301 rows, the #19 clause verbatim) and the
switch is recorded. Venue: local GPU, first quiet window **after
the noise-ladder rung-2 obligations** (stage-2 confirm + seating —
that pre-reg's launches come first in the post-close window); never
co-located with a training run's eval chain; launch via
`run_detached.sh` with a `babysit.toml` entry at launch;
first-poll util+rate check per standing rule.

## Instrument delta (to land, oracle-gated, before launch)

The eligible-list rule lands in the selection layer
(`SelectedSubgoalPolicy._pick` and the offline scorer recomputes in
the dump/read path), leaving pass-1 decode and dump bytes
untouched; the candidates dump gains the eligible flags and the
live filtered picks; the read script
(`subgoal_draws_results.py` family) gains the filter-aware
recompute. **Oracles (abort-on-red before launch):** rung (b)'s
(i)–(vi) inherited verbatim — in the draws-0 limit the eligible
list is exactly [greedy], so the bit-exact reproduction of the
rung-(a) self arm carries; plus (vii) **banked-table
pick-invariance**: re-scoring the real stage-1 table changes 0/60
SC picks and 0/60 ceiling picks (the priors above, as a regression
fixture on real data); (viii) **the filter binds structurally**: a
planted fixture whose full-list SC argmax IS a truncated candidate
must yield a different filtered pick (and same for the ceiling
scorer); (ix) **all-truncated fallback**: a planted all-9-truncated
row yields the greedy candidate as decoded, recorded as a fallback
row; (x) the stage-1 re-adjudication script reproduces every
written prior above exactly (60/60, 57/60, 23/425, 0/60 + 0/60).
