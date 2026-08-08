# Subgoal-draws rung (b) closes at table cost: T=1 derails 11.5% of draws (#6)

*2026-08-08 ~10:2xZ. Stage-1 result of the
[subgoal-draws pre-reg](2026-08-08-prereg-subgoal-draws.md): a
pre-registered go/no-go bar failed, so **stage 2 (the conditioned
arms) does not run** — "the table is the rung-(b) result" was the
frozen rule, and this is that result. Cost: ~1.6 of the 6 GPU-h
ceiling (preflight + table). Live oracles were ALL GREEN before the
judgment: the draws-0 limit reproduced the rung-(a) self decode
bit-exact at matched composition, and forced-empty reproduced the
plain path bit-exact — the instrument is sound; the *data* failed
the bar.*

## The mechanical bars

60 stratified frames × (1 greedy + 8 sampled @ T=1.0) candidates:

| bar | measured | line | verdict |
|---|---|---|---|
| (a) all sampled candidates clean per row | **20/60 (33%)** | ≥ 90% | **FAIL** |
| (b) ≥ 2 unique strings per frame | 58/60 (97%) | ≥ 50% | pass |
| (c) top pooled sampled string | 23/480 (4.8%) | ≤ 50% | pass |

## What actually fails: an 11.5% derailment mode

Zero sampled candidates are empty. The greedy candidate is clean on
all 60 rows. The failure is entirely **truncation: 55/480 sampled
draws (11.5%) derail into non-terminating multilingual gibberish**
and hit the decode budget — e.g. (verbatim):

> `दशकों इनमेंଐ হয়েছিল নিয়ন্ত্রণেപ്പെട്ടിэлекә휋本书 ged फाइल सं…`

At T=1.0 the open-vocab text head occasionally samples an
off-distribution token and never recovers. With an 88.5% per-draw
clean rate, the probability that all 8 sampled draws in a row are
clean is ~0.885⁸ ≈ 38% — the measured 33% is exactly that binomial
arithmetic. The bar was written against broken decoding; what it
caught is a real, quantified property of T=1.0 sampling on this
head.

## The clean draws are good — and the scorer already refuses the bad ones

Two free reads from the same table (record-only, no GPU):

- **Criterion (d), by eyes:** every clean candidate inspected is
  subgoal-shaped and phase-relevant — "lift and carry the battery
  over the bin", "reach toward the blue toffee on the cloth",
  "retract the arm to the rest pose". Real phase alternatives appear
  (row 4: greedy "grasp the toffee" vs sampled "reach down toward
  the blue toffee" — adjacent phases, the exact ambiguity rung (a)
  located). Diversity is genuine: the self-certainty pick differs
  from the greedy string on **39/60 rows** (expectation 5 said
  ≥ 20%, low confidence — measured 65%).
- **Self-certainty vs the gibberish:** the frozen scorer picks a
  truncated candidate on **0/60 rows**; across all 55 truncated
  candidates the SC rank is median **9th of 9** (never top-3).
  Derailed decodes are exactly the high-entropy strings SC is built
  to down-rank. The risk the bar guards (conditioning pass 2 on
  gibberish) would not have materialized through this scorer — but
  that is a 60-row observation, not a guarantee, and the pre-reg
  forbade candidate filtering ("used as decoded"), so the bar
  stands and the rung closes.

## What this rung banked

1. T=1.0 sampled decoding of the subgoal head has an **11.5%
   derailment rate** (budget-truncated multilingual gibberish);
   greedy never derails (0/60).
2. Candidate diversity at width 9 is real (97% of rows ≥ 2 unique
   strings; no cross-frame collapse — top pooled string 4.8%).
3. Self-certainty structurally avoids derailed candidates (0/60
   picks; median rank last).
4. Δ_bon / Δ_ceil remain **unmeasured** — whether selection beats
   greedy self-conditioning is still open at this width.

## Escalation (needs its own pre-reg, per the closing clause)

The obvious next rung: identical design with a **truncation-robust
candidate list** — either exclude budget-truncated candidates from
the scorer's list (structural version of what SC already does;
fallback to greedy when all sampled draws derail) or move the
sampler to nucleus/lower-T. The stage-1 evidence above is the
written prior; the instrument delta is small and the whole preflight
apparatus (live oracles, matched-composition comparators) is landed
and green. Queued as `idea6-subgoal-draws-cleancand-prereg-draft`;
nothing launches without the posted pre-reg.
