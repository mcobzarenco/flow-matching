# Noise-ladder rung 2: per-dataset tickets FALSIFIED out-of-sample

*2026-08-08 19:2x–19:4xZ. Stage-2 confirm eval ran per the
[pre-reg](2026-08-08-prereg-noise-ladder-perdataset.md) (launched
18:34Z after the owner cleared the credit-cap wait, ~0.83 GPU-h of
the ≤ 1.5 gate; frozen reads chained at rc=0, adjudication banked in
`analysis__noise_ladder_rung2.json`). The pre-registered primary
read fired its falsifier — and not marginally: the effect is
significantly in the WRONG direction. The per-dataset rung of the
noise-structure ladder is closed.*

## The one number

**Read 1 (primary, frozen):** Δ_route = routed − ticket 33, pooled
over the **qualifying complement core rows** (6,014 rows the map
never saw during selection), dataset-clustered bootstrap CI95, seed
0. Pass rule: CI95 entirely below 0.

**Result: +0.129, CI95 [+0.060, +0.205] — entirely *above* zero.**
Routing each qualifying dataset to its own probe-selected ticket is
significantly worse than giving every dataset the single global
winner, ticket 33.

![Per-dataset Δ_route, sorted — 34 wins vs 54 losses, pooled CI
entirely above zero](../img/noiseladder/rung2_per_dataset_delta.svg)

The supporting reads agree with the primary rather than softening
it. **Read 3** (win table): routing wins 34 qualifying datasets,
loses 54, ties 9 — win rate 0.386, two-sided sign p = 0.042.
**Read 4** (record-only mirrors): Spearman between a dataset's
draw-dispersion and its routing gain is −0.05 — the dispersion
signal that ordered R4b's quartiles carries nothing about *which
datasets* benefit from their own ticket.

## The selection-transfer inversion, quantified

The probe rows that *selected* the map showed a selection-biased
delta of **−0.60**. On rows the selection never touched, the same
map delivers **+0.13**. That inversion is the median-2-frame caveat
from R4a cashing out: stage 0's F=6 permutation-null floor
guaranteed the per-dataset argmins beat *shuffled* argmins on the
banked data, but with ~6–20 probe frames per dataset the argmin
still memorizes its cell. Within-dataset row-holdout was the honest
test, and it failed it.

Two record-only reads keep the result well-framed rather than
over-read:

- **Read 2:** routed vs the stable-key baseline is still **−0.756**
  [−0.876, −0.649]. The golden-ticket effect itself (one shared
  structured noise beats per-sample keys, R2's −0.924) reproduces on
  this fresh decode. What failed is *specialization*, not tickets.
- **The horizon mirror** shows structure the pooled number hides:
  routing actually wins the first ~8 steps of the chunk and loses
  increasingly from step ~15 on — per-dataset tickets help the
  chunk's opening and hurt its long tail.

![Horizon mirror: routed vs ticket 33, with the per-step
difference](../img/noiseladder/rung2_horizon_mirror.svg)

## What this closes, and what it doesn't

**Closed:** the per-dataset-tickets rung. Board row stays with
global ticket 33 (5.6524 full-panel on this decode, consistent with
the banked 5.6468). No amendment, no re-run: the pre-reg's cell
sizes were the named risk, the falsifier was built for exactly this
outcome, and it fired with room to spare.

**Not closed:** the ladder above it — but the bar moved. LAFM-style
learned mode priors and DSRL-style state-conditioned noise now
inherit a measured prior: naive per-context specialization at small
per-context n *inverts* out-of-sample on our panel. Any future
specialization rung must show transfer on held-out rows at
selection time, not only a permutation-null clearance. The
early-vs-late horizon split is the one genuinely new lead (a
chunk-position-dependent noise policy would be a *different*,
cheaper axis than per-dataset routing), recorded here as a
record-only observation — it gets no arm without its own pre-reg.

**Independent and unaffected:** the seating arm (launched 19:25Z at
stage-2's rc=0, ~3.0 GPU-h) — it re-runs the *random-noise* draws-10
config with dumps to give R3's mean-of-top-10 vs mean-of-random-10
the paired read the board seating requires. Its verdict concerns the
top-10 *ensemble*, not per-dataset routing, and lands tonight.
