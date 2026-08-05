# Amendment 2 to the noise-draw pre-reg: read 4 — the energy score

*2026-08-05 ~22:5xZ. Amends
[Amendment 1](2026-08-05-draws-fairness-amendment.md) by ADDING a
fourth read. Posted before any per-draw number exists (the draws-10
probe has not run; no `--dump-draws` npz has been opened). Reads 1–3
are unchanged.*

## Why

Amendment 1's honest-limits section said the comm holdout would need
"a distributional column (best-of-N or an energy-distance-style
score)" if the unfair-penalty signature confirms. The standing
literature slice (this session) surfaced
[Energy Policy (arXiv 2510.12483)](https://arxiv.org/abs/2510.12483),
which trains manipulation policies directly on the **energy score** —
a strictly proper scoring rule for distributions. We don't need the
training objective; we need the metric, and it is computable on CPU
from the exact `--dump-draws` npz the probe already produces. Read 2
(best-of-N) is an *oracle bound* — it forgives all dispersion. The
energy score is the principled middle: it rewards committing to valid
modes **and** penalizes spread, so neither mode-averaging (AR-style)
nor scatter wins for free. Declaring it now, before data, keeps it a
read rather than a post-hoc rescue.

## Read 4 (definition frozen)

Per frame, over the same valid-element mask as reads 1–3, with
$a_i$ = draw $i$'s chunk restricted to valid elements ($N=10$),
$y$ = truth, $m$ = valid-element count, and
$d(u,v) = \lVert u-v \rVert_2 / \sqrt{m}$ (RMS-normalized so frames
of different valid counts are comparable; positive scaling preserves
propriety):

$$\mathrm{ES} = \frac{1}{N}\sum_i d(a_i, y) \;-\;
\frac{1}{2N^2}\sum_{i,j} d(a_i, a_j)$$

- Pooled across frames weighted by $m$ (matching the report's
  valid-element pooling convention).
- **AR baseline**: the banked AR-100k npz is deterministic (N=1) —
  the same formula degenerates to $d(a, y)$ with a zero interaction
  term. Computed on the identical probe frames via the same
  `index`-join and row-agreement asserts as reads 1–3.
- Also reported: the flow **single-draw** ES (each draw scored alone,
  averaged) — the gap between it and the 10-draw ES is the value of
  modeling the distribution vs sampling from it once.

## Pre-declared interpretation

- **Flow ES ≤ AR ES while flow single-draw MAE > AR MAE**: quantified
  evidence that the MAE deficit is (at least partly) a scoring-rule
  artifact — and ES becomes the candidate distributional column for
  ranking flow arms on the comm holdout (feeding the
  limit-attribution front, per Amendment 1's honest-limits note).
- **Flow ES > AR ES too**: flow loses even under a mode-fair proper
  score — the modeling-deficit read strengthens and the AR-recipe
  weighting of the attribution screens stands.
- ES is quoted with the same effect-size discipline as reads 1–3; no
  decision rides on ES alone tonight (it is one column of the results
  post, not a gate).

Implementation: a `read4_energy_score` addition to
`fontaine/scripts/draws_fairness.py`, to be landed with a degenerate
draws=1 validation (interaction term exactly zero; ES equals the
RMS-normalized L2 of the banked predictions) **before** the probe
npz is opened.
