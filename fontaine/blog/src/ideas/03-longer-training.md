# 3. Longer training on the best recipe — `queued`

*Tag: `longer-training` · idea #3 · [index](../ideas.md)*

- **Hypothesis:** rcond-100k was still improving at 100k
  (75k→100k bought 0.05–0.3); an extension banks a cheap win.
- **Cost:** a multi-day 1×H100 run (own-baseline rule: needs the
  eff-10/11 reference arm first, charter §4). Resume traps: fresh
  `--seed`, `--steps` = new TOTAL, cosine re-heat semantics.
- **Falsification:** panel MAE at matched eval cadence vs the
  own-baseline arm's curve; kill if the extension's curve is flat
  over its first 10–15k steps.

**2026-08-09 — lit `0814`: the horizon-churn recipe, published
([Anytime Pretraining page](../papers/anytime-pretraining.md),
2602.03702):** LR decay and weight averaging are two implementations
of the same implicit sample-weighting (exact for quadratics);
constant-LR or 1/√t trunks + online EMA match per-horizon-tuned
cosine at every intermediate budget (150M/300M scale, val loss
only). Our 40k→60k→100k lineage — each extension restarting from an
already-annealed floor — is this paper's motivating pathology. Two
banked consequences: (1) the next fresh trunk run's candidate recipe
is a constant-LR trunk + branch decays (last ~10% to 10% of peak)
from our every-5k saves, extendable without retuning; (2) a priced,
unqueued CPU-side read — uniform-average mid-run checkpoints (e.g.
30k–50k) as a "decayed endpoint preview" while the run continues;
their evidence is dense online EMA, ours would be 5k-sparse, so it
needs its own pre-reg before any panel eval. No status change; the
own-baseline entry condition stands.
