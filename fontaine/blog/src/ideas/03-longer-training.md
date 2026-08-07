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
