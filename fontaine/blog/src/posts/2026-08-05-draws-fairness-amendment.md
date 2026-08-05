# Amendment 1 to the noise-draw pre-reg: the mode-averaging fairness reads

*2026-08-05, ~22:1xZ. Amends
[the noise-draw ensembling pre-registration](2026-08-05-prereg-noise-draw-ensembling.md)
(its method step 1 — the unimodality probe — is superseded by the
sharper protocol below). Posted before any per-draw number exists;
the probe launches at the first quiet local-GPU boundary after the
draws chain completes, never concurrently.*

## Why now

Owner challenge 21:49Z, mid-exchange: **is chunk MAE unfair to flow
because it forgives mode-averaging?** An AR decode that splits the
difference between two valid modes and a flow draw that commits to
one valid-but-different mode get very different MAE for the same
task competence. Three reads were pre-declared in-channel; this
amendment freezes their definitions and instruments before any of
them produces a number.

## Instrument finding (the reason this needed code)

The pre-declared reads assumed "the draws-10 per-draw dumps" would
exist. **They could not have**: the draws chain launcher passes no
dump flag at all, and `--dump-predictions` stores the *post-average*
prediction — per-draw chunks were averaged away inside
`BijouPolicy.predict_with_text` and never left the process. Landed
tonight (check.py green, 184 tests):

- **`bijou.eval --dump-draws PATH`** — writes the bijou policy's
  pre-average `[frames, draws, chunk, dim]` stacks + truth/valid +
  the full frame-identity columns (#18.1 conventions, plus scoring
  semantics scalars so the npz is standalone). Loud constraints:
  requires `--checkpoint` and `--sample-draws > 1`. The prediction
  path is untouched — the mean is taken once on the full stack before
  the per-item split (`collapse_draws`, unit-tested: dumped draws
  average back byte-identically to the predicted chunks).
- **Oracles**: banked AR-100k panel report recomputed through the
  edited scoring path (12 cells, tolerance 1e-4); the fairness
  script's degenerate draws=1 run on the banked flow-80k npz
  reproduces the **6.6232** anchor exactly on reads 1 and 2 with
  all-zero dispersion
  (`reports/analysis__draws_fairness_k4l2_validate.json`).

## Probe protocol (frozen)

- **Plan**: `plans/holdout_curated_v0_k4l2_drawsprobe_s7.json` —
  every 7th core frame of the k4l2 panel plan, file order, offset 0:
  **2,458 frames, 792 repos, 2,458 of 4,301 panel episodes**; labeled
  panel empty (built by `fontaine/scripts/draws_probe_plan.py`,
  deterministic).
- **Run**: flow-80k checkpoint, `--sample-draws 10 --sample-steps 30`
  (Heun), `--noise-key index`, `--seed 0`, same corpus flags as the
  chain, plus `--dump-draws`. Cost ≈ 30 min on 1×H100 at the
  measured draws-10 pacing (~1/7th of a full-panel draws run).
- **Instrument gate** (E1-style): the probe's draw 0 re-decodes the
  banked single-draw predictions — mean per-frame chunk-MAE drift vs
  the banked flow-80k npz rows (`draw0_vs_banked_frame_mae_drift`)
  expected **< 0.05**; larger = instrument finding, diagnose before
  any read is quoted.
- **Analysis**: `fontaine/scripts/draws_fairness.py --draws <npz>`
  (pure CPU; joins the probe rows to the banked AR-100k / flow-80k
  npzs on the corpus concat `index` — the paired-analysis join, with
  a hard assert that truth/valid rows agree).

## The three reads (definitions frozen)

All pooling valid-element-weighted, matching the report's
`chunk_mae` exactly (validated against the anchor above).

1. **Mean-of-draws MAE** — pooled MAE of the 10-draw ensemble mean.
   Ensembling manufactures the mode-averaged predictor flow "should
   have been" under an MAE-fair comparison. Primary magnitude comes
   from the chain's full-panel draws-10 run (pre-reg E3); the probe
   read cross-checks it on the subset.
2. **Best-of-N MAE** — per frame, the best of the 10 draws by that
   frame's chunk MAE (first_mae selects its own best draw,
   independently); pooled. The oracle mode-match bound: how good is
   flow when "sampled a different valid mode" is forgiven entirely.
3. **Dispersion-conditioned deficit** — per-frame dispersion = masked
   element-mean of the across-draw std; probe frames cut into
   dispersion quartiles; per-quartile mean paired deficit
   (flow-single-draw − AR-100k, per-frame, from the banked npzs) +
   flow win rate + Spearman(dispersion, deficit). Per-step dispersion
   curve reported alongside (the #1 prediction: spread should grow
   with horizon).

## Pre-declared interpretation

- **Unfair-penalty signature**: deficit concentrating in the high-
  dispersion quartiles (monotone quartile trend, positive Spearman) —
  flow is being punished where it commits to modes. Read 2 sizes it:
  best-of-10 at or below AR's paired chunk MAE on the probe frames
  says a valid-mode within 10 draws matches AR.
- **Modeling-deficit signature**: deficit flat across dispersion
  quartiles and best-of-10 still well above AR — the gap is not a
  metric artifact; attribution screens proceed on the AR recipe
  (owner steer 21:48Z).
- Effect sizes quoted with everything; quartile noise floor read off
  the quartile n (~615 frames each).

## Honest limits (stated in-channel, kept here)

MAE cannot settle actual rig performance either way; the owner's
comm-MAE→rig bridge was built on AR checkpoints. If the unfair-
penalty signature confirms, the comm holdout needs a distributional
column (best-of-N or an energy-distance-style score) before it can
rank flow arms — that column feeds the limit-attribution front, not
a new benchmark.
