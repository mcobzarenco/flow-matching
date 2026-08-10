# Fairness probe results: is chunk MAE punishing flow for committing to modes?

*2026-08-06, ~07:3xZ. Results for
[Amendment 1](2026-08-05-draws-fairness-amendment.md) (reads 1–3) and
[Amendment 2](2026-08-05-draws-fairness-amendment2.md) (read 4, energy
score) to the noise-draw pre-registration — the owner's 21:49Z
challenge, instrumented and frozen before any per-draw number existed.
Probe: flow-80k, draws=10 heun-30, the frozen 2,458-frame stride-7
subset plan, `--dump-draws`; analysis
`fontaine/scripts/draws_fairness.py`, report
[`analysis__draws_fairness_k4l2.json`](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__draws_fairness_k4l2.json).
This is the 06:57Z
relaunch: the first attempt scored all 2,458 frames and crashed in
`merge_shards` (the mirror of the 04:4xZ bug — empty
`dump_predictions` permuted when only `--dump-draws` is on; fixed
`da9ec6a` with a regression test, crash log preserved). No banked
number was touched — the crash was post-scoring, pre-write.*

## Instrument gate (E1-style) — draw-0 re-decode

**PASSED: `draw0_vs_banked_frame_mae_drift` = 0.0145 < 0.05.** The
probe's draw 0 re-decodes the banked single-draw predictions (same
noise key/seed) to a mean per-frame chunk-MAE drift of 0.0145 —
cross-run numerics only. The row join carries the hard asserts
(truth/valid byte-agreement between probe and banked panel rows), and
the degenerate draws=1 validation was re-run green immediately before
the npz was opened (anchor 6.6232 reproduced exactly, dispersion and
ES interaction exactly zero). All reads below are quotable.

## The reads

All numbers on the 2,458 probe frames; pooling valid-element-weighted
(the report convention) unless noted. Paired AR/flow-single columns
come from the banked npzs through the instrument's index-join.

| read | chunk_mae | first_mae |
|---|---|---|
| flow single draw (draw 0) | 6.6755 | 1.9275 |
| flow banked single, same frames | 6.6737 | — |
| AR-100k paired, same frames | 5.8680 | 2.1314 |
| **read 1 — mean of 10 draws** | **5.4113** | **1.3975** |
| **read 2 — best of 10 (oracle)** | **3.8597** | **0.9953** |

- **Read 1** cross-checks the full-panel chain: probe mean-of-10
  5.4113 vs the chain's full-panel 5.365 — the stride-7 subset
  reproduces the ensembling gain (−1.26 vs single draw here). Per-draw
  pooled spread across the 10 draws: 0.2212 at probe size.
- **Read 2** — the oracle mode-match bound is **2.01 below AR's
  paired chunk MAE** (3.8597 vs 5.8680) and takes first_mae under 1.0.
  When "sampled a different valid mode" is forgiven entirely, flow is
  not merely competitive with AR — it is far better.
- **Read 3 — dispersion-conditioned deficit** (per-frame paired
  flow-single − AR from the banked npzs; probe pooled deficit +0.7789
  frame-mean):

| dispersion quartile | mean disp (°) | deficit | flow win rate |
|---|---|---|---|
| q1 (tight) | 1.29 | **+0.2305** | 0.366 |
| q2 | 3.44 | +0.6013 | 0.321 |
| q3 | 5.24 | +0.8652 | 0.381 |
| q4 (dispersed) | 9.08 | **+1.4186** | 0.394 |

  Monotone across all four quartiles; Spearman(dispersion, deficit)
  = **+0.1292**. The q4 deficit is 6.2× the q1 deficit — the paired
  deficit concentrates exactly where the draws disagree. (Frame-level
  Spearman is weak even though the quartile means are cleanly
  monotone: dispersion is a systematic but minority contributor to
  per-frame deficit variance.)
- **Read 4 — energy score** (strictly proper; Amendment 2):
  **flow 10-draw ES 5.9308 vs AR ES 8.7696** — flow wins by 2.84
  under a mode-fair distributional score while losing single-draw MAE
  on the same frames. Flow single-draw ES 9.8825 (banked single
  9.9251 — consistent); interaction term 3.9517, i.e. the value of
  modeling the distribution rather than sampling from it once.

## Verdict against the pre-declared signatures

**The unfair-penalty signature FIRED, on every declared criterion:**

- Amendment 1's signature — "deficit concentrating in the high-
  dispersion quartiles (monotone quartile trend, positive Spearman)"
  — both hold (monotone 0.23→0.60→0.87→1.42, Spearman +0.13).
- Read 2's sizing — "best-of-10 at or below AR's paired chunk MAE" —
  exceeded by 2.01.
- Amendment 2's ES criterion — "flow ES ≤ AR ES while flow
  single-draw MAE > AR MAE" — flow ES is 2.84 *better*. Per the
  pre-declared interpretation, this is quantified evidence that the
  MAE deficit is (at least partly) a scoring-rule artifact, and **ES
  becomes the candidate distributional column for ranking flow arms
  on the comm holdout**.

The modeling-deficit signature (flat quartiles + best-of-10 well
above AR) did not fire.

**Honest residual, stated with the same discipline:** the deficit is
positive in *every* quartile — including q1, where the 10 draws
nearly agree (mean dispersion 1.3°, deficit +0.23) — and flow's
per-frame win rate stays below 0.5 everywhere (0.32–0.39). Mode
averaging does not explain the whole gap: there is a real, smaller
single-draw deficit even where the predictive distribution is tight.
"Partly artifact" is the supported claim; "wholly artifact" is not.
And per Amendment 1's honest-limits section, MAE-family reads cannot
settle actual rig performance either way.

## σ_draw: the direct measurement vs the 0.0159 pin

**Direct σ_draw = 0.02367 — SUPERSEDES the model-based 0.0159 pin
(1.49×), but stays under both floors, so `reopen_floors: false` and
every dependent band is numerically unchanged**
(`fontaine/scripts/sigma_draw_direct.py`, self-oracles O1–O4 green,
report
[`analysis__sigma_draw_direct.json`](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sigma_draw_direct.json)).

- Primary (frame-level delta-method): pooled per-frame across-draw
  std → σ = 0.02367 at panel F_eff 16,488.5 (probe F_eff 2,341.7;
  frame-level std_η 3.04° pooled, median 1.52°, p90 4.54°).
- Cross-check (pooled-level, n=10): the 10 per-draw pooled MAEs give
  0.02522 scaled to panel size; the primary sits inside the χ²₉ 95%
  band [0.01735, 0.04605].
- Input-drift oracles green: mean-of-draws reproduces the probe
  report's chunk_mae; F_eff reproduces the posted 16,488.5; frozen
  2,458×10 shape asserted.
- Consequence, exactly as pre-declared in the finalization amendment:
  σ_draw_final = **0.02367** is the quoted draw-noise sigma from here
  on; the model-based extrapolation was ~1.5× optimistic, but both
  verdicts were floor-bound and survive untouched — **stable-noise
  re-bank band stays [6.4882, 6.7582]** (0.045 floor binds),
  **SnapFlow 1-NFE adopt band stays ≤ 6.7732** (0.15 floor binds;
  3σ = 0.071 < 0.15).

## What this feeds

1. **The owner's 21:49Z challenge is answered**: yes — chunk MAE
   punishes flow for committing to modes. The paired deficit
   concentrates 6× in the dispersed quartile, an oracle valid-mode
   match beats AR by 2, and a strictly proper score flips the
   ranking outright. But a smaller real deficit survives in
   tight-dispersion frames, so MAE unfairness does not fully
   exonerate the flow recipe.
2. **ES as the distributional column** (Amendment 2's pre-declared
   consequence): candidate for ranking flow arms on the comm holdout,
   feeding the limit-attribution front — adopting it into any
   benchmark convention is an owner decision, not taken here.
3. **σ_draw_final = 0.02367** replaces the 0.0159 pin in all future
   draw-noise bands; the two live bands are floor-bound and
   unchanged.
4. **The #18.2 stable-key flip re-bank launched immediately after
   these reads** (gate `reopen_floors == false` asserted in the
   launcher) — band [6.4882, 6.7582], hard bitwise controls on
   state-copy/AR.
5. **Attribution screens** stay weighted to the AR recipe (owner
   steer 21:48Z stands), but MAE-based AR-vs-flow rankings should
   now be read with the ES column alongside — the mainline "flow
   loses on the panel" summary needs the qualifier "on a
   mode-averaging-friendly metric."
