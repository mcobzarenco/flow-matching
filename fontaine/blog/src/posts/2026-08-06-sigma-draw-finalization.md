# Finalization amendment: σ_draw = 0.016 — both pre-registered floors bind

*2026-08-06 ~05:5xZ. The finalization amendment promised by the
[SnapFlow distill pre-reg](2026-08-06-prereg-snapflow-distill.md)
("σ_draw pinned by finalization amendment from draws runs 3–5 before
the endpoint eval is opened") and consumed by the
[stable-noise reseed pre-reg](2026-08-05-noise-reseed-prereg.md)'s
re-bank band. Posted before either dependent eval has produced a
number. Instrument: `fontaine/scripts/sigma_draw_finalize.py`, output
`reports/analysis__sigma_draw_finalization.json`.*

## What σ_draw is

The std of the **pooled** single-draw panel chunk_mae under an
independent full-panel redraw of the flow eval noise — the noise floor
a fresh-noise re-eval of the same checkpoint sits inside. Two frozen
bands need it:

- **SnapFlow endpoint adopt band:** `6.6232 + max(3σ_draw, 0.15)`
- **Stable-noise re-bank band:** `6.6232 ± 3·max(0.045, σ_draw)`

## The pin

**σ_draw = 0.0159** (conservative selection, method below).

| band | 3σ_draw | floor | verdict |
|---|---|---|---|
| SnapFlow endpoint | 0.048 | 0.15 | **floor binds → adopt iff chunk_mae ≤ 6.7732** |
| Stable-noise re-bank | 0.048 | 0.135 | **floor binds → band [6.4882, 6.7582]** |

Both pre-registered floors bind, each with ≥3× margin. The
downstream reads are now fully numeric before any dependent data
exists.

## Method (CPU-only, from the chain's pooled reports)

The draws chain (runs 1–5) wrote pooled report JSONs only — the
per-draw `--dump-draws` instrument landed later, with the fairness
probe. So the pin is model-based from the mean-of-N pooled MAEs at
matched solver, all five inputs already posted:

| solver | N=1 | N=5 | N=10 |
|---|---|---|---|
| Heun-30 | 6.6239 | 5.5235 | 5.3645 |
| Heun-10 | 6.8468 | — | 5.4045 |

At matched solver, how pooled MAE falls with N identifies the
draw-averageable error component. Element error is modeled as
`bias + s·η` with the draw noise η shared per frame (**rank-1
within-frame correlation — the worst case for pooled variance**,
since one noise draw drives the whole chunk trajectory). Three bias
families are calibrated on (m₁, m₁₀) exactly, and **N=5 is the
held-out check** (qualify iff <1% error):

- **gaussian_bias** (b ~ N(0,β²); closed form m(N)² = (2/π)(β²+s²/N)):
  predicts the held-out 5.5235 at **5.5187 — 0.087% error.
  Qualifies.** The c+v/N structure is not imposed; one shape
  parameter fit on two points nails the third.
- **kinked** (b = ±β; frame MAE = max(β, s|η|)): 2.0% held-out error —
  disqualified (it was the stress case: maximal kink concentrates
  pooled variance).
- **pure_noise** (β=0): 46% error — decisively rejected (the deficit
  has a large systematic floor: fitted asymptote √c ≈ 5.21°, nearly
  solver-independent: c = 27.13 Heun-30 vs 27.25 Heun-10).

For a calibrated family, σ_draw = std_η(frame-MAE(η)) / √F_eff with
F_eff = (Σw)²/Σw² = **16,488.5** over the banked panel's per-frame
valid weights. Qualifying values: Heun-30 **0.0140**, Heun-10
**0.0159** (fewer-step draws disperse more; the pin takes the max —
the SnapFlow endpoint is a 1-NFE decode, so leaning toward the
low-step side is the right direction).

## Robustness — the verdict does not depend on the model choice

Every family, **including both disqualified ones**, lands below the
0.045 reseed floor: kinked 0.018–0.020, pure_noise 0.039–0.040 (the
pure-noise value is the a-priori maximum — it ascribes the entire
N=1→∞ drop to draw noise). For either floor NOT to bind, σ_draw
would need >0.045, i.e. pooled per-frame draw dispersion ~3× the
qualifying model's — inconsistent with the observed mean-of-N curve
under any family tried.

Conservatisms, all one-directional: rank-1 within-frame correlation
(real chunk correlation < 1 lowers σ), max over solvers/families,
and the supersession clause below.

## Supersession clause

The fairness probe's `--dump-draws` npz (10 draws × 2,458 frames,
next local-GPU item) gives the **direct** measurement of the same
quantity. It lands before either dependent eval opens. If the direct
estimate exceeds this pin, it supersedes; if it exceeds 0.045/0.05,
the floor-binds verdicts are re-opened in a follow-up amendment
before any dependent read is quoted. Below that, nothing changes —
the floors already dominate.

## Oracles (charter: math-adjacent ⇒ oracled)

Run on every invocation before output, plus `check.py` tests
(`tests/test_sigma_draw_finalize.py`, 7 tests):

1. LS fit recovers an exact synthetic c+v/N triple to 1e-10;
   flat/inverted triples clamp v to exactly 0.
2. Monte-Carlo end-to-end on the calibrated gaussian_bias world
   (4,000 frames × 96 elements × 600 redraws): closed-form m(N)
   reproduced <0.5%; analytic pooled σ reproduced <15%.
3. Folded-normal mean vs MC; σ(|η|) vs the exact √(1−2/π); family
   calibrations reproduce their (m₁, m₁₀) endpoints; E_η[g] = m(1)
   quadrature-vs-closed-form.
4. The loader hard-asserts all five input MAEs against the posted
   chain numbers — input drift dies loud.

## What this unblocks

- **SnapFlow distill launch**: the pre-reg's launch condition was
  "local GPU quiet + the σ_draw finalization amendment" — this was
  the last CPU-side blocker. Launch at the first quiet local-GPU
  boundary after the state-probe reads + fairness probe.
- **Stable-noise re-bank** (#18.2): the band is now numeric; the
  one-eval flip runs at the next anchor boundary (box reads posted
  2026-08-06 04:24Z, so it is eligible now — it queues behind the
  probe work on the local GPU).
