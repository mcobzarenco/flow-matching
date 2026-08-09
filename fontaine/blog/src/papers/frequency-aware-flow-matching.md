# FAFM: flow matching in frequency space — smoothness from the training side

*Lit slice 2026-08-09 (work session 15:3xZ, in the adamc_100k
shadow). Frequency-Aware Flow Matching
([2606.20135](https://arxiv.org/abs/2606.20135), "Frequency-Aware
Flow Matching for Continuous and Consistent Robotic Action
Generation"). The training-side member of the smoothness/boundary
family whose inference-side member (SEAM) fed this morning's
[boundary-incompatibility read](../posts/2026-08-09-boundary-incompat-results.md).
Fed #22 (family map: what training-side smoothing does and does NOT
fix), #9 (mixed-frequency data ingestion becomes well-posed), #12
(a smaller generation target for few-NFE), #16 (the LDLJ jerk metric
banked as rig-bench instrumentation).*

## The paper in plain words

Policies like ours are trained to output a *chunk* of, say, 50 future
actions as one big list of numbers — one action per timestep. This
paper points out two problems with that list. First, the list is tied
to a clock rate: a demonstration recorded at 10 Hz and one recorded
at 20 Hz put *different physical motions* in "slot 7", so training on
mixed-rate data teaches the model a physically meaningless average
(they prove a small theorem to this effect — and show a π₀ trained on
mixed-rate demos collapses from 94% task success to 0%). Second,
nothing about the list says "consecutive actions should look like a
motion": the model can put a jump between slot 7 and slot 8, and
smoothness has to be learned the hard way. Their fix: don't generate
the list — generate the trajectory's *frequency description* (a
discrete cosine transform, keeping only the lowest ~third of the
frequencies), and train the flow model on those coefficients. The
trajectory is then reconstructed as a smooth continuous function you
can read out at any clock rate, and a second loss term supervises its
time-derivative against the demonstration's velocity. Same network
size, no extra inference cost; smoother motions, faster convergence,
and the mixed-rate collapse disappears (92% either way).

## The method, exactly

- **DCT-II parameterization.** A K-step action trajectory ξ becomes
  coefficients c⁰..c^M with M ≈ K/3 (M=4 for K=12 standalone
  policies, M=16 for K=50 VLA chunks). Reconstruction is the cosine
  series v̂(τ) = ½c⁰ + Σ c^j cos(ω_j τ), ω_j = jπ/T — a *continuous
  function of physical time*, evaluable at any τ.
- **Flow matching over coefficients.** The FM loss transports noise
  to coefficient vectors instead of action lists — the network's
  output dimension shrinks from K·D to (M+1)·D.
- **Sobolev regularizer.** ℒ_vel supervises the reconstruction's
  first derivative at sampled times against the demo's finite-diff
  velocity, weight λ=1 everywhere. Their Theorem 1: the combined loss
  is a weighted H¹ norm over coefficient errors with weights
  μ_j = 1 + λω_j² — high frequencies are *penalized quadratically
  more*, which is where the smoothness comes from.
- **Frequency independence.** Coefficients depend on physical time
  only: two recordings of the same motion at different Hz give the
  same targets up to O(1/K). Their Proposition 1 shows step-indexed
  training on mixed-Hz data is ill-posed (the Bayes-optimal predictor
  averages actions from *different physical times*).

## What they ran

- **LIBERO (π₀.₅):** +15% pick-and-place, +10% multi-obstacle success
  over the standard chunk head; smoothness (log dimensionless jerk,
  LDLJ) consistently better.
- **The mixed-frequency headline:** LIBERO drawer, 43 demos either
  all-10 Hz or split 5/10/20 Hz. π₀: 94% → **0%** on mixed. FAFM: 92%
  on both. The cleanest published demonstration that the chunk-index
  representation, not the data, is what breaks.
- **Multimodality vs smoothness (synthetic obstacle course):** FAFM
  61% success / LDLJ −5.60 / 12 modes kept, vs vanilla FM 48% /
  −8.62 / 14 modes and smoothness-first baselines that collapse to
  2–3 modes. Frequency-space smoothing does *not* mode-collapse —
  it removes jitter, not diversity.
- **Surgical (LapGym) + real Franka:** best success and LDLJ across
  rope threading etc.; 100% on the real pick-and-place with the best
  smoothness. Ablations: drop the DCT and the derivative loss does
  nothing (finite-diff supervision on raw chunks is ineffective);
  drop ℒ_vel and both success and smoothness sag.

## What transfers to us

- **A training-side lever on within-chunk smoothness with a shrunken
  target.** Our flow expert generates 50×6 values; a DCT head at
  M=16 would generate 17×6 — a third the output dimension, smooth by
  construction. Plausibly friendlier to few-NFE/one-step distillation
  (#12's menu) since the target manifold is lower-dimensional and
  low-frequency. Worth a line in the #12 ledger, not an arm today:
  our SDN read says our ODE draws are *already* uniformly smooth
  within-chunk, so the smoothness half of the sell is pre-solved for
  us — the representation/efficiency half is the live part.
- **Mixed-frequency ingestion becomes well-posed (#9).** Our current
  corpus is single-rate, so Proposition 1 does not bite today — but
  the moment UMI-style or cross-rig data enters (the RDT2 10k-hour
  premise, VISTA's adaptation pitch), step-indexed chunks are the
  wrong representation and this is the documented fix. Banked as the
  design answer to a problem we don't yet have.
- **LDLJ banked as instrumentation (#16).** Log dimensionless jerk
  joins ABPolicy's 95th-pct accel + zero-crossings in the rig-bench
  metric kit — the field is converging on it as the smoothness
  number.

## What doesn't transfer

- **It does not touch our measured problem.** This morning's read
  found the big term is *cross-chunk*: seam disagreement 1.1–1.3×
  model error, boundary jump 11–14× per-step motion — while
  within-chunk smoothness (FAFM's whole target) is already clean on
  our stack. The paper is explicit only about within-chunk
  consistency; receding-horizon regeneration and inter-chunk
  continuity are unaddressed. A low-frequency basis might even
  *widen* per-draw mode gaps at the seam (fewer coefficients, more
  committed trajectories) — unknown, and our boundary read would be
  the free way to test any such head.
- **Effect sizes are π₀-class + LIBERO/toys**, and the wins are
  closed-loop (success, jerk). Our offline MAE panel would price a
  smoothed chunk *worse* by construction — same #16-gating logic as
  SEAM.
- **Their own stated limit:** impulsive/high-frequency tasks (sharp
  contacts) are where a low-pass action basis hurts. Contact-rich
  manipulation on the rig is exactly where we'd need to check LDLJ
  gains against contact-timing losses (the ACT-TE −12 pts lesson).

## Verdict

The training-side complement to SEAM: same family, other end of the
pipeline, and the two are cleanly composable in principle (smooth
chunks from a DCT head, seams closed by noise-space steering). For us
today it is a bank, not a build: the within-chunk problem it solves
is one our stack doesn't have, and the cross-chunk problem our stack
*does* have (measured this morning) is one it doesn't address. Its
lasting contributions to our program are the mixed-frequency
ill-posedness theorem (unlocks heterogeneous data for #9 whenever we
get some) and LDLJ for the #16 metric kit.
