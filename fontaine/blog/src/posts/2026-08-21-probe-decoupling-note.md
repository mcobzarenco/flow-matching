# The probe decoupling: our offline instruments cannot see grasp collapse

*2026-08-21. Methods note, consolidating a pattern the pdnorm mix
lineage has now banked twice. No new experiments — every number
below is a banked read with its artifact in `reports/`. Feeds the
eventual VLA eval-design doc (the north star's rig eval will need
this decided up front).*

**Plain words**: we train grasping policies and we measure them two
cheap ways (how well they predict held-out demonstration actions)
and one expensive way (asking a simulator to actually run the arm
100 times and counting grasps). The lineage now has two banked
cases where a policy that had catastrophically stopped grasping
scored *as well as or better than* the healthy policies on both
cheap instruments. The cheap instruments are still worth running —
they catch normalization bugs and training divergence loudly — but
they can no longer be allowed to answer the question "did the model
get better at the task?". Only rollouts answer that.

## The four cells

Same recipe throughout (v2-joint + insulate-flow, per-dataset flow
norm, eff-96, 3000 steps, seed 0); the only lever moved across
cells is *which data rides with the demos*:

| cell | mix | sim100 | panel truth-fit | probe endpoint |
|---|---|---:|---:|---:|
| onerig | demos + v2 ×4 | **28**/100 | 27.26 | 4.53 |
| control | demos only | **11**/100 | 27.40 | 5.90 @1000 |
| democlean | demos + clean ×4 | **8**/100 | 28.43 | 4.68 |
| convicted | three-way mix | **1**/100 | 27.44 | 6.17 |

![Panel and probe columns for the four cells vs their sim100
rows](../img/probe-decoupling/decoupling_columns.png)

The sim100 column spans **28×** (1 → 28 grasps, paired per-seed
McNemar reads CI-excluding-zero at every step of the ladder). The
two offline columns are flat — and *interleaved*: the collapsed
cells do not even sort to one end.

Receipts: sim100 paired reads
`analysis__sim100_paired_democlean3000_vs_{onerig3000,disc1000,pdnorm3000}.json`;
panel rows `analysis__{onerig,democlean,pdnorm}_endpoint_truthfit_wear.json`
+ `analysis__disc1000_panel_row_audit.json` (control) +
`analysis__released_row_honest_wear.json` (anchors); probe curves
from the four runs' `train_log.jsonl`.

## Banked miss #1 — the k4l2 panel

The k4l2 panel measures chunk-MAE against held-out community
episodes, estimator-consistent via the truth-fit rewear instrument
(same estimator both sides of the ladder). The banked ladder:

- **democlean, 8/100 collapsed: 28.43** — the *highest* (most
  drifted) of the four, but within ~1 point of everything
- convicted, 1/100: 27.44
- control, 11/100: 27.40
- **onerig, 28/100 healthy: 27.26**
- released baseline: 27.14; shuffled-repo null midpoint: 25.15

The between-cell spread (1.17 deg) is *smaller than the within-cell
estimator seam* (native-minus-truth-fit gap: 1.55–1.91 deg across
these cells). Reading a grasp ranking out of this column isn't just
unsupported — it's below the instrument's own noise floor. The
panel put the 8/100 model within ~1 point of the healthy class, and
put the 1/100 convict *between* the 11/100 and 28/100 cells.

## Banked miss #2 — the eval probe curve

The in-train probe (eval-250 chunk-MAE on held-out demo frames)
looked like it had a signature: the convicted three-way cell showed
a clean elevation, 5.45/5.47 plateau → 6.83 peak in the 2250–2750
window → 6.17 close. The democlean cell was the test of that hope:
same collapse class (8/100 vs 1/100), and its probe curve **fell
monotonically to 4.6848 — closing at the healthy onerig cell's
level (4.5266)** — while grasping sat collapsed the whole time.

So the probe's positive read is real but its silence proves
nothing: one collapsed cell showed the elevation, the other tracked
the healthy curve to the digit. A signature that fires on some
poisons and not others, with no way to know which kind you have, is
a drift alarm — not a verdict.

## What the offline instruments DO see

They stay in the battery — because what they do see, they see
loudly and cheaply:

- **Normalization seams and table bugs.** The disc-1000 checkpoint
  wearing the raw released table scored 58.14; re-worn on its
  training rows, 27.40. A ~30-deg signal for a wear bug, caught in
  seconds without a rollout. This is the panel guard's actual job,
  and it has caught real bugs (the pdnorm table fix was verified
  exactly this way: Δ −28.96, receipts on the named rows).
- **Training divergence.** The convicted cell's probe elevation was
  a true positive — the mix was fighting the demos and the curve
  said so mid-run, for free.
- **Estimator seams.** The truth-fit rewear instrument decomposes
  native-vs-truth-fit wear (1.5–1.9 deg here), which bounds what
  panel deltas can mean before any claim is made.

## Why the miss is structural, not fixable by trying harder

Both instruments score action prediction *on states drawn from the
demonstration distribution*. Grasp collapse doesn't live there: a
collapsed policy still imitates demo frames almost perfectly — the
failure is a small systematic error (here, gripper amplitude
compression carried by ~0.7% of training frames) that only
compounds once the policy visits its *own* states in closed loop.
Add the dilution: chunk-MAE averages over 6 joints × a full chunk,
so a grasp-critical error at one channel over a handful of contact
frames moves the aggregate by less than the estimator seam. The
instrument isn't badly built — it's answering a different question
(distribution match) than the one we're asking (competence).

## The standing rule

Effective immediately, already encoded in the gripfix babysit
registry entry ("record-only — the probe canNOT clear this cell"):

1. **For any mix / recipe / data cell, the verdict instrument is a
   rollout eval.** sim100 today; the rig eval when the north-star
   VLA gets there. Pre-registration gates must be stated in rollout
   numbers, with paired per-seed reads.
2. **Panels and probes are drift guards only**: explicit thresholds
   against banked anchors (wear bugs, divergence), record-only for
   everything else.
3. **No offline read can clear a cell.** A green panel plus a
   healthy-looking probe curve is compatible with 8/100 — that is
   now a measured fact, twice. Silence from these instruments
   changes nothing about the verdict.

The cost asymmetry that used to justify probe-based verdicts has
also collapsed: sim100 is ~2.5 GPU-h per endpoint, and the Squint
result (lit `0819`) points at twin rollouts on trivial compute for
*relative* screens. Rollout verdicts are affordable; wrong verdicts
are not.

*Related: [democlean pre-reg + verdict
append](2026-08-20-prereg-demos-plus-clean.md) (the probe-curve
contrast chart), [onerig pre-reg](2026-08-19-prereg-demos-plus-one-rig.md)
(the panel ladder), [truthfit rewear
instrument](2026-08-18-prereg-grasp-sft-v2-joint-pdnorm.md).*
