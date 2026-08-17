# sft-v1 flow regression isolated: it was the normalization window, twice, in two different ways

*2026-08-17 07:2xZ, run in-flight during the demo_gen_v2 regen ride.
Queue item `sft-v1-flow-regression-isolation`; verdict posted
in-channel (msg 1538811601153425469). Data record:
[analysis__sft_v1_flow_isolation_tables.json](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sft_v1_flow_isolation_tables.json).*

**Plain words.** Our best small model grasped the boat 44 times out of
100. Two bigger retrains — with 16× the data — almost never grasp. The
serving audit had already cleared the delivery pipeline, so the fault
was baked in at training. The three suspects were: the two-headed
training objective, the sim+real data mix, and the way action values
get squashed to a standard range before the model learns them (the
"normalization table"). Today's cheap experiment — re-evaluating an
archived retrain that shared everything with the broken run *except*
the table — plus a close read of the tables themselves, points firmly
at the squashing: **each broken run squashed the sim actions through a
window that didn't fit sim data, and each time the distortion landed
on a wrist joint — exactly where a grasp lives or dies.** The
training objective is formally cleared; the data mix remains possible
but has no mechanism evidence.

## The cells

| run | objective | data | action table | flow grasps |
|---|---|---|---|---|
| probe `joint_corrected@2000` | joint + insulate-flow | demos-only | demos-native (corrected) | **44/100** |
| run-1b `remaponly@2000` | joint + insulate-flow | 3-dataset mix | rig-lineage remap | **0/20** (new today) |
| run-2 `@3000` | joint + insulate-flow | 3-dataset mix | pooled `--recompute-stats` | 5/100 box, 0/20 local |

The new cell: run-1b's step-2000 weights (13 GB, pulled from the box
archive) evaluated on the audit's 20 unseen seeds (100–119, local
H100, ~0.5 GPU-h): **0/20, median final distance 9.18 cm** —
statistically the same collapse as run-2's 0/20 / 8.9 cm.

Two logical consequences:

1. **The pooled table is not the sole lever.** Run-1b never saw
   `--recompute-stats` and is equally broken — the named suspect (c')
   cannot be the whole story.
2. **The joint objective is exonerated.** Chasing the probe's
   provenance pinned the 44/100 checkpoint as
   `joint_corrected/step_002000` — the joint objective itself, trained
   demos-only on the corrected demos-native table. (a) is clean.

## The mechanism, quantified

Per-channel occupancy = (sim demos' q01–q99 action width) / (serving
table's width) in normalized space; flow-MSE gradient weight scales
as occupancy².

- **Run-2 (pooled)**: wrist_flex occupancy **48.9%** → **0.24×**
  gradient weight; every other channel 0.96–1.01×. Pooling real+sim
  widened exactly one channel's window, and it's the grasp-critical
  one. Surgical.
- **Run-1b (rig-lineage)**: wrist_roll occupancy **288%** — the sim
  demos' ±157° rolls overflow the table's ±66°-ish window, so training
  targets clip at ±1 *and* serving can never command a roll beyond
  ~66°. A different channel, a different distortion, the same class:
  sim supervision squeezed through a foreign window.
- The one run whose table fits its own (sim-only) data is the one
  that grasps.

## What it means for v2

The recommendation to the owner (pending): **per-dataset (per-item
row) flow-target normalization**, or a demos-native table for the sim
slice — with training and serving consistent, the `b779ba4` lesson.
The caveat is disclosed: the sim+real mix (b) rides along in both
broken runs and is not formally exonerated. But the elegant part is
that **SFT-v2 is itself the clean fourth cell**: 3-dataset mix + a
sim-fit table. If it grasps, (b) is exonerated for free; if it
doesn't, the mix becomes the prime suspect — either way the next run
is the experiment, and no extra GPU-hours are spent on isolation.
