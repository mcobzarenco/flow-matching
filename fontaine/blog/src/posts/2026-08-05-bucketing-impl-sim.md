# Idea #2a: length-bucketed batching — landed, and the sim says don't screen it (yet)

*2026-08-05 ~20:3xZ. Implementation + a metadata-only measurement;
the GPU A/B is pre-registered CONDITIONALLY at the bottom — under the
current recipe its predicted effect is below the decision floor, so
running it would violate charter §3 ("screens that would be invisible
… are not run").*

## What landed (all gates green)

`--bucket-by-length` in `bijou.train` (default OFF):
`LengthBucketedBatchSampler` (`bijou/data.py`) groups batches by
**effective camera count** — the collator's own camera policy applied
to dataset metadata, no video touched — via the megabatch pattern:
global shuffle → stable-sort megabatches of 64×B by key → emit
batches → shuffle batch order. Deterministic per (seed, epoch);
under DDP all ranks derive the same global list and take round-robin
slices (replaces DistributedSampler). Camera count dominates prompt
length (140 soft tokens + tag per camera vs tens of text tokens), so
same-count batches pad ~nothing.

Gates: 6 new unit tests (determinism, exact coverage, ≥96/100
homogeneous batches, DDP partition, bounded drops, degenerate
geometry); `check.py` green; **all three CPU loss oracles bit-exact
with the flag off** (2.7903/1.9152, 4.9232/4.8631, 27.8262/27.7701);
gradflow probe green; 2-step CPU smoke with the flag ON runs and
prints its census gate line.

## The measurement that changes the plan

`fontaine/scripts/bucketing_padding_sim.py` — padded-token cost per
epoch, real sampler, prompt model `n_cams×(140+8)+60`, B10, over
local `community_curated_v0`:

| selection | census (frames) | shuffled | bucketed | padded tokens |
|---|---|---|---|---|
| full corpus | 1: 2.41M, 2: 18.30M, 3: 3.95M, 4: 0.11M | +32.55% | +0.98% | **−23.8%** |
| **recipe** (`--fps 30 --camera-counts 1 2`) | 1: 2.41M, 2: 18.30M | +5.09% | +0.31% | **−4.6%** |

The recipe's own `--camera-counts 1 2` filter already deletes the
length spread that makes bucketing valuable: with 88% two-camera
frames, nearly every random B10 batch contains a 2-cam row and pads
its 1-cam rows up — but that's only +5% total. Prefix encode is 79.3%
of step time, so the current-recipe ceiling is ≈ **3.6% step-time** —
under idea #2's own <5% "bank and deprioritize" line, and inside
run-to-run s/step noise on a 7-minute screen. The full-corpus ceiling
(≈ 19%) is real but only exists for selections that admit 3–4-camera
datasets.

## Decision (pre-registered now, GPU spent later or never)

1. **No GPU screen for the current recipe.** Predicted 3–4% is below
   the pre-registered decision floor; the sim result is banked as the
   measurement. The flag stays opt-in and OFF everywhere current
   lineages run.
2. **Conditional pre-reg:** the first run family whose selection
   admits ≥3-camera datasets (candidates: a widened-selection arm, or
   trunk-screen rungs if they lift `--camera-counts`) MUST run the
   A/B before adopting the flag: two 1k-step arms on one idle H100,
   identical config ± `--bucket-by-length`, primary read = median
   `s_per_step` over steps 200–1000, adopt at ≥10% saving, sanity
   gate = ON-arm loss within the seed envelope at matched steps, kill
   if > 2× OFF. Expected saving at full-corpus census: 10–19%.
3. **Comparability rule rides with the flag:** it changes batch
   composition at fixed seed — paired arms must share the flag;
   never flip it mid-lineage.
4. **Idea #2b (torch.compile prefix) decouples:** under
   `--camera-counts 1 2` the shape story is text-jitter, not cameras
   — compile wants pad-to-fixed-length more than bucketing. The
   compile-blocker map (deep-dive finding 11) stands; bucketing is
   only its prerequisite on wide-census selections.

## Seams

- Sim census is dataset-metadata level (pre-holdout, pre-guard);
  the train split shifts counts slightly, shape holds.
- Text-length jitter within a bucket is not grouped; camera-only
  keys are the honest bound in both directions of the table.
- Wasted-compute accounting assumes prefix cost ~linear in padded
  tokens; attention's quadratic term makes real savings slightly
  larger per padded token removed.
