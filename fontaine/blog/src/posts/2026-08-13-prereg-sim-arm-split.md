# Pre-reg — arm sub-part split: which rendered arm pixels carry the sim signature

*2026-08-13, posted in-channel 06:18Z (id 1537344697809240134), before
any read. Queue item `sim-arm-appearance-leg`; follows the fg content
split (leg (a)) and the clutter-patch legs (b)+(c).*

**Plain words.** After patching the fake desk clutter with real photo
crops, the biggest remaining chunk of "rendered stuff the encoder can
tell is fake" is the robot arm itself (~7.1% of the image). Before
spending anything on a fix, we ask the same question we asked about
the clutter: *which part* of the arm looks fake — the gripper, the
long links, or the little camera mounts — and is it one arm instance
or both? Same trick as before: re-run the image composite with only
some parts shown, everything else identical to the pixel, and see
which subset moves the "looks fake" score.

## Question

Rendered arm reads only_arm 0.654 vs plate_only 0.866 (leg (a));
patched surface 0.556 still ≫ real-fg 0.328. Which sub-part carries
it — gripper, links, or camera mounts? Which instance — follower or
leader?

## Design

Same hooked harness as leg (a): ONE production v3 instance (numpy
post backend), `_composite` hook, segmentation-restricted masks, 20
seeds × 5 appearance draws, noise-RNG state restored per arm — frames
pair 1:1. Encoder probe: er_60k trunk, top cam, knn5 AUROC vs
held-out real-B (A/B split as before). Two exact partitions of the 96
arm-class geoms (in-run tiling oracle):

- **part axis**: gripper+jaw (46 geoms), links base→wrist (44),
  camera_mount (6)
- **instance axis**: follower (48), leader (48)

Arms (14): v3, plate_only, bridges no_arm/only_arm, no_/only_ ×
{gripper, links, mount, follower, leader}. Per-part pixel fractions
recorded per slot.

## Registered anchors / aborts

In-run v3 must read 0.713 ± 0.005 else **ABORT** (no claims). Bridge
bands (fresh noise realization, non-abort sanity, flagged if missed):
plate_only 0.865 ± 0.02, only_arm 0.654 ± 0.02, no_arm 0.825 ± 0.02.

## Decision rule

Rank parts by paired Δknn5 CI95 (10k resamples, rng 0) of
only_<part> vs plate_only; a part is NAMED the fix target iff its CI
excludes 0 AND it carries ≥ 60% of the only_arm − plate_only paired
delta; if two parts each carry ≥ 35%, both are named (split verdict →
photometric fix targets both). Instance axis is context: the
follower-vs-leader read decides whether a fix must treat both
instances. No promotion, no production change from this leg — the
output is the named target(s) for the photometric ladder rung already
queued.

## Cost

CPU render + ~0.03 GPU-h embeds (14 arms × 100 frames + real groups).
Launch immediately per no-idle rules (GPU idle).
