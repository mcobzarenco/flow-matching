# Pre-reg — arm link photometrics: a real-arm-derived material grade

*2026-08-14, drafted 01:4xZ, posted in-channel before the gate read.
Queue item `sim-arm-photometric-links`; executes the target the
[arm sub-part split](2026-08-13-prereg-sim-arm-split.md) named — links
carry 88% of the arm's keep-only delta on 6.1% of pixels, both
instances must be treated (follower/leader sub-additive ~77–79% each).*

**Plain words.** The encoder can tell our rendered robot arm is fake,
and the long links are the part that gives it away. Until now the sim
painted them a single flat near-black color. We went and *measured*
what the real arm's pixels actually look like — by posing the simulated
arm exactly where the real arm was in 142 real video frames, projecting
its silhouette onto those frames, and pooling the real pixels under it.
The real "black" arm is not flat at all: it is brighter than our paint,
cool-blue in cast (window daylight), and about a fifth of its pixels
are shiny highlights. Then we tuned the simulator's material (base
color, shininess) until the rendered arm's pixel statistics match the
measured real ones, and now we ask the registered question: does the
"looks fake" score actually improve?

## Instrument (landed before this post)

- **Mining** (`fontaine/scripts/sim_arm_photometric_fit.py mine`,
  `reports/analysis__arm_photometric_mine.json`): 26 reference-half v2
  episodes × 6 strided frames; follower qpos set from recorded
  `observation.state`, seg render through the production fisheye remap;
  **per-body darkness snap** (exact FFT argmin over ±60 px, ≥80% of the
  mask must stay in-frame) absorbs the image-space registration offset;
  per-body guards: ring ratio < 0.8 AND masked median luma < 100;
  wrist body excluded (dark distractors: PCB, cables); 142/156 frames
  harvested ≥1 confident body → 436k PLA px + 77k servo px.
- **Real reads**: PLA links median luma 65.7, channel medians
  [60, 66, 83] (cool cast), highlight fraction 0.159, p97 149.6.
  Servo casings: median 64.9, highlight fraction 0.183, p97 205.6.
  The flat recolor renders read hf 0.05/0.00 — the arm is matte-flat in
  sim and glinting in reality.
- **Fit** (`fit` subcommand, `reports/analysis__arm_photometric_fit.json`):
  albedo per channel by 2-point linear solve THROUGH the production v3
  composite at real-registered poses, specular × shininess by 4×4 grid,
  albedo re-solved at the winner. Frozen result (now
  `SO101Sim.ARM_PHOTOMETRICS_V1`):
  - PLA: rgba (0.1197, 0.1607, 0.2182), specular 1.0, shininess 0.1
  - servo: rgba (0.02, 0.02, 0.0661), specular 1.0, shininess 0.1
  - loss (weighted luma-percentile + channel-median SSE): PLA
    37180 → 4355, servo 99887 → 43624. Both populations chose the
    specular ceiling with the broadest highlight — the missing term was
    shine, not paint.
- **Sim surface** (`arm_photometrics="v1"`, opt-in, default untouched):
  material-level grade on the link PLA materials + STS3215 servo
  materials, BOTH instances; moving jaws and `wrist_roll_follower`
  (gripper/mount territory) untouched. 5 oracles
  (`tests/test_arm_photometrics.py`): default path byte-identical,
  graded materials exact, excluded materials untouched, zero RNG draws
  (qpos + noise-stream state bit-equal), validation. check.py 879.

## Question

Does the fitted material grade move the rendered arm toward real on the
pinned encoder probe — and is the movement attributable to the links?

## Design

`fontaine/scripts/sim_arm_photometric_read.py`: TWO production v3
instances (numpy post) over the same 20 seeds × 5 appearance draws —
default vs `arm_photometrics="v1"` — hooked at `_composite` exactly
like the arm-split leg (noise-RNG state restored per arm; frames pair
1:1 across arms AND instances). Arms (5): v3, plate_only, only_links
(baseline instance); v3_photo, only_links_photo (patched instance).
Encoder probe: er_60k trunk, top cam, knn5 vs held-out real-B. In-run
oracles: hooked frames bit-exact vs production observations; qpos
bit-equal across instances per slot; frames bit-equal outside the
12-dilated arm-class mask.

## Registered anchors / aborts

In-run v3 must read 0.713 ± 0.005 else **ABORT** (no claims). Anchors:
arm-split only_links 0.705, plate_only 0.866, no_mount removal best
0.654, real_fg 0.328.

## Decision rule (frozen before the read)

- **PRIMARY**: paired Δknn5 CI95 (10k resamples, rng 0) of v3_photo
  vs v3 entirely **below 0** (toward real) → the grade lands; queue the
  production-default promotion ask (owner sign-off, as with the clutter
  patch — no default flip without it).
- **MECHANISM**: only_links_photo vs only_links paired CI95 below 0 —
  attribution to the named target. Primary-pass with mechanism-fail is
  reported as "lands, attribution unclear" (the grade also touches
  gripper-servo casings via the shared material — same physical part).
- **Fail**: PRIMARY CI includes or exceeds 0 → the grade-only model is
  insufficient; the registered follow-up is texture (print-layer local
  contrast: real 8.4 vs graded 4.7; and the servo glint tail p97 206 vs
  125) — queued, not auto-run.
- **Record-only riders**: v3_photo AUROC vs the no_mount 0.654 removal
  best (does photometrics beat amputation?); the camera mount is WHITE
  in reality while the sim paints it black (mining overlays) — its
  material is shared with the gripper wrist-roll piece, so a mount fix
  needs its own material split; noted for the queue, not executed here.

## Cost

CPU renders (~100 paired slots × 2 instances) + ~0.02 GPU-h embeds
(5 sim arms × 100 + 400 real frames) on the er_60k trunk — run
alongside R1-A (34 GiB / 100%; the embed job is minutes-long and fits
the 45 GiB headroom; R1-A's ~48-min step pace absorbs the contention
noise).
