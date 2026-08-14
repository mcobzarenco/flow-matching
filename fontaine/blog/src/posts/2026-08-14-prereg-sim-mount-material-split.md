# Pre-reg — camera-mount material split: the white bracket, measured

*2026-08-14, drafted 03:0xZ, posted in-channel before the gate read.
Queue item `sim-mount-material-split`; executes the arm-split rider the
[photometric-links close](2026-08-14-prereg-sim-arm-photometric-links.md)
queued — the mount is the per-pixel most sim-distinctive class:
`no_mount` was the ONLY removal that moved v3 toward real
(0.713 → 0.654 on ~0.66% of pixels), despite the absence-OOD confound.*

**Plain words.** The wrist camera on the real robot sits in a white
3D-printed bracket. Our simulator paints that bracket black — it
inherited the "recolor the whole arm black" rule, because in the
shipped robot model the bracket shares one material with a black part
of the gripper. So the single most tell-tale patch of rendered pixels
is a part we've been coloring exactly wrong: white in reality, black in
sim. The fix has two steps. First, an ownership split: give the gripper
piece its own copy of the color (the hand-off is invisible — verified
byte-identical) so the shared material now belongs to the bracket
alone. Second, measure what the real bracket's pixels look like — by
posing the simulated arm exactly where the real arm was in real video
frames and pooling the pixels under the projected bracket silhouette —
and tune the sim material until the rendered bracket matches. Then the
registered question: does the "looks fake" score improve?

## Instrument (landed before this post)

- **The split** (`mount_material="v1"`, opt-in, default untouched):
  the mount's visual geom shares `wrist_roll_follower_so101_v1_material`
  with the gripper's wrist-roll piece — and that material carries
  exactly mjv's material-less defaults (spec 0.5, shin 0.5, refl 0,
  emis 0, no texture). So the gripper geom detaches to `matid = -1`
  with the material's rgba copied into `geom_rgba` — **render
  byte-identical** (oracle: full top+wrist frame bit-equality) — and
  the material becomes mount-exclusive, fully gradeable with zero new
  slots, zero recompile, zero physics/RNG impact. 6 oracles
  (`tests/test_mount_material.py`): default path untouched, split +
  grade exact, nothing else touched, zero RNG draws, validation,
  split-alone byte-identity.
- **Mining** (`fontaine/scripts/sim_mount_material_fit.py mine`,
  `reports/analysis__mount_material_mine.json`): 26 reference-half v2
  episodes × 6 strided frames at recorded follower poses, production
  fisheye remap — the links-mine machinery with one structural change:
  the mount is WHITE, so the darkness snap cannot register it directly.
  The mount mask **rides its rigid dark neighbor's lock** — the gripper
  body (its parent, tried first) or the wrist — each snapped per-body
  (a single wrist+gripper union saturated its ±60 px search on ~2/3 of
  frames; the leader's identical dark cluster is in frame). Guards: the
  usual ring ratio < 0.8 + median luma < 100 on the locked dark body,
  plus a mount-plausibility guard — mount pixels must read ≥ 1.4× the
  locked body's luma (a wrong lock lands the mask on arbitrary
  content). **81/156 frames kept → 91k mount px** (bars: ≥60 frames,
  ≥20k px); locks: gripper 54, wrist 27; 17 frames rejected by the
  brightness guard.
- **Real read**: the mount region is **neutral light gray-white** —
  channel medians [123, 120, 125] vs the sim recolor's flat black
  (0.13·255 ≈ 33); luma p50 121, p90 193, p97 240; highlight fraction
  0.121; local contrast 24.3 (the mask region includes the dark camera
  PCB inside the bracket ring — the sim has no PCB model, so the fit
  targets the region's statistics as the encoder sees them). Overlays
  (12 dumped) confirm registration: the white bracket reads periwinkle
  under the 50% blue mask tint.
- **Fit** (`fit` subcommand,
  `reports/analysis__mount_material_fit.json`): albedo per channel by
  2-point linear solve (probes 0.3/0.8 — bright part) THROUGH the
  production v3 composite at real-registered poses, specular ×
  shininess by 4×4 grid, albedo re-solved at the winner. Frozen result
  (now `SO101Sim.MOUNT_MATERIAL_V1`): **rgba (0.4546, 0.430, 0.4311),
  specular 1.0, shininess 0.1** — the specular ceiling with the
  broadest highlight, exactly what both link populations chose; loss
  (weighted luma-percentile + channel-median SSE) **177188 → 9028**
  (~20×). The recolor-black mount composited at luma p50 55 vs real
  121; the fitted grade lands p50 121.4 and channel medians
  [124, 120, 126] vs real [123, 120, 125] — dead on. Known residual:
  highlight fraction 0.075 vs real 0.121 (the uniform geom can't
  reproduce the PCB-and-bracket structure).

## Question

Does the measured white-bracket material move the rendered frame toward
real on the pinned encoder probe — and is the movement attributable to
the mount pixels?

## Design

`fontaine/scripts/sim_mount_material_read.py`: THREE production v3
instances (numpy post) over the same 20 seeds × 5 appearance draws —
default, `mount_material="v1"`, and a **record-only combo** stacking
`arm_photometrics="v1"` on top (the pending promotion asks would flip
together; this prices the stack) — hooked at `_composite` exactly like
the links read (noise-RNG state restored per arm; frames pair 1:1
across arms AND instances). Arms (6): v3, plate_only, only_mount
(baseline); v3_mount, only_mount_v1 (patched); v3_full_fix (combo).
Encoder probe: er_60k trunk, top cam, knn5 vs held-out real-B. In-run
oracles: hooked frames bit-exact vs production observations; qpos
bit-equal across all three instances per slot; patched frames bit-equal
outside the 16-dilated MOUNT-class mask (the fix is mount-local); combo
frames bit-equal outside the 16-dilated arm-class mask.

## Registered anchors / aborts

In-run v3 must read 0.713 ± 0.005 else **ABORT** (no claims). Anchors:
arm-split only_mount 0.821, no_mount removal best 0.654, plate_only
0.866, real_fg 0.328.

## Decision rule (frozen before the read)

- **PRIMARY**: paired Δknn5 CI95 (10k resamples, rng 0) of v3_mount vs
  v3 entirely **below 0** (toward real) → the split+grade lands; joins
  the pending promotion asks (owner sign-off, no default flip without
  it).
- **MECHANISM**: only_mount_v1 vs only_mount paired CI95 below 0 —
  attribution to the named target (the fix touches ONLY mount pixels by
  the locality oracle, so a primary-pass with mechanism-fail would mean
  the whole-frame gain rides interaction with other content — reported
  as such).
- **Fail**: PRIMARY CI includes or exceeds 0 → a color+shine grade is
  insufficient for this part; residual candidates (bracket geometry
  mismatch, the missing camera-PCB dark mass inside the ring) go to the
  queue with the measured stats, not auto-run.
- **Record-only riders**: v3_full_fix vs v3 and vs v3_mount (the
  two-flag stack the promotion asks would flip); v3_mount AUROC vs the
  no_mount amputation best 0.654 (does painting it right beat cutting
  it off?).

## Amendment 1 (03:2x–04:2xZ, pre-read, in-run oracle only)

The first read attempt ABORTED itself at the patched pass's first slot:
8 pixels differed outside the 16-dilated mount mask. Diagnosed, not a
bug — the **tabletop plane carries reflectance 0.02**, so the table
faintly mirrors the scene and ANY arm color change moves its table
reflection; the mount is the tallest arm part, so its reflection lands
~35 px below the bracket, outside its own halo (the links read never
saw this only because the arm-class halo swallowed the arm's own
reflection). Measured: 8 px |Δ| = 1 at slot 0; up to 20 px |Δ| = 5 at
seed 2 (the fitted specular 1.0 puts glints on the mount, and 0.02 ×
full-scale 255 ≈ 5 counts bounds their reflection). The locality
oracle is AMENDED from bit-equality to the physical bound with the
mechanism named: outside the halo, ≤ 3000 px (~1% of frame) may differ
by ≤ 6 counts; per-pass leak maxima are recorded in the results JSON.
The registered decision reads (PRIMARY/MECHANISM CIs, abort band) are
untouched.

## Cost

CPU renders (~100 paired slots × 3 instances) + ~0.02 GPU-h embeds
(6 sim arms × 100 + 400 real frames) on the er_60k trunk — run
alongside R1-A (34 GiB / 100%, ~41 GiB headroom; minutes-long embed
job, absorbed by R1-A's ~48-min step pace).

---

## RESULTS (04:4xZ 08-14, executed same session — split verdict, adjudicated by the frozen rule)

All gates green: in-run v3 **0.713** dead-center; bridges reproduce the
arm-split anchors exactly (plate_only 0.866, only_mount 0.821); clean
anchor 0.283; qpos bit-equal across all three instances × 100 slots;
locality within the amended reflection bound (≤24 px, ≤5 counts, both
passes, vs 3000/6 allowed).

- **PRIMARY — FAIL.** v3_mount vs v3 paired Δknn5 **+6.8e-08, CI95
  [−0.07e-07, +1.42e-07] includes zero** (45/100 closer); AUROC
  0.713 → 0.713. At ~0.66% of pixels, the fixed part sits below the
  whole-frame read's detection floor. Per the frozen rule: **no
  promotion ask for `mount_material` alone.**
- **MECHANISM — PASS, decisively.** only_mount_v1 vs only_mount
  **−1.03e-06, CI95 [−1.16e-06, −0.90e-06] entirely below zero**,
  93/100 closer, AUROC **0.821 → 0.793**. And vs the bare plate the
  graded mount reads **−2.67e-06 with 100/100 slots closer** — the
  amputation confound REVERSED: with the measured color, mount
  *presence* beats absence. The grade is right; the frame just can't
  see 0.66% of pixels through this probe.
- **Record-only rider (the stack).** v3_full_fix (mount + photometrics)
  vs v3: **−1.49e-07, CI95 [−2.45e-07, −0.57e-07] entirely below
  zero**, 61/100, AUROC **0.713 → 0.702**; vs v3_mount −2.17e-07
  CI-excludes-zero (72/100) — the photometrics term carries the stack.
  Implication for the pending promotion asks: if `arm_photometrics`
  flips, `mount_material` rides at zero measured frame-level cost —
  the owner's call, stated without a recommendation to flip it alone.

Disposition: the part is fixed at the mechanism level and banked
opt-in; no follow-up mount item queued (nothing further to execute
unless the promotion flips it into production). The next arm item
remains the registered texture follow-up (print layers + servo glint
tail).
