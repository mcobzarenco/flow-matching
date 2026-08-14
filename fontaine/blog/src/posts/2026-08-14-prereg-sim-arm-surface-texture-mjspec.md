# Pre-reg — TRUE arm surface texture via the mjSpec recompile path

*2026-08-14, drafted 09:2xZ, posted in-channel before the gate read.
Queue item `sim-arm-surface-texture-mjspec`; the escalation registered
by the [micro-texture
refutation](2026-08-14-prereg-sim-arm-texture-followup.md) (05:4xZ
08-14): statistically-matched screen-space grain read MORE fake — the
encoder wants coherent surface-tracking structure, not matched
marginals. This read prices that structure.*

**Plain words.** The real robot arm is 3D-printed, and printing leaves
fine horizontal ridges — layer lines — on every surface. Our simulated
arm is perfectly smooth. Last time we tried to fake the ridges by
sprinkling matched noise over the arm's pixels on screen, and the
policy's vision encoder called the bluff: the sprinkle looked *less*
real, because the encoder cares about structure that sticks to the
surface, not about pixel statistics. So this time we did it properly:
we baked a real striped texture into the arm's materials inside the
physics model itself, so the stripes live on the surface and move with
it. The physics is provably untouched — the recompiled model is
bit-identical in every physical field. Now we render 100 identical
scene pairs — smooth arm vs striped arm — and ask the encoder which
looks more like real robot footage.

## What landed before this read (instrument facts, commit `e408f9e`)

- **`arm_texture='v2'`**: a quasi-periodic layer-line texture asset
  (256², pinned private RNG seed 20260814) built at compile time and
  cube-mapped onto the 18 PLA link materials via an mjSpec recompile.
  Cube shrink-wrap anchors the bands in OBJECT space — they track the
  surface through every pose, the property the refutation demanded and
  screen-space fields cannot have. Servo casings out of scope (their
  residual is a specular glint tail; no specular-map path in the
  classic renderer).
- **Physics hard bar: 11/11 oracles green** — every physics field of
  the recompiled model bit-equal to the `from_xml_path` baseline, ids
  unrenumbered, qpos trajectories bit-equal over settled resets AND a
  scripted 60-tick excursion, shared RNG streams untouched.
- **Zero-clip generator**: tanh soft-bound keeps the texture strictly
  inside `center·(1±amplitude)` — zero clipped texels (clipping would
  silently break the grade-time mean compensation; the renderer
  MODULATES lit color by the texture, so PLA albedo is divided by the
  realized texture mean and the fitted photometric grade is preserved
  exactly).
- **Reflection rider (registered)**: the tabletop reflects the arm
  (`mat_reflectance` 0.02), and a TRUE surface texture rides that
  physical light path — mechanism confirmed by zeroing the reflectance
  (0 out-of-halo diff px). The locality oracles therefore allow
  out-of-arm-halo diffs ONLY on the dilated reflective-geom mask,
  |Δ| ≤ 24, < 1% of frame; anything else is a leak and aborts.
- **Fit honesty
  (`reports/analysis__arm_surface_texture_fit.json`)**: period chosen
  by lc-response probe over {6, 10, 16, 24, 32} texture rows/line —
  the response is monotonic in period (fine bands die in the composite
  blur chain); 32 is frozen at the registered plausibility bound (≥ 8
  bands per link face; coarser stops being print-layer-like).
  Amplitude quadrature-solved on the real PLA local-contrast median
  8.36 and **capped at the 0.42 no-clip headroom: realized lc 6.43**
  vs 4.66 graded — the instrument closes **~41% of the quadrature lc
  gap and cannot close the rest without clipping** (same finding class
  as the v1 servo contrast cap). This read prices a *partial-fidelity*
  coherent texture, stated as such.

## Question

Does TRUE surface-tracking layer-line structure on the printed links —
at the maximum no-clip amplitude — move the top view toward real on
the pinned encoder probe, where the statistically-matched screen-space
grain moved it AWAY?

## Design

`fontaine/scripts/sim_arm_surface_texture_read.py`: TWO production v3
instances (numpy post) over the same 20 seeds × 5 appearance draws —
`arm_photometrics='v1'` alone vs grade + `arm_texture='v2'` — hooked
at `_composite` exactly like the micro-texture read (arms: v3_photo /
plate_only / only_links_photo vs v3_surf / only_links_surf). Encoder
probe: er_60k trunk, knn5 vs held-out real-B, top camera. In-run
oracles: hooked frames bit-exact vs production observations; qpos
bit-equal across instances per slot; strict outside-arm-halo
bit-equality off the reflective rider region; rider bounds as
registered above.

## Registered anchors / aborts

**ABORT (no claims)** unless in-run v3_photo knn5 AUROC in
**0.698 ± 0.005** (the banked photometric-read anchor; same gate the
micro-texture read used and passed at 0.6977). Anchors: banked v3
0.713, only_links_photo 0.652, real_fg 0.328, clean anchor; the
REFUTED micro-texture primary **+9.33e-07 CI95 [+8.27, +10.42]e-07
(AUROC 0.698 → 0.751)** — the delta this escalation must NOT
reproduce.

## Decision rule (frozen before the read)

- **PRIMARY**: paired top Δknn5 CI95 (10k resamples, rng 0) of
  v3_surf vs v3_photo entirely **below 0** → coherent
  surface-tracking structure reads more real where matched marginals
  read more fake — the refutation's mechanism hypothesis is
  CONFIRMED; `arm_texture='v2'` becomes a promotion candidate at its
  declared partial fidelity, and closing the residual lc gap (a
  clip-free channel: normal-map-free band contrast is exhausted)
  becomes a priced follow-up.
- **CI95 entirely ABOVE 0** → SECOND refutation, now WITH coherent
  structure: the failure was never about coherence, and the
  arm-texture direction goes cold at this abstraction level —
  v1-graded stays the arm frontier; magnitude compared against the
  micro-texture's +9.33e-07.
- **CI95 straddles 0** → no measurable effect at partial fidelity:
  the texture stays opt-in and unclaimed; the residual-fidelity
  follow-up is priced but NOT auto-queued (the channel is capped —
  a straddle at 41% of the gap does not license extrapolation).
- **MECHANISM (record + interpretation aid)**: same rule for
  only_links_surf vs only_links_photo — the links-only composite is
  the sharper version of the same question (the micro-texture's
  mechanism read was +1.30e-06, unambiguous).
- **Record-only riders**: only_links_surf vs plate_only; clean
  anchor; reflection-rider per-slot stats (max |Δ|, changed
  fraction).

## Cost

CPU renders (100 paired slots × 2 instances + per-slot seg renders) +
~0.02 GPU-h embeds (500 sim + 400 real frames) on the er_60k trunk.
GPU is idle by design (R1-A boundary pends the owner call) — the
embed job does not conflict.

---

## RESULTS (09:2xZ 08-14, executed same session — SECOND REFUTATION, adjudicated by the frozen rule)

All gates green: in-run v3_photo knn5 AUROC **0.698** dead-center in
the abort band; hooked frames bit-exact vs production ×100; qpos
bit-equal across instances ×100; strict outside-halo equality held
(zero leak pixels); reflection rider **silent in the composited
output** (max |Δ| 0 — the ≤14-count raw reflection is fully absorbed
by the PSF blur + uint8 quantization; the rider bound was needed for
the raw-frame oracles only).

- **PRIMARY — CI entirely ABOVE zero → the registered second
  refutation.** v3_surf vs v3_photo paired Δknn5 **+3.07e-07, CI95
  [+2.42, +3.71]e-07** (14/100 slots closer); AUROC **0.698 →
  0.718**. TRUE surface-tracking structure at the maximum no-clip
  amplitude reads MORE fake, exactly like the screen-space grain did —
  the coherence hypothesis is disconfirmed as the missing ingredient.
  Per the frozen rule: **the arm-texture direction goes cold at this
  abstraction level; `arm_photometrics='v1'` (0.698) stays the arm
  frontier.** Magnitude context: about a third of the micro-texture's
  harm (+9.33e-07, 0.751) — coherent structure hurts *less*, but it
  confidently hurts.
- **MECHANISM — consistent.** only_links_surf vs only_links_photo
  **+1.98e-07, CI95 [+1.36, +2.59]e-07** (27/100 closer), AUROC 0.652
  → 0.671 (micro-texture: +1.30e-06, 0.740).
- **Mechanism observations (diagnostic, not new claims):**
  1. The cube shrink-wrap does NOT render clean horizontal layer
     lines — on several link faces the bands come out as radial
     "sunburst" fans (visible in the strip's Δ panel). The realized
     structure is coherent and surface-tracking but not
     print-layer-like everywhere.
  2. Our bands are pure albedo modulation. Real print layers are
     RELIEF — their contrast is shading/specular structure that moves
     with the light, which the classic renderer cannot express
     without a normal-map path (the same limitation that put servo
     glints out of scope). Two refutations now say the encoder
     rejects added arm content at this rendering abstraction; the
     surviving hypothesis is that the residual lives in
     light-transport structure, not albedo statistics or albedo
     geometry.
- **What this buys the ledger**: the escalation branch registered by
  the micro-texture refutation is now CLOSED with a measured answer,
  the v1-graded arm stays the production frontier, and no
  further texture rung is auto-queued (the clip-capped albedo channel
  is exhausted; a normal-map/renderer-upgrade rung would be a new
  design decision, priced only if the owner wants it).

Artifacts: [analysis](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_arm_surface_texture_read.json) ·
[fit record](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__arm_surface_texture_fit.json) ·
[chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__arm_surface_texture_read.png) ·
[strip](https://mcobzarenco-fontaine-reports.static.hf.space/strip__arm_surface_texture_read.png)
