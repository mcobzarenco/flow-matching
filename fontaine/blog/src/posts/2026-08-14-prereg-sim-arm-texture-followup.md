# Pre-reg — arm micro-texture: print layers and servo glints

*2026-08-14, drafted 05:0xZ, posted in-channel before the gate read.
Queue item `sim-arm-texture-followup`; executes the residual branch the
[photometric-links close](2026-08-14-prereg-sim-arm-photometric-links.md)
registered — the grade closed the albedo/shine gap but left the graded
surfaces locally FLAT: real PLA carries print-layer relief (local
contrast 8.36 vs 4.66 graded) and the STS3215 servo casings a specular
glint tail (luma p97 205.6 / p99 250.0 real vs 125.2 / 127.2 graded).*

**Plain words.** Last session we matched the real arm's overall color
and shininess in the simulator, and the "looks fake" score improved.
But zoom into any real photo of the arm and the surfaces aren't smooth:
the 3D-printed parts show fine ridges from the printing process (like
the grain in corduroy), and the servo motors' glossy plastic casings
throw tiny bright sparkles. Our rendered arm is perfectly smooth — a
freshly airbrushed version of a well-used machine. MuJoCo can't add a
surface texture to an already-built scene without rebuilding it (which
risks disturbing the physics), so we paint the texture on at the
compositing stage instead: a fixed, deterministic grain pattern is laid
over exactly the arm's pixels — nothing else in the frame changes, and
the physics can't tell it's there. We tuned the grain strength and the
sparkle brightness so the *statistics* of the rendered arm pixels match
what we measured from real video. Then the registered question: does
the textured arm read less fake to the frozen encoder?

## Instrument (landed before this post)

- **Composite-stage micro-texture** (`arm_texture="v1"`, opt-in,
  requires `arm_photometrics="v1"` — fitted and gated as that
  combination; default path untouched): deterministic static
  screen-space fields built once at init from a PRIVATE pinned
  Generator (seed 20260814; the spawn/appearance/noise streams are
  untouched — zero per-frame draws). Two components, applied to the
  rendered top source frame under per-population segmentation masks
  (18 PLA + 12 servo geoms, both instances, count-pinned) BEFORE the
  production remap/blur/noise chain: a multiplicative zero-mean
  band-limited relief (~2 px correlation) and, on the servo casings, a
  graded push-toward-white speckle for the glint tail. Honest scoping:
  this is a **statistics stand-in**, not a physical model — the fields
  are screen-fixed rather than surface-tracking, so pooled per-pixel
  statistics (what the frame probe sees) are the target; video
  coherence is out of scope and stated so. Top camera only (the wrist
  view is `sim-wrist-view-material-read`, queued). 6 test oracles
  (`tests/test_arm_texture.py`) + init-time field checks: validation,
  pinned geom counts (18 PLA + 12 servo), field determinism +
  normalization, speckle density/peak bounds, identity at zero
  parameters, speckle never darkens/overshoots, zero shared-RNG draws.
- **Fit** (`fontaine/scripts/sim_arm_texture_fit.py`,
  `reports/analysis__arm_texture_fit.json`): solve-based, THROUGH the
  production v3 composite at the same real-registered poses ×
  appearance draws the photometric fit used, against the same mined
  real statistics (`reports/analysis__arm_photometric_mine.json`,
  436k PLA + 77k servo px). Amplitude per population by two-probe
  quadrature solve on the local-contrast target; servo speckle
  (density, gain) by a registered 2×2 grid on the p97/p99 tail loss
  plus a linear gain refine; servo amplitude re-solved with the chosen
  speckle live. The speckle *profile* took two pre-read iterations
  (recorded here, nothing was gated on them): a graded ramp left most
  speckle pixels too faint to survive the PSF blur (p97 moved ~8
  counts); raw binary blobs reached p97 175 but their sharp edges
  tripled local contrast (16.7 vs the 9.22 target); the frozen profile
  is binary blobs **softened into smooth peak-1 bumps** — full push at
  the center, gradient spread over the skirt, like a glint blooms.
  Frozen result (now `SO101Sim.ARM_TEXTURE_V1`): **PLA amplitude
  0.2321 (modulation only); servo speckle-only (amplitude re-solved to
  0), density 0.08, gain 1.0, soften 2** — confirm pass reads PLA
  local contrast **8.24** (real 8.36, graded floor 4.66) and servo
  **10.46** (real 9.22, floor 2.20); servo glint tail p97 **141.7** /
  p99 159.7 vs real 205.6 / 250.0 (floor 125.2 / 127.2) — the tail
  closes ~20% of its gap; the remainder is registered as the known
  residual of the screen-space stand-in (pushing single glints through
  the PSF to 250 triples local contrast first — the mjSpec
  surface-texture route is the queued escalation if this read
  underdelivers). The photometric guard loss (percentiles + medians)
  *improves* under the texture on both populations: PLA 4355 → 2358,
  servo 43624 → 27944.

## Question

Does closing the measured texture gap (print-layer local contrast, servo
glint tail) on top of the photometric grade move the rendered frame
toward real on the pinned encoder probe — and is the movement
attributable to the link pixels?

## Design

`fontaine/scripts/sim_arm_texture_read.py`: TWO production v3 instances
(numpy post) over the same 20 seeds × 5 appearance draws —
`arm_photometrics="v1"` alone (the banked baseline) and
`arm_photometrics="v1"` + `arm_texture="v1"` — hooked at `_composite`
exactly like the photometric read (noise-RNG state restored per arm;
frames pair 1:1 across arms AND instances). Arms (5): v3_photo,
plate_only, only_links_photo (baseline); v3_tex, only_links_tex
(patched). Encoder probe: er_60k trunk, top cam, knn5 vs held-out
real-B. In-run oracles: hooked frames bit-exact vs production
observations; qpos bit-equal across both instances per slot; patched
frames bit-equal outside the 16-dilated ARM-class mask (the texture is
arm-local; the arm halo swallows the table-reflection leak the mount
read documented, as it did for the photometric read).

## Registered anchors / aborts

In-run v3_photo must read **0.698 ± 0.005** (the banked
photometric-read anchor) else **ABORT** (no claims). Anchors: banked
v3 0.713, only_links_photo 0.652, plate_only ~0.866, real_fg 0.328.

## Decision rule (frozen before the read)

- **PRIMARY**: paired Δknn5 CI95 (10k resamples, rng 0) of v3_tex vs
  v3_photo entirely **below 0** (toward real) → the texture lands;
  joins the pending `sim-arm-photometrics-promotion` ask as a stacked
  option (owner sign-off, no default flip without it).
- **MECHANISM**: only_links_tex vs only_links_photo paired CI95 below
  0 — attribution to the named target.
- **Fail**: PRIMARY CI includes or exceeds 0 → screen-space statistics
  matching is insufficient (or the probe can't see relief at this
  scale); the mjSpec recompile route (true surface-tracking texture
  assets) stays queued as the escalation with the measured stats — not
  auto-run.
- **Both-ways report**: a CI *above* 0 (texture reads MORE fake) is
  reported as such — over-texturing is a real failure mode of
  screen-fixed grain.

## Cost

CPU renders (~100 paired slots × 2 instances) + ~0.02 GPU-h embeds
(5 sim arms × 100 + 400 real frames) on the er_60k trunk. GPU is
otherwise idle by design (R1-A boundary pends the owner call), so the
embed job has the card to itself.

---

## RESULTS (05:4xZ 08-14, executed same session — FAIL, the registered over-texturing direction)

All gates green: in-run v3_photo **0.698** dead-center in the abort
band; plate_only 0.866 and only_links_photo 0.652 reproduce the banked
anchors exactly; clean anchor 0.283; qpos bit-equal across both
instances × 100 slots; frames bit-equal outside the dilated arm mask.

- **PRIMARY — FAIL, decisively, in the *more fake* direction.** v3_tex
  vs v3_photo paired Δknn5 **+9.33e-07, CI95 [+8.27e-07, +1.04e-06]
  entirely ABOVE zero**, only 3/100 slots closer; AUROC
  **0.698 → 0.751**. The texture doesn't just fail to help — it undoes
  most of the photometric grade's gain (v3 was 0.713 before the grade).
- **MECHANISM — same direction, stronger.** only_links_tex vs
  only_links_photo **+1.30e-06, CI95 [+1.22e-06, +1.38e-06], 0/100
  closer**; AUROC **0.652 → 0.740**.
- **Reading.** The pooled per-pixel statistics moved *toward* real
  (local contrast dead-on for PLA, servo tail ~20% closer, photometric
  guard loss improved on both populations) while the encoder moved
  *away* — the probe is sensitive to the **spatial structure** of the
  texture, not just its pooled statistics. The zoom crop shows why:
  screen-fixed band-limited grain at amplitude 0.23 reads as blotchy
  mottling, not as coherent print-layer ridges; real relief is
  anisotropic, surface-tracking, and shading-coupled. Matching
  marginal pixel statistics at the composite stage is the wrong
  instrument class for texture. That kills this branch cleanly and
  cheaply (one session, ~0.02 GPU-h).
- **Disposition (per the frozen rule).** `arm_texture="v1"` stays
  opt-in and unpromoted; **no ask to the owner**. The escalation is
  queued, not auto-run: `sim-arm-surface-texture-mjspec` — a TRUE
  surface texture (mjSpec recompile path, UV-mapped anisotropic layer
  lines that track the geometry and couple to shading), with this
  read's stats and the physics-preservation oracles as its bar. The
  photometric grade (0.698/0.652) remains the arm-appearance frontier
  and its promotion ask stands unchanged.

Artifacts: [analysis](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_arm_texture_read.json)
· [fit record](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__arm_texture_fit.json)
· [chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__arm_texture_read.png)
· [frame strip](https://mcobzarenco-fontaine-reports.static.hf.space/strip__arm_texture_read.png)
· [arm zoom](https://mcobzarenco-fontaine-reports.static.hf.space/zoom__arm_texture_read.png)
