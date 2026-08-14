# The renderer-class decision — what an arm the encoder believes would cost, and what it would buy

*2026-08-14 · decision brief, banked numbers only (no run, no new
claims). Queue item `renderer-class-decision-brief`. The appearance
screen is closed and the wrist follow-ups are banked; this
consolidates them into the one decision they all point at — and that
decision is the owner's to make.*

**Plain words.** Over three days we measured exactly which pixels let
a real-photo-trained network catch our simulator lying. Cheap fixes
are done: pasting real photo crops over the fake clutter recovered
most of what the overhead camera could recover, and a measured
re-paint of the robot arm rides along for free. What's left is not
paint. The arm's *shape and shading* — the ridged texture of 3D-printed
plastic, the way light glints off and shadows curl around it — is
beyond what our renderer can draw, and it is now the whole remaining
gap on both cameras. Fixing that means upgrading how the simulator
draws, not what it draws: a "renderer-class" change. This brief prices
that decision: what it would plausibly buy on each camera, what the
tiers of effort look like, what stays unknown until it's done — and a
recommendation to pilot it small before buying it whole.

![Where each camera stands, and the three facts that price the upgrade](https://mcobzarenco-fontaine-reports.static.hf.space/chart__renderer_class_decision.png)

*Left: the banked position of each camera on the shared instrument
(knn5 AUROC vs held-out real, er_60k probe; lower = reads more real),
with the span a renderer-class fix addresses. Right: the three paired
reads that price it, on one axis — the pose-switched arm term dwarfs
both the content term (nil) and the material-stack regression.*

## The case, in three banked facts

**1. Top camera: the stack lands at 0.552; the floor is 0.328.** The
[full opt-in stack](2026-08-14-prereg-sim-full-optin-stack.md#results-1058z-08-14-executed-same-session--middle-branch-stack-beats-v3-misses-the-best-single-bar-by-0001)
(clutter patches + arm photometrics + mount fix) is measured safe and
banked; the [foreground-swap
decomposition](../reports.md#top-cam-gap-decomposition--the-whole-0713-residue-lives-in-the-rendered-foreground-pixels-08-13)
showed that substituting *real* dynamic pixels reaches 0.328 through
the same compositor. The remaining **−0.224 AUROC is entirely the
rendered arm** — and the [two texture
refutations](2026-08-14-appearance-screen-report.md) pinned what kind
of problem it is: not albedo. Statistically-matched grain read *more*
fake (0.698 → 0.751); true surface-tracking bands baked into the
materials still read more fake (0.698 → 0.718). The surviving
hypothesis is **relief and light transport** — print-layer ridges as
shading structure, specular glints (the real arm's pixels are 16–18%
glint), soft self-shadowing.

**2. Wrist camera: 0.877 at manipulation poses, and it is the arm.**
At reset poses the wrist camera is near-honest (0.523). Posed at the
exact joint states the real robot recorded mid-episode, it reads
**0.877** ([rollout-pose
read](2026-08-14-prereg-sim-rollout-pose-wrist.md), calibration
direction understating it) — and the paired pose-effect rider puts the
whole gap on the pose switch: +8.71e-06 Δknn5, 1/100 slots closer. The
[content split](2026-08-14-prereg-sim-wrist-content-split.md) then
discharged the one registered caveat in the strengthening direction:
deleting the benchy moves nothing (+3.28e-07, CI straddling zero,
blind-slot control ≈ 0, benchy-px↔Δ correlation 0.011). **The rendered
arm itself carries the manipulation-pose wrist gap.** Scene-content
matching is off the table as a fix.

**3. The measured materials stop working exactly where the arm fills
the frame.** The photometric grade is fitted from real-arm pixel
measurements and validated on the top camera — yet at manipulation
poses the same flags *regress* the wrist read (+3.99e-07, CI entirely
above zero, 22/100). The measurement isn't wrong; the renderer can't
cash it. A flat-shaded surface wearing measured brightness/glint
values still doesn't shade like a ridged, glinting surface up close.
This is the classic renderer failing on its own terms, at the poses a
wrist-conditioned policy actually sees.

## What "renderer-class" concretely means here

The scene renders through MuJoCo 3.11's built-in fixed-function
OpenGL renderer (`mujoco.Renderer`), then the top view goes through
the production compositor onto real background plates; the wrist view
is the raw render. Three facts bound the engineering:

- **The arm meshes are STLs** (17 of them, Menagerie `robotstudio_so101`)
  — STL carries no UV coordinates, so nothing 2D can be mapped onto
  them today. The banked texture read had to use a *cube-mapped*
  procedural texture for exactly this reason. Any per-surface detail
  (normal maps included) needs the meshes re-exported with UVs first.
  We have this pipeline already: `convert_benchy.py` does
  STL → decimate → xatlas-UV → OBJ for the benchy. Extending it to
  the arm is a one-off script, not research.
- **The model format already speaks PBR; the renderer doesn't.**
  MuJoCo 3.x materials accept texture roles — `NORMAL`, `ROUGHNESS`,
  `METALLIC`, `ORM`, `EMISSIVE` — but the built-in renderer consumes
  only the RGB role; the others exist for external renderers. So the
  asset side of a PBR upgrade is native MJCF/mjSpec work, while the
  *drawing* side necessarily means a second render path (an external
  PBR/raytrace renderer consuming the posed scene — per-frame pose
  export is trivial; our reads are offline CPU renders of ~200
  frames, so render throughput is a non-issue for read harnesses).
- **The compositor stays.** Plates, fisheye lens model, grade, sensor
  noise are all validated (the pipeline reaches 0.328 with real
  foreground pixels). A renderer swap only replaces the rendered-
  foreground input to the existing, anchored pipeline.

## Cost tiers

**Tier 0 — composite/albedo texture (spent, refuted twice).** The
cheap tier is measured cold: both refutations above. Further spend
here buys negative value. *Zero cost, zero (or negative) return —
done arguing.*

**Tier 1 — in-classic mjSpec work (cheap, can't reach the target).**
The texture read already built the mjSpec recompile path
(`_compile_arm_surface_texture`). What classic rendering can still
express — per-material specular/shininess, light placement — is
either already fitted (the photometric grade *is* the measured
specular story) or can't encode relief: the classic renderer has no
normal-map input at all. This tier cannot express the surviving
hypothesis. *Days of work available, but the screen says the payload
isn't here.*

**Tier 2 — asset re-export + PBR render path (the actual decision).**
Three parts, roughly independent:

1. *Meshes*: arm STLs → UV-mapped OBJs via the existing benchy
   pipeline (script-sized; the gripper + camera-mount meshes — the
   per-pixel worst offenders in the [sub-part
   split](../reports.md#arm-sub-part-split--the-links-carry-88-of-the-arms-signature-mounts-are-the-per-pixel-worst-pre-reg-08-13)
   — first).
2. *Detail*: print-layer relief is parametric (known layer height,
   known print orientation per part) — normal maps can be *baked
   procedurally*, no scanning required for the links. Real-gripper
   geometry (wear, chamfers, the taped cable runs the STLs don't
   model) is the open-ended part; a scan/photogrammetry pass is the
   honest version, and it's the least bounded cost in the tier.
3. *Renderer*: a PBR path (external renderer consuming the posed
   scene each frame) for the foreground layer, feeding the existing
   compositor. The plumbing is bounded; **the real cost is
   validation**: re-fit the wrist lens model against the new
   renders, re-fit the photometric grade under a shading model that
   can finally cash it, re-pin the render oracles, re-replicate the
   anchor bands (0.713/0.523 replicated to the digit four times —
   that discipline is what makes the banked ladder comparable, and a
   new render path restarts it).

My honest sizing: parts 1–2 (links only, procedural relief) are a
few sessions; part 3's validation tail is the majority of the tier
and is the part that recurs into every future read.

## What it would plausibly buy — and what stays unpriced

- **Top camera**: the addressable span is measured on both ends —
  **0.552 → 0.328 at perfection** (−0.224). Real but modest: the top
  camera already has its payload (clutter patches), and the
  compositor bounds how wrong the arm can look from above.
- **Wrist camera**: the addressable span is **0.877 → toward 0.523**
  (−0.355) — but only the top end is measured. No wrist-side floor
  read exists (it would need real-frame arm segmentation, the rider
  we registered as out of scope), and residual content terms the
  benchy split can't touch (no clutter in the sim wrist view, leader
  arm at home, no boat-in-jaw states) sit somewhere inside that
  span. The upgrade's wrist ceiling is genuinely unknown.
- **Unpriced either way**: whether 0.877 *costs policy success*. The
  whole instrument is an encoder-honesty proxy — the north star is
  transfer on the rig, and no read yet connects wrist-camera
  dishonesty to success-rate loss. That connection is measurable
  (deterministic-seed relative screens à la the Squint note in
  `ideas.md`) without any renderer work.

## Recommendation

**Don't buy Tier 2 outright on these numbers — pilot it.** The
top-side case alone doesn't justify the validation tail (−0.224
ceiling on a camera whose payload is already banked). The wrist-side
case is the real one (−0.355 with the calibration direction
understating it, materials regressing, content ruled out) — but its
ceiling is unmeasured and its policy cost is unpriced. Two bounded
moves dominate the decision tree, in either order:

1. **The pilot** (CPU + ~0.02 GPU-h class, reuses the banked
   harness verbatim): re-export *only* the wrist-visible meshes
   (gripper, camera mount, forearm links) with UVs + procedurally
   baked layer-line normal maps, render the same 100 pose-matched
   manipulation slots through a PBR path, score on the same
   instrument against the same gates. That directly measures the
   one number that decides the tier: how much of 0.877 a
   relief-and-light-transport arm recovers. If it moves materially
   toward the reset band, Tier 2 is bought with evidence; if it
   doesn't, the surviving-hypothesis story is wrong and we saved
   the whole tier.
2. **The transfer read**: a closed-loop relative screen pricing
   whether wrist dishonesty moves success rate at all. If it
   doesn't, the renderer decision stops mattering for the north
   star and 0.877 is a known, tolerated proxy artifact.

The pilot is the natural next pre-registration when a window opens;
the transfer read needs its own design work (sim-adaptation sanity
arm included) before it's pre-registrable. Neither launches anything
until the GPU reserve lifts, and both are the owner's call — this
brief is the priced menu, not a commitment.

## Artifacts

- Lead chart: [per-camera position + the three pricing facts](https://mcobzarenco-fontaine-reports.static.hf.space/chart__renderer_class_decision.png)
  (`renderer_decision_chart.py`, banked JSONs only)
- Every input number: the [appearance-screen consolidated
  report](2026-08-14-appearance-screen-report.md), the [rollout-pose
  wrist read](2026-08-14-prereg-sim-rollout-pose-wrist.md), and the
  [wrist content split](2026-08-14-prereg-sim-wrist-content-split.md),
  each with frozen analysis JSONs linked from the
  [reports ledger](../reports.md).
