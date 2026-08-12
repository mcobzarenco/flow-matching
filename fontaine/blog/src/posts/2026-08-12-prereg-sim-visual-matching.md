# Pre-registration: sim visual matching v1 — closing the appearance gap at the policy's eyes

*Registered 2026-08-12 ~04:3xZ (work session), the pre-reg promised
in-channel 01:30Z. Successor of the
[encoder OOD probe](../reports.md#encoder-ood-probe--sim-vs-real-at-the-policys-eyes-rides-the-sim100-pre-reg-owner-ask-0111z-08-12)
(which measured the baseline this work must move) and the
[sim100 eval](2026-08-12-sim100-results.md) (whose 0/500 close named
visual familiarity as the lever). Design citations:
[sim-as-eval](../papers/sim-as-eval.md) — SIMPLER's ablation puts
visual matching second after controller sysid (done:
[servo sysid](2026-08-11-sim-servo-sysid.md)); the REAL|SIM
side-by-side convention is LIBERO/SIMPLER's.*

## Plain words

Our simulated robot rig now moves like the real one, but it does not
*look* like it: the simulated camera images are pale, flat-lit, and
laid out subtly wrong, and we measured (with the policy's own vision
encoder) that the policy can tell sim frames from real ones almost
perfectly. That matters because our best policy barely touches the
boat in sim while moving confidently on the real rig — the images it
sees in sim are, to its eyes, foreign. This work re-paints and
re-frames the sim: real table wood baked into the scene, the clutter
laid out where it really sits, warm directional light with
episode-to-episode variation, and both cameras re-posed to match the
real views. The success test is the same encoder probe, re-run: if
the sim frames stop being separable from real frames (AUROC falling
from 0.885 toward 0.5, the can't-tell-them-apart point), the
matching landed — and only then do we spend GPU hours re-running the
100-seed eval.

## Baseline (measured, frozen)

From the closed encoder OOD probe
(er_60k eval-mount vision trunk, 5-NN cosine distance to the real_v2
reference half, AUROC sim-vs-held-out-real):

| camera | 5-NN AUROC | k(sim) | k(realB) | ratio |
|---|---|---|---|---|
| top | **0.885** | 1.87e-5 | 1.22e-5 | 1.54× |
| wrist | 0.828 | 2.21e-5 | 1.69e-5 | 1.33× |

Clean-repo control sits INSIDE the real spread (AUROC 0.26/0.28) —
the shift is sim-specific. Sim distances are ~7× too homogeneous
(render diversity is part of the gap). Per-tick flat — the gap is
the scene, not arm poses.

## Instrument: the reset-render probe (pinned)

Per-iteration read = `fontaine/scripts/sim_encoder_ood_probe.py`
with a new `--render-resets 100` mode: render the settled reset
frame of seeds 0..99 live (both cameras, 640×480, the exact
`SO101Sim.observe()` path the policy sees), embed through the same
frozen er_60k mount, same pinned real A/B/clean reference frames,
same reads. Justification, computed from the banked probe's stored
per-frame distances: restricting the banked sim set to tick 0 only
reproduces the full read — top 5-NN AUROC 0.887 (full 0.885), wrist
0.831 (full 0.828); centroid 0.809/0.718 (full 0.802/0.707). Reset
frames carry the whole signal.

Protocol order (each probe read ~0.02 GPU-h, foreground, idle local
H100; gate for the whole item: **≤ 0.5 GPU-h**):

1. **v0-render baseline** — run the reset-render probe BEFORE any
   visual change. Expected ≈ the tick-0 numbers above; this also
   prices the render-vs-H.264-video pipeline delta. **Tripwire: if
   the v0-render top 5-NN AUROC differs from 0.887 by more than
   0.05, stop and investigate before crediting any visual change.**
2. Visual matching passes (axes below), re-rendering side-by-sides
   per pass; probe re-read after each major pass.
3. **v1 read** — the registered read, same command on the final
   scene.
4. **Texture-sensitivity read** (record-only): seeds 0..19, 5
   appearance draws each (`--appearance-seed` decoupled), report the
   spread of per-draw mean k and the sim-internal pairwise
   homogeneity vs the real reference's (the 7× figure).

## Success bar (registered)

- **v1 lands** if the top-cam 5-NN AUROC on the reset-render
  pipeline drops by **≥ 0.10 absolute** vs the v0-render baseline
  (direction: toward 0.5). Wrist AUROC and both k-ratios recorded;
  wrist is secondary (smaller measured gap).
- If the bar is missed, the matching did NOT land:
  `sim100-v1-rerun` stays gated (its go/no-go is exactly this probe
  re-read) and the miss is reported as the result.
- AUROC below 0.5 (sim MORE typical than held-out real) would mean
  over-fitting the reference half — reported as a warning, not a
  win.

## Change axes (v1 scope, appearance-only)

Physics is untouched by construction: only lights, cameras,
materials, textures, and image post-processing may change. Oracle
pinned with the change: `reset(seed)` qpos is bit-identical across
appearance seeds and matches the pre-change scene.

1. **Table texture rebuild** — re-crop from the pinned real_v2 probe
   frames: the real walnut is far darker than the current bake, and
   the plank lines must run along +x (they currently read rotated
   90° in the top view).
2. **Top-cam scene layout** — clutter stand-ins moved/added to the
   real layout (mouse up-table center-right, dark laptop at the
   image-right edge, PCB + cable between the arms); floor/background
   tone.
3. **Camera pose/FOV** — top cam re-posed against the real frame
   (disk position/scale, arm scale in image); wrist cam re-posed:
   the real wrist view looks over the jaw tips at the table with the
   orange moving jaw on the image-LEFT; the sim currently views the
   gripper body from behind with the jaws mirrored.
4. **Lighting** — warm directional daylight look + per-reset jitter
   (direction, intensity, color temperature) from a dedicated
   appearance RNG; addresses the 7× homogeneity read directly.
5. **Object albedo** — disk toward the real beech; benchy tint
   distribution re-centered on the real light-gray print.
6. **Image post-processing (optional pass, only if the scene passes
   stall short of the bar)** — barrel distortion matched to the real
   130°-module fisheye (the real table edge visibly curves; sim is
   pinhole) and a fixed per-channel affine color grade computed once
   from v0 renders vs the real reference stats. Both applied inside
   `SO101Sim.observe()` as a `render_style="v1"` option — the same
   frames every downstream consumer sees.

## Determinism & seed policy

Spawn draws (benchy x/y/yaw) keep their RNG stream and order —
seed → identical benchy pose and qpos as the sim100 v0 runs.
Appearance draws move to a dedicated
`default_rng(appearance_seed ?? seed)` stream: same seed gives a
new (but deterministic) look, decoupled from physics. This is the
fresh-appearance analog of the house seed policy: physics
comparability preserved, appearance intentionally re-drawn.

## Deliverables

- `so101_sim.py` / scene deltas (all runtime or scene-XML —
  vendored menagerie files stay untouched).
- Before/after REAL | SIM side-by-side page (both cameras, several
  seeds) + probe numbers — blog post, house dark charts if a chart
  earns its place.
- Probe JSONs on fontaine-reports; `reports.md` section.
- `sim100-v1-rerun` boundary updated with the measured v1 read
  (go/no-go per its own item).

## Not in scope

Physics changes of any kind; re-running the 100-seed eval (successor
item, own gate); domain randomization for *training* (this is
eval-side matching); real-frame inpainting à la SIMPLER-RT (bake
approximations first — cheaper, and the probe tells us if they
suffice).
