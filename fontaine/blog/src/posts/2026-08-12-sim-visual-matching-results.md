# Sim visual matching v1: the sim looks real now — the encoder still isn't fooled

*2026-08-12 ~06:5xZ work session. Executes the
[visual-matching pre-reg](2026-08-12-prereg-sim-visual-matching.md)
(registered ~04:3xZ, same session). Verdict up front: the registered
bar — top-cam 5-NN AUROC down ≥0.10 from the v0-render baseline —
was **MISSED** (0.890 → 0.876). Every appearance axis we named
landed, the renders are dramatically closer to the rig frames, and
the policy-eye probe barely moved. The discriminating signal lives
somewhere appearance-matching at this level doesn't reach.*

## Plain words

We rebuilt the simulated scene to look like the real robot rig: real
table wood cropped from actual rig photos, the clutter (mug, mouse,
laptop, circuit board) laid out where it really sits, warm daylight
that varies episode to episode, both cameras re-aimed to match the
real views — including reproducing the real cameras' wide-angle lens
distortion, which visibly bows the table planks in every real frame,
and their auto-white-balance color response. Side by side, the new
sim frames are hard to mistake for the old ones ([top
camera](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_visual_match_top_before_after.png),
[wrist
camera](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_visual_match_wrist_before_after.png)).
Then we asked the policy's own vision encoder whether it can still
tell sim from real — and it can, almost exactly as well as before.
The test we registered in advance (drop the tell-them-apart score by
at least 0.10) failed. That is a real and useful negative: it says
the gap is not in scene layout, lens geometry, or color statistics,
but in something finer — most likely the micro-texture of real
camera images and the sheer variety of real content (hands, cables,
reflections) that our clean renders lack. The next lever, which the
field's standard recipe (SIMPLER) jumps straight to, is pasting
*actual real camera pixels* into the sim views rather than
approximating them.

## Registered reads (reset-render probe, top-cam 5-NN AUROC primary)

| config | top 5-NN | top centroid | wrist 5-NN | wrist centroid |
|---|---|---|---|---|
| v0-render baseline | **0.890** | 0.813 | 0.835 | 0.708 |
| + scene pass (texture, layout, lights, poses) | 0.892 | 0.845 | **0.786** | **0.677** |
| + fisheye remap | 0.874 | 0.813 | 0.870 | 0.736 |
| + color grade | 0.881 | 0.819 | 0.904 | 0.760 |
| + sensor blur/noise (labeled amendment) | 0.876 | **0.801** | 0.900 | 0.749 |

- **Baseline tripwire passed**: v0-render top 0.890 vs the banked
  tick-0 0.887 — the live-render pipeline reproduces the
  rollout-video read, so the iteration loop measures what the eval
  measures.
- **Registered bar: top ≤ 0.790. Missed** — best top config differs
  from baseline by ~0.015, inside session-to-session noise.
- **Wrist DID respond to content**: the scene pass (mostly the
  camera re-pose — the menagerie wrist cam had the moving jaw
  mirrored to the wrong side and stared into the gripper body) moved
  wrist 5-NN 0.835 → 0.786 and centroid 0.708 → 0.677. The fisheye +
  grade passes then *hurt* wrist (0.900) — with the wider source
  FOV, more sim-specific periphery (arm body, table edge) enters the
  frame. Content composition, not image statistics, is what the
  wrist read tracks.
- **Sensitivity read (registered)**: 20 seeds × 5 appearance draws —
  per-draw mean k varies **0.4%** (top) / 0.8% (wrist); per-seed
  spread across draws ~3%. And the sim set remains ~10× more
  homogeneous than real at the encoder (k std/mean 4% vs 45%).
  Lighting/tint jitter does not diversify the embedding; real-frame
  diversity comes from *content* (hands, boat pose, motion,
  reflections), not illumination.

## What landed in the repo (ships as `render_style="v1"`, default)

1. **Table texture rebuilt** from the pinned real frame — correct
   plank direction (along +x), ~7 cm plank scale, contrast
   compressed to the measured real std; central-band table stats now
   match to ~2/255 per channel (mean 166/159/150 vs real
   164/158/150).
2. **Real clutter layout** — white mug + dark mouse up-table, laptop
   at the image-right edge, PCB between the arms, office-chair and
   floor-bag stand-ins in the background band the fisheye brings
   into view.
3. **Wrist camera re-posed** (runtime, vendored XML untouched) —
   looks over the jaw tips at the table, orange moving jaw on the
   image-left like every real frame; 16:9 sensor model swapped for
   the module's 4:3 fovy.
4. **Fisheye remap** — both cameras render a 72° pinhole source
   remapped through a center-matched equidistant model (center
   magnification = the previously-matched 52° view); real-frame
   plank bowing and periphery reproduced.
5. **Per-reset appearance jitter** from a dedicated RNG stream
   (lighting direction/intensity/temperature, table tone, benchy
   tint re-centered on the real light-gray print).
6. **Color grade + sensor emulation** — fixed per-channel affine to
   the real AWB/contrast response; Gaussian PSF + sensor noise
   (deterministic per seed). The sensor pass is an **amendment** to
   the registered post-process axes, labeled as such.
7. **Physics untouched, oracle-pinned** (`tests/test_sim_appearance.py`,
   5 green): settled qpos bit-identical across appearance seeds and
   render styles; the spawn stream still bit-matches the banked
   sim100 v0 spawns.

## What this means for the rerun gate

`sim100-v1-rerun`'s go/no-go is exactly this probe read, so by its
own gate the ~2–4 GPU-h rerun should NOT auto-launch. One honest
argument the other way, for the owner to weigh: the probe measures
*encoder separability*, not *policy behavior*. The camera-geometry
fixes (fisheye + wrist re-pose) change **where things appear in the
image** — the er60k arm's reach-over-the-table fingerprint (96/100
seeds untouched, systematic overshoot along +x) is exactly what a
pinhole-vs-fisheye spatial mismatch would produce, and a policy can
be geometrically mis-aimed without the encoder read moving at all.
SIMPLER's own fidelity metric was policy-behavior correlation, not
an OOD score. A 20-seed er60k spot-check (~0.5 GPU-h) would answer
it cheaply. Held for owner steering; by the registered gate, we do
not spend it unilaterally.

## Next lever (named, not queued)

Real-frame **inpainting** à la SIMPLER-RT: bake actual rig pixels as
the static scene (table + background billboard from real frames,
render only arm/boat/disk), rather than approximating materials and
optics. The pre-reg's "Not in scope" note said *"bake approximations
first — cheaper, and the probe tells us if they suffice."* The probe
has now answered: they don't.

## Artifacts

- Probe JSONs (all five configs + sensitivity), before/after
  composites: `analysis__sim_visual_match_*.json`,
  `chart__sim_visual_match_{top,wrist}_before_after.png` on
  [fontaine-reports](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_visual_match_v1sensor.json).
- Instrument: `sim_encoder_ood_probe.py --render-resets N
  [--appearance-draws K]` — the per-iteration read for any future
  matching work (~0.02 GPU-h).
- Total GPU spend this item: ~0.12 GPU-h of probe reads (gate 0.5).
