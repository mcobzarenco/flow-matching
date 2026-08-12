# Pre-registration: sim wrist-cam periphery re-tune under the v1 fisheye

*Registered 2026-08-12 ~05:5xZ (work session; in-channel 05:59Z). Queue item
`sim-wrist-periphery-fix`, queued at the
[v1 close](2026-08-12-sim-visual-matching-results.md) where the
wrist was explicitly secondary. Instrument, references and bar
semantics inherited unchanged from the
[v1 pre-reg](2026-08-12-prereg-sim-visual-matching.md); the
[v2 close](2026-08-12-sim-visual-inpainting-results.md) adds the
wrist-composite honest negative (0.951 — mush plate) that keeps this
a *render-path* item, not an inpainting one.*

## Plain words

The robot's wrist camera in the simulator now bends its image like
the real wide-angle lens — but widening the view also changed *what
is in the picture*. The real wrist camera at an episode start looks
almost straight down at the table: wooden planks fill the whole
frame and only the two fingertip jaws poke into the bottom edge
(orange on the left, black on the right). Our simulated wrist camera
instead sees a huge dark mass of its own gripper body filling the
bottom half of the picture, because the wider lens pulls the
camera's own mount hardware into view. The policy's vision encoder —
our referee for "does sim look real?" — got *better* at telling sim
from real when we widened the lens (0.835 → 0.900), and we believe
this self-view is why. The fix attempted here is small and physical:
move and tilt the simulated camera on its bracket so the picture
shows what the real one shows — table everywhere, fingertips only at
the bottom. The referee score must come back down to at least the
level the camera had before the wide lens went in (0.786).

## Baseline (measured, frozen)

All numbers = wrist 5-NN AUROC, sim-vs-held-out-real, from the
banked probe jsons (`analysis__sim_visual_match_*.json`,
`analysis__sim_encoder_ood_probe_v2_shipped.json`):

| config | wrist 5-NN AUROC |
|---|---|
| v0 render | 0.835 |
| + scene pass (incl. first wrist re-pose, 52° pinhole) | **0.786** |
| + fisheye (72° source) | 0.870 |
| + grade | 0.904 |
| + sensor (= the SHIPPED wrist path, unchanged in v2) | **0.900** |
| v2 wrist composite (REJECTED, plate coverage 0.36) | 0.951 |

Top on the shipped default: 0.773 (v2 composite, registered bar
≤ 0.790 MET). Draw-to-draw noise from the registered sensitivity
read: per-draw mean k varies 0.8% (wrist).

**Diagnosis** (from the frame galleries): the 72° source pulls
sim-specific periphery into the wrist frame — the gripper/arm body
mass fills the bottom ~40% of the sim image, where real episode-start
frames show only slim jaw tips inside the bottom quarter over
full-frame table planks. Content composition, not image statistics,
is what the wrist read tracks (v1 close finding).

## Method (single axis: the wrist camera, runtime only)

Iterate ONLY `SO101Sim._repose_wrist_cam` (mount-local pos/quat,
both arms; vendored XML untouched), against the real episode-start
framing:

1. **Pose/height under the 72° source** — re-derive the lookat:
   camera forward/up on the bracket, pitch steeper toward the table,
   so the gripper body mass exits the frame and the jaw tips sit in
   the bottom quarter (orange moving jaw image-left, black fixed jaw
   image-right, like every real start frame).
2. **Periphery content** — after the pose is right, check what the
   frame edges hold vs real (table extent to every edge at the
   start pose; no sim floor band / table far edge unless real shows
   one). Any scene-XML delta this forces is named in the results
   post; none is expected.
3. **Visual iteration is encoder-free**: candidate poses are
   compared against pinned real_v2 **A-half** episode-start frames
   (the reference half — held-out B stays untouched, the AUROC read
   keeps its meaning). Only shipping candidates get an encoder read.

Not in scope: a per-episode-aligned wrist clean plate (noted at the
v2 close as a possible future axis), any top-cam change, any
physics/spawn change.

## Instrument (pinned, inherited)

`fontaine/scripts/sim_encoder_ood_probe.py --render-resets 100` on
the shipped default (`render_style="v2"`; wrist = v1 full-render
path + the candidate pose), same frozen er_60k eval-mount trunk,
same pinned A/B/clean references, ~0.02 GPU-h per read. Primary =
wrist `knn5_secondary.auroc_sim_vs_real`; the same json's top read
is the guard. Final pose also gets the 20-seed × 5-draw sensitivity
read (record-only).

## Success bar (registered)

- **Lands** if wrist 5-NN AUROC ≤ **0.786** (the scene-only level)
  on the 100-seed reset-render probe at the shipped default config.
- **Guard (tripwire)**: top 5-NN AUROC from the same read must stay
  ≤ **0.790** (its registered line). The pose change touches no
  top-cam state, so any top move beyond noise is an instrument red
  flag — stop and investigate, credit nothing.
- **Ship rule**: bar met → the pose delta ships as the
  `_repose_wrist_cam` default (all render styles share the repose;
  historical probe jsons remain the record of the old pose). Bar
  missed but final ≤ 0.88 (clearly below shipped 0.900 beyond the
  0.8% draw noise) → ship as a strict improvement, report the miss
  honestly. Final > 0.88 → no ship, honest negative, pose reverted.

## Budget & oracles

- GPU gate for the whole item: **≤ 0.2 GPU-h** of foreground probe
  reads (expect 2–4 reads + sensitivity).
- Oracles before commit: qpos bit-identity across v0/v1/v2 and
  across old-vs-new pose (the camera is render-only — physics must
  not move), spawn-stream identity, `check.py` green.
