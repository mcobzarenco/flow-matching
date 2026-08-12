# Sim visual matching v2: real-frame inpainting — the registered bar is MET

*2026-08-12 ~05:4xZ work session (real `date -u` at write: 05:46).
Executes the
[inpainting pre-reg](2026-08-12-prereg-sim-visual-inpainting.md)
(registered ~05:1xZ, same session). Verdict up front: the registered
bar — top-cam 5-NN AUROC ≤ 0.790 — is **MET**: 0.890 (v0) → 0.876
(v1) → **0.773 (v2)**. Compositing rendered dynamic content over a
real clean plate moved the encoder read further in one pass than
every v1 appearance axis combined. The wrist composite read *worse*
than the v1 wrist path (0.951 vs 0.900) — reported as the honest
secondary, and the shipped style falls back accordingly.*

## Plain words

Version 1 repainted the simulator to imitate the real scene and the
policy's vision encoder still wasn't fooled. Version 2 stops
imitating: we computed a "clean plate" from the real recordings —
the empty set, with everything that moves averaged away, like a film
studio's background shot — and now paste only the simulator's moving
parts (the arms, the toy boat, the goal disk, the desk clutter) onto
that real photograph. The wood grain, the room, the lighting falloff
in the corners: all literally real pixels now. The encoder's
tell-sim-from-real score dropped from 0.876 to 0.773, past our
pre-registered success line of 0.790 — the first registered win on
this axis. It is not finished (0.5 would mean indistinguishable, and
a perfectly re-recorded real dataset scores ~0.27), but the lever
finally moved, and it confirms where the remaining gap lives: in the
moving content itself and its variety, not the background.

## Registered reads (reset-render probe, top-cam 5-NN AUROC primary)

| config | top 5-NN | top centroid | wrist 5-NN | wrist centroid |
|---|---|---|---|---|
| v0-render baseline | 0.890 | 0.813 | 0.835 | 0.708 |
| v1 shipped (scene+fisheye+grade+sensor) | 0.876 | 0.801 | 0.900 | 0.749 |
| **v2 inpainting composite** | **0.773** | **0.730** | 0.951 | 0.844 |

k-ratios (sim vs held-out real): top 1.54× (v0) → 1.16× (v2);
centroid ratio 1.11×. The overfit tripwire did not fire (0.773 is
well above 0.5) — composites are closer to real but still on the
sim side of the held-out spread.

- **Top: bar met.** ≤ 0.790 registered, 0.773 read (100 seeds; a
  20×5 appearance-draw re-read gives 0.774 — the number is stable).
- **Wrist: honest negative.** The wrist clean plate is a
  cross-episode mush — episode-start wrist poses differ by degrees
  between episodes, so the median smears the wood grain into a
  featureless wash (per-pixel coverage 0.36 vs 0.56 for top) — and
  the composite regressed to 0.951. The wrist gap remains
  content/viewpoint-shaped, consistent with v1's finding; the
  queued wrist item owns it.
- **Homogeneity (record-only, registered)**: unchanged — sim k
  std/mean ~4% vs real ~45%, per-draw mean k spread 0.5%. A fixed
  real background does not diversify the embedding any more than
  lighting jitter did. The diversity axis is *content variation*
  (the real rows of the gallery include operator hands and, one
  day, a pile of mail on the table); plate banks / clutter states
  are the named lever.
- **A/B integrity**: plates are mined ONLY from episodes wholly
  inside the probe's reference half — verified in video-timeline
  frame indices (last plate frame 17066 < first held-out frame
  17100), so the held-out real set stays pixel-disjoint from
  everything a composite can contain.
- **Physics oracles green before any read**: settled qpos
  bit-identical across v0/v1/v2 and across appearance seeds; spawn
  stream bit-matches the banked sim100 spawns
  (`tests/test_sim_appearance.py`).

## Side by side

[Top camera — REAL | v1 | v2](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_visual_inpaint_top_before_after.png)
· [wrist](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_visual_inpaint_wrist_before_after.png).
The v2 column carries the real table, its corner falloff and the
off-table periphery verbatim; the visible tells are the rendered
clutter stand-ins (the too-white mug), missing contact shadows
under composited objects, and faint plate ghosts where the real
arms park.

## What landed (ships as `render_style="v2"`, the new default)

1. **Clean plates** (`fontaine/scripts/make_clean_plates.py` →
   `assets/real_plates/`): per-pixel median over the 26 A-half
   episodes — top from 1081 strided frames, wrist from 312
   episode-start frames — plus per-pixel coverage sidecars and a
   manifest pinning episodes/strides/commit. The real disk and
   on-table clutter *move between episodes* and median away (the
   operator repositions them), which is exactly why the composite
   renders them.
2. **Segmentation composite** (`SO101Sim._composite`): dynamic mask
   = every geom on a non-world body (both arms, benchy) + the named
   on-table statics (disk, mouse, mug, laptop, PCB) whose real twins
   left the plate; mask and frame share the fisheye remap (bilinear
   = ~1 px feather); foreground gets the v1 grade + PSF blur; sensor
   noise goes on the full frame (the median plate is denoised below
   single-frame noise).
3. **Wrist falls back to the v1 render path** inside v2 — the
   composite is measurably worse there (0.951 vs 0.900, same
   instrument). The pure-composite wrist read is reproducible at
   commit `f75c341`.
4. Default flipped `v1` → `v2`: on the pinned probe, v2 strictly
   dominates (top 0.876 → 0.773, wrist identical by construction).

## What this means for the rerun gate

`sim100-v1-rerun`'s go/no-go is exactly this probe re-read, and the
read now clears the registered line. The item stays **owner_hold**
(the 20-seed behavioral spot-check ask from the v1 close is still
pending, and the geometry argument it was probing is unchanged),
but the gate fact flips: by its own registered criterion the rerun
is now GO — with v2 frames, and the spot-check remains the cheaper
first step if the owner prefers.

## Next levers (named)

Content diversity: per-episode plate banks (needs a mining pass
that doesn't bake boat ghosts), real clutter-state variation, disk
position drawn from the real between-episode distribution (task
semantics — needs its own pre-reg, not appearance-only). Contact
shadows under composited objects. Wrist: the queued
`sim-wrist-periphery-fix` owns the wrist view.

## Artifacts

- Probe JSONs:
  [v2 primary](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v2_inpaint.json)
  · [20×5 homogeneity](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v2_homog.json)
  · [shipped config (wrist fallback)](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_encoder_ood_probe_v2_shipped.json).
- Total GPU spend this item: ~0.06 GPU-h of probe reads (gate 0.3).
