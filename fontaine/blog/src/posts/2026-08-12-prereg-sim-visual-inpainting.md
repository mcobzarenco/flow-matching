# Pre-registration: sim visual matching v2 — real-frame inpainting

*Registered 2026-08-12 ~05:3xZ (work session). Successor of
[visual matching v1](2026-08-12-sim-visual-matching-results.md),
whose registered bar was missed (top-cam 5-NN AUROC 0.890 → 0.876 vs
≤ 0.790): approximating the real scene's materials, optics and color
statistics did not move the encoder read. This is the lever named at
that close — stop approximating the background and bake the real
pixels in (the SIMPLER-RT recipe; see
[sim-as-eval](../papers/sim-as-eval.md)). Instrument, references and
bar semantics are inherited unchanged from the
[v1 pre-reg](2026-08-12-prereg-sim-visual-matching.md).*

## Plain words

Version 1 tried to make the simulator's picture look real by
imitation: repainting the table, moving the lamps, bending the lens.
The policy's own vision encoder still told sim from real almost
perfectly. Version 2 stops imitating the background and uses the
real thing: we take the real camera recordings, compute a "clean
plate" — the scene with the moving parts averaged away, like a film
studio's empty-set shot — and paste the simulator's moving parts
(the two robot arms, the toy boat, the wooden disk) onto that real
photo, using the simulator's own knowledge of which pixel belongs to
which object. Everything that made the sim look fake but never
mattered to the task — wood grain, room clutter, light — is now
literally real; only the objects that move are still drawn. The test
is unchanged: if the encoder can no longer separate composited sim
frames from held-out real frames, the gap is closed.

## Baseline (measured, frozen)

Reset-render probe on the shipped v1 default (sensor read,
05:0xZ 08-12): **top 5-NN AUROC 0.876**, wrist 0.900 (scene-only
wrist was 0.786 — wrist is content-sensitive and stays secondary).
v0 baseline 0.890/0.835. Sim is ~10× too homogeneous under the
encoder; lighting jitter does not fix it.

## Method (v2 scope)

1. **Clean plates** (`fontaine/scripts/make_clean_plates.py`): per
   camera, per-pixel median over frames drawn ONLY from real_v2
   episodes lying wholly inside the probe's reference half A (first
   half of the concatenated timeline, episode boundaries from the
   dataset parquet) — the held-out B episodes stay pixel-disjoint
   from everything the composite can contain, so the AUROC read
   keeps its meaning. Top: strided frames across those episodes
   (arm/boat/operator move → median removes them; the static disk
   stays). Wrist: episode-START windows only (arm at rest = the
   settled-reset viewpoint the probe renders). Plates + a coverage
   sidecar (per-pixel fraction of frames near the median) land in
   `assets/real_plates/` with the generating command pinned.
2. **`render_style="v2"`** (`SO101Sim`): render the source pinhole
   frame plus a segmentation pass; dynamic mask = every geom on a
   non-world body (both arms, benchy) **plus the disk** (the queue
   item's arms/benchy/disk set — the rendered disk overlays its real
   twin, and any misalignment shows as a double edge in the
   composite gallery rather than silently biasing physics-vs-visual
   target). Fisheye-remap frame AND mask (bilinear = free ~1 px
   feather), grade + PSF-blur the rendered foreground only (the
   plate is already through the real optics), composite over the
   plate, then sensor noise on the FULL frame (the median plate is
   denoised below single-frame noise; restoring it is part of
   matching).
3. Appearance jitter continues to act on the rendered foreground
   (lighting on arm/boat, boat tint). The background is a fixed
   photo per camera in this pass.

## Reads (instrument pinned, order registered)

Same command class as v1, foreground, idle H100; **gate ≤ 0.3 GPU-h
for the whole item** (~0.02 GPU-h per read):

1. **v2 read (registered primary)** — reset-render probe, seeds
   0..99, v2 default: top/wrist 5-NN AUROC + k-ratios + centroid
   secondaries.
2. **Homogeneity read (record-only)** — seeds 0..19 × 5 appearance
   draws: does a fixed real background move the ~10× sim-internal
   homogeneity figure, given jitter now only touches the foreground?
3. Physics oracles before any read is credited: reset qpos
   bit-identical across render styles v0/v1/v2 and across appearance
   seeds; spawn stream bit-matches banked sim100.

## Success bar (registered, inherited)

- **v2 lands** if top-cam 5-NN AUROC ≤ **0.790** (v1's registered
  line: ≥ 0.10 absolute below the 0.890 v0 baseline). The
  interesting regime is further: ≤ 0.65 would put sim inside sight
  of the clean-repo control (0.26–0.28 = same-rig re-record).
- **Overfit tripwire**: AUROC < 0.5 means composites sit closer to
  the A reference than held-out real does — reported as a warning,
  not a win (the composite contains literal A pixels; that is the
  method, stated openly — the B-side pixel-disjointness is what
  keeps the read honest).
- Miss → reported as the result; `sim100-v1-rerun` gate updates
  either way with the measured v2 read.

## Not in scope (named follow-up levers)

Per-episode plate banks / plate warps for content diversity (risk:
baked boat ghosts — needs its own mining pass); shadow synthesis for
the composited foreground (rendered shadows fall on a rendered
table that is no longer shown; the pasted look this leaves is
accepted and inspected in the gallery); grade re-fit against
foreground-only statistics; re-running the 100-seed policy eval
(successor item, own gate).
