# Sim wrist-cam periphery: bar smashed — wrist AUROC 0.900 → 0.548, inside the real spread

*2026-08-12 ~06:2xZ (work session). Closes queue item
`sim-wrist-periphery-fix` per its
[pre-registration](2026-08-12-prereg-sim-wrist-periphery.md)
(registered ~05:5xZ, in-channel 05:59Z). Result: the registered bar
(≤ 0.786) was passed on the **first candidate pose** — the wrist
camera read is now at 0.548, a hair above the 0.5
can't-tell-them-apart point, and the sim wrist frames sit *inside*
the real embedding spread (k-ratio 0.97×).*

## Plain words

The fix was moving a camera ten centimeters. Our simulated wrist
camera sat on top of the wrist joint, behind the gripper — so when
we gave it the real rig's wide-angle lens, the picture filled up
with the robot's own dark plastic body, something the real camera
never sees. We slid the camera forward so it sits over the base of
the fingers, tilted it a little steeper toward the table, and the
picture became what the real one shows: wooden table everywhere,
two fingertips poking into the bottom edge. The policy's vision
encoder — which yesterday told sim wrist images from real ones 90%
of the time — now barely does better than a coin flip (55%). By
this measure the wrist view is the first camera whose sim images
are statistically inside the cloud of real ones.

## The read (registered instrument, 100-seed reset-render probe)

| config | wrist 5-NN AUROC | wrist centroid | k-ratio |
|---|---|---|---|
| v0 render | 0.835 | 0.708 | — |
| + scene pass (52° pinhole; the old bar) | 0.786 | 0.677 | — |
| shipped v1 wrist path (72° fisheye + grade + sensor) | 0.900 | 0.749 | 1.33× |
| **+ re-tuned pose (this item)** | **0.548** | **0.587** | **0.97×** |

- **Bar: MET** — 0.548 ≤ 0.786, and far past it: k(sim) 1.647e-5 vs
  k(realB) 1.691e-5 means the average sim wrist frame is now
  *closer* to the real reference set than the average held-out real
  frame is.
- **Guard: green** — top 5-NN AUROC 0.773, bit-identical to the
  shipped v2 read (the top path is untouched; also a determinism
  check of the instrument).
- **Sensitivity (registered, record-only)**: 20 seeds × 5 appearance
  draws reads 0.550 — the result is stable across appearance draws,
  not a lighting accident.

## What changed

One runtime pose in `SO101Sim._repose_wrist_cam` (vendored XML
untouched, both arms): the camera moves from the wrist top *behind*
the gripper (world ≈ (0.096, −0.004, 0.160) at home, 55° below
horizontal) to over the jaw base (≈ (0.150, 0.000, 0.150), 65°),
same image-right = −y roll convention. Under the 72° fisheye source
the old pose filled the bottom ~40% of frame with gripper-body mass;
the new pose drops the body out of frame, leaving the orange moving
jaw and black fixed jaw tips in the bottom quarter over full-frame
table — the composition of every real episode-start frame.

[REAL | old pose | new pose gallery](https://mcobzarenco-fontaine-reports.static.hf.space/chart__sim_wrist_periphery_before_after.png)
— three reset seeds vs three real episode starts.

The candidate was found in three encoder-free iteration rounds
(contact sheets vs pinned real_v2 A-half start frames — held-out B
untouched); only the shipping candidate got encoder reads. Total
GPU spend: ~0.04 GPU-h of probe reads (gate 0.2).

## What it means

- **Content composition is confirmed as the whole wrist story.** The
  v1 close said the wrist read tracks what is in the frame, not
  image statistics — this lands it: no texture, grade, or plate work
  moved the wrist below 0.786; a 10 cm camera move took it to 0.548.
- **The per-episode-aligned wrist plate axis (named at the v2 close)
  is retired.** A composite cannot beat inside-the-real-spread, and
  the mush-plate negative (0.951) is explained: the plate was built
  for a viewpoint whose own periphery was the problem.
- **The sim100 rerun gate now reads double-GO**: top 0.773 ≤ 0.790
  and wrist 0.548 ≤ 0.786 — both cameras at or under their
  registered lines. The rerun (`sim100-v1-rerun`) stays owner-held
  pending the 20-seed spot-check call; a pre-reg amendment draft is
  queued so the eval is launch-ready on unhold.
- **Caveat (stated at every close of this series):** the probe
  measures encoder separability of *reset* frames. Mid-episode wrist
  content (closed jaws, lifted boat, motion blur) is unmeasured, and
  encoder-indistinguishable does not imply behavior transfers — that
  is exactly what the held rerun would measure.

## Artifacts

- Probe jsons: `analysis__sim_wrist_periphery_fix.json` (100-seed
  primary), `analysis__sim_wrist_periphery_sensitivity.json` (20×5)
  on [fontaine-reports](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_wrist_periphery_fix.json).
- Gallery: `chart__sim_wrist_periphery_before_after.png` (link above).
- Oracles: 10 green (qpos bit-identity across render styles, spawn
  stream vs banked v0 — the pose is render-only), `check.py` 704
  green.
