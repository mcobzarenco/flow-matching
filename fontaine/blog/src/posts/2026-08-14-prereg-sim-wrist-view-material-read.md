# Pre-reg — wrist-view read of the arm material fixes

*2026-08-14, drafted 05:5xZ, posted in-channel before the gate read.
Queue item `sim-wrist-view-material-read`; the wrist-side fact the two
pending promotion asks
([photometrics](2026-08-14-prereg-sim-arm-photometric-links.md),
[mount](2026-08-14-prereg-sim-mount-material-split.md)) currently
assume rather than measure — the photometrics results post named the
wrist view the promotion sanity.*

**Plain words.** We fixed two things about how the simulated robot arm
looks — its surface colors and shine, and the color of the white
camera bracket — and both fixes are opt-in flags waiting on a
promotion decision. Those fixes were measured through the robot's
*top* camera. But the robot has a second eye: a camera on its own
wrist, inches from the very surfaces we recolored. Before anyone flips
the flags on, we should check the fixes don't make the wrist view
*worse*. This read renders 100 paired wrist frames — identical scenes,
flags off vs on — and asks the policy's own frozen vision encoder
which version looks more like the real robot's wrist footage.

## What the feasibility probe found (pre-read, shapes the design)

- **Both flags are model-level material writes** — they change every
  camera. The wrist frame is a *raw* render (bit-identical v2↔v3 by
  the registered wrist guard), so no composite hook is needed: the
  production `reset()` observations pair 1:1 across instances.
- **The reset-pose wrist effect surface is small**: the stack changes
  ~0.5% of wrist pixels (~1,670 of 307,200), mostly |Δ| 1–2 counts —
  at the settled home pose the wrist camera sees graded surfaces only
  in the periphery/distance.
- **Anchor honesty**: the queue item's banked wrist anchors (knn5
  AUROC 0.828, ratio 1.33×, centroid 0.707) are from ROLLOUT frames
  (ticks 0/300/600 of the sim100 videos — mid-episode poses). Under
  settled RESET renders — this read's protocol, the only one that
  pairs — the wrist baseline is near-chance: **0.5442 / 0.5476** on
  the two banked 100×1 reads. The gate is anchored to the reset
  numbers; the rollout gap is registered as an explicit limitation.
- **Appearance draws do vary the wrist frame** (scene-wide lighting
  draws; measured ~100% of pixels across draws), so the 20×5 schedule
  gives 100 distinct wrist slots, matching the established protocol.

## Question

Does the two-flag stack (`arm_photometrics='v1'` +
`mount_material='v1'` — the exact combination the promotion asks would
flip) move the WRIST view toward or away from real on the pinned
encoder probe?

## Design

`fontaine/scripts/sim_wrist_material_read.py`: TWO production v3
instances (numpy post) over the same 20 seeds × 5 appearance draws —
default and the two-flag stack — production observations, both
cameras. Encoder probe: er_60k trunk, knn5 vs held-out real-B per
camera (wrist real = `observation.images.wrist`, 300 v2 + 100 clean
strided frames). In-run oracles: qpos bit-equal across instances per
slot (the grades consume no RNG draws); per-slot wrist changed-pixel
fraction ≤ 5% (an RNG-stream divergence flips ~100% of pixels through
sensor noise; measured ~0.5%). Diagnostics recorded: per-slot raw-seg
visibility of the graded classes (pla / servo / mount) in the wrist
view; changed-px stats.

## Registered anchors / aborts

**ABORT (no claims) unless BOTH**: in-run v3 TOP knn5 AUROC in
**0.713 ± 0.005** (the established 20×5 protocol gate) AND in-run v3
WRIST knn5 AUROC in **[0.50, 0.60]** (band around the banked 100×1
reset baselines 0.5442/0.5476 — wide because no 20×5 wrist anchor
exists and appearance draws redraw scene lighting). Anchors: wrist
rollout baseline 0.828 (context only — different pose distribution,
NOT this read's gate); mount-read top rider (the same stack, top cam)
−1.49e-07 CI95 [−2.45, −0.57]e-07.

## Decision rule (frozen before the read)

- **PRIMARY** (as queued): paired wrist Δknn5 CI95 (10k resamples,
  rng 0) of v3_stack vs v3 entirely **below 0** → the stack helps the
  wrist view too; the promotion asks gain a wrist-side plus.
- **CI95 entirely ABOVE 0** → wrist-side REGRESSION: flagged on both
  pending promotion asks (the texture lesson — fixes can read more
  fake); magnitude reported against the top-side gain.
- **CI95 straddles 0** → wrist-neutral: the promotion-relevant
  finding — the flags don't perturb the wrist view measurably; the
  asks proceed on top-side evidence alone, stated as such.
- **Record-only riders**: top paired stack vs v3 (should replicate the
  mount read's −1.49e-07 rider — a protocol cross-check); wrist clean
  anchor; graded-visibility diagnostic.
- **Limitation (registered)**: this read is at settled RESET poses.
  The 0.828 rollout-pose wrist gap — where the gripper fills the frame
  mid-manipulation — is a DIFFERENT fact this read does not touch; if
  the owner wants the rollout-pose read it needs banked rollout
  trajectories (only videos exist) or fresh policy rollouts, priced
  separately.

## Cost

CPU renders (100 paired slots × 2 instances) + ~0.02 GPU-h embeds
(400 sim + 800 real frames) on the er_60k trunk. GPU is idle by
design (R1-A boundary pends the owner call) — the embed job does not
conflict.

---

## RESULTS (06:0xZ 08-14, executed same session — wrist-neutral, adjudicated by the frozen rule)

All gates green: in-run v3 TOP knn5 AUROC **0.713** dead-center
(band 0.708–0.718); in-run v3 WRIST **0.561** inside [0.50, 0.60];
qpos bit-equal across both instances × 100 slots; wrist changed-px
tripwire quiet (max 1,714 px = 0.56% of frame, |Δ| max 12, mean
1,670 px — right at the feasibility measurement).

- **PRIMARY — CI STRADDLES ZERO → wrist-neutral.** Paired wrist
  Δknn5 of the stack vs v3: **−1.39e-08, CI95 [−4.53e-08, +1.73e-08]**
  (46/100 slots closer); AUROC 0.561 → 0.560. Per the frozen rule:
  the promotion-relevant finding — **the two-flag stack does not
  perturb the wrist view measurably in either direction** at reset
  poses. No wrist-side regression (the texture-lesson failure mode
  did NOT fire); no wrist-side plus to claim either. The pending
  promotion asks proceed on top-side evidence alone, now stated as
  measured rather than assumed.
- **Why it's null, mechanically (diagnostic, registered
  record-only):** at the settled home pose the wrist camera sees
  **~230 raw px of graded surface total** — servo 208, PLA links 21,
  mount **1 px** (of a 640×480 raw frame; visible in all 100 slots
  but never more than ~0.08% of the frame). The stack changes ~0.5%
  of output wrist pixels at |Δ| 1–2 counts (PSF spillover around the
  servo edge, cf. the strip's amplified-Δ panel). There is nearly
  nothing for the encoder to read.
- **Record-only rider — top REPLICATED EXACTLY.** Top paired stack
  vs v3: **−1.4937e-07, CI95 [−2.451, −0.570]e-07** (61/100), AUROC
  0.713 → 0.702 — bit-for-bit the mount read's combo rider. Expected
  in hindsight (deterministic renders, same schedule, same weights)
  but it is a real cross-check all the same: the mount read's
  `_composite` hook path and this read's plain production `reset()`
  observations produce **identical frames and identical embeddings**
  — the hook was bit-exact, and the two code paths agree.
- **Anchors:** wrist clean 0.265, wrist held-out knn5 1.691e-05
  (both matching the banked probe exactly); the wrist reset baseline
  landed at 0.561 vs the banked 100×1 reads' 0.544/0.548 — inside
  the registered band, the 20×5 lighting draws worth ~+0.015.

**Disposition.** Item closed; no follow-up wrist item queued from
this result — the reset-pose wrist fact is measured and null. The
**registered limitation stands**: the 0.828 ROLLOUT-pose wrist gap
(gripper filling the frame mid-manipulation) is a different,
still-open fact — reading it needs banked rollout trajectories or
fresh policy rollouts; it remains priced separately and is noted on
the promotion asks, not auto-queued.

Artifacts: [analysis](https://mcobzarenco-fontaine-reports.static.hf.space/analysis__sim_wrist_material_read.json)
· [chart](https://mcobzarenco-fontaine-reports.static.hf.space/chart__wrist_material_read.png)
· [frame strip](https://mcobzarenco-fontaine-reports.static.hf.space/strip__wrist_material_read.png)
