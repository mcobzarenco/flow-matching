# Pre-reg — wrist content split: how much of the 0.877 is the benchy, not the arm?

*2026-08-14, drafted 15:2xZ, posted in-channel before the read. Queue
item `sim-manip-wrist-content-split` — the registered caveat of the
[rollout-pose wrist read](2026-08-14-prereg-sim-rollout-pose-wrist.md)
(banked 12:2xZ: manipulation-pose wrist AUROC **0.877**, gap real):
scene content was unmatched, so the 0.877 bundles "the rendered arm
looks fake up close" with "the sim scene contains a benchy at its
spawn while real mid-grasp frames hold the boat elsewhere". This read
prices the benchy term of that bundle.*

**Plain words.** The last read showed the wrist camera looks clearly
fake during manipulation — but the comparison wasn't entirely fair to
the simulator's arm. In the sim images the toy boat sits untouched at
its starting spot on the plate; in the real images, mid-episode, the
robot has usually picked the boat up or knocked it around, and the
table is messier. Some of the "fakeness" score might therefore be the
*scene contents* differing, not the *arm rendering* being bad. The
test: re-render the exact same 200 images with the boat deleted from
the scene, and see how much the score moves. If deleting the boat
barely moves the score, the arm rendering carries the blame and the
expensive renderer upgrade stays justified. If it moves a lot, the
camera is more honest than 0.877 suggested and cheap scene fixes
climb the priority list.

## Design (one new knob on the banked harness, everything else verbatim)

Harness = `sim_rollout_pose_wrist_read.py` run 3, verbatim: same 100
pose-matched slots (held-out episodes 26–49, mid-band `[0.3T, 0.7T)`
picks, timestamp-exact real-frame decode), same 150-frame manipulation
reference (episodes 0–19), same 50-pick episode-disjoint calibration
holdout (episodes 20–25), er_60k trunk knn5, 20×5 (seed,
appearance-draw) schedule.

TWO production v3 instances, BOTH default materials + fitted
curve-only lens + re-tuned pose + numpy post — i.e. both are the
banked read's *default* arm; the instances differ by zero flags:

- **PRESENT (in-run anchor)**: the banked default instance verbatim —
  reset pass (banked 20×5 protocol) + manip pass (arm qpos overwritten
  to slot i's recorded real state, ctrl-clipped, `mj_forward`,
  production `observe()`), benchy at its seeded spawn.
- **ABSENT**: identical call sequence; in the manip pass only, after
  the arm-qpos overwrite the benchy free joint is relocated to
  **(0, 0, −10)** (10 m below the scene) before `mj_forward` +
  `observe()`. Reset pass untouched (benchy at spawn) — so the reset
  passes of the two instances are bit-identical by construction and
  the RNG streams (appearance, content, sensor noise) stay aligned:
  every paired manip slot differs ONLY in benchy presence, same
  lighting draw, same noise draw.

Kinematic removal (no settle) matches the banked pass-2 semantics.

## Feasibility (verified pre-reg, no claims)

- Benchy visible in the wrist raw segmentation in **61/100** manip
  slots: mean 4,337 px, median 1,967, max 57,409 (18.7% of frame)
  when visible. The read is not vacuous — and the 39 benchy-blind
  slots come along as a free within-run control (their paired deltas
  isolate the indirect term: shadows/bounce, not silhouette).
- Relocation to (0, 0, −10): **0 benchy px in all 100 slots**;
  production `observe()` runs clean post-relocation.

## Registered gates (frozen before the read)

**ABORT (no claims) unless ALL of:**

- In-run reset TOP knn5 AUROC in **0.708–0.718**; reset WRIST AUROC
  in **[0.49, 0.57]** (the banked bands, third+1 replication).
- Calibration (ref-holdout vs held) **≤ 0.65**, directional per
  banked Amendment 2; below 0.35 the low-note applies to
  AUROC-vs-real readings (see SECONDARY).
- Cross-instance oracles: reset frames **bit-identical** (changed-px
  = 0 — stricter than the banked 5% cap; the instances differ by zero
  flags, any nonzero is RNG divergence and aborts); arm-qpos
  bit-equality ×100 manip slots; benchy px = **0** in every ABSENT
  wrist segmentation.
- In-run replication anchor: PRESENT manip AUROC in **[0.86, 0.89]**
  (banked 0.877; sim-side spread across the three banked runs
  0.874–0.877).

## Decision rules (frozen)

- **PRIMARY — the content term**: paired Δknn5 per slot,
  **ABSENT − PRESENT**, CI95 (10k resamples, rng 0).
  - CI entirely **< 0** (removal moves sim closer to real) → the
    benchy-at-spawn is a measured fake cue. Content share =
    point-Δ / banked pose-effect (+8.71e-06). Share **≥ 50%** → the
    0.877 materially overstates the arm term and the honest wrist
    number is meaningfully lower — cheap content matching outranks
    the renderer-class decision wrist-side. Share < 50% → the arm
    still carries the majority; renderer-class keeps its wrist price,
    now net of the measured content term.
  - CI **straddles 0** → the benchy content term is NIL — the
    rendered arm (+ residual unmatched content) carries the
    pose-switched gap; the renderer-class arm-appearance decision
    keeps its full wrist-side price.
  - CI entirely **> 0** (clean table reads MORE fake) → removal is
    anti-matching (real mid-grasp scenes contain the boat somewhere);
    the arm term stands at least at its measured share, and content
    matching means matching the grasp state, not deleting the object.
- **SECONDARY (descriptive, calibration-caveated)**: ABSENT manip
  AUROC vs held real, next to PRESENT's. Under calibration < 0.35
  only a fake-side (≥ 0.75) reading is claimable (banked Amendment-2
  logic); an honest-side ABSENT number would be recorded as
  directional evidence only, not a verdict.
- **Record-only riders**: Δ split by benchy-visible (61) vs blind
  (39) slots — the blind-slot deltas price the indirect
  (shadow/bounce) term; per-slot benchy px vs |Δ| relation; ABSENT
  changed-px stats vs PRESENT (uncapped at manip by design).

The queue item's optional "real-frame arm-crop rider" is **dropped**
(registered): separating arm pixels in real frames needs a real-frame
segmenter the harness doesn't have — out of scope for this read.

## Cost

CPU renders (2 instances × 200 obs) + **~0.02 GPU-h** embeds (~1,200
frames, er_60k trunk). The GPU is **OWNER-RESERVED** (12:54Z): renders
+ frame cache land now; the embed step waits for the in-channel
release or an explicitly offered gap, per the 12:55Z commitment.

---

## RESULTS (15:2xZ 08-14, single run, all gates green — the content term is NIL; the rendered arm carries the whole manipulation-pose gap)

![Wrist content split](https://mcobzarenco-fontaine-reports.static.hf.space/chart__wrist_content_split.png)

*Execution note: the owner 👍'd the pre-reg post (which carried the
gap ask); read as pre-reg ack + gap-go, stated in-channel 15:21Z with
a veto window before the embeds fired at 15:26Z. GPU use: ~30 s on an
otherwise 0 MiB card; the reserve was otherwise untouched.*

All gates green: reset top **0.713** (band 0.708–0.718) and wrist
**0.523** (band [0.49, 0.57]) — fourth consecutive replication to the
banked digit; calibration **0.268** ≤ 0.65 (low-note active, caveating
AUROC readings only); render-stage oracles all green (reset frames
bit-identical ×100, arm qpos bit-equal ×100, benchy px 0 ×100 in
ABSENT); in-run replication anchor: PRESENT manip AUROC **0.877** —
the banked number to the digit, in [0.86, 0.89].

- **PRIMARY — CONTENT NIL.** Paired Δknn5, ABSENT − PRESENT:
  **+3.28e-07, CI95 [−2.26e-07, +8.39e-07]** — straddles zero, only
  20/100 slots closer after removal. Content share of the banked pose
  effect: **−3.8%** (and pointing the wrong way — removal reads
  trivially *more* fake, not less). Deleting the benchy does not make
  the wrist view honest: the ABSENT arm still reads **0.888** AUROC
  (fake-side, claimable under the low-note). Under the frozen rule the
  benchy content term is NIL — **the rendered arm (plus residual
  unmatched content that deletion can't touch) carries the
  pose-switched gap, and the renderer-class arm-appearance decision
  keeps its full wrist-side price.**
- **Riders**: the split by visibility confirms the null is not
  dilution — the 61 benchy-visible slots read +5.34e-07 CI
  [−3.91e-07, +1.36e-06] (straddling), and the 39 blind slots
  +6.4e-09 (the shadow/bounce term is ~zero, 1/39 slots moved at
  all); benchy-px↔|Δ| correlation 0.011 — even the slots where the
  benchy fills up to 19% of the frame don't move when it vanishes.
  Manip changed-px mean 9,003 / max 99,443 — the removal did change
  what the camera saw; the encoder just doesn't care.

**What this closes.** The banked 0.877's registered caveat is
discharged in the direction that strengthens it: the number is not a
scene-content artifact, and with the calibration direction understating
it, **0.877 stands as the wrist camera's honest-camera failure at
manipulation poses, attributable to the rendered arm itself**. The
renderer-class decision (normal-map/PBR + gripper geometry) now owns
the full wrist-side price; cheap content matching is off the table as
a fix (deleting the distractor moves nothing, so placing/matching it
better is not where the gap lives either — the blind-slot null and
the zero px↔Δ relation both say the encoder's attention is on the
arm). The three pending promotion asks are unchanged from the banked
read's disposition.

Ledger: CPU renders alongside the owner-reserved (idle) GPU window +
**~0.005 GPU-h** embeds in an explicitly-cleared ~30 s gap; single
run, zero aborts, all four banked anchors replicated to the digit.
