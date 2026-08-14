# Pre-reg — rollout-pose wrist read: is the wrist camera honest where it matters?

*2026-08-14, drafted 11:5xZ, posted in-channel before the read. Queue
item `sim-rollout-pose-wrist-read` — the one unmeasured leg the
[appearance-screen consolidated
report](2026-08-14-appearance-screen-report.md) flags: every wrist
number so far is a settled RESET pose (0.548–0.561 band, fitted lens
0.523), while the banked 0.828 wrist anchor is from ROLLOUT frames —
mid-episode poses where the gripper fills the frame.*

**Plain words.** The robot's wrist camera has looked honest in every
test so far — but every test posed the arm at its resting position,
where the wrist camera mostly sees empty table. During actual
manipulation the arm bends over the workspace and the camera stares
at the gripper, the arm's own surfaces, and whatever it is grabbing,
from inches away. An old measurement on outdated renders said the
wrist view looks very fake at exactly those poses. This read re-asks
that question with today's best visuals, and fairly: we pose the
simulated arm at the *exact joint angles the real robot recorded*
mid-episode, so the comparison is appearance against appearance, not
pose against pose. We also re-test the two pending arm-material
fixes at these poses — at reset they touched ~230 pixels and read
neutral; mid-manipulation the same surfaces fill the frame.

## Premise correction (registered, from the git audit)

The queue item says "render at banked rollout trajectories' recorded
qpos" — **no such traces exist**. The banked sim100 artifacts are
videos + distance/grip traces; `EpisodeResult` has no qpos field, and
the 0.828 anchor was measured on rollout VIDEO frames (old visuals:
pre-lens-fit, pre-pose-retune, ticks 0/300/600). Re-deriving sim
rollout qpos means re-running policy inference (GPU-h class — not
this item). The executable and *stronger* pose source: the REAL
held-out episodes' recorded `observation.state`. Rendering the sim
arm at real recorded joint angles pose-matches every slot — it
removes the pose-distribution confound that the 0.828-vs-reset-band
comparison always carried, and it lands exactly on the item's
question: is the wrist camera honest at the manipulation poses the
policy will actually see?

## Design

Feasibility verified pre-reg (no claims): real `observation.state`
[T,6] degrees round-trips exactly into sim qpos via
`np.deg2rad` + `mj_forward` (kinematic pose, physics-free —
registered); a mid-episode pose changes ~100% of wrist pixels vs
reset (the arm fills the frame, as expected); real video↔parquet
alignment is exact via the episodes-meta timestamps (per-file
segments contiguous, decode index = round(from_ts·30)+frame_index —
video segments carry trailing extra frames, so naive global striding
is NOT frame-exact; verified per file).

- **Pose slots (100)**: mid-band frames (`frame_index` in
  [0.3·T, 0.7·T) per episode), evenly strided across the concatenated
  mid-band pool of HELD-OUT v2 episodes 26–49 (pool 6,733; the
  encoder probe's half-A boundary at global frame 16200, manifest
  `assets/real_plates/manifest.json`). Subtask labels exist for only
  12/50 episodes — the mid-band selector is the objective one;
  per-slot jaw angle recorded as the manipulation diagnostic.
- **Manipulation reference (150)**: same mid-band selector strided
  over reference-half episodes 0–25 (pool 6,327).
- **Real frames**: decoded at exact (episode, frame) per the
  alignment rule above; the paired real wrist frame for slot i is the
  frame whose recorded state the sim renders.
- **Sim**: TWO production v3 instances, `lens_model='fitted'`
  (curve-only refit, the item's registered lens), re-tuned wrist pose
  (default), numpy post — default materials vs
  `arm_photometrics='v1'` + `mount_material='v1'` (the exact pending
  promotion stack). Established 20×5 (seed, appearance-draw)
  schedule, identical call sequence in both instances. **Pass 1**
  (in-run anchor): settled production resets, the banked protocol
  verbatim. **Pass 2**: same slots, arm qpos overwritten to slot i's
  recorded real state (clipped to ctrl range), `mj_forward`,
  production `observe()` — same grade/noise/lens path.
- **Probe**: er_60k trunk knn5 (established). Reset pass scored
  against the standard real_v2 300-strided A/B split; manipulation
  pass scored against the manipulation reference, AUROC vs the 100
  held-out real manipulation frames.

## Registered anchors / aborts (frozen before the read)

**ABORT (no claims) unless ALL of:**

- In-run reset TOP knn5 AUROC in **0.708–0.718** (banked 20×5 band;
  the fitted lens is wrist-only — top must replicate).
- In-run reset WRIST AUROC (fitted default) in **[0.49, 0.57]**
  (banked curve-only gate read 0.523).
- Real-real manipulation calibration — held-out manip pool vs manip
  reference — AUROC in **[0.35, 0.65]**: outside it, episode-level
  shift dominates and the manip pools cannot adjudicate sim numbers.
- Cross-instance qpos bit-equality ×200 slots (the materials consume
  no RNG draws; covers the benchy spawn too). Reset-pass wrist
  changed-px ≤5% tripwire stands; at manipulation poses the cap is
  REMOVED by design (the flags SHOULD move many pixels there) —
  changed-px is recorded, not gated.

Context anchors (not gates): rollout-frame banked **0.828** (old
visuals, sim policy poses); reset band 0.548–0.561 (equidistant) /
**0.523** (fitted); top 0.713.

## Decision rules (frozen)

- **PRIMARY 1 — the headline number**: manipulation-pose wrist AUROC,
  v3 fitted default vs held-out real manipulation frames.
  - **≤ 0.65** → the wrist camera is HONEST at manipulation poses
    under current visuals: the 0.828 banked anchor does not survive
    visual fixes + pose matching, and the consolidated report's open
    leg closes (with the registered caveat that the 0.828 read also
    differed in pose distribution).
  - **≥ 0.75** → the manipulation-pose wrist gap is REAL under
    current best visuals — the renderer-class arm-appearance
    decision (normal-map/PBR) is priced for the wrist too, and
    wrist-conditioned policy transfer inherits it.
  - Between → narrowed but open; reported with the content-mismatch
    riders below.
- **PRIMARY 2 — the promotion-relevant paired read**: paired
  manipulation Δknn5 CI95 (10k resamples, rng 0), stack vs default.
  - Entirely **< 0** → the material flags help exactly where the arm
    fills the frame — wrist-side value REVIVES for the two pending
    promotion asks (reset-neutrality was a visibility floor, as
    suspected).
  - **Straddles 0** → neutral even at fill-the-frame poses — the
    "reset poses couldn't see the flags" objection CLOSES and the
    absorbed-materials story from the stack read stands unqualified.
  - Entirely **> 0** → wrist regression at manipulation poses —
    flagged on both promotion asks (the texture lesson).
- **Record-only riders**: paired manip-vs-reset Δknn5 within the
  default instance (the pose effect, scene held constant); graded
  pla/servo/mount visibility px at manip poses (reset measured ~230);
  manip changed-px stats; reset-pass paired stack-vs-default (should
  replicate the 08-14 wrist-neutral read, now on the fitted lens);
  slot jaw-angle distribution.

## Limitations (registered)

Scene content is NOT matched: the sim benchy sits at its seed spawn
on the plate while real mid-grasp frames may hold the boat in the
jaw; the leader arm is fixed at its home pose; real distractor
clutter is absent from the sim wrist view (the wrist rides the raw
render — no composite). Every one of these pushes sim-real distance
UP, so a LOW manipulation AUROC is conservative evidence of honesty;
a HIGH one leaves camera-vs-content unresolved and must say so — the
graded-visibility and pose-effect riders are the partial separators.
Kinematic posing (no settle) is registered; poses are the real
servo's own recorded states, so they are physically attained
configurations.

## Cost

CPU renders (2 instances × 200 slots) + ~0.02 GPU-h embeds (~950
frames, er_60k trunk) alongside R1-B — the established headroom
pattern (stack read ran identically at 10:58Z).

---

## Amendment 1 — first run ABORTED on the calibration gate; episode-disjoint fix (posted 12:0xZ, BEFORE the re-read)

The first execution (12:00Z) ABORTED exactly as registered: the reset
anchors replicated dead-on (top **0.713** in 0.708–0.718, wrist
**0.523** in [0.49, 0.57] — the harness is sound), but the real-real
manipulation calibration read **0.129** vs the [0.35, 0.65] band. No
claims were taken from that run.

**Diagnosis** (from the run's own diagnostics): the calibration
holdout was drawn INTERLEAVED from the same episodes as the knn
reference — every fourth of 200 even picks over episodes 0–25. A
holdout frame therefore sits ~40 frames from reference frames of the
same episode: temporal near-duplicates. Its knn5 distances collapse
toward zero while the held-out episodes (26–49) have no same-episode
neighbors in the reference at all — the AUROC measured temporal
leakage, not pool comparability. A design flaw in my gate
instrument, caught by the gate's own band.

**Registered fix (frozen before the re-read)**: the calibration
holdout becomes EPISODE-DISJOINT — reference = 150 mid-band picks
from episodes **0–19**, calibration holdout = 50 mid-band picks from
episodes **20–25**, held-out slots (26–49) unchanged. Now both the
calibration pool and the held pool relate to the reference the same
way (different episodes, same rig/protocol), which is the
comparability the gate was meant to test. Every other element —
bands, both PRIMARY rules, riders, sim passes (which are unchanged
by this fix) — stands verbatim.

---

## Amendment 2 — run 2 aborted on the same gate; the band was mis-set against the protocol's own banked behavior (posted 12:1xZ, BEFORE any adjudication)

Run 2 (episode-disjoint calibration) ABORTED again: calibration
**0.268** vs [0.35, 0.65]. Reset anchors replicated exactly again
(top 0.713, wrist 0.523); the sim-side numbers were stable across
both runs (manip AUROC 0.874 → 0.877 — decode-set jitter only). No
claims taken from run 2 under the frozen rule.

**What 0.268 means.** With leakage removed, real manipulation frames
from episodes 20–25 STILL read closer to the reference (episodes
0–19) than the held episodes 26–49 do — a genuine along-the-dataset
drift. This is not an anomaly of my pools: it is this protocol
family's banked real-real norm — the clean-repo anchors read
**0.26/0.28** against the strided A/B split on the same harness (the
08-12 probe pre-reg records them as "inside the real spread"). My
symmetric [0.35, 0.65] band assumed an exchangeability the banked
protocol never had. The band was the flaw, run to run; the
instrument kept catching it.

**Directional analysis (the honest part).** The calibration gate
exists to protect the PRIMARY from a misleading held pool. The two
failure directions are NOT symmetric:

- Calibration HIGH (> 0.65: held pool unusually CLOSE to the
  reference) → held scores deflate → sim AUROC **inflates** → a
  "gap_real" verdict would be unsafe. This side must abort.
- Calibration LOW (held pool far from the reference, as measured) →
  held scores inflate → sim AUROC is **understated** → a "gap_real"
  verdict is conservative; it is the "honest" (≤ 0.65) verdict that
  would be unsafe to claim.

**Registered amendment (frozen before adjudication)**: the
calibration gate becomes directional — ABORT iff calibration
AUROC > **0.65**; below 0.35 is recorded with the mandatory caveat
that only the fake-side (≥ 0.75) verdict is claimable and any
honest-side (≤ 0.65) reading would be void. Run 2's numbers are
adjudicated under this gate (the sim side is untouched by
calibration design; a fresh run re-executes for the clean artifact).
This IS a post-hoc band correction after two looks — it is posted as
such, before any claim, with the banked 0.26/0.28 anchors as the
external justification and with the correction working AGAINST the
only verdict it permits: the measured 0.877 can only be an
understatement in this direction.

---

## RESULTS (12:2xZ 08-14, run 3 under the amended gate — the gap is REAL, and the material stack REGRESSES the wrist where the arm fills the frame)

![Wrist camera honesty by pose regime](https://mcobzarenco-fontaine-reports.static.hf.space/chart__rollout_pose_wrist.png)

All gates green under Amendment 2: reset top **0.713** (band
0.708–0.718), reset wrist **0.523** (band [0.49, 0.57]) — both
anchors replicated to the banked digit for the third consecutive
run; calibration **0.268** ≤ 0.65 with the low-note active (only the
fake-side verdict is claimable — and that is the verdict); qpos
bit-equal across instances ×200 slots; reset changed-px max 1,605
(0.5%, at the feasibility measurement).

- **PRIMARY 1 — GAP REAL.** Manipulation-pose wrist AUROC **0.877**
  (knn5 2.67e-05 vs real held 1.85e-05), far above the 0.75 bar —
  and in this calibration direction an *understatement*. The banked
  0.828 was not an old-visuals artifact: with today's best visuals
  (fitted lens + re-tuned pose, both reset-honest at 0.523) and the
  pose-distribution confound REMOVED by exact pose matching, the
  wrist camera still reads clearly fake exactly where the policy
  needs it during manipulation. The pose-effect rider isolates it:
  paired manip-vs-reset within the same instance, same scenes,
  **+8.71e-06 CI [+7.74, +9.68]e-06, 1/100 slots closer** — the pose
  change alone carries the whole gap.
- **PRIMARY 2 — REGRESSION.** Paired manipulation Δknn5, stack vs
  default: **+3.99e-07 CI95 [+2.00e-07, +6.30e-07]**, only 22/100
  slots closer — entirely above zero. The
  `arm_photometrics`+`mount_material` stack, top-validated and
  wrist-neutral at reset, makes the wrist view read MORE fake at
  manipulation poses — the texture lesson fires at the exact poses
  the promotion asks care about. Graded-surface visibility explains
  the reset blindness: pla+servo+mount ≈ **3,200 px** mean (up to
  ~34k) at manipulation poses vs ~230 px at reset — the reset read
  physically could not see this.
- **Rider replications (harness cross-checks)**: reset TOP paired
  stack **−1.49e-07 CI [−2.45, −0.57]e-07** — digit-for-digit the
  banked mount-read rider; reset WRIST paired stack **+6.7e-09 CI
  [−2.1e-08, +3.4e-08]** straddling zero — replicating the 08-14
  wrist-neutral read, now on the fitted lens.

**What this changes for the three pending promotion asks:**

1. **Clutter-patch promotion (top composite): untouched** — the
   wrist rides the raw render; this finding does not bear on it. It
   remains the payload, promote first or alone.
2. **arm_photometrics / mount promotion: now carries a measured
   wrist-side COST at manipulation poses** (+4e-07 paired, CI
   excl. 0) against its top-side gain (−1.5e-07 solo, absorbed next
   to clutter in the stack read). The 08-14 "wrist-neutral" line was
   a reset-pose visibility floor, not a clearance. If the owner
   flips the material flags, the wrist regression at manipulation
   poses is the price paid where a wrist-conditioned policy looks.
3. The 0.877 itself is the bigger fact: **the wrist camera is not
   honest mid-manipulation regardless of the flags** — the
   renderer-class arm-appearance decision (normal-map/PBR, real
   gripper geometry) now has a wrist-side number attached, not just
   the top-cam 0.552→0.328 residual.

**Registered caveat**: scene content is unmatched (sim benchy at its
spawn, no boat in the sim jaw, no real clutter in the sim wrist
view) — the 0.877 bundles arm appearance with content mismatch; the
pose-effect rider shows the bundle switches on with pose, but the
split between "rendered arm looks fake up close" and "scene contents
differ mid-grasp" is not priced here. That split is the natural
follow-up read (benchy-present vs benchy-absent at the same poses,
same harness) and is queued as the refill.

Ledger: 3 runs × ~0.02 GPU-h embeds ≈ **0.06 GPU-h**, CPU renders
alongside R1-B; two registered aborts banked as instrument findings
(temporal-leakage calibration, drift-floor band), zero sim-side
variance across runs.
