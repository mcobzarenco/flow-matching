# Pre-registration: 100-seed sim policy eval — er_60k in the SO-101 sim

*Registered 2026-08-11 ~21:1xZ (work session), executing the owner
goal (17:07Z 08-11): "evaluate one good policy in sim on 100 fixed
seeds; primary metric = boat→disk distance reduction (continuous),
success rate secondary." Protocol frozen at this post; param sheet
in-channel with a stated objection window before the first GPU
minute. Design citations: [sim-as-eval](../papers/sim-as-eval.md)
(SIMPLER lineage); groundwork:
[sim review findings](2026-08-11-sim-review-findings.md),
[sim fixes batch 1](2026-08-11-sim-fixes-batch1.md),
[servo sysid](2026-08-11-sim-servo-sysid.md).*

## Plain words

We now have a physics sim of the owner's robot arm and its
pick-up-the-boat task, tuned this week until the start state is
clean, the physics is stable, and the simulated servos move like the
real ones (measured by replaying real episodes through the sim). This
eval drives our best policy — the `er_60k` model that all our offline
numbers say is the strongest — through 100 simulated episodes, each
from a different randomized boat position, and measures how much
closer to the goal disk the boat ends up. The point of the continuous
metric (centimeters of progress, not just success/failure) is
statistical: near-misses and partial progress separate policies far
faster than a pass/fail bit at n=100. As a built-in honesty check we
also run three earlier snapshots of the same model: our offline panel
says they rank 15k ≪ 35k < 55k ≈ 60k, and if the sim reproduces that
ordering it is measuring something real about policy quality — with
no robot time spent.

## Arms

All policy arms are `BijouPolicy` on `fontaine_molmo2_er_60k_ddp4`
checkpoints (weights-only, `fontaine-checkpoints`), fast path (no
narration), heun-10, bf16 expert, policy seed 0, batch 1:

| arm | checkpoint | role | banked panel MAE (fast path, k4l2 core) |
|---|---|---|---|
| `er60k` | `step_060000` (local disk) | **primary** — the reference trunk | 5.7782 |
| `er55k` | `step_055000` (hub dl) | validation ordering | 5.8269 |
| `er35k` | `step_035000` (hub dl) | validation ordering | 6.2892 |
| `er15k` | `step_015000` (hub dl) | validation ordering | 7.5283 |
| `hold` | none | metric floor: command the settled reset state every tick | — |

The `hold` arm is the sim analog of state-copy: under servo dynamics,
commanding the current pose holds still, so the boat should not move
— it prices reset artifacts and pins the metric zero.

## Environment (v0 physics, frozen)

- `sim.SO101Sim` at repo HEAD (this commit; stamped in every output
  JSON), scene `assets/robotstudio_so101/bijou_pickplace.xml`.
- v0 physics = widened joint limits (`_widen_joint_limits`) + scene
  solver caps 50/50 + the 340-hull CoACD benchy build +
  **SERVO_SYSID** (kp 108.18 / kv 13.377 / fr 3.478 / damping 0.722 /
  frictionloss 0.0183 / armature 0.2045) — the replay-identified set,
  held-out arm MAE 1.76° vs 3.31° vendored
  ([sysid post](2026-08-11-sim-servo-sysid.md)); SIMPLER's
  sysid-before-freeze is done.
- **Machine pin** (assets are per-machine artifacts — menagerie
  unpinned + CoACD regenerated locally): this box (`68-209-75-143`,
  H100, EGL rendering), MuJoCo 3.11.0, torch 2.11.0+cu130. Asset
  manifest: 372 files, combined sha256
  `1af281e3a9591352…` (sorted `sha256sum` of `assets/`). Cross-machine
  trajectory repro is NOT claimed.
- Observation seam as the contract check verified it: cameras
  `top`/`wrist` 640×480, state = 6 joints in degrees rig order, task
  string "Pick up the toy boat and place it on the wooden disk.",
  norm stats `mcobzarenco/so101_pick_place_v2`.

## Episode protocol

- **Seeds: 0–99 inclusive**, identical list for every arm (paired
  design). Env seed drives spawn x,y ∈ (0.195,0.27)×(−0.005,0.04),
  yaw ∈ (−π,π], benchy tint. `reset()` settles the arm first, then
  places the boat (strike-free by construction; strikes counted).
- **Horizon: 30 replans × 30 executed ticks = 900 ticks = 30.0 s** at
  30 Hz, early stop when `success()` latches. Real rig episodes
  (v2, n=50): median 19.6 s, p90 40.0 s, max 44.9 s — 30 s covers
  ~1.5× the median; truncation is priced by the min-distance column.
  Chunk size 50; executing 30 matches the real-rollout replan cadence.
- **Noise**: flow noise is per-(policy-seed, replan-index),
  batch-independent — every episode reuses the same 30-draw noise
  sequence; deterministic per config.

## Metrics

Per seed, distances in cm (XY, benchy base → disk center, from the
settled post-reset state as initial):

- **PRIMARY: progress_final = initial − final**, mean over the 100
  seeds, per arm. Paired per-seed deltas between arms with bootstrap
  CI95 (10k resamples, seed 0).
- Secondary: progress_min = initial − min (per-tick series recorded);
  **success rate** + median success tick. Caveat, pre-declared: the
  sim `success()` lacks the gripper-open check its docstring claims
  and its stillness clause reads all joint velocities — success rate
  is reported with that caveat until fixed; the distance metrics are
  unaffected.
- Recorded per seed for the report: per-tick distance series, spawn
  pose, final boat height + upright, reset strike count, per-replan
  inference latency; per-seed video (top|wrist).

## Gates and expectations

- **Validity gate**: reset strikes = 0 on every (arm, seed) — the
  sim-fixes batch measured 0/100 on these exact seeds; any strike
  excludes the seed from all arms and is reported. Expected exclusions: 0.
- **Metric-floor oracle**: `hold` arm |mean progress_final| < 0.5 cm.
  Fail = reset/settle artifact contaminates the metric → fix before
  reading policy arms.
- **Validation read (the fidelity headline)**: rank the four rungs by
  mean progress_final and compare to the banked panel ordering. The
  five pairs with panel gap ≥ 0.1 MAE — (15k,35k), (15k,55k),
  (15k,60k), (35k,55k), (35k,60k) — must all rank correctly
  (sim-better = panel-lower); the (55k,60k) pair (gap 0.0487) is
  record-only either way. Spearman ρ and the SIMPLER-style rank
  violation weight (max panel-MAE gap among misranked pairs) are
  reported. AutoEval caveat stands: fidelity is per-policy-family —
  this validates the sim for the er lineage only.
- **Interpretation caveat, pre-declared**: grasp-phase physics is
  sim-fidelity-limited (phantom collision margin p99 3.78 mm;
  gripper-priority friction override → in-grip spin) — absolute
  success rates carry that asterisk; the paired/ordering reads are
  the robust product.

## Cost and abort

- GPU: inference + EGL rendering only (the owner's inference-only
  steer for the local H100 — rendering is the sim's own workload).
  **Gate ≤ 6 GPU-h wall.** Estimate: 900 ticks × 28 ms ≈ 25 s
  sim+render per episode + 30 predicts × measured latency (smoke
  measures before launch) ≈ 35–60 s/episode → ~1–1.7 h per 100-seed
  arm, 4 policy arms + cheap hold arm ≈ 4–7 h wall, run sequentially
  in one detached unit (order: er60k, hold, er15k, er35k, er55k — the
  headline and the floor first, then max-contrast ordering).
- Abort rule: if the er60k arm exceeds 2 h wall, pause after the
  in-flight arm and reassess in-channel before continuing.
- Record-only: nothing here gates or repoints any run. What it feeds:
  if the ordering holds, the sim panel becomes a standing
  policy-quality metric (and `sim-visual-matching` inherits a
  validated baseline to improve on).

## Deliverables

Per-arm JSON (config header + per-seed rows + distance series) and
the reads analysis JSON on `fontaine-reports`; HTML report with the
house dark-mode charts (per-arm distance-over-time mean curves,
progress distributions, ordering-vs-panel scatter) + a video gallery
(best/median/worst seeds per arm); results post; numbers in-channel.

## Amendment 4 (2026-08-12 22:3xZ): camera-channel asymmetry is protocol, not accident

Owner-decided 22:31Z (after the wrist-compositing investigation,
`wrist_composite_feasibility.py`): the two observation channels are
*deliberately* produced by different visual pipelines, and every
consumer of sim eval rows should know it.

- **`top` is a composite**: real clean-plate photograph (which
  therefore carries the true rig lens for every background pixel)
  with the sim-rendered arm + objects inpaint-composited over it
  (v2/v3 render styles).
- **`wrist` is fully rendered**: scene-matched render through the
  center-matched equidistant fisheye + fixed grade. It sits inside
  the real spread on the encoder probe (5-NN AUROC 0.548 after the
  08-12 re-pose), but every pixel carries the *synthetic* lens model.

A wrist composite was investigated and rejected: episode-start plate
poses spread 20.8 mm / 5.1° median (static plates mush — the 0.951
read), and although plane-homography warping from the 26-plate bank
is geometrically sound (the wrist is table-plane-dominated, median
100% of rays), nearest-plate warp fill is p10 49% before
arm-footprint and parked-boat holes — residual sim-texture seams are
SIMPLER Table III's partial-matching hazard. The residual synthetic-
lens risk on the wrist channel (pixel-scale-as-distance-ruler,
2603.02139) is instead addressed render-side: `sim-fit-real-lens-model`
(plumb-line θ→r fit on the pinned real frames + cubemap two-stage
render), probe-gated, queued 08-12.
