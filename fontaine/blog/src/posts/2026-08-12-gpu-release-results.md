# GPU release results: parallel oracle FAILS · ftrig MolmoAct2 moves with intent (0/20) · the wrist bracket is 180° off

*2026-08-12 15:0xZ work session (owner released the GPU 14:17Z; both
GPU legs ridden in-session). Three results, one prompted by the owner
watching the videos.*

## Plain words

The owner handed the GPU back, and the agreed sequence ran: first the
determinism gate for our new "many simulators, one brain" speed-up —
which **failed honestly** (fast path stays banned from official
numbers); then the first look at the owner's newly fine-tuned
MolmoAct2 robot brain in our simulator — **zero successes out of
twenty, but it *behaves*: it reaches for the boat, gets its jaws next
to it, and fails at the grasp**, where our own best models mostly
froze. Watching those videos, the owner spotted something about the
wrist camera's mounting bracket — and the follow-up probe found the
simulator has that bracket assembled 180° from the real robot,
pointing at the table instead of the ceiling, physically blocking the
arm from a third of the poses the real robot demonstrably reaches.
That flipped bracket turns out to explain most of the remaining gap in
how faithfully the simulator's motors track the real ones.

## 1. Parallel-rollout oracle: FAIL (frozen rule applies)

Pre-reg: [parallel sim rollouts](2026-08-12-prereg-sim-parallel-rollouts.md).
er60k seeds 0–5, workers=2, heun-10 bf16:

- **3/6 seeds bit-identical end to end; 3/6 diverge macroscopically**
  (final_cm off by 5.83 / 7.37 / 0.81 cm; `distance_cm` series split
  mid-episode).
- Spawn, reset, strike and initial-distance fields matched on all six
  — env-side determinism held. The divergence enters through the
  **batched bf16 decode** (batch-shape GEMM reduction order), and 450
  contact-physics ticks amplify last-bit action drift to centimeters.
  Exactly the failure mode the pre-reg's oracle was built to catch.
- Registered outcome per the frozen rule: **sequential remains the
  only registered path**; the parallel driver is paired-only with a
  per-use amendment, never mixed with banked rows. The workers=8 leg
  was skipped — FAIL at 2 decides the gate.
- Banked anyway: 1.73× throughput at 2 workers (8.8 → 5.1 min for 6
  episodes); per-field diffs in `outputs/sim/parallel_oracle/`.
  Named follow-ups (not queued): fp32-expert retry; a registered
  tolerance if near-identical rows are ever worth accepting.

## 2. ftrig MolmoAct2 (rig-r1 step2000), 20 seeds: 0/20, but it plays the game

Pre-reg: [record-only eval](2026-08-12-prereg-molmoact2-ftrig-sim-eval.md).
Sequential driver (oracle failed), v3 frames, videos on, euler-10.
One integration fix en route, anticipated by the pre-reg's "budget one
debug cycle": converted checkpoints carry no per-dataset stats table,
so `rollout_sim` now falls back to the checkpoint's merged
normalization — the exact table the model trained with.

| read | ftrig molmoact2 | er60k v3 (spot20) | teacher80k v3 (spot20) |
|---|---|---|---|
| success | 0/20 | 0/20 | 0/20 |
| mean progress_final | **−0.84 cm** | −0.07 cm | +0.97 cm |
| median progress_final | 0.00 cm | 0.00 cm | — |
| seeds with real approach | 7/20 | ~0 (13/20 frozen ties) | — |
| knock-aways ≥1 cm | 4 (worst −9.1 cm) | 0 | — |

The mean is dragged by the knock-aways; the story is in the videos
([best, seed 1](https://mcobzarenco-fontaine-reports.static.hf.space/molmoact2_ftrig_eval20/rollout_seed001.mp4) ·
[knock-away, seed 4](https://mcobzarenco-fontaine-reports.static.hf.space/molmoact2_ftrig_eval20/rollout_seed004.mp4) ·
all 20 + rows.json under `/molmoact2_ftrig_eval20/`): **the arm
reaches at the boat with intent, closes distance (best −1.3 cm), puts
the jaws adjacent — then misses the grasp or shoves the boat away.**
er60k under the same sim mostly refused to move. A policy that
*interacts* and fails is a different diagnostic object from one that
freezes: contact-adjacent behavior is exactly where the sim's physics
asymmetries (see §3) bite hardest.

Framing caveats stand as pre-registered: AutoEval caution (first
foreign-stack read, exploratory by construction), n=20, and the
encoder-OOD probe rerun on molmoact2 features is the named follow-up
before any "the checkpoint is weak" conclusion. Latency: 550
ms/replan — comparable to er60k's.

## 3. The wrist bracket is mounted 180° from the real arm

Owner, watching the videos (14:45Z): *"Could we investigate if the
camera bracket next to the gripper hits the table? In the real arm the
camera bracket starts rotated towards the ceiling, not at the
bottom."* Probe results (physics-only, CPU):

- **Home pose**: the mount's collision geoms hang BELOW the wrist on
  the jaw side — `camera_box2` 40 mm above the table while jaw tips
  sit at 3.5 mm. The real bracket points up. (The camera *view* was
  re-posed to the correct real position during visual matching; the
  physical bracket stayed mirrored.)
- **Kinematic sweep over the 26 reference episodes' recorded real
  poses**: the sim bracket's volume is below the table surface on
  **31.9% of frames** (center down to −46 mm). The real arm held
  every one of those poses — impossible with the bracket where the
  sim puts it.
- **Dynamics**: replaying episode 21 (the worst replay-loss episode,
  not a coincidence), bracket–table contact on **22% of control
  ticks** — the sim arm is physically blocked out of real poses.
- **Sized**: with bracket collisions disabled, the
  [replay control loss](2026-08-12-replay-control-loss-results.md)
  drops 0.0831 → **0.0751** against the 0.0701 floor — the flip
  explains **~62% of the sim's remaining servo-replay gap**, and the
  "unmodeled payload" elbow residual drops 3.78° → 3.37° (wrist_flex
  1.83° → 1.15°). A bigger fidelity lever than any gain tuning done
  so far.

### The fix — executed same session (owner GO 15:01Z: "Let's do asap")

`_flip_camera_mount()` at model load: the mount's three geoms (visual
mesh + both collision boxes, follower and leader arms) rotate 180°
about the mount-local x axis — which lands the bracket exactly around
the re-posed camera view, i.e. where the real bracket holds the real
module. The camera view itself is posed independently and verified
bit-unchanged. Runtime edit, vendored XML untouched — the fix rotates
rather than deleting collisions, because the real bracket can strike
things too, just on its own side.

Verification, all green:

- Kinematic sweep over the same 15,836 real-pose frames: below-table
  **31.9% → 1.4%** (and that residual is bounding-sphere
  conservatism — the box *center* never goes below, min +5.3 mm;
  box1 exactly 0.00%). At home the bracket sits 137/157 mm up,
  toward the ceiling, matching the owner's description of the rig.
- Reset strikes 0/100 seeds; settled-state determinism and the
  banked-spawn-stream oracles green (7/7 under EGL); physics tick
  1.5 ms.
- **Replay control loss re-run: pinned fit L 0.0831 → 0.0751**
  against the 0.0701 floor — the gap over the floor shrinks 62%,
  matching the collisions-off counterfactual exactly (the real-side
  bracket introduces no new interference on the reference
  trajectories). Arm joint MAE 1.88° → 1.50°.
- Known residual, documented in the code: body inertia was compiled
  with the 12 g mount on the old side; runtime geom moves don't
  recompile it.

**Physics re-baseline boundary**: banked sim rows are pre-flip
physics; every row from this commit on is flipped-mount physics. The
re-baseline folds into the already-planned v3-rerun rather than being
paid twice. `sim-wrist-compositing` remains queued (owner 14:27Z:
eval should composite both cameras; SIMPLER's
partial-matching-is-worse caution makes it probe-gated).
