# Pre-registration: stage-2 sign-convention probe — optical-flow cross-check

*2026-08-05 ~23:3xZ. Posted BEFORE any probe code runs on the
candidate cells. Follow-up to the
[stage-1 screen](2026-08-05-sign-convention-stage1.md) (owner
hypothesis 14:55Z: flipped sign conventions, esp. wrist_roll on
mirrored wrist-cam mounts; the two-stage plan was agreed then —
this freezes stage 2's scope). CPU-only, runs beside the live box
batch and draws chain on spare cores, ionice'd. Zero GPU.*

## Question

Stage 1 found model-vs-truth mirror *signatures* but is structurally
blind in both directions: it cannot see an internally-consistent
mirror-world repo the model partially fit, and a genuine-looking
anti-correlation can still be a model failure on a hard repo rather
than a data fault. Stage 2 asks the pixels: **does the recorded joint
velocity move the world in the same direction as it does in healthy
repos?** A flipped-convention repo shows optical flow *opposite* to
what its recorded velocities predict, relative to the population.

## Candidate cells (primary reads)

The stage-1 mirror-signature shortlist, verbatim:

| cell | stage-1 evidence |
|---|---|
| kantine/domotic_dishTidyUp_anomaly · wrist_flex | median frame corr −0.75, 5/8 anti |
| kantine/domotic_groceriesSorting_expert · wrist_roll | 3/8 anti, same uploader family |
| aractingi/push_cube_square_light_reward · shoulder_lift | 3/8 anti |

**Specificity controls (pre-declared expected outcomes):**
Dongkkka/koch_arm_gripper_pick_red_pen · shoulder_pan (stage-1
"tracked-but-offset", med frame corr +0.76) must read **NORMAL**;
kevin510/lerobot-cat-toy-placement · wrist_roll (the ±180° wraparound
pathology, [wrap census](2026-08-05-wrap-census.md)) must read
**NORMAL** — wraps are not mirrors. Either control reading MIRRORED ⇒
instrument-fault presumption: no candidate verdicts ship, a debug
post does. The remaining 4 stage-1 cells run record-only.

All three candidates verified present in the local corpus copy
(`~/datasets/mcobzarenco/community_curated_v0`), robot_type so100,
30 fps, two 480×640 cams (`image`, `image2` — no wrist label, see
ego-cam rule), AV1 videos decode via torchcodec, state+action
parquet streams intact (checked this session before posting).

## Instrument (`probes/probe_sign_convention_stage2.py`, to be written)

Per repo, per episode:

1. **Joint velocity** from the *state* stream (follower = physical):
   v_d(t) = state_d(t+1) − state_d(t), deg/frame at 30 fps.
2. **Isolated-motion pairs** for target dim d: |v_d| ≥ 0.5 deg/frame
   AND |v_d| ≥ 2·|v_j| for every other non-gripper dim j. If a repo
   yields < 30 pairs, relax dominance to 1.5× once; still < 30 ⇒
   that read is **inconclusive-by-data** (reported, not forced).
   Pairs capped at 400/repo by uniform stride (deterministic, no
   RNG in selection).
3. **Flow**: decode frames t, t+1 (torchcodec), grayscale, downscale
   to 320×240, Farneback (pyr_scale .5, levels 3, winsize 21, iters
   3, poly_n 7, poly_sigma 1.5 — frozen here).
4. **Flow statistics**: ω(t) = least-squares image-plane angular
   velocity about center, Σ[(x−c)×f]/Σ|x−c|² (for wrist_roll);
   t_y(t) = mean vertical flow (wrist_flex, shoulder_lift);
   t_x(t) = mean horizontal flow (shoulder_pan).
5. **Ego-cam rule** (cams are unlabeled): over all motion pairs of
   the repo, ego cam = argmax Spearman corr(mean |flow|, Σ_d |v_d|),
   requiring margin ≥ 0.15 over the other cam. No margin ⇒ compute
   the read on both cams; verdicts only if both agree in sign, else
   inconclusive-by-camera.
6. **Signed read** per (repo, dim): Spearman ρ between v_d(t) and
   the dim's flow statistic over the isolated pairs, ego cam.
7. **Stream-consistency check** (no video; classifies mirror type):
   Spearman corr(action_d(t) − state_d(t), state_d(t+3) − state_d(t))
   over pairs with |action−state| ≥ 0.5°. Positive = servo-consistent
   (state and action share the convention — a calibration-level
   mirror); negative = action-stream-only flip (worse: contradictory
   supervision against honest pixels+state).

**Reference population** (defines the healthy sign per dim,
deterministic): all so100 repos with ≥ 8 stage-1 panel frames and
stage-1 MAE ratio ≤ 2.0 on the target dim, sorted lexicographically,
first 15 yielding ≥ 30 isolated pairs each. **Population validity
gate:** ≥ 80% sign agreement (≥ 12/15) AND median |ρ| ≥ 0.2.
Population not sign-consistent ⇒ the dim's read is
**invalid-by-population** — no mirror verdict, and the diversity
itself escalates to the owner (it would mean sign conventions vary
corpus-wide, a bigger finding than three repos).

## Decision rules (per candidate cell)

- **MIRRORED**: sign(ρ_cand) opposite to sign(median ρ_ref), |ρ_cand|
  ≥ 0.3, and an episode-level bootstrap (resample episodes with
  replacement, 1000 draws, SeedSequence(13)) puts ≥ 90% of mass on
  the flipped sign.
- **NORMAL**: sign matches the reference, |ρ_cand| ≥ 0.3, ≥ 90%
  bootstrap mass.
- **INCONCLUSIVE** otherwise (reported as such — no verdict forcing).

## Hard validation gate (before any candidate cell is opened)

Synthetic-flip oracle: pick the lexicographically-first valid
reference repo, negate its state stream on the target dim in-memory.
The probe must read **MIRRORED on the doctored copy and NORMAL on the
original** for each of the three statistics (ω, t_y, t_x). Fails ⇒
fix the instrument, no candidate reads. (Implementation + this gate +
reference population may all run before candidate cells are opened;
candidate verdicts come last, in one shot.)

## Pre-declared consequences

- **≥ 1 MIRRORED** ⇒ ideas #13 repair arm becomes eligible:
  flip-corrected derived corpus (through the #18.8 leakage-cert
  path) + paired screen — its own pre-reg, GPU only at a quiet
  boundary. Repo list flagged for any curated-v1 exclusion set and
  reported to the owner as a transferable data-quality finding.
- **All three NORMAL/refuted** ⇒ the stage-1 mirror signatures were
  model failures on hard repos, not data faults; #13 → `falsified`
  for these candidates, repair arm dies unlaunched.
- **Inconclusive-by-data/camera** on any cell ⇒ recorded honestly;
  no repair eligibility from that cell.
- Controls misbehave ⇒ instrument fault, debug post, no verdicts.

## Cost & babysitting

~18 repos × ≤ 400 pairs × 2 cams of Farneback at 320×240 ≈ 20–40 min
CPU on spare cores (nice/ionice'd — the box batch and local draws
chain own the GPUs and are untouched). Runs in a later session as a
normal work item; results post carries the verdict table, per-cell ρ
with bootstrap intervals, the population histogram, and the two
control reads.
