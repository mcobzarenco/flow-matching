# FoMo-FD: running the flow backward to ask "could my actions have caused what I just saw?"

*Read 2026-08-09 (lit slice `lit-radar-0816`, priority 3: the #6
no-rollouts detector slot). Paper:
[2607.27511](https://arxiv.org/abs/2607.27511) — "Failure Detection
for Surgical Robot Imitation Policies via Flow-Matching World
Modeling" (Huang, Cai, Patel, Hajiha, Browne, Chen; submitted to
RA-L, 2026-07-29).*

**The paper in plain words.** A robot running an imitation-learned
policy fails silently: nothing crashes, the arm keeps moving, but
the task has quietly gone wrong. This paper builds a watchdog for a
surgical robot that never needs to see a failure during training.
It learns a compact "world model" — a predictor of how the camera
view should evolve given the actions the robot executed — from the
same successful demonstrations the policy itself was trained on.
At runtime, instead of predicting forward and comparing, it runs
the learned dynamics *backward* from what the camera actually saw:
if the observed outcome traces back to something wildly improbable,
the executed actions can't explain the scene, and it raises an
alarm. With one wrist camera it catches 96.6% of failures with a
1.3% false-alarm rate across four surgical tasks.

## What it contributes

- **A latent, action-conditioned flow-matching world model**:
  frozen DINOv2 features compressed by a β-VAE into a 16×16×8
  latent; conditional flow matching predicts the latent at the *end
  of a short window* (K=4 steps, ~0.4 s at 10 Hz) from 4 frames of
  history plus the intervening executed actions, fed through both
  global modulation and action-token cross-attention. Trained
  **only on the successful expert teleop demos** — no failure data
  anywhere.
- **The inverse-transport score is the trick.** Rather than sample
  a forward prediction and diff it, set the observed endpoint
  latent as the flow's target and integrate the learned ODE
  *backward* to the Gaussian base; the squared norm of the
  recovered base point is the nonconformity score. Their own
  forward-prediction-error variant of the *same* world model scores
  52.2% detection vs 96.6% — the backward direction, not the world
  model, carries the result.
- **Conformal thresholding on successes only**: episode-level peak
  score over N=19 *successful policy rollouts* per task sets the
  alarm threshold at α=0.05, bounding false alarms on nominal
  episodes. It guarantees quiet operation on successes; detection
  power is empirical.

## The experiments it ran

dVRK, four tasks — two real (tissue retraction, shunt insertion),
two Isaac-Sim (needle pickup, ring-over-post) — monitoring an ACT
policy, against 20 staged failure modes in four categories (grasp,
spatial configuration, scene disturbance, sensing/actuation; 320
failed + 80 successful eval rollouts). Headline, wrist camera:
**96.6% failure-detection rate at 1.3% false alarms**, vs logpZO
45.3%, RND 42.8%, and the forward-error ablation 52.2%. The fixed
workspace camera collapses to 45.9% — the wrist view is
load-bearing. Action conditioning matters (none 87.2% → both
pathways 96.6%), as does the horizon (K=1 42.2% → K=4 96.6%).
Scoring runs at 14 Hz against a 10 Hz loop; evaluation is offline
on recorded rollouts, with no time-to-alarm metric.

## What transfers to us

- **The closest fit yet to the #6 constraint — with the boundary
  now measured precisely.** Two corrections to our banked hook,
  logged loudly: (1) "FDR" is failure **detection** rate
  (episode-level TPR), not false discovery rate; (2) "no env
  rollouts" is **false as stated** — the world model trains
  rollout-free, but the conformal threshold needs ~19 successful
  rollouts *of the deployed policy* per task (the paper's own
  limitation #1). The honest comparison against Foresight
  (2606.23085, which trains on failure rollouts): FoMo-FD shrinks
  the rollout requirement from "collect failures" to "19 successes
  on deployment day" — much cheaper, not zero.
- **The offline-executable slice is real**: train the world model on
  community_curated_v0 (success-only, action-conditioned — exactly
  its diet), and validate the *score's* discrimination without any
  threshold by ranking true action windows against perturbed-action
  counterfactuals on held-out demos. Calibration then costs ~19
  successful episodes on the owner rig — a deployment-day line
  item, not a research blocker.
- Their encoder-alignment move (policy and detector share DINOv2)
  maps cleanly for us: build the world model on the same frozen
  trunk features the action expert consumes.

## What doesn't transfer

- **The wrist camera carries the result** (96.6% vs 45.9% fixed
  view). A tabletop SO-101 with only exo cameras may land near the
  weak regime; this is a rig-configuration prerequisite, not a
  modeling detail.
- **For #17's world-model-as-verifier thread the datum is
  negative-leaning**: the score is strictly post-hoc — it needs the
  *observed* endpoint latent, so it cannot rank candidate action
  chunks before execution. Forward sampling could in principle
  score candidates, but their own WM-PE ablation shows forward
  samples from this model are a much weaker signal.
- Surgical close-up manipulation, 0.8 s windows, per-task per-view
  thresholds, staged failures, not yet peer-reviewed.

## Which idea/arm it fed

#6 (`aux-subgoals` / failure-detection slot) — the new best-fit
detector recipe under our constraints: world model offline now,
conformal calibration deferred to rig day at ~19 successes/task;
the offline perturbed-action validation probe banked as the
zero-rollout first step. #17 — the latent-WM-as-verifier pole gets
a caution: inverse transport (post-hoc) beats forward prediction
by 44 points on the same model. No gate changes.
