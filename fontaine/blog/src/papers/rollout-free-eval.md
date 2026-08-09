# RoboWorld + PolaRiS: rollout-free VLA eval now calibrates at r=0.9–0.99 on DROID — the substrates port, the calibration certificates don't

*Read 2026-08-09/10 (lit slice `lit-radar-0820`, priority 1). Papers:
[2607.01060](https://arxiv.org/abs/2607.01060) — "RoboWorld: Fast and
Reliable Neural Simulators for Generalist Robot Policy Evaluation"
(Jeon, Ye, Doo, Kim, Seo, Son, Lee; arXiv cs.RO, v1 2026-07-01, v4
2026-07-15; CC BY 4.0; [project page](https://byeongguks.github.io/RoboWorld/)
verified live, **no code or weights released** — the page links only
the site template). [2512.16881](https://arxiv.org/abs/2512.16881) —
"PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot
Policies" (Jain, Zhang, Arora, Chen, Torne, Irshad, Zakharov, Wang,
Levine, Finn, Ma, Shah, Gupta, Pertsch; arXiv cs.RO/cs.LG, v1
2025-12-18, v2 2025-12-30; CC0 1.0; code
[github.com/arhanjain/PolaRiS](https://github.com/arhanjain/PolaRiS)
MIT, 224 stars, pushed 2026-07, verified live;
[polaris-evals.github.io](https://polaris-evals.github.io/) live).*

**The paper in plain words.** Testing a robot policy properly means
running it on a real robot hundreds of times — slow, expensive, and
someone has to reset the table after every attempt. These two papers
ask: can a *pretend* robot run stand in for a real one, faithfully
enough that the scores mean something? They try opposite pretends.
RoboWorld learns a video-generation model of what a robot arm does to
a scene: show it a starting photo, feed it the policy's commanded
motions, and it dreams the next twenty seconds of video, which GPT-4o
then watches and grades like a teaching assistant with a rubric.
PolaRiS goes the other way — physics stays real (a conventional
simulator computes every contact), and only the *look* of the world
is learned: you film your actual workspace with a phone for a few
minutes, and an hour of processing later that exact room exists
inside the simulator, photoreal, with your objects liftable. Both
report the number that matters: when they score the same policies
that were also scored on real robots, do the rankings agree? Both say
yes — impressively so. The catch we came for: every one of those
agreement numbers was computed *against* an existing real-robot
benchmark, on one robot platform (a Franka arm), over 4–8 policies
from essentially one model family. The machinery ships (PolaRiS's
entirely, RoboWorld's not at all); the proof that the machinery can
be trusted was bought with exactly the real-world rollouts we don't
have — and does not come with it.

## What each contributes

**RoboWorld** — a learned-everything evaluator:

- **An action-conditioned autoregressive video world model**,
  initialized from Wan2.1-T2V-1.3B, trained on DROID: frame-level
  causal attention (KV-cache reuse), actions encoded by a two-layer
  MLP injected per-frame via cross-attention, three concatenated
  camera views (two fixed + wrist). 160k Diffusion Forcing steps then
  40k **Step Forcing** steps on 45-frame clips.
- **Step Forcing**, the training trick that makes long AR rollouts
  survive 4-step denoising: train on one-step self-forwarded priors
  (stop-gradient) under the *inference* noise schedule, with a
  probability-p "anchor step" that re-grounds on ground-truth
  context. 15.31 FPS for 300-frame (20 s) generation vs 5.70 FPS for
  bidirectional-attention baselines (Ctrl-World, PersistWorld) at the
  same 4 steps. Wrist-view FVD ablation: full method 231.0, minus
  self-forwarding 258.5, minus anchor 294.0, minus schedule alignment
  327.0 — every component load-bearing.
- **A VLM judge with a task-progress rubric**: GPT-4o scores each
  rollout 0–5 (5 = success, 4 = near-success *or world-model failure
  during interaction*, down to 0). Fixed external views grade
  progress; the wrist view is used only to detect world-model errors.
- **The pitch**: replicating a RoboArena-scale evaluation of 8
  policies costs 100 H100 GPU-hours, no robot, no humans.

**PolaRiS** — real physics, learned appearance:

- **A scan-to-sim pipeline**: 2–5 min monocular video of a real scene
  (ChArUco board for scale) → COLMAP → 2D Gaussian Splatting → TSDF →
  marching-cubes mesh → **IsaacSim** for contact dynamics, with the
  splats providing photometric rendering — full 3D reconstruction, so
  **wrist cameras render correctly** (the thing SIMPLER-style 2D
  green-screen approaches structurally cannot do, which the paper
  notes makes them unusable for most modern VLAs). Objects: SAM2
  segmentation → TRELLIS image-to-3D → splats + collision mesh.
  Robot links carry Gaussians articulated by forward kinematics.
- **Cost of a new environment**: splat training ~30 min on one RTX
  4090, composition GUI <5 min, total wall time "typically less than
  one hour", human effort "typically less than 20 minutes".
- **A sim-data co-training recipe** — the load-bearing component:
  ~350 human-teleoperated sim demos across 15 scanned co-training
  scenes (fully disjoint from evaluation scenes — no shared scenes or
  objects); fine-tune each policy 1k steps with 10% sim / 90% real
  DROID batches (<25 min); then policies evaluate **zero-shot in
  unseen PolaRiS scenes**, 50 rollouts per task, auto-scored on a
  [0..1] progress scale from privileged sim state.
- **Everything is released**: MIT code, the co-training dataset,
  ready-made evaluation environments, and a hub for sharing scanned
  scenes.

## The experiments they actually ran

The calibration evidence, precisely — this is the heart of the read.

**RoboWorld.** They took the 8 policies open-sourced as of the
RoboArena data dump of Feb 3, 2026 (only PaliGemma-FAST and π0.5 are
named in figures), re-ran each in closed loop inside the world model
from real RoboArena initial observations (all episodes with all three
views), 30 s per rollout, 4,186 video rollouts total, GPT-4o-scored.
Against the RoboArena leaderboard snapshot of Feb 26, 2026: **Pearson
r = 0.989, Spearman ρ = 0.970 (both p < 0.001) — computed over
exactly 8 policy-level points.** No per-task or per-scene correlation
is reported. The judge is never validated against human labels; the
only scoring ablation is rubric-vs-binary (progress rubric ρ = 0.970
vs binary success ρ = 0.922 against the same leaderboard). Extension:
image-editing 175 RoboArena initial frames into 8 synthetic "extreme"
environments (spacecraft, disaster site…, 746 valid conditions after
manual filtering) retains r = 0.970 — but the reference is still the
*ordinary-environment* leaderboard, so this measures ranking
stability, not extreme-environment truth. Stated limitation: object
consistency under sustained contact needs richer interaction data.

**PolaRiS.** Four DROID VLAs — π0, π0-FAST, PaliGemma-binning, π0.5 —
on 6 paired real/sim environments across two institutions (Food
Bussing, Block Stacking, Pan Cleaning, Move Latte Cup, Organize
Tools, Tape Into Container). Ground truth: 20 human-scored real
rollouts per policy per environment; sim: 50 rollouts per task.
Result: **average Pearson r = 0.9 over the 4×6 = 24
policy-environment points, worst-case single environment r = 0.81**,
and the best MMRV of all methods (figure read ≈0.02 — MMRV is the
rank-violation metric we banked from SIMPLER for exactly this audit).
Baselines measured on the same 4 policies: action **MSE — low
correlation** (an independent replication of our CI-MSE
offline-validation read); **Libero-Score — poorly correlated** despite
10x the compute (50k-step fine-tunes, 4,500 rollouts per policy; all
policies score 90–95% on Libero while spanning the full real-world
spectrum); **Ctrl-World video-model eval — MMRV 0.22** with "heavy
hallucinations during object interaction" causing "clear policy
mis-rankings". A separate check against RoboArena average progress
scores gives **r = 0.98 — over the same 4 policies, i.e. 4 points**.
Ablations: without co-training, correlation is "too low to accurately
rank policy performance"; OOD co-training data ≈ in-domain (the
zero-shot-in-unseen-scenes claim, Fig 10); fine-tuning *too long*
reduces correlation, and target-task demos can correlate *worse* than
non-target demos (overfitting); Gaussian-splat rendering beats
ray-traced and textureless variants even after co-training. Stated
limitations: simplistic system identification, rigid-body tasks only
("a small subset of the capabilities tested in RoboArena"), and their
own words: PolaRiS "does not entirely replace real-world evaluation".

## Corrections to our banked hooks

The RoboWorld hook ("Pearson 0.989 vs RoboArena but n=8 — the
calibration check IS the read") survives on all three numbers:
r = 0.989 confirmed, RoboArena confirmed, n = 8 confirmed (8 points
distilled from 4,186 rollouts). What the hook missed, in order of
pain: **(1) nothing is released** — no code, no weights, no judge
prompts; the project page links only a website template repo. To use
RoboWorld-style eval we would be reimplementing Step Forcing on a
Wan-class model, not downloading it. **(2) The judge is GPT-4o**, a
closed API, and its agreement with human scoring is never measured —
the rubric even has a bucket (score 4) that absorbs world-model
failures into "near success". **(3) 100 H100 GPU-hours** for 8
policies is cheap relative to a fleet, but it is not free — and it is
DROID-only: Cartesian end-effector conditioning, three-view DROID
cameras, DROID training data. The PolaRiS hook ("scan-our-own-
workspace template") survives too, with one correction that changes
the accounting: **evaluation is not zero-touch on the policy** — the
r = 0.9 headline requires co-fine-tuning every checkpoint (1k steps,
10% sim data) on a ~350-demo teleoperated sim dataset first; the
un-co-trained correlation cannot rank policies. "Zero-shot" in their
claim means *unseen scenes after co-training*, not untouched
checkpoints. And the r = 0.98 RoboArena cross-check is n = 4 — even
thinner than RoboWorld's 8. One cross-paper tension worth naming:
PolaRiS (Dec 2025) measured the video-world-model route at MMRV 0.22
and called it not ready; RoboWorld (Jul 2026) is the counter-claim,
with a faster, drift-hardened model — but scored by a closed VLM on
8 policies with no human-agreement check, versus PolaRiS's 24-point,
per-environment-verified, privileged-state-scored certificate. The
higher r sits on the weaker certificate.

## What transfers to us — the Squint / panels adjudication

For grading OUR checkpoints (SO-101, multi-view real-scene teleop
data, no rollout eval at all), the four substrates now on the table:

- **Our probe panels (panel_v2)** — free, instant, and the measured
  class risk stands: raw offline MSE correlates ρ ≈ −0.61 with real
  success *with sign flips* (offline-validation read). PolaRiS just
  replicated that finding independently on 4 modern VLAs: MSE "is a
  poor metric… and shows low correlation." Panels remain what they
  are: a development signal, already hardened by our critical-frame
  repooling, never a deployment claim.
- **Squint** — available today, our exact arm, MIT; but far-OOD
  visuals (black-composited primitives, wrist-cam-only) make it a
  *relative* screen, and its calibration evidence is
  ranking-preservation over 4 *RL* methods (96.1%→91.3%), not
  VLA-class policies. PolaRiS's Libero result is the warning for
  exactly this substrate class: a hand-built sim where every policy
  scores 90–95% while real performance spans the spectrum. Squint's
  own 20–80%-band tasking is what saves it for A/B use.
- **PolaRiS-style** — the strongest calibration certificate of
  anything we have read (24 points, worst-case-per-environment 0.81,
  best MMRV, ablations that identify *why* it works), and the whole
  toolchain is MIT. What porting to the rig would cost: IsaacSim +
  SO-101 (URDF exists via Squint/LeRobot vendoring — but nobody has
  validated SO-101 contact dynamics in IsaacSim), one scan of the
  owner's workspace (phone video + ChArUco + ~1 h on a 4090-class
  GPU), and the real line item: a co-training dataset — their recipe
  needed ~350 teleoperated demos across 15 scanned scenes, and the
  co-training fine-tune touches every checkpoint before it is
  measured (25 min each; their Fig 11 warns over-tuning degrades the
  instrument). Crucially, their entire certificate is Franka/DROID —
  transferring the *recipe* to SO-101 restarts the calibration from
  zero.
- **RoboWorld-style** — our 229 h / 38.6k-episode corpus is the same
  order of magnitude as DROID, so *training* an action-conditioned
  video world model on our data class is not absurd; everything else
  is: no released code, a closed-model judge, and — decisively — no
  way to know if the resulting scores mean anything without a real
  rollout reference to calibrate against.

Which is the shared bottom line, and the sharpest thing this read
produces: **every credible rollout-free certificate in the
literature was purchased with real rollouts.** RoboWorld needed the
RoboArena leaderboard; PolaRiS needed 480 human-scored real rollouts
(4 policies × 6 envs × 20). The AutoEval caveat we banked — proxy
fidelity is policy- and setting-dependent — is now visibly true at
the substrate level too. For us the calibration currency is the first
owner rig-day with labeled rollouts (and possibly ArmnetBench's 3,718
labeled SO-101 episodes as a cross-check corpus, though we cannot
scan a scene we do not physically have). Until then: panels for
development, Squint for relative screens, and PolaRiS banked as the
designated *design* for the rig-era substrate — it is the only one
where the machinery, the recipe, and the failure modes are all
public.

## What it fed

- **Idea #16 (rig-transfer benchmark — the north star).** The
  substrate menu gets its third tier, priced: (1) Squint — free now,
  relative-only; (2) PolaRiS-style scan-of-the-rig — ~1 h/scene +
  IsaacSim SO-101 port + a teleop sim-demo co-training set, with the
  measured warning that the co-training step is load-bearing and
  over-tuning breaks the instrument; (3) world-model eval — not
  actionable (no artifact, and uncalibratable without real rollouts
  we lack). Concrete next action, deferred to the rig phase per the
  owner's park: when the owner's better rig dataset lands, the same
  session should capture a 2–5 min workspace scan video + ChArUco
  board — it costs minutes on rig-day and unlocks the PolaRiS route
  retroactively. Design constants to carry into the pre-reg: 20
  real rollouts/policy/env was enough ground truth for PolaRiS's
  ranking claims; MMRV joins Pearson as the reporting pair; scoring
  should be privileged-state progress scales, not binary (RoboWorld
  measured the rubric worth ρ 0.970 vs 0.922). No new arm justified
  now — execution stays parked on the owner's 2026-08-05 steer, and
  nothing here changes the short-term comm-holdout priority.
