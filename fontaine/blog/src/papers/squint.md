# Squint: a real, MIT-licensed SO-101 sim — built for 16x16 wrist-cam RL, not for grading VLAs

*Read 2026-08-09 (lit slice `lit-radar-0819`, priority 1). Paper:
[2602.21203](https://arxiv.org/abs/2602.21203) — "Squint: Fast Visual
Reinforcement Learning for Sim-to-Real Robotics" (Abdulaziz Almuzairee,
Henrik I. Christensen; UC San Diego; arXiv cs.RO/cs.CV/cs.LG, submitted
2026-02-24; CC BY 4.0; code
[github.com/aalmuzairee/squint](https://github.com/aalmuzairee/squint),
MIT, verified live).*

**The paper in plain words.** Teaching a robot arm by trial and error
directly from camera images is normally slow — hours or days of
compute. Squint makes it take minutes: it runs 1,024 simulated copies
of the SO-101 hobby arm at once inside the ManiSkill3 simulator,
shrinks the camera image down to a squint-worthy 16x16 pixels (small
enough that training is cheap, still enough to see a red cube on a
black table), and stacks up known RL speed tricks — a distributional
critic, layer normalization, a tuned update ratio, compiled PyTorch.
A policy for one task trains from scratch in 2–9 minutes on a single
RTX 3090, then drives the real arm with no real-world training at
all: 73 of 80 trials succeed across eight tasks. The catch for anyone
hoping this is a general benchmark: each policy does exactly one task,
sees only a wrist camera, and the world is deliberately impoverished —
the background is composited to solid black in sim and the real table
is black to match, objects are color-coded printed primitives. It is
a beautifully engineered narrow corridor between sim and real, not a
recreation of reality. For us the paper is almost beside the point:
the released task set is the first credible closed-loop, success-
scored simulation of our exact arm — and it is actually downloadable.

## What it contributes

- **Squint, a visual SAC recipe** for wall-clock-fast training:
  1,024 parallel ManiSkill3 GPU envs, batch 512, UTD ~0.25 (256
  updates per iteration), C51 distributional critic, LayerNorm on
  all linear layers, two-layer CNN encoder updated only by the
  critic's TD loss, torch.compile + cudagraphs + bf16 AMP ("more
  than a 5x speedup" from the systems work alone). 1.5M env steps in
  ~15 minutes on one RTX 3090; most tasks converge in 2–6 minutes
  (Stack Can, the hardest, 9).
- **"Resolution squinting":** render at 128x128, area-downsample to
  16x16, rather than rendering 16x16 natively — identical sample
  efficiency, better final performance ("natural anti-aliasing"),
  and images small enough to keep a 1M-transition replay buffer on
  GPU (a 100K buffer costs 7% asymptotic success).
- **The SO-101 Task Set:** eight ManiSkill3 tasks registered as
  `SO101{Reach,Lift,Place,Stack}{Cube,Can}-v1`, 50 steps max at 10Hz
  control (5-second episodes), dense rewards plus binary success
  predicates built from position thresholds and contact-force grasp
  checks. Ships with the SO-101 URDF + meshes, a digital twin of
  their rig, per-task domain-randomization configs, a real-robot
  deploy script (LeRobot calibration), a sim/real camera-alignment
  tool, and STLs for printing the task objects. MIT, 88 stars.
- **Zero-shot sim-to-real on a real SO-101:** 73/80 (91.3%) vs 96.1%
  average success in sim, from a single wrist camera at 16x16 plus
  noisy joint positions. Deployment scales all actions by 0.15 and
  triples control to 30Hz — a deliberate sim/real mismatch for
  safety and smoothness that transfer survives.

## Corrections to our banked hook

Our hook clause "SO-101 arm integrated into ManiSkill3" is **wrong in
the way that matters**: upstream ManiSkill3 contains the SO-100 only.
SO-101 support originated in a community PR (credited to @jackvial)
against `StoneT2000/lerobot-sim2real`, and Squint **vendors** the
URDF, meshes, and agent class inside its own repo
(`envs/robot/so101.{urdf,py}`). Nothing was upstreamed; you get the
arm by installing their repo, and `-v1` task IDs live only there.
Second correction: "zero-shot sim-to-real" is **single-task visual RL
from dense rewards** — not BC, not a VLA, no language, one policy per
task. Third: the "heavy domain randomization" is heavy in *jitter*,
narrow in *scope* — millimetre/degree wrist-cam pose + FOV noise
applied every step, ambient light 0.2–0.5, color jitter, object
size/friction, gripper stiffness/damping (500–2000 / 50–200), joint
noise sigma=5 degrees — but backgrounds are composited to solid black
via segmentation greenscreen, object colors are fixed (red cube, blue
can), and there is no texture or scene randomization at all. The rest
of the hook survives: 8 tasks, released, <15 min on one 3090 all
check out — the first banked hook in nine deep reads where the
artifact story got *stronger* on contact.

## The experiments it ran

- **Sim (Table I, 8-task average):** Squint 96.1% vs optimized SAC
  88.3%, PPO 60.2%, DrQ-v2 4.5% (single-env, sequential — the
  wall-clock strawman), BC 41.9%. Per task Squint is 95–100%
  everywhere except Stack Can (81.2%, where SAC collapses to 18.7%).
- **Real (Table II, 10 trials x 8 tasks):** Squint 73/80 (91.3%),
  SAC 65/80, PPO 50/80, DrQ-v2 8/80, BC 38/80 (47.5%),
  State-to-Visual DAgger 53/80 (66.3%). Sim ranking is preserved on
  the real arm; the gap concentrates in stacking (Stack Cube
  95.0%→8/10, Stack Can 81.2%→6/10 — can grasping suffers "tipping
  and insufficient gripper friction").
- **Color jitter is load-bearing:** removing it drops real success
  from 73/80 to 58/80 (−18 points) with sim success unchanged — the
  agents remain "brittle to visual changes" by the authors' own
  admission.
- **Imitation struggles here:** BC and DAgger distill a state-based
  SAC expert; the paper attributes their weakness to distribution
  mismatch — "wrist cameras require active vision, and a different
  exploration movement than an all-seeing state agent would take."
  A wrist-cam-only substrate is genuinely hard for policies that did
  not learn to point the camera.

## Could WE run rollouts in it? (the question we banked it for)

**PARTIAL — yes mechanically, not yet meaningfully as an absolute
benchmark.** The load-bearing facts, from the code, not the paper:

- The envs are standard registered ManiSkill3 gym environments.
  Obs modes include `rgb+state`; sensor resolution is a constructor
  kwarg (their own train script passes
  `sensor_configs=dict(width=..., height=...)`), so rendering 224+
  for a VLM trunk is a flag, not a fork. Success/`info` comes back
  every step. An evaluation-only harness for an external policy is
  a ~100-line gym loop with zero dependency on their training code.
- Action interface matches us: 6-dim joint space (5 arm + gripper),
  and the SO101 agent exposes `pd_joint_pos` with
  `normalize_action=False` — absolute joint positions, the LeRobot
  convention our data and policies already use — alongside the delta
  controllers they trained with.
- Compute is a non-issue: they run 1,024 envs on a 3090; at our
  scale the cost of eval rollouts is Molmo2-4B inference, not sim.
- **The gap is visual, and it is large.** Default observations are
  one wrist camera (71-degree FOV) over a black-composited scene
  with color-coded primitives. Our policies consume multi-view RGB
  of real cluttered rigs from 229h of community teleop, plus
  language. A zero-shot drop-in would measure our policy far out of
  distribution — a failure would not separate "policy is bad" from
  "renderer is alien." Their own in-domain BC baseline at 41.9%
  sim success is the warning label.
- The repo softens this more than the paper does: a
  `ThirdCameraEnv` (128x128, per-step pose randomization) ships
  behind a one-line switch, the black overlay is a config flag
  (`apply_overlay=False` gives raw renders; the overlay image is
  swappable per the README), and DR configs are dataclasses. A
  both-cameras, high-res, no-overlay variant is a small subclass.
- Tasks are 5-second single-primitive episodes — reach/lift/place/
  stack — not our task distribution, and there is no language. So
  the honest near-term use is **relative** measurement: A/B deltas
  between our checkpoints/variants under a constant (wrong) domain
  gap, plus unlimited ground-truth-labeled success/failure rollouts.
  Absolute "would this work on the rig" numbers need a sim-
  adaptation step (co-training or fine-tuning on sim-rendered
  frames) that changes what is being measured — that is idea #16's
  design problem now, not a blocker to touching the substrate.

## Which idea it feeds

- **Idea #16 (rig-transfer benchmark — the north star).** The
  blocking dependency "no rollout substrate for our arm class" is
  gone in the mechanical sense: MIT-licensed, verified-installable
  SO-101 digital twin with success predicates, arbitrary-resolution
  RGB, and a LeRobot-convention absolute-joint controller. The
  design question #16 inherits: bridge the visual gap (third-person
  camera + no overlay + small sim co-train arm) and define what a
  sim success rate is allowed to claim about the rig. The paper's
  own measured transfer (96.1% sim → 91.3% real, ranking preserved
  across four methods) is the first quantitative sim-real
  correlation on this exact arm — weak evidence, right arm.
- **Idea #6 (failure-detector calibration).** The substrate
  generates unlimited success/failure rollouts with free
  ground-truth labels (contact-force grasp checks, position
  predicates, per-step `info`). Calibrate residual-stream probes on
  sim rollouts, then cross-check the calibration against
  ArmnetBench's 2,288 real labeled SO-101 failures — a two-sided
  test neither corpus supports alone.
- **Idea #22 (async/staleness chunk-switch screens).** Closed-loop
  by construction, deterministic seeds, 1,024 parallel envs, 10Hz
  control. Staleness/chunk-switch ablations become measurable as
  success-rate deltas with the domain gap held constant across
  arms — exactly the relative-measurement regime where an
  out-of-distribution sim is still a fair judge.
