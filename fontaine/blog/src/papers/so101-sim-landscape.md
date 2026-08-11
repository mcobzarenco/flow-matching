# The SO-101 sim landscape: nobody has published our eval

**Scope:** a census (2026-08-11) of simulators and task suites usable
for SO-100/SO-101-class tabletop manipulation — every repo below was
fetch-verified live today. Anchor papers: Squint
([2602.21203](https://arxiv.org/abs/2602.21203), [own page](squint.md)) ·
REALM ([2512.19562](https://arxiv.org/abs/2512.19562)) · Benchmarking
VLAs on SO-101 ([2606.08881](https://arxiv.org/abs/2606.08881),
[own page](so101-vla-benchmark.md)) · ArmnetBench
([2607.24481](https://arxiv.org/html/2607.24481v1)) · VLA-REPLICA
([2605.20774](https://arxiv.org/abs/2605.20774)).
**Read:** 2026-08-11, sim lit lane (owner directive 17:07Z). **Fed:**
the `sim-policy-eval-100seeds` protocol (what to borrow, what to
ignore), and the publish-later option for our benchy suite.

**The landscape in plain words.** Before betting a week on our own
home-made simulation of the SO-101 arm, we checked whether someone
else already built what we need: a simulated SO-101 you can drop a
trained policy into, run a fixed set of scripted scenarios, and get
back a number that means something. The answer, after crawling every
SO-100/SO-101 sim repo and benchmark we could find: **no.** There are
about a dozen projects that put this arm in a simulator, but each is
missing something we need — most score only pass/fail instead of
measuring *how far* the task progressed, several do only one task,
one only reaches toward things, and the biggest ones require NVIDIA's
heavyweight Isaac stack. The published SO-101 *benchmarks* of 2026
all evaluate on real robots, not in sim. So our little MuJoCo scene
with a continuous distance metric is not reinventing a wheel — as far
as the public record goes, the wheel doesn't exist yet. What the
census did surface: two projects worth reading for design ideas, a
new Hugging Face channel for *distributing* sim environments, and one
important fact about our own robot model's provenance.

## The SO-101-specific field

**LeIsaac (LightwheelAI)** —
[github.com/LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac),
Apache-2.0, ~700 stars, very active (v0.4.0 Dec 2025, GR00T
inference Jan 2026). The largest SO-101 sim ecosystem: Isaac
Lab/PhysX, single- and bi-arm SO-101, teleop with a physical leader
arm, four task families (`PickOrange`, `LiftCube`, `CleanToyTable`,
`FoldCloth-BiArm`), LeRobot dataset export. Per-task `check_success`
is binary; no continuous metric. Closest to an "official" SO-101 sim
stack — and a heavyweight dependency we don't want for a 100-seed
CPU-friendly eval.

**so101-nexus** —
[github.com/johnsutor/so101-nexus](https://github.com/johnsutor/so101-nexus),
Apache-2.0, pushed *yesterday* (2026-08-10), self-declared beta. Our
closest architectural cousin: MuJoCo (plus optional MuJoCo-Warp GPU
backend), SO-101, wrist + overhead cameras, Gymnasium API, LeRobot v3
dataset output, six tasks (PickLift, PickAndPlace, StackCube, Touch,
LookAt, Move). Worth reading for task/success design; too young to
depend on.

**so-frame (LiveKit)** —
[github.com/livekit-examples/so-frame](https://github.com/livekit-examples/so-frame),
no license declared, pushed 2026-08-05. A cheap aluminum eval *frame*
for the SO-101 with a URDF/MJCF/USD triplet of the whole cell, wrist
+ overhead cameras, and — the interesting part — a REAL|SIM|OVERLAY
calibration tool that composites the sim render over the live camera.
Two sim tasks: state-based PPO in mjlab, and a ManiSkill vision task
implementing Squint's algorithm. A working template for the
sim/real visual-matching step the eval-fidelity lineage recommends.

**lerobot-sim2real (Stone Tao)** —
[github.com/StoneT2000/lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real),
no license, ManiSkill3, SO-100 only — the README now points SO-101
users at Squint as the better successor. One cube-grasp task, kept
deliberately minimal. Historically important (the first credible
zero-shot RGB sim-to-real for this arm class), superseded for our
purpose.

**NVIDIA's Sim-to-Real SO-101 Workshop** —
[github.com/isaac-sim/Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop),
Apache-2.0, frozen educational artifact (5 commits). Teleop demos →
GR00T fine-tune → *sim eval* → real eval on a vial-to-rack task. Not
infrastructure, but NVIDIA's documented endorsement of exactly our
protocol shape: sim eval as the gate before real eval.

**The rest, quickly:** `isaac_so_arm101` (BSD-3, reach-only RL,
basis of the Seeed/LycheeAI tutorials); `pick-101` (MIT, one MuJoCo
cube-lift SAC task, success = z > 8 cm); `gym-hil` (HF official,
Franka-only despite living in the LeRobot orbit — human-signaled
success, no SO-101); `slobot` (Genesis, hobby-grade bidirectional
sim↔real); assorted ≤12-star topic repos. PyBullet is a dead end for
this embodiment. None is a usable suite.

**And the 2026 SO-101 benchmarks are all real-world.** Benchmarking
VLAs on SO-101 (2606.08881) positions itself explicitly *against*
sim evals; ArmnetBench (2607.24481) is a parallel real arm-farm;
VLA-REPLICA (2605.20774) is a reproducible real-world cell. Nobody
has published an SO-101 *sim* benchmark beyond Squint's task set —
and Squint's is single-task-per-policy RL at 16×16 wrist-only
resolution, not a VLA eval.

## The general suites, as reference designs

- **ManiSkill3** (SAPIEN/PhysX, Apache-2.0 code): best embodiment
  flexibility (~35 robots, SO-100 upstream; SO-101 only as Squint's
  vendored copy), GPU-parallel, per-env seeding. The host we'd pick
  if we ever abandoned MuJoCo — we won't lightly, given our training
  contract seam is already verified against our own scene.
- **LIBERO** (MuJoCo/robosuite, MIT): Franka-fixed, but *the*
  reference design for fixed-seed suite evaluation — frozen
  init-state files per task, which is exactly the "100 fixed seeds"
  pattern our protocol wants. Now integrated into LeRobot's eval
  stack.
- **RoboCasa** (MuJoCo, MIT code / CC-BY assets): kitchen-scale,
  oversized for a 5-DoF hobby arm, but a raidable CC-BY object/scene
  library.
- **mjlab** (MuJoCo-Warp, Apache-2.0, 2.8k stars, pushed today) and
  **mujoco_playground** (MJX/Warp): the GPU-parallel upgrade path
  that keeps MuJoCo physics if we ever need thousands of parallel
  envs; so-frame already registered an SO-101 pick-place task in
  mjlab, proving the port is mechanical.
- **Genesis** (own engine, 29.7k stars): imports MJCF directly, no
  task/metric layer, no determinism guarantees documented — a
  ceiling, not a tool, for us.
- **LeRobot EnvHub** (new,
  [docs](https://huggingface.co/docs/lerobot/envhub)):
  `make_env("org/repo:envs/task.py")` loads community sim envs from
  the Hub; the LightwheelAI SO-101 tasks ship this way, with
  physical-leader teleop into sim. This is the natural channel if we
  ever publish the benchy suite.

## The asset layer — and one correction about our own model

The canonical SO-101 model chain: TheRobotStudio's
[SO-ARM100 repo](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101)
(Apache-2.0) generates `so101_new_calib.{urdf,xml}` from Onshape,
with STS3215 servo parameters adapted from the Open Duck Mini
project and an explicit ±0.5° backlash joint class; mujoco_menagerie
carries a DeepMind-tuned derivative as `robotstudio_so101` — which
is what **our** sim loads (the census agent's first pass missed it by
searching for `so_arm101`; menagerie's SO-100 lives at
`trs_so_arm100`, the SO-101 at `robotstudio_so101`). Two facts from
diffing our local copy against upstream:

- The friction/inertia identification **carried over**: our model has
  the `damping 0.60 / frictionloss 0.052 / armature 0.028` class and
  the ±0.5° backlash joints.
- The *controller* did not: our menagerie copy runs position
  actuators at `kp 998.22, kv 2.731, forcerange ±2.94`, vs
  TheRobotStudio's published `kp 17.8, kv 0, forcerange ±3.35`. A
  56× stiffness disagreement between the two published models of the
  same servo is not a detail — the ±2.94 force ceiling is exactly
  the saturation the [sim review](../posts/2026-08-11-sim-review-findings.md)
  measured in the jammed home pose, and the
  [contact-fidelity page](sim-contact-fidelity.md) has the
  SIMPLER ablation showing controller gains are the *first-order*
  term for eval fidelity. Which gains match the real arm is an
  empirical question a few real trajectories can answer.
- Known upstream caveats worth inheriting deliberately: base
  collision meshes removed ("problematic collision behavior"), and
  LeRobot's 0–100 linear gripper mapping is not reflected in the
  MJCF.

## Bottom line for the 100-seed protocol

1. **Build on our own sim.** Nothing public does closed-loop SO-101
   eval with a continuous metric; the field's SO-101 benchmarks are
   real-world. Our contract-seam-verified MuJoCo scene plus the
   distance read is ahead of the public field, not behind it.
2. **Borrow three designs:** LIBERO's frozen init-state files (the
   clean way to pin 100 seeds — materialize and commit the settled
   start states, don't just pin RNG seeds); so-frame's
   REAL|SIM|OVERLAY calibration (the cheap visual-matching step);
   so101-nexus's task/success definitions (same engine, same arm —
   free second opinions).
3. **The controller-gain question is now open** and pre-dates any
   contact fix: menagerie-vs-TheRobotStudio disagree 56× on kp for
   the same servo, and our review's home-pose saturation sits right
   on the menagerie forcerange. Sysid against a few real episodes
   belongs in the protocol's fix list.
4. **EnvHub** is where an eventual public benchy suite would live —
   parked, not planned.
