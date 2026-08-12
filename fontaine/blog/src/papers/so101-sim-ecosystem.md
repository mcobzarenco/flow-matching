# The SO-100/SO-101 sim ecosystem, take 2: the training-in-sim angle

*Lit slice `0820`, 2026-08-12 (owner-called 09:23Z: "what are the
typical environments people use for benchmarking that we could also
benchmark on... ideally close to our embodiment"). This page is the
UPDATE to the [08-11 census](so101-sim-landscape.md) ("nobody has
published our eval" — still true): where that census asked *can we
drop our policy into someone's eval suite* (answer: no), this one
asks the questions the owner's GRPO thread makes live — who TRAINS
in sim on this arm class, at what throughput, with what published
transfer, and where a benchmark bridge is cheap. Theme cluster:
[lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real)
· [ManiSkill3](https://arxiv.org/abs/2410.00425) · the LeRobot sim
env family (gym-hil, gym-so100/lowcostrobot) · protocol references
(LIBERO, SimplerEnv). Survey depth: READMEs/docs; deep reads queued
where flagged.*

## Plain words

Our simulator was built for one job: measuring whether OUR policies
work before risking the one physical robot on the desk. But other
people also simulate this exact cheap 3D-printed arm, and if we can
run their tasks — or they ours — our numbers stop living on an
island: a "0/500 successes" headline means more when someone else's
training recipe scores 90% on the same arm in the same kind of
scene. This page maps who simulates the SO-100/SO-101 family, in
what engine, and what it would take for our results to sit next to
theirs.

## The map

| project | engine / renderer | embodiment | what it offers | distance from us |
|---|---|---|---|---|
| **lerobot-sim2real** (StoneT2000) | ManiSkill3 (SAPIEN, GPU-parallel) | **SO-100** | RL in sim → zero-shot real RGB cube-grasp, **91.6%** real success avg of 3 runs; sysid tools; robot-color DR | closest published sim2real result on our arm class |
| **ManiSkill3** | SAPIEN, GPU-parallel sim+render | many incl. SO-100 tasks | thousands of env steps/s WITH rendering; the engine under lerobot-sim2real and Squint | engine alternative to our MuJoCo path; not a drop-in |
| **gym-hil** (LeRobot) | MuJoCo | SO-101-adjacent (Franka default; HIL focus) | human-in-the-loop RL envs; **records LeRobotDataset format** | same physics engine + data format as our stack |
| **gym-so100 / gym-lowcostrobot** | MuJoCo | SO-100/low-cost arms | community task envs (cube lift/push) | same engine, simpler scenes than ours |
| **Isaac Lab SO-101** (community + NVIDIA guides) | Isaac Sim/PhysX | **SO-101** | sim2real guides (e.g. vial-to-rack), teleop demo collection | different engine; useful as a protocol reference |
| **LIBERO / SimplerEnv** | MuJoCo / SAPIEN | Franka / Google-robot, WidowX | the benchmarks VLA-RL papers actually report on (e.g. SimpleVLA-RL hits 97.6 on LIBERO-Long) | wrong embodiment — protocol value only |

## What transfers to us

1. **The evaluation-protocol gap is ours, not the ecosystem's.** No
   standard benchmark exists for SO-101 tabletop pick-place with
   *imitation-trained* policies — lerobot-sim2real benchmarks RL
   *training* in sim, LIBERO benchmarks VLAs on Franka. Our
   100-seed paired-arm protocol (spawn distributions, paired seeds,
   engagement/direction reads, strike gates) is already more
   instrumented than what these repos ship. The bridge worth
   building is the cheap one: **port their task definition (cube
   grasp, their success predicate) into our sim** and report our
   arms on it — one page of XML + a success function — rather than
   porting our stack into ManiSkill.
2. **lerobot-sim2real's result calibrates expectations.** 91.6%
   real cube-grasp from pure-sim RL on an SO-100 says the
   embodiment is NOT the blocker — with enough sim interaction, this
   arm class does transfer. Their recipe is RL-from-scratch with
   task rewards, not VLA imitation; the delta to our 0/500 is
   recipe+task difficulty, not hardware. (Deep read queued: their
   sysid tooling and camera alignment vs our servo-sysid/plate
   approach.)
3. **ManiSkill3's GPU-parallel rendering is the scaling answer** if
   sim RL (the GRPO thread) becomes real for us: thousands of
   env-steps/s with cameras vs our ~10 ticks/s/env after the GPU
   compositor. But its visuals are stylized — our real-plate
   compositing is the *fidelity* play, theirs is the *throughput*
   play. A GRPO experiment could use v0-style rendering (no
   composite) for training throughput and our v3 for eval fidelity.
4. **gym-hil's LeRobotDataset recording** means sim demos and HIL
   corrections land in exactly our training format — relevant the
   moment we train (not just eval) in sim.

## What doesn't transfer

- Engine migration: SAPIEN/Isaac ports of our sysid'd servo model +
  plate compositor would restart the visual-matching work from
  zero; nothing in the ecosystem beats our measured 0.673/0.548
  encoder reads for OUR cameras.
- LIBERO/SimplerEnv numbers: wrong embodiment; useful only as
  protocol patterns (SimplerEnv's real-vs-sim correlation
  methodology is the one to imitate when we claim "sim predicts
  rig").

## Fed into

- `ideas.md` hook `0820`; queue: `lit-so101-benchmark-envs`
  (this page's deep-read follow-ups: lerobot-sim2real sysid +
  camera alignment; SimplerEnv correlation methodology),
  `grpo-on-sim-design-research` (ManiSkill3 throughput datum;
  SimpleVLA-RL/πRL pointers banked there).
- Concrete next arm this suggests: **a cube-grasp task port into
  our sim** with their success predicate — a second task axis for
  the panel at ~1 day of work, making our numbers comparable to the
  only published SO-100 sim2real result.

Sources:
[lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real) ·
[ManiSkill3 paper](https://arxiv.org/abs/2410.00425) ·
[ManiSkill sim2real docs](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/sim2real/index.html) ·
[gym-hil / SO-101 HIL guide](https://note.com/npaka/n/n6efe8890aa84) ·
[Isaac Lab SO-101 guide](https://vnrobo.com/en/blog/so101-isaac-lab-lerobot-sim2real) ·
[SimpleVLA-RL](https://arxiv.org/abs/2509.09674) ·
[πRL](https://arxiv.org/abs/2510.25889)
