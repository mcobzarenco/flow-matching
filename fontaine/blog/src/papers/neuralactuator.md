# NeuralActuator: the virtual force sensor reaches our exact arm — SO-101, load registers, everything released — but it still needs telemetry our corpus never logged

*Read 2026-08-10 (lit slice `lit-radar-0821`, priority 3). Paper:
[2607.11734](https://arxiv.org/abs/2607.11734) — "NeuralActuator:
Neural Actuation Modeling for Robot Dynamics and External Force
Perception" (Zhiyang Dou, John U. Onyemelukwe, Hangxing Zhang, Heng
Zhang, Minghao Guo, Yunsheng Tian, Michal Piotr Lipiec, Joshua Jacob,
Chao Liu, Peter Yichen Chen, Yuri Ivanov, Wojciech Matusik; MIT CSAIL
+ Amazon Robotics (Ivanov, work unrelated to the position); arXiv
cs.RO, submitted 2026-07-13, v2 2026-07-17; CC BY 4.0; **RSS 2026
Outstanding Systems Paper Award**. Artifacts verified 2026-08-10:
code public under MIT at
[frank-zy-dou/Dynamics-Modeling](https://github.com/frank-zy-dou/Dynamics-Modeling)
(NeuralActuator subdir, last push 2026-07-22) with the Neural
Actuation Dataset shipping **in-repo** plus a Hugging Face mirror
([frankzydou/NAD](https://huggingface.co/datasets/frankzydou/NAD),
HTTP 200), eleven pretrained checkpoints incl. three SO-101 ones
([frankzydou/NeuralActuator](https://huggingface.co/frankzydou/NeuralActuator),
HTTP 200), and leader–follower teleop + hardware sourcing code for
both the OpenManipulator-X and the SO-101.)*

**The paper in plain words.** Cheap robot arms can't feel. They have
no force sensors, so when they bump into something — or pick
something up — the software only finds out indirectly, if at all.
This paper builds a "virtual" sense of touch from signals the servos
already produce for free: how hard the motor is working (its current
draw, or on the cheapest servos a built-in "load" register), its
temperature, its supply voltage, and how far the joint is lagging
behind where it was told to go. A small transformer reads a
nine-frame history of these signals at ~60 Hz and predicts three
things at once: a stand-in for joint torque that can drive a physics
simulator forward (so you can predict where the arm will actually
end up, friction and sag included), the outside force pushing on the
gripper (gated by a learned "is there even contact?" switch so it
reads zero in free space), and a per-motor health score that catches
a joint that has become mechanically stiff. The clever training
trick: nobody measures true torque on a $100 servo, so instead of
regressing torque labels, they push the predicted torque through a
differentiable simulator and only require that the *simulated
motion* match the recorded one — no torque sensor, no
current-to-torque calibration constant. They validate on three arms
spanning $500 (OpenManipulator-X) to $30,000+ (Franka) — and,
crucially for us, on the **SO-101 itself**, the LeRobot hobby arm
with the same Feetech STS3215 servos as ours, using only the servos'
signed load registers. Estimated forces land within ~0.1–0.3 N on
the Dynamixel arm and ~0.5–0.7 N on the SO-101; feeding the
estimated force to a behavior-cloning policy lifts pick-and-place
from 80% to 92.5%. Everything — code, data, teleop rig,
checkpoints — is genuinely released. The catch for us: every mode of
this model eats live servo telemetry. Our 229-hour community corpus
logged positions only — no load, no voltage, no temperature — so
NeuralActuator cannot run over it, at all. What it changes is the
rig: on collection day, our own arm can have this sense of touch
nearly off the shelf.

## What it contributes

- **One model, three heads, telemetry in.** Input per timestep, per
  joint: commanded (goal) position, measured position, velocity,
  effort telemetry (motor current on OMX/Franka; the **signed load
  register** on the SO-101 — "the raw current registers are not
  used"), bus voltage, coil temperature; tracking error
  `e = q_cmd − q` is an explicit feature. Nine-token history (8 past
  frames + current) into a 4-layer transformer encoder (d=192, 4
  heads, ff=384, gated attention), sub-millisecond inference
  (0.25 ms mean) at the ~60 Hz control rate. Heads: (1) **torque
  surrogate** `τ_pred` fed into a differentiable simulator (MuJoCo
  MJX; a second Newton/NVIDIA-Warp backend shipped in the release)
  to propagate dynamics; (2) **external force** as a two-stage
  product `f̂ = g · f̂_raw` with a contact-probability gate trained
  on `‖f_gt‖ > 0.01 N`; (3) **motor-condition score** per actuator.
- **Torque supervision without torque.** The surrogate "is
  supervised through differentiable simulation using pose
  trajectories, without assuming reliable current–torque
  calibration" — this deletes FACTR 2's stated K-calibration
  limitation instead of solving it, by never producing calibrated Nm
  in the first place. The cost: `τ_pred` is a surrogate that
  "absorbs external loads" absent a contact model, not a physical
  torque.
- **The Neural Actuation Dataset (NAD), released.** 450 task
  assignments (430 distinct trajectories) across 45 tasks: OMX ~90
  min / ~330k frames at ~58.8 Hz (free motion + F/T-sensor-labeled
  pushes + known-payload tasks + motor-condition tasks); **SO-101
  100 trajectories / ~66k frames / ~18 min at ~62.3 Hz** across 10
  task-payload combinations; Franka 35 lift-and-hold trajectories
  (200–600 g). Collected on a twin-arm leader–follower teleop rig
  whose code and hardware configs are also released.
- **Force labels are the cheap kind on the cheap arm.** The OMX gets
  real six-axis F/T-sensor labels for pushes; the **SO-101 rig has
  no force sensor** — its labels are gravity synthetics
  (`force_z = −mg` while a known payload is held, `−999` sentinel
  otherwise, masked from the loss). So the SO-101 head learns
  vertical payload force only, never lateral contact.

## The experiments they actually ran

- **Dynamics rollout (OMX):** propagating the simulator with
  predicted torques for 600 steps (~10 s) gives ~**3.1° average
  per-joint error** (J1–J4: 3.1/2.8/3.2/3.1°; gripper 0.2 mm).
  Parameterization ablation: direct torque prediction 0.30°/0.39°
  (avg/worst per-joint MAE) beats a residual-around-prior form at
  0.49°/0.64°.
- **Force estimation:** OMX known-payload MAE **0.11 N** average
  (0.02–0.20 N range, 600 steps), F/T-sensor contact trajectories
  ~0.23 N average, force-gauge pushing ~0.10 N; vs the best adapted
  classical baseline (GMO, a generalized-momentum observer) at
  **0.66 N** — ~5.5×. Franka payload benchmark 0.28 N average.
  **SO-101: 0.47–0.64 N (go-up-and-stay) and 0.54–0.73 N
  (pick-and-place) on 300–500 g payloads** — i.e. ~10–20% relative
  error on 2.9–4.9 N loads, from load registers alone. On no-contact
  reference trajectories the gated output stays at 0.00–0.02 N.
- **Motor condition (OMX):** rubber-band-restricted Joint 3 vs
  normal, trained on 32 trajectories: **91.0% accuracy** (precision
  84.5%, recall 96.2%, AUC 0.95) — the restricted joint draws more
  current along a near-identical trajectory, and the model reads it.
- **Behavior cloning (OMX only, not SO-101):** two tasks, 40 trials
  each, policy = joint-position history + gripper aperture, with vs
  without the frozen NeuralActuator's `f̂_ext` appended:
  pick-and-place **80% → 92.5%**, go-up-and-stay **85% → 95%**.
  That is force-*conditioning* only — there is **no FIRST-style
  re-sampling anywhere in this paper** — against a position-only
  baseline. Clean isolation of the input, narrow everything else.
- **Stated limitations:** long-horizon telemetry-conditioned
  rollouts accumulate error; online inference "requires live
  effort-related actuator telemetry" and cannot do counterfactual
  rollouts of unexecuted commands; the force head is a single 3D
  resultant at the EE (no wrench, no multi-point localization) and
  needs force labels; payloads ≤~50 g (~0.5 N) sit at the noise
  floor of the low-cost platform.

## What transfers to us — and what doesn't

- **Nothing runs on the corpus. Full stop, again.**
  community_curated_v0 logs cameras + joint positions + commanded
  positions at 30 fps. NeuralActuator's input vector needs load (or
  current), voltage, and temperature per joint at ~60 Hz — none
  logged, and even our kinematic channels are at half their rate.
  The FACTR 2 verdict repeats one rung lower: the method now exists
  at our exact servo class, and our corpus still can't feed it.
- **It does NOT validate the currentless Δq_d variant either.**
  Tracking error is an explicit input, but always *alongside* effort
  telemetry — the paper has parameterization and force-coupling
  ablations (Table XIV) but **no kinematics-only ablation**. Whether
  `(q, q̇, Δq_d)` histories alone carry a usable contact signal — the
  one question our positions-only corpus needs answered — is still
  answered by nobody, including this award winner. Idea #9's
  zero-GPU gate stands exactly as banked.
- **Rig-side, this is the biggest de-risking event yet.** Our
  FACTR 2 page called a Present_Load-based contact detector on the
  SO-101 "plausible and rig-day-sized, but unproven at this servo
  class by anyone." Now it is proven, published, and awarded:
  signed load registers at ~62 Hz on STS3215 servos support ~0.5 N-
  class force estimation, and the released stack includes the SO-101
  teleop/data-collection code, training configs (`current_source:
  load`), the 46-column CSV schema to log, and three pretrained
  SO-101 checkpoints. What the bus reports at rig time —
  Present_Position/Speed/Load/Voltage/Temperature — is exactly and
  only what their SO-101 configs consume. Caveats that survive:
  their SO-101 labels are vertical gravity synthetics (payload
  weight while held), so lateral-contact sensing at our class is
  still undemonstrated, and light contacts below ~0.5 N are
  explicitly beyond the platform's noise floor.
- **The BC gain is real but two platforms and one input away from a
  VLA claim.** +12.5/+10 points is measured on a $500 Dynamixel arm,
  2 tasks, 40 trials, a tiny position-history policy, and payload-
  dominated tasks where `f̂` is nearly a "holding mass m" indicator.
  No SO-101 BC, no image-conditioned policy, no re-weighting lever.
  It says force input helps a policy that would otherwise infer load
  from proprioception alone; it does not say what it adds on top of
  a vision trunk that can see the object.

## Hook corrections

> "neural actuation model: torque dynamics + external-force
> detection on platforms from ~$500 to $30K, teleop dataset,
> improves BC — torque-from-current at exactly our cost class; the
> FACTR 2 successor niche."

1. **"~$500 to $30K" undersold it — the paper contains the SO-101
   itself.** The ~$500 arm is the OpenManipulator-X; the third
   platform is the LeRobot SO-101 on Feetech STS3215s, i.e. *our*
   arm, below FACTR 2's $2,500 floor and below the hook's own
   stated range.
2. **"Torque-from-current" is wrong twice at our cost class.** On
   the SO-101 the current registers are *not used* — the effort
   input is the signed load register — and no head produces
   calibrated torque from current anywhere: the torque surrogate is
   trained through differentiable simulation explicitly *because*
   current-to-torque calibration is unreliable on cheap servos.
3. **"Improves BC" — verified but narrow:** OMX only (no SO-101
   BC), two payload-centric tasks, 40 trials, 80→92.5% and 85→95%,
   pure force-conditioning vs a position-only baseline. Because
   there is no re-sampling in this paper, it isolates the input in a
   way FACTR 2's entangled +17% never did — but it also leaves
   FACTR 2's re-weighting lever untested here.
4. **"FACTR 2 successor" is the wrong genealogy.** Concurrent work
   (submitted ~4 weeks after FACTR 2, does not cite it) from a
   different family: forward-dynamics-through-diffsim, not
   free-motion residual subtraction; no teleop force feedback; adds
   heads (contact gate, motor health) FACTR 2 doesn't have. It does
   occupy — and win — the below-FACTR-2 cost niche the hook pointed
   at.
5. **"Teleop dataset" — released and then some** (in-repo + HF
   mirror, MIT code / CC BY 4.0 paper, teleop + hardware configs,
   11 checkpoints), but it is an *actuation* dataset, not a skills
   corpus: ~2 h total, ~18 min of SO-101, payload/push tasks with
   gravity-synthetic or fixture-mounted force labels — telemetry-
   rich, task-poor. Not a BC data source for us.

## What it feeds

- **Idea #9 (contact-segmentation gate) — unchanged, with sharper
  framing.** The adjudication the queue asked for: NeuralActuator
  neither validates nor displaces the currentless Δq_d variant — it
  never runs without effort telemetry, and our corpus has none. The
  zero-GPU screen (per-episode `Δq_d = action − state`, free-motion
  tracking-error model, residual thresholding, gripper-close
  spot-checks) remains the only path on community_curated_v0 and
  remains unvalidated by the literature. One upgrade from this
  paper worth stealing at screen time: the **two-stage gate**
  (separate contact-probability classifier multiplying a magnitude
  regressor) is a cleaner detector shape than raw hysteresis
  thresholding, and trains on the same weak labels.
- **Idea #16 (rig benchmark, parked) — the rig-day rider goes from
  speculative to shovel-ready.** Supersedes the FACTR 2 rider: on
  collection day, log their 46-column SO-101 schema (positions,
  goal positions, velocities, signed load, voltage in decivolts,
  temperature at ~62 Hz) alongside our own format — their teleop
  code is for the same leader–follower LeRobot rig we'd use. That
  single decision buys: (a) a virtual force sensor via their
  released SO-101 checkpoints or a retrain (their configs, hours of
  data, not GPUs); (b) contact-phase labels for FIRST-style
  re-weighting of rig fine-tuning data; (c) a free motor-health
  monitor (91% accuracy on a stiff joint at the $500 class) — cheap
  insurance on hobby servos that degrade. Known limits to carry:
  vertical-payload validation only at our servo class, ~0.5 N
  noise floor, no lateral-contact ground truth without buying a
  force gauge (~$30 fixture, worth it that day).
- **No new arXiv ids for triage** — the load-bearing references are
  either pre-cutoff classics (Hwangbo actuator nets, momentum-
  observer collision detection) or already in our read set
  (LeRobot); the reference list carries no arXiv ids we lack.
