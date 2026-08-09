# FACTR 2: torque sensing from 10 minutes of free motion — no force sensor, but the current sensor it does need is one we don't log

*Read 2026-08-09/10 (lit slice `lit-radar-0820`, priority 2). Paper:
[2606.12406](https://arxiv.org/abs/2606.12406) — "FACTR 2: Learning
External Force Sensing for Commodity Robot Arms Improves Policy
Learning" (Steven Oh, Jason Jingzhou Liu, Tony Tao, Philip Han,
Kenneth Shaw, Satoshi Funabashi, Ruslan Salakhutdinov, Deepak Pathak;
CMU; arXiv cs.RO, submitted 2026-06-10; arXiv nonexclusive
distribution license; project page
[jasonjzliu.com/factr2](https://jasonjzliu.com/factr2) verified live,
code **"Coming Soon" — not released** as of 2026-08-09; predecessor
[FACTR](https://jasonjzliu.com/factr/) verified live with released
training/teleop/hardware repos).*

**The paper in plain words.** Feeling what you touch is most of what
makes contact-rich manipulation work, and most robot arms can't do
it — real force sensors are expensive, so cheap arms ship without
them. This paper's trick: every motor already leaks a force signal
through its electrical current, but that raw signal is polluted by
everything else the motor is doing — fighting gravity, fighting its
own friction, accelerating the arm. So they drive the arm around in
free space for ten minutes, touching nothing, and train a small
recurrent network (one minute of training) to predict what the motor
torque *should* be when no contact is happening. At run time,
whatever the measured torque shows *beyond* that prediction must be
the outside world pushing back. On a $30,000 Franka this residual
tracks the factory torque sensors closely enough to drive
force-feedback teleoperation that 20 study participants rated as easy
as the real-sensor version. Then they close the loop into learning:
label each demonstration frame as free-space, about-to-touch, or
in-contact using the estimated torque, and feed the
behavior-cloning sampler five times more of the about-to-touch
frames. That re-weighting — plus giving the policy the torque signal
as an input — beats prior force-aware policies by over 17% task
progress on five long-horizon assembly tasks. The catch for us: "no
force sensor" does not mean "no sensor." The method's raw material is
motor current at 100 Hz, and the cheapest arm it ever touches is a
$2,500 AgileX Piper — our SO-101's hobby servos and our
positions-only community corpus are both below its floor. What
survives the descent is the *idea*: contact phases are learnable from
motor-side residuals, and up-weighting the moments just before
contact is where the policy gains live.

## What it contributes

- **NEXT (Neural External Torque Estimation):** train
  `f_θ(x) → τ̂_free` on ~10 minutes of contact-free motion, where
  `x` is a 50-step history of joint position, joint velocity, and
  the commanded-minus-measured position difference `Δq_d`; at run
  time `τ̂_ext = τ_m − τ̂_free`, with measured motor torque obtained
  as current times torque constant, `τ_m = K·I_m`. Architecture:
  2-layer LSTM (hidden 128) + 2-layer MLP head (hidden 256), L2
  regression, AdamW; **1 minute of training on an RTX 3090**; runs
  at 100 Hz in deployment (568 Hz capable). The pitch is that the
  learned model absorbs what analytical models can't: "nonlinear
  friction, stiction, backlash, hysteresis, temperature-dependent
  drive behavior, sensing noise, torque ripple, deadzones, and
  saturation."
- **Force-feedback teleop without force sensors:** NEXT's estimate
  drives FACTR-style bilateral teleoperation on arms that lack
  dedicated sensing (Franka with sensors ignored; AgileX Piper
  which never had them).
- **FIRST (Force-Informed Re-Sampling Training):** segment each
  demo into free-space / pre-contact / contact by hysteresis
  thresholding the L1 norm of `τ̂_ext` (pre-contact = the 1-second
  window before contact onset), then up-sample the contact-relevant
  phases during BC batch construction — sampling weights
  `w_F : w_PC : w_C` = 1:5:1 on most tasks, 1:3:3 for cap screwing.
  The policy itself is a flow-matching head (conditional velocity
  field) over DINOv3-Base image tokens plus MLP-projected
  proprioception **and the estimated external torque** —
  `o_t = (images, q_t, τ̂_ext,t)`.

## The experiments it actually ran

- **Torque-estimation accuracy (Franka, ground truth = the
  factory-calibrated external-torque estimate from built-in joint
  torque sensors; 5 min held-out data with a human applying
  forces):** NEXT contact error **0.547±0.348 Nm** vs FILIC
  4.395±1.531 and a disturbance observer 1.471±0.761; free-space
  error 0.414±0.278 Nm, *below* the dedicated external sensor
  estimate's 0.449±0.208. On the Piper, free-space error
  0.018±0.012 Nm. Input ablation: `(q)` < `(q, q̇)` <
  `(q, q̇, Δq_d)` — the position-tracking-error feature
  "consistently outperforms," i.e. it is the load-bearing input.
  History length swept 10/25/50, 50 selected.
- **Teleoperation user study:** a wiping task on the Franka, 20
  participants, five conditions (no feedback, disturbance observer,
  leader-follower position feedback, FACTR teleop with dedicated
  sensors, FACTR teleop with NEXT). NEXT-based feedback rated
  easier to use than the baselines on 1–5 ratings, applied joint
  torques comparable to the sensor-based condition; repeated on the
  Piper.
- **Policy learning:** five long-horizon contact-rich tasks on a
  bimanual Piper rig — LEGO assembly, NIST belt assembly, NIST
  insertion, tool clean-up, cap screwing — 250 demos per task, 20
  rollouts per task, metric = **task progress** (fraction of
  completed stages per rollout; e.g. LEGO has 6 stages). Baselines:
  base policy, base + torque input, FACTR, TA-VLA. FIRST is highest
  on all five tasks; the abstract's claim is "outperforms prior
  force-aware policies by over 17% in task progress." Exact
  per-arm averages live only in Figure 6 (my figure read: base
  ~0.55, base+torque ~0.60, FACTR ~0.63, TA-VLA ~0.65, FIRST
  ~0.81 — treat as approximate).
- **The phase ablation (Table 2, the number that matters for
  curation):** up-sampling **pre-contact** frames averages **0.818**
  task progress vs **0.670** for contact-only up-sampling and 0.811
  for both — the second before touch is where the useful gradient
  lives, not the contact plateau itself.
- **Stated limitations:** absolute torque scale depends on the
  motor torque constant K ("if K is inaccurate, absolute scale
  requires calibration"), and NEXT is robot-specific — retrain per
  arm.

## Corrections to our banked hook

1. **"No force sensor" has a hidden clause: a current sensor is
   load-bearing.** `τ_m = K·I_m` is an *input*, at both training
   and inference, at 100 Hz. "Commodity" here means a $2,500 AgileX
   Piper and a Franka — the paper never touches the hobby-servo
   class, and SO-100/SO-101 appear nowhere in it.
2. **The "+17%" is not re-sampling alone.** FIRST both feeds
   `τ̂_ext` to the policy as an observation *and* re-weights the
   sampler; the +17% is over prior force-aware baselines (FACTR,
   TA-VLA) that also consume force. The paper never ablates
   re-sampling *without* the torque input — so the corpus-curation
   transfer our hook proposed (sampling-only, no force at
   inference) is unvalidated even inside the paper. The clean
   re-sampling datum is Table 2's 0.818 (pre-contact) vs 0.670
   (contact-only), both arms torque-conditioned.
3. **"~10 min of motion data" verified, but it is 10 minutes of
   *instrumented* motion** — free-space trajectories with motor
   current logged, plus 1 minute of training. Motion alone (our
   corpus's positions-only logs) is not enough.
4. **No artifact yet:** project-page code is "Coming Soon"
   (checked 2026-08-09); only FACTR 1's repos are released. The
   hook's "cheapest force recovery" framing priced in an
   implementation that does not currently exist in public.

## What transfers to us and what doesn't

- **Not NEXT itself, on the corpus.** community_curated_v0 logs
  cameras + `observation.state` (joint positions) + `action`
  (commanded positions) at 30 fps. No motor currents, no torques —
  the residual `τ_m − τ̂_free` is uncomputable, full stop. Joint
  velocities aren't logged either, though finite-differencing at
  30 fps is serviceable.
- **But the load-bearing feature IS in every episode.** The
  paper's own input ablation crowns `Δq_d` — commanded minus
  measured position — and that is exactly `action −
  observation.state`, present in all ~52.5k corpus episodes. A
  NEXT-shaped variant needs no current: train the same tiny LSTM to
  predict *free-motion tracking error* from `(q, q̇)` history, and
  use the residual of actual vs predicted tracking error as a
  contact score. On position-controlled servos with high gear
  friction, raw `|Δq_d|` spikes on fast motion and gravity load
  too — which is precisely why the residual-vs-learned-free-model
  structure, the paper's actual idea, is the part worth copying.
- **Rig-side (SO-101 servos):** Feetech STS3215s report
  Present_Load — a PWM-duty proxy, not calibrated current — at bus
  rates well below the paper's 100 Hz, with no trustworthy K and
  gearbox friction/backlash far beyond a Piper joint. Absolute Nm
  is out of reach (the paper's own K-calibration limitation, squared);
  a *relative* contact detector from the same recipe (10 min free
  motion, predict the load signal, threshold the residual) is
  plausible and rig-day-sized, but unproven at this servo class by
  anyone, including this paper.
- **The curation idea transfers as a hypothesis, not a result.**
  Phase-aware up-weighting with pre-contact >> contact is a
  measured sign at 250-demo single-task scale, force-conditioned,
  on one rig — three qualifiers away from "re-weight 229 h of
  community BC data by an estimated contact signal." It earns a
  screen, not a corpus.

## What it fed

- **Idea #9 (data levers) — a new candidate for the
  weighted-sampling slot, with a zero-GPU gate first.** #9's
  "judge-score-weighted sampling (never yet run)" now has a
  literature-backed sibling: phase-weighted sampling by estimated
  contact proximity, with FIRST's 1:5:1 pre-contact weighting and
  the 0.818-vs-0.670 pre-contact>contact ordering as the design
  prior. **Cheapest next action (VISTA-screen pattern, zero GPU):**
  an offline segmentation screen over the corpus — per-episode,
  compute `Δq_d = action − state`, fit the free-motion
  tracking-error model (or start even cheaper: per-joint normalized
  `|Δq_d|` hysteresis), and validate the derived contact onsets
  against gripper-close commands as weak grasp labels on a
  spot-check sample. Falsification is cheap and loud: if phase
  shares are degenerate (contact ≈ everywhere or ≈ nowhere) or
  onsets don't line up with grasp events, the lever dies before any
  training. Only a clean screen buys the paired 40k re-sampling
  arm — and that arm must carry the caveat that sampling-only
  (no force input) is unvalidated even in FACTR 2.
- **Idea #16 (rig benchmark, parked) — a rig-day note, not an
  action.** When the owner's better rig dataset gets collected, the
  10-minute free-motion protocol is worth folding into the
  collection day: log Present_Load + positions during free motion,
  train the 1-minute LSTM, and check whether the residual spikes on
  contact. If it does, the rig gets a free contact channel for
  FIRST-style weighting of its own fine-tuning data — and a force
  signal our panels have never had. No arm justified now: execution
  on #16 is owner-parked, and the servo-class feasibility question
  is exactly what the 10-minute protocol answers for free later.
