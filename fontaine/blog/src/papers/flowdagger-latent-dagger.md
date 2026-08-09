# FlowDAgger: fixing a frozen policy in noise space, five corrections at a time

*Read 2026-08-09 (standing lit slice, during the K-smoke ladder
attempt-2 wait window — the session the stage-2 attachment steer
window opens). Paper: FlowDAgger,
[2607.08877](https://arxiv.org/abs/2607.08877), July 2026.*

**The paper in plain words.** When a robot policy makes a mistake, a
person can grab the arm and show it the right move. The obvious way
to use those corrections is to retrain the policy's weights — but
that is slow, needs big GPUs, and (this paper measures it) makes the
policy *forget* the other things it already knew. FlowDAgger never
touches the weights. Its trick: a "denoising" policy turns random
noise into actions, so any corrected action can be run *backwards*
into the exact noise that would have produced it. Collect a handful
of those (correction → noise) pairs and train a tiny helper network
that, at run time, hands the frozen policy *better noise*. The big
model stays intact; the helper is small enough to train on the same
consumer GPU that runs the robot. With 5–20 corrections per task it
turns failing skills into working ones — and the skills it wasn't
correcting stay at full strength, where weight retraining collapsed
them.

## What it contributes

- **Action inversion.** An expert correction `a` is mapped to the
  latent noise `z` that the frozen flow policy would have denoised
  into `a` — reverse-time integration through the policy's own
  velocity field, then local refinement. The workhorse detail is
  *per-step fixed-point inversion* (M=5 iterations per step): action
  reconstruction MSE 0.00168 vs 0.0329 for the naive single-step
  Euler reverse — an order of magnitude, and their ablations say the
  cheaper variants underperform end-to-end, so the inversion quality
  is load-bearing.
- **A latent noise policy.** A small MLP + encoder maps observations
  to noise adjustments; the frozen base then denoises the adjusted
  noise as usual. All adaptation capacity lives *before* the frozen
  model, in its input-noise space — the same latent surface our
  draws instrumentation samples (each draw = one noise seed; their
  helper *chooses* the noise instead of sampling it).
- **The DAgger loop, made cheap.** Rollout, human intervenes on
  failure, invert the intervention, add the pair, retrain the helper
  (~8 GB GPU total — the memory already needed for deployment).

## The experiments

- **MetaWorld, π0.5 base, 12 contact-rich tasks, 50 intervention
  rollouts:** mean success 0.53 → 0.78 (+0.25). Baselines with the
  same interventions: SFT +0.18, LoRA-DAgger +0.15, Residual-DAgger
  +0.11, DSRL (latent-space RL) +0.02.
- **Real hardware, two bimanual rigs (FR3 Duo, Dual UR5e), 8 tasks,
  5–20 intervention episodes each:** Toolbox Packing 13% → 80% with
  10 corrections; Glassware Stacking 26% → 76% with 5.
- **The retention headline.** After adapting on Hammer (50
  episodes), held-out already-working tasks stay at 0.88 mean for
  FlowDAgger while the target task gains +0.44. The weight-space
  baselines *destroy* the held-out set: LoRA-DAgger −0.66, SFT
  −0.94. This is the paper's sharpest measurement and its whole
  argument in one number.
- **Generality:** comparable gains (+0.21 mean) adapting
  Cosmos-Policy, a world-action model with latent video diffusion —
  the recipe is not π-specific; appendix runs cover Gr00t N1.7 and a
  vanilla diffusion policy.

## What transfers to us, and what doesn't

- **The steer-window angle (#4).** Today we choose how to attach the
  stage-2 expert: frozen trunk (F) vs KI-joint (K). FlowDAgger is
  *not* evidence about which seam trains a better expert — their
  base policies are complete post-attach VLAs, and nothing here
  touches trunk-vs-expert gradient routing. What it IS: the
  strongest measurement yet of the **aftermarket value of frozen
  capital** — SFT's −0.94 retention collapse is the same disease
  KI insulates against at training time, measured at adaptation
  time. If F ties K on the screen, this line of work (with
  [[qguided-flow-critic]], same conclusion from the
  inference-guidance side) says frozen-capital recipes keep
  composing after deployment: correction, guidance, and steering
  methods all assume an intact base.
- **The rig path (#16).** 5–20 physical interventions per task and
  an 8 GB training budget is *exactly* the data and compute scale of
  the owner rig. We already bank this family: **UniSteer**
  (2605.10821, rig lever #3 on the
  [noise-steering II page](noise-space-steering-2.md)) inverts
  corrections through the frozen flow decoder the same way (per-step
  fixed point; UniSteer M=16, FlowDAgger M=5) and trains a noise
  actor on them. FlowDAgger's deltas over the banked lever: (a) the
  **retention measurement** — UniSteer never quantified what
  weight-space adaptation costs, FlowDAgger's 0.88-vs-−0.94 is that
  number; (b) the explicit DAgger loop at 5–20 interventions; (c)
  generality beyond π-family (Cosmos world-action model, Gr00t,
  vanilla diffusion). Same hardware rung on the rig-time menu
  (teleop corrections needed); the retention evidence upgrades the
  whole rung's case, not just this paper's.
- **The noise-space thread (#19/#1).** A *learned chooser of input
  noise* on a frozen sampler — read now from three sides:
  golden-ticket screens (fixed noise reuse), Q-guided critics
  (gradient steering mid-sample), and inversion-trained noise
  policies (UniSteer, now FlowDAgger). Our own reads said draw
  diversity is real but selection is the bottleneck;
  inversion-from-corrections is a label source none of our banked
  screens tried — remembered *if* the selection thread re-opens (it
  is currently parked on the rung-(b′) NO-SCORER verdict).
- **What doesn't transfer.** MetaWorld and bimanual tabletop rigs,
  not our data; interventions need a human in the loop (nothing for
  panel-eval land); and the inversion needs the flow ODE run
  backwards per correction — cheap at their chunk sizes, unmeasured
  at ours.

## Fed

- **#16 (rig-transfer):** the named few-intervention adaptation
  recipe + its evidence bar (retention 0.88 vs SFT −0.94).
- **#4 (stage-2 attachment):** steer-window context note — the
  frozen-capital aftermarket argument, weighed only if the screen
  reads F≈K.
- Cross-links: [[qguided-flow-critic]] (frozen-policy steering from
  the inference side), the golden-ticket pages (noise-space value,
  selection bottleneck).
