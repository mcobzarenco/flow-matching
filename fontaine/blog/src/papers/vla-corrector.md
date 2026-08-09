# VLA-Corrector: a 40M drift monitor that decides when the chunk is going wrong

*Read 2026-08-09 (lit slice `lit-radar-0812b`, priority 1: verifier
family / async adjacency, #6/#19/#22). Paper:
[2607.01804](https://arxiv.org/abs/2607.01804) — "VLA-Corrector:
Lightweight Detect-and-Correct Inference for Adaptive Action
Horizon" (cs.RO).*

**The paper in plain words.** Robot policies like ours emit actions
in chunks: predict a second or two of motion, execute it blind, then
look again. If something drifts mid-chunk — the object slips, a
human nudges the scene — the robot keeps executing a stale plan.
This paper bolts a small external "monitor" onto a frozen policy:
a 40M-parameter MLP that watches the camera features and continuously
asks *"is the scene changing the way the executed actions said it
should?"* When the answer is persistently no, it cuts the chunk
short and replans — and during that one recovery replan, it nudges
the policy's denoising toward the direction that undoes the
accumulated drift. No retraining of the policy, no environment
labels — the monitor learns from the same demonstrations the policy
did. The result is an *event-triggered* action horizon: long chunks
while things are fine, short corrective ones when they aren't.

## What it contributes

- **Latent-space Vision Monitor (LVM)**: a 40M MLP over frozen VLA
  encoder features that predicts the *residual* visual change
  ΔZ expected from the executed actions (residual, not future state
  — static scene content cancels out). Deviation score
  E_t = 1 − cos(ΔZ_expected, ΔZ_observed); the trigger is adaptive
  (median + MAD sliding window, hysteresis via λ_on > λ_off) and
  fires only after p=5 consecutive exceedances.
- **Online Gradient Guidance (OGG)**: on the single recovery replan
  after an interrupt, inject the gradient of a cosine loss between
  predicted action effect and the corrective direction
  ΔZ_corr = ΔZ_exp − ΔZ_dev into the flow head's *velocity field*
  (v ← v − η∇L). Amortized cost +7.93 ms per environment step,
  because recovery replans are rare.
- **Numbers**: MetaWorld on π0.5 48.7% → 64.35% (+15.65 pp) at
  *fewer* policy calls (4.98 vs 5.15); transfers as a wrapper to
  SmolVLA (+4.75) and X-VLA (+4.05). LIBERO few-shot 94.0 → 97.8,
  above the full fine-tune (96.95). Real-robot disturbance recovery
  +28.3 pp.
- **The ablation that matters**: truncation-only (detect + cut, no
  gradient steering) already gets 60.35% of the 64.35% — most of the
  win is *when to stop*, not *how to steer*. And a decoupled
  external monitor beats an internal auxiliary head grafted onto the
  VLA by +14.8 pp (49.55% vs 64.35%) — the policy's own features
  make a poor judge of the policy.

## What transfers to us

1. **A verifier that escapes the closed-family verdict.** Our #6
   scorer rung closed the zero-training scorer family: SC and
   masked-contrast both *anti-select* among candidates. VLA-Corrector
   is a different animal on both axes that mattered: it is *trained*
   (40M, from demos alone — no env or success labels, the same data
   diet RoVer proved viable), and it judges *temporal drift of the
   executing plan*, not candidate quality at decode time. The
   post-mortem calibration bar (beat |ρ|≈0.03 by an order of
   magnitude) applies to candidate rankers; a drift monitor is
   scored on interrupt precision instead. If a learned-verifier arm
   ever gets its affirmative case, the two design data to steal are:
   **predict residuals, not states** (static content cancels) and
   **keep the judge decoupled from the policy** (+14.8 pp for
   external).
2. **The adaptive-horizon result reframes a #22 question.** Our
   async/staleness ladder treats execution horizon as a fixed design
   constant per arm. Here the horizon is an *output* of a cheap
   monitor, and truncation alone — just cutting stale chunks early —
   is worth +11.65 pp before any steering. That is adjacent to the
   SV-VLA gate (runtime-plan-verification page) but cheaper: no VLM
   judge, ~ms-scale MLP. The banked boundary-incompatibility read
   gives us the complementary datum (our seam jump is 11–14×
   per-step motion): they cut chunks on *scene* drift; our measured
   pathology is *decode* drift at the seam. Both argue against
   fixed-horizon blind execution.
3. **OGG is the gradient-guidance family again** (third sighting
   after Q-guided flow critic and the RLDT SVGD update): gradients
   injected into the velocity field at decode, policy frozen. The
   recipe is becoming standard; ours would differ only in the guide
   signal.

## What doesn't transfer

- **Needs closed-loop execution.** Everything here is measured in
  rollouts with disturbances; our panel is open-loop frozen frames —
  there is nothing for a drift monitor to catch. This is a #16
  rig-time capability, parked exactly like the rest of the ladder.
- **Cross-domain correctors are weak** (+3.1 pp LIBERO-trained on
  MetaWorld vs +10.0 domain-matched) — the monitor is a per-domain
  artifact, one more thing to train at deployment, not a portable
  module.
- **OGG cannot invent recovery behaviors** the frozen policy lacks
  (their stated limitation) — it re-aims the prior, it doesn't
  extend it. Consistent with our Δ_self finding: the bottleneck is
  what the policy can generate, not how you steer among its outputs.

## Which idea/arm it fed

**#6 (aux attribution / scorer escalations)**: two design constraints
banked for any future learned-verifier arm (residual target;
decoupled external judge, +14.8 pp). **#22 (async staleness)**:
event-triggered truncation logged as a menu adjacency — the
truncation-only ablation (+11.65 of +15.65 pp) is the citable datum
that *when to cut* dominates *how to steer*; no arm (closed-loop,
parked on #16). **#19**: verifier-family sighting only. Cross-refs:
[runtime plan verification](runtime-plan-verification.md) (SV-VLA's
heavier gate), [RoVer](rover-learned-verifier.md) (same
demos-only-training diet),
[Q-guided flow critic](qguided-flow-critic.md) (same velocity-field
guidance mechanism), [label-free selection
signals](label-free-selection-signals.md) (the closed candidate-scorer
family this monitor is *not* a member of).
