# Action space design, finally measured: chunk-wise delta beats our absolute-joint folklore by 8 points in our exact policy class

*Read 2026-08-09 (lit slice `lit-radar-0819`). Paper:
[2602.23408](https://arxiv.org/abs/2602.23408) — "Demystifying Action
Space Design for Robotic Manipulation Policies" (Yuchun Feng, Jinliang
Zheng, Zhihao Wang, Dongxiu Liu, Jianxiong Li, Jiangmiao Pang, Tai
Wang, Xianyuan Zhan; Tsinghua AIR / Shanghai AI Lab / PKU; arXiv
cs.RO, v1 2026-02-26, v2 2026-04-23; ICML 2026 poster; CC BY 4.0).*

**The paper in plain words.** Every robot-learning team has to decide
what the policy's outputs actually mean: target positions or nudges
from where the arm is now ("absolute vs delta"), joint angles or
gripper pose in 3D space ("joint vs task/EE space"), and how many
future steps to predict at once ("chunking"). Almost everyone —
including us — inherits these choices from whatever codebase they
started from. This group spent 16,000+ A100 GPU-hours and 13,000+
real-robot rollouts testing the combinations properly, on real 6-DoF
arms with a grid protocol that controls where objects start. The
verdict: predicting *deltas* — but only deltas measured from the
arm's pose at the start of each chunk, never step-to-step increments
— beats absolute targets everywhere, by around 8–15 points. Joint
space beats gripper-pose space when you train one robot on plenty of
data (especially with generative policy heads like flow matching);
gripper-pose space wins when transferring across robots or from a
pretrained foundation model. And the best open-loop execution window
depends on which you picked: delta wants short, absolute wants long.
We predict absolute joint positions. This paper says that is the
wrong default for our exact policy class — and they released code
and data, so the claim is auditable.

## Hook corrections — the banked clause vs the paper (9th deep read, 9th correction set)

- **"13,000+ real rollouts on a bimanual robot" — misleading
  platform claim (their own abstract's fault).** The main platform is
  a **single-arm** AgileX PiPER (6-DoF); the dual-arm PiPER runs only
  1 of 4 real tasks (Bimanual Cube Transfer), plus an AIRBOT arm for
  one cross-embodiment task. Every reported real number is 3 trials
  × 10 rollouts = 30 rollouts, so 13,000+ rollouts ≈ 430+ evaluated
  configurations. 2,000+ demos collected, 250/task standard.
- **"500+ trained models" — true but they are small models.** One
  in-house base architecture (FiLM-conditioned ResNet-18 + 6-layer
  Transformer decoder) in two loss variants they *label* "ACT" (MSE
  regression) and "DP" — and their "DP" is **flow matching, not
  DDPM Diffusion Policy**. Plus π0-LoRA transfer runs (30k steps,
  batch 32). No large VLA is trained from scratch anywhere.
- **"Delta consistently wins" — true on averages, with a decisive
  asterisk: only CHUNK-WISE delta.** Step-wise delta (increments
  relative to the previous predicted step) loses ~10 points to
  chunk-wise and would erase the headline. "Consistently" holds for
  scenario averages, not every cell (multi-task Cube, Lift Pot,
  Shake Bottle flip to absolute).
- **"Joint/task complementary" — real, but the numbers for the
  transfer half live only in Figure 6**, which the HTML does not
  render numerically. The in-domain half is fully tabulated; the
  cross-embodiment/π0 half we can verify in direction only.
- **"Absolute needs longer horizons" — it's the EXECUTION horizon,
  not the training chunk.** All policies are trained with k=60
  chunks (2 s at 30 Hz, following π0 practice); they grid-search the
  *inference* execution window 15–60 and find delta peaks near 30,
  absolute near 60. **Training chunk length is never swept** — the
  question we most wanted answered is out of scope.
- **Release audit: PASSED — a first for this hook series.** Project
  page ([cathyf9600.github.io/empirical](https://cathyf9600.github.io/empirical/)),
  code ([github.com/CathyF9600/DemystifyActionSpace](https://github.com/CathyF9600/DemystifyActionSpace)),
  and dataset ([hf.co/datasets/cfeng9600/DemystifyActionSpace](https://huggingface.co/datasets/cfeng9600/DemystifyActionSpace))
  all resolve (curl 200, checked 2026-08-09). Repo is thin (3 stars,
  README doesn't document which action-space variants ship) but it
  exists, with the AgileX teleop data.

## What it contributes

- **A clean two-axis taxonomy with a stability theorem.** Temporal
  axis (absolute 0th-order vs delta 1st-order) × spatial axis (joint
  vs task space), all under chunking. Proposition 4.1: step-wise
  delta decoding multiplies prediction noise by the cum-sum matrix
  L_k with spectral norm ≈ (2k+1)/π — error grows O(k) with chunk
  length — while chunk-wise delta and absolute decode through the
  identity, O(1). At our k=50-class chunks that is a ~32× worst-case
  noise amplification for the step-wise variant. This is the
  mechanism behind the ~10-point chunk-wise > step-wise gap, and it
  is cross-validated with a flow-matching backbone (their Fig. 9).
- **The headline table (progress score, mean ± SE, overall averages
  across single-arm, multi-task, bimanual, RoboTwin-2.0):**

  | head | space | abs | delta | gap |
  |---|---|---|---|---|
  | regression ("ACT") | EE | 63.4 ± 2.7 | 78.4 ± 1.4 | **+15.0** |
  | regression ("ACT") | joint | 71.2 ± 2.9 | 79.7 ± 2.5 | **+8.5** |
  | flow matching ("DP") | EE | 71.9 ± 4.8 | 82.9 ± 1.6 | **+11.0** |
  | flow matching ("DP") | joint | 79.6 ± 2.2 | **88.0 ± 2.3** | **+8.4** |

  Best overall cell: **flow matching + joint + chunk-wise delta** —
  exactly our head and space, minus the delta.
- **Horizon–abstraction coupling.** Delta control degrades when
  executed open-loop too long (drift, stale reference); absolute
  keeps improving to the full 60-step window, then saturates
  (information decorrelation: mutual information between o_t and
  a_{t+k} decays in k). They standardize on exec-30 for delta,
  exec-60 for absolute thereafter.
- **Scaling behavior (Tables 4–5, real robot, flow-matching head,
  joint space):** rel-joint > abs-joint at every data scale — 77.2
  vs 64.4 (100 demos), 90.8 vs 82.6 (250), 93.5 vs 84.2 (500) — and
  at every epoch budget (87.8 vs 71.5 @300 up to 94.3 vs 79.9
  @1200). The delta win does not close with scale; the joint-over-EE
  win *grows* with scale and model capacity.
- **The complementarity result:** joint space wins in-domain;
  task/EE space wins under cross-embodiment (AgileX + AIRBOT
  co-training) and π0-LoRA transfer, attributed to embodiment
  invariance. Their guideline: fixed rig with enough data → joint +
  chunk-wise delta; cross-robot/transfer → EE.

## The experiments it ran

- 4 real tasks (Touch Cube, Pick Cup, Pick & Place, Bimanual
  Transfer) with partial-credit progress scores; workspace uniformly
  partitioned into a 6×6 grid for both collection and eval — an
  unusually honest initial-condition protocol. 30 Hz position
  control throughout.
- RoboTwin-2.0 sim (AgileX embodiment, hard mode), 10 tasks, 50
  demos/task: rel-qpos best overall for both heads (ACT 46.3, DP
  48.0 vs abs-ee 26.7/26.0) — sim agrees with real.
- Grid searches over data (100/250/500 demos), compute
  (300–1200 epochs), single- vs multi-task, plus five
  cross-validation suites in Appendix F.
- **Noise floor to keep in mind:** each cell is 30 rollouts; SEs of
  ±2–10pp are typical, and several per-task flips sit inside noise.
  The averages, not the cells, carry the conclusions.

## What transfers to us

- **Class match is unusually good.** Flow-matching head, chunked
  decoding, joint space, 30 Hz — that is our stack minus the trunk
  size, and the π0-LoRA experiments extend the delta finding to a
  3.3B flow VLA. The delta result is not a single-step-policy
  artifact: everything here is chunked.
- **The specific configuration they crown (joint + chunk-wise
  delta) differs from ours (joint + absolute) by exactly one bit**,
  and in their tables that bit is worth ~8pp progress score for our
  head/space, robust across data and compute scales.
- **A free hypothesis for our corpus:** chunk-wise delta subtracts
  the chunk-start proprio state, which also subtracts any per-rig
  calibration offset. On 229 h of heterogeneous community SO-100/101
  teleop (many rigs, many calibrations), delta targets may be
  *better-distributed* for us than for their single-rig setup.
- **If we ever co-train across morphologies** (SO-100 vs SO-101 is
  near-identical kinematics, so this is latent for now), the EE-wins-
  under-transfer result is the relevant prior.

## What does NOT transfer

- **Hardware class.** AgileX PiPER and AIRBOT are proper 6-DoF arms
  with decent position tracking. Nothing here runs on hobby-servo,
  no-force-control hardware like the SO-101's STS3215s. Chunk-wise
  delta's only feedback dependence is the chunk-start reference
  state — but on sloppy servos that reference (commanded vs actual
  position mismatch under load) is exactly what's noisy. The 30 Hz
  frequency matches; the hardware claim is unmeasured for our class.
- **Training chunk length.** Fixed at 60 everywhere. Our
  chunk-length convention gets no direct evidence; only the
  execution-window coupling transfers as a prior.
- **Their eval is rollouts; ours is offline MAE — and this paper's
  mechanisms live in the gap.** Both headline effects are
  *deployment* effects: step-wise noise amplification happens at
  decode time, and the horizon coupling is drift over open-loop
  execution. A per-frame offline MAE (i) is incommensurable across
  parameterizations unless predictions are decoded to a common
  absolute-joint space first, (ii) cannot see execution-horizon
  effects at all, and (iii) will likely *flatter* delta (better-
  conditioned targets shrink raw regression error) while missing
  delta's real deployment cost (drift). Any offline delta-vs-abs
  comparison we run must decode to absolute joint space before
  scoring and carry this caveat pre-registered.

## Which idea it feeds

- **New pre-registered arm (cheapest justified change to our action
  parameterization): `delta-joint`.** Retrain the action expert with
  chunk-wise delta joint targets — a_{t+k} := q_{t+k} −
  q_{chunk_start} — same data, same frozen trunk, per-dim
  normalization recomputed on the delta distribution. Score on the
  existing held-out panels **after decoding predictions back to
  absolute joint positions** (add chunk-start proprio), paired CI95
  against the current absolute-joint baseline. Pre-register: chunk-
  wise only (step-wise is theoretically and empirically dominated —
  do not spend a run on it); a panel win is necessary-but-not-
  sufficient (offline metric plausibly overstates delta's deployment
  win); a panel *loss* despite this flattery would be strong
  evidence against switching. Cost: one tiny-config probe run first,
  then one full run. This is the single most evidence-backed cheap
  arm in the current queue: +8.4pp in their flow+joint class, robust
  100→500 demos and 300→1200 epochs, direction confirmed in sim, on
  a released codebase.
- **Idea #5 (tokenizer/normalization):** action parameterization is
  a normalization decision in disguise — step-wise and chunk-wise
  delta are *bijective reparameterizations of the same targets* that
  differ by ~10 points at rollout time via decode-time noise
  amplification ((2k+1)/π growth). Whatever we do to action targets
  (normalize, tokenize, reparameterize), the question is what the
  decode map does to prediction noise, not just what the encode map
  does to the training distribution.
- **Offline-eval validity (metric-trust thread):** this paper is the
  cleanest evidence yet that action-space rankings can invert
  between per-frame regression error and rollouts — the two best
  decoders (chunk-wise delta, absolute) are *identical* in offline
  error-propagation terms yet differ by 8–15pp in rollouts via
  learning-difficulty and horizon effects our panels can't see.
  Log as a standing caveat on any panel-based action-space claim.
