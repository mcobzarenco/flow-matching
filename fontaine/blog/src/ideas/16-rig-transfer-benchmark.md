# 16. Few-shot rig-transfer benchmark — `parked` for execution (owner 2026-08-05 21:43Z), instruments banked; **the north star** (owner 2026-08-05 17:20–17:23Z)

*Tag: `rig-benchmark` · idea #16 · [index](../ideas.md)*

- **OWNER STEER 2026-08-05 21:43Z — execution PARKED, priorities
  reweighted:** the rig datasets are small/noisy and a 12-ep fixed
  holdout is high-variance ("really depends on which episodes you
  choose"); owner will collect a better rig dataset later. **Short
  term: improve MAE on the comm holdout and/or attribute where the
  limit comes from** (bigger trunk? bigger image embeddings?
  video-trained trunk? is the flow expert needed at all vs pure AR?).
  Empirical anchor from the owner: lower comm-holdout MAE has always
  translated to good rig fine-tunes; current failure mode is gripper
  *placement* accuracy (grounding), motion is fine; aux tasks
  generalize strikingly (4k ft on AR-100k produced sensible subgoals
  for a fully-OOD instruction — "coiled USB-C cable", "glass pot").
  The instruments below stay banked: they are corpus-agnostic and
  re-run on the future dataset in minutes.

- **External anchor for the premise (deep read 2026-08-07,
  [post](../posts/2026-08-07-pi05-deep-read.md)):** π0.5's Fig. 8 is
  the diversity-buys-transfer bet measured at production scale —
  held-out-home performance scales monotonically with training
  locations (3→104), and at 104 locations MATCHES a control trained
  on the test homes; 97.6% of their phase-1 examples are not the
  target embodiment. Evidence, not proof (their scale: ~400 h,
  ~100 homes) — but the north-star premise now has a citable
  production-scale precedent.

- **Pre-reg draft posted 2026-08-05 ~21:2xZ**
  ([post](../posts/2026-08-05-prereg-rig-fewshot-benchmark.md)): design
  frozen — 12-ep holdout (SeedSequence(16)) + nested N ∈ {10,25,45}
  materialized derived corpora (leakage-checked, the #18.8 consumer);
  owner `run_ft_rig.sh` protocol constants; best-checkpoint-at-200
  selection; co-primary chunk_mae + first-4 pooled MAE; 3·σ_ft
  decision rule (σ_ft from N25 seed replicates); **eligibility gate:
  init pretrain corpus must certifiably exclude the rig repos —
  flow-80k is contaminated (rig data in its pretrain mix), rcond/box
  arms qualify.** Two slots (init selection rule + E5 noise scale)
  fill by finalization amendment after the box reads; execution at
  the first quiet GPU boundary after.
- **Instruments LANDED 2026-08-05 ~21:5xZ (Amendment 1 on the
  pre-reg):** plan frozen (`plans/rig_fewshot_v0_k4l2.json`, 12 eps /
  48 core + 24 labeled; holdout = native split 0.212/seed 16 → v2
  {1,2,3,6,11,15,20,24,25,30,41} + clean {2} — mechanism amendment:
  the draft's bespoke SeedSequence draw could not feed the leakage
  checker); subsets materialized + verified
  (`~/datasets/rig_fewshot_v0/`, n10 6,223 / n25 15,881 / n45 29,107
  frames, videos hardlinked bit-identical, judgments remapped, stats
  recomputed w/ oracle worst |Δ| 1.2e-4); **leakage certs PASSED ×3**
  (#18.8 provenance path, doctored-provenance negative control fails
  loud); loader smoke bit-exact incl. shifted mid-file video decode;
  **wrap census CLEAN on both rig repos** (hygiene gate 1). Remaining
  before launch: launcher gen + finalization amendment (slots 1–2)
  after the box reads.

- **Goal statement (owner):** "build a VLA for my rig… prove transfer
  so you can fine-tune a task on a new SO101 arm with tens of
  examples." Community-panel MAE is the proxy; **the
  sample-efficiency curve is the product metric.**
- **Design sketch (pre-reg to write after the box batch lands):**
  fine-tune the best lineage on N ∈ {10, 25, 50} episodes of a
  held-out rig task; measure panel-style MAE on that task's holdout
  (and eventually rollout success) vs N. Protocol precedent: the
  owner's ft-rig lineage (4–5k-step fine-tunes, `run_ft_rig*.sh` on
  the second box, both AR and flow variants).
- **Dependencies:** tonight's aux-off answer + seed-noise floor pick
  the trunk and set the minimum detectable effect for paired ft
  comparisons; sign/calibration hygiene (ideas #13, #14) bites
  hardest on a new arm — keep them warm.
- **Falsification:** the curve itself — if MAE at N=50 is no better
  than zero-shot, transfer is not proven and the pretraining recipe
  (not the ft protocol) is the suspect.
- Reweights the whole list: rig-transfer relevance now outranks
  community-panel micro-optimization at equal cost.
- **Metric note (2026-08-05 paired analysis):** the pre-reg must fix
  the deployment replan interval k and quote first-k pooled MAE next
  to chunk_mae — the flow-vs-AR ranking *flips* at k≤3 vs k≥5
  ([post](../posts/2026-08-05-flow-vs-ar-paired.md)); chunk_mae alone
  is the most AR-favorable point on that axis.
- **Literature (2026-08-05 slice): the ft-protocol arm should
  include LoRA-r32 + full vision-encoder ft** (arXiv:2607.10172,
  π0 on UR5e precision assembly): LoRA saturates at r=32 with no
  significant FFT advantage; **freezing or LoRA-restricting the
  vision encoder significantly degrades** (independent external
  support for our grounding-bottleneck reads, idea #11); static
  peak VRAM 36.2→10.8 GiB — on 1×H100 that headroom converts
  directly to batch for the few-shot fine-tunes. **Papers-page
  re-read 2026-08-07 ([page](../papers/data-and-trunks.md)):
  CONFIRMED on all counts, now numeric** — r=32 at 0.74 vs FFT 0.76
  (p=1.000); SigLIP frozen 0.14 / SigLIP-LoRA 0.43 vs 0.74 fully
  trainable; metric is ATP (sub-goal progress), not success rates;
  plateau beyond r=32 may partly be the α=r scaling rule.
- **Proxy-validity slice (2026-08-07,
  [page](../papers/offline-validation.md), 5 sources): our metric
  class is measured.** CI-MSE (2606.29898) correlated raw
  validation MSE against real+sim rollout success over 27 VLA
  checkpoints (π0.5, X-VLA, GR00T N1.7): Spearman −0.61 — and in
  their data-scale family raw MSE ranked checkpoints *backwards*
  (+0.90). Their repair (score only task-critical frames +
  rollout-like alignment) reaches −0.87. **Rung EXECUTED 2026-08-07
  same-day
  ([pre-reg + results](../posts/2026-08-07-prereg-critical-frame-repooling.md)):
  every published ranking HOLDS on the critical-frame pool** — all
  10 pairwise gaps keep their sign with CI95 excluding 0, coverage
  99.9%, and the model-vs-state-copy separation *widens* on critical
  frames (the opposite of CI-MSE's easy-frame-dilution failure
  mode). Robustness citation banked on the leaderboard; instrument
  `critical_frame_repooling.py` reusable for future rows (molmo2
  endpoint). Rollout-vs-offline stays open until the rig benchmark
  exists. Also
  banked: MMRV (SIMPLER 2405.05941) as the scoring rule for any
  future proxy-vs-rig audit, weighted by real margins rather than
  Pearson; AutoEval's (2503.24278) caveat that proxy fidelity is
  policy-dependent — a proxy validated on one family doesn't
  transfer free to the next.
- **Frozen-trunk few-shot lever #2 banked (2026-08-07 ~20:2xZ,
  [noise-space-steering page](../papers/noise-space-steering.md)):**
  FRS/DSBC (2606.13675) — recover the latent noise behind ~10 good
  reference trajectories by reverse-ODE through the frozen flow
  policy, distill into a tiny auxiliary noise policy (<1 min
  training, ~1 GB; up to +95% absolute success on real tasks, +60%
  avg across 6 DROID tasks from 10 human-steered rollouts). Needs
  reference actions, not rewards — closer to our rig-data reality
  than DSRL-style RL (which stays gated on this benchmark existing).
  Joins VLA-Talker's evidence-injection few-shot hook; flow-family
  only (explicitly inapplicable to the AR trunk).
- **Frozen-trunk rig lever #3 banked (2026-08-07 ~20:3xZ,
  [noise-steering II page](../papers/noise-space-steering-2.md)):**
  UniSteer (2605.10821) — human corrective actions converted to
  noise-space supervision by per-step fixed-point inversion through
  the frozen flow decoder (M=16 iterations, ~0.1 s/sample), then
  SFT-then-RL on a lightweight noise actor. Real π₀/AgileX: 20%→90%
  average over four tasks in ~66 min, vs DSRL 55% and DAgger 60%;
  needed ~1 pure-human trajectory per round where DAgger needed 8;
  **OOD positions 100% where DSRL drops to 0–25%** — the strongest
  evidence yet that the *supervised* rungs of the noise ladder beat
  pure noise-RL at small budgets. Ordering prior banked: SFT→RL 95%
  vs RL-only 60% (corrections first, reward polish second). Needs
  live teleop corrections — one step up the hardware ladder from
  DSBC's 10 recorded demos (lever #2). The rig-time menu is now:
  ticket (zero machinery) → DSBC (10 demos) → UniSteer (teleop
  pedal + ~1 h/task) → DSRL (rewards + rollouts). Flow-family only.
- **Lever #3 corroborated + retention number banked (2026-08-09,
  [FlowDAgger page](../papers/flowdagger-latent-dagger.md),
  2607.08877):** the same inversion-of-corrections recipe as
  UniSteer (per-step fixed point, M=5 vs 16) run as an explicit
  DAgger loop at 5–20 interventions/task on π0.5, Cosmos-Policy,
  Gr00t and vanilla diffusion — MetaWorld +0.25 mean vs SFT +0.18 /
  LoRA-DAgger +0.15 / DSRL +0.02; real bimanual 13%→80% w/ 10
  corrections. The new number the rung was missing: **held-out
  retention 0.88 under latent adaptation vs LoRA −0.66 / SFT −0.94**
  — weight-space adaptation's forgetting cost, measured. ~8 GB
  train budget = deployment hardware.
- **The weight-space pole of the post-SFT menu banked (2026-08-09,
  [Hy-Embodied stack page](../papers/hy-embodied-stack.md),
  2606.14409):** FlowPRO/RPRO — preference RL on a flow policy with
  the flow loss itself as implicit reward
  (`r = (β/2)(ℓ_ref − ℓ_θ)`, no reward model), labels from an
  intervention-and-rollback teleop loop (failure + correction =
  preference pair; same intervention currency as levers #3/#3′).
  Real bimanual: 94–99% SR, +6–12 pts over DAgger with the same
  interventions, and *faster* executions (16 s vs 27 s — preference
  pairs penalize dithering, positive-only imitation can't). Caveat
  loud: **retention never measured** — FlowDAgger's −0.94 SFT
  forgetting critique stands unanswered against any weight-space
  recipe; if this menu ever runs on the rig, held-out retention is
  the first read to demand. Same page banks the deployment lever:
  latency-aware cubic-Bézier chunk stitching + async
  producer-consumer loop at exactly our H=50 chunk length — the
  chunk-boundary-continuity piece our decode-cost story doesn't
  measure yet.
- **Rollout-eval design inputs banked (2026-08-09,
  [async execution II](../papers/async-execution-2.md)):** (1) the
  reaction-time identity E[Δt_react] = Δt_infer + ½·Δt_exec (FASTER,
  2603.19199) — on the rig the *execution horizon we choose* will
  likely dominate decode latency, so TTFA (time-to-first-action) is
  the metric to instrument, not raw inference ms; FASTER's streaming
  numbers + [HyperVLA's 4 ms pole](../papers/hypervla-hypernetwork-inference.md)
  bracket the latency design space. (2) ABPolicy's (2602.23901) jerk
  instruments — 95th-percentile acceleration + velocity
  zero-crossing rate — a ready-made smoothness read for rig
  rollouts, same family as the SDN jerk read that showed our ODE
  draws already uniformly smooth within-chunk (boundary jerk is the
  open term). Zero measurement now; both slot into the bench design
  when the owner's better rig dataset lands.

- **2026-08-09 — weight-space post-training pole, second recipe
  ([ForesightFlow page](../papers/foresightflow-self-scored-bestofk.md),
  2606.04968):** decoupled advantage-weighted flow matching
  (advantage weights on action velocities ONLY; uniform on the
  self-scoring potential channel — coupled weighting demonstrably
  hallucinates value, staged ablation 42.0 vs 51.0 final-stage).
  One joint stage, −38% compute vs critic-based IDQL, ~1K added
  params. Sits beside FlowPRO in the post-SFT menu; retention
  unmeasured in both (the FlowDAgger critique stands). Needs stage
  labels + mixed-quality rollouts — a rig-data-era option, not a
  panel-era one.

- **2026-08-09 fresh sweep — RL pole data-efficiency datum
  ([Z-1 page](../papers/z1-selective-joint-rl.md), 2606.31846):**
  task-wise GRPO over a flow-SDE conversion of the flow decode
  (Gaussian noise into intermediate transitions → per-action
  log-probs) lifts a π0.5-based policy +13.2 pts over its SFT init
  (67.4 → 80.6 avg on 24 RoboCasa tasks) from 1,199 public demos and
  sparse binary success rewards with a 0.998 success-aware decay —
  no reward engineering. Shared-prefix rollouts + tree branching are
  the cost levers; paper reports zero compute figures and is
  sim-only. Sits in the post-SFT menu beside FlowPRO/ForesightFlow
  (weight-space) and the noise-space column.

- **2026-08-09 `lit-radar-0811` — two post-SFT menu entries, the
  poles priced at both infrastructure extremes.**
  ([RLDT page](../papers/rldt-density-transport-rl.md), 2606.08602):
  RL-pole roster entry #3 — SVGD density transport on flow policies;
  the only update *native* to flow matching (no likelihoods, no
  backprop-through-time; per-depth gradients stay well-conditioned;
  repulsion term preserves multimodality by construction). Honest
  price: 64–1,000 parallel envs + trained critic + 30–48 GPU-h per
  task at SMALL policy scale — the whole RL pole is sim-first;
  parallel-env infrastructure, not sample count, is the blocker.
  Its expected-target trick is the same 1-NFE endpoint estimate
  ForesightFlow benchmarked (τ 0.80–0.86), now used for gradients.
  ([FAN page](../papers/fan-feasible-action-neighborhood.md),
  2604.01570, CVPR26): the ZERO-infrastructure pole — one KL term
  at SFT time toward a Gaussian around the policy's own argmax
  (self-referential smoothing, no rollouts/critic/labels); modest
  ID gains, real OOD/perturbation wins (+5–6 pts; 1/30→7/30 on
  their hardest real task). Discrete-token heads only → AR-trunk
  candidate for a future rig fine-tune pre-reg; α benchmark-tuned,
  unimodality-per-state assumption untested on bimodal states.

**2026-08-09 — lit `0812b`: RL-pole entry 4, and the pole's first
measured IND-vs-OOD trade
([π-StepNFT page](../papers/pi-stepnft.md), 2603.02083):**
critic-free step-wise contrastive updates on flow-SDE transitions
(binary success only, no value net, no likelihoods, one forward
pass) roughly match PPO in-distribution but beat it +11.1 pp OOD on
ManiSkill (semantic shift 49.1 vs 25.4, π0) — value-based buys peak
IND, critic-free buys OOD retention. For the few-demo/shifted rig
regime that trade favors the critic-free end. Price unchanged:
8×H100, co-located sim rollouts, sparse success flags; the pole
stays sim-first. Weakness noted: LIBERO-Long 86.7 vs PPO 90.2 —
sparse credit assignment degrades on long horizons. Also filed
([DFM-VLA page](../papers/dfm-vla.md)): iterative-refinement
decoders nearly double AR at 10% data (CALVIN 3.21 vs 1.71) — a
few-shot-regime prior for the head axis.

**2026-08-09 — lit `0813`: RL-pole entry 5, the first measured
NEGATIVE sign ([SA-VLA page](../papers/sa-vla.md), 2602.00743):**
sparse-reward actor-critic PPO on a π0.5 flow policy lands *below*
no-RL (77.5 vs 81.0 on LIBERO-Plus spatial OOD) — the pole's
emerging shape is that the RL update itself is the risk and
published gains are protective machinery (dense geometric rewards
+5.5, frozen spatial injection +2.25 zero-shot, learned exploration
noise +0.75; full pipeline nets +2.75 over SFT, 154 GPU-h, 64
parallel envs, privileged sim rewards). Reusable design pattern:
the noise-parameterization taxonomy — external SDE noise is
invisible to PPO's likelihood ratio; the variance must be a learned
policy output (annealed floor for early coverage). Also
([silent-failures page](../papers/silent-failure-observability.md),
2606.03134): a bench-design constraint for the north star —
telemetry-style success flags run 32–48% false-positive among
flagged successes even with scripted policies in clean sim, so any
rig benchmark (and every RL-pole recipe trained on binary success)
needs an exteroceptive label audit; cheapest sufficient check is a
final-frame scene read.

**2026-08-09 — lit `0814`: RL-pole entry 6
([FPO page](../papers/fpo-flow-policy-optimization.md),
2510.09976, ICRA 2026):** the missing gradient route — a
likelihood-free PPO ratio from the *change in CFM training loss* on
the action (batch-normalized, exponentiated; "mild local
monotonicity" assumed, not proven), no SDE conversion, no BPTT.
π₀-FPO in sim: ALOHA Transfer Cube ~40% → 65%+ own-baseline sparse
reward (the bankable number); LIBERO 87.2 avg is cross-base-model.
The ablation is the roster datum: removing the ratio proxy costs 46
pp and clipping 33, while dropping the Q-ensemble to one critic
costs 7 — **the gradient route carries the method, critic
elaboration is seasoning**. Third independent frozen-trunk vote
(decoder frozen, actor-only), and its degraded variants collapse
below SFT level — consistent with SA-VLA's negative sign. Env
count/compute unreported (the pole's open cost axis gets nothing);
zero OOD/retention measurement. Pole stays sim-first.

**2026-08-09 — lit `0815`: RL-pole entry 7, the first fully
offline + real-robot entry ([RedFlow page](../papers/redflow.md),
2607.27782):** failed deployment rollouts become action-level
corrective supervision with no environment, no teleop, no critic —
an off-the-shelf progress model (Robo-Dopamine GRM) scores chunks,
HDBSCAN clusters matched proprio+progress contexts, and
correctable failures get advantage-weighted attraction / margin
suppression / correction-redirection targets on the flow endpoint.
Real-world average 56.7% → 74.7% across three AgileX tasks from
100–200 deployment rollouts + binary outcomes per task; matches
PPO/GRPO/DDPO on LIBERO-Spatial at ~10× fewer samples (1,536
offline vs 13K–24K on-policy). This re-prices the pole: parallel-env
infra is no longer the universal entry fee, and it bridges to the
intervention levers (UniSteer/FlowDAgger) — corrections without a
human in the loop. Sharpest ablation repeats the
protective-structure-carries-the-sign pattern: dropping the
uncorrectable-failure separation costs −11.5 avg (−20.4 on Goal) —
knowing which failures NOT to correct is the biggest single
component. Caveats for the roster: retention/OOD unmeasured (the
FlowDAgger critique stands), the headline gain is from a
*deliberately weakened* base policy, per-task real numbers are
figure reads, and the GRM is unvalidated on rig-like scenes.

**2026-08-09 — lit `0816`: RL-pole entry 8, the fleet-scale tier
([Learning While Deploying page](../papers/learning-while-deploying.md),
2605.00416, AgiBot):** the pole's first offline-to-online entry on
real hardware at fleet scale — 16 dual-arm G1 robots stream
experience to a central learner (policy broadcast every 50 steps),
humans intervene reactively, and the **VLM trunk stays frozen with
only the flow expert updating, in production RL**. Avg task score
SFT 0.76 → offline RL 0.88 → online 0.95 (short-horizon 0.99,
long-horizon 0.91) after ~60 robot-hours online. The load-bearing
ablation: their novel DIVL critic (categorical distribution over
dataset action-values, quantile-extracted implicit max, entropy-
adaptive τ) vs plain expectile is a wash short-horizon but
**+9.7/+16.7 pts on long-horizon** — the distributional
representation keeps rare successes visible in heterogeneous fleet
data. Policy extraction is QAM — flow-native critic-gradient-to-
velocity-field regression via adjoint dynamics — **adopted from Li
& Levine, not theirs** (hook corrected). Borrowable pre-rig: the
whole offline column (0.88 beats SFT before any online loop), with
the stated prerequisite that their offline buffer contains failures
+ play data with terminal labels — success-only corpora collapse
the advantage signal, so our entry runs through banked rig-day
failure rollouts. Honesty flags: 0.95 mixes binary success with
human rubric scores, trial counts and intervention rates
unreported, per-task robot pools not one generalist deployment.

**2026-08-09 — lit `0817`: the RL-pole's missing ingredient goes
public, and two rig-benchmark metrics join the design set
([ArmnetBench](../papers/armnetbench.md) 2607.24481 +
[Legato](../papers/legato.md) 2602.12978):** LWD's stated
prerequisite — failure rollouts with terminal labels, which
success-only corpora can't provide — now exists as a public
artifact on our exact embodiment: 2,288 labeled failures + 106
suboptimal across 3,718 LeRobot-v3.0 episodes (Apache 2.0, 7
policy families incl. flow-based π0/π0.5 and Molmo-trunk
MolmoAct 2 — which ranked 6/7 at 18.9% under the 50-demo budget,
with a camera-conditions asterisk). Banked as the designated
offline calibration/eval corpus for the pole's pre-rig column.
Flags carried: no inter-rater agreement, n≈30 per task–policy
cell (±15–18 pt CIs), task confounded with cell. From Legato:
completion time (−19–23% vs RTC at equal scores = hesitation, not
frame-level smoothness) and boundary-overlap RMSE join the
benchmark's candidate metric set — offline chunk-MAE panels are
structurally blind to seam behavior. Menu unchanged; still the
benchmark-design ledger.

**2026-08-09 — lit `0819`: the rollout-substrate blocker is
mechanically GONE, and the rig phase gets its binding forgetting
precedent ([Squint](../papers/squint.md) 2602.21203 +
[SO-101 VLA benchmark](../papers/so101-vla-benchmark.md) 2606.08881 +
[CL triangle](../papers/cl-triangle.md)):** Squint ships an MIT
SO-101 digital twin as registered ManiSkill3 gym envs — success
predicates, arbitrary-resolution RGB (`sensor_configs` kwarg),
`pd_joint_pos` with `normalize_action=False` (LeRobot-convention
absolute joints, 5+gripper), verified installable file-by-file; sim
compute is negligible next to Molmo2-4B inference. Correction: SO-101
was never upstreamed to ManiSkill3 (vendored from a community
`lerobot-sim2real` PR into their repo). What #16 inherits is a
*design* problem, not an access problem: the default visual world is
one wrist cam over black-composited primitives — far OOD for our
multi-view 229h policies (their in-domain BC baseline: 41.9% sim) —
so first use is relative A/B screens + probe-label generation, with
mitigations already in-repo (ThirdCameraEnv one-line switch,
`apply_overlay=False`, swappable overlay). Their 96.1%→91.3%
ranking-preserving transfer (4 methods) is the first quantitative
sim↔real correlation on our exact arm. From the benchmark paper, the
anti-pattern list with one keepable axis set: pilot tasks into the
20–80% success band (2 of their 4 tasks wasted on ceiling/floor),
≥50 trials/cell or paired designs (n=20 = ±22pp), pre-register
multi-label vs primary-label failure annotation; keep their
execution-dimension framing (control fidelity / grounding / temporal
consistency / precision). From the CL triangle, the rig phase is
literally 2605.26820's experiment: pre-register that rig FT carries
229h-corpus episode replay at ρ ∈ [0.02, 0.2] on ~20% of batches —
naive rig-only FT wipes prior competence within a few thousand steps
(BWT −81 by 4×4k), and replay beat joint retraining at matched
compute.

Lit `0820` 2026-08-09 ([rollout-free eval](../papers/rollout-free-eval.md)
2607.01060 + 2512.16881, + [FACTR 2](../papers/factr2-torque-estimation.md)):
the eval-substrate menu gets its priced third tier. PolaRiS (MIT
code live, 224 stars) scans a real scene into IsaacSim in <1 h
(2DGS→mesh + TRELLIS assets, wrist cams render — the SIMPLER
blocker gone) and calibrates at r=0.9 over 24 policy-env points
(worst env 0.81, best MMRV) — but the certificate needs
per-checkpoint co-training (1k steps, 10% sim, ~350 teleop sim
demos; over-tuning degrades the instrument) and is DROID-only, so
SO-101 restarts calibration from zero. The world-model route
(RoboWorld r=0.989 vs RoboArena, n=8) is not actionable: no
artifact, GPT-4o judge never human-validated. Shared lesson: every
rollout-free certificate was purchased with real rollouts. Two
rig-day riders banked: (a) capture a 2–5 min workspace scan + a
calibration board when the better rig dataset is collected —
minutes of cost, unlocks the PolaRiS route retroactively; (b)
FACTR 2's 10-min free-motion protocol — log Present_Load +
positions, train the 1-minute LSTM, check residual spikes on
contact (unproven at the STS3215 servo class; paper floor is a
$2,500 Piper). Design constants worth keeping: 20 real
rollouts/policy/env sufficed for ranking ground truth;
progress-scale scoring beat binary (ρ 0.970 vs 0.922); report
Pearson + MMRV. PolaRiS also independently replicates our
offline-validation read (action MSE poorly correlated; sim-success
saturation with real performance spanning the spectrum). No new
arm — execution stays parked.

**Lit `0821` 2026-08-10
([Curse of Precision](../papers/curse-of-precision.md), 2607.23108 +
[NeuralActuator](../papers/neuralactuator.md), 2607.11734 +
[GigaWorld-1 / WMBench](../papers/gigaworld-wmbench.md),
2607.02642).** Three bench-design inputs in one slice. (1) Precision
tasks get a design rule: build ONE task at 2–3 tolerance levels
(re-sleeving the peg/hole is the knob that keeps cells inside the
banked 20–80% band), fit the precision ceiling c across levels as a
target-SR-independent headline metric, and report config changes as
Δc rather than ΔSR-at-one-tolerance (their wrist-cam removal =
+1.5 mm on c; the smooth-vs-erratic degradation curve doubles as a
debug instrument). Caveat carried: c is a rollout-sweep fit —
sim-only, Franka-only, diffusion-only in the paper — so it is a
rig-phase instrument, not a pre-rig computable. (2) The FACTR 2
rig-day rider is SUPERSEDED by a shovel-ready one: NeuralActuator's
third platform is our exact arm — force MAE 0.47–0.73 N from
Feetech load registers alone (no current sensor; torque via
differentiable simulation, no calibration), MIT code + 3 SO-101
checkpoints + teleop code all verified live. Rig day should log
their 46-column servo schema (pos/goal/vel/load/volts/temp
@~62 Hz); that makes a virtual force sensor + motor-health monitor
nearly off-the-shelf. Caveats: vertical-payload-only validation at
our class, ~0.5 N noise floor; and the corpus still can't feed it —
the #9 zero-GPU Δq_d contact gate stands exactly as banked (their
two-stage contact-probability gate shape is the one upgrade). (3)
The eval-substrate menu's world-model tier updates: the banked
"no artifact" half of the verdict is dead (GigaWorld-1 Nano 1.3B /
Pro 5B Apache-2.0 weights + LeRobot-format pipeline + a VLM judge
with measured 87.8% human agreement, all verified live 08-10;
Ctrl-World MIT + DROID checkpoint live too) — and WMBench
contributes a zero-rollout *pre-trust replay screen* that runs on
our corpus as-is (replay held-out actions, compare generated vs
real video). But the "uncalibratable" half stands: its 324K
"rollouts" are human-graded world-model videos under replayed
actions — no policy drives, and Corr(real policy success, WM score)
is defined in the paper and never computed. Screen ≠ certificate
(Ctrl-World's MMRV 0.22 is the proof); policy-ranking calibration
still costs real rollouts. No new arm; execution stays parked.

New 2026-08-10 (lit `0822`, the final slice before the owner pause;
[PhAIL](../papers/phail.md) 2605.29710, Positronic Robotics — full
release verified: ~990-episode dataset + `build/stats.py` analysis
pipeline + Rerun-based annotation audit tooling, phail.ai live):
the bench's statistical-protocol question ANSWERED. Their protocol
— per-event time-to-success instead of binary outcomes, Kaplan–Meier
CDFs with timeouts right-censored and hard failures absorbed at
T=∞, macro-averaged two-sample KS across objects with
episode-clustered bootstrap p-values — resolves 2 of 3 close VLA
pairs at 25–30 episodes/cell where their binary-test sizing needs
600–1500 paired rollouts (the closest pair still fails at N=30).
The radar's human-anchor worry DISSOLVES: the KS machinery is
purely model-vs-model; the human teleop reference only normalizes
the headline scalar (HRT = RMST ratio at τ=240 s, best VLA 13.8% of
human pace) — collect one teleop block per rig day for the readable
number, skip it with zero statistical cost. Adopted as design
inputs (not commitments): keep ≥50 single-attempt trials/cell as
the BUDGET (their N counts ~4.4-event episodes ⇒ ~130 correlated
events per cell — 30 episodes ≠ 30 trials; SO-101 servo noise
pushes required N up), adopt KS-on-CDFs as the ANALYSIS that lets
some comparisons close early; blinded same-session policy rotation
+ spatial-nuisance logging become hard protocol requirements
(their camera/tote side swap moved GR00T 22.2 pp — larger than the
model gap under study); per-item timestamps from synchronized
video with telemetry as proposer-not-truth (their 42%
telemetry/operator disagreement independently replicates our
32–48% telemetry false-positive finding); lift `build/stats.py`
rather than re-derive. Rider: their aggregation-disagreement
result (macro-AUC and RMST rank the same three models in opposite
order) is the loudest argument yet for publishing the full CDF
panel, not one scalar.
