# Papers

Reviews of the papers behind the literature slices. One page per
paper — or per tight theme cluster where the papers only make sense
together — each covering four things: **what the paper contributes**,
**what experiments it actually ran**, **what transfers to our setup
and what doesn't**, and **which idea or experimental arm it fed**
(the `#N` references point into [Ideas](../ideas.md)).

These pages are written for a reader with less context than the
research log assumes. The one-line hooks in `ideas.md` remain the
index into the backlog; the page here is the record of what was
actually read and why it mattered. Standing rule (owner,
2026-08-07): every literature slice lands its Papers page in the
same session it is banked. Standing rule (owner, 2026-08-08
18:57Z): every page opens with **"The paper in plain words"** — a
short, jargon-free summary of what the paper is about and what it
showed, before the dense analysis.

## Pages

| Page | Papers | Fed |
|---|---|---|
| [π0.5 + Knowledge Insulation](pi05-knowledge-insulation.md) | 2504.16054, 2505.23705 | #4 attachment arms, #6 self-subgoal probe, #16 north star, #5 |
| [LabVLA](labvla.md) | 2606.13578 | #4 — the KI-joint arm is the field's incumbent |
| [Q-VGM](qvgm.md) | 2606.08015 | #4 — the frozen arm keeps an offline-RL escalation path |
| [Test-time selection for VLAs](test-time-selection.md) | 2510.05681, 2605.01194, 2602.12281, 2506.17811, 2605.25547, 2607.03751, 2605.28527 | #19 — the six selection flavors behind the best-of-10 ceiling gate |
| [Self-Certainty](self-certainty.md) | 2502.18581 | #6 rung (b) — the frozen verifier-free scorer for subgoal-draws selection |
| [Progress from logits](progress-from-logits.md) | 2602.19313, 2605.28231 | #6 rung-(b) escalation routing — history fixes phase (zero-shot, our trunk family); masked-contrast prerequisite verified met; SC scorer cell survives |
| [SnapFlow](snapflow.md) | 2604.05656 | #12 — replicated on our stack; 1-NFE distillation adopted-signal |
| [The seam debate](seam-debate.md) | 2604.16067, 2605.30877 | #4 — the escalation branches on both sides of the F/K screen |
| [Encoder winners don't transfer](encoder-grafting.md) | 2606.14153 | #4 — the scale-transfer caveat on reading Δ_seam |
| [Hierarchy & subgoals](hierarchy-subgoals.md) | 2606.10267, 2607.04816 | #6 — design constraints + escalation map for the self-subgoal probe |
| [The one-step menu](one-step-menu.md) | 2603.12480, 2603.01469, 2606.05737, 2603.14245 | #12 — the fallback menu SnapFlow made moot, and why it worked |
| [Sampling beyond selection](sampling-beyond-selection.md) | 2603.15757, 2606.03847, 2510.12483 | #1, #19 — noise tickets, variance gates, the adopted ES column |
| [The state shortcut](state-shortcut.md) | 2506.23944, 2509.18644, 2601.16667, 2602.12032, 2602.06575, 2606.22836 | #9, #11 — the crutch we measured, and the mis-banked p=0.8 citation |
| [Grounding & conditioning placement](grounding-conditioning.md) | 2601.16207, 2509.04996, 2602.04208, 2506.01844 | #11 — the acuity-probe triangulation; arm B's published baseline |
| [Action tokenization](action-tokenization.md) | 2501.09747, 2512.04952 | #5 — the v3 refit spec + the learned-VQ falsifiers |
| [Data & trunks](data-and-trunks.md) | 2602.09722, 2604.23001, 2606.31382, 2607.10172 | #9, #16, #17, #18.7 — two skim-banked claims corrected loudly |
| [The attachment frontier](attachment-frontier.md) | 2603.10126, 2607.13429, NVIDIA WAM post | #4, #6, #17 — expert memory, the anchoring third recipe, π0.7 |
| [LP-FT: the schedule with a matched control and a theorem](lpft-two-phase-schedules.md) | 2202.10054, 2405.16747 | #4 f-then-joint (third citation, first with matched control + theory); the a(t),b(t) compute framing |
| [VLM4VLA: nine trunks + module freezing](vlm4vla-trunk-ablation.md) | 2601.03309 | #17 vision-unfreeze prior; #10/#17 proxy-collapse; NOT compute-matched caveat |
| [APT — seam damage as initialization](apt-expert-pretraining.md) | 2606.12366 | #4 — why random-init experts wreck trunks; the F-then-joint escalation rung |
| [ActionX — pre-train the expert, then unfreeze](actionx-rl-expert-pretraining.md) | fnbot.2026.1806605 | #4 — F-then-joint's second same-shape citation (+38 Long over joint-from-scratch); #16 — expert-scoped RL pole |
| [The initialization thread](vla-initialization.md) | 2605.25802, 2601.03309 | #17 trunk criterion, #4 — F's frozen-vision caveat + the vision-first diagnostic |
| [Checkpointing without stalling](checkpointing-systems.md) | CheckFreq, Gemini, 2406.10707, 2511.07035, 2605.17821, 2512.24511 | #18.9 async saves (landed `e3bdc93`) — design corroborated; pinned-buffer + save-frequency hooks banked |
| [ELASTIC — adaptive test-time compute](elastic-adaptive-compute.md) | 2606.31132 | #1 rung-3 candidate (dispersion-gated draw allocation), #19, #6 — R4b's monotone-dispersion read is this paper's premise, measured free on our panel |
| [RoVer — a 0.2B learned verifier](rover-learned-verifier.md) | 2510.10975 | #6 escalation rung (learned PRM if "scorer is the gap"; chunk-step caveat pre-registered ammunition), #19, #1 |
| [Label-free selection signals](label-free-selection-signals.md) | 2605.10158 (+ 2606.14084 re-read) | #6 scorer rung design constraint (score the SET, not the candidate — read the same session as the NO-SCORER verdict); masked-conditioning scorer sketch; jerkpick already banked on [noise-space III](noise-space-steering-3.md) |
| [VLAFlow: training-objective bake-off](vla-training-objectives.md) | 2607.01586 | #4 attachment decision brief (stop-grad −26 pts, frozen-VLM trade table); #6 aux corroboration; NEW future-latent-alignment hook (#6/#17) |
| [Q-guided flow critic](qguided-flow-critic.md) | 2607.02092 | #6 escalation map third scorer shape (gradient guidance, flow-side); #1/#19 rung-3 note + uncertainty gate |
| [FlowDAgger: latent-space DAgger](flowdagger-latent-dagger.md) | 2607.08877 | #16 rig few-intervention adaptation recipe (retention 0.88 vs SFT −0.94); #4 steer-window frozen-capital note; #19/#1 inversion-as-label-source |
| [Hy-Embodied-0.5-VLA: the full stack](hy-embodied-stack.md) | 2606.14409 | #16 — FlowPRO preference RL (weight-space pole, retention-unmeasured caveat) + H=50 Bézier chunk-stitch deployment lever; #4 joint-pole ledger entry under APT's condition |
| [Decode-time stochasticity](decode-temperature.md) | 2605.22493, 2605.29766, 2603.20538, 2605.30660, 2508.20072 | #19 — the dT read's directional prior; 2nd strike on cheap probe selectors; q-token theory anchor |
| [Offline validation: does the panel predict the robot?](offline-validation.md) | 2606.29898, 2605.00066, 2405.05941, 2503.24278, 2602.12691 | #16 — raw-MSE proxy measured at ρ −0.61 (sign flips exist); critical-frame re-pooling rung banked; MMRV for future proxy audits |
| [LAFM: learned prior libraries](latent-action-priors.md) | 2606.23420 | #1 — the rung above the ticket screen on the noise-structure ladder; R4's task-locality read reinterpreted; DSRL named as the next read if stage 1 CONFIRMs |
| [Where should the words come from? HiRoC + VLA-Talker](subgoal-sourcing-post-training.md) | 2608.05999, 2608.05738 | #6 — two fresh directional priors for tonight's self-subgoal probe (cold-start misalignment; injected-vs-supervised language); #16 evidence-injection few-shot hook |
| [Noise-space steering: the ladder above the ticket](noise-space-steering.md) | 2506.15799, 2606.01151, 2606.13675 | #1 — DSRL read (the named next-read); LP-DS trust-region guard banked for any CEM escalation; #16 — FRS/DSBC 10-demo frozen-trunk rig lever |
| [Noise-space steering II: execution + the human loop](noise-space-steering-2.md) | 2606.19774, 2605.10821 | #22 — PAINT banked as the new training-free first arm (chunk-50 π₀, beats the TT-RTC fallback on cost); #16 — UniSteer rig lever #3 (corrections→noise, SFT-then-RL prior); #1 — locality probe noted, no gate change |
| [Runtime plan verification: gate, refresh, recover](runtime-plan-verification.md) | 2604.02965, 2510.16281, 2512.03913 | #6 — the escalation ladder above rung (a) priced (gate needs recovery; subgoal-draws width scaling); #22 — SV-VLA as a drift-monitor competitor; #19/#1 subgoal-draws bridge |
| [Noise-space steering III: attribution + a judge-free selector](noise-space-steering-3.md) | 2603.11642, 2606.14084 | #1 — three pre-reg priors for the per-dataset-tickets rung (interaction-dominated locality 39.4% vs 1.4% noise main effect; path-intact sampler is why the channel exists; boundary artifact = named panel-blind unknown of ticket 33); #19 — SDN's jerk-pick selector queued as a free record-only read on banked draw stacks |
| [The loss and the mask: CCE + FlexAttention](memory-efficient-loss-attention.md) | 2411.09009, FlexAttention docs | #2b, #18 — CE-memory escalation ladder w/ entry condition (row added retroactively 08-08: the page landed 08-08 without its table row) |
| [VEGA: encoder-level 3D-aware alignment](vega-encoder-grounding.md) | 2605.10485 (+ FiT3D 2407.20229 context) | #17 vu5k — the aux-alignment third pole between freeze and thaw (interpretation lever + named cheap escalation); #11 placement echo; #6 aux-family sighting; Spatial Forcing 2510.12276 banked as a new radar hook |
| [HyperVLA: hypernetwork inference](hypervla-hypernetwork-inference.md) | 2510.04898 | #17 trunk ledger — inference-efficiency pole (understand-once/execute-tiny) + the generated-update normalization design rule; #16 rig latency existence proof; MSE-vs-diffusion ablation explicitly NOT read onto AR-vs-flow |
| [Async execution II: shrink, smooth, or train](async-execution-2.md) | 2603.19199, 2602.23901, 2605.19294 | #22 arm menu re-ranked (HAS-on-decode new rung 2; DEFLECT's restart-corrected +1.6–2.3 pp; d≈18 still untested by anyone); #16 TTFA accounting + jerk instruments; #12 fourth pole (one-step head, many-step tail) |
| [Spatial Forcing: convergence, not score](spatial-forcing.md) | 2510.12276 | #17 — the aux pole's second recipe (teacher×depth interaction: VGGT works at LLM-24, collapses at encoder); the 3.8× is a fewer-steps lever, teacher overhead unreported; #11 aux-family; SF may fit single-tower Molmo2 better than VEGA's |
| [RDT2: 10k hours of UMI + the F-shaped recipe](rdt2-umi-scaling.md) | 2602.03310 | #4 F-pole ledger context (AR-first + frozen-trunk expert + distill, no joint stage) pre-Δ_seam; #16 hours-scale data premise + β≈0.23; #5 RVQ priced-first; #12 second 1-NFE production point |
| [QDepth-VLA: predict quantized depth tokens](qdepth-vla.md) | 2510.14836 | #11 aux-family third recipe (generative expert, monocular pseudo-labels); #17 — the only aux-spatial recipe needing no encoder seam (single-tower fallback); #5 quantized-beats-regression +3.9; the −2.9 loss vs −8.5 expert ablation split carried loudly |
| [ForesightFlow: teaching the flow to score its own draws](foresightflow-self-scored-bestofk.md) | 2606.04968 | #19/#1 — seventh selection flavor; the K-sweep evidence anchor (separate 500M critic FLAT K=1→5, self-scored +5.0 — selector shape > size, third strike on post-hoc probes); #12 — 1-NFE endpoint preview with measured ranking fidelity (τ 0.83); #16 — decoupled-AWFM weight-space recipe |
| [Fewer layers than you think (CLP)](fewer-layers-clp.md) | 2606.20246 | #17 — trunk-redundancy ledger opens (33–50% of finetuned-VLA depth is CKA twins; 8 of 16 DiT expert layers free); throughput accounting fourth lever class (fewer layers, train+inference, FLOP-count mechanism); #4 — prune-then-attach named sequel arm |
| [SEAM: closing the chunk seam in noise space](seam-boundary-steering.md) | 2607.04609 | #22 — cheapest bridging arm (closed-form, 1.01× vs RTC's 1.22×, no training); #1 — the cross-chunk half of the boundary term the SDN read couldn't see; boundary-incompatibility CPU read on banked npz banked as a free hook |
| [Robot Critics that Sweat the Small Stuff](robot-critics-small-stuff.md) | 2606.21572 | #19/#6 — trained-critic pole placed and PARKED (needs rollout labels + a video model; ceiling reads cap the payoff on our decodes); one more point that learning the judge is what makes judging work |
| [Qwen-VLA: the early-fusion pole](qwen-vla-early-fusion.md) | 2605.30280 | #17 trunk ledger — early-fusion pole staked (Qwen3.5-4B + 1.15B single-stream DiT; OOD 76.9 vs π₀.₅ 41.5, no-fusion-ablation confound loud); #4 — F-then-joint production vote #2 (Stage I frozen-trunk expert warm-start) filed pre-Δ_seam; #19 τ=0.6 deploy sharpening; #16 embodiment prompts + data mixture |
| [Observation aliasing: when the frame alone can't tell you what to do](observation-aliasing.md) | 2605.14712, 2605.14598 | `fieldcond-subgoal-meta-report` — NN-divergence frame-mining protocol + the delta-concentration chart as the report's central claim; #6 — external baseline shape for the subgoal channel (frame-only 9% → intent-conditioned 45.8% on aliased states; DSSP's strict floor-gap theorem); #11 — aliasing census banked as the entry condition for any history/memory arm |
| [Correcting corrected weight decay](weight-decay-correction.md) | 2512.08217 | `adamc-100k-live` readout — grad-norm chart interpretive frame (flat norms expected, ~nil loss effect; head-exclusion partition validated twice; 10% LR floor on the recommended side; no-steady-state-at-100k caveat); ScionC radar-only |
| [Z-1: unfreeze the trunk only when diagnostics say so](z1-selective-joint-rl.md) | 2606.31846 | #4 fjoint rung — joint phase as diagnostic-gated conditional escalation (4th frozen-first vote); #16 post-SFT menu RL pole (+13.2 pts from 1,199 demos, GRPO over flow-SDE log-probs) |
| [VLA-Corrector: a 40M drift monitor](vla-corrector.md) | 2607.01804 | #6 learned-verifier design constraints (residual target; decoupled external judge +14.8 pp); #22 event-triggered truncation datum (+11.65 of +15.65 pp is *when to cut*); #19 verifier-family sighting |
| [π-StepNFT: step-wise critic-free RL](pi-stepnft.md) | 2603.02083 | #16 RL-pole entry 4 — the pole's first measured IND-vs-OOD trade (critic-free +11.1 OOD over PPO, −5.5 IND); #1 ticket-informed-exploration footnote |
| [DFM-VLA: discrete tokens that get to change their mind](dfm-vla.md) | 2603.26320 | #17 head-axis fourth quadrant (commitment, not discreteness, is the expensive property); #5 MAAT metric-aligned embedding +4.4 pp datum; #16 low-data column (10%: 3.21 vs AR 1.71) |
| [OneWM-VLA: a world model on one token per frame](onewm-vla-one-token.md) | 2605.07931 | #17 predictive-supervision pole, self-anchored variant (14.7M LoRA, no teacher; monotone bandwidth sweep; unsupervised scaffold < nothing); #11 dynamics-aux adjacency |
| [HiF-VLA: codec motion vectors as temporal context](hif-vla.md) | 2512.09928 | #11 history-arm candidate representation (MPEG-4 MVs + decode-stage AdaLN), behind the aliasing-census gate |
| [Muon-SW: the AdamC correction, re-derived for Muon](muon-sw.md) | 2607.23777 | `adamc-100k-live` readout — weight-norm chart expected shape (plateau-then-flat = correction working); λ ∝ η now derived 3 independent ways; alignment-cosine probe banked as free second opinion |
| [AsyncVLA: re-noise the tokens you don't trust](asyncvla.md) | 2511.14148 | #17 commitment-axis datum 3 (within-model: revisability ≫ more denoise compute; coin-flip selector keeps 2/3 of gain); #6 verifier ledger (dense per-token ≫ outcome labels; relative-confidence blind spot); #22 negative placement (not async execution) |
| [Silent failures: proprio vs vision observability](silent-failure-observability.md) | 2606.03134 | #16 bench constraint (telemetry success flags 32–48% false-positive in clean sim → exteroceptive label audit); #6 verifier ledger (modality > capacity; final-state exteroception carries the precision signal) |
| [SA-VLA: spatially-aware flow-matching RL](sa-vla.md) | 2602.00743 | #16 RL-pole entry 5 (naive sparse RL measured NEGATIVE, 77.5 vs 81.0 no-RL; protective-machinery framing; noise-parameterization taxonomy); #11/#17 aux-family fourth mode (frozen feature injection, erosion-proof under RL) |
| [StreamVLA: completion-state gating](streamvla.md) | 2602.01100 | #6 phase-estimation constraint (completion-anchored gate sidesteps the measured mid-execution bottleneck; event-triggered refresh ≈ always-reason at half latency); #22 adjacency (re-reasons, never cuts the chunk) |

## Retroactive backlog

The owner asked (2026-08-07) for retroactive pages covering every
lit slice banked so far. Grouped by theme, most load-bearing first;
landed in three work-session batches the same day. **Cleared
2026-08-07 (batch 3): all 42 sources covered.** The table stays as
the per-paper index; from here the standing rule applies — every
new lit slice lands its page in the same session.

**The attachment seam (#4)** — how to attach a flow expert to a
pretrained trunk:

| Paper | arXiv | Status |
|---|---|---|
| π0.5 | [2504.16054](https://arxiv.org/abs/2504.16054) | ✅ [page](pi05-knowledge-insulation.md) |
| Knowledge Insulation | [2505.23705](https://arxiv.org/abs/2505.23705) | ✅ [page](pi05-knowledge-insulation.md) |
| LabVLA | [2606.13578](https://arxiv.org/abs/2606.13578) | ✅ [page](labvla.md) |
| Q-VGM | [2606.08015](https://arxiv.org/abs/2606.08015) | ✅ [page](qvgm.md) |
| AEGIS (gradient asymmetry) | [2604.16067](https://arxiv.org/abs/2604.16067) | ✅ [page](seam-debate.md) |
| Wall-OSS-0.5 | [2605.30877](https://arxiv.org/abs/2605.30877) | ✅ [page](seam-debate.md) |
| Encoder winners don't transfer across scale | [2606.14153](https://arxiv.org/abs/2606.14153) | ✅ [page](encoder-grafting.md) |
| AR-VLA (history-aware AR expert) | [2603.10126](https://arxiv.org/abs/2603.10126) | ✅ [page](attachment-frontier.md) |
| Representation anchoring | [2607.13429](https://arxiv.org/abs/2607.13429) | ✅ [page](attachment-frontier.md) |
| VLAFlow (objective bake-off) | [2607.01586](https://arxiv.org/abs/2607.01586) | ✅ [page](vla-training-objectives.md) |

**Test-time selection & sampling (#19, #1):**

| Paper | arXiv | Status |
|---|---|---|
| MG-Select | [2510.05681](https://arxiv.org/abs/2510.05681) | ✅ [page](test-time-selection.md) |
| VLA-ATTC | [2605.01194](https://arxiv.org/abs/2605.01194) | ✅ [page](test-time-selection.md) |
| CoVer | [2602.12281](https://arxiv.org/abs/2602.12281) | ✅ [page](test-time-selection.md) |
| RoboMonkey | [2506.17811](https://arxiv.org/abs/2506.17811) | ✅ [page](test-time-selection.md) |
| TapSampling | [2605.25547](https://arxiv.org/abs/2605.25547) | ✅ [page](test-time-selection.md) |
| Look Before You Leap | [2607.03751](https://arxiv.org/abs/2607.03751) | ✅ [page](test-time-selection.md) |
| What Frozen VLAs Already Know About Success | [2605.28527](https://arxiv.org/abs/2605.28527) | ✅ [page](test-time-selection.md) |
| Self-Certainty (best-of-N without a judge) | [2502.18581](https://arxiv.org/abs/2502.18581) | ✅ [page](self-certainty.md) |
| DVAC (variance-gated replanning) | [2606.03847](https://arxiv.org/abs/2606.03847) | ✅ [page](sampling-beyond-selection.md) |
| Golden Ticket (noise search) | [2603.15757](https://arxiv.org/abs/2603.15757) | ✅ [page](sampling-beyond-selection.md) |
| Energy Policy (energy-score training) | [2510.12483](https://arxiv.org/abs/2510.12483) | ✅ [page](sampling-beyond-selection.md) |
| Guided Action Flow (Q-guided critic) | [2607.02092](https://arxiv.org/abs/2607.02092) | ✅ [page](qguided-flow-critic.md) |
| FlowDAgger (latent-space DAgger) | [2607.08877](https://arxiv.org/abs/2607.08877) | ✅ [page](flowdagger-latent-dagger.md) |

**One-step decoding & distillation (#12):**

| Paper | arXiv | Status |
|---|---|---|
| SnapFlow | [2604.05656](https://arxiv.org/abs/2604.05656) | ✅ [page](snapflow.md) |
| One-Step Flow Policy (OFP) | [2603.12480](https://arxiv.org/abs/2603.12480) | ✅ [page](one-step-menu.md) |
| MeanFlow one-step VLA | [2603.01469](https://arxiv.org/abs/2603.01469) | ✅ [page](one-step-menu.md) |
| Let It Be Simple | [2606.05737](https://arxiv.org/abs/2606.05737) | ✅ [page](one-step-menu.md) |
| GoldenStart | [2603.14245](https://arxiv.org/abs/2603.14245) | ✅ [page](one-step-menu.md) (screened out) |

**Hierarchy & subgoals (#6):**

| Paper | arXiv | Status |
|---|---|---|
| Hi-VLA (hierarchy design study) | [2606.10267](https://arxiv.org/abs/2606.10267) | ✅ [page](hierarchy-subgoals.md) |
| CAC-VLA (gated latent-action conditioning) | [2607.04816](https://arxiv.org/abs/2607.04816) | ✅ [page](hierarchy-subgoals.md) |
| π0.7 / world-action models | [NVIDIA WAM post](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/) | ✅ [page](attachment-frontier.md) |

**State shortcut & modality imbalance (#9, #11):**

| Paper | arXiv | Status |
|---|---|---|
| Adapt Your Body (proprio masking p=0.8) | [2506.23944](https://arxiv.org/abs/2506.23944) | ✅ [page](state-shortcut.md) (withdrawn paper) |
| State-free policy | [2509.18644](https://arxiv.org/abs/2509.18644) | ✅ [page](state-shortcut.md) |
| ReViP (state-dominant bias) | [2601.16667](https://arxiv.org/abs/2601.16667) | ✅ [page](state-shortcut.md) |
| GAP (phase-guided gradient scaling) | [2602.12032](https://arxiv.org/abs/2602.12032) | ✅ [page](state-shortcut.md) |
| ThinkProprio | [2602.06575](https://arxiv.org/abs/2602.06575) | ✅ [page](state-shortcut.md) |
| Cloak (visual EE masking) | [2606.22836](https://arxiv.org/abs/2606.22836) | ✅ [page](state-shortcut.md) |

**Grounding & conditioning placement (#11):**

| Paper | arXiv | Status |
|---|---|---|
| IVRA (patch-affinity injection) | [2601.16207](https://arxiv.org/abs/2601.16207) | ✅ [page](grounding-conditioning.md) |
| FLOWER (deep-layer pruning) | [2509.04996](https://arxiv.org/abs/2509.04996) | ✅ [page](grounding-conditioning.md) |
| SCALE (adaptive temperatures — banked title was wrong) | [2602.04208](https://arxiv.org/abs/2602.04208) | ✅ [page](grounding-conditioning.md) |
| SmolVLA (mid-stack conditioning) | [2506.01844](https://arxiv.org/abs/2506.01844) | ✅ [page](grounding-conditioning.md) |

**Data, tokenization & trunks (#5, #9, #16, #17, #18):**

| Paper | arXiv | Status |
|---|---|---|
| FAST (local canon) | [2501.09747](https://arxiv.org/abs/2501.09747) | ✅ [page](action-tokenization.md) |
| FASTer (learned VQ tokenizer) | [2512.04952](https://arxiv.org/abs/2512.04952) | ✅ [page](action-tokenization.md) |
| Rethinking VLA scaling (negative transfer) | [2602.09722](https://arxiv.org/abs/2602.09722) | ✅ [page](data-and-trunks.md) |
| Data-engine survey | [2604.23001](https://arxiv.org/abs/2604.23001) | ✅ [page](data-and-trunks.md) |
| VLM-to-VLA parameter redundancy | [2606.31382](https://arxiv.org/abs/2606.31382) | ✅ [page](data-and-trunks.md) |
| LoRA-r32 fine-tuning study (π0 on UR5e) | [2607.10172](https://arxiv.org/abs/2607.10172) | ✅ [page](data-and-trunks.md) |

**Vision-encoder freeze/unfreeze (#17, owner question 08-07):**

| Paper | arXiv | Status |
|---|---|---|
| OpenVLA (vision-FT ablation) | [2406.09246](https://arxiv.org/abs/2406.09246) | ✅ [page](vision-encoder-freeze.md) |
| MAPS (module-wise proximity scheduling) | [2511.19878](https://arxiv.org/abs/2511.19878) | ✅ [page](vision-encoder-freeze.md) |
| Dual-encoder representation preservation | [2509.11417](https://arxiv.org/abs/2509.11417) | ✅ [page](vision-encoder-freeze.md) |
| VEGA (encoder grounding alignment) | [2605.10485](https://arxiv.org/abs/2605.10485) | ✅ [page](vega-encoder-grounding.md) |
| HyperVLA (hypernetwork inference) | [2510.04898](https://arxiv.org/abs/2510.04898) | ✅ [page](hypervla-hypernetwork-inference.md) |
| ActionX (RL expert pre-training) | [fnbot.2026.1806605](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1806605/full) | ✅ [page](actionx-rl-expert-pretraining.md) |

**Unfreezing schedules under a compute budget (owner steering 08-09 10:38Z, a(t)/b(t) framing):**

| Paper | arXiv | Status |
|---|---|---|
| LP-FT (feature distortion + two-phase schedule) | [2202.10054](https://arxiv.org/abs/2202.10054) | ✅ [page](lpft-two-phase-schedules.md) |
| LP-FT mechanism via NTK (LLMs) | [2405.16747](https://arxiv.org/abs/2405.16747) | ✅ covered in [page](lpft-two-phase-schedules.md) |
| VLM4VLA (9-trunk sweep, module freezing, proxy collapse) | [2601.03309](https://arxiv.org/abs/2601.03309) | ✅ [page](vlm4vla-trunk-ablation.md) |

**Smoothness / boundary family (fed by the 08-09 boundary-incompat read):**

| Paper | arXiv | Status |
|---|---|---|
| SEAM (inference-side seam steering) | [2607.04609](https://arxiv.org/abs/2607.04609) | ✅ [page](seam-boundary-steering.md) |
| FAFM (training-side frequency-space smoothness) | [2606.20135](https://arxiv.org/abs/2606.20135) | ✅ [page](frequency-aware-flow-matching.md) |

**Data ingestion / heterogeneous collection (radar set 08-09):**

| Paper | arXiv | Status |
|---|---|---|
| VISTA (UMI adaptation: fisheye VQA + physics validation) | [2606.04708](https://arxiv.org/abs/2606.04708) | ✅ [page](vista-umi-validation.md) |
| LAFP (latent-action flow policy) | [2606.10517](https://arxiv.org/abs/2606.10517) | ✅ [page](lafp-latent-flow-policy.md) |
| Flowing With Purpose (latent-action FM) | [2606.23420](https://arxiv.org/abs/2606.23420) | ✅ already covered: [LAFM page](latent-action-priors.md) (dup caught 08-09) |

**Fresh sweep 0810 (adamc readout + fjoint sequencing):**

| Paper | arXiv | Status |
|---|---|---|
| Correction of Decoupled Weight Decay (AdamC successor) | [2512.08217](https://arxiv.org/abs/2512.08217) | ✅ [page](weight-decay-correction.md) |
| Z-1 (efficient GRPO for flow VLAs, selective joint training) | [2606.31846](https://arxiv.org/abs/2606.31846) | ✅ [page](z1-selective-joint-rl.md) |

**Radar 0811 (banked hooks from the 0810 fresh sweep):**

| Paper | arXiv | Status |
|---|---|---|
| TCFM (trajectory-consistent flow matching, RK4 decode) | [2605.08511](https://arxiv.org/abs/2605.08511) | ✅ [page](trajectory-consistent-flow-matching.md) |
| RLDT (SVGD density-transport RL on flow policies) | [2606.08602](https://arxiv.org/abs/2606.08602) | ✅ [page](rldt-density-transport-rl.md) |
| FAN (feasible-action-neighborhood prior) | [2604.01570](https://arxiv.org/abs/2604.01570) | ✅ [page](fan-feasible-action-neighborhood.md) |
| HiFlow (tokenization-free scale-wise AR-via-FM) | [2603.27281](https://arxiv.org/abs/2603.27281) | ✅ [page](hiflow-scalewise-ar-flow.md) |
| VLA-JEPA (latent world model) | [2602.10098](https://arxiv.org/abs/2602.10098) | ✅ [page](vla-jepa-latent-world-model.md) |

**Radar 0812b (banked hooks from the 0811 refill sweep):**

| Paper | arXiv | Status |
|---|---|---|
| VLA-Corrector (detect-and-correct inference, adaptive horizon) | [2607.01804](https://arxiv.org/abs/2607.01804) | ✅ [page](vla-corrector.md) |
| π-StepNFT (step-wise negative-aware online RL for flow VLAs) | [2603.02083](https://arxiv.org/abs/2603.02083) | ✅ [page](pi-stepnft.md) |
| DFM-VLA (discrete flow matching iterative refinement) | [2603.26320](https://arxiv.org/abs/2603.26320) | ✅ [page](dfm-vla.md) |
| One-Token-Per-Frame / OneWM-VLA (visual bandwidth in world models) | [2605.07931](https://arxiv.org/abs/2605.07931) | ✅ [page](onewm-vla-one-token.md) |
| HiF-VLA (hindsight/insight/foresight motion representation) | [2512.09928](https://arxiv.org/abs/2512.09928) | ✅ [page](hif-vla.md) |

**Radar 0814 (banked hooks from the 0813 refill sweep):**

| Paper | arXiv | Status |
|---|---|---|
| Hyperball (Fantastic Pretraining Optimizers II, weight-norm equilibria) | [2606.16899](https://arxiv.org/abs/2606.16899) | ✅ [page](hyperball-optimization.md) |
| Anytime Pretraining (horizon-free schedules + weight averaging) | [2602.03702](https://arxiv.org/abs/2602.03702) | ✅ [page](anytime-pretraining.md) |
| VLA-FAIL (zero-failure-data detection: Mahalanobis + chunk consistency) | [2606.21386](https://arxiv.org/abs/2606.21386) | ✅ [page](vla-fail.md) |
| FPO (likelihood-free RFT of flow-matching VLAs, ICRA 2026) | [2510.09976](https://arxiv.org/abs/2510.09976) | ✅ [page](fpo-flow-policy-optimization.md) |
| X-Tokenizer (multimodal action tokenizer as auxiliary supervision) | [2606.14752](https://arxiv.org/abs/2606.14752) | ✅ [page](x-tokenizer.md) |

**Radar 0815 (banked hooks from the 0814 refill sweep):**

| Paper | arXiv | Status |
|---|---|---|
| Weight-norm criticality (loss spikes from decay+normalization driving scale-invariant norms below a critical floor) | [2607.21005](https://arxiv.org/abs/2607.21005) | ✅ [page](weight-norm-criticality.md) |
| Weibull weight-scale (three-force norm decomposition; spline recovery of alignment force from sparse checkpoints) | [2606.19367](https://arxiv.org/abs/2606.19367) | ✅ [page](weibull-weight-scale.md) |
| Decoupled Action Expert (5M MLP ≈ 244M U-Net; task knowledge fits in the conditioning pathway) | [2511.12101](https://arxiv.org/abs/2511.12101) | ✅ [page](decoupled-action-expert.md) |
| Foresight (learned failure detection over action-conditioned world-model latents, outcome labels only, conformal FPR band) | [2606.23085](https://arxiv.org/abs/2606.23085) | ✅ [page](foresight-failure-detection.md) |
| RedFlow (offline failure→correction RL for flow VLAs) | [2607.27782](https://arxiv.org/abs/2607.27782) | ✅ [page](redflow.md) |
| Weight decay improves LM plasticity (pretrain λ 0.5–1.0 beats 0.1 downstream; base loss under-predicts post-finetune quality) | [2602.11137](https://arxiv.org/abs/2602.11137) | ✅ [page](weight-decay-plasticity.md) |
| Learning While Deploying (16-robot fleet offline-to-online RL; DIVL distributional critic + QAM flow-native extraction, frozen trunk) | [2605.00416](https://arxiv.org/abs/2605.00416) | ✅ [page](learning-while-deploying.md) |
| FoMo-FD (inverse-transport nonconformity on a success-only flow world model; 96.6% detection @1.3% FA, wrist-cam-dependent) | [2607.27511](https://arxiv.org/abs/2607.27511) | ✅ [page](fomo-fd.md) |
| VLA-GSE (spectral-init adapter-MoE from the frozen backbone's SVD; init carries the gain, Gaussian-init lands below LoRA) | [2605.06175](https://arxiv.org/abs/2605.06175) | ✅ [page](vla-gse.md) |
| ActionCache (training-free retrieval cache over the flow decode; head-only speedups, trunk untouched — our bottleneck unaddressed) | [2607.06370](https://arxiv.org/abs/2607.06370) | ✅ [page](actioncache.md) |
| MolmoAct2 (AI2 VLA on the Molmo2 trunk: Molmo2-ER backbone, 621M per-layer-KV flow expert, SO-100/101 checkpoint + curated 184h pool) | [2605.02881](https://arxiv.org/abs/2605.02881) | ✅ [deep-dive post](../posts/2026-08-09-molmoact2-deep-dive.md) |
