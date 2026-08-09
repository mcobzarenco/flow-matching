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
| [APT — seam damage as initialization](apt-expert-pretraining.md) | 2606.12366 | #4 — why random-init experts wreck trunks; the F-then-joint escalation rung |
| [The initialization thread](vla-initialization.md) | 2605.25802, 2601.03309 | #17 trunk criterion, #4 — F's frozen-vision caveat + the vision-first diagnostic |
| [Checkpointing without stalling](checkpointing-systems.md) | CheckFreq, Gemini, 2406.10707, 2511.07035, 2605.17821, 2512.24511 | #18.9 async saves (landed `e3bdc93`) — design corroborated; pinned-buffer + save-frequency hooks banked |
| [ELASTIC — adaptive test-time compute](elastic-adaptive-compute.md) | 2606.31132 | #1 rung-3 candidate (dispersion-gated draw allocation), #19, #6 — R4b's monotone-dispersion read is this paper's premise, measured free on our panel |
| [RoVer — a 0.2B learned verifier](rover-learned-verifier.md) | 2510.10975 | #6 escalation rung (learned PRM if "scorer is the gap"; chunk-step caveat pre-registered ammunition), #19, #1 |
| [Label-free selection signals](label-free-selection-signals.md) | 2605.10158 (+ 2606.14084 re-read) | #6 scorer rung design constraint (score the SET, not the candidate — read the same session as the NO-SCORER verdict); masked-conditioning scorer sketch; jerkpick already banked on [noise-space III](noise-space-steering-3.md) |
| [VLAFlow: training-objective bake-off](vla-training-objectives.md) | 2607.01586 | #4 attachment decision brief (stop-grad −26 pts, frozen-VLM trade table); #6 aux corroboration; NEW future-latent-alignment hook (#6/#17) |
| [Q-guided flow critic](qguided-flow-critic.md) | 2607.02092 | #6 escalation map third scorer shape (gradient guidance, flow-side); #1/#19 rung-3 note + uncertainty gate |
| [Decode-time stochasticity](decode-temperature.md) | 2605.22493, 2605.29766, 2603.20538, 2605.30660, 2508.20072 | #19 — the dT read's directional prior; 2nd strike on cheap probe selectors; q-token theory anchor |
| [Offline validation: does the panel predict the robot?](offline-validation.md) | 2606.29898, 2605.00066, 2405.05941, 2503.24278, 2602.12691 | #16 — raw-MSE proxy measured at ρ −0.61 (sign flips exist); critical-frame re-pooling rung banked; MMRV for future proxy audits |
| [LAFM: learned prior libraries](latent-action-priors.md) | 2606.23420 | #1 — the rung above the ticket screen on the noise-structure ladder; R4's task-locality read reinterpreted; DSRL named as the next read if stage 1 CONFIRMs |
| [Where should the words come from? HiRoC + VLA-Talker](subgoal-sourcing-post-training.md) | 2608.05999, 2608.05738 | #6 — two fresh directional priors for tonight's self-subgoal probe (cold-start misalignment; injected-vs-supervised language); #16 evidence-injection few-shot hook |
| [Noise-space steering: the ladder above the ticket](noise-space-steering.md) | 2506.15799, 2606.01151, 2606.13675 | #1 — DSRL read (the named next-read); LP-DS trust-region guard banked for any CEM escalation; #16 — FRS/DSBC 10-demo frozen-trunk rig lever |
| [Noise-space steering II: execution + the human loop](noise-space-steering-2.md) | 2606.19774, 2605.10821 | #22 — PAINT banked as the new training-free first arm (chunk-50 π₀, beats the TT-RTC fallback on cost); #16 — UniSteer rig lever #3 (corrections→noise, SFT-then-RL prior); #1 — locality probe noted, no gate change |
| [Runtime plan verification: gate, refresh, recover](runtime-plan-verification.md) | 2604.02965, 2510.16281, 2512.03913 | #6 — the escalation ladder above rung (a) priced (gate needs recovery; subgoal-draws width scaling); #22 — SV-VLA as a drift-monitor competitor; #19/#1 subgoal-draws bridge |
| [Noise-space steering III: attribution + a judge-free selector](noise-space-steering-3.md) | 2603.11642, 2606.14084 | #1 — three pre-reg priors for the per-dataset-tickets rung (interaction-dominated locality 39.4% vs 1.4% noise main effect; path-intact sampler is why the channel exists; boundary artifact = named panel-blind unknown of ticket 33); #19 — SDN's jerk-pick selector queued as a free record-only read on banked draw stacks |
| [The loss and the mask: CCE + FlexAttention](memory-efficient-loss-attention.md) | 2411.09009, FlexAttention docs | #2b, #18 — CE-memory escalation ladder w/ entry condition (row added retroactively 08-08: the page landed 08-08 without its table row) |
| [Observation aliasing: when the frame alone can't tell you what to do](observation-aliasing.md) | 2605.14712, 2605.14598 | `fieldcond-subgoal-meta-report` — NN-divergence frame-mining protocol + the delta-concentration chart as the report's central claim; #6 — external baseline shape for the subgoal channel (frame-only 9% → intent-conditioned 45.8% on aliased states; DSSP's strict floor-gap theorem); #11 — aliasing census banked as the entry condition for any history/memory arm |

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
| VEGA (encoder grounding alignment) | [2605.10485](https://arxiv.org/abs/2605.10485) | radar hook, unread |
| HyperVLA (hypernetwork inference) | [2510.04898](https://arxiv.org/abs/2510.04898) | radar hook, unread |
