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
same session it is banked.

## Pages

| Page | Papers | Fed |
|---|---|---|
| [π0.5 + Knowledge Insulation](pi05-knowledge-insulation.md) | 2504.16054, 2505.23705 | #4 attachment arms, #6 self-subgoal probe, #16 north star, #5 |
| [LabVLA](labvla.md) | 2606.13578 | #4 — the KI-joint arm is the field's incumbent |
| [Q-VGM](qvgm.md) | 2606.08015 | #4 — the frozen arm keeps an offline-RL escalation path |
| [Test-time selection for VLAs](test-time-selection.md) | 2510.05681, 2605.01194, 2602.12281, 2506.17811, 2605.25547, 2607.03751, 2605.28527 | #19 — the six selection flavors behind the best-of-10 ceiling gate |
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
| DVAC (variance-gated replanning) | [2606.03847](https://arxiv.org/abs/2606.03847) | ✅ [page](sampling-beyond-selection.md) |
| Golden Ticket (noise search) | [2603.15757](https://arxiv.org/abs/2603.15757) | ✅ [page](sampling-beyond-selection.md) |
| Energy Policy (energy-score training) | [2510.12483](https://arxiv.org/abs/2510.12483) | ✅ [page](sampling-beyond-selection.md) |

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
