# Qwen-VLA: the early-fusion pole, and another frozen-first recipe

*Lit slice 2026-08-09 (work session 12:15Z, fourth read — the last
banked hook cleared; the radar backlog is empty again). Qwen-VLA
([2605.30280](https://arxiv.org/abs/2605.30280), May 2026). Fed #17
(trunk ledger: the early-fusion pole, staked at production scale),
#4 (Stage I trains the expert under a frozen trunk before any joint
stage — a second production F-shape vote filed pre-Δ_seam), #19 (a
production decode-temperature datapoint: train at τ=1.0, deploy at
τ=0.6), #16 (embodiment-aware prompts + the data recipe).*

## The paper in plain words

Most robot-policy models bolt a vision encoder onto a language
model and then attach an action generator. Qwen-VLA starts from a
*natively multimodal* base instead — Qwen3.5, where image tokens
are interleaved straight into the text stream from the start — and
attaches a 1.15B-parameter flow-matching action expert that reads
the trunk's hidden states. One model handles manipulation,
navigation, and trajectory prediction across different robot
bodies, told apart by plain-text "embodiment prompts" describing
each robot's morphology. Training walks four stages: first the
action expert alone learns text-to-action with the backbone
*frozen*; then everything unfreezes for multimodal pretraining on
~15,000 hours of robot, human-egocentric, and synthetic data; then
supervised finetuning; then a narrow RL stage. The result posts
strong numbers nearly everywhere it's evaluated — LIBERO 97.9%,
RoboTwin-hard 87.2%, and a large out-of-distribution gap over
π₀.₅ on real ALOHA (76.9% vs 41.5%).

## Contribution

- **Trunk:** Qwen3.5 (4B) — early fusion (ViT tokens with spatial
  merging interleaved into the text stream; no separate
  encoder+projector seam), hybrid attention (gated linear attention
  in most layers, grouped-query softmax at intervals).
- **Expert:** single-stream DiT, 16 blocks, ~1.15B params —
  concatenates VLM hidden states with the noisy action chunk into
  one sequence under joint self-attention, AdaLN timestep
  conditioning, flow-matching objective, few Euler steps at
  inference.
- **Recipe (the part we care about):** Stage I "text-to-action" —
  expert trains, **backbone frozen**; Stage II continued
  pretraining — joint, VL data mixed in (loss weight 0.1 VL vs 1.0
  action at SFT); Stage III SFT; Stage IV PPO with a
  flow-matching log-prob, stop-gradient guarding the value head
  from the backbone.
- **Embodiment-aware prompt conditioning:** robot morphology and
  control conventions described in text, not baked into heads.

## Experiments

- **Manipulation:** LIBERO 97.9 (π₀.₅ 97.6, ABot-M0 98.6);
  RoboCasa-GR1 56.7 (π₀.₅ 37.0); Simpler-WidowX 73.7 (π₀.₅ 46.9);
  RoboTwin easy/hard 86.1/87.2 (π₀.₅ 82.7/76.8 — note the *hard*
  split holds up, where most methods sag).
- **Real ALOHA:** in-domain 83.6; **OOD 76.9 vs π₀.₅'s 41.5** —
  the headline gap. Pretraining ablation: 48.5 → 83.6 (+35.1)
  with the multimodal pretrain stages.
- **Navigation:** beats StreamVLN on R2R/RxR (59.6 vs 52.9 RxR
  SR) — same weights, no navigation-specific head.
- **Base vs Instruct:** the pretrain-only base already posts 90.8
  LIBERO / 64.3 WidowX; instruction stages add +7 to +22.
- **Decode note:** RL rollouts sample actions at τ=1.0; deployment
  sharpens to **τ=0.6**.

## What transfers to us, what doesn't

- **A second production frozen-first vote, filed hours before the
  Δ_seam read.** Stage I is exactly the F-then-joint shape (#4's
  named escalation rung): warm-start the expert under a frozen
  trunk, *then* unfreeze — the recipe APT motivated
  (random-init experts wreck trunks) and ActionX replicated. With
  RDT2 (frozen-trunk expert, no joint stage ever) this makes two
  production stacks this week whose first stage protects the trunk
  from a random-init expert. Ledger context only — the frozen
  read stays untouched, and note the *disanalogy*: their Stage I
  is language-only text-to-action, not our full-recipe F arm.
- **The early-fusion pole is now staked with numbers (#17).** Our
  trunk ledger had late-fusion entries (Molmo2-style
  encoder→projector→LLM) and VLM4VLA's nine-trunk sweep; Qwen-VLA
  is the first production VLA on a *natively* early-fused trunk,
  and its OOD gap (76.9 vs 41.5) is the kind of generalization
  claim the fusion choice is supposed to buy. Confound carried
  loudly: the comparison π₀.₅ has a different trunk, different
  data (their 15k hours vs π₀.₅'s), different everything — this
  is a stack-vs-stack result, not a fusion ablation. No
  fusion-controlled experiment exists in the paper.
- **τ=1.0 train / τ=0.6 deploy is a production dT datapoint
  (#19).** Our dT table found mild monotone improvement toward
  cooler temperatures (mean-collapse direction); a production
  stack independently landing on 0.6 for deployment sharpening is
  consistent with that shape — banked beside the table, not as
  evidence (different mechanism: theirs is RL-rollout exploration
  vs deployment exploitation).
- **VL co-training at weight 0.1 beside action 1.0** is a
  same-shape datapoint for our α-weighted CE aux in the K arm
  (theirs guards language capability, ours guards the trunk — but
  both are "keep the trunk's original objective alive at ~1/10
  weight during action training").
- **Doesn't transfer:** every absolute number (their data scale
  is ~15k hours plus 7.2M synthetic trajectories — orders beyond
  ours); the embodiment-prompt mechanism (single-embodiment
  program); the RL stage (SimplerEnv rewards, deliberately
  narrow); latency claims (unquantified — "few Euler steps" with
  no ms figure, and the 1.15B expert is not obviously rig-fast).

## Where it lands

- **#17**: trunk-ledger entry — early-fusion pole staked
  (Qwen3.5-4B, gated-linear hybrid attention, 1.15B single-stream
  DiT expert at ~22% of total params); stack-vs-stack OOD gap
  carried with the no-fusion-ablation confound loud.
- **#4**: F-then-joint production vote #2 (Stage I frozen-trunk
  expert warm-start), filed as Δ_seam ledger context beside RDT2;
  disanalogy (language-only Stage I) noted.
- **#19**: τ=0.6 deployment sharpening banked beside the dT
  table as a production sighting of the cool-side preference.
- **#16**: embodiment-prompt conditioning + the 74/6/7.5/3.7 data
  mixture filed for the rig-data-era design conversation.
