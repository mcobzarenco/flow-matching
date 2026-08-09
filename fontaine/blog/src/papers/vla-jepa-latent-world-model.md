# VLA-JEPA: pretrain the trunk to predict the future in latent space, then bolt the head on

*Read 2026-08-09 (lit slice `lit-radar-0811`, priority 4: trunk
pretraining family, #17). Paper:
[2602.10098](https://arxiv.org/abs/2602.10098) — "VLA-JEPA:
Vision-Language-Action model with latent world model" (Sun et al.,
9 authors).*

**The paper in plain words.** Most robot policies learn to map "what
I see + what I'm told" to "what I do" and hope the vision-language
trunk's general knowledge fills the gaps. This paper adds a stage in
between: before any action learning, the trunk is trained to
*predict the near future* — not future pixels (which would reward
modeling irrelevant flicker and camera shake) but future *latent
states*, as computed by a frozen video encoder (V-JEPA2) that was
itself trained to capture what changes meaningfully in videos. The
key hygiene rule is leakage-free supervision: future frames are only
ever used to make training *targets*; the model never gets them as
input, so it cannot cheat by copying. Because the targets need no
action labels, the stage can eat ordinary human video
(Something-Something) alongside robot data (DROID). After
pretraining, a flow-matching action head attaches and fine-tunes.
The claimed payoff is mostly *robustness*: near-parity in
distribution, clearly better under perturbations — and, notably,
most of the robustness comes specifically from the human-video
share of the diet.

## What it contributes

- **The recipe**: Qwen3-VL-2B trunk (trainable) + learnable latent
  action tokens; a 12-layer time-causal transformer predicts future
  V-JEPA2 latents (frozen target encoder, horizon T=8 frames);
  then a DiT-B flow head fine-tunes on top. Pretraining: 220K
  Something-Something videos + 76K DROID trajectories, 50K joint
  steps. The world model is training-time scaffolding — deployment
  is a plain VLA forward pass.
- **Numbers**: LIBERO 97.2 (π0.5 96.9, OpenVLA-OFT 97.1 — a wash);
  SimplerEnv Google-Robot 65.2 best-overall; **LIBERO-Plus
  (perturbation suite) 79.5 vs OpenVLA-OFT's 69.6** — the real
  headline. Real Franka: best ID (80%) and layout-OOD (70%), but
  *loses to π0.5 on task-level OOD* — the authors own this: latent
  dynamics buys physical robustness, not textual reasoning.
- **The ablation that matters**: drop human video and LIBERO barely
  moves (97.2→96.1) but LIBERO-Plus collapses 79.5→62.9, improving
  ~linearly as the human-video share rises. The actionless-video
  stage is a *robustness* diet, not a dynamics-knowledge diet.
- Horizon ablation: T=8 optimal; T=4 underfits, T=16 redundant.

## What transfers to us

1. **A predictive third entry in the representation-supervision
   family (#17/#11).** Our ledger has Spatial Forcing (align
   current latents to a 3D encoder) and the encoder-grafting set
   (swap/align the encoder itself). VLA-JEPA is the same move with
   a *time-shifted* target: predict a frozen external encoder's
   latents at t+k. Same integration point as Spatial Forcing (an
   aux loss on trunk latents), same frozen-target hygiene, and the
   robustness-not-capability payoff profile matches what the
   perturbation column shows in both papers.
2. **The leakage-free discipline is our own oracle pattern,
   confirmed externally** — future frames as targets-never-inputs
   is exactly the shape of our eval leakage checker's contract.
   Their attention-map comparison (LAPA's dense visual leakage vs
   their operation-focused maps) is a nice qualitative
   demonstration of why the hygiene matters.
3. **A priced claim about actionless human video.** RDT2/VISTA
   framed human data as a *scaling* resource needing heavy
   validation; VLA-JEPA gives it a second, cheaper role —
   robustness regularizer via latent-future prediction, no action
   labels, no physics validation pipeline. If a trunk-pretraining
   arm ever opens on the #17 ledger, this is the lowest-friction
   way to spend human video.

## What doesn't transfer

- **Wrong stage for us today.** This is a *pretraining* recipe;
  both our trunks are past it, and retrofitting means a full
  trunk-scale run — exactly the shape the startup-velocity rule
  blocks without an owner-level reason.
- **The task-OOD loss to π0.5** is a real ceiling: latent dynamics
  does not substitute for instruction-following breadth. For the
  north-star rig VLA, language robustness is not the binding
  constraint anyway — but it means this recipe is not a general
  upgrade, it's a trade.
- **2B trunk, 7-dim actions, single-arm suites** — one scale point,
  no compute figures, no evidence at 4B+ or bimanual.

## Which idea/arm it fed

**#17 (new trunks)**: pretraining-recipe family-map entry — the
predictive pole of representation supervision, with the
human-video-buys-robustness ablation as the citable datum. **#11
(visual grounding)** cross-ref: same integration point as Spatial
Forcing; if that family ever runs, the current-vs-future target
choice is the first fork. Cross-refs:
[Spatial Forcing](spatial-forcing.md),
[encoder grafting](encoder-grafting.md),
[RDT2](rdt2-umi-scaling.md) / [VISTA](vista-umi-validation.md) (the
scaling-role of human data this paper complements),
[latent action priors](latent-action-priors.md) (LAPA — the leaky
baseline their attention maps indict).
