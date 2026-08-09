# VEGA: align the vision tower to a 3D-aware teacher, then throw the projector away

*Read 2026-08-09 (lit slice, standing allocation — one of the two
unread #17 radar hooks). Paper:
[2605.10485](https://arxiv.org/abs/2605.10485) — "VEGA: Visual
Encoder Grounding Alignment for Spatially-Aware Vision-Language-Action
Models" (Wang et al., submitted 2026-05-11, no venue listed). Sim
benchmark is RoboTwin 2.0 (bimanual) + 4 real ALOHA tasks.*

**The paper in plain words.** VLA models are usually built on vision
encoders that were pretrained on flat 2D images, so they are bad at
judging depth and spatial layout — which matters when a robot has to
grasp things. A recent fix ("Spatial Forcing") teaches the model
spatial awareness by nudging its *middle-of-the-language-model*
features to match a 3D foundation model, but nobody knows which
middle layer to pick, and by that point vision is already mixed up
with language. VEGA's move: apply the same nudge **directly at the
vision encoder's output**, before language ever touches it. The
teacher is DINOv2-FiT3D — a DINOv2 that was itself fine-tuned to be
3D-aware via Gaussian-splat rendering
([FiT3D, ECCV'24](https://arxiv.org/abs/2407.20229)). A tiny 2-layer
MLP projector (~2.1M params) maps student features into teacher
space, a cosine loss pulls them together during fine-tuning, and at
inference the projector is deleted — zero deploy cost. Result:
modest but consistent gains over Spatial Forcing, biggest on "hard"
randomized scenes and real-robot tasks.

## The experiments it ran

- **Setup**: OpenVLA-OFT (Prismatic-7B, LoRA rank 32, dual
  DINOv2+SigLIP encoders). Alignment applies **only to the DINOv2
  branch** (the "spatial" branch), student features from the
  second-to-last encoder block, teacher = final block of frozen
  DINOv2-FiT3D. Loss `L_action + λ·L_align`, λ=0.1, co-trained
  through fine-tuning. 4×H100, ~28 h, 100k steps.
- **RoboTwin 2.0** (6 bimanual tasks × easy/hard, 100 trials):
  averages — vanilla OFT 56.0/22.7, OFT+Spatial-Forcing 64.2/27.8,
  VEGA **67.5/30.7**. The hard-split gaps are where it earns its
  keep (e.g. Place Shoes 0.09 → 0.13 → 0.25).
- **Real ALOHA** (4 tasks, 20 trials each): 0.48 → 0.55 → **0.60**
  average. Small n; treat as directional.
- **Teacher ablation is the interesting one**: no teacher 0.70/0.34
  (per-task Move Card), VGGT-as-teacher 0.76/**0.04** — a geometry
  foundation model as teacher *catastrophically fails* the hard
  split — FiT3D 0.77/0.43. The teacher must live in a feature space
  the student can actually reach; a raw geometry model doesn't.
- **λ sensitivity**: 0.2 already degrades (61.3 easy). The aux must
  stay subordinate to the action loss.
- **Frozen-vs-unfrozen FiT3D probe**: swapping the student's encoder
  for FiT3D directly, frozen ≈ unfrozen — once the features are
  spatially right, unfreezing buys ~nothing.

## What transfers to us

- **A third pole for the #17 freeze-vs-thaw question.** VLM4VLA said
  frozen vision loses uniformly; VEGA's frozen-FiT3D ≈ unfrozen-FiT3D
  probe sharpens *why*: unfreezing pays when the features lack
  something control needs (here: 3D structure), and an aux loss that
  injects the missing structure can substitute for unfreezing. If
  our vu5k screen reads thawed > frozen, "what did the tower learn?"
  has a testable candidate answer — and a VEGA-style aux (teacher →
  cosine → discard projector) is a *cheaper, retention-friendlier*
  escalation than full unfreezing. Costs one frozen teacher forward
  per step, nothing at deploy.
- **Placement evidence rhymes with #11.** Encoder-level alignment
  beats LLM-token-level alignment (Spatial Forcing) at both
  difficulty tiers — inject structure *before* it entangles with
  language. Same shape as our conditioning-placement reads.
- **Aux-riding-the-main-loss** with λ small and the scaffold
  discarded at inference is exactly the #6 aux-attribution pattern
  (aux HELPS actions, +0.462 when off) — third independent sighting
  of "cheap auxiliary supervision, zero deploy cost".

## What doesn't transfer

- **The dual-encoder detail.** VEGA aligns the DINOv2 branch of a
  DINOv2+SigLIP pair. Molmo2 has a single ViT tower — the "align
  only the spatial branch" cleanliness isn't available; an alignment
  aux on our tower would tug the *same* features language reads.
  Whether that fights the VQA-inherited semantics is exactly what a
  rung would need to measure.
- **Gains are small and the base is not ours**: +3.3/+2.9 over
  Spatial Forcing on RoboTwin averages, autoregressive OpenVLA-OFT
  with LoRA, bimanual sim + 20-trial real tasks. No flow expert, no
  seam, nothing that touches F-vs-K or Δ_seam.
- **Teacher fragility is a live caveat**: the VGGT collapse and
  their own "FiT3D transferability may degrade in unstructured
  environments" limitation mean the recipe is really
  "FiT3D-or-nothing" as published.

## What it fed

**#17 vision-unfreeze (vu5k)**: banked as an *interpretation lever
and escalation option* for the readout — if thawed wins, a
spatial-alignment-aux arm (frozen tower + VEGA-style loss) is the
named cheap alternative before committing to unfrozen-vision
lineages; if frozen ties, VEGA's frozen≈unfrozen-once-spatially-right
probe is corroborating context. Cited into the finalization
amendment's set alongside VLM4VLA. **#11** placement echo and **#6**
aux-family sighting noted in their ledgers. New radar hook banked:
[Spatial Forcing 2510.12276](https://arxiv.org/abs/2510.12276)
(ICLR'26) itself — the 3.8× *training-acceleration* claim was not
examined here and would need its own read before anyone quotes it.
