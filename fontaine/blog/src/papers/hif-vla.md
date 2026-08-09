# HiF-VLA: motion vectors from the video codec as free temporal context

*Read 2026-08-09 (lit slice `lit-radar-0812b`, priority 5,
skim-to-place per the banked hook). Paper:
[2512.09928](https://arxiv.org/abs/2512.09928) — "HiF-VLA:
Hindsight, Insight and Foresight through Motion Representation for
Vision-Language-Action Models" (CVPR 2026).*

**The paper in plain words.** Most robot policies look only at the
current camera frame — they are blind to how the scene has been
*moving*, which matters in long tasks where the current frame alone
is ambiguous. Feeding in a stack of past frames fixes that but
triples inference cost. This paper's trick: video codecs (MPEG-4)
already compute a compact summary of inter-frame motion — the
"motion vectors" used for compression, a coarse 16×16-block
displacement field that comes essentially free with the video
stream. HiF-VLA encodes those past motion vectors as "hindsight"
tokens (a small ViT), injects them into the action expert via AdaLN
conditioning ("insight"), and also has the model predict *future*
motion tokens as an auxiliary ("foresight"). Modest but consistent
gains over OpenVLA-OFT at half the latency of multi-frame stacking.

## What it contributes

- **The representation**: MPEG-4 macroblock motion vectors —
  h×(H/16)×(W/16)×2 displacement tensors, decoded from the codec,
  no optical-flow network — as the temporal-context carrier.
- **Wiring**: 4-layer ViT with 3D convs encodes MV history →
  hindsight tokens condition a 6-layer joint expert via AdaLN
  (injected at the *decoding* stage, not into the VLM — their key
  ablation: the residual-like decode-stage path wins, VLM injection
  disrupts pretrained alignment); learnable queries predict future
  motion + action tokens (foresight aux). Prismatic-7B backbone.
- **Numbers**: LIBERO-Long 94.4/96.4 (third-view/multi-view) vs
  OpenVLA-OFT 91.0/94.0; CALVIN ABC-D 4.08/4.35 vs 3.65/4.10.
  Latency 121.6 ms (1.67× base) vs multi-frame baselines 229.5 ms
  (3.15×).
- **Limitation (theirs)**: codec MVs are compression artifacts, not
  physics — noisy in highly dynamic scenes.

## What transfers to us

1. **#11's history-arm entry condition just got a cheap candidate
   representation.** The observation-aliasing page banked an
   aliasing census as the gate for any history/memory arm; if that
   census ever fires, codec motion vectors are the cheapest
   representation on the menu — no flow network, no frame stack,
   ~free at data-loading time (our episodes are stored as video).
   The decode-stage AdaLN injection point (not VLM-side) is the
   design datum to carry with it.
2. **Foresight-as-aux is the motion-space cousin of OneWM-VLA's
   latent forecast** — same slice, same shape: predict a compact
   future summary as auxiliary supervision. The pole is getting
   crowded, which raises confidence in the shape and lowers the
   novelty of any one recipe.

## What doesn't transfer

- **Gains are small** (+2.4 to +3.4 pp on saturated suites) against
  a 7B backbone; nothing here is arm-priced.
- **The Markov-blindness premise is unmeasured on our stack** — the
  aliasing census (#11) is exactly the measurement, and it hasn't
  fired; buying temporal context before measuring the ambiguity
  would be backwards.

## Which idea/arm it fed

**#11 (visual grounding)**: history-arm candidate representation
banked (codec MVs + decode-stage AdaLN), strictly behind the
aliasing-census gate; no arm. Cross-refs:
[observation aliasing](observation-aliasing.md) (the entry
condition), [OneWM-VLA](onewm-vla-one-token.md) (same
forecast-as-aux shape, same slice),
[VLA-JEPA](vla-jepa-latent-world-model.md) (the pole's
teacher-anchored end).
