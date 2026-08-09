# HiFlow: autoregression over scales, not time — the tokenization-free middle ground

*Read 2026-08-09 (lit slice `lit-radar-0811`, priority 3: trunk /
decode family, #17). Paper:
[2603.27281](https://arxiv.org/abs/2603.27281) — "HiFlow:
tokenization-free scale-wise autoregressive policy learning via flow
matching" (cs.RO).*

**The paper in plain words.** There are two standard ways to make a
robot policy generate an action chunk: turn the actions into
discrete tokens and predict them one by one like words (our AR
trunk's family), or keep them continuous and denoise the whole chunk
at once (our flow expert's family). This paper stakes out a middle
position: keep actions continuous, but generate them
*coarse-to-fine* — first one number per dimension summarizing the
whole chunk's average motion, then two halves, then four quarters,
then the full resolution — with each level predicted
autoregressively from the levels above it, and a small flow-matching
network doing the continuous generation at every level. The
autoregression is over *resolution*, not over time, so the model
commits to the gist of the motion before it fills in the wiggles.
The pitch: you get AR-style factorized structure without a
quantizer's rounding errors, and their fine-grained tasks (threading
a needle) are where the win over token-based scale-AR shows up.

## What it contributes

- **The scale ladder**: chunk T=8 pooled into scales {1,2,4,8} by
  window-averaging; p(a⁽¹⁾)·p(a⁽²⁾|·)… factorized over scales with a
  scale-causal attention mask in a 12-block transformer (ScaleAR);
  a shared 6-block ActionFlowNet runs conditional flow matching per
  scale (25 Euler steps each → ~104 NFE total). ResNet vision +
  AdaLN task conditioning; no VLM anywhere.
- **Numbers**: MimicGen avg 88% vs CARP (VQ-token scale-AR) 85%;
  threading 90% vs 70% — the quantization-error story lands where
  precision matters. RoboTwin bimanual "place basket" 39% vs
  diffusion policy 18%; real HSR avg ~59% vs CARP ~44%.
- **Scale-count ablation**: 4 scales optimal (88%); 2 scales 84%,
  5 scales 85% — the ladder helps, but it saturates fast and
  over-decomposition hurts.

## What transfers to us

1. **A third pole for the #17 head-architecture axis.** Our ledger
   has discrete-token AR (FAST family, our AR trunk) and one-shot
   continuous flow (our expert). HiFlow demonstrates the
   *AR-over-scales, flow-per-scale* hybrid works and beats its
   quantized twin (CARP) exactly where quantization should hurt —
   fine precision. That is the cleanest controlled evidence yet on
   the tokenize-or-not question, because CARP holds the scale-AR
   structure fixed and only swaps discrete-vs-continuous.
2. **A conceptual cousin of FAFM, on a different axis.** Temporal
   window-pooling is a crude low-pass; the scale ladder is
   coarse-to-fine in *time* what FAFM's DCT-coefficient flow is in
   *frequency*. Both say: generate the slow structure first,
   condition the detail on it. If a future flow-head redesign ever
   opens, these two are the same bet in two coordinate systems —
   compare before picking.
3. **The saturation ablation is a useful prior**: most of the
   benefit is captured by 2–4 scales. Any hierarchy we ever bolt on
   should start minimal.

## What doesn't transfer

- **No trunk.** ResNet + task embedding; nothing about VLM
  conditioning, so it says nothing about how the ladder composes
  with a 4B trunk — the integration cost (a new head + ScaleAR
  stack) is entirely unpriced at our scale.
- **~104 NFE** against our 1-NFE deployment direction; they don't
  even report wall-clock. As with TCFM, decode-side cost discipline
  is absent from this literature.
- **+3% average over CARP at 1K–10K-demo scale** is a modest
  headline; the threading subscore carries the story. Fine for a
  family-map entry, nowhere near an arm.

## Which idea/arm it fed

**#17 (new trunks)**: head-architecture family-map entry — the
continuous-vs-quantized controlled comparison (vs CARP) is the
citable datum; no arm. Cross-refs:
[action tokenization](action-tokenization.md) (FAST/FASTer — the
pole this paper argues against),
[FAFM](frequency-aware-flow-matching.md) (same coarse-to-fine bet in
frequency space), [one-step menu](one-step-menu.md) (the NFE
discipline this family ignores).
