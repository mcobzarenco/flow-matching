# QDepth-VLA: predict quantized depth tokens, keep the expert at inference

*Lit slice 2026-08-09 (work session 12:15Z). QDepth-VLA
([2510.14836](https://arxiv.org/abs/2510.14836), Li et al., CASIA) —
the last banked radar hook, read as the third corner of the
aux-spatial grid opened by [VEGA](vega-encoder-grounding.md) and
[Spatial Forcing](spatial-forcing.md). Fed #11 (aux-family ledger),
#17 (the aux-spatial pole gains a generative-expert recipe that fits
a single-tower trunk), #5 (another discretization-beats-regression
datapoint).*

## The paper in plain words

Same disease as VEGA and Spatial Forcing — VLM-based robot policies
know *what* they see but not *where* it is in 3D — but a different
cure. Instead of nudging the policy's internal features to *look
like* a 3D teacher's features (alignment), QDepth-VLA makes the
model *predict depth outright*: a pretrained VQ-VAE compresses each
depth map into a small grid of discrete tokens, and a dedicated
"depth expert" running beside the policy learns to predict those
tokens from the RGB stream. No depth camera is needed — the depth
maps themselves come from a monocular estimator (Video Depth
Anything) run over ordinary RGB training data. The twist that
separates it from its siblings: the depth expert is *not* thrown
away after training. Its predicted depth tokens sit in the
attention context, and the action head reads them while acting — a
geometry scratchpad the policy consults at deploy time, paid for
with extra inference compute.

## Contribution

The mechanism, precisely: an off-the-shelf monocular video-depth
model (ViDA, ViT-L) pseudo-labels depth for OXE/LIBERO frames; a
per-dataset VQ-VAE (codebook K=256, dim 160, 16×16 latent grid)
turns each map into 256 discrete tokens. A depth expert (18 layers,
hidden 1024 — a second π₀-style expert coordinated MoE-fashion with
the PaliGemma backbone) reads the SigLIP vision tokens *before
language fusion* and predicts the 256 depth codes by cross-entropy.
Total loss = flow-matching action loss + λ_t·CE, with λ_t decaying
exponentially from 0.01 — geometry pressure front-loaded, action
refinement takes over late. Hybrid attention mask: depth tokens see
image+text only; action tokens see everything *including the depth
tokens* — which is why the expert must run at inference.

## Experiments

- **LIBERO single-view** (open-π₀ base): 85.4 avg vs base 77.7 —
  +8.8 Spatial, +10.4 Goal, +6.6 Long. Beats CoT-VLA-7B (81.1) and
  3D-CAVLA (82.6) in the single-view setting.
- **LIBERO multi-view**: 94.9 — above DreamVLA (92.6), below
  3D-CAVLA (98.1). Their own framing: depth aux partially
  *compensates for missing views*.
- **SimplerEnv**: Google-robot avg 75.1 vs 71.4; WidowX avg 68.5 vs
  60.0, driven by the precision task (stack-block 39.6 vs 15.8,
  +23.8). One regression hidden in the average: open/close drawer
  58.0 vs 68.0.
- **Real 6-DoF Piper arm**: 42.5 vs 32.5 avg — but 4 tasks × 10
  trials, 50 demos; carry the sim numbers, not these.
- **Ablations on WidowX** (the valuable part, full model 68.5):
  remove the depth *expert* −8.5 (largest, −23.8 on stack-block);
  remove hybrid attention −5.5; swap quantized tokens for pixelwise
  depth *regression* −3.9; remove the depth *loss* only −2.9.

## What transfers to us, what doesn't

- **The aux-spatial grid now has a third corner, and it's the one
  that fits a single-tower trunk.** VEGA aligns at the encoder
  output (needs an encoder/LLM seam Molmo2 doesn't have); Spatial
  Forcing aligns LLM-interior features; QDepth adds *generative
  prediction by a parallel expert* — and an expert attached beside
  the trunk is exactly the attachment machinery we already run for
  actions (#4). If the vu5k readout ever opens the aux-spatial
  conversation for our stack, this recipe needs no architectural
  seam at all: vision tokens in, depth codes out.
- **The ablation table undercuts the paper's own story, and that's
  the load-bearing read.** Removing the depth supervision costs only
  −2.9 of the +8.5; the *expert with hybrid attention* — extra
  parameters and tokens the action head can attend into, trained by
  whatever gradient flows through — carries ~5.6 on its own. So
  most of the measured win is *architecture* (a scratchpad in the
  context), not *geometry* (the depth signal). Any pre-reg citing
  this paper as "depth supervision buys X" must quote the −2.9, not
  the −8.5. The honest depth-specific claims are the stack-block
  delta and quantized-vs-regression.
- **Quantized-beats-regression (+3.9) is a mean-collapse story we
  recognize.** Pixelwise L2 regression onto noisy monocular
  pseudo-depth averages away detail exactly the way our #19 reads
  show sampled-mean action pooling collapsing structure; CE over
  discrete codes keeps the prediction multimodal and — their
  motivation — absorbs teacher noise (ViDA pseudo-labels are
  temporally inconsistent). Same argument family as FAST/RVQ on the
  action side (#5): discretize the continuous signal, predict
  distributions over codes.
- **Zero-cost is traded away, and nobody measured the exchange
  rate.** VEGA and SF's selling point was aux structure at zero
  inference overhead; QDepth keeps an 18-layer expert plus 256
  extra context tokens in the deploy path and reports no latency
  numbers — and no head-to-head against VEGA/SF exists in either
  direction (their baselines are CoT-VLA, DreamVLA, 3D-CAVLA). The
  teacher×depth×recipe grid stays unresolved by direct evidence;
  escalation order for us stays VEGA-first on published wins,
  QDepth-style expert as the fit-our-architecture fallback.
- **Doesn't transfer:** absolute LIBERO/Simpler numbers (open-π₀
  base, different data regime); the real-robot deltas (4×10
  trials); the per-dataset VQ-VAE requirement is real preprocessing
  debt (a codebook per data source, plus a ViT-L labeling pass over
  the whole corpus) that any cost accounting must include.

## Where it lands

- **#11**: aux-family ledger — third recipe class (generative
  quantized-depth via parallel expert, monocular pseudo-labels, no
  sensors), evidence strongest on precision/spatial tasks, weakest
  as a *depth-supervision* claim (see the −2.9 vs −8.5 split).
- **#17**: the aux-spatial pole's menu is now
  {encoder-align, LLM-interior-align, expert-generative}; the third
  is the only one that needs no encoder seam — named fallback for
  single-tower Molmo2 if the family is ever pre-registered.
- **#5**: quantized-beats-regression +3.9 banked as a
  discretization datapoint on the *perception* side, echoing FAST's
  action-side argument.
- **#4 (context only)**: one more production sighting of the
  parallel-expert attachment pattern — MoE-coordinated second
  expert beside a frozen-ish trunk, same shape as our F/K screen's
  subject.
