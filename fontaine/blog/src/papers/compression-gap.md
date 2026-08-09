# The Compression Gap: a tidy bottleneck story, measured on models 1000× smaller than the claim

*Read 2026-08-09 (lit slice `lit-radar-0817`, priority 5: the
flow-over-AR mechanism hook). Paper:
[2604.03191](https://arxiv.org/abs/2604.03191) — "The Compression
Gap: Why Discrete Tokenization Limits Vision-Language-Action Model
Scaling" (Takuya Shiba, single author, 2026-04-03).
[Code](https://github.com/Shibattic/the-compression-gap).*

**The paper in plain words.** Robot policies turn camera images
into actions through two different kinds of "action head": some
output actions as continuous numbers (diffusion/flow heads), others
compress actions into a small vocabulary of discrete tokens first
(like words in a dictionary). This paper asks: if you give the
policy better eyes — swap a weak vision encoder for a strong one —
which head lets the improvement through? On one benchmark suite,
with small models: the continuous head converts the better encoder
into +21 to +26 points of success, while the discrete-codebook head
gains only +4 to +10. The proposed explanation is an information
bottleneck — a 1000-word action dictionary can only carry ~80 bits
per motion chunk, so extra visual detail has nowhere to go. It's a
clean story, but the evidence is thin: tiny non-VLA models, one
benchmark, single seeds, and the paper's own ablations wobble in
ways the bottleneck theory doesn't explain.

## What it contributes

- **A factorial claim**: encoder (ResNet-18 → SigLIP/SigLIP2/
  DINOv2) × action head (Diffusion Policy vs OAT's FSQ codebook)
  on LIBERO-10. Encoder upgrade through DP: +21.2 (size M) / +26.0
  (size L). Through the 80-bit OAT codebook: +3.6 / +10.4
  (Table 1).
- **A mechanism proposal, not a measurement**: the data-processing
  inequality — the discrete channel caps I(Z;T) at H_l·log2|V| ≈
  80 bits per 32-step chunk at OAT defaults, so once the codebook
  is the tightest bottleneck, encoder gains are "blocked at
  quantization." No mutual information, reconstruction error, or
  probing is ever measured; the evidence is endpoint success
  deltas whose pattern is consistent with the story.
- The one attempted causal test (grow the codebook, watch the
  encoder delta return) half-works: |V|=1920 → Δenc +15.2, but
  |V|=4375 → back down to +4.0, unexplained. Table 2 also has the
  codebook head getting *worse* under a better encoder (SigLIP2:
  44.2, below the ResNet-18 baseline 53.8) — also unexplained.

## The experiments it ran

Everything on LIBERO-10 (50 demos/task), with **small language-free
transformer policies — not VLAs**: 4-layer/256-dim and
6-layer/384-dim decoders, no LLM trunk, no language conditioning.
Strong encoders are frozen with pre-extracted cached features;
ResNet-18 is trained end-to-end (an asymmetry inside the
factorial). Metric: peak success across training, **single seed
per cell**, 500 rollouts. The often-omitted reversal: with the
weak encoder, **the discrete head wins by 17.4 points** (53.8 vs
36.4) — the paper's own reading is that structured tokenization
*compensates* for poor perception. The continuous head also
saturates in the low 60s even with the best encoders.

## What transfers to us

- **A weak directional prior, banked as "consistent-with," not
  "predicts":** in a good-trunk regime, continuous heads extract
  more from trunk improvements than tight discrete codebooks do.
  It rhymes with our #19 finding (flow draws gains ~9× AR's — the
  mean-collapse asymmetry), and it points at the vision-unfrozen
  adamc regime as where flow-vs-AR panel divergence would be most
  visible. Both of those connections are *our* inference — the
  paper contains no flow matching, no VLA, and never trains an
  improving encoder.
- **The bit-budget arithmetic is worth keeping** as a lens: our
  AR decoder's binned action vocabulary carries on the order of
  1,800 bits per chunk — ~22× OAT's 80 — so the paper's bound
  plausibly *never binds* for our AR head. If our panels show AR
  attenuation anyway, the mechanism would have to be something
  other than raw channel capacity.

## What doesn't transfer

- **The headline numbers, wholesale.** Tiny non-VLA policies, one
  benchmark, single seed (±5-pt noise plausible), peak-success
  metric, floor-effect confound (DP's big delta starts from a
  17-pt-lower baseline — more headroom), unmatched head parameter
  counts, and the frozen-vs-trained encoder asymmetry. The
  "compression gap" may be more about OAT's specific 32×7→8-token
  compression ratio than about discreteness.
- **Any direct vote on #4 (frozen vs joint)**: the strong encoders
  are always frozen. Joint training opens exactly the channel the
  setup excludes — the encoder co-adapting to the codebook's
  needs — which would *shrink* the gap. If adamc's unfrozen-vision
  AR run beats this paper's frozen-swap story, that's evidence for
  co-adaptation, not against the paper.

## Hook corrections

The banked one-liner ("encoder upgrades give >21-pt gains through
continuous action heads but attenuated through discrete codebooks —
mechanism-level flow-over-AR prediction that sharpens exactly when
the trunk improves") was oversold on every clause: the >21 pts is
tiny language-free Diffusion Policy on LIBERO-10 at a single seed;
the attenuation is one 80-bit FSQ codebook (bigger codebooks
largely un-attenuate, and AR binning has orders of magnitude more
bits); the "mechanism-level" evidence is arithmetic plus a
non-monotonic ablation; and the paper never trains an improving
trunk — the "sharpens as the trunk improves" extrapolation was
ours, and it now carries an explicit conditional: with a *weak*
encoder the discrete head won by 17 points.

## Which idea/arm it fed

#19 (`ar-sampled-draws`) — a weak external rhyme for the
mean-collapse asymmetry, filed with its conditionals (good-trunk
regime only; not flow matching; our AR head's bit budget likely
escapes the bound). Watch note for the adamc k4l2 panel readout:
if flow-vs-AR divergence appears under unfrozen vision, this paper
is a *consistent-with* citation, never a *predicted-by*. No gate
changes.
