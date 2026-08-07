# Action tokenization — FAST and its learned successor

**Papers:** FAST
([2501.09747](https://arxiv.org/abs/2501.09747), Physical
Intelligence) and FASTer
([2512.04952](https://arxiv.org/abs/2512.04952), v2). FAST is local
canon (#15) — our AR baseline *is* the π0-FAST recipe — and FASTer
was banked from the 2026-08-06 lit radar as the natural rung after
the tokenizer-v3 quantile refit. Both re-read at full-text depth
for this page. **Fed:** #5 (tokenization-v3; the learned-VQ
follow-on rung now has concrete falsifiers) and #8/#12 (the decode-
latency picture).

## The theme

Discrete action tokens are what let a language-model trunk treat
control as next-token prediction — our whole AR arm rests on them.
FAST is the field's answer to *why naive discretization fails* and
the fix that made AR VLAs competitive; FASTer is the December-2025
bid to replace FAST's fixed transform with a learned one. Read
together they say something specific to us: the tokenizer's
*reconstruction* quality stopped being the frontier (ours
round-trips near-losslessly — owner measurement, 2026-08-05); the
live axes are **token distribution quality** and **decode
latency**.

## 1. FAST — why binning fails, and the DCT fix (2501.09747)

**The problem it solves.** Per-dimension binning — the OpenVLA/RT-2
default — collapses at high control frequency. Their account is
information-theoretic: as frequency rises, smooth action signals
change little per step, so each token's marginal information
approaches zero; highly correlated consecutive tokens make
next-token prediction a near-copy task and convergence crawls.
Their didactic study makes it vivid: at 800-timestep sampling a
binning policy "simply copies the first action." At 20–50 Hz on
real tasks, binning policies were "unable to make progress" at all.

**What it contributes.** A four-step invertible pipeline:
per-dimension quantile normalization (1st→99th percentile mapped to
[−1,1] — outlier-robust; exactly the spec our v3 refit targets),
DCT per action dimension over the chunk, scale-and-round
quantization (scale γ=10 trades lossiness against compression),
then BPE over the flattened low-frequency-first sequence (vocab
1024). Compression is strongest exactly where binning dies: 700
naive tokens → 53 for a 50 Hz bimanual shirt-fold chunk (13.2×);
rule of thumb ~30 tokens per second per arm. FAST+ is the same
pipeline BPE-trained on ~1M cross-embodiment chunks and "closely
matches" dataset-specific fits.

**What they ran.** Five real tasks up to bimanual 50 Hz laundry
folding, LIBERO, zero-shot DROID. Against a learned FSQ baseline,
FAST is "as good or at times better … despite requiring no separate
neural network training." The headline everyone quotes — π0-FAST
matches diffusion π0 with **up to 5× fewer GPU-hours** — holds
("5x fewer GPU hours" for their flagship comparison; convergence in
3× fewer steps on table bussing), and on zero-shot DROID π0-FAST
actually *beats* diffusion π0 at language following. The
counterweight they print themselves: **inference is ~7.5× slower**
— ~750 ms per 1-second chunk on a 4090 vs ~100 ms for the diffusion
expert, because 30–60 tokens decode sequentially through the full
2B trunk. Caveats: main results live in bar charts (no numeric
task tables), the 5× is a single-run "up to," and there is no
representation-quality measurement anywhere — the evidence that
token quality shapes the *backbone* (KI's ~95% vs ~85% with
FAST-CE vs binning-CE) comes from the KI paper, not this one.

**Transfers.** Confirmed as our recipe's foundation. Two working
numbers: our v3 refit should reproduce the quantile-normalization
spec exactly, and deployment budgeting should treat **AR decode
latency, not training cost, as the binding constraint** — which is
precisely the #12 SnapFlow result's other half (our 1-NFE flow
student sidesteps the 750 ms class of cost entirely).

## 2. FASTer — a learned VQ that internalizes the DCT (2512.04952)

**What it contributes.** Two separable things. (a) **FASTerVQ**:
the action chunk is reshaped into a 2-D time×dimension grid (their
"single-channel image" — not literal pixels), patchified, and passed
through a transformer autoencoder into a 3-level residual VQ
(codebook 4096). The training loss keeps a **DCT-domain L1 term**
alongside raw-action L1 — the learned tokenizer doesn't discard
FAST's frequency insight, it internalizes it. Fixed-length output:
21 tokens per single-arm chunk, 84 bimanual, 126 whole-body. (b)
**Block-wise AR decoding** in the VLA: fixed-length tokens make
block prediction possible — 21 tokens decode in 3 forward passes
instead of 21, plus a small action-expert head.

**What they ran.** LIBERO average **97.9%** vs their π0-FAST
reproduction's 94.2; Simpler-Bridge 87.9% vs 76.5; real XArm 97.5%,
whole-body R1Lite 81.0%. Latency (RTX 5090): 112 ms vs π0-FAST's
197–556 ms single-arm; 237 ms vs **1,100–3,000 ms** at 21-DoF
whole-body. Tokenizer analytics are the strongest part: FAST's BPE
on Bridge uses only 48% of its vocab with one token eating 9.6% of
all occurrences (normalized entropy 0.69); FASTerVQ hits 100%
utilization, max frequency 1.35%, entropy 0.91, and reconstructs
~95% of chunks within tolerance where FAST+ manages ~70%.

**The scope corrections.** Three things the abstract won't tell
you. The speed win is **vs AR-FAST, not vs diffusion** — diffusion
π0 is still faster than FASTer at whole-body (225 vs 237 ms). Their
own ablation attributes **2.2 points of the LIBERO headline to the
decoding scheme + action expert**, not the tokenizer (token-wise
95.5 → block-wise 96.7 → +expert 97.7) — and the expert head
*collapses to 23.6%* if not pretrained. And the cross-backbone
table shows the tokenizer gain **shrinks as the FAST baseline gets
stronger**: +17.3 points on their weak InternVL FAST fit, +1.3 on
well-tuned Paligemma. All π0-family numbers are author
reproductions; no limitations section; no training wall-clock.

**Transfers — and the cheap falsifiers.** Our FAST fit
round-trips near-losslessly, so FASTerVQ's headline reconstruction
advantage is not the operative axis for us. The +1.3-on-a-good-fit
row is the honest prior for what a learned-VQ arm would buy. Before
any such arm, two CPU-cheap measurements decide (banked into #5):
compute our v3 fit's **Table-8-style stats** — vocab utilization,
max token frequency, normalized unigram entropy — and compare
against FAST-on-Bridge's pathology (48% / 9.6% / 0.69). Near 0.9
entropy → no headroom, the arm dies before it's born; near the
pathology → a real rung. Separately, **block-wise decoding of
fixed-length tokens** is the latency mechanism worth copying
independent of the VQ — though for us the 1-NFE flow student (#12)
already occupies that deployment slot.
