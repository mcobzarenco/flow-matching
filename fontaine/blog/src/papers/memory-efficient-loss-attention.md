# The loss and the mask: Cut Cross-Entropy + FlexAttention (2 papers)

*Lit slice 2026-08-08, same-session with the
[molmo2 perf review](../posts/2026-08-08-molmo2-perf-review.md) it
feeds — the review found our two "big tensor" costs are the
full-vocab CE chain and dense additive attention masks; both have
published, engineered answers. Read: abstracts + repo docs + the
PyTorch engineering posts; numbers quoted are the papers' own.*

## Cut Your Losses in Large-Vocabulary Language Models (CCE)

[arXiv 2411.09009](https://arxiv.org/abs/2411.09009), Apple, ICLR
2025 oral; [code](https://github.com/apple/ml-cross-entropy)
(`cut-cross-entropy` on PyPI).

**The problem it names** is precisely our finding 5: as vocabularies
grew (32k → 128k → 256k), the [B·T, V] logit matrix became the
single largest training allocation — for Gemma-2 2B the loss-side
tensors consume "an order of magnitude more memory than the rest of
the model combined."

**Contribution.** Never materialize the logits. Cross-entropy needs
only (a) the correct-token logit ⟨h_i, e_{y_i}⟩ — an indexed
matmul — and (b) log-sum-exp over the vocabulary, which a custom
Triton kernel computes blockwise in SRAM, streaming over vocabulary
tiles and never writing the [T, V] product to global memory. The
backward exploits softmax sparsity: vocabulary blocks whose softmax
mass is below bf16 numerical precision (the overwhelming majority
after a few training steps) contribute no representable gradient and
are skipped; a vocabulary-sort groups the surviving blocks so the
skip is block-granular, not element-granular.

**Experiments.** Gemma-2 2B: loss-computation memory 24 GB → 1 MB;
classifier-head training footprint 28 GB → 1 GB. Loss/convergence
parity demonstrated (training curves indistinguishable), throughput
parity with the fused baselines (the gradient filter is what buys
back the recomputation). Evaluated across the modern vocab range
(Llama-3/Phi/Gemma/Qwen-class heads) — the effect scales with V.

**What transfers to us.**

- The *shape* of the fix transfers exactly: our AR suffix loss
  assembles [B, W, 153,090] logits via `torch.cat` (base head + gap
  + FAST block, `ar_molmo2.py:238-244`) then fp32-upcasts all of it
  including 40–60% ignore rows (`ar_backbone.py:1124-1177`).
- **But the magnitude does not, at today's config** — honesty first:
  `--backward-chunks 6` on batch 12 means the per-forward CE tensor
  is [2, W≲80, 153k] ≈ 50 MB bf16, chunk-small by design. The review
  already ranked the S-effort valid-row-select fix first for this
  reason. CCE's real opening here is the *coupling*: backward
  chunking exists partly to bound step memory — a CCE-class loss
  would let us cut the chunk count (fewer chunked allreduces, less
  re-padding variety) or grow batch, which is where its memory win
  converts to throughput.
- **The complication:** CCE assumes one linear classifier. Our head
  is a *composite* (base `lm_head` + FAST block at an offset, with a
  dead gap) — a drop-in needs the lse composed across two segments
  (`logaddexp`, gap contributes −inf), which is exactly the
  "segmented CE" the review sketched. The pip kernel won't do this
  out of the box; a hand-rolled two-segment lse in T-slices is the
  M-effort middle rung, CCE proper is the L rung if we ever unify
  the head.

**What doesn't transfer.** The gradient-filtering throughput trick
assumes a trained-ish softmax (early-training softmax is flat —
their curves show it's fine, but our aux-text weak-label rows may
keep more mass spread); and none of this touches the flow expert,
whose loss has no vocabulary.

**Fed into:** `molmo2-perf-fix-prereg` (queue) — pass 1 keeps the
valid-row select; this page banks the escalation ladder
(valid-row → two-segment lse → CCE) with its entry condition: the
day we want backward-chunks < 6 or batch > 12/GPU on the AR trunk.

## FlexAttention: a programming model for optimized attention

[arXiv 2412.05496](https://arxiv.org/pdf/2412.05496) + the
[PyTorch blog series](https://pytorch.org/blog/flexattention-for-inference/);
shipped in stable torch (we run 2.11).

**Contribution.** Attention variants (causal-OR-block, padding,
prefix-LM, document masking — *our multimodal mask is literally
their example class*) expressed as a Python `score_mod`/`BlockMask`,
compiled via `torch.compile` into fused block-sparse Triton kernels
— so a mask is a *program*, not a materialized [B, 1, S, S] tensor.
Reported: >2.4× end-to-end training speedup replacing SDPA in
gpt-fast/torchtune at long context; native GQA (`is_gqa=True`);
block-sparsity skips fully-masked tiles. Counterpoint from the
issue tracker: at short sequences or dense masks it can *lose* to
cuDNN SDPA (mask-build overhead, Triton vs cuDNN kernel quality) —
it's a long-context, structured-sparsity tool.

**What transfers / what doesn't.** Our prefix S ≈ 1.1k is short and
our mask is dense-ish (causal OR image-block, no long masked runs),
and the prefix already rides the cuDNN flash kernel *with* the
additive mask — so FlexAttention buys us nothing today; the
measured cheap win remains re-admitting cuDNN on the suffix (review
finding 1). The entry condition that changes the answer:
torch.compile adoption (idea #2b) or much longer prefixes (video,
multi-frame history — the #14 direction), where the S² additive
mask and its rebuild-per-forward genuinely bind. Banked as the
named successor to the dense-mask design, not a current action.

**Fed into:** the perf-fix ladder's "later rungs" note + idea #2b's
compile file (FlexAttention is compile-native — the two land
together or not at all).

**Sources:** [CCE arXiv](https://arxiv.org/abs/2411.09009) ·
[apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy) ·
[FlexAttention paper](https://arxiv.org/pdf/2412.05496) ·
[PyTorch FlexAttention blog](https://pytorch.org/blog/flexattention-for-inference/) ·
[pytorch#138493](https://github.com/pytorch/pytorch/issues/138493) ·
[pytorch#141129](https://github.com/pytorch/pytorch/issues/141129)
