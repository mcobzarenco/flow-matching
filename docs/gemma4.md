# Gemma 4 — what the trunk actually is

**Why this doc exists.** Gemma 4 was released after the assistant's
training-data cutoff (January 2026). An assistant reasoning from priors
will silently substitute Gemma-2/3 facts and mis-describe the model
(e.g. calling the trunk "Gemma-3-class"). Owner directive 2026-08-05:
this doc is the antidote. Everything below is distilled from
`bijou/gemma4/` — a pure-torch reimplementation verified bit-exact
against the HF reference (`python -m bijou.gemma4.verify_parity`) — and
from the released checkpoints' `config.json`. **When in doubt, the code
is ground truth, not model memory.** Per the owner: Gemma 4 is
universally better than Gemma 3; do not treat Gemma-3 intuitions
(sizes, context behavior, quality ceilings) as applicable.

## The family

| Model | Status in this repo |
|-------|--------------------|
| **E2B** (`google/gemma-4-e2b-it`) | Implemented; **the bijou trunk** |
| **E4B** (`google/gemma-4-e4b-it`) | Implemented |
| 12B | Not implemented (`use_bidirectional_attention="vision"` — bidirectional attention within image blocks) |
| 26B-A4B | Not implemented (MoE blocks) |

The E-series ("E" ≈ effective/on-device) are multimodal: text + vision
towers are implemented in `bijou/gemma4/`; the checkpoints also carry
an audio tower, which is not. Architectures are fully config-driven —
`e2b_config()` / `e4b_config()` build them in code for from-scratch
use.

## E-series text decoder — the facts (from `config.py` / `text.py`)

Shared: vocab 262,144, head_dim 256, tied embeddings, no attention
bias, bf16, final logit softcapping **30.0** (it's back —
Gemma 3 had dropped it), sliding window **512**, RMSNorm eps 1e-6.

| | E2B | E4B |
|---|-----|-----|
| layers | 35 | 42 |
| hidden | 1536 | 2560 |
| MLP intermediate | 6144 | 10,240 |
| attention heads / KV heads | 8 / **1** (MQA) | 8 / 2 |
| sliding:full pattern | 4:1 (period 5) | 5:1 (period 6) |
| KV-shared layers (last N) | 20 | 18 |
| double-wide MLP on KV-shared layers | yes | no |

Structural features an older mental model won't predict:

- **PLE (Per-Layer Embeddings):** a packed auxiliary embedding table
  (own vocab 262,144, 256-dim per layer) plus a projection of the input
  embeddings feeds a small residual signal into *every* decoder layer.
- **Hybrid attention, two geometries:** sliding layers use plain RoPE
  (θ=10k) over head_dim 256; global (full) layers use **p-RoPE**
  (partial rotation, factor 0.25, θ=1M) over a **wider
  `global_head_dim` of 512**.
- **KV sharing:** the last `num_kv_shared_layers` layers have no K/V
  projections at all — they reuse the K/V states of the last non-shared
  layer of the same layer type. The final layer is always full
  attention.
- **Norms:** Q/K RMSNorm with scale, V RMSNorm *without* scale,
  attention scaling 1.0. RMSNorm applies the weight as **`x * w`** —
  unlike Gemma 2/3's `x * (1 + w)` (`bijou/nn.py:104`).

## E-series vision tower (from `vision.py`)

Encoder-free patch pipeline, *not* a SigLIP bolt-on: linear patch
embedder (16×16 raw-RGB patches) + learned 2D positions → 16
bidirectional transformer layers with 2D RoPE and clipped linears →
spatial average pooling to the soft-token budget (kernel 3) →
RMSNorm + projection into LM space. hidden 768, 12 heads.

## How bijou uses it

- Trunk = **E2B**, truncated: prefix encode runs layers < 15 (of 35);
  export streams for the action expert are the global-attention prefix
  layers **{4, 9, 14}** (see `docs/architecture.md` §1).
- `bijou/gemma4/` takes explicit `device`/`dtype` factory args so
  from-scratch parts (action expert) build directly on-device.
- Parity: greedy tokens match HF exactly; logits within bf16-ULP
  tolerance (on H100 with eager attention, bitwise today).

## Standing reminder

Memory entry `gemma4-post-cutoff` points here and is loaded every
wake-up. If a Gemma claim matters to a decision, check this doc or the
code — never answer from pre-cutoff memory.
