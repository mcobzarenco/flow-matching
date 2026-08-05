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

Five models, each in base + instruction-tuned versions, all Apache 2.0
(lineage facts from the [HF launch blog](https://huggingface.co/blog/gemma4),
read 2026-08-05; repo status from `bijou/gemma4/`):

| Model | Params | Context | Modalities | Status in this repo |
|-------|--------|---------|------------|--------------------|
| **E2B** (`google/gemma-4-e2b-it`) | 2.3B effective (5.1B with embeddings) | 128k | text + image + audio | Implemented; **the bijou trunk** |
| **E4B** (`google/gemma-4-e4b-it`) | 4.5B effective (8B with embeddings) | 128k | text + image + audio | Implemented |
| **12B Unified** | 11.95B dense | 256k | text + image + audio, **encoder-free** | Not implemented (`use_bidirectional_attention="vision"` — bidirectional attention within image blocks) |
| **26B-A4B** | 26B total, **4B active** (MoE, 8 of 128 experts) | 256k | — | Not implemented (MoE blocks) |
| **31B** | 31B dense | 256k | — | Not implemented |

Per the blog: **only 26B-A4B is MoE** — the rest are dense. **PLE is in
E2B, E4B, *and* 12B Unified** (not just the E-series). **KV sharing
applies across the family.** Audio: E2B/E4B use USM-style conformer
encoders, trained only on *speech* QA (music and non-speech sounds were
not in the training data); the **12B Unified also takes audio** — it has
no separate vision or audio encoder at all, projecting raw image patches
and raw audio waveforms into the LM's embedding space through
lightweight linear layers, so the whole model fine-tunes in one pass.
Attention family-wide alternates local sliding-window and global
full-context layers — 512-token windows on the smaller dense models,
1024 on the larger — with dual RoPE (standard on sliding layers, pruned
on global) for long context. Capability reference points: MMLU-Pro 85.2
(31B) vs 60.0 (E2B); Arena Elo ~1452 (31B) with the 26B MoE at ~1441 on
just 4B active params.

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
