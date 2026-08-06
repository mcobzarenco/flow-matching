# Molmo2-4B — what the candidate trunk actually is

**Why this doc exists.** Molmo2 was released 2025-12-11 — after the
assistant's training-data cutoff (January 2026 for the current model,
but the family is easy to confuse with Molmo-1 2024 priors: different
vision feed, different decoder, different crops/pooling). Per the
charter (§6, post-cutoff epistemics): primary sources beat priors.
Everything below was fetched 2026-08-06 from the
[HF repo](https://huggingface.co/allenai/Molmo2-4B) (`config.json`,
`preprocessor_config.json`, `chat_template.jinja`,
`modeling_molmo2.py`) and the
[paper arXiv:2601.10611](https://arxiv.org/abs/2601.10611) /
[Ai2 blog](https://allenai.org/blog/molmo2) as read for the
[trunk survey](../fontaine/blog/src/posts/2026-08-05-trunk-survey.md).
Facts sourced from the fast-model summaries of those fetches are
re-verified against the raw files before any implementation consumes
them (the port plan's rule R1). **When in doubt, the fetched files are
ground truth, not model memory.**

## The family (2025-12-11, Ai2)

- **Molmo2-4B** — SigLIP-so400m (~400M) + **Qwen3-4B-Instruct-2507**
  decoder; 4.85B total. The quality pick in the ≤7B tier: 15-benchmark
  avg 62.8 vs Qwen3-VL-4B 58.1 vs InternVL3.5-4B 53.4 (model card).
- **Molmo2-O-7B** — Olmo-3-7B decoder, 7.76B, fully open data lineage,
  59.7 avg. The transparency sibling.
- 8B flagship — the video-grounding headline SKU (video pointing F1
  38.4, tracking J&F 56.2 vs Gemini 3 Pro's 20.0/41.1).

Weights **Apache 2.0**; training mix includes third-party
academic-use datasets and the card states research-use intent —
inert for our research use, a real consideration for anything
commercial. **No base-stage checkpoints released** (instruct SKUs
only). Video-native: ≤128 frames @≤2 fps, 9M+ new open examples
incl. video pointing and multi-object tracking with persistent IDs —
spatio-temporal grounding is the pretraining objective nearest to
"where is the gripper and what is it doing".

## Molmo2-4B config facts (fetched `config.json`)

Decoder (**stock Qwen3-4B geometry**, shared with InternVL3.5-4B and
Qwen3-VL-4B — one port amortizes across all three):

| field | value |
|---|---|
| layers / hidden / MLP | 36 / 2560 / 9728 (SwiGLU, silu) |
| attention | GQA 32 q-heads : 8 kv-heads, head_dim 128, SDPA |
| QK-norm | `qwen3` style: per-head RMSNorm(head_dim) on q and k, **before RoPE** |
| RoPE | theta 5,000,000; max_position 36,864 (trains at 16k) |
| vocab | 151,936 + **128 additional** (separate `Molmo2Embedding` matrix for the extension, Molmo-1 convention) |
| embeddings | **untied** (`tie_word_embeddings: false`), **no input-embedding scaling** (no Gemma-style sqrt(hidden)) |
| norms | RMSNorm eps 1e-6; full attention every layer (no sliding windows, no softcapping, no PLE, no KV sharing) |
| dtype | fp32 shards (4×~4.85 GB ≈ 19.4 GB); bf16 ≈ 9.7 GB |

Special token ids (in the additional-vocab range): image_start
151936, image_end 151937, **image_patch 151938**, image_column
151939, low-res start 151940, frame start/end 151943/151944.

Vision tower (SigLIP-so400m class; card says "SigLIP 2", metadata
links the SigLIP-1 so400m repo — flagged in the survey, unresolved):

| field | value |
|---|---|
| layers / hidden / MLP | 27 / 1152 / 4304, gelu_pytorch_tanh, 16 heads |
| input | 378×378, patch 14 → 27×27 = 729 patches, LN eps 1e-6 |
| feature taps | `vit_layers [-3, -9]`, **concatenated** on feature dim → 2304 |

Connector (from `modeling_molmo2.py`): 2×2 **attention pooling**
(`image_pooling_2d`, multi-head dot-product attention with the mean
of each 2×2 patch group as query) → gated MLP (`ImageProjectorMLP`,
`w2(act(w1(x)) * w3(x))`, no biases) → text hidden 2560. Image
features are **added into the input-embedding sequence at
`image_patch` placeholder positions — single injection at layer 0**,
no DeepStack-style per-layer injection. Decoder hidden states are
therefore clean token streams — the property the export-stream /
residual-tap protocol needs.

Attention mask: vision tokens attend **bidirectionally within image
blocks** (`token_type_ids`-driven mask function: query and key both
image ⇒ unrestricted), text stays causal. A prefix-mask detail, same
family as Gemma 4 12B-Unified's `use_bidirectional_attention="vision"`.

## Processor facts (fetched `preprocessor_config.json`)

- Molmo-style multi-crop: **max_crops 8** at 378×378,
  `overlap_margins [4, 4]`, plus a low-res global view.
- **pooling_size [2, 2]** for images → ceil(27/2)² = **196 tokens per
  crop/view** before overlap trimming (the survey's "3×3 pooling" is
  the *video* path → 81 tokens/frame; images pool 2×2).
- Normalization mean/std 0.5 (not SigLIP's usual per-channel stats —
  take the file's word), resample 2 (bilinear).
- Token cost per image ≈ 196 (global view only) up to ≈ 196×9 ≈ 1.7k
  (8 crops + global) — vs our current Gemma soft-token budgets
  {70, 140, 280, 560, 1120}. Our 480p sources sit below 378² × 8
  crops native, so small crop counts are the honest operating point
  (same argument as the arch-batch img280 amendment).

## Chat template (fetched `chat_template.jinja`)

ChatML/Qwen convention: `<|im_start|>user … <|im_end|>`,
`<|im_start|>assistant`; images inserted as `<|image|>` placeholders
*before* the conversation text ("Image 1<|image|>Image 2<|image|>"
when multiple); videos as `<|video|>`; optional style markers and a
subtitle role (video captioning artifacts — unused by us). Contrast
Gemma's `<start_of_turn>` — prompt-assembly code cannot be shared.

## Loading

`trust_remote_code=True` required: `modeling_molmo2.py` (71.6 kB),
`configuration_molmo2.py`, `processing_molmo2.py`,
`image_processing_molmo2.py`, `video_processing_molmo2.py`.
Tokenizer is Qwen2-family BPE (`tokenizer.json` 11.5 MB). One
non-stock decoder detail to check at port time: separate rotary
embeddings for `rope_scaling_layers` (per-layer dynamic RoPE) —
confirm whether the 4B SKU uses it (config shows plain theta 5M).

## Port status

See the port plan post
(`fontaine/blog/src/posts/2026-08-06-molmo2-port-plan.md`) for the
implementation plan: Qwen3 decoder port shared across the trio,
SigLIP tower + connector, stream-export mapping, parity harness,
memory budget.
