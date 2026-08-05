# Trunk survey — open-weights VLM candidates for the next-generation trunk

*2026-08-05, work sessions ~19:10Z (framing; killed by the 19:08Z
usage-cap 429) + ~19:35–20:00Z (candidate deep-reads via six
parallel web readers + one follow-up, resumed after the credit
top-up). Owner mandate 17:50–18:01Z: deep
review of in-scope open-weights models as candidate trunks. Budget
**<7B total params, ideally ~3B**; **video-trained preferred**; method:
**read the arXiv paper (if any) + the HF `config.json` per candidate,
not just model cards**. This doubles as the literature slice (charter
§0 standing allocation). Everything below is post-cutoff-sensitive and
was researched on the live web this session; config numbers are quoted
from the fetched `config.json` files, not from model memory (charter
§6).*

## Why a trunk survey (and what "trunk" means here)

The stage-2 protocol (mainline §8.11, banked 6.62 panel @80k) separates
the **trunk** — a pretrained VLM that encodes (images, state, prompt)
into hidden states — from the **action expert** — a flow-matching head
reading a few intermediate layers (today: E2B global-attention prefix
layers {4, 9, 14}). Swapping the trunk under a fixed expert is the
cheapest structurally-different bet we can make, and the grounding
probes (ideas #11) say the visual stack is the current bottleneck. The
north star (idea #16) is few-shot transfer to the owner's rig, so
sample-efficiency of adaptation — not leaderboard position — is what a
trunk is for.

## Rubric

Ranked on, in order:

1. **Expected grounding/dynamics quality at ≤7B** — video/dynamics
   pretraining is the owner-preferred signal, because manipulation is a
   dynamics problem and the acuity probes point at the visual stack.
2. **Integration cost against our stack** — bijou is a pure-torch
   Gemma-4 reimplementation with bit-exact parity; a same-family swap
   (E2B→E4B) is nearly free, any new family costs a new implementation
   + parity harness (~the largest single cost on the table).
3. **Structural fit for the export-stream protocol** — layer count,
   hidden size, attention layout; `head_dim > 256` today falls off the
   fused-attention path (the deep-dive found even our global 512
   doesn't take flash), and exotic layer types (MoE, conv hybrids,
   encoder-free towers) change what "read layer k" means.
4. **License + checkpoint availability** — base (pre-IT) checkpoint
   with the vision tower shipping openly is strongly preferred (the
   base-vs-IT question is idea #10).

## The incumbent: Gemma 4 E2B (what a challenger must beat)

From `docs/gemma4.md` (code-derived, parity-verified): 2.3B effective
(5.1B with embeddings), text+image+audio, Apache 2.0. 35 layers, hidden
1536, MQA 8/1, head_dim 256 (global layers widen to 512 with p-RoPE),
sliding:full 4:1 @512 window, PLE, KV-sharing on the last 20 layers,
encoder-free 16×16-patch vision pipeline (16 bidirectional layers,
hidden 768). Bijou truncates it: prefix encode = layers <15, export
streams {4, 9, 14}. Panel anchors: AR-100k 5.8026 / sealed v2 5.6903;
flow-80k 6.6232 (heun-30) but first_mae 1.9331 beats AR's 2.1431.

The within-family upgrade (E4B: 42 layers, hidden 2560, 8/2 KV,
5:1 @512, no double-wide-MLP trick) is already implemented in
`bijou/gemma4/` — it is the zero-integration-cost challenger and the
control against which any cross-family swap must justify its
implementation tax.

## Candidate: Ministral 3 3B (owner-flagged 17:57Z)

*Sources: [HF instruct repo](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)
+ base-repo `config.json` (fetched), [Mistral 3 blog](https://mistral.ai/news/mistral-3/),
[arXiv:2601.08584](https://arxiv.org/abs/2601.08584) (Jan 2026).*

- **Size/license:** 3.4B LM + 0.4B vision ≈ 3.8B total, Apache 2.0 —
  dead-center in the owner's budget. Released 2025-12-02 alongside
  Mistral Large 3; trained by **cascade distillation** (iterative
  prune + distill from a larger parent), not from scratch.
- **Base checkpoint: YES** — `Ministral-3-3B-Base-2512` ships BF16
  *with* the full vision tower (config fetched and checked). The
  instruct repo is FP8-quantized (vision tower/projector/lm_head kept
  high-precision). Base-vs-IT (idea #10) is actually runnable here.
- **Config facts** (from `config.json`): text — 26 layers, hidden
  3072, GQA 32 heads / 8 KV, **head_dim 128**, MLP 9216, **full
  attention on every layer** (no sliding window), YaRN ×16 → 256k
  (16k native), vocab 131k, tied embeddings. Vision — Pixtral
  lineage: 24 layers, hidden 1024, patch 14, variable res up to
  1540px, 2D RoPE, 2×2 spatial merge into a bias-free GELU projector.
- **Video: NO.** Nothing in the blog, model card, or paper abstract
  mentions video training — images only. Misses the owner-preferred
  signal outright.
- **Structural fit:** clean — uniform full-attention decoder,
  head_dim 128 (on the fused-attention fast path, unlike our global
  512), no PLE/KV-sharing exotica; "read layer k" is unambiguous.
  26 layers vs E2B's 35 means the {4,9,14} export map would need
  re-tuning (relative depths ~15/35/55% → ~4/9/14 of 26 ≈ same
  indices, conveniently). Attention inner dim (4096) is wider than
  the residual stream (3072) — a wrinkle for hidden-state hooks but
  export streams read the residual stream, so inert.
- **Integration cost:** full new-family port (Mistral-3 decoder +
  Pixtral encoder + parity harness). `transformers` v5 only —
  the HF-reference side of a parity harness needs a version bump.



## Candidate: Qwen3-VL 2B / 4B (dense)

*Sources: [tech report arXiv:2511.21631](https://arxiv.org/abs/2511.21631)
(Nov 2025, read from the PDF — the HTML build is empty),
[4B config](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/config.json)
and 2B config (fetched), HF API enumeration of the Qwen org.*

- **Size/license:** 2B = **2.13B** total (28 layers, hidden 2048,
  GQA 16/8, head_dim 128, MLP 6144); 4B = **4.44B** total (36
  layers, hidden 2560, GQA 32/8, **decoupled head_dim 128** —
  attention width 4096 > residual 2560, MLP 9728). Both Apache 2.0,
  vocab 152k, tied embeddings, **native 256k ctx**.
- **Video: YES, the real thing.** Four-stage pretrain ≈2.2T tokens;
  S2 (~1T @32k seq) carries a "significantly larger volume of video";
  S3 (100B @262k seq) emphasizes long video, post-training includes
  2-hour videos. Timestamps are *textual* (`<3.0 seconds>` prefixes
  — T-RoPE dropped), positional encoding is **interleaved-MRoPE**
  (t/h/w interleaved across dims). Caveat: absolute video volume is
  never quantified — qualitative stage descriptions only.
- **Vision tower:** SigLIP2-Large ~300M (24 layers, hidden 1024,
  patch 16, **temporal_patch_size 2**, 2×2 merge → one LM token per
  32×32px per 2 frames), continued-trained at native/dynamic
  resolution (CoMP-style interpolation). Video-aware down to the
  patching.
- **Structural fit — two real wrinkles:** (1) **DeepStack**: multi-
  level ViT features (indexes {5,11,17}) are *added into the hidden
  states of the first three LM layers* — a reimplementation must
  reproduce this, and it sits below our export depths (streams
  {4,9,14}-equivalent unaffected, but "layer k hidden state" ≠ pure
  token stream for k<3). (2) Interleaved-MRoPE is a new positional
  scheme for the parity harness. On the plus side: head_dim 128
  everywhere = fused-attention fast path, uniform full-attention
  GQA decoder, no PLE/KV-sharing exotica.
- **Base checkpoint: NO.** Exhaustive org enumeration: only
  Instruct/Thinking (+quants) exist; `-Base` repos 404. Idea #10
  (base-vs-IT) is unrunnable in-family; the Thinking SKU is the only
  "different post-training" contrast available.
- **Integration cost:** full new-family port (Qwen3 decoder +
  SigLIP2 encoder + DeepStack + M-RoPE + parity harness) — the
  highest-quality-per-param challenger, at the full implementation
  tax.

## Candidate: InternVL3.5-4B

*Sources: [paper arXiv:2508.18265](https://arxiv.org/abs/2508.18265),
[HF repo](https://huggingface.co/OpenGVLab/InternVL3_5-4B) `config.json`
(fetched), HF API. Newest InternVL trunk family as of Aug 2026 (the
Mar-2026 InternVL-U is a separate unified-generation lineage, not a
trunk successor).*

- **Size/license:** **4.73B total** = InternViT-300M (24 layers,
  hidden 1024, patch 14, 448px tiles) + **Qwen3-4B** decoder (36
  layers, hidden 2560, GQA 32/8, head_dim 128, MLP 9728, untied
  embeddings, 40k ctx). Apache 2.0.
- **The headline: a real base checkpoint.** Four stages ship per
  size: `-Pretrained` (CPT only, pre-SFT), `-Instruct`, `-MPO`, and
  unsuffixed (full Cascade-RL). The `-Pretrained` SKU is exactly
  what idea #10 wants and what Qwen3-VL refuses to give us.
- **Video: present but under-documented.** Video benchmarks are
  quoted (MVBench/VideoMME/MLVU) but neither the 3.5 paper nor the
  InternVL3 paper enumerates video data volume or tokenization
  (InternVL3 CPT ≈ 200B tokens total, 1:3 text:multimodal). Video at
  inference = 256 tokens per 448² frame, no temporal patching.
  Weaker video-pretraining signal than Qwen3-VL, better than
  Ministral/Gemma.
- **Structural fit:** tiling is the wrinkle — dynamic 448² tiles
  (1–12 + thumbnail, worst case ~3.3k visual tokens/image) with
  pixel-shuffle ×0.5; no temporal patching; `select_layer: -1`
  (single-level vision feed — no DeepStack-style injection, so LM
  hidden states are clean token streams at every depth). Decoder is
  vanilla Qwen3: head_dim 128, uniform full attention.
- **Integration synergy:** the decoder is byte-for-byte the same
  config family as Qwen3-VL-4B's (both Qwen3-4B: 36/2560/9728,
  GQA 32:8). **One Qwen3 decoder port + parity harness unlocks both
  candidates**; they differ only in vision tower and injection
  scheme. That halves the marginal cost of whichever is tried
  second.

## Candidate: SmolVLM2 2.2B

*Sources: [paper arXiv:2504.05299](https://arxiv.org/abs/2504.05299),
[blog](https://huggingface.co/blog/smolvlm2),
[HF repo](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
`config.json` (fetched).*

- **Size/license:** ≈2.2B = SigLIP-SO400M (~400M, 27 layers, hidden
  1152 inferred, patch 14, 384px) + SmolLM2-1.7B (24 layers, hidden
  2048, **MHA 32/32 inferred — config omits the head fields**,
  head_dim 64, MLP 8192). Apache 2.0. **Context 8k** (config;
  paper's 16k claim doesn't match the shipped checkpoint).
- **Video: YES, explicitly** — the training mix is 33% video (LLaVA-
  video-178k, Vista-400k, MovieChat, FineVideo, …), frames rescaled
  to 384 (no tiling for video), pixel-shuffle ×9 → 81 tokens/frame.
  Video-MME 52.1 — best-in-class *for 2B at release* (Feb 2025), but
  that class has moved (Qwen3-VL-2B is a year newer).
- **Robotics pedigree (unique on this list):** LeRobot's **SmolVLA
  uses SmolVLM2-500M as its trunk** with a flow-matching action
  expert — the closest existing analogue to our stage-2 protocol in
  open source. The family is *proven* as a VLA trunk; note the
  flagship robotics use picked the 500M for latency, not the 2.2B.
- **Weaknesses:** older-generation 1.7B LM, MHA (no GQA), 8k ctx,
  no base checkpoint for v2 (v1 base exists but isn't video-trained),
  and **no successor** — no SmolVLM3 exists as of today; the family
  looks dormant since SmolLM3 (text-only).
- **Integration cost:** low-moderate — Llama-style decoder (the
  simplest port on this list) + SigLIP encoder + pixel shuffle.

## Candidate (different species): V-JEPA 2 / 2.1 — dynamics-pretrained encoder

*Sources: [paper arXiv:2506.09985](https://arxiv.org/abs/2506.09985),
[GitHub](https://github.com/facebookresearch/vjepa2), HF collection
configs (fetched), [V-JEPA 2.1 arXiv:2603.14482](https://arxiv.org/html/2603.14482v1)
(Mar 2026, post-cutoff).*

- **What it is:** encoder-only video JEPA — ViT-L/300M, ViT-H/600M,
  ViT-g/1B (patch 16, tubelet 2), pretrained on >1M hours of video
  (VideoMix22M). MIT/Apache (card vs repo disagree slightly; both
  permissive). **No language interface ships** — the paper's
  LLaVA-style alignment (Llama-3.1-8B) was never released.
- **V-JEPA 2-AC:** a 300M action-conditioned predictor post-trained
  on a frozen ViT-g with **<62 h of unlabeled DROID robot video** →
  zero-shot Franka pick-and-place (80%/65% cup/box) via latent-space
  planning. Proof that these latents carry manipulation-relevant
  dynamics with tiny robot-data budgets — precisely the few-shot
  transfer property the north star (idea #16) wants.
- **V-JEPA 2.1 (Mar 2026)** is the pick if we go this way: dense
  predictive loss + **deep self-supervision across intermediate
  layers** — i.e. mid-layer features are *trained* to be predictive,
  which is exactly what an export-stream action head reads. New size
  ladder ViT-B/80M → ViT-G/2B @384px. Checkpoints on Meta's file
  server (HF hosting still an open issue).
- **Role in our stack:** not a trunk swap — a **vision-stack
  replacement/augmentation arm** (idea #17 bullet 2). Concrete
  shape: keep the Gemma trunk for language+state, feed V-JEPA 2.1
  ViT-L features into the expert alongside (or instead of) the
  trunk's visual stream, and let the grounding probes (idea #11)
  arbitrate. Tests the dynamics-vs-image-language pretraining
  hypothesis directly, at 300M marginal params.

## Candidate: Molmo2-4B (the sweep's headline find)

*Sources: [paper arXiv:2601.10611](https://arxiv.org/abs/2601.10611)
(Jan 2026), [Ai2 blog](https://allenai.org/blog/molmo2),
[HF repo](https://huggingface.co/allenai/Molmo2-4B) `config.json`
(fetched), HF API. Released 2025-12-11 — post-cutoff; found by the
completeness sweep, not the seed list.*

- **Size/license:** **4.85B total** = SigLIP-so400m ~400M (27
  layers, hidden 1152, patch 14, 378px; card says "SigLIP 2" but
  metadata links the SigLIP-1 so400m repo — flagged) + **Qwen3-4B-
  Instruct-2507** decoder (36/2560, GQA 32:8, head_dim 128, MLP
  9728 — the *same decoder family* as InternVL3.5-4B and
  Qwen3-VL-4B). Weights **Apache 2.0**; but trained on third-party
  academic-use datasets, card states research-use intent — a real
  consideration for anything commercial, inert for our research use.
  Context 36,864 (trains at 16k — far short of Qwen3-VL's 256k,
  irrelevant for our ~2k-token frames).
- **Video: YES, with grounding.** Up to 128 frames @≤2 fps, patches
  pooled 3×3, interleaved with text + timing info, **bidirectional
  attention among vision tokens**; 9M+ new open examples (dense
  video captioning, video pointing, multi-object tracking with
  persistent IDs), collected without closed-VLM distillation, all
  datasets released. The robot-relevant part: spatio-temporal
  referring — the blog's own demo is *"how many times does the robot
  grasp the red block?"* answered with points + timestamps.
- **Benchmarks:** 15-benchmark average **62.8 vs Qwen3-VL-4B 58.1 vs
  InternVL3.5-4B 53.4** (card); video grounding wins are video-
  native (8B flagship: video pointing F1 38.4 vs 20.0, tracking J&F
  56.2 vs 41.1 — vs *Gemini 3 Pro*). Per-benchmark modality split of
  the average not published — flagged.
- **Structural fit:** vision feed is Molmo-style two-level
  (`vit_layers [-3, -9]`) through a pooling connector — no DeepStack
  injection into LM layers, so decoder hidden states stay clean
  token streams; decoder is vanilla Qwen3 (fused-attention-friendly
  head_dim 128, uniform full attention, QK-norm). The bidirectional
  vision-token attention is a prefix-mask detail, cheap to
  reimplement.
- **Base checkpoint: none found** (instruct-tuned SKUs only; the
  data is open but stage checkpoints aren't). Sibling
  **Molmo2-O-7B** (Olmo-3-7B backbone, 7.76B, fully open data
  lineage) scores 59.7 — the transparency option at 3 points and
  ~3B params extra.

## Screened at the sweep stage (checked, not deep-read)

A completeness sweep over the rest of the ≤7B open-weights landscape
(each entry live-checked on HF today; one-line verdicts):

- **Cosmos-Reason1-7B** (NVIDIA, 2025): 7.29B (Qwen2.5-VL-7B-based),
  video-trained @4 fps + SFT/RL post-training for *physical common
  sense and embodied reasoning* — the most VLA-aligned post-training
  on the list, at the exact top of the budget. NVIDIA Open Model
  License (commercial OK, guardrail clause). **Held as a
  second-round deep-read**: at 7.3B it's ~3.4× E2B's effective
  params, so it only enters if the E4B rung shows scale is what the
  panel wants. Successors (Reason2-8B, Cosmos 3) are over budget.
- **Hy-Embodied-VLM-1.0** (Tencent, **Jul 2026**): the closest
  training mix to the north star (action-centric embodied data,
  manipulation/navigation/spatial), Apache 2.0 — but **30B total /
  3B active MoE**; total params blow the budget ~4× and MoE breaks
  the export-stream story (per-token expert routing means "layer k
  features" are not a stable object). Watch the lineage (a MoT-2B
  predecessor exists); not a candidate today.
- **Kimi-VL-A3B** (Moonshot, 2025): video-trained, MIT, but 16B
  total / 2.8B active MoE — same two objections as Hy-Embodied.
- **LFM2-VL-3B** (Liquid, Nov 2025): right size, but image-only (no
  video), custom non-OSI license, and a hybrid conv+attention
  backbone that muddies "read layer k". Out.
- **Phi-4-multimodal 5.6B** (MIT): text+image+audio, **no video**;
  LoRA-adapter modality mixing; 2025-era image stack. Out.
- **Apple FastVLM 7B**: image-only, research-oriented `apple-amlr`
  license. Out (the FastViTHD token-efficiency trick is worth
  remembering separately).
- **AuroraEdge-V-2B** (Jan 2026): paper only, no weights on HF. Out.
- **PaliGemma 3 / newer Google small VLM: does not exist** as of
  today — Gemma 4 E-series *is* Google's current open frontier here;
  Gemini Robotics stays closed.
- **Being-H05-2B** (Jan 2026, Apache 2.0): a ~3B *VLA* (InternVL
  vision + Qwen LM + unified action space) — downstream of a trunk,
  not a trunk; relevant as prior art for the north star, not this
  list.

## Ranked verdict

The single most useful structural fact the survey turned up: **three
of the top candidates share one decoder.** Molmo2-4B, InternVL3.5-4B,
and Qwen3-VL-4B all sit on Qwen3-4B (36 layers / hidden 2560 /
GQA 32:8 / head_dim 128 / MLP 9728). One Qwen3 decoder port + parity
harness amortizes across all three; they differ only in vision tower
and injection scheme. That collapses what looked like three
independent integration taxes into one tax plus two toppings.

The queue (each rung enters via its own pre-reg; nothing here is
pre-registered yet):

1. **Gemma 4 E4B** — the mandatory first rung. Zero integration cost
   (already in `bijou/gemma4/`), same tokenizer/protocol, isolates
   pure trunk-scale effect (2.3B→4.5B effective). No video — so it
   also calibrates how much the video-pretraining axis matters when
   rung 2 lands. Run at the 40k screen rung after the box batch
   frees GPUs.
2. **Molmo2-4B** — the cross-family pick. Best measured quality in
   tier (62.8 avg, +4.7 over Qwen3-VL-4B), genuinely video-trained
   *with spatio-temporal grounding* (pointing/tracking — the
   nearest pretraining objective to "where is the gripper and what
   is it doing"), cleanest structural fit of the trio (no DeepStack,
   no tiling, plain Qwen3 hidden states), Apache weights. Costs:
   the Qwen3 port (shared), SigLIP-so400m encoder, no base ckpt,
   research-use data caveat (fine for us).
3. **InternVL3.5-4B** — the science sibling, nearly free once the
   Qwen3 port exists (InternViT-300M + tiling instead of SigLIP +
   pooling). Its unique asset is the **`-Pretrained` base
   checkpoint** — the only way to run idea #10 (base-vs-IT) on a
   modern 4B trunk. Weaker documented video mix; treat as the
   base-vs-IT vehicle rather than the quality bet.
4. **V-JEPA 2.1 ViT-L (300M)** — the structurally-different arm, not
   a trunk swap: bolt dynamics-pretrained features into the expert's
   visual stream beside the Gemma trunk and let the grounding probes
   (idea #11) arbitrate. Mid-layers are *trained* predictive in 2.1
   (deep self-supervision), matching the export-stream read; 2-AC's
   <62 h-of-robot-video → zero-shot Franka result is the strongest
   external evidence for the north-star thesis (few-shot transfer
   lives in dynamics pretraining). Cheapest params on the list.
5. **Qwen3-VL-4B (or 2B)** — held in reserve. Strongest native-res/
   long-video engineering, but most reimplementation surface
   (DeepStack injection into LM layers 0–2, interleaved-MRoPE), no
   base ckpt, and Molmo2 beats it on measured quality from the same
   decoder. Enters only if rungs 2–3 point at native-resolution or
   long-horizon context as the binding constraint.

**Screened out of the launch queue:** Ministral 3 3B (clean arch,
base ckpt, right size — but images-only and cascade-distilled;
dominated by the Qwen3 trio on rubric #1), SmolVLM2 2.2B (video-
trained and the only candidate with shipped VLA pedigree via
SmolVLA, but older-gen 1.7B MHA LM, 8k ctx, dormant family —
its pedigree is an argument for our *protocol*, not for the trunk),
Cosmos-Reason1-7B (second-round candidate iff E4B says scale is
what the panel wants), and the MoEs/others per the sweep section.

**What would change this ranking:** E4B failing to beat E2B at 40k
(then trunk scale isn't the lever and V-JEPA jumps the queue);
a Molmo2 base-stage release (removes InternVL's unique asset); a
Qwen3-VL successor with a base SKU.

## Method note

Per-candidate sources: arXiv paper (where one exists) + HF
`config.json` fetched this session + launch blog where no paper
exists. Where a claim below is load-bearing for a launch decision it
gets re-verified against the fetched config at pre-registration time —
this survey ranks the queue, it does not pre-register anything.
