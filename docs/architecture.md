# Bijou architecture

Bijou is a vision-language-action (VLA) model for SO-100/101 robot
arms, built by reusing one pretrained multimodal transformer in every
role it can serve. Two trunk backbones are supported behind one
encoder/decoder seam:

- **Gemma-4 E2B-IT** — vision tower + 35-layer text stack, 262k
  vocabulary, per-layer input embeddings, tied LM head, ~5.2B params.
  The original trunk; most of the measured history in §7.
- **Molmo2-4B** (`bijou/molmo2/`, fact sheet in `docs/molmo2.md`) —
  SigLIP-so400m tower + Qwen3-4B decoder, 4.85B params, untied
  embeddings/LM head. Adopted 2026-08 after a matched-topology screen
  (§7): at 2.5× fewer training steps it decisively beat the equivalent
  Gemma composition on the frozen evaluation set, and it is the trunk
  of the current training runs.

Each observation (a task instruction, a variable set of camera frames,
the arm's proprioceptive state) is rendered as one chat-templated user
turn and prefill-encoded once per control step; action generation then
follows one of two trained families:

- **Decoder-only (`ar_backbone` on Gemma, `Molmo2ARDecoder` on
  Molmo2) — the phase-1 recipe (§2.3, §7).** The full backbone simply
  continues its own prefill as a model turn: the trunk's real
  generation opener, then — under full-vocabulary cross-entropy
  through the frozen shipped LM head — one headerless VALUE line per
  field the prompt's `[generate|…]` conditioning requested (subgoal /
  holding / progress / event / per-camera visibility: weak labels from
  the LLM judging pipeline), then `[BOA]` and the action chunk as
  **FAST tokens** (BPE over quantized DCT coefficients of the 50-step
  chunk). On Gemma these occupy a ~1k-id reserved-unused block at the
  tail of the vocabulary (~11M new params: prompt-side state
  projection + the block's embedding/PLE rows); on Molmo2 — whose
  embeddings and LM head are untied and carry no spare block — the
  FAST block gets fresh untied embedding rows and head rows
  (`fast_embed`/`fast_head`), mean-initialized from the frozen shipped
  tables. Everything else is the backbone itself fine-tuning at low LR
  (bf16 autocast, fp32 masters). What speaks is COMMANDED by the
  prompt: the model learns p(value | observation, asked) — never "was
  this frame judged" — and inference requests exactly the fields it
  wants (`[generate|actions]` = the deployment fast path, straight to
  grammar-constrained greedy action decoding).

- **Flow expert (§2.1–2.2) — the continuous-action head, phase 2.** A
  separate fp32 flow-matching expert (shape configurable; the trained
  instances are ~404M at hidden 1024) denoises the action chunk
  (velocity at flow time τ, Heun integration, τ: 1 → 0), conditioned
  on the trunk in one of two ways: *cross-attention* over exported
  K/V streams (on Gemma, the trunk truncated at layer 14 — 2.55B —
  encodes the prompt; E-series KV sharing makes the deep half
  query-only, so the truncated prefix's exported K/V is
  bitwise-identical to a full forward), or *residual-stream adapters*
  (`--conditioning-streams residual`): learned RMSNorm + K/V
  projections over raw residual taps of a **frozen** phase-1-trained
  trunk, one stream per prefix layer. Both recipes have trained
  instances: the expert over the frozen decoder-only **Gemma** trunk
  is the best banked checkpoint overall (§7, decoded as a 10-draw
  ensemble), and the expert over the frozen decoder-only **Molmo2**
  trunk (residual adapters) trained cleanly with its readout in
  progress. A matched joint-fine-tuning arm was stopped at ~4× the
  frozen arm's step cost — frozen-trunk attachment is the working
  recipe. (An AR FAST decoder over the same expert blocks, `ar_fast`,
  was the family's token-based variant until its retirement — §2.2.)

The prompt side is shared and annotation-aware (§1, §2.4): every
camera's image block is tagged with its judge-voted semantic kind
(`[wrist camera|…]`, `[top camera|…]`, … `[unknown camera|…]`;
per-camera dropout to "unknown"), the instruction appears both before
and after the content (the extended "sandwich": causal attention then
yields instruction-conditioned image/conditioning K/V and
content-conditioned instruction K/V), instructions are stochastically
swapped for judge-suggested rewrites, and the sandwich wraps a
`[key|value]` conditioning block — `[subgoal|…]` (an operator/planner
hint, heavily dropped out so the planner-less context stays the
well-trained default), hindsight
`[outcome|success∣partial∣failure][smoothness|high∣medium∣low]` from
the episode verdict, and the always-present `[generate|…]` request —
so inference asks for the behavior AND the outputs it wants instead
of the corpus marginal. The user turn ends with one soft **state
token** (the normalized proprioceptive vector through a
zero-initialized projection, injected like an image soft token) —
state rides the prefix, where stage 2 and the multi-turn direction
want it.

Data is a curated union of ~1000 community LeRobot datasets recorded
on miscalibrated hobbyist rigs, plus the owner's own SO-101 rig
datasets; per-dataset normalization (§4) removes the between-rig
calibration offsets images cannot see, and the LLM judge pipeline
(docs/episode-annotations.md) supplies the episode verdicts,
relabeled instructions, camera kinds, subgoal segments and sparse
frame labels that the paragraphs above consume. The deployment
target is the owner's physical SO-101 (§6).

This document is the deep reference for the model and training system
as they exist, the measured results that shaped them (§7), and the
directions under evaluation (§8). Per-module contracts live in
docstrings; code conventions in `code-styleguide.md`;
collaboration/operating conventions in `working-together.md`.
Transient state (in-flight runs, machine inventory, the queue) lives
in wandb, the HF hub, and `reports/` — not in docs.

Package layout (strict downward-only imports):
`train`/`eval`/`rollout`/`judge` → `loading` → `model` →
`encoders`/`decoders` → `interface` → `gemma4`/`molmo2` (`data` beside `model`,
imported by `loading`; `judge` touches only `data`; `aux_text` and
`annotations` are leaves beside `gemma4` — `annotations` is the
judge-annotation artifact CONTRACT, the shapes both the judge writer
and the training readers share; production concerns — the prompt and
its hash — stay in `judge`), with
`interface.py` as the encoder×decoder seam
and `model.py` the composition root: `BijouModel` owns the backbone
ONCE and composes a prompt-side encoder strategy (which receives the
backbone as an argument) with an action decoder — one network serves
several roles (prefix encoder for the cross-attention decoders; prefix +
suffix runner for the decoder-only path). The root also owns the
objective dispatch (`BijouModel.loss` / `loss_components`) and the named
trainable-group routing (`param_groups`: decoder / backbone_text /
backbone_vision).
Naming: **backbone** is the one identifier for the pretrained trunk
network (Gemma-4 or Molmo2) — the artifact (`--backbone`,
`BackboneConfig.id`, `backbone.safetensors`) and the mounted module
(`model.backbone`) alike; "trunk" survives only as informal prose.

The diagram below shows the Gemma composition; Molmo2 is analogous
with a full-depth prefix (no truncation point exists — Qwen3 layers
all carry K/V) and the decoder-only suffix running against the
retained prefix KV cache.

```
{task}[{kind_i} camera|<imgs_i>]..[{cond}|{v}][generate|{fields} actions]{task}⟨state⟩
  one chat-templated user turn — kinds/conditioning/request fill per
  sample from the judge annotations (§1); ⟨state⟩ = soft token
      │  E2B prefix: layers 0..14 (bf16), LEFT-padded batches
      ▼
  prefix K/V — ObservationMemory, encoded once per observation
      │
      ├─ cross-attention family: exported GLOBAL streams {4, 9, 14}
      │    FlowDecoder (404M fp32), 16 layers, each =
      │      cross-attn(one stream) → self-attn([state][a_1..a_50]) → MLP
      │      → velocity at flow time τ → Heun integration τ: 1 → 0
      │
      └─ decoder-only (ar_backbone): the FULL E2B continues the suffix
           [<|turn>model\n][value\n per requested field][BOA][t_1..t_k]
           (fields: subgoal/holding/progress/event/visible — judge
           labels, headerless — request order pins line meaning)
           through all 35 layers against the retained prefix cache;
           ~11M new params; decode scaffolds the request: value lines
           under per-field budgets, forced BOA, FAST-grammar chunk
```

## 0. Tensor dimension notation

Canonical identifiers for tensor axes, used in every `nn.Module` method
docstring in the ML code (single letters where standard, names
otherwise). A docstring lists each tensor argument as
`- name: [axis, axis, ...]`.

Batch & sequence:
- `B` — batch size
- `S` — query sequence length: the tokens a module processes this call
  (backbone prompt tokens; the expert's `suffix`)
- `T` — total key/value length in self-attention: `seen + S` with a KV
  cache, `= S` at prefill / no cache
- `P` — memory width: the encoded-observation tokens the decoder
  cross-attends (the `ObservationMemory` width; for the Gemma trunk this
  is the prompt length)
- `suffix` — decoder-side token count: the expert's `[state][a_1..a_chunk]`
  (`= 1 + chunk`); variable-width for ar_backbone
  (`[state][opener][aux?][BOA][tokens]`)
- `chunk` — action-chunk length in timesteps (`chunk_size`, default 50)
- `images` — camera images in a batch, summed over samples (Σ per-sample
  cameras — NOT `B`: samples contribute variable camera counts and the
  processor flattens them; verified: a 2-cam + 1-cam pair collates to a
  leading dim of 3). Inside the vision tower each image is one "batch"
  row, so this is also the tower's leading axis.
- `patches` — image patch tokens per padded grid (pre-pool)
- `soft_tokens` — pooled vision soft tokens (valid tokens, flattened
  across the batch as the tower returns them)

Feature & head axes:
- `hidden` — model hidden size (Gemma trunk 1536; Molmo2 trunk 2560; expert default 768, trained instances 1024/1536)
- `head_dim` — per-attention-head dimension
- `heads` — query attention heads; `kv_heads` — key/value heads
  (`kv_heads ≤ heads` under GQA)
- `intermediate` — MLP intermediate (GLU) size
- `action_dim` / `state_dim` — action / state dimensionality (6/6 here)
- `time_embed_dim` — sinusoidal time-embedding dimension
- `vocab` — token vocabulary size
- `num_layers` — decoder layer count (also the PLE per-layer axis on Gemma; Molmo2 has no per-layer embeddings)
- `ple_dim` — per-layer-embedding dimension (`hidden_size_per_layer_input`)

Inline literals where an axis is a fixed small constant: `2` = the (x, y)
spatial pair in `image_position_ids`; `head_dim/2` = RoPE inverse-freq
length; `3·patch_size²` = a raw-RGB vision patch row.

## 1. Prompt side — the prefix encode

This section describes the Gemma-4 E2B prompt path in detail; the
prompt *semantics* (camera kinds, conditioning block, request set,
soft state token) are shared by both trunks. The Molmo2 encoder
strategy (`bijou/encoders/molmo2.py`) differs where the architecture
forces it: Molmo2's own chat template and image processor (SigLIP
tower, pooled patches, `--max-crops`), a full-depth prefill (no
truncation point exists), and — for the flow expert —
residual-stream taps with learned adapters instead of exported K/V
(§2.1); trunk fact sheet in `docs/molmo2.md`.

**Molmo2 prompt format.** Molmo2 follows the ChatML/Qwen convention,
which hoists every image to the front of the sequence — Gemma's
inline "instruction sandwich" cannot be rendered, so the same
semantic content is re-mechanized (`MOLMO2_PROMPT_FORMAT`, a
namespaced format, not a version bump of Gemma's):

```
<bos> {hoisted images} <|im_start|>user\n
{task}[kind1 camera|Image 1][kind2 camera|Image 2]{condition}{task}
<state> <|im_end|>\n
```

Images expand per the shipped chat template's exact bytes (`Image
1<img>Image 2<img>` labels when there are several); camera *kinds*
bind to images through `[kind camera|Image i]` bracket groups in the
text, replacing Gemma's inline camera tags. Two checkpoint quirks are
load-bearing: the sequence-initial `<bos>` is actually the
`<|im_end|>` token (id 151645 — the checkpoint's own convention, per
its `tokenizer_config.json`), and tokenization is native
(`tokenizer.json` through the `tokenizers` backend, no remote code);
assembling text segments around pinned special-token ids is exactly
equivalent to tokenizing the full templated string because the
specials are added tokens that always split first (golden-fixture
test against the reference processor). The soft state token splices
just inside the user-turn close, and the collator LEFT-pads so an AR
suffix can continue directly from the prompt.

**One backbone, mounted at a depth (Gemma).** `BackboneConfig.depth ∈ {prefix,
full}`: the cross-attention decoders mount layers 0–14 only
(**truncated**); `ar_backbone` mounts the full 35-layer stack (its
suffix runs the KV-shared deep half). Either way the prompt encode
itself runs layers 0–14 and produces the same prefix K/V.

**Why truncation is exact.** Gemma-4 E-series layers ≥15 (E2B, of 35)
carry no K/V weights — the deep half runs query-only against the K/V
produced at layer 13/14. Bijou truncates exactly at that boundary
(`first_kv_shared_layer_idx = 35 − num_kv_shared_layers(20) = 15`), so it
keeps layers 0–14 (~2.1–2.55B params vs 5.2B) and the exported streams
are **bitwise-identical to a full forward**. The expert is then, in
effect, "more KV-shared layers" grafted on with the backbone's own
geometry — and the same argument makes the ar_backbone prefill exact:
no loss is taken on prompt positions and nothing consumes prompt hidden
states above layer 14, so prompts NEVER need the deep half (the
dead-half argument is per-token). `bijou/gemma4/` is a pure-torch
reimplementation, bit-exact vs HF on greedy text+image generation
(`verify_parity.py`).

**Exported streams = global-attention prefix layers {4, 9, 14}.** Gemma-4
uses a hybrid 5-period schedule (every 5th layer is FULL/global, the rest
sliding-window-512). Only the global layers are exported because they
have uniform 512 `head_dim`, p-RoPE trained for arbitrary range, and are
never window-truncated — the expert can adopt their geometry exactly.
The observation encode runs the decoder with `kv_stop_layer = max(streams) = 14`:
it caches that layer's K/V and skips its attention/MLP and all deeper
layers (dead compute for a K/V export; ~1/15 of decoder FLOPs saved, more
if the schedule stops lower). `KVCache.update()` is functional (no
in-place writes), so this same path is autograd-transparent when the
trunk is trained (§8.1).

**Prompt = extended instruction sandwich, pipe-unified brackets, soft
state token** (prompt format 3, `prompt.format` + `prompt.state_dim`
in the checkpoint; formats < 3 — colon conditioning, no `[generate|…]`,
no state token — are REFUSED at load: their prompt-side parameter set
does not exist and no artifact predating 2026-08-03 is preserved). One
chat-templated user turn, **LEFT-padded** across a batch with
per-sample LOGICAL position ids (cumsum of the real-token mask). The
structure (verified byte-for-byte against the real E2B processor,
2026-08-03) — every `{…}` slot fills PER SAMPLE from the data:

    <bos><|turn>user\n{task}[{kind_1} camera|<imgs_1>]..[{kind_k} camera|<imgs_k>][subgoal|{v}]?[outcome|{v}]?[smoothness|{v}]?[generate|{fields}]{task}⟨state⟩<turn|>\n

`{kind_i}` is camera i's judged semantic kind, resolved per dataset
from its `meta/camera_kinds.json` (vocabulary wrist|top|front|side|
unknown — dataset-specific ASSIGNMENT over a fixed global vocabulary,
never dataset names or free text); the conditioning brackets are the
sample's value conditioning (§2.4) — every part `[key|value]`;
`[generate|…]` is ALWAYS present (`actions` terminal). A rig-v2 sample
renders e.g.:

    …{task}[top camera|<imgs>][wrist camera|<imgs>][subgoal|reach toward the boat][outcome|success][generate|holding progress actions]{task}⟨state⟩<turn|>\n

(its recorded `front` camera key is judged kind `top`).

Each camera's soft tokens live INSIDE its kind-tagged bracket group —
bracketed on purpose: the Gemma chat template TRIMS every text part's
edge whitespace, so groups must self-delimit (~4.5 tokens/camera).
Tokenizer discipline: `[`/`]`/`|` are single ids that never
cross-merge into letters/digits/`\n` (measured 2026-08-03); adjacent
`][` seams merge (id 2585) but the prompt is always jointly encoded —
merges are deterministic compression, never a train/decode split
hazard (those live suffix-side, §2.4); angle brackets are BANNED
inside our bracket syntax (`|<`/`>]` merge). Kinds come from the
judge's per-dataset `meta/camera_kinds.json` (majority vote, verified
against the dataset's own annotation stamp) and travel WITH the items
(`item["camera_kinds"]`, attached by StatsAttachedDataset exactly like
the stats); a missing file/camera/tie renders `unknown`, and
`--camera-kind-dropout` (default 0.1) replaces resolved kinds with
`unknown` per camera per visit at train time so unjudged rigs stay
in-distribution at inference (probes render true kinds). Camera slot
ORDER is **(semantic kind, short name)** — `camera_prompt_order`, the
single source shared with the positional `visible` indices (§2.4):
same-kind rigs order identically whatever their private key names,
and the sort uses RAW kinds so a kind-dropout draw retags but can
never reorder images; untagged datasets degenerate to plain
short-name order, and multiple unknown-kind cameras keep a stable,
key-derived sub-order.

The **soft state token** ends the user turn: the per-sample-normalized
proprioceptive vector through `GemmaEncoder.state_proj`
(zero-initialized — inert at start, gradients via its K/V use; the
prompt-side "new parameter", `prompt.safetensors` in checkpoints,
decoder-LR group). Mechanically the collator splices one
attention-masked-in pad id just inside the user-turn close (fixed
offset from the sequence end under left padding) and the encoder
overwrites that position's embedding — the image-soft-token precedent;
the templated tail is verified at collator construction against a
probe conversation, so a template change breaks loudly, never shifts
the slot silently. State-in-prompt puts proprioception in the prefix
K/V — exactly where stage 2 (frozen-backbone flow expert, §8.11) and
the multi-turn direction (§8.12) need it; the flow decoder ALSO keeps
its own suffix state token (redundant conditioning, harmless, a
stronger direct path under frozen backbones).

Left padding is
load-bearing (decided 2026-08-01, test-gated in
`tests/test_backbone_continuation.py`): Gemma's sliding-window masks are
physical-index, so right padding puts a suffix appended after the batch
max at DIFFERENT physical distances per sample and silently corrupts
windowed attention for any suffix continuation; with left padding every
sample's suffix is physically adjacent to its real prompt. Correct for
the cross-attention consumers too (they read positions from the mask).
Under causal attention the extended sandwich (conditioning INSIDE the
task copies) yields instruction-conditioned image/conditioning K/V
*and* content-conditioned instruction K/V for a few extra tokens, and
each tag opens its image group so the image K/V is tag-conditioned.
Camera NAMES remain positional slots (community image/image2
keys carry no reliable wrist-vs-scene semantics — SmolVLA precedent);
the kind TAG is the first reliable viewpoint signal.

**What the decoders consume.** The cross-attention family reads the
exported streams of an `ObservationMemory`; `ar_backbone` additionally
retains the FULL prefix `KVCache` on the memory (`retain_cache=True`,
set by `BijouModel.encode` from the decoder kind — the exported streams
are zero-copy views into it) and extends it in place while decoding.

**Vision geometry** (encoder-free E-series tower, 768 hidden, 16-px
patches, 3×3 spatial pool): a 640×480 frame → resized 624×480 → 39×30
patches → **130 soft tokens** (13×10, one per 48×48-px cell), under the
140-token/camera budget. Prompt ≈ 292 tokens for 2 cameras tag-less
(+~3.5/camera for kind tags, +~5–25 for the conditioning block); padded
batches reach ~452. The acuity probe (§8) found position is *sharpest at
the tower output* (8.4 px linear readout) and degrades through the LM
layers — the pool is not the bottleneck; the text stack's handling of
visual tokens is.

## 2. Action decoders

Four decoder classes share the seam — `FlowDecoder`,
`ARBackboneDecoder`, `Molmo2ARDecoder`, `MolmoFlowDecoder` — behind
two trainable CLI kinds (`--decoder flow | ar_backbone`; `molmo_flow`
is inherit-only via `--init-from`/`--resume`, §8.13): `ar_backbone`
dispatches to the trunk-matching decoder-only class
(`ARBackboneDecoder` on a Gemma backbone, `Molmo2ARDecoder` on
Molmo2). A checkpoint's `decoder.kind` tags which one it carries;
retired kinds (`ar_fast`, §2.2) are refused by name with the git tag
that still loads them.

### 2.1 Flow-matching expert (cross-attention)

A narrow decoder over the suffix `[state][a_1..a_50]` (`suffix_length =
1 + chunk_size`). CLI default shape: **hidden 768, 6 self-attn heads,
intermediate 3072 (GLU), 4 cross-attn heads, 15 layers**
(`--decoder-{hidden,heads,intermediate,cross-heads}`,
`--stream-counts 4 4 7`); the trained mainline instances override to
**hidden 1024, 8 heads, intermediate 4096, 8 cross-heads, 16 layers,
~404M fp32** with a 4-4-8 schedule. Freshly initialized, never loaded
from the backbone.

Each `SuffixBlock` (`bijou/decoders/blocks.py`, flow-private since
ar_fast's retirement) is
a Gemma-style sandwich of three sublayers, each
`residual → pre_RMSNorm → sublayer → post_RMSNorm → +residual`:

- **Cross-attention** over one exported stream. Queries adopt the
  backbone's global geometry exactly: `head_dim 512`, q-RMSNorm, p-RoPE
  continuing at positions after each sample's REAL (unpadded) prefix
  length, scaling 1.0. GQA against the stream's single K/V head. The
  per-layer stream assignment is `cross_attention_schedule` — CLI
  default **4-4-7**, trained mainline **4-4-8** (4 layers on stream 4,
  4 on 9, the rest on 14 — deepest-heavy, since layer 14 is the one
  the backbone's own deep half consumes). Its length is the expert
  depth; cycle/hybrid schedules are config diffs.
- **Self-attention** over the suffix. `SelfAttentionMode`:
  CAUSAL_ACTIONS (default — state visible to/from all; actions attend
  only earlier actions) or BIDIRECTIONAL. State is a token at position 0.
- **MLP** — gated GLU, `gelu_pytorch_tanh`.

Padding-position note: cross-attention query positions use each sample's
real memory width (`padding_mask.sum(1)`), not the padded batch width —
otherwise a sample's prediction depends on batch-mates' prompt lengths
(measured max|Δ| 0.55 before the fix; batch-1 rollout unaffected).

**State placement (π0 layout).** State enters the expert, not the VLM
prompt: the frozen backbone only ever sees in-distribution (image+text)
inputs, and slow visual context is decoupled from fast proprioception.

**Time conditioning (default: input-additive).** τ → sinusoidal
(geometric periods 4e-3..4, π0's unit-interval choice) → MLP → added to
the action-token embeddings at the input; the state token gets no time;
layers are unconditioned. An alternative per-layer adaRMS scheme is
implemented and the trained flow mainline uses it (§8.2).

**Conditioning is the exported-K/V path above, only.** A
residual-tap alternative (learned adapters over raw trunk hidden
states — the Molmo2 FlowDecoder attachment arm, with its
`--seam-stop-grad`/`--joint-ce` live-trunk seam flags) was removed
2026-08-13 at tag `pre-decoder-simplify`, superseded by `molmo_flow`
(§8.13) as the flow-on-Molmo2 story; its measured outcomes survive in
§7/§8.11 (the joint arm cost ~4× the frozen arm per step —
frozen-trunk attachment is the working recipe). Checkpoints recording
`residual_exports` are refused by name and load at the tag.

**One-step distillation** (`--distill snapflow`). SnapFlow-style
self-distillation trains the expert to jump straight from noise to the
endpoint (1 network evaluation instead of a Heun integration):
stop-gradient two-step-Euler shortcut targets mixed with the standard
flow loss (α=0.5, λ=0.1, no EMA teacher). The distilled student holds
within ~0.2 chunk MAE of the teacher's best ensemble at ~30× less
decode compute (§7 curated-plan ledger).

Params live ~50% in the MLPs, ~33% in cross-attention (8 heads × 512
over the residual 1024), ~17% in self-attention.

### 2.2a MolmoAct2 discrete head (`ar_molmoact2`, suffix format 6)

The MolmoAct2 family's second pathway (§8.13 step-8 closure):
`MolmoAct2ARDecoder` continues the trunk's own action rows
(`<action_start>` + bins + `<action_end>`, empty opener — their
serving prompt carries the whole scaffold) against the retained
prefix cache; grammar-masked decode by symbol-budget arithmetic over
the released FAST codec (1005 reachable of 2048 block rows, 7
quantization holes loud-by-default, reference-verbatim short
tokenization opt-in for training). Zero decoder parameters — the
trainable surface is the trunk (`--backbone-text-lr`); checkpoints
record the format-6 `ar_backbone` section (no expert/prompt weight
files). Joint runs mount it as the parameterless `joint_ce` rider
beside `molmo_flow` (L_flow + λ·L_CE, `--insulate-expert` for KI).

### 2.2 AR FAST decoder (cross-attention) — RETIRED 2026-08-13

`ARFastDecoder` (the same sandwich blocks over a causal FAST-token
suffix, grammar-constrained greedy decode) was removed at tag
`pre-decoder-simplify` after being superseded by `ar_backbone` on
quality (5.656 vs 5.96 community holdout), parameter count (~11M
vocabulary patch vs a 404M expert) and deployment. Its grammar mask
and `IGNORE_INDEX` convention live on in §2.3 (`ar_backbone.py` owns
them now); its checkpoints load at the tag, and its results stay in
the §7 Gemma-era ledger. History and the tokenizer artifact story:
§8.3.

### 2.3 Decoder-only path (`ar_backbone`)

`ARBackboneDecoder` (`bijou/decoders/ar_backbone.py`): the FULL backbone
is the decoder — the prompt (which carries the `[generate|…]` request
and the soft state token, §1) is prefill-encoded once (layers 0–14,
cache retained), then the suffix runs ALL 35 layers against that
cache, and next-token logits come from the backbone's own frozen tied
LM head over the **full vocabulary**. There is no separate decoder
network; the module owns only **~11M new parameters** at E2B scale:

- `fast_embed` / `fast_ple` — input-embedding and per-layer-embedding
  rows for the FAST block, scaled like the backbone's own tables (√dim).
  Warm-started around the real tables' row mean + 0.02 noise
  (`init_tables_from_backbone`) so block logits start near the average
  text logit under full-vocab CE.

(The state projection moved PROMPT-side with format 3 —
`GemmaEncoder.state_proj`, §1 — and the format-4 mode tables are
deleted: the request conditioning subsumed the fed `[ACT]`/`[AUX]`
mode.)

**Molmo2 variant** (`Molmo2ARDecoder`,
`bijou/decoders/ar_molmo2.py`): the same recipe on the Molmo2 trunk,
with the differences the architecture forces. The prefill is
full-depth (Qwen3 has no KV-sharing truncation point) with the cache
retained; the suffix runs all 36 layers against it. Molmo2's
embeddings and LM head are **untied**, ship no reserved-unused block,
and both stay frozen — so the FAST block gets *fresh untied rows*: a
`fast_embed` input table and `fast_head` output rows appended beside
the shipped head (base-vocab logits from the frozen `lm_head`, block
logits from `fast_head`), both mean-initialized from the corresponding
frozen tables. No input-embedding scaling and no per-layer embeddings
(Molmo2 conventions). Auxiliary text reads the frozen shipped head —
for the original vocabulary both embedding and head sides stay frozen
by design.

**FAST block placement.** The action vocabulary (BPE + BOA + PAD,
`vocab_total` = 1026 for fast_tokenizer_v2) is TAIL-anchored at
`block_base = vocab_size − vocab_total` — E2B: ids 261118..262143,
inside the 3259-id reserved-unused run starting at 258885. No embedding
resize, no magic constant, adapts to any backbone; recorded in the
checkpoint's decoder section. The backbone never consumes FAST ids as
ids (suffix tokens enter as embeddings); the ids exist so actions and
text share ONE softmax: `lm_head` logits are computed, the block's
columns are overwritten from the patch, and the softcap is applied
AFTER the overwrite (block capped identically to text).

**Suffix format 5** (`aux_text.SUFFIX_FORMAT`, recorded per checkpoint;
formats ≤ 4 — fed mode tokens, suffix state slot, header bytes — are
REFUSED: their parameter sets are incompatible and no artifact worth
loading exists, owner call 2026-08-03):

    [<|turn>model\n][value\n per requested field][BOA][t_1..t_k]

The opener is Gemma-4's REAL generation prompt — the exact 3 tokens
(`[105, 4368, 107]` on E2B) `apply_chat_template(...,
add_generation_prompt=True)` appends, verified against the real
checkpoint 2026-08-02. What follows is exactly what the prompt's
`[generate|…]` requested (§2.4): one HEADERLESS `value\n` line per
requested field, in request (= template) order, then **BOA** —
retained as the action block's own single-id begin-marker
(constrained-decode anchor; codec output consumed verbatim; ar_fast
convention unchanged) — then the FAST tokens. "What speaks" is
COMMANDED by the prompt, per field: the model learns
p(value | observation, asked) and is never asked to predict from
appearance whether a judge happened to label a frame; the fast path
`[generate|actions]` trains on its own sample mass (unjudged frames +
request dropout). Teacher-forced full-vocabulary CE: opener positions
are IGNOREd except the last (its target is the first value token, or
BOA on `[generate|actions]` rows); PAD is batch padding, always
ignored; no EOA (action length is fixed by the FAST grammar). Loss
components: `total = action + w·aux` with per-position mean CE split
by the collator's aux mask — value lines are aux, BOA + block tokens
action (`--aux-loss-weight`, default 0.5); aux is logged as a
position-weighted mean (CE sum / token count, all-reduced), so
sparsely-labeled corpora don't dilute `train/loss_aux` toward 0.

**Decoding is fully scaffolded by the request**
(`predict_chunk(generate=…)`, which must equal the request the
prompt was collated with — `Collator.generate_override`, one tuple,
one source): per requested field in order — constrained fields
(holding) are pure classification (score the candidates' first ids
once, force the winner + terminator); free-text fields decode
greedily over text ids under the field's `VALUE_BUDGETS` cap until
the `\n` terminator (exhaustion forces it, loudly, incrementing the
cumulative `fallback_count`) — then **BOA is FORCED** (its target is
trained; its identity is not a decision) and the ACTION phase runs
the ar_fast grammar mask (each step masks to tokens whose BPE symbol
expansion fits the remaining chunk×dim budget; a full-length chunk is
guaranteed by construction). `generate=()` IS the deployment fast
path — forced `[opener][BOA]` prefill, straight to actions.
Requesting a field the checkpoint never trained is a loud error.
Never compare chunk numbers across different request sets — they are
different measurement conditions by design (probes score
`generate=()`; the samples table decodes all trained fields).

Prompt-side geometry is what makes the suffix exact: left padding +
logical positions (§1) put every suffix token physically adjacent to
its sample's real prompt, positions continuing after each sample's real
prefix length.

### 2.4 Auxiliary text tasks (`bijou/aux_text.py`)

Trained text outputs rendered from the LLM-judge annotations
(`docs/episode-annotations.md`): the collator draws each sample's
REQUEST SET (which fields the prompt's `[generate|…]` asks for,
requested ⊆ labeled always) and the matching HEADERLESS value lines
(template v4). For `[generate|subgoal holding progress event actions]`:

    reach toward the toy boat\n
    no\n
    30%\n
    none\n

- **Request-conditioned, per field.** Which field a line answers is
  pinned by request order (template order always), not by generated
  header bytes — v3's `[field]` headers were fully predictable given
  the request, i.e. pure padding, and were dropped (owner call
  2026-08-03; parsing zips lines with the request; report tables
  re-attach field names at the DISPLAY layer). Label availability:
  subgoal exists on every frame of a judged episode
  (piecewise-constant `language_persistent` rows);
  holding/progress/visible only on judge-sampled frames (the finite
  mask IS the sampled-frame set — never interpolated); event wherever
  its status is KNOWN — the firing frame's text (multi-event frames
  are real data, 441 in community_curated_v0, rendered "; "-joined;
  lerobot's single-row `emitted_at` RAISES on them — it killed a
  corpus run 2026-08-02) or the **explicit `none`** on judge-sampled
  no-event frames (a TRUE negative; unsampled frames are unknown and
  never requested). Label PRESENCE is conditioning, never a
  prediction target — the model learns p(value | obs, asked), and at
  inference any trained field can be requested on any frame.
  Unjudged samples request nothing and train the fast path — mixed
  sparsely-annotated corpora train one format, and an aux fine-tune
  extends a pretrained base rather than fighting it.
- **Tokenizer contract** (the v2 scar, one seam left): training
  assembles suffixes from PER-FIELD encodings (never one joint
  string), so cross-line merges cannot exist by construction; the one
  remaining split point — value|`\n` terminator — is asserted at
  `build_aux_runtime` construction (`enc(value) + enc(\n) ==
  enc(value + \n)`, real candidates for constrained fields; measured
  to hold on E2B for every field's value class). Values are
  \n-SANITIZED at render: headerless lines mean a stray newline in a
  judge string would silently shift every later line's supervision
  (with v3 headers that was a local parse failure). The v2
  `field: value` template broke the equivalent property (`" yes"`
  merged) and shipped a silently-wrong holding metric — the tripwire
  test injects a merging stub so the class can only recur loudly.
- **The "event" lerobot language style is registered by
  `bijou.annotations`** (idempotent import-time set-adds) — the DAG
  leaf under both the judge writer and the training reader, so either
  import order resolves event rows.
- **Request dropout** (`--aux-dropout`, default 0.1 on aux runs): a
  labeled sample's request collapses to `{actions}` with probability
  p — keeps the deployment fast path trained even at 100% annotation
  coverage. **Field dropout** (`--field-dropout`, default 0.1):
  each labeled field independently drops from the request (request
  and target move together — a requested field is always supervised
  and vice versa), so all SUBSETS of the labeled set appear in
  training and inference-time partial requests stay in-distribution.
  Draws are per-visit from a generator seeded by the dataloader
  worker seed (pure function of --seed, rank, worker); probe-side
  collators run dropout-0 clones.
- **Field set and order.** `--aux-fields` selects a subset of
  {subgoal, holding, progress, event, visible} but never reorders
  (template order is validated at the CLI boundary and re-guarded in
  `AuxSpec`; new fields APPEND); free-text values are truncated
  loudly (subgoal 20, event 24 — multi-event joins; the same numbers
  are the decode-side `VALUE_BUDGETS`). `visible` renders which
  cameras see the task object and the gripper as PROMPT POSITIONS
  (`object 0,1; gripper 1` — ascending indices into the §1 camera
  order): positional on purpose, since kind names collide (two
  unknown-kind cameras), short names are dataset-internal vocabulary
  the prompt never shows, and indices stay invariant under
  camera-kind dropout. View-binding is the most directly
  grounding-targeted label available; "none" on a sampled frame is a
  TRUE negative (occlusion is signal). Surface disagreements (kinds
  map ≠ vector slots ≠ item cameras) skip the field LOUDLY once per
  dataset per worker (guessing through misaligned slots would label
  the wrong cameras); aux training rejects `--cameras`/`--max-cameras`
  outright — camera selection would silently shift every index.
- **Template versioning.** `AUX_TEMPLATE_VERSION` (4) rides in the
  checkpoint's decoder section (`AuxDecodeConfig`: version, fields,
  prompt hash, judge model); loading a version this code doesn't know
  is a loud error — a byte-level convention change on an existing
  checkpoint would silently break decoding, so conventions change
  only with a version bump.
- **Label provenance is the dataset's own stamp** (revised 2026-08-02
  with docs/episode-annotations.md: the stamp is the in-band blessed
  materialization, and a code-level pin was wrong-by-construction —
  `bijou.judge.PROMPT_HASH` advances with the judging code, not with
  materialized labels, so the old tripwire would have silently
  disowned every existing corpus at the next judge-prompt change). At
  selection time `data.annotation_stamp` parses each dataset's
  `meta/judge_annotations.json`; absent/unparseable ⇒ trains as
  unjudged, loudly. The checkpoint records the DISTINCT stamps
  (`AuxDecodeConfig.prompt_hash`/`judge_model`, "+"-joined); mixed
  stamps across a corpus are legitimate (each dataset's labels match
  its own stamp). `--aux-prompt-hash` is the opt-in per-run pin for
  sweeps that must fail loudly on a mid-sweep re-materialization.
  `--aux-fields` with zero stamped datasets is a startup error.
- **Id spaces.** Aux ids are ordinary text-vocabulary ids — the
  full-vocab head exists for exactly this; the collator
  (`assemble_suffix`) builds one mixed suffix tensor in backbone id
  space (guarded: aux ids must sit below `block_base`) plus the
  aux-position mask the loss splits on.
- **Metrics.** Component losses `train/loss_action` / `train/loss_aux`
  (the aux mean is position-weighted across batches and ranks);
  in-run eval logs the all-trained-fields scaffolded decode as
  display-string columns (`aux_generated` vs `aux_label`, field names
  re-attached) inside the `eval/samples` wandb table — the table's
  chunks and text come from that one decode (self-consistent rows),
  while the scalar MAE probes score the `[generate|actions]` fast
  path (comparable across aux-on/off/less arms) — and
  `eval/samples_holding_acc`: the constrained holding value of the
  table decode vs the label over labeled rows (request conditioning
  elicits every requested field in its training context, so the old
  separate likelihood probe dissolved into the main path). Labels are
  weak supervision (~80% inter-judge agreement on holding, ±15%
  progress MAE): weight modestly, expect an accuracy ceiling near the
  label noise.
- **Offline eval surfaces** (shipped 2026-08-04): `bijou.eval` runs a
  NARRATED PASS automatically on aux-trained checkpoints (a second
  policy sharing the loaded model, prompt requesting every trained
  field) — its paired chunk MAE vs the fast path is the full-sample
  does-narration-help answer; generations feed holding-accuracy and
  progress-MAE vs the weak labels over every labeled sampled frame
  (the proper-n version of the in-run 12-row probes) and appear next
  to labels in the HTML sample blocks. Q2 outcome slices and the Q3
  sensitivity counterfactual ride the same run (JSON + HTML + stdout;
  Q3 auto-measures when outcome conditioning is trained and no manual
  --condition-override is given).

## 3. Flow matching — objective and sampling

lerobot's π0/SmolVLA convention. With ε ~ N(0, I) and clean chunk a:

    x_τ = τ·ε + (1−τ)·a          target  u = ε − a
    τ ~ Beta(1.5, 1) → (0.001, 1]   (mass toward τ=1)

**τ direction (read this before interpreting any per-τ statement): τ=1 is
PURE NOISE, τ=0 is data** — π0's diffusion-flavored direction, INVERTED
relative to the Lipman/rectified-flow convention (t=1 = data). "High τ"
anywhere in these docs = the noise end = the FIRST integration steps
(initial chunk placement from prefix context alone); Beta(1.5, 1)
therefore up-weights the noisy regime during training.

MSE of the expert's velocity against u over the FULL chunk: episode-
boundary chunks carry repeat-last-action targets (lerobot's delta-
timestamps query clamps indices to the episode range, so tail positions
hold the final real action — verified elementwise on v1). This replaced
masked-out padding: the expert attends every chunk position (directly
under bidirectional self-attention), so masked padding still shaped
predictions invisibly, and "hold the last action" is the correct
post-completion behavior. The scan
(`reports/episode_lengths.json`) sized the alternative — full-chunk-only
start sampling — at ~12% of start positions lost; rejected. Eval still
scores real steps only. Sampling integrates τ from 1
(noise) to 0 with exact endpoints; **Heun** (2nd-order predictor-
corrector, 2 evals/step) is default — ~2× lower integration error than
Euler at equal cost; Heun-10 is the eval convention. The τ→0 corrector
evaluates exactly at τ=0 (negligible extrapolation for the smooth time
embedding).

Sampling analyses (§8) show the field is rougher than assumed:
Heun-5→30 recovers ~1 MAE (integration error ≈ model-error floor at
Heun-10), and across-noise-draw std (~5.9°) exceeds single-draw error —
mean-of-N draws is the single largest known accuracy lever (§8.7).

## 4. Data and normalization

**Selection & holdout** (`data.select_datasets`, shared by train/eval):
collection roots `<user>/<dataset>`; loud drops for incompatible dims /
missing features / corrupt stats / duplicate repo ids. `--holdout-episodes
F --split-seed S` is a deterministic per-dataset episode split, a pure
function of (S, repo_id, count, F) — reproducible train/eval agreement
with no persisted files; every ≥2-episode dataset contributes ≥1 holdout.
`--fps` optionally keeps only datasets at given frame rates (the 50-step
chunk spans a fps-dependent wall-clock horizon; ~88% of community frames
are 30fps). `--camera-counts` optionally keeps only datasets with the
given camera COUNTS: prompt length is ~160 + 140/camera, so mixed
counts inside a batch pad every short prompt to the longest (wasted
prefix compute + DDP rank stragglers — the measured 65–97% util
spread); on community_curated_v0, `--camera-counts 1 2` keeps 878/981
datasets, 42,872/52,507 episodes (81.7%), 83.6% of frames (measured
2026-08-03 from per-dataset info.json). Both filters change the
concatenated frame indexing — eval numbers are only comparable between
runs with identical filters.

**Per-dataset MEAN_STD normalization (load-bearing).** 59–95% of the
aggregate action variance across community rigs is between-rig calibration
offset that images cannot see; aggregate normalization leaves a trained
model *behind* state-copy. Each sample is normalized with its own
dataset's stats (`StatsAttachedDataset` attaches them per item, in the
worker). Checkpoints store the full per-dataset table + a count-weighted
aggregate fallback; inference normalizes with the deployment rig's stats
(rollout `--stats-repo-id`).

**Quantile stats** (q01/q10/q50/q90/q99, action+state) ride in the same
`meta/stats.json` and the same per-item mechanism — needed by the FAST
tokenizer (§8.3). lerobot's native dataset-level quantiles are a
count-weighted *mean of per-episode quantiles*, which is WRONG (quantiles
don't compose by averaging; extremes regress toward the median, measured
−54° vs exact −120° on rig v2); the corpus is corrected to exact
quantiles by `ldtools.backfill_quantile_stats`. mean/std/min/max compose
correctly and are untainted.

**Prompt conditioning** (`--condition-fields subgoal outcome
smoothness`, default off): the `[key|value]` bracket block inside the
sandwich (§1). **Subgoal-conditioned execution** (C2): the frame's
current segment label renders as `[subgoal|…]` with its OWN dropout
(`--subgoal-dropout`, default 0.5 — deployment mostly runs
planner-less, so the unconditioned context must stay well-trained),
resolvable from a planner/operator at inference (`rollout --subgoal`,
or an explicit `item["condition_subgoal"]`); **anti-copy coupling**:
when the subgoal rides the prompt, the aux draw EXCLUDES it from the
request set — prompt-conditioning and prediction are exact
complements, so `loss_aux` never trains or scores copying, and the
self-conditioning loop (feed the generated subgoal into the next
replan's prompt) stays a rollout-side option. **Outcome conditioning**
(C1; `--condition-dropout` 0.1): hindsight labels from each
episode's verdict —
`outcome ∈ {success, partial, failure}` from `task_completion_visible`
(`unclear` renders nothing), `smoothness ∈ {high, medium, low}` from
the 1–10 score bucketed 8–10/5–7/1–4. Train-time, failed/partial demos
train under their own label instead of as-if-good — on
community_curated_v0 that is 13.7% partial + 4.1% failure of 56,328
judged episodes (≈10k episodes made usable); deployment asks for the
behavior it wants (rollout `--outcome success --smoothness high`
defaults). Per-field dropout keeps the unconditioned marginal trained;
encoder-side placement makes it work for every decoder kind (a suffix
placement would be ar_backbone-only). The checkpoint's `prompt` section
records `condition_fields`; loaders render matching conditioning.
**Eval semantics (three questions, three measurements):** Q1 fit —
probes/eval condition on each frame's TRUE labels (scoring against
recorded actions while conditioned on "success" would penalize exactly
the trained deviation from failures); Q2 deployment proxy — the same
pass sliced per outcome (`eval/chunk_mae_success` ≈ open-loop
deployment, `_unlabeled` = continuity anchor vs pre-conditioning
checkpoints; zero extra compute); Q3 conditioning sensitivity — THE
tripwire for silent conditioning collapse: on labeled non-success rich
rows, decode again with outcome forced to "success" and log mean
|Δprediction| (`eval/condition_sensitivity`; pre-registered: > 0 and
growing — ≈ 0 means the label is ignored and the failed-demo mass
trained as-if-good after all). Offline, `bijou.eval
--condition-override outcome=success` runs the full counterfactual.
Never compare numbers across conditioning contexts.

**Instruction augmentation** (`--instruction-augment P`, default 0.0):
with probability P a judged episode's recorded task string swaps for a
uniformly drawn judge-suggested rewrite (2–3 grounded alternatives per
judged episode, sidecar `suggested_instructions`, parsed typed through
the `bijou.annotations` contract and attached per episode by
StatsAttachedDataset). Uniform on purpose — phrasing DIVERSITY is the
goal, not quality filtering; unjudged episodes always keep the recorded
string, the CLI `--instruction` override beats both, and probes/eval
score the recorded instruction (augment-0 clone), so eval numbers stay
comparable across P. The rewrite pool also includes the judge's
`observed_task` description — hindsight relabeling to what actually
happened — GATED to completion==yes episodes until outcome
conditioning ships (an instruction describing a failure must not train
as instruction-following without the outcome label explaining it). Directly attacks the rollout brittleness that the
task string must match the recorded instruction. Pre-registered
expectation for the first augmented run: recorded-string probe MAE may
read marginally worse early while phrasing robustness improves —
expected, not a regression.

**Robustness.** Workers spawn (never fork — torchcodec is fork-unsafe);
decoder cache capped (default OOMs hosts); `tolerance_s = 0.5/fps` (v3
concatenated files break 1e-4 tolerance — and 80× above the
timestamp-desync drift class the curation census found, which only
trips lerobot's default); unfetchable items (corrupt videos, desynced
frames) are substituted with a far index from the same dataset, loudly,
with bounded retries (two runs died before this guard), and a
**circuit breaker** aborts when a (worker, dataset)'s failure rate
exceeds 5% over ≥100 fetches — substitution absorbs rare pathologies,
it must not paper over systematic breakage or a bad refactor.

## 5. Training system

DDP via torchrun, per-rank batch/workers; loss and MAE probes all-reduced,
checkpoints/logging on rank 0. Optimizer AdamW (0.9, 0.95), fused on CUDA
(CPU keeps the reference path so the loss oracle is stable), cosine
schedule to 10% after linear warmup, grad-clip 10.0 (a never-bites safety
net; a tighter clip renormalizes most steps and injects moment noise) —
except ar_backbone, whose full-vocab CE runs larger grad norms:
convention is `--grad-clip 100` there (step-1 norms in the 10^4 range
are softcap saturation, clipped in practice).

**Optimizer variants.** `--optimizer {adamw,adamc}` (default adamw).
AdamC ([arXiv 2506.02285](https://arxiv.org/abs/2506.02285)) corrects
decoupled weight decay for the LR schedule: with AdamW, a layer's
steady-state gradient-to-weight ratio is `√(2λ/γt)`, so LR decay
*raises* equilibrium gradient norms late in training; AdamC scales the
decay coefficient with the schedule (`λ̂_t = λ·γt/γmax`) on hidden
("normalized") layers, while the output head keeps standard decay and
1-D parameters stay undecayed. Implemented as stock fused AdamW with a
per-group time-varying `weight_decay` written immediately before each
step — bit-exact, no custom kernel. Tied/shared parameters are handled
by construction: each parameter object sits in exactly one group
(asserted, both modes — a tied embedding/head pair decayed from two
groups is the classic failure), and decoder types whose output-layer
partition hasn't been audited are refused. Oracles:
`tests/test_adamc.py` (bitwise AdamW equivalence at peak LR, exact λ̂
trajectory, ZeRO-1 group-sync contract).

**Memory and throughput machinery** (all oracle-gated, semantics
exact): `--zero1` shards Adam moments across ranks (torch
ZeroRedundancyOptimizer: each parameter's state lives on one rank,
updated shards broadcast after each step — per-rank optimizer memory
~1/world); with chunked backward, `--chunk-grad-allreduce` replaces
the DDP wrapper with one explicit in-place gradient all-reduce per
step (DDP's reducer buckets — a full fp32 gradient copy — are
allocated at construction even when unused; measured 13.6 GiB on the
Molmo2 recipe); `--activation-checkpointing` recomputes trunk blocks
in backward (memory-only, gradient bitwise-pinned).

**Asynchronous checkpointing** (`bijou/async_save.py`, default on;
`--sync-save` = legacy path). A save boundary captures a device→CPU
snapshot in seconds and gathers/merges/writes on a background thread
over a dedicated gloo group (never the training NCCL communicator);
under ZeRO-1 the background gather replaces a measured ~15.5 min/save
synchronous consolidate (~14% of wall time on the Molmo2 4×DDP
recipe). Written bytes are oracle-pinned identical to the sync path;
publishes are atomic (`.tmp` dir rename); the final save joins before
process-group teardown.

**Resume seed discipline.** Nothing restores the data-stream position
on `--resume`: the loop restarts at epoch 0 with the `--seed` shuffle
and per-rank noise streams, so resuming with the checkpoint's own seed
would replay exactly the batches (and flow-matching τ/ε draws) already
trained on. Resume therefore *demands* a fresh `--seed`
(`--allow-same-seed-resume` exists for deliberate reproduction).

**Component learning rates.** `--decoder-lr` (always > 0), plus optional
`--backbone-text-lr` / `--backbone-vision-lr` — omitting a component's
lr keeps it frozen (explicit 0 is rejected); the flags mirror the
param-group names {decoder, backbone_text, backbone_vision} (renamed
2026-08-01 from --expert-lr/--text-lr/--vision-lr; recorded train_args
load under both spellings forever). All groups share the cosine shape
scaled to their own peak. Backbone training uses fp32 master weights
with a bf16 autocast prefix encode (bf16 updates vanish below bf16
resolution at ~1e-5); the decoder stays fp32-with-TF32 outside the
autocast region. One
`BijouTrainStep` module owns prefix-encode + objective in BOTH regimes
(`backbone_trained` selects no-grad native-dtype encode vs grad + autocast),
so a single DDP wrapper (`static_graph`) hooks everything trained;
single-process frozen math is byte-identical to the historical
decoder-only wrap (oracle-exact), multi-rank frozen runs changed
gradient bucketing composition at the 2026-08-01 refactor (declared
re-baseline). See §8.1.

**Chunked backward (`--backward-chunks N`, default 1).** The memory
fallback when a loader batch doesn't fit (E4B screen pre-reg
2026-08-05): each step's item list splits at COLLATE time into N equal
sub-batches (each pads to its own max — position ids are padding-mask
cumsums, so width is inert to the math), forward/backward runs per
chunk with DDP `no_sync` on all but the last, and each chunk backwards
its SUM-form loss divided by the FULL-step normalizer counts
(`loss_count_normalizers`: data-only, computed before any forward).
Global-count normalization — not a mean of chunk means — makes the
accumulated gradient exactly the unchunked one even when chunks carry
unequal valid-token counts (token-weighted CE pooling; the aux ratio
uses the global aux count the same way). Sample composition, effective
batch and the LR schedule are invariant; must divide `--batch-size`;
N > 1 drops `static_graph` (plain DDP is the well-trodden
accumulation path). Equivalence contract: "equal up to fp reduction
order" — measured 2026-08-05 on the tiny CPU fixture: chunk math with
bit-identical memory reproduces gradients to rel ~5e-7; per-chunk
collation widths shift the prefix-encode fp realization, amplifying to
rel ~2e-4 on the RANDOM saturated fixture (forward identical to 1e-6;
ar_fast CLI A/B bitwise at printed precision, flow draws chunk-shaped
noise — same law, different realization). The N=1 path is byte-exact
(loss oracles reproduced). Tests: `tests/test_chunked_backward.py`.

**Decoder-kind CLI semantics.** flow accepts the `--decoder-*` shape
flags and `--stream-counts`; AR decoders require `--fast-tokenizer
<artifact>`; **ar_backbone rejects the shape flags and stream counts**
(the backbone IS the architecture — flags describing a model the run
doesn't build are errors, not ignored). `--aux-fields` (ar_backbone
only; incompatible with `--cameras`/`--max-cameras` — positional
visible indices) enables aux text training (§2.4);
`--aux-loss-weight` sets w (default 0.5 — the labels are weak
supervision); `--aux-dropout` sets the request-collapse rate and
`--field-dropout` the per-field request dropout (defaults 0.1/0.1);
`--aux-prompt-hash` is the opt-in provenance pin (§2.4); `--camera-kind-dropout` (default 0.1,
all decoder kinds) is the prompt-side kind→unknown dropout (§1);
`--instruction-augment` (default 0.0, all decoder kinds) samples
judge-suggested task rewrites (§4); `--condition-fields` /
`--condition-dropout` / `--subgoal-dropout` (default off / 0.1 / 0.5)
render prompt conditioning — subgoal hint + hindsight
outcome/smoothness (§4). Train step returns
component losses; `train/loss_action` + `train/loss_aux` log beside
`train/loss` on aux runs (aux aggregates as CE-sum/token-count across
the window and all ranks — a position-weighted mean, immune to the
sparse-batch dilution a mean-of-means would suffer). Later additions
not detailed above:
`--distill snapflow` + `--target-time-embed` (one-step distillation,
§2.1), `--state-dropout` (proprioception dropout), `--max-crops`
(Molmo2 image crops), `--bucket-by-length` (length-bucketed batching),
`--rewarmup-steps` (resume re-warm ramp, see the resume paragraph),
`--prompt-generate-bracket`.

**In-training probes.** `--eval-samples N` sizes two MAE probes
(eval_chunk_mae on holdout, train_mae on train), drawn exactly as
`bijou.eval --seed` would, sharded and all-reduced, CPU-resident.
ar_backbone probe prompts render `[generate|actions]`
(`generate_override=()`) and decode the fast path, so scalar MAE is
comparable across aux-on/off arms and agrees with offline eval. Two
one rank-0 wandb table over EVAL_TABLE_ROWS = 12 rich rows (down from
32 — the matplotlib figures were a measured ~34s/eval rank-0
straggler): `eval/samples` = chunk columns straight off the scalar
pass (fast path — actions condition on the user message only, no aux
text in the suffix; chunk_mae matches the scalar's measurement
condition) PLUS `aux_generated`/`aux_label` side-channel columns from
the all-fields decode of the same items (what the model says for this
observation next to the fast-path chunk — deliberately mixed
conditions, owner-requested pairing). The all-fields decode also
yields `eval/samples_all_fields_mae` — masked MAE of the chunks that
followed the model's SELF-generated field lines over those 12 rows
(paired does-narration-help signal; small-n, directional, never
compared to the full-probe scalar; a dedicated all-fields table was
dropped 2026-08-03 as visually redundant) — and
`eval/samples_holding_acc` (generated holding vs label over labeled
rows — the constrained value in the MAIN decode; the separate
likelihood probe dissolved with request conditioning).

**Checkpoint schema** (`loading.py` dataclasses): `expert.safetensors` +
`bijou_config.json` (format 3: role-sectioned — `backbone` {id,
depth: prefix|full}, `prompt` {kind, exports, max_soft_tokens},
`decoder` = the tagged config with stream-name schedules; ar_backbone's
records the tokenizer artifact ref, block placement, `suffix_format`
and the `aux` provenance record (§2.4; absent keys parse as format 1 /
no aux) — plus per-dataset + aggregate stats, train args, step);
`optimizer.pt` for lossless `--resume`; and `backbone.safetensors`
(bf16 trunk snapshot) **iff the checkpoint's trunk differs from pristine
HF — trained in-run OR inherited frozen from an adapted `--init-from`**
(the inherited file is hardlinked/copied, byte-identical; conditioning
on trained-in-this-run-only once shipped fine-tunes that silently loaded
the pristine trunk — the invariant is test-gated). Directories are
self-contained: loading one must need no other directory. Format-1/2
checkpoints: pre-format-3 prompts and pre-format-5 ar_backbone
suffixes are REFUSED at load (2026-08-03, no back-compat — their
parameter sets no longer exist: state_proj moved prompt-side, mode
tables deleted); schema parsing is guarded by state-dict key fixtures
and section tests. `--init-from` = warm
start (decoder config-guarded, loud SystemExit — except the data-side
format keys {aux}, which may differ with a printed note: enabling aux
on an aux-less format-5 base
is the sanctioned warm-start pattern; NOT guarded: `--max-soft-tokens`,
`--backbone` — known footguns); `--resume` =
lossless continuation (CLI lr ignored, printed; cosine re-evaluated
over the new `--steps`, so extending re-heats LR — accepted when
reusing moments, else init-from + warmup; resume stays STRICT about
decoder keys — never resume across a format change).

**Holdout/eval CLI semantics.** `--holdout-episodes F --split-seed S`:
deterministic per-dataset episode split, a pure function of (S,
repo_id, count, F); train loads the train side; `bijou.eval --episodes
{all,train,holdout}` with the same flags reproduces it exactly.
`--eval-samples` (required when holdout > 0) sizes the two in-training
probes; `--eval-seed` is deliberately separate from `--seed`. Eval
always scores the state-copy baselines alongside; `--fps` filters
change the concatenated frame indexing, so numbers are only comparable
between same-filter runs. Rollout: `--stats-repo-id` must be a repo id
the checkpoint trained on (per-dataset table lookup); AR decoding
additionally requires its quantiles.

**Regression gates.** `check.py` (ruff + pyright + pytest, final verdict
line only). The **loss oracles** — a 2-step tiny-backbone CPU run after
any change near the math (tied to the current tiny-gemma4; regenerate ⇒
re-baseline loudly):

    uv run python -m bijou.train --train-data ~/datasets/mcobzarenco/so101_pick_place_v2 \
      --backbone outputs/tiny-gemma4 --decoder-hidden 64 --decoder-heads 2 \
      --decoder-intermediate 128 --decoder-cross-heads 2 --stream-counts 1 1 2 \
      --steps 2 --batch-size 2 --num-workers 2 --log-every 1 --eval-every 5 \
      --save-every 1000 --eval-samples 4 --device cpu --seed 0 \
      --save-dir outputs/train/oracle_tmp

must reproduce **flow 2.7903 / 1.9152** exactly; with `--decoder
ar_backbone --fast-tokenizer tests/fixtures/tiny_fast_tokenizer` (and
the `--decoder-*` shape flags OMITTED — ar_backbone rejects them),
**27.8306 / 27.767** (random tiny weights under full-vocabulary CE —
an anchor, not a quality signal).

The **molmoact2 objective matrix** (retirement phase 3) anchors on the
tiny molmoact2 fixture (`PYTHONPATH=. uv run python
probes/generate_tiny_molmoact2.py` → `outputs/tiny-molmoact2/` —
per-checkout like tiny-gemma4; regenerating re-baselines these
loudly). Same 2-step CLI shape with `--init-from
outputs/tiny-molmoact2/checkpoint` on the rig v2 data, seed 0, batch
2:

- `--objective flow`: **1.3906 / 1.3305**
- `--objective ar --backbone-text-lr 1e-5`: **12.2254 / 12.3317**
- `--objective joint --insulate-expert --backbone-text-lr 1e-5`:
  **13.6160 / 13.6621**, with the built-in cross-oracle: its
  `loss_action` ≡ the flow anchors and `loss_aux` ≡ the ar anchors
  BITWISE, and total = flow + λ·CE exactly (λ = 1) — the decision-5
  ordering and λ-composition proven inside the real trainer (recorded
  2026-08-14 at phase-3 landing; the run also exercises the
  released-BPE quantization-hole policy — real rig chunks DO hit
  holes, tokenized short + counted like the reference recipe, loud
  never silent). Re-baselined 2026-08-13 at the T1
(ar_fast-retirement) pre-deletion measurement: flow reproduced its
2026-08-05 anchor bitwise, ar_backbone had MOVED from 27.8262/27.7701
somewhere in the 08-05→08-13 window with no re-baseline note — the
exact change is UNBISECTED (the perf-pass CE-reduction changes are
the natural suspects, but their landing gate verified 118/118 bitwise
hashes on its own fixture — and the gradflow probe's single-forward
flags-on anchor still reproduces its 2026-08-05 value exactly, so the
mover sits in the train-step/collation path, not the model math). The drift
predates the T1 deletions — measured before them, twice,
bitwise-stable — which is what this re-baseline pins; a tripwire that
moves without a note has failed at its one job, so future oracle-adjacent
changes must carry the re-run in the same commit. The retired ar_fast
anchors (4.9232/4.8631 on this corpus) retired with the decoder — tag
`pre-decoder-simplify` reproduces them.
Re-baselined 2026-08-05: the oracle corpus is now the rig v2 dataset
`so101_pick_place_v2` at its standard box-mirror path — staged on
every machine, unlike the laptop-only `community_dataset_v1_v3` the
oracles used before (owner call, #fontaine; the corpus change moves
every anchor, same code — verified bitwise-reproducible twice on the
fontaine box at ML-code parity with main). Prior anchors on the
retired corpus: 2026-08-03 formats 3+5 flow 1.7766/1.6235, ar_fast
4.8795/4.8750, ar_backbone 27.8513/27.7803; format-2+4 flow
1.8896/1.7237, ar_fast 4.8917/4.8683, ar_backbone 27.7483/27.7840;
earlier: format-3 27.7622/27.7245, format-2 27.8116/27.8348, ar_fast
pre-tags 4.8803/4.8656.
Flags-on (unfreeze) oracles live in
`probes/probe_unfreeze_gradflow.py` — COMMITTED 2026-08-05: the
gitignored laptop copy had rotted invisibly two ways (TrainArgs
drift; the state_proj partition check still looked decoder-side
after the format-3 prompt-side move), so doc-cited instruments now
graduate to `probes/`, inside ruff + pyright's blast radius. Anchors
on the rig-v2 oracle corpus, re-recorded 2026-08-05 and
bitwise-reproduced twice: **flow 1.6948, AR 4.8395, ar_backbone
27.8546**, asserted in the probe alongside the FULL-depth partition
checks (retired-corpus asserts were 1.5966/4.8269/27.8524; the
1.5825/4.8345/27.7346 this paragraph cited before were stale
transcriptions of an earlier probe revision — prose copies rot, the
asserts are the source of truth). Cross-machine portability was
MEASURED, not assumed, during the corpus migration: the laptop
reproduces the box-recorded base flow oracle 2.7903/1.9152 bitwise,
so these CPU anchors are machine-portable. Regenerate the tiny backbone
with
`uv run python -m bijou.gemma4.testing --output outputs/tiny-gemma4`
(per checkout; changing it re-baselines every oracle). gemma4 changes
additionally gate on `verify_parity` (needs a big GPU). Any new
architecture path records its own oracle loudly.

**Step split** (Gemma E2B trunk, H100, batch 64, measured):
observation encode 79.3% / expert fwd 4.6% / bwd 15.4% / opt 0.7%.
(Molmo2 anchors, 4×H100 DDP: decoder-only training ~2.25 s/step at
12/rank; the frozen-trunk expert attach ran 0.92 s/step vs 3.74 with
the trunk trained jointly.) The frozen backbone forward dominates —
expert width is nearly free wall-clock; the perf wins are prefix-side
(§8.8). Unfrozen-text steps run ~2× frozen at matched batch; live-trunk
DDP runs measured 69–79 GB/rank (batch 32–64, H100) —
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps. ar_backbone
trains one teacher-forced pass over a ~30–80-token suffix instead of a
404M-expert denoise: the 100k mainline measured **median 0.444 s/step**
at B11/rank ×4×H100 with live text trunk. Its VRAM is
composition-dependent (3–4-camera community samples spike past 2-camera
estimates): B16/rank OOM'd twice on the community mix; **B11/rank holds
(~75.6 GiB peak)**; standing rule — OOM ⇒ `--resume` latest checkpoint
at B10.

**Hard-won library constraints** (do not re-learn): lerobot workers
spawn, never fork (torchcodec is fork-unsafe); decoder cache capped
(default OOMs hosts); `tolerance_s = 0.5/fps`; corrupt community videos
strike randomly (loud substitution with bounded retries); the
pretrained-branch `make_pre_post_processors` drops dataset_stats.
torch: pickled CPU tensors cost 1 shm fd each (DatasetStats stays
plain floats; file_system sharing set worker-side); SDPA with
head_dim > 256 needs materialized-KV or silently runs ~3× slower;
fused-SDPA vs additive-mask paths differ at bf16 ULP scale;
`Module.__getattr__` stubs return `Tensor | Module` (narrow after
ModuleList iteration); `Linear.bias` stubs lie (cast with comment).

## 6. Inference and deployment

`bijou.rollout` on the physical SO-101 (`docs/rollout_so101.md`): laptop
RTX 3000 Ada (8 GiB) fits the bf16 backbone (4.77 GiB) + bf16 expert; peak
6.29 GiB; ~233 ms/replan warm. `--max-relative-target` is a lerobot per-
tick rate limiter (not a safety system); camera names are positional
prompt slots; task string must match the recorded instruction. Deployment
always fine-tunes on rig data first (zero-shot cross-rig transfer is the
wall, §7) — so the operative metric for any change is fine-tuned-then-
scored rig MAE, not zero-shot. All decoder kinds serve `predict_chunk`
behind one policy interface, returning a `BijouPrediction` (`actions`,
mirroring the batch's ground-truth field, + aux `generations`);
ar_backbone picks its request set via `--generate [fields…]` on eval
and rollout (§2.3: omitted = the `[generate|actions]` fast path;
requested fields cost ~1 suffix forward per field plus its value
tokens per replan; no suffix KV-cache reuse across replans yet — the
known optimization if it deploys), and
AR checkpoints need the deployment rig's exact q01/q99 quantiles (the
tokenizer's fit normalization), which old-format stats tables don't
carry. Camera kinds at rollout derive from the operator's own
`--camera` names via `rollout.camera_kinds_from_names`: a name inside
the kind vocabulary IS its kind, anything else tags `unknown` with a
LOUD warning — name cameras by viewpoint (top/wrist/front/side) to
give the model the signal. Kinds ride each item
(`item["camera_kinds"]`), so no policy-level plumbing exists.
Conditioning at rollout: `--outcome success --smoothness high`
defaults (ask for the behavior you want; rendered only by
condition-trained checkpoints) and optional `--subgoal "…"` for
planner/operator hints — omitted, the dropout-trained unconditioned
context applies. Noise control for flow checkpoints: `--seed` fixes
the per-replan noise sequence; `--noise-ticket FILE` (an npz with a
`tickets` float32 `[count, chunk, dim]` array — the same format the
eval CLI consumes) replaces fresh noise entirely, so every replan
integrates from the same fixed vector (maximally consistent chunk
seams, no draw diversity; the file's sha256 prints in the banner for
attribution). `--sample-draws N` batches N flow draws per replan and
executes their mean; `--async-inference` overlaps planning with
execution and switches chunks at the horizon boundary.

**The offline eval system** (`python -m bijou.eval`) scores frozen
checkpoints on dataset frames — open-loop chunk MAE against recorded
actions, with state-copy baselines computed on the identical frames.
The surface, briefly:

- **Frame selection**: either `--num-samples N` (seeded uniform) or a
  frozen *sample plan* (`--sample-plan plans/*.json`) — a committed
  list of `(repo_id, episode, frame)` rows, so different checkpoints
  and modes score byte-identical frame sets. Episode-level
  train/holdout splits (`--episodes`, `--holdout-episodes`,
  `--split-seed`) hash episodes exactly the way training does; the
  standalone leakage checker (`bijou/eval/leakage.py`, its own CLI)
  certifies that a training corpus never contains panel-holdout
  episodes, including through filtered/renamed derived corpora
  (whose re-hashed splits silently move episodes across sides).
- **Decode control**: `--sample-steps/--sample-method/--sample-draws`
  (flow solver and ensembling), `--target-time zero` (one-forward
  endpoint decode for shortcut-trained checkpoints),
  `--ar-temperature` (sampled AR decodes), `--noise-key
  {stable,index}` (noise keyed by frame identity vs plan index),
  `--noise-tickets` + optional `--noise-ticket-map` (decode every
  frame from committed noise vectors — one shared vector, or one
  routed per dataset; the bank file's sha256 rides in every output).
- **Ablation probes**: `--mask-state` (zero the proprioceptive state
  input), `--subgoal-mode {oracle,self,draws,mcselect}` (condition
  the language slot on ground-truth segment labels, the model's own
  generated subgoal, or scored candidate subgoals), `--smolvla` (an
  external-baseline scorer on the same frames).
- **Outputs**: summary JSON (`--output-json`), a browsable HTML panel
  with rendered trajectories (`--report`), and per-frame npz dumps —
  `--dump-predictions` (one chunk per frame per policy + truth,
  validity, and frame identity columns) and `--dump-draws` (the
  pre-average `[frames, draws, chunk, dim]` stack). The npz dumps are
  the substrate for all downstream analysis scripts; they carry
  enough provenance (policy name, decode settings, noise/ticket
  hashes) to be interpretable standalone.

**Artifacts.** Checkpoints + tokenizers:
[`mcobzarenco/bijou-checkpoints`](https://huggingface.co/mcobzarenco/bijou-checkpoints)
(seed checkpoints keep `optimizer.pt`, the rest are weights-only).
Datasets: `mcobzarenco/community_curated_v0` (judge-filtered v1∪v2∪v3:
981 datasets, ~52.5k episodes fully materialized @ one stamp — the
pretrain corpus), the three raw community collections, and the two rig
repos (`mcobzarenco/so101_pick_place_{v2,clean}`, private) — all with
backfilled exact quantiles; boxes mirror them under
`~/datasets/mcobzarenco/`. Runs log to wandb `bijou-dev`
(entity `aristotle1337`). Laptop dev data:
`/home/marius/w/community_dataset_v1_v3` (3 datasets, 44k frames — the
oracle corpus).

## 7. Empirical grounding and results ledger

All numbers: open-loop chunk MAE, raw degrees, 256 frames, seed 0
(Heun-10 for flow; AR decodes greedily), state-copy baselines scored on
the identical frames. **Comparability rules**: numbers only compare
within one frame set — the three sets below (fps-30 + ≤2-camera;
fps-30; unfiltered legacy) index frames differently, and EVERY
selection filter defines a new set (`--camera-counts` changes the
concatenated indexing exactly as `--fps` does); token-level metrics
never cross
tokenizer versions; the 256-frame probe has a ±0.3 noise floor and the
in-run probe matches offline eval to ~0.01 when settings agree (per-
rank sharding once read ~0.3 high on the unftext r2 lineage — trust
offline for decisions). Full reports/JSONs in `reports/`; superseded
run narratives live in git history. Detailed setup + fit narratives for
mainline runs live in "Experiment reports" at the end of this section —
the ledger tables are the index, those are the record.

### Leaderboard — best checkpoint per family, community holdout (Gemma era, pre-2026-08-06 frame sets)

| family | best checkpoint | frame set (copy baseline, frames) | chunk_mae | ×copy | first_mae (copy) |
|---|---|---|---|---|---|
| **ar_backbone, request-conditioned** | `bijou_arb_rcond_100k_ddp4` @100k | fps-30 ≤2-cam (10.88, 16384) | **5.640** | 0.52 | **2.11** (2.56) |
| ar_backbone, full-vocab (suffix fmt 1) | `bijou_arb_fullvocab_100k_ddp4` @100k | fps-30 (11.50, 256) | 5.656 | 0.49 | 1.95 (2.73) |
| ar_fast, live text trunk | `bijou_ar_fast_v2_unftext2_50k_ddp2` @50k | fps-30 (11.50, 256) | 5.96 | 0.52 | 2.06 (2.73) |
| flow, adaRMS frozen trunk | `bijou_adarms_bidir_h1536` @100k | fps-30 (11.50, 256) | 6.92 (Heun-30) | 0.60 | — |
| flow, legacy mainline + unftext r2 | r2 @15k | legacy (10.30, 256) | 6.47 | 0.63 | — |

How to read it: rows come from the per-frame-set ledgers below (same
measurements, not re-runs); **×copy is the only column that crosses
frame sets, and only coarsely** — the 0.52-vs-0.49 inversion between
the two ar_backbone rows is within cross-set distortion (the ≤2-cam
filter drops hard multi-camera scenes AND lowers the copy baseline;
they never met on a shared frame set, and pre-format-3 checkpoints are
refused by current code, so they can't). The request-conditioned row
was the project best of the Gemma era: current-format, on the hub with
optimizer, scored at 16× the sample size of every other row, and the
only row whose first_mae beats its own copy baseline. For the current
picture — the curated-corpus frozen plan, the Molmo2 trunk swap, and
the flow-side bests that now lead overall — see the next subsection.

### Ledger — curated-corpus frozen plan (2026-08; copy 11.785 / first 2.620, 17,204 core frames)

The current evaluation standard: `community_curated_v0` holdout scored
against one *frozen, versioned frame plan* (25,800 frames drawn once,
17,204 "core" frames pooled for headline numbers — identical rows for
every entry, so per-frame paired deltas with bootstrap CIs replace
cross-run eyeballing). All later results, including everything the
research agent banks, live on this set; the living version of this
table (with decode-cost columns and per-result links) is the
[leaderboard on the agent's blog](https://mcobzarenco-fontaine-blog.static.hf.space/leaderboard.html).

| checkpoint × decode | chunk MAE | first_mae | note |
|---|---|---|---|
| Flow expert on frozen AR-pretrained Gemma trunk @80k, Heun-30, best 10-draw ensemble | **5.185** | **1.383** | best banked overall (`bijou_flow_artrunk_h1024_40k_ddp2` — the §8.11 recipe realized); noise-vector selection over a searched bank |
| SnapFlow 1-NFE student, mean-of-10 | 5.368 | 1.593 | one-step distilled flow — ~30× cheaper than the teacher config above |
| Gemma AR-100k, greedy (deployment anchor) | 5.803 | 2.143 | `bijou_arb_rcond_100k_ddp4` rescored on this plan |
| **Molmo2 AR 60k, greedy** | **5.860** | **2.072** | 40k + 20k fresh-data continuation; paired Δ(60k−40k) −0.139 [CI95 −0.194, −0.090] |
| Molmo2 AR 40k, greedy | 6.008 | 2.187 | the trunk-swap screen readout (below) |
| state-copy control | 11.785 | 2.620 | byte-matched on every eval |

**The trunk swap (2026-08-06→08).** Molmo2-4B replaced Gemma-4 E2B as
the training trunk on a matched-topology screen: the same decoder-only
recipe at 40k steps beat an *equivalent-topology Gemma control*
(7.797 on this plan) by a paired per-frame **−1.717 [CI95 −1.80,
−1.63]** — and reached within 0.21 of the Gemma line's fully-trained
100k checkpoint at 2.5× fewer steps, with ~3× cheaper greedy decode
(measured ~678 vs ~2157 ms per single-frame decode, same GPU class).
The 60k continuation closed most of the remaining gap on first-step
error while chunk MAE still trails the 100k anchor by ~0.06
(cross-trunk, unpaired — treated as parity pending longer training).

**Sampled decoding, both trunks.** Sampling the AR decode at T=1 and
averaging 10 draws buys only −0.145 (Gemma) / −0.154 (Molmo2) chunk
MAE — the same "mean collapse" on both trunks (greedy AR decode
already sits near the predictive mean), versus a ~9× larger
multi-draw gain in the flow families. Ensembling is a flow-side
lever here, not an AR-side one.

**Flow expert over the frozen Molmo2 trunk (2026-08-09).** The 404M
expert attached via residual-stream adapters to the frozen Molmo2
decoder-only checkpoint trained 10k steps cleanly (probes beat their
pre-registered bars throughout; state-copy beaten decisively on the
banked endpoint eval). A matched joint-fine-tuning arm (trunk
continues its CE objective beside the expert's flow loss, stop-grad
seam) was stopped on cost: ~4× the frozen arm's step time, consistent
with production recipes that also train the expert against a frozen
trunk. The attachment-recipe decision memo is in progress on the
agent's blog.

### Ledger — fps-30 + ≤2-camera frame set (Gemma-era mainline; copy 10.88 holdout / 10.81 norm-copy, 16,384 frames)

| checkpoint | comm holdout | note |
|---|---|---|
| **ar_backbone request-conditioned 100k** (`bijou_arb_rcond_100k_ddp4`) | **5.640** (first_mae 2.113, p50 4.08, p90 11.51) | prompt fmt 3 / suffix fmt 5 / aux v4; decoder 1e-4 / backbone 2e-5; B12→10 @20k; **project best**, −48% vs copy, first_mae beats copy's 2.557; narrated pass +0.043 (46% win) — see the report below. (Earlier 1024-frame read: 5.328 vs copy 10.463 — a slightly easy draw; the 16k numbers supersede it) |

This set drops 3–4-camera datasets (99 + 4 of 981; 18.0% of episodes),
which also lowers the copy baseline (10.46 vs 11.50 on the all-camera
fps-30 set) — the ratio to copy, not the absolute, is what carries
across the two sets.

### Ledger — fps-30 frame set (all camera counts; copy 11.50 holdout / 10.72 train / 12.00 rig-holdout-3403)

| checkpoint | comm holdout | rig holdout (3403) | note |
|---|---|---|---|
| adaRMS flow 40k (`bijou_adarms_bidir_h1536_40k_ddp2`) | 8.07 | 15.64 | from-scratch, bidir, h1536, frozen trunk |
|  … resumed 50k | 7.80 h10 / 7.72 h30 | — | Heun-gap collapse = adaRMS signature |
|  … resumed 100k | 7.11 h10 / 6.92 h30 | 14.55 | train 7.14 ≈ holdout: zero episode-fit gap when trunk frozen |
| AR FAST v2, frozen trunk, 10k | 8.34 | — | ≈ flow@40k quality in 10k steps; plateaued ~8.0 |
| **AR FAST v2 + live text trunk, 50k** (`bijou_ar_fast_v2_unftext2_50k_ddp2`) | **5.96** (first_mae 2.06) | **11.69** (first_mae 3.99) | decoder 1e-4 / text 2.5e-5; first pretrain to beat copy on rig holdout; train probe ~4.9 = episode-fit gap (live trunk memorizes scenes) |
| **ar_backbone 100k** (`bijou_arb_fullvocab_100k_ddp4`) | **5.656** (first_mae 1.951) | 12.458 vs copy 11.973† | decoder-only path, decoder 1e-4 / text 2.5e-5, B11×4, clip 100; suffix format 1 (pre-opener) — warm-start seed, not an inference target |
|  … rig ft 5k (`bijou_arb_ft_rig_5k_ddp4`) | — | min **11.48 @ step 250**† | decoder 2e-5 / text 1e-5; crosses copy at 250 then memorizes (same episode-fit arc as the AR ft round) |

† Rig rows above the dagger line scored the pre-rename `marius/*` split
("rig-holdout-3403"); the daggered rows scored the post-rename
`mcobzarenco/so101_pick_place_{v2,clean}` holdout (0.1/seed 0 — repo id
enters the split hash, so the rename IS a new frame set; its copy
baseline is 11.973). Rig comparisons cross the dagger only coarsely.

The live-trunk AR lines are the project's strongest evidence: the
ar_backbone 100k row is the ledger best (−5% vs the ar_fast line on
community holdout, first_mae 1.951 vs copy 2.726), and the decoder-only
path gets there with ~11M new params instead of a 404M expert — though
steps/architecture confound the margin over ar_fast-50k. The
frozen-trunk AR plateau at ~8.0
locates the gain in the trunk, not the decoder. Attribution is still
2-factor (live trunk × AR objective; flow-side unfreeze at 2e-5 was
only a modest win — legacy table below). LR sensitivity is extreme:
6e-4/3e-4 destroyed a warm 8.34 init to 15.95 within 1k steps (worse
than from-scratch at matched steps — the trunk outran the decoder);
1e-4/2.5e-5 dipped to 10.07 then delivered. Zero-shot rig transfer
remains the wall for every pretrain (12.458 loses to copy; the ft
crosses it within 250 steps).

Rig fine-tunes from the AR-unftext 50k base (5k steps, decoder 2e-5,
frozen-vs-1e-5 trunk arms): **no measurable holdout headroom** — both
arms' minima (11.51 / 11.02 @ step 500) tie the base's 11.69 within
the 5–6-episode holdout's noise, then memorize (train MAE → 1.2 by
5k). Unlike every earlier ft, this base trained on the rig data — the
pretrain banked the ft's gains. The effective sample unit is episodes
(57 on the rig), not frames; a live trunk opens a train/holdout gap at
~13% of a "frame epoch" while the frozen-trunk flow run closed 100k at
zero gap.

### Ledger — legacy (unfiltered) frame set (copy 10.30 holdout / 11.10 train / 9.54 rig-clean-256 / 11.39–12.92 rig-both)

| checkpoint | comm holdout | rig | note |
|---|---|---|---|
| mainline cont45k (frozen, ~27M samples) | 6.85 | 11.71 clean-256 / 18.76 both-256 | best frozen-trunk flow; still loses to copy cross-rig |
| unftext r2 15k / 25k (flow, text-lr 2e-5 from cont45k) | 6.47 / 6.49 | 18.18 / 17.97 both-256 | modest monotone win both sides; plateaued |
| rig ft `ft_marius_4k_init45k` @4k | — | 10.38 (h30: 10.10) vs copy 11.39 | first honest rig-holdout copy-crossing; init45k beat init40k by 0.22 (pretrain carries downstream) |

(Unexplained: rig-holdout copy reads 12.00 on the fps-30 ledger vs
11.39 on the ft round despite nominally identical frames — verify
frame-set identity before any cross-ledger rig comparison.)

### Findings that shaped the design

- **Scale dominates architecture.** The matched 4-arm ablation (causal
  vs bidirectional, 140 vs 280 soft tokens, 4-4-8 vs all-deepest;
  20k→40k, `docs/ablation_20k_results.md`) moved 0.6–1.2 MAE per
  doubling of steps; no variant separated from control where it counts.
  Kept: causal (bidir catastrophic cross-rig, 17.1 vs 13.3 — and the
  bidir adaRMS run's rig failure at scale echoes it), 140 tokens
  (280 = 1.9×/step, edge reversed), 4-4-8 (streams0016's rig edge is a
  hint, §8.4).
- **The wall is grounding, not action space.** Re-anchor probe:
  cross-rig error is a frame-dependent *level* mis-estimation (chunk
  shape is right; a per-frame oracle offset beats copy, no per-rig
  offset recovers anything) → the model mis-localizes the working point
  visually on unseen rigs. **Falsified delta-actions** (§8.9); pointed
  at the representation — and the AR-unftext result above is the first
  large confirmation that shaping the trunk attacks exactly this
  (community first_mae 2.06 vs copy 2.73; rig first_mae still behind,
  3.99 vs 2.76 — rig grounding remains the open front).
- **Acuity probe:** position is sharpest at the vision-tower output
  (8.4 px linear readout) and degrades through the LM stack (K4 10.8 →
  K14 17.3 px in-scene; 25–32 px cross-background); not object-centric
  (task-object motion ≈ 1.9× background at K14). The pool is exonerated;
  the text stack's use of visual tokens is the bottleneck — motivates
  trunk adaptation (§8.1) and conflicts with deepest-heavy schedules
  (§8.4).
- **τ-diagnostic:** ~1 MAE recoverable integration error in-domain; OOD
  cost concentrates at high τ (initial placement from context);
  fine-tuning roughens mid-τ transport; ft forgetting on community
  ≈ +0.85. Motivates adaRMS (§8.2 — whose Heun-gap collapse, −0.08 at
  10→30 vs additive's −0.28, confirmed the pre-registered signature)
  and sample-draws (§8.7).
- **Sampling analyses:** across-draw std ~5.9° exceeds single-draw
  error; mean-of-10 took a ft's 5.30° to 2.88° on motion frames — the
  largest known zero-training lever (§8.7). Fine-tuned fields want more
  integration steps (Heun-30 re-score −0.28 for +39% s/frame).
- **Fine-tuning is always done before deployment** — but its headroom
  is whatever the pretrain didn't bank (see the AR ft arc above). §8
  changes are judged by fine-tuned-then-scored rig MAE, with rollout as
  the final arbiter.

### Experiment reports

One block per mainline run: the exact setup (so it can be re-run), the
fit narrative including what went wrong, and the offline eval that
scored it. Ledger rows above are the index; these are the record.
Superseded runs stay in git history.

#### `bijou_arb_rcond_100k_ddp4` — request-conditioned ar_backbone, 100k (2026-08-03/04)

First run of the request-conditioned stack (prompt format 3, suffix
format 5, aux template v4 — §1/§2.3/§2.4). **Project best**; the
checkpoint is on the hub at
`mcobzarenco/bijou-checkpoints/bijou_arb_rcond_100k_ddp4/step_100000`
(backbone + expert + prompt + optimizer), reports in
`reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__*` (naming
rule: docs/working-together.md § Artifacts).

**Setup.** Code `68c762b` (eval-table split `d51a3b5` applied from 20k).
Corpus `community_curated_v0` @ `--fps 30 --camera-counts 1 2` ⇒ 878 of
981 datasets, 42,853 labeled episodes (annotation stamp `9b796de`,
judge opus-5), `--holdout-episodes 0.1 --split-seed 0`. Decoder
`ar_backbone` + `fast_tokenizer_v2`; aux fields subgoal, holding,
progress, event, visible; `--aux-dropout 0.0 --field-dropout 0.1`;
conditioning `--condition-fields subgoal outcome smoothness
--condition-dropout 0.1 --subgoal-dropout 0.5`;
`--instruction-augment 0.5 --camera-kind-dropout 0.1`. LRs
`--decoder-lr 1e-4 --backbone-text-lr 2e-5`, `--grad-clip 100`,
`--warmup-steps 1000`, 4×H100 DDP, `--num-workers 20
--prefetch-factor 4`, `--eval-samples 256 --eval-every 500
--save-every 2500 --log-every 20 --seed 0`. Batch 12 → **10** (see
below). Wall ~13.5 h; 0.46–0.49 s/step at B12, 0.42–0.47 at B10.
Verbatim (the `MALLOC_*`/`PYTORCH_CUDA_ALLOC_CONF` env is the standard
box preamble from `docs/init_gpu_machine.md`):

```sh
uv run torchrun --standalone --nproc-per-node=4 -m bijou.train \
    --train-data ~/datasets/mcobzarenco/community_curated_v0 \
    --fps 30 --camera-counts 1 2 \
    --holdout-episodes 0.1 --split-seed 0 \
    --decoder ar_backbone \
    --fast-tokenizer mcobzarenco/bijou-checkpoints/fast_tokenizer_v2 \
    --aux-fields subgoal holding progress event visible \
    --aux-dropout 0.0 --field-dropout 0.1 \
    --condition-fields subgoal outcome smoothness \
    --condition-dropout 0.1 --subgoal-dropout 0.5 \
    --instruction-augment 0.5 --camera-kind-dropout 0.1 \
    --decoder-lr 1e-4 --backbone-text-lr 2e-5 --grad-clip 100 \
    --steps 100000 --warmup-steps 1000 --batch-size 12 \
    --num-workers 20 --prefetch-factor 4 \
    --eval-samples 256 --eval-every 500 --save-every 2500 --log-every 20 \
    --seed 0 --wandb-project bijou-dev \
    --wandb-run-name bijou_arb_rcond_100k_ddp4 \
    --save-dir outputs/train/bijou_arb_rcond_100k_ddp4
```

and the eval that produced the table below (4-GPU sharded, 22m50s;
the earlier single-GPU 1024-frame read lives in
`reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__1024.*` — kept
because in-run probes were compared against it, superseded for
decisions):

```sh
uv run torchrun --standalone --nproc-per-node=4 -m bijou.eval \
    --data ~/datasets/mcobzarenco/community_curated_v0 \
    --episodes holdout --holdout-episodes 0.1 --split-seed 0 \
    --fps 30 --camera-counts 1 2 \
    --checkpoint outputs/train/bijou_arb_rcond_100k_ddp4/step_100000 \
    --num-samples 16384 --batch-size 24 --num-workers 8 --seed 0 \
    --report-samples 32 \
    --output-json reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__16384.json \
    --report reports/eval__bijou_arb_rcond_100k_ddp4__step_100000__16384.html
```

**Interruption.** Rank-2 CUDA OOM at step 20,160 (77.5 of 79.2 GiB, a
1.78 GiB allocation) — the pre-registered B12 gamble; `step_020000`
had just been written, so the run resumed there at B10 with the SAME
`--seed 0` (optimizer/scheduler restored; only batch composition
shifted). Effective batch 48 → 40 mid-run is a visible seam at 20k in
the wandb curves. Per the launcher's pre-registration, batch roulette
ends here: the structural fix for headroom is the output-vocab
shortlist head (§8.8-adjacent, queued), not another batch guess.

**Fit.** In-run `eval_chunk_mae` (256-frame probe): 31.4 @500 → 9.43
@5k → 7.54 @10k → 7.33 @20k → **6.57 @30k** (pre-registered gate
"clearly below 7.8 by 30k": passed) → 6.03 @40k → 5.79 @50k → 5.86
@60k → 5.60 @75k → 5.54 @90k → 5.55 @100k, best **5.29 @99.5k**.
Final loss 2.60 = action 2.51 + 0.5·aux 0.174; train_mae 4.41 vs
holdout 5.55 (healthy gap — the killed format-4 predecessor was
plateaued at 7.8–8.8 with a memorization signature). Run hygiene over
100k steps: **1** loader substitution, **0** value-budget fallbacks, 4
subgoal-truncation prints, no cuDNN asserts. Aux/conditioning
sub-metrics @100k: `chunk_mae_{success,partial,failure,unlabeled}` =
5.66 / 4.93 / 6.12 / 5.78; `samples_all_fields_mae` 5.70 vs 5.55
fast-path; `condition_sensitivity` 0.235 final (>0 all run = no
conditioning collapse, but it is a ≤12-row metric — read the trend,
not the point). `samples_holding_acc` never logged: the 12 fixed rich
rows contained no judge-sampled (holding-labeled) frame — fixed
offline by the full-sample aux metrics below; stratified rich-row
selection is queued for in-run.

**The 40.5k–45.5k excursion** (recorded because the mechanism is
reusable diagnostic knowledge). Eval-led and slice-localized:
`chunk_mae_partial` blew out 5.9 → **11.26 @41k** and stayed ~8.7
through 44.5k while success (6.15→6.45) and failure (~5.5) barely
moved; headline eval followed (6.03 → 7.41 @41.5k); rolling train loss
rose only +0.15 and **3.5k steps later** (44.1k); grad norms stayed
flat (6.1–6.8 in-window; post-30k mean 7.35, max 22.9 — clip@100
never engaged); `condition_sensitivity` peaked 2.28 @40.5k. Reading
(inference, not proof): a **repricing of what `[outcome|partial]`
implies**, which compounds autoregressively at decode — hence a large
eval move with almost no teacher-forced CE move, and no grad spike
expected under that mechanism. Not data or optimizer (no loader event,
smooth cosine, B-seam was 20k). It resolved: the partial slice
repriced back to trend. (The 1024-frame eval's "partial finished
best" read did not survive 16× the sample — see the correction below.)

**Offline eval** — 16,384 holdout frames, seed 0, identical
split/filters, 4-GPU sharded `bijou.eval` with the narrated pass
auto-enabled:

| policy | chunk_mae | p50 | p90 | first_mae | chunk_mse | ms/frame |
|---|---|---|---|---|---|---|
| state-copy | 10.882 | 8.24 | 22.98 | 2.557 | 450.8 | 0 |
| state-copy-norm | 10.813 | 8.14 | 22.92 | 2.369 | 450.0 | 0 |
| **bijou@100000** | **5.640** | **4.08** | **11.51** | **2.113** | **141.3** | 107 |
| bijou@100000+fields | 5.684 | 4.08 | 11.53 | 2.146 | 147.0 | 149 |

Paired: bijou vs copy **−5.018** mean Δ, p50 −2.544, **77% win rate**;
narrated vs fast path **+0.043**, p50 +0.000, 46% — self-generated
narration does NOT help the actions at this scale, but costs even less
than the 1024-frame read suggested (that +0.242 was small-n plus
batch-composition numerics; see `bijou/eval/sharding.py`). The fast
path stays the deployment default and the headline metric. Q2 slices
(true-label conditioning, bucketed by verdict): success 5.527
(n=12536), partial 5.994 (n=2458), failure 7.006 (n=498), unlabeled
5.500 (n=892) — **small-n corrections**: "partial best slice" (4.72 at
n=162) inverted at n=2458 (partial sits +0.47 above success — the
difficulty ordering is success < unlabeled < partial < failure, which
matches intuition: messier executions are harder to clone), and
failure's 6.712@n=25 firmed up to 7.006@n=498. Q3 sensitivity
**1.609** mean |Δ| over 2956 labeled non-success frames — conditioning
is decisively live offline. Aux vs the weak judge labels (narrated
generations, every labeled sampled frame): **holding accuracy 0.805**
(n=365 — exactly at the 0.807 inter-judge agreement ceiling; the
1024-frame 0.880@n=25 was a lucky draw), **progress MAE 0.066**
(n=365 — well inside the ±0.15 inter-judge noise). The report's
per-dataset table (869 datasets, median 13 frames each) puts the
worst residuals at `willnorris/bbox-2` (47.8 vs copy 111.1 — an
extreme-motion outlier both policies fail) and a tail of
hard-manipulation sets at 13–26; best sets sit at 1.5–1.8.

**Takeaways.** (1) The request-conditioned format trains stably
end-to-end and clears the format-4 plateau decisively — 5.64 vs copy
10.88 (−48%), with `first_mae` 2.113 **beating** copy's 2.557 (the
initial-placement measure that used to be the wall). (2) Narration is
free of action cost but of no action benefit; keep it for the aux
metrics. (3) Aux heads track the judge as closely as judges track each
other — the weak-label ceiling, not the model, is now the limit there.
(4) Still improving at 100k (75k→100k bought 0.05–0.3 depending on
smoothing) — longer runs remain the cheapest known win. Open flags:
the partial-slice repricing event class (watch for recurrence), ≤12-row
in-run conditioning metrics, FAST clip rate ~1.94% of chunks (≤4 of
300 coefficients; tokenizer predates curated-v0's exact quantiles —
refit queued).

## 8. Directions under evaluation and proposed changes

Each subsection: the change, its justification, status, and the key
design decisions. Full blow-by-blow lives in git history (these subsume
the retired `plan_*.md`). Smaller queued arms not detailed below:
`--trim-leading-idle` (~6.7% of frames are leading idle), a lerobot
policy plugin (`--policy.type=bijou`), suffix KV-caching for AR decode.
(State-noise augmentation shipped as `--state-dropout` — train-time
proprioception dropout against the state-shortcut failure mode.) Hygiene owed: rotate the wandb API key (it
reached a since-deleted box's shell history).

### 8.1 Trunk unfreezing (IMPLEMENTED; the live-trunk AR result is the headline)

**Change.** `--backbone-text-lr` trains E2B text layers 0–14 + the
multimodal projector (embeddings and per-layer-embedding tables stay
frozen — few rows/step, dense Adam waste, cheapest forgetting control);
`--backbone-vision-lr` adds the tower (expected to stay off — the acuity probe puts position
sharpest at the tower output, so adaptation is needed *downstream*).
**Justification.** The in-dist-vs-cross-rig gap and the acuity probe both
localize the bottleneck in the text stack's use of visual tokens, not the
expert; π0/SmolVLA both train their trunks. **Numerics/plumbing.** fp32
masters + bf16 autocast; `BijouTrainStep` + single DDP wrap
(static_graph); `backbone.safetensors` rides in the checkpoint; frozen
path stays byte-identical (oracle exact).
**Status/finding.** With the flow objective, text-lr 2e-5 from cont45k
was a modest monotone win (§7 legacy ledger). With the **AR CE
objective** the same flags produced the project's best results (§7:
ar_fast 5.96, then ar_backbone 5.656 comm holdout; first rig
copy-crossings) — next-token CE
shapes the trunk far better than flow MSE did, through the exported-K/V
pathway and natively in the decoder-only path alike.
LR regime is narrow: decoder at its native LR, trunk ≤2.5e-5; hotter
(3e-4) churns features faster than any decoder tracks and lands below
from-scratch. **Open dials.** freeze-then-thaw. (ZeRO-1 and
activation checkpointing both shipped — §5 memory machinery; the
flow-decoder-on-frozen-adapted-trunk attribution test executed as
stage-2, §8.11 — the win transfers.)

### 8.2 adaRMS time conditioning (IMPLEMENTED; signature confirmed)

**Change.** `--time-conditioning adarms`: DiT-style per-layer modulation
of the expert by τ. A per-layer zero-init head `SiLU → Linear(hidden →
6·hidden)` yields, per sublayer, a **scale** on the pre-norm output
(`rmsnorm(h)·(1+γ)`) and a **gate** on the residual contribution
(`residual + g·sublayer`); a final-norm scale head too. **No shift β** —
that is a LayerNorm artifact (LayerNorm has a bias to condition); RMSNorm
is bias-free by design, so the additive τ-injection is supplied by the
gate instead, and scale modulates the one thing an RMSNorm output is (a
magnitude). NOTE (verified against openpi source): π0.5's
adaRMS **keeps the shift** — `normed·(1+scale) + shift`, gated residual,
zero-init, applied at their pre-norm-only Gemma blocks. Our no-shift
design is a deliberate deviation, not a match (their gate has no
post-norm to act through; ours does). β's unique capability is injecting
a τ-dependent vector independent of current activations — one
composition step more direct than gate-through-state-token, plausibly
relevant exactly at high τ (where OOD cost concentrates). If the per-τ
mechanism probe on an adarms checkpoint shows the high-τ gap NOT
shrinking, a β arm is the targeted follow-up (+hidden per sublayer head:
229M→343M modulation at h1536). **Identity at init** (γ=gate=0 ⇒ every block is the identity ⇒
zero velocity field), so it is a from-scratch-only architecture (cannot
warm-start additive→adarms; the guard enforces this). **Justification.**
The τ-diagnostic's ~1 MAE integration gap + mid-τ roughening: the marginal
field changes character along τ, and per-layer modulation lets each block
reweight by τ rather than propagating one input-added vector through 16
residual blocks. **Cost.** +~101M (~+25%), few % step time (expert is
~20% of the step). **Pre-registered signature.** Shrinks the Heun-5→30
gap and the mid-τ bump; chunk-MAE −0.1..−0.4; rig zero-shot unchanged
(grounding ≠ conditioning). Param-confounded (win = modulation OR
capacity) until a bottleneck-head follow-up.
**Status.** Additive stays
default and byte-identical. The adarms lineage (bidirectional, h1536,
fps-30, 40k→100k) delivered the pre-registered signature — Heun-10→30
gap −0.08 vs additive's −0.28 — and the ledger's best flow numbers
(§7), but confounded (adaRMS + bidir + width + fps filter together),
and bidir's rig failure persisted at scale. A causal adaRMS-only arm
would isolate the conditioning effect. Forward-compat: if AR co-training
ever shares the expert (§8.3 option A), modulation must be MASKED to the
flow positions (per-position gating, not per-sample broadcast).

### 8.3 Autoregressive FAST decoding (SHIPPED — ar_backbone §2.3; ar_fast retired 2026-08-13, §2.2; tokenizer artifact story here)

**Shipped decoder.** `ARBackboneDecoder` (§2.3) decodes the action
phase under the FAST grammar mask — a valid sequence expands to
exactly chunk×dim quantized DCT coefficients, so there is NO EOA and
no malformed generations by construction (each step masks to tokens
whose BPE symbol-expansion fits the remaining budget; the mask
originated in `ARFastDecoder`, whose fresh-cross-attention variant
retired 2026-08-13 after ar_backbone superseded it — §2.2). Train:
`--decoder ar_backbone --fast-tokenizer <artifact>`; eval/rollout work
unchanged via `predict_chunk` (greedy — no Heun knobs); AR inference
additionally needs quantile stats (rides the checkpoint's per-dataset
table). Results in §7: ar_fast's frozen-trunk plateau ~8.0 at 10k
(≈ flow@40k quality); with the live trunk, the project's best
Gemma-era lines (ar_fast 5.96, then ar_backbone 5.656).
**Tokenizer:** owned DCT+BPE (`bijou/fast/`, arXiv:2501.09747), fit on
1040 datasets / 4.9M chunks. **Use `fast_tokenizer_v2`**
(`mcobzarenco/bijou-checkpoints/fast_tokenizer_v2`: alphabet 159 + 865
merges, 22.3 tok/chunk, 13.4×, recon 0.48° ≪ model error). v1 is
degenerate — min/max alphabet derivation let outlier coefficients eat
the merge budget (1019 base symbols + 5 merges, 53 tok/chunk); the fit
now bounds the alphabet by corpus-coefficient quantiles
(`--alphabet-coverage`) and hard-errors when merges would starve.
Artifacts are immutable (a refit changes token semantics); fit
normalization = per-dataset exact q01/q99 from `stats.json` (backfilled
by `ldtools`; lerobot's native quantiles average per-episode quantiles
and are wrong for corpus use); constant-dim guard + normalized clip
(parked joints have ~0 span). PI's published universal tokenizer
converts to our format (`outputs/convert_pi_fast.py`; ByteLevel
mid-character merges drop, ~24%) — a cross-embodiment ablation arm and
an independent cross-check of the implementation.

**Co-training resolution.** The once-proposed CE+flow mixture on one
backbone (π0.5/knowledge-insulation style) was overtaken by its own
"option C": the full-VLM path shipped as the standalone `ar_backbone`
decoder (§2.3) — trained with the deep half LIVE under
`--backbone-text-lr` rather than frozen-with-thaw-dial, which dissolved
the frozen-deep-half risk the option table worried about. What remains
of the co-training idea is the two-stage form: §8.11 (flow decoder on
the frozen AR-pretrained trunk). The literal joint-loss path has since
shipped (`--joint-ce` + `--seam-stop-grad`, §2.1) and a matched joint
arm was run and stopped on step cost (~4× the frozen arm), not
quality — the two-stage form stands.

### 8.4 Cross-attention stream schedule re-test

**Change.** Re-test `--stream-counts 0 0 16` (all cross-attention on the
deepest global layer 14) and a shallow-heavy variant (e.g. 8-4-4) at
scale. **Justification.** streams0016 had the best rig transfer in the
4-arm round (13.15, −1.01) despite trailing on holdout — a rig-transfer
hint; and the acuity probe found the *shallowest* stream (K4) carries the
sharpest position, which argues the opposite (shallow-heavy) — the two
hints conflict, so measure. Cheap: a config diff, no code.

### 8.5 Expert shape (width, cross-head count)

**Change.** Expert width 1024→1536 (E2B-matched); cross-attention head
count 8→4/6. **Justification.** Width is nearly free wall-clock (expert is
~20% of step) and capacity may bind at 27M+ cumulative samples (the 40k
ablation said it didn't at 2.56M). The 8 cross-heads are ~1/3 of expert
params and GQA-share E2B's single global K/V head — 8 views of one keyhole
has diminishing returns; 4–6 likely captures it, freeing params. But
retrieval is exactly where the measured problems live, so shrink only with
a matched arm. (Width 1536 shipped in the adaRMS lineage; the AR lineage
runs 1024.)

### 8.6 Backbone variants (EXECUTED — Molmo2-4B adopted)

The variant question was answered by a full trunk swap rather than the
Gemma-family arms sketched here (E4B for capacity, base-vs-IT): a
matched-topology screen against Molmo2-4B (§7, curated-plan ledger)
came out decisively for Molmo2, which is now the trunk of the current
training runs. The Gemma E-series compositions remain fully supported
and are the measured history of §7.

### 8.7 Inference-time noise-draw ensembling (SHIPPED)

`--sample-draws N` shipped in eval AND rollout (draws batched through
the expert, prefix encoded once; measured 576 ms for a mean-of-10
replan on the deployment laptop). It is the decode config of the best
banked checkpoint (§7: mean-of-10 and searched pinned-noise-vector
ensembles on the flow families) — while the same lever on the AR
families buys almost nothing (mean collapse, §7). Original
justification held up: mean-of-10 took a fine-tune's single-draw
5.30° → 2.88° on motion frames.

### 8.8 Throughput (prefix-side)

Length-bucketed batching shipped as `--bucket-by-length`; a
`kv_stop_layer` win landed earlier. Still open: `torch.compile` of the
frozen backbone (the backbone forward dominates the step — 79% on the
Gemma profile). Expert-side autocast is dead (expert is only ~20%).

### 8.9 Rejected / deprioritized

- **Delta-action targets** — falsified: the cross-rig error is
  frame-dependent visual level mis-estimation, not a state-relative
  offset the model can't express; per-dataset normalization + a state
  token already make state-copy ≈ the identity in normalized space, so
  delta is an affine reparameterization of information the model has. The
  deployment-smoothness benefit (chunk starts at the measured state) is
  better had by a ~5-line rollout-side anchor blend, no retraining.
- **280 soft tokens** — 1.9×/step for a reversed holdout edge and worse
  rig.
- **Symmetric bidirectional self-attention** — catastrophic cross-rig in
  the ablation; being re-tested only as one factor inside the adaRMS run.

### 8.10 Aux text tasks (IMPLEMENTED §2.4; first corpus-scale read-out in §7, paired attribution still owed)

**Status.** The full annotation stack shipped 2026-08-02 — five aux
fields (§2.4: subgoal/holding/progress/event/visible), prompt
conditioning (§4: camera kind tags, subgoal hint with anti-copy
coupling, hindsight outcome/smoothness), instruction augmentation
(rewrites + success-gated observed_task) — on the fully-materialized
curated corpus, superseded 2026-08-03 by request conditioning (the fed
mode token became the prompt's `[generate|…]` list; §2.3/§2.4).
Format-2 smokes on
rig v2 validated the aux mechanism (loss_aux 22.3 → 0.07, coherent
self-emitted subgoals, zero budget fallbacks; aux-less arm: BOA on
16/16). **Corpus-scale read-out: `bijou_arb_rcond_100k_ddp4` (§7
experiment reports; 16,384-frame numbers).** Measured there: aux
quality sits AT the weak-label ceiling (holding 0.805 at n=365 — the
inter-judge agreement is 0.807; progress MAE 0.066 vs the judge),
conditioning is live (Q3 1.609, n=2956), and narrating costs actions
nothing but buys them nothing either (+0.043 MAE, 46% win over 16,384
frames) — so aux is confirmed as free interpretability, which is what
the pre-registration asked.
**Value attribution STILL stays the paired experiment** (aux-on vs
aux-off fine-tunes from a common base on rig v2, matched
seed/steps/LRs, one variable at a time): the 100k run answers "does
narrating at inference help" but NOT "does aux supervision shape the
representation" — the recipe is deliberately multivariate, so that
claim needs the arms. Pre-registered: aux-on action
MAE within probe noise (±0.3) of aux-off at matched steps (aux as free
interpretability), holding accuracy approaching but not
exceeding the ~0.8 label-noise ceiling (met: 0.805 at n=365),
`condition_sensitivity` climbing off zero (§4's Q3 — ≈0 means
conditioning collapsed; met: 1.609 offline).
**Queued follow-ups:** un-gate `observed_task` rewrites for failure
episodes once Q3 validates; `[quality: …]` conditioning if smoothness
shows signal; next-subgoal aux (A2 — planning signal for chunk endings
crossing segment boundaries); camera-kind prediction under kind-dropout
(predict the true kind of `unknown`-tagged views — the C2 anti-copy
pattern applied to viewpoints); the self-conditioning rollout loop
(feed the aux-generated subgoal into the next replan's `--subgoal`).
**Owed alongside:** a wandb series for the value-budget fallback counter (log-grep
only today); "vision frozen" wording in the model-summary line.

### 8.11 Stage-2: flow decoder on the frozen AR-pretrained backbone

**Change.** Train a flow expert (§2.1) against the FROZEN backbone of
a decoder-only pretrain — the deployment answer if AR decode latency
ever binds (flow replans in a few hundred ms; AR is ~30–80 sequential
backbone forwards). **Status: EXECUTED, twice.** (1) On the Gemma
trunk via `--backbone-init-from` (warm start from the AR-pretrained
snapshot, unfreeze flags off): `bijou_flow_artrunk_h1024_40k_ddp2`
@80k is the **best banked checkpoint overall** (§7 curated-plan
ledger, 5.185 as a 10-draw ensemble). (2) On the Molmo2 trunk via
residual-stream adapters over the frozen decoder-only checkpoint —
trained 10k steps cleanly. A matched joint arm — the trunk continuing
its CE objective beside the expert's flow loss through a stop-gradient
seam — was stopped at ~4× the frozen arm's step cost; frozen-trunk
attachment is the working recipe, consistent with production systems
(RDT2, Qwen-VLA Stage I) that also train the expert against a frozen
trunk. **The Molmo2 attachment arm is RETIRED (2026-08-13, tag
`pre-decoder-simplify`)**: residual conditioning and its seam flags
(`--conditioning-streams`, `--seam-stop-grad`, the flow-side
`--joint-ce` wiring) were removed — `molmo_flow` (§8.13) supersedes
it as the flow-on-Molmo2 story, and its 10k checkpoint loads at the
tag. The BijouModel `joint_ce` slot survives dormant as §8.13 step
6's narration vehicle.

### 8.12 Multi-turn action context (K interaction pairs)

**Design sketch (chat history, 2026-08-02).** Format each chunk
generation as a (user, assistant) pair — observation turns and action
turns interleaved, ≤K pairs retained, Δt embeddings between pairs,
trained with a mixture of step offsets to match replan cadence. The
KV-sharing horizon makes it cheap: observation tokens never need layers
15–34 (no loss on them, nothing consumes prompt hidden states above 14
— the dead-half argument is per-token), so the interleaved forward is
layers 0–14 over everything, 15–34 over action-turn tokens only:
deep-half compute independent of context length, total step cost
≈2–2.5× single-turn at K=3. **Status.** Parked — design only; revisit
after aux value is measured.

### 8.13 `molmo_flow` — the MolmoAct2 action expert as a first-class decoder

**Change.** Adopt MolmoAct2's action-expert scheme wholesale as a new
decoder kind beside §2.1: a DiT expert (adaLN-Zero 9-way modulation,
RoPE+QK-norm self-attention, SwiGLU) whose block *i* cross-attends the
trunk's layer-*i* post-RoPE prompt KV through ONE shared bias-free
context projection — the whole prefix cache as conditioning, no
residual taps, no per-stream adapters, no exports. Their measured case
for the deep read: flattened per-layer KV beats final-hidden-state by
+1.9 LIBERO at ceiling; our executed-horizon analysis says our flow's
headroom is trajectory modeling, not grounding — a bigger,
deeper-conditioned expert is the right next arm. Hard requirement:
their released checkpoints load into our models (via conversion).
**Status: plan approved 2026-08-11 (chat). Step 1 SHIPPED (the CLI
checkpoint-inferred-flag rule: `TrainArgs.from_namespace` +
`__post_init__` single-encoding validation, `ARCH_FLAGS`/
`CheckpointTrainArgs` write/read sync test, verified flag-free against
both real rig-ft lineages — the AR checkpoint infers its implied
bracket, the stage-2 flow checkpoint resolves h1024/adaRMS). Step 2
SHIPPED (`bijou.convert_molmoact2` + the `molmoact2`/`molmo_flow`
section schema; gate run on the H100 box 2026-08-11: released
SO-100/101 and rig-ft rung-2000 both convert — 588/588 expert tensors
byte-equal source by independent re-read, mask flavors 'both'/
'continuous' captured, metadata round-trips, fp32 experts 2.31 GB;
converted dirs staged at `~/marius-convert-gate/converted/` on that
box for the step-3/5 gates). Step 3 SHIPPED (`decoders/molmo_flow.py`:
the owned architecture copy on the ascending convention, sampler with
Heun flagged, sum-form loss, recorded-t-law `TimeLaw`, the clamp tail;
flow.py convention-frozen with a cross-reference note; box gate
2026-08-11: forward ×3 timesteps + Euler-10 loop BYTE-EQUAL vs the
port on both real checkpoints' weights — transitively byte-exact vs
their HF module via the port's G1). Step 4 SHIPPED
(`encoders/molmoact2.py`: `MolmoAct2InputsCollator`/`MolmoAct2Encoder`
— assembly owned, leaf transforms imported from the golden-pinned
port; per-row assembly byte-equals `pack_action_example`; both
split-point layouts pinned, off-ids == on-ids + `<action_output>`;
`conditioning_mask` rides `ObservationMemory` carrying the
`action_mode` flavor while `attention_mask` keeps counting EOS — the
positions source is untouched; uint8 TRUNCATION coercion serves both
train and inference, zero train/serve skew; the prompt section gained
the `narration` split-point field). Step 5 PART 1 SHIPPED (eval-side
integration): `from_checkpoint` assembles converted checkpoints
(`build_molmo_flow_decoder` bridge; expert compat injection; no
prompt.safetensors — the encoder has zero parameters), BijouModel
grew the molmo_flow arms (loss/sums/normalizers, predict dispatch with
per-kind operating-point defaults, `retain_cache`, the
`insulate_expert` KI run property), and the KI gradient contract is
test-pinned both ways (insulated ⇒ trunk grads exactly zero; open ⇒
nonzero through the cached K/V, composing with activation
checkpointing — the fixture expert perturbed off zero-init, where the
open-seam test is provably vacuous). **E2E parity gate GREEN** (box,
2026-08-11, 240 anchors × both arms, bf16): OURS vs the PORT live =
**0.0 pooled, 0.0 max** — byte-identical end-to-end (collation →
encode → KV extraction → flow loop → tail) — and vs the banked HF
npz exactly the port's own kernel floors (released 0.041 / rung
0.0541 ≤ 0.075), anchor MAEs reproduced (28.9456 / 3.2321). One real
bug caught by the gate and fixed under diagnosis: the q01/q99 clamp
table briefly lived in buffers and was swept to bf16 by the
deployment `module.to()` — denorm constants rounded to ~3 significant
digits (measured 0.027/0.119 pooled divergence) — now plain fp32
tensors, data not weights. Step 5 PART 2 SHIPPED (train wiring):
molmo_flow is INHERIT-ONLY at the CLI (never a --decoder choice; it
resolves from an --init-from/--resume checkpoint per the step-1 rule —
from-scratch stays step 7); `--insulate-expert` (refused with an
unfrozen trunk until the step-6 CE rider); the shared Collator grew
the merged-table state mode (q01/99-clamp, their scheme; None keeps
every existing path byte-identical); model build from source sections;
the init block loads via `load_expert_state` and tolerates the absent
prompt.safetensors iff the encoder is stateless; the save side writes
the MolmoAct2 prompt section and SPLICES the in-use q01/q99 tables
into the normalization row (the run aggregate honestly carries none —
without the splice a descendant checkpoint would lose its clamp).
**GPU train smoke GREEN** (box 2026-08-11: 40 steps, batch 8, frozen
trunk, --init-from the converted rung-2000 on the 2 rig repos — loss
opens at 0.0132, exactly the reference run's endpoint band (their
0.135@20 → ~0.008@2000 corridor), grad norms ≤3, 0.31 s/step, 24.2
GiB peak; async save landed and the written checkpoint ROUND-TRIPS:
from_checkpoint reload → predict works, rig tables verbatim in the
saved normalization row). Two integration bugs caught by the smoke and
fixed: the frozen-trunk `state_proj` freeze assumed every encoder has
one, and the launch banner lacked the encoder arm. Remaining for
§8.13: the rig-rung repeat (gate d, ≤6 GPU-h — the pre-registered
2k-step recipe through bijou.train), then step 6 (narration).**

**STEP-8 CLOSURE (2026-08-14, docs/molmoact2-retirement.md executed
phases 0–5).** The port package `bijou/molmoact2/` is DELETED. Its
discrete head is first-class: `MolmoAct2ARDecoder`
(`decoders/ar_molmoact2.py`) — the third `ARSuffixDecoder` concrete,
decoder kind `ar_backbone` at suffix format 6, ZERO own parameters
(trunk-native rows; `MolmoAct2ActionCodec`'s −2/−1 special offsets
make the scaffold arithmetic land on `<action_start>`/`<action_end>`
with `block_base = action_token_start_id` — capture stays [B, 2048]
block-relative). Byte-parity vs the port's decode gated on the frozen
fixture (ids/bins/actions equal ×6, logprobs 2.4e-7). The molmoact2
family trains under `--objective {flow, ar, joint}` (+
`--joint-ce-weight`, `--expert-init {inherit, fresh, <ckpt>}`) with
decision-5 ordering (flow KV extracted before the CE append) and KI
test-pinned both ways; CPU anchors in §5's regression gates. GRPO
rides `bijou/grpo_replay.py` (`MolmoAct2DiscreteStack` + the thin
replay; row NPZ + loop `.pt` formats frozen) — frozen-wave replay
gate PASSED on both R1 banks (masks bit-equal 1904+904 rows;
port-vs-first-class ≤5.7e-5 under the re-baselined 1e-4
cross-decomposition bound; fresh v2 wave through the new driver ran
end-to-end). Full record: docs/molmoact2-retirement.md.**

**Decisions (register).**

1. **Naming**: `decoders/molmo_flow.py`, `MolmoFlowDecoder`,
   `MolmoFlowConfig`, checkpoint wire kind `"molmo_flow"` — named for
   the ARCHITECTURE's provenance (there are many ways to build a
   KV-conditioned expert; this is theirs). Not a trunk requirement:
   the module's contract is trunk-neutral — N post-RoPE (K, V) pairs
   `[B, S, kv_dim]` + prompt mask, N == block count, one uniform
   `kv_dim`. Molmo2 (36 uniform full-attention layers, kv_dim 1024) is
   the only wired producer; a gemma4 variant is possible-not-built
   (KV sharing leaves only 15 real surfaces, mixed 256/512 kv_dim,
   sliding windows — the few-stream §2.1 design is what "read the
   trunk's KV" collapses to on that architecture).
2. **Time convention**: ascending t (0 = noise → 1 = data, target
   x − ε, left-endpoint Euler 0→1, t ~ 0.001 + 0.999·Beta(1, 1.5))
   is the repo standard for ALL new flow code. `flow.py` keeps its
   name and its π0 convention (τ = 1 − t, target ε − x, integrate
   1→0), frozen, retiring with its checkpoint lineage — flipping it
   in place would break every deployed flow checkpoint or plant a
   sign-shim landmine. Known escape hatch, documented not built: an
   EXACT weight-space converter (sinusoid features of 1−t are a fixed
   per-frequency rotation absorbable into `time_in_proj`; the sign
   flip absorbs into `action_out_proj`; same for φ_s). The two τ-laws
   are near-mirrors with opposite endpoint asymmetry (ours includes
   pure noise and excludes pure data; theirs the reverse) — adopted
   as-is, pinned by a convention-direction test so nobody "fixes" it.
3. **Parallel copy, not delegation**: `bijou/molmoact2/` stays
   untouched as the frozen parity reference; the decoder owns its
   architecture copy (it must be free to grow — narration
   conditioning, horizon, Heun, φ_s). A byte-parity oracle (same
   weights into both modules → byte-equal forwards) pins them while
   both exist.
4. **Conversion-first loading**: runtime never reads their HF layout.
   A converter emits a normal bijou checkpoint: trunk verbatim
   (backbone always present — never pristine), expert section with
   tensor names preserved 1:1, q01/q99 table + horizon + setup/control
   strings into metadata, THEIR tokenizer in-checkpoint (it re-homes
   image ids and carries the state-token block), the `'both'`-mode
   EOS-strip mask flavor (load-bearing for the weights — the expert
   trained with BOS/EOS stripped from its context), `n_obs_steps == 1`
   and dropout-zero asserted at convert time, full provenance.
5. **Prompt format is an encoder mode**, `{bijou, molmoact2}`,
   checkpoint-recorded. molmoact2 mode = their verbatim template,
   256-bin discrete state tokens, their UINT8 single-view 378×378
   image path (double quantization is part of the trained
   distribution — not our molmo2 crops pipeline), special ids resolved
   from the checkpoint tokenizer. bijou mode stays available for
   from-scratch runs on our trunks (the decoder never sees the prompt,
   only the cache).
6. **Normalization is decoder-owned** for this kind: checkpoint-
   resident q01/q99 buffers, clamp on input and output (their tail).
   The per-dataset mean/std machinery is untouched for every other
   decoder; fine-tunes recompute ONE merged table (their semantics,
   rig-ft precedent). Accepted: loss values are not comparable across
   decoder kinds; eval MAE (raw units) is.
7. **Narration = the `joint_ce` rider, format-aware.** OFF (default,
   converted checkpoints): `<action_output>` sits in the prefill and
   the expert reads the prompt-only KV slice — byte-for-byte their
   serving path (the parity gate). ON: the prefill stops at the ChatML
   opener (== `MOLMO2_GENERATION_OPENER`, no new contract), aux text
   decodes as suffix, `<action_output>` appends, and the expert
   conditions on the EXTENDED slice — teacher-forced at train,
   generated at inference, so both see the same kind of context.
   FAST-token spans are always masked out of the expert's context
   (their own `'both'`-mode span-strip concept). Narration on their
   trunk requires trunk training — it never narrated — which composes
   as their own staging: knowledge insulation on the flow gradients,
   CE flowing to the trunk (the discrete co-training rider's analog).
8. **Knowledge insulation is a seam flag**: `--insulate-expert`
   detaches the extracted KV before the expert (their post-train);
   off = expert gradients reach the trunk THROUGH the cached K/V
   (their finetune; the checkpointed-block cache machinery already
   carries gradients through the cache). Same pattern as
   `--seam-stop-grad`. `ar_molmo2.py` needs zero changes — the
   `joint_ce` slot already composes an AR CE rider with a training
   decoder.
9. **CLI inferred-args rule** (general, built first): `--resume`
   refuses every architecture-determining flag (run-policy flags —
   steps, LRs, `--rewarmup-steps`, fresh `--seed`, cadence — stay
   legal); `--init-from` refuses flags for inherited sections and
   REQUIRES them for explicitly replaced sections (the stage-2
   decoder-swap path). Sentinel `None` defaults + one reviewable
   arch-vs-policy partition table. Upgrades "validate equality if
   passed" to "refuse at the door"; the prompt-format-change guard
   falls out as a special case.
10. Behind flags, zero-cost at parity: Heun, K>1 noise draws per
    chunk (their monotone +1.75 ablation; batched over one KV
    extraction), continuous-state reconditioning via their dormant
    `state_encoder` path. Out of scope: depth gate, their discrete
    head, CUDA graphs.

**Steps (each gates the next; every step lands green through
check.py).**

- **1 — CLI inferred-args rule.** Gate: resume-with-arch-flag and
  inherited-section init-from flags error naming the checkpoint value;
  decoder-replacement init-from still works; a flag-free resume of a
  mainline checkpoint parses unchanged.
- **2 — Converter** (pure-CPU, deterministic, idempotent, P2 guards
  at convert time). Gate: released SO-100/101 + one rig-ft rung
  convert; trunk tensors byte-verified; metadata round-trips through
  `bijou.loading`.
- **3 — Decoder module** (config with released shape as staticmethod,
  copied architecture with load-compatible tensor names, ascending
  loss + sum form for chunked backward, Euler sampler with Heun
  flagged, clamp+unnormalize tail). Gates: byte-equal forward vs
  `bijou.molmoact2.ActionExpert` on shared random tiny weights (CPU)
  and released weights (script); solver-loop equality vs
  `wiring.generate_actions` under a shared generator; the
  convention-direction test.
- **4 — Encoder mode** (molmoact2 template, discrete state, uint8
  image path, dynamic ids, mask flavor, the PREFILL SPLIT POINT —
  narration-off through `<action_output>`, narration-on through the
  opener — `retain_cache` for the new kind, empty streams). Gate:
  collator byte-equals `pack_action_example` on the golden fixtures;
  both split-point byte-layouts pinned.
- **5 — Integration** (BijouModel match-arms, loading schema,
  bijou.train: expert-only default group / `--insulate-expert` /
  q01q99 flow / joint_ce slice wiring; eval + rollout dispatch;
  draws-ensembling tiles the cache). Gates in order: converted
  released checkpoint vs `MolmoAct2Predictor` on the 240 banked
  anchors — expect 0.0, budget ≤ 0.075; 2-step training-loss corridor
  vs `molmoact2/train.py`; KI gradient tests both ways (insulated ⇒
  flow-term trunk grads exactly zero; uninsulated ⇒ nonzero through
  cached K/V, composing with activation checkpointing); rig-ft rung
  repeat through bijou.train reproducing the G4 result class
  (≤ 6 GPU-h).
- **6 — Narration on** (flip slice + split point; CE via the rider;
  `aux_loss_weight` pre-registered — CE nats vs clamped-MSE are
  different units, decide the balance up front). Gates: narration-off
  bitwise unchanged from step 5; narration-on smoke + eval aux
  columns.
- **7 — Experiments** (separate pre-regs, post-migration): their-init
  vs from-scratch on our corpus; molmo_flow-from-scratch on our AR
  trunk + bijou prompt vs §2.1 at matched compute (577M-vs-367M
  params confound handled); K ∈ {1,4,8}; KI on/off at unfrozen trunk;
  narration-as-conditioning value; horizon-50 extension (RoPE
  extrapolation — an experiment, not an assumption); SnapFlow distill
  of molmo_flow.
- **8 — Retirement checkpoint.** After a shipped rung:
  `molmoact2/train.py` first; `predictor`/`processing`/`action_expert`
  stay until the parity scripts migrate; then the package folds and
  the decoder's copy is the only implementation.

**Known consequences.** Two mirrored flow conventions coexist until
`flow.py` retires (mitigated: no shared code, different variable
vocabulary — `tau` vs `t` — convention notes at both module tops, one
direction-pinning test per kind). Laptop rollout of converted
checkpoints is GPU-infeasible at 8 GiB (4.9B trunk + 578M expert bf16
≈ 11 GB, no PLE to offload) but works on CPU: measured 66–69 s/replan
on the physical rig (Core Ultra 9 185H, bf16 trunk + fp32 expert,
Euler-10, 2 cameras, sync loop) — a validation mode, not a control
mode (~1 s of motion per replan at chunk 30). Deploying the released
checkpoint on a post-PR#777-calibrated arm (lerobot ≥ 0.5) requires
`--joint-frame v30-to-v21`: the global q01/q99 table bakes in the
pre-0.5 degrees frame, so rollout remaps state/actions at the robot
boundary and gates the first observation against the checkpoint's own
state band in model frame (docs/rollout_so101.md). Horizon 30 vs 50
makes matched-window reporting a first-class eval flag. Checkpoint schema grows (new kind,
embedded q01/q99, prompt-format + mask-flavor fields, foreign
tokenizer in-checkpoint). Estimated cost (estimate, not budget):
steps 1–6 ≈ 5–6 focused sessions, ≤ 10 GPU-h, dominated by step-5
gates.
