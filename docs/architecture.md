# Bijou architecture

Bijou is a vision-language-action model for SO-100/101 arms built on
**Gemma-4 E2B-IT**, with one prompt side and two trained action-path
families. The **cross-attention family**: a truncated backbone (layers
0–14) encodes camera images + a language instruction once per
observation and exports the K/V of a few of its layers; a **404M fp32
flow-matching action expert** (or a narrow AR FAST decoder) cross-attends
that K/V and emits a 50-step action chunk. The **decoder-only path
(`ar_backbone`)**: the FULL backbone plays both roles — the same prompt
is prefill-encoded once, then the backbone itself continues the suffix
autoregressively into FAST action tokens (and, when trained with judge
annotations, an auxiliary text segment first — subgoal / holding /
progress) under full-vocabulary CE through its own tied LM head.
Per-dataset normalization makes ~1000 miscalibrated community rigs
trainable together; the ultimate target is the owner's physical SO-101.
The decoder-only path holds the current ledger best (§7).

This document is the deep reference for the model and training system as
they exist, the measured results that shaped them (§7), and the proposed
changes under evaluation (§8). Per-module contracts live in docstrings;
code conventions in `code-styleguide.md`; collaboration/operating
conventions in `working-together.md`. Transient state (in-flight runs,
machine inventory) lives in wandb, the HF hub, and the chat — not in
Package layout (strict downward-only imports):
`train`/`eval`/`rollout`/`judge` → `loading` → `model` →
`encoders`/`decoders` → `interface` → `gemma4` (`data` beside `model`,
imported by `loading`; `judge` touches only `data`; `aux_text` is a
leaf beside `gemma4`, imported by `interface`/`data` and above), with
`interface.py` as the encoder×decoder seam
and `model.py` the composition root: `BijouModel` owns the backbone
ONCE and composes a prompt-side encoder strategy (which receives the
backbone as an argument) with an action decoder — one network serves
several roles (prefix encoder for the cross-attention decoders; prefix +
suffix runner for the decoder-only path). The root also owns the
objective dispatch (`BijouModel.loss` / `loss_components`) and the named
trainable-group routing (`param_groups`: decoder / backbone_text /
backbone_vision).
Naming: **backbone** is the one identifier for the Gemma network — the
pretrained artifact (`--backbone`, `BackboneConfig.id`,
`backbone.safetensors`) and the mounted module (`model.backbone`) alike;
"trunk" survives only as informal prose.

```
[instruction][cam_1]..[cam_k][instruction]     chat-templated user turn
      │  E2B prefix: layers 0..14 (bf16), LEFT-padded batches
      ▼
  prefix K/V — ObservationMemory, encoded once per observation
      │
      ├─ cross-attention family: exported GLOBAL streams {4, 9, 14}
      │    FlowDecoder (404M fp32), 16 layers, each =
      │      cross-attn(one stream) → self-attn([state][a_1..a_50]) → MLP
      │      → velocity at flow time τ → Heun integration τ: 1 → 0
      │    (ARFastDecoder: same blocks, FAST tokens, constrained greedy)
      │
      └─ decoder-only (ar_backbone): the FULL E2B continues the suffix
           [state][<start_of_turn>model\n][MODE][aux text?][BOA][t_1..t_k]
           through all 35 layers against the retained prefix cache;
           ~11M new params; fed [ACT]|[AUX] mode commands speak-vs-act;
           free-until-BOA under [AUX], then FAST-grammar decode
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
- `hidden` — model hidden size (backbone 1536; expert 1024/1536)
- `head_dim` — per-attention-head dimension
- `heads` — query attention heads; `kv_heads` — key/value heads
  (`kv_heads ≤ heads` under GQA)
- `intermediate` — MLP intermediate (GLU) size
- `action_dim` / `state_dim` — action / state dimensionality (6/6 here)
- `time_embed_dim` — sinusoidal time-embedding dimension
- `vocab` — token vocabulary size
- `num_layers` — decoder layer count (the PLE per-layer axis)
- `ple_dim` — per-layer-embedding dimension (`hidden_size_per_layer_input`)

Inline literals where an axis is a fixed small constant: `2` = the (x, y)
spatial pair in `image_position_ids`; `head_dim/2` = RoPE inverse-freq
length; `3·patch_size²` = a raw-RGB vision patch row.

## 1. Prompt side — the Gemma-4 E2B prefix encode

**One backbone, mounted at a depth.** `BackboneConfig.depth ∈ {prefix,
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

**Prompt = instruction sandwich** `[task][cam_1..N][task]` in one
chat-templated user turn, **LEFT-padded** across a batch with per-sample
LOGICAL position ids (cumsum of the real-token mask). Left padding is
load-bearing (decided 2026-08-01, test-gated in
`tests/test_backbone_continuation.py`): Gemma's sliding-window masks are
physical-index, so right padding puts a suffix appended after the batch
max at DIFFERENT physical distances per sample and silently corrupts
windowed attention for any suffix continuation; with left padding every
sample's suffix is physically adjacent to its real prompt. Correct for
the cross-attention consumers too (they read positions from the mask).
Under causal attention the sandwich yields instruction-conditioned image
K/V *and* image-conditioned instruction K/V for a few extra tokens.
Camera NAMES are positional slots (sorted); community image/image2 keys
carry no reliable wrist-vs-scene semantics (SmolVLA precedent), so slot
order is the only camera signal.

**What the decoders consume.** The cross-attention family reads the
exported streams of an `ObservationMemory`; `ar_backbone` additionally
retains the FULL prefix `KVCache` on the memory (`retain_cache=True`,
set by `BijouModel.encode` from the decoder kind — the exported streams
are zero-copy views into it) and extends it in place while decoding.

**Vision geometry** (encoder-free E-series tower, 768 hidden, 16-px
patches, 3×3 spatial pool): a 640×480 frame → resized 624×480 → 39×30
patches → **130 soft tokens** (13×10, one per 48×48-px cell), under the
140-token/camera budget. Prompt ≈ 292 tokens for 2 cameras; padded
batches reach ~452. The acuity probe (§8) found position is *sharpest at
the tower output* (8.4 px linear readout) and degrades through the LM
layers — the pool is not the bottleneck; the text stack's handling of
visual tokens is.

## 2. Action decoders

Three decoder kinds share the seam (`--decoder flow | ar_fast |
ar_backbone`); a checkpoint's `decoder.kind` tags which one it carries.

### 2.1 Flow-matching expert (cross-attention)

A narrow decoder over the suffix `[state][a_1..a_50]` (`suffix_length =
1 + chunk_size`). Default shape: **hidden 1024, 8 self-attn heads
(head_dim 128), intermediate 4096 (GLU), 8 cross-attn heads, 16 layers,
~404M fp32**. Freshly initialized, never loaded from the backbone.

Each `SuffixBlock` (`bijou/decoders/blocks.py`, shared with ar_fast) is
a Gemma-style sandwich of three sublayers, each
`residual → pre_RMSNorm → sublayer → post_RMSNorm → +residual`:

- **Cross-attention** over one exported stream. Queries adopt the
  backbone's global geometry exactly: `head_dim 512`, q-RMSNorm, p-RoPE
  continuing at positions after each sample's REAL (unpadded) prefix
  length, scaling 1.0. GQA against the stream's single K/V head. The
  per-layer stream assignment is `cross_attention_schedule`, default
  blocks **4-4-8** (4 layers on stream 4, 4 on 9, 8 on 14 — deepest-heavy,
  since layer 14 is the one the backbone's own deep half consumes). Its
  length is the expert depth; cycle/hybrid schedules are config diffs.
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
implemented and under evaluation (§8.2).

Params live ~50% in the MLPs, ~33% in cross-attention (8 heads × 512
over the residual 1024), ~17% in self-attention.

### 2.2 AR FAST decoder (cross-attention)

`ARFastDecoder` (`bijou/decoders/ar_fast.py`): the same sandwich blocks
over suffix `[state][BOA][t_1..t_k]`, fully causal, teacher-forced CE
over the FAST token vocabulary (state/PAD positions ignored), greedy
decode constrained by the FAST grammar. Tokenizer artifact story and
results: §8.3.

### 2.3 Decoder-only path (`ar_backbone`)

`ARBackboneDecoder` (`bijou/decoders/ar_backbone.py`): the FULL backbone
is the decoder — the prompt is prefill-encoded once (layers 0–14, cache
retained), then the suffix runs ALL 35 layers against that cache, and
next-token logits come from the backbone's own frozen tied LM head over
the **full vocabulary**. There is no separate decoder network; the
module owns only **~11M new parameters** at E2B scale:

- `state_proj` — the normalized state enters as suffix position 0 via a
  **zero-initialized** linear projection (an inert token at init: the
  prompt-conditioned computation starts undisturbed, gradients flow
  through its K/V use). State stays out of the VLM *prompt* (π0 layout,
  as in §2.1) — the prompt cache stays replan-reusable and
  in-distribution.
- `fast_embed` / `fast_ple` — input-embedding and per-layer-embedding
  rows for the FAST block, scaled like the backbone's own tables (√dim).
  Warm-started around the real tables' row mean + 0.02 noise
  (`init_tables_from_backbone`) so block logits start near the average
  text logit under full-vocab CE.

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

**Suffix format 3** (`aux_text.SUFFIX_FORMAT`, recorded per checkpoint;
formats 1–2 = legacy, loadable for warm starts — the mode tables
fresh-init loudly — and decoded via their own trained sequence with a
loud warning):

    [state][<start_of_turn>model\n][MODE][aux text — iff AUX][BOA][t_1..t_k]

The opener is the IT chat template's own generation prompt (format 2's
contribution). `MODE ∈ {[ACT], [AUX]}` is **fed, never predicted**:
two reserved ids directly below the FAST block (`mode_base =
block_base − 2` — E2B: 261116/261117, same unused tail), embedded
through their own tiny patch tables. A row feeds [AUX] iff it carries
aux supervision (label presence × aux dropout, decided at collation),
so "speak vs act" is COMMANDED, never inferred: the model is never
asked to predict from appearance whether a judge happened to label a
frame, and both conditionals — p(actions | ACT) and p(aux, actions |
AUX) — train on their own sample mass. Teacher-forced full-vocabulary
CE: state, opener and the fed MODE position are IGNOREd; the MODE
position's own logits are the trained transition (first aux token
under [AUX], BOA under [ACT] — aux-less runs feed [ACT] everywhere);
PAD is batch padding, always ignored; no EOA (action length is fixed
by the FAST grammar). Loss components: `total = action + w·aux` with
per-position mean CE split by the collator's aux mask
(`--aux-loss-weight`, default 0.5); aux is logged as a
position-weighted mean (CE sum / token count, all-reduced), so
sparsely-labeled corpora don't dilute `train/loss_aux` toward 0.

**Decoding is ONE loop with two entry modes** (`predict_chunk`):

- `ACT` — feed `[state][opener][ACT][BOA]` in a single prefill (BOA is
  the only trained continuation of [ACT], so it is fed, not sampled),
  then the ACTION phase under the ar_fast grammar mask (each step
  masks to tokens whose BPE symbol expansion fits the remaining
  chunk×dim budget; a full-length chunk is guaranteed by
  construction). The deployment fast path — and a TRAINED context,
  unlike format 2's force-BOA (measured OOD 18.2 vs 13.5 on the
  aux-saturated smoke arm).
- `FREE` — feed `[state][opener][AUX]`, then the FREE phase where only
  text ids and BOA are legal (mode ids and the rest of the FAST block
  are masked) under a `MAX_FREE_TOKENS = 48` budget — exhaustion
  forces BOA, loudly, and increments a cumulative `fallback_count` (a
  persistent rate means the model stopped closing its aux segment) —
  then the ACTION phase. Conditioned on [AUX] the model has never seen
  immediate BOA, so the aux fields come out at every frame — not at
  the labeled-frame frequency. FREE on an aux-less checkpoint is a
  loud error ([AUX] is untrained there).

Never compare numbers across decode modes — they are different
measurement conditions by design.

Prompt-side geometry is what makes the suffix exact: left padding +
logical positions (§1) put every suffix token physically adjacent to
its sample's real prompt, positions continuing after each sample's real
prefix length; the state slot borrows the pad token's PLE row (the
precedent set by image soft tokens).

### 2.4 Auxiliary text tasks (`bijou/aux_text.py`)

Trained text outputs rendered from the LLM-judge annotations
(`docs/episode-annotations.md`), emitted before BOA in the format-2
suffix:

    subgoal: reach toward the toy boat\n
    holding: no\n
    progress: 30%\n

- **Presence-based rendering, mode-conditioned.** A field appears iff
  its label exists at the frame: subgoal on every frame of a judged
  episode (piecewise-constant `language_persistent` rows);
  holding/progress only on judge-sampled frames (the finite mask IS
  the sampled-frame set — never interpolated). A sample with ≥1
  rendered field feeds [AUX]; unjudged/dropped samples render nothing
  and feed [ACT] — mixed sparsely-annotated corpora train one format,
  the mode explains label presence away, and an aux fine-tune extends
  a pretrained base rather than fighting it.
- **Mode dropout** (`--aux-dropout`, default 0.1 on aux runs): a
  labeled sample trains as [ACT] with probability p — keeps the
  deployment fast path trained even at 100% annotation coverage.
  Draws are per-visit from a generator seeded by the dataloader worker
  seed (pure function of --seed, rank, worker); the probe-side
  collator runs a dropout-0 clone so eval tables always show true
  labels.
- **Field set and order.** `--aux-fields` selects a subset of
  {subgoal, holding, progress} but never reorders (template order is
  validated at the CLI boundary and re-guarded in `AuxSpec`); subgoal
  text is truncated at `max_subgoal_tokens` (16), loudly.
- **Template versioning.** `AUX_TEMPLATE_VERSION` (2) rides in the
  checkpoint's decoder section (`AuxDecodeConfig`: version, fields,
  prompt hash, judge model); loading a version this code doesn't know
  is a loud error — a byte-level header change on an existing
  checkpoint would silently break elicitation, so headers change only
  with a version bump.
- **Label provenance is pinned.** `aux_text.PINNED_PROMPT_HASH` is a
  literal (the import DAG points judge → data, never the reverse),
  test-asserted equal to `bijou.judge.PROMPT_HASH` — a judge-prompt
  change fails check.py instead of silently mixing label
  distributions. At selection time, `data.verified_annotation_stamp`
  admits a dataset's annotation surfaces only when its
  `meta/judge_annotations.json` stamp matches the pin; stale/absent
  stamps train as unjudged, loudly. `--aux-fields` with zero verified
  datasets is a startup error.
- **Id spaces.** Aux ids are ordinary text-vocabulary ids — the
  full-vocab head exists for exactly this; the collator
  (`assemble_suffix`) builds one mixed suffix tensor in backbone id
  space (guarded: aux ids must sit below `block_base`) plus the
  aux-position mask the loss splits on.
- **Metrics.** Component losses `train/loss_action` / `train/loss_aux`
  (the aux mean is position-weighted across batches and ranks);
  in-run eval logs the FREE-decode generations as raw-string columns
  (`aux_generated` vs `aux_label`) inside the `eval/samples` wandb
  table — the table's chunks and text come from the same decode
  (self-consistent rows), while the scalar MAE probes score the ACT
  fast path (comparable across aux-on/off/less arms) — and
  `eval/samples_holding_acc`: teacher-forced likelihood accuracy
  p(yes) vs p(no) at the holding value position under the label
  context ([AUX] + preceding trained fields; sparse fields appear in
  free decode at every frame now, but likelihood scoring stays the
  clean per-field measurement). Labels are weak supervision (~80%
  inter-judge agreement on holding, ±15% progress MAE): weight
  modestly, expect an accuracy ceiling near the label noise.
- **Owed:** the offline `bijou.eval` report has no aux section yet
  (generations + aux metrics in HTML/JSON); in-run wandb is the current
  surface.

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
are 30fps).

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

**Robustness.** Workers spawn (never fork — torchcodec is fork-unsafe);
decoder cache capped (default OOMs hosts); `tolerance_s = 0.5/fps` (v3
concatenated files break 1e-4 tolerance); corrupt community videos are
substituted with a far index from the same dataset, loudly, with bounded
retries (two such runs were killed before this guard).

## 5. Training system

DDP via torchrun, per-rank batch/workers; loss and MAE probes all-reduced,
checkpoints/logging on rank 0. Optimizer AdamW (0.9, 0.95), fused on CUDA
(CPU keeps the reference path so the loss oracle is stable), cosine
schedule to 10% after linear warmup, grad-clip 10.0 (a never-bites safety
net; a tighter clip renormalizes most steps and injects moment noise) —
except ar_backbone, whose full-vocab CE runs larger grad norms:
convention is `--grad-clip 100` there (step-1 norms in the 10^4 range
are softcap saturation, clipped in practice).

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

**Decoder-kind CLI semantics.** flow accepts the `--decoder-*` shape
flags and `--stream-counts`; AR decoders require `--fast-tokenizer
<artifact>`; **ar_backbone rejects the shape flags and stream counts**
(the backbone IS the architecture — flags describing a model the run
doesn't build are errors, not ignored). `--aux-fields` (ar_backbone
only) enables aux text training (§2.4); `--aux-loss-weight` sets w
(default 0.5 — the labels are weak supervision); `--aux-dropout` sets
the [ACT] mode-dropout rate (default 0.1). Train step returns
component losses; `train/loss_action` + `train/loss_aux` log beside
`train/loss` on aux runs (aux aggregates as CE-sum/token-count across
the window and all ranks — a position-weighted mean, immune to the
sparse-batch dilution a mean-of-means would suffer).

**In-training probes.** `--eval-samples N` sizes two MAE probes
(eval_chunk_mae on holdout, train_mae on train), drawn exactly as
`bijou.eval --seed` would, sharded and all-reduced, CPU-resident.
ar_backbone probes decode through the unified free-until-BOA path, so
in-run MAE and offline eval agree (~0.02 at 100k); aux runs add the
generations table columns and holding likelihood accuracy (§2.4),
rank-0-only, bounded to the rich table rows.

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
checkpoints load forever via `checkpoint_sections`' read-side synthesis
(format 1 through the train-args synthesizer), no file conversion;
guarded by state-dict key fixtures and cross-format section tests. `--init-from` = warm
`--init-from` = warm
start (decoder config-guarded, loud SystemExit — except the data-side
format keys {aux, suffix_format}, which may differ with a printed
note: enabling aux / adopting a newer suffix format on an older base
is the sanctioned warm-start pattern, and format-added params — the
mode tables, for pre-format-3 checkpoints — fresh-init loudly with an
exact allowed-missing-keys check; NOT guarded: `--max-soft-tokens`,
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

    uv run python -m bijou.train --train-data /home/marius/w/community_dataset_v1_v3 \
      --backbone outputs/tiny-gemma4 --decoder-hidden 64 --decoder-heads 2 \
      --decoder-intermediate 128 --decoder-cross-heads 2 --stream-counts 1 1 2 \
      --steps 2 --batch-size 2 --num-workers 2 --log-every 1 --eval-every 5 \
      --save-every 1000 --eval-samples 4 --device cpu --seed 0 \
      --save-dir outputs/train/oracle_tmp

must reproduce **flow 1.8896 / 1.7237** exactly; with `--decoder ar_fast
--fast-tokenizer tests/fixtures/tiny_fast_tokenizer` added, **AR
4.8803 / 4.8656**; with `--decoder ar_backbone --fast-tokenizer
tests/fixtures/tiny_fast_tokenizer` (and the `--decoder-*` shape flags
OMITTED — ar_backbone rejects them), **27.7622 / 27.7245** (random tiny
weights under full-vocabulary CE — an anchor, not a quality signal; the
huge step-1 grad norm is softcap saturation, clipped in practice;
re-baselined 2026-08-02 for suffix format 3 — the fed mode token —
from format 2's 27.8116/27.8348).
Flags-on (unfreeze) oracles live in
`outputs/probe_unfreeze_gradflow.py` (flow 1.5528, AR 4.8689,
ar_backbone 27.6946 — format-3 re-baseline — with the FULL-depth
partition checks, asserted in the probe). Regenerate the tiny backbone
with
`uv run python -m bijou.gemma4.testing --output outputs/tiny-gemma4`
(per checkout; changing it re-baselines every oracle). gemma4 changes
additionally gate on `verify_parity` (needs a big GPU). Any new
architecture path records its own oracle loudly.

**Step split** (H100, batch 64, measured): observation encode 79.3% / expert
fwd 4.6% / bwd 15.4% / opt 0.7%. The frozen backbone forward dominates —
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
behind one policy interface, returning a `ChunkPrediction` (chunks +
aux generations); ar_backbone picks its decode mode via `--aux-mode
{act,free}` on eval and rollout (§2.3: act = fast path, free = speak
first, ~30–45 extra suffix forwards per replan; no suffix KV-cache
reuse across replans yet — the known optimization if it deploys), and
AR checkpoints need the deployment rig's exact q01/q99 quantiles (the
tokenizer's fit normalization), which old-format stats tables don't
carry.

**Artifacts.** Checkpoints + tokenizers:
[`mcobzarenco/bijou-checkpoints`](https://huggingface.co/mcobzarenco/bijou-checkpoints)
(seed checkpoints keep `optimizer.pt`, the rest are weights-only).
Datasets: three community collections + the two rig repos
(`mcobzarenco/so101_pick_place_{v2,clean}`, private) — all with
backfilled exact quantiles; boxes mirror them under
`~/datasets/mcobzarenco/`. Runs log to wandb `bijou-dev`
(entity `aristotle1337`). Laptop dev data:
`/home/marius/w/community_dataset_v1_v3` (3 datasets, 44k frames — the
oracle corpus).

## 7. Empirical grounding and results ledger

All numbers: open-loop chunk MAE, raw degrees, 256 frames, seed 0
(Heun-10 for flow; AR decodes greedily), state-copy baselines scored on
the identical frames. **Comparability rules**: numbers only compare
within one frame set — the fps-30-filtered set and the unfiltered
legacy set index frames differently; token-level metrics never cross
tokenizer versions; the 256-frame probe has a ±0.3 noise floor and the
in-run probe matches offline eval to ~0.01 when settings agree (per-
rank sharding once read ~0.3 high on the unftext r2 lineage — trust
offline for decisions). Full reports/JSONs in `reports/`; superseded
run narratives live in git history.

### Ledger — fps-30 frame set (current mainline; copy 11.50 holdout / 10.72 train / 12.00 rig-holdout-3403)

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

## 8. Directions under evaluation and proposed changes

Each subsection: the change, its justification, status, and the key
design decisions. Full blow-by-blow lives in git history (these subsume
the retired `plan_*.md`). Smaller queued arms not detailed below:
`--trim-leading-idle` (~6.7% of frames are leading idle), state-noise
augmentation, a lerobot policy plugin (`--policy.type=bijou`), suffix
KV-caching for AR decode. Hygiene owed: rotate the wandb API key (it
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
from-scratch. **Open dials.** freeze-then-thaw; ZeRO-1 / activation
checkpointing for batch; whether a flow decoder trained on the FROZEN
adapted trunk inherits the win — the decisive attribution test, now
specced as stage-2 (§8.11).

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

### 8.3 Autoregressive FAST decoding (SHIPPED — ar_fast §2.2, ar_backbone §2.3; tokenizer artifact story here)

**Shipped decoders.** `ARFastDecoder` (§2.2): grammar-constrained
greedy decode — a valid sequence expands to exactly chunk×dim quantized
DCT coefficients, so there is NO EOA and no malformed generations by
construction (each step masks to tokens whose BPE symbol-expansion fits
the remaining budget). `ARBackboneDecoder` (§2.3) reuses the same
grammar mask for its action phase. Train: `--decoder ar_fast|ar_backbone
--fast-tokenizer <artifact>`; eval/rollout work unchanged via
`predict_chunk` (greedy — no Heun knobs); AR inference additionally
needs quantile stats (rides the checkpoint's per-dataset table).
Results in §7: frozen-trunk plateau ~8.0 at 10k (≈ flow@40k quality);
with the live trunk, the project's best lines.
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
the frozen AR-pretrained trunk). A literal joint-loss run stays
available if stage-2 disappoints, but no code path for it exists today.

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

### 8.6 Backbone variants

E4B (4 exported streams, needs a 4-entry `--stream-counts`) for capacity;
E2B **base vs IT** to test whether the instruct tuning matters for our
narrow instruction distribution (prediction: ±0.2 MAE, IT edge grows only
with language-diverse data; verify the -pt checkpoint ships the vision
tower). Both are backbone-swap arms, not code changes beyond config.

### 8.7 Inference-time noise-draw ensembling (no training)

**Change.** `--sample-draws N` in rollout: batch N noise draws through the
expert (prefix encoded once), average the chunks. **Justification.** The
strongest measured accuracy lever anywhere: mean-of-10 took a fine-tune's
single-draw 5.30° → 2.88° on motion frames. ~20 lines; prefix cost
unchanged; check unimodality on the sampling report first (averaging
multi-modal draws is wrong). Directly attacks the level-uncertainty the
re-anchor/τ probes found.

### 8.8 Throughput (prefix-side)

Length-bucketed batching (batches pad to ~452 vs ~292 typical) and
`torch.compile` of the frozen backbone — the backbone forward is 79% of
the step, so these are the real multipliers; a `kv_stop_layer` win already
landed. Expert-side autocast is dead (expert is only ~20%).

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

### 8.10 Aux text tasks (IMPLEMENTED §2.4; value measurement next)

**Status.** Rendering, loss, decode, metrics and provenance pinning all
shipped (§2.4). Format-2 smokes on the annotated rig v2 dataset
validated the mechanism (aux arm: loss_aux 22.3 → 0.07, coherent
self-emitted subgoals, zero budget fallbacks; aux-less arm: BOA
immediately on 16/16 probe rows); **suffix format 3** (the fed mode
token + `--aux-dropout`, built for sparsely-annotated corpora — §2.3)
supersedes those arms and needs its own smoke before the value
experiment. Aux labels exist on rig v2 only today; the
community-corpus judging pass is the data-side dependency for aux at
pretrain scale.
**Next: the paired value experiment, ON FORMAT 3** — aux-on vs aux-off
fine-tunes
from the 100k base on rig v2, matched seed/steps/LRs (5k, eval every
250, decoder 2e-5 / text 1e-5), one variable. Pre-registered: aux-on
action MAE within probe noise (±0.3) of aux-off at matched steps (aux
as free interpretability), holding likelihood accuracy approaching but
not exceeding the ~0.8 label-noise ceiling, both minima vs copy 11.973.
The interesting outcome in either direction: a measurable action-MAE
WIN would motivate aux at pretrain scale; a LOSS bounds the
task-interference cost.
**Owed alongside:** the offline `bijou.eval` aux report section
(generations + aux metrics in HTML/JSON; wandb is the only surface
today).

### 8.11 Stage-2: flow decoder on the frozen AR-pretrained backbone

**Change.** Train a flow expert (§2.1) against the FROZEN backbone of
an ar_backbone pretrain: slice a full-depth snapshot to the prefix
layers, then `--init-backbone-from` it with unfreeze flags off. The
decisive attribution test from §8.1 (does the AR-shaped trunk transfer
through exported K/V?) and the deployment answer if AR decode latency
ever binds (flow replans in ~233 ms; AR is ~30–80 sequential backbone
forwards). **Status.** Parked pending §8.10; needs the snapshot-slicing
+ init plumbing (small, loading-side) before any run.

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
