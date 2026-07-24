# Bijou: architecture reference & thread handoff

Purpose: complete context from the Bijou build thread (July 2026) — the
implemented ML architecture, the design alternatives we weighed, a survey of
the code, the training setup, and the infrastructure state — in enough depth
to continue in a fresh session. Section D isolates machines/paths so the ML
content (A–C) reads without operational noise. Predecessor context:
`docs/handoff_gemma4_vla.md`.

**TL;DR state**: `bijou/` contains (1) a pure-torch, HF-parity-verified
Gemma 4 E-series implementation (`bijou/gemma4/`), and (2) the Bijou VLA — a
frozen truncated E2B prefix encoder exporting global-attention KV streams, a
~380M flow-matching action expert cross-attending them, and a multi-dataset
train CLI. Everything is validated end-to-end on the H100 box; a 20k-step
run on the community v2 corpus crashed at step 300 on two corrupt datasets
(guard added, `e00e151`) and is being relaunched by the owner.

---

## A. The model

### A.1 Backbone: Gemma 4 E-series, what matters for Bijou

Family (June 2026): E2B, E4B (on-device, `model_type: gemma4`), 12B
"Unified" (`gemma4_unified`), 26B-A4B MoE, 31B. transformers ≥5.14. We use
**`google/gemma-4-e2b-it`** (E4B supported and parity-verified too).

Text decoder (E2B → E4B where different):

| | E2B | E4B |
|---|---|---|
| layers | 35 | 42 |
| hidden / MLP | 1536 / 6144 | 2560 / 10240 |
| heads / KV heads | 8 / 1 | 8 / 2 |
| sliding : full pattern | 4:1 (full at 4,9,14,…) | 5:1 (full at 5,11,17,…) |
| KV-shared tail layers | 20 (from layer 15) | 18 (from layer 24) |
| double-wide MLP on shared layers | yes (12288) | no |

Shared across E-series: vocab 262144, sliding window 512 (θ=10k RoPE,
head_dim 256), **global layers head_dim 512 with p-RoPE** (θ=1M,
`partial_rotary_factor=0.25` — only the first 64 of 256 frequency pairs
rotate; the rest are position-invariant), Q/K RMSNorm with scale + V RMSNorm
without, attention scaling 1.0, final logit softcap 30, tied LM head,
`sqrt(hidden)`-scaled embeddings (bf16-rounded), RMSNorm applies `x·w` (NOT
`x·(1+w)` as in Gemma 2/3), **PLE** (per-layer embeddings: a packed
`[262144, L·256]` table + a projection of inputs feed a gated residual into
every decoder layer), causal-only attention (`use_bidirectional_attention:
null`; only the 12B has blockwise-bidirectional image attention).

**KV sharing** (the architectural hook Bijou exploits): layers ≥15 (E2B) have
*no K/V weights*; every deep sliding layer reuses layer 13's K/V and every
deep full layer reuses layer 14's. The deep half of the network is an
existence proof that many query-only layers can run productively against a
couple of fixed KV tensors.

Vision tower (identical E2B/E4B): encoder-free — `pixel_values` are rows of
raw 16×16 RGB patches (768 features), `Linear(768→768)` + learned 2D
position embeddings, 16 bidirectional layers with 2D RoPE (θ=100) and
checkpoint-provided clip bounds on all linears, 3×3 spatial average pooling
scaled by `sqrt(768)` in fp32, then RMSNorm(scale-less) + `Linear(768→1536)`
into LM space. Soft-token budget per image `max_soft_tokens ∈
{70,140,280,560,1120}`: 640×480 → 266 tokens at 280, 133 at 140. Audio tower
exists in checkpoints but is deliberately not implemented.

### A.2 Bijou architecture

```
[instruction][cam_1]..[cam_k][instruction]      (chat-templated user turn)
        │ frozen truncated backbone: layers 0..14 only (E2B)
        ▼
  KV streams of the GLOBAL prefix layers {4, 9, 14}   ← PrefixKV, cached
        │ cross-attention                              per observation
        ▼
ActionExpert: 15 narrow layers, each =
  cross-attn(one scheduled stream) → self-attn([state][a_1..a_50]) → gated MLP
        ▼
velocity of the action chunk at flow time τ  →  Heun τ:1→0 (default 5 steps)
```

- **Prefix encoder** = the backbone truncated at `first_kv_shared_layer_idx`
  (15 for E2B, 24 for E4B): exactly the layers that compute K/V. Loading
  drops layers ≥15 and slices the packed PLE tensors to the kept layers —
  **2.55B params instead of 5.2B** (vision tower kept, head tied=free). The
  exported streams are **bitwise identical** to a full-model forward (eager).
- **Streams**: the global layers within the prefix — E2B (4, 9, 14), E4B
  (5, 11, 17, 23). Uniform geometry (head_dim 512, 1 KV head, k-normed,
  p-RoPE), unbounded cache (full-attention layers never window), trained for
  arbitrary-range attention. ~2 KB/token/stream bf16 → a 600-token prefix
  costs ~3.5 MB to cache across replans.
- **Expert cross-attention** adopts backbone geometry exactly: queries
  `Linear(hidden→heads·512)` + q-RMSNorm + p-RoPE at positions
  `P, P+1, …` (P = prefix length), scaling 1.0, groups = heads (1 KV head).
  Per-layer stream assignment is an explicit `cross_attention_schedule`
  tuple; default blocks `(4,4,4,4, 9×4, 14×7)` — the three counts are the
  agreed knobs, deepest stream weighted heaviest.
- **Expert suffix** `[state][50 actions]`: state and actions enter via
  learned `Linear(6→hidden)` (π0-style; state deliberately NOT in the VLM
  prompt — keeps the frozen backbone fully in-distribution and decouples
  slow visual context from fast state for future async replanning). Flow
  time τ: sinusoidal embedding (periods 4e-3..4.0) → MLP → added to action
  embeddings only. Self-attention has its own θ=10k RoPE over suffix
  positions and two mask modes (`SelfAttentionMode`): `CAUSAL_ACTIONS`
  (state↔everything, actions causal; SmolVLA's ablation winner, our
  default) and `BIDIRECTIONAL` (π0's choice). No inference-cost difference
  — flow matching denoises the chunk jointly (no autoregression over
  actions), so bidirectional is inference-safe; this is a pure quality
  ablation. Velocity head is zero-initialized.
- **Flow matching** (lerobot π0/SmolVLA conventions, so recipes port):
  `x_τ = τ·ε + (1−τ)·a`, target `u = ε − a`, ε~N(0,I), train
  τ ~ Beta(1.5,1)·0.999+0.001; sampling integrates τ 1→0 with
  `SamplingMethod.HEUN` (default, 5 steps = 10 NFE, same cost as π0's
  Euler-10 but ~2x lower worst-case integration error — measured vs a
  Heun-64 reference; Euler available; Heun wasteful below ~4 steps);
  `sample_actions(noise=, generator=)` for seeded determinism (eval harness
  pattern from the SmolVLA work ports directly).
- **Prompt**: instruction sandwich. Under E2B's causal-only attention this
  yields instruction-conditioned image KV *and* image-conditioned
  instruction KV for ~15–30 extra tokens; the tail `[images][text]` matches
  the model card's recommended ordering. Chat template applied (IT model).
- **Normalization contract**: state/actions arrive MEAN_STD-normalized
  (flow matching mixes with N(0,I) noise, so this is load-bearing);
  `sample_actions` returns normalized actions. Stats live in the
  training-side processor, saved with every checkpoint.

Current run sizing: expert hidden 1024, 8 heads, MLP 4096, 8 cross-heads,
15 layers → **378.9M trainable** (fp32) on the 2.55B frozen bf16 prefix
(π0 pairs ~300M expert with a 2.6B LM; the 190M default felt undersized).

### A.3 Design decisions, alternatives, trade-offs

1. **Backbone choice — E2B** (E4B ready): encoder-free patches with
   per-camera token budgets, 256k-trained global layers, on-device sizing.
   12B Unified rejected for v0 (latency; bidirectional-vision masks not
   implemented).
2. **What the expert attends** — three candidates:
   (a) layer-matched per-layer KV (SmolVLA/π0 style, 15 streams,
   hierarchical features; 12/15 streams sliding-windowed, mixed head dims);
   (b) the two shared streams (layers 13+14 — "the expert = more KV-shared
   layers", validated by the backbone's own deep half);
   **(c) global-only streams {4,9,14} — chosen**: uniform 512 geometry, no
   window truncation ever, p-RoPE trained at long range, multi-depth
   features, and it sidesteps the sliding-KV out-of-distribution issue
   (sliding K beyond 512 relative distance were never consumed in
   pretraining; keeping them is mechanically trivial — cache trimming — but
   statistically unvetted). Caveat noted: layers 4/9 KV only ever served
   their own layer in pretraining; layer 14 KV is the
   architecture-validated general-purpose memory (hence the heavier
   schedule weight). (a) remains the upgrade path; the schedule tuple makes
   all variants config diffs.
3. **Schedule blocks vs cycle**: blocks (shallow→deep) chosen by owner
   preference with the three counts as knobs; cycling (4,9,14,4,…) is the
   first schedule ablation; theory doesn't decide (earliest expert layers
   hold the noisiest actions and might want the deepest KV).
4. **Cross+self in every layer** vs SmolVLA's alternating: kept both per
   layer (cross first — actions read the scene before mixing); alternating
   halves attention cost and is a cheap ablation.
5. **State in expert vs in VLM prompt**: expert (π0 layout) — frozen
   backbone sees only in-distribution inputs; SmolVLA puts state in the VLM.
6. **Instruction sandwich** vs single instruction: sandwich; cost trivial,
   both conditioning directions materialize in KV under causality.
7. **Camera slots**: positional by sorted key with per-sample discovery —
   verified empirically that the community collections' `image`/`image2`
   names carry no wrist-vs-scene semantics (visual check: `image2` is wrist
   in ~half, external side view in the rest; sample strips in
   `outputs/cam_check/`, local only). SmolVLA pretrained in exactly this
   noisy-positional regime on the same data. The proper fix is a
   camera-role sidecar produced by the Gemma judge ("is this camera
   arm-mounted?") feeding a future `--camera-map`; deferred to the curation
   phase.
8. **Embodiment scope**: 6-DoF SO-100/101 only; non-6-DoF datasets dropped
   loudly (20 in community v2). Cross-embodiment (pad-to-max à la SmolVLA's
   32 dims vs per-embodiment heads) deliberately out of scope for now.
9. **Frozen backbone for v0** (no LoRA): keeps the stream-parity invariant
   checkable during development and isolates expert learning; unfreezing is
   a later experiment.
10. **Chunking**: `chunk_size=50` denoised, `n_action_steps` (how many to
    execute before replanning) kept as a separate deployment knob; dataset
    dictates 33ms/step at 30fps → 1.7s horizon.

### A.4 Measured properties

- **Backbone parity** (`bijou.gemma4.verify_parity`, H100): eager backend
  bitwise-identical to HF transformers 5.14.1 for E2B *and* E4B — prefill,
  600-token sliding-window cached decode, greedy generate, image path.
  SDPA backend: single-forward max|Δ| ≈ 0.6–1.8 (5–14 bf16 ULPs at
  softcapped-logit scale), greedy tokens identical except genuine near-ties
  (top-2 gap ≤ tolerance in both models — observed gaps 0.125–0.375).
  CPU-only caveat: HF is nondeterministic against *itself* in cached decode
  (oneDNN bf16; measured seed 1–5 ULP), so tolerance gates + token
  agreement are the contract, not bitwise.
- **Attention backend perf** (H100, E2B): SDPA vs eager — 8k-token prefill
  1.91× faster / 4.65× less peak memory; image prefill 1.58×; decode parity
  (q_len==1 dispatches to eager — fused kernels are launch-bound there).
  Global layers need a repeat_kv fallback above head_dim 256 (flash/cudnn
  cap; mem-efficient rejects enable_gqa → silent math-backend 3× trap).
- **VLA latency** (E2B, H100, warm, batch 1): `encode_prefix` 41 ms @ 566
  tokens (2 cams @ 280/140); 10-step `sample_actions` 156 ms; ≈ 197 ms per
  50-action chunk. Expert forward 19 ms (launch-bound; compile headroom).
- **Stream export parity**: truncated prefix encoder ≡ full model, bitwise.
- **Training smoke** (7-episode pick-place, 300 steps, batch 8): flow loss
  2.44→1.33 (init ≈ E[(ε−a)²] ≈ 2 sanity-checks the zero-init head);
  chunk-MAE ~24 raw units (predict-mean level — expected from scratch;
  SmolVLA reference reached ~2 only from its pretrained base).

---

## B. The code

### B.1 Layout and dependency structure

```
bijou/
  gemma4/                  pure-torch Gemma 4 E-series (no transformers dep)
    config.py              frozen dataclasses, no defaults; e2b_config()/e4b_config()
    cache.py               KVCache (HF DynamicCache semantics; sliding trim)
    masks.py               MaskSpec (+is_causal flag), additive mask builders
    layers.py              RMSNorm, RoPE math, eager/SDPA attention, AttentionBackend
    text.py                decoder: PLE, hybrid attention, KV sharing, TextModel
    vision.py              patch embedder, 2D-RoPE encoder, pooler, MultimodalEmbedder
    model.py               Gemma4Model (embed_multimodal + lm_head), set_attention_backend
    generation.py          greedy/sampled generate with cache
    loading.py             meta-device loader; truncate_layers + PLE slicing
    testing.py             synthetic images; write_tiny_checkpoint (~96MB, loadable)
    verify_parity.py       HF-comparison CLI (tolerance + near-tie token gates)
    bench.py               eager-vs-SDPA benchmark CLI
  expert.py                ExpertConfig, ActionExpert, PrefixKV, SelfAttentionMode
  model.py                 BijouModel: encode_prefix / forward / sample_actions
  loading.py               from_backbone(), default_expert_config(), prefix_global_layers()
  train.py                 multi-dataset train CLI (see §C)
```

Imports form a **verified strict DAG** (AST + Kahn check done in-session):
`config → cache → masks → layers → {text, vision, expert} → models →
loaders/testing → CLIs`. `__init__.py` files are re-export hubs only;
submodules never import package roots.

### B.2 Key implementation notes

- **gemma4 numerics**: mirrors HF expression-for-expression in the eager
  path (fp32 RMSNorm/softmax round-trips, bf16 mask values `finfo.min`,
  bf16-rounded embed scales, p-RoPE zero-frequency tail). Rope inverse
  frequencies always fp32; under meta-device construction, computed buffers
  materialize on CPU (`buffer_device`) since they can't be loaded.
- **Loader** (`gemma4/loading.py`): builds on meta device, streams
  safetensors directly to the target device, casts to config dtype, drops
  audio tower + the checkpoint's redundant shared-layer k/v weights, ties
  the head. `truncate_layers=N` validates N ≤ first_kv_shared and slices
  the two packed PLE tensors via `safetensors.get_slice` (reads only kept
  bytes).
- **KV sharing implementation**: `is_kv_shared_layer` /
  `is_kv_source_layer` on the config; a per-forward `shared_kv` dict is
  written by the source layers (13/14 E2B) and read by the tail. The
  Bijou prefix encoder ignores all of this by truncating before the shared
  region and exporting per-layer K/V from a `KVCache` (full-attention
  layers cache unbounded, so exports are complete).
- **Attention backends**: `AttentionBackend.{EAGER,SDPA}` is a runtime
  constructor arg (never in config/checkpoint), mutable in place via
  `set_attention_backend`. SDPA is default; eager is the parity reference.
- **Tiny checkpoints** (`python -m bijou.gemma4.testing --output ...`):
  structurally faithful miniature (hybrid layers, KV sharing → exercises
  truncation + PLE slicing, p-RoPE, tied head, three global prefix layers
  so default `stream_counts` fits; real vocab + patch size so the real
  tokenizer/processor work). Train smokes: ~0.45 s/step CPU vs ~43 with
  the real backbone. No test-only branches anywhere in production code.

### B.3 Conventions (enforced)

- **Typing**: full parameter+return annotations everywhere — ruff
  flake8-annotations (`ANN`, ANN401 off) + pyright
  `reportMissingParameterType=error`; `fmatch/` legacy exempt per-file.
  Dataclasses over dicts wherever all fields are present
  (`CollatedBatch`, `Normalizers`, configs, `PrefixKV`, `MaskSpec`); dicts
  only for genuinely variable schemas (raw LeRobot items, external stats).
- **Configs carry no defaults** — they describe checkpoints; released
  architectures come from `e2b_config()`/`e4b_config()` (asserted equal to
  the shipped config.json files) or `Gemma4Config.from_json`.
- **Explicit `device=`/`dtype=` factory kwargs on every module** (owner
  preference over context managers) — enables meta-device loading and
  mixed-precision composition (bf16 backbone + fp32 expert casts inputs/
  streams to its own dtype).
- Imports at top of file, no `TYPE_CHECKING` guards (none needed — DAG).
- Run `uv run pyright` **unpiped** (piping to `tail` swallowed nonzero
  exits twice in this thread and let a broken commit through once).

### B.4 Verification tooling

- `python -m bijou.gemma4.verify_parity --device cuda [--long-context 600]
  [--image path] [--attn-backend both|eager|sdpa] [--raw]` — chat-templated
  prompts by default, synthetic deterministic image if none given; hard
  gates = single-forward logits within `--tolerance` (2.0) + near-tie-aware
  greedy token agreement; image interior-token logits are info-only.
- `python -m bijou.gemma4.bench --device cuda` — eager vs SDPA table
  (prefill time/peak-mem at several lengths, decode tok/s, image prefill).

---

## C. Training setup

### C.1 CLI and data selection

`python -m bijou.train` — key arguments:

- `--train-data PATH...`: dataset dirs and/or collection roots, mixed;
  roots scanned for `*/meta/info.json` and `*/*/meta/info.json`. Subsetting
  via shell globs; `--exclude` fnmatch on `<user>/<dataset>`. (A future
  `--val-data` with the same semantics is anticipated; `--data-list file`
  is the planned form for judge-curated corpora.)
- Startup census: datasets/episodes/frames + camera-set histogram; **loud
  drops** with reasons: dims ≠ first dataset's (20 in community v2:
  7/12/14-DoF), missing stats, and metadata-vs-parquet frame-count
  mismatches (2: `zaringleb/*`, 269 phantom trailing frames each — crashed
  the first 20k run at step ~300 via ConcatDataset indexing).
- `--cameras` (filter, keys or suffixes), `--max-cameras`,
  `--max-soft-tokens` (per camera; 140 default → 133 tokens per 640×480
  cam), `--instruction` (override per-frame task strings).
- Expert knobs: `--expert-{hidden,heads,intermediate,cross-heads}`,
  `--stream-counts`, `--self-attention-mode`, `--chunk-size`.
- `--eval-samples N`: eval set = N items at evenly-strided concat indices
  (spans datasets/episodes); chunk-MAE (raw units, seeded model-default
  sampling) always reported to tty+jsonl. Orthogonal to wandb.
- `--wandb-project` (off by default) adds scalars + a rich per-eval table:
  positional camera columns with key captions, task, state, per-sample MAE,
  per-joint predicted-vs-truth chunk plots.

### C.2 Data pipeline (the performance-relevant details)

- Sampling: `ConcatDataset` + shuffle (frame-uniform; per-dataset weights
  deliberately deferred). Per-dataset `delta_timestamps` from each fps give
  `[50,6]` action chunks + `action_is_pad`; loss masks padded steps.
- `PrefixCollator` (in dataloader workers): per-item sorted camera keys →
  instruction-sandwich conversations → Gemma4 processor. **transformers
  5.14 trap**: per-call processor kwargs must be nested AND a flat
  `padding=True` silently discards `processor_kwargs` — both go inside
  (this silently ignored the soft-token budget until caught: 564 → 292
  prefix tokens).
- **Workers must be spawned, not forked**: torchcodec/ffmpeg deadlocks or
  throws "Could not push packet to decoder" in forked children of a parent
  with live CUDA + in-process decode state (verified empirically; applies
  to both AV1 community data and own recordings — all lerobot video is
  AV1). Single-threaded workers via `worker_init_fn`.
- H2D path: workers emit pinned `CollatedBatch`es (custom `pin_memory`
  hook); `DevicePrefetcher` does all transfers — one-batch lookahead on a
  side CUDA stream (`record_stream`'d), so the loop receives
  device-resident batches. No `.to(device)` anywhere in loss/eval code.
  Per-step hidden syncs removed: `has_padding` decided CPU-side in
  workers; loss/grad-norm sync once per `log_every`.
- Known bottleneck: at batch 64 the run is **decode-bound** (~128 AV1 GOP
  decodes/step) — the GPU-utilization sawtooth's dips are worker
  starvation, not GPU-side serialization. Levers: `--prefetch-factor`
  (default 4), workers ≈ cores−6; structural fixes (episode-chunked
  sampling, pre-decoded caches, uint8 pixel transport) are open items.
- Multi-dataset scaling measured: 303 datasets construct in 26.4 s once;
  the ConcatDataset pickles at 5 MB / 0.4 s into spawn workers → lazy
  per-dataset init not needed.

### C.3 Normalization and checkpoints

- MEAN_STD **per dataset**: every sample is normalized with its own
  dataset's stats (`StatsAttachedDataset` → batch fields → loss/eval).
  Rationale (measured): 294/303 community datasets record old-lerobot
  DEGREE-calibrated angles whose DIY homing offsets spread the SAME
  physical rest pose by 20–63° std across rigs — camera-invisible; under
  aggregate normalization the model must route the offset from the state
  token at gain exactly 1 and was only at ~0.56 by 7k steps (predictions
  "flat at the wrong constant", trivial state-copy baseline unbeaten).
  Per-dataset stats subtract it in the pipeline. Stds floored at 1e-2;
  non-finite stats → dataset dropped loudly.
- Checkpoints (`--save-every`): `expert.safetensors` + `optimizer.pt`
  (Adam moments, LambdaLR state, step) + `bijou_config.json` (expert
  config, backbone id, aggregate stats as rollout fallback,
  `per_dataset_normalization` table, train args, step).
- **Stats-table contract (decided)**: the table covers the datasets of
  THIS run only — no inheritance/merging across `--init-from` chains. A
  fine-tune checkpoint carries the fine-tune rigs' stats (what deployment
  needs); pretraining stats live in the pretraining checkpoint; eval on
  any dataset gets stats from the dataset itself. Rollout must normalize
  with the deployment rig's stats.
- **Restarts**: `--resume <ckpt>` = lossless continuation (weights +
  optimizer + schedule + step; checkpoint's optimizer hyperparameters win
  over CLI, printed loudly; `--steps` counts total). `--init-from <ckpt>`
  = warm start (weights only, CLI honored, fresh step count — use a new
  `--save-dir`, short re-warmup ~300 steps for cold Adam moments; the only
  option for checkpoints predating `optimizer.pt`). Both fail early on
  expert-config mismatch. Training never reads normalization from the
  checkpoint (per-dataset stats come from the data), so `--train-data`
  may change freely across restarts — only action/state dims and chunk
  size must match.

### C.4 Hyperparameters and runs so far

- Recipe: AdamW lr 1e-4 peak, betas (0.9, 0.95), wd 1e-5, grad-clip 10,
  linear warmup (≈500–1000 for real runs) → cosine to 10% of peak; batch
  64; fp32 expert on bf16 frozen backbone; τ ~ Beta(1.5,1).
- Runs: (i) 300-step single-dataset smoke — plumbing validation, loss
  2.44→1.33; (ii) 20k-step community-v2 run (expert 1024/8/4096/8, ≈379M):
  crashed at step ~300 on the zaringleb corruption (nothing saved — first
  checkpoint was at 2500), guard landed, **relaunch pending/in the owner's
  hands**; watch `s_per_step` (was 1.58 before the pipeline work) and the
  eval table on wandb (`bijou-dev`, run naming `community-v2-20k-h1024`).

### C.5 Dataset landscape (for training decisions)

Community v2 converted corpus: 323 datasets / 12.2k episodes / 5.83M
frames; 303 are 6-DoF (5.37M frames usable), all `robot_type` so100.
Camera sets: 241× (image,image2), 37× (image,), 35× (image,image2,image3),
10× (image2,) — see §A.3(7) for semantics. Frame counts per camera set in
the training census. Community v1 converted corpus not yet colocated with
the box (open item). Owner recording `so101_pick_place_clean` (7 eps, 3399
frames, cams front/wrist) mixes in cleanly (padding path exercised).

---

## D. Infrastructure & operations (the noise section)

- **Laptop**: repo at `/home/marius/w/flow-matching` (this thread ran in
  worktree `major-yak`); RTX 3000 Ada 8 GB (fits tiny-backbone CUDA
  smokes); 60 GB RAM. uv project, Python 3.13, transformers 5.14.1 pinned
  via override. `uv run pyright` / `uv run ruff check` are the gates.
- **H100 box**: `ubuntu@209.20.156.82` (ssh -A for GitHub; `uv` at
  `~/.local/bin`). Repo `~/flow-matching` synced to origin/main
  (`git fetch && git reset --hard origin/main`). Bills while up.
- **Data on the box**: `~/datasets/mcobzarenco/community_dataset_v2_v3`
  (121 GB, 323 datasets); `~/datasets/marius/so101_pick_place_clean`;
  `~/community_dataset_v1_v3/ZGGZZG/so100_drop0` (old 1-dataset dev
  sample). **v1_v3 full collection not found on the box** despite being
  uploaded to the hub — locate/download before training on v1+v2.
- **Laptop data**: `/home/marius/w/community_dataset_v1_v3` (3 converted
  datasets — dev sample), `/home/marius/w/datasets/marius/so101_pick_place_clean`.
- **Checkpoints cached** both machines: `google/gemma-4-e2b-it`,
  `google/gemma-4-e4b-it` (box also has 12B). Tiny test checkpoint:
  regenerate with `uv run python -m bijou.gemma4.testing --output
  /tmp/tiny-gemma4` (not committed).
- **wandb**: project `bijou-dev`, entity `aristotle1337`, key via
  `WANDB_API_KEY` env only (never in code/commits). The key was pasted in
  chat/shell history during this thread — **rotate it** when convenient.
- Command cheat-sheet:
  - parity: `uv run python -m bijou.gemma4.verify_parity --device cuda
    --long-context 600` (add `--model google/gemma-4-e4b-it` for E4B)
  - bench: `uv run python -m bijou.gemma4.bench --device cuda`
  - fast train smoke: tiny checkpoint + `--backbone /tmp/tiny-gemma4
    --expert-hidden 64 --expert-heads 2 --expert-intermediate 128
    --expert-cross-heads 2 --stream-counts 1 1 2`
  - the 20k run command: see §C.4 / shell history on the box (expert
    1024/8/4096/8, batch 64, workers 20, warmup 1000).

---

## E. Open threads, roughly prioritized

1. **Relaunch + babysit the 20k community-v2 run** (guard landed; verify
   `s_per_step` post-pipeline-work; consider `--save-every 1000`).
2. **Open-loop eval harness** on held-out episodes (port
   `eval_smolvla.py`'s seeded-noise, chunk-MAE-per-frame pattern; the
   `--val-data` argument design is agreed-in-principle).
3. **lerobot policy wrapper** (`PreTrainedPolicy` + processor pipeline +
   registry) so `lerobot-rollout`/`lerobot-train` work; then real-robot
   rollouts on the SO-101.
4. **Resume support** (optimizer/scheduler state in checkpoints).
5. **Ablations queue**: bidirectional vs causal-actions (owner interest),
   schedule blocks vs cycle, stream sets (global-only vs +sliding vs
   all-15), expert width/depth, soft-token budgets per camera.
6. **Camera-role sidecar** via the Gemma judge (wrist vs scene
   classification) → `--camera-map`; belongs to the curation phase along
   with judge-filtered corpora (`--data-list`).
7. **Throughput**: decode-bound training (episode-chunked sampling,
   pre-decoded caches, uint8 pixels); `torch.compile` on the expert;
   flex-attention for sliding-window prefill in gemma4.
8. **Locate/download community v1_v3** to the box; later v3 (multi-
   embodiment → requires the padding/per-embodiment-head decision).
9. **E4B Bijou** (config-driven already: prefix 24 layers, 4 streams —
   needs `stream_counts` of length 4) if E2B quality caps out.
10. Gemma4 gaps, deliberate: audio tower, MoE (26B-A4B), 12B
    bidirectional-vision masks, `standardize=True` vision option.

## F. Hard-won gotchas (read before debugging anything)

- transformers 5.14: nested `processor_kwargs` required; flat
  `padding=True` **silently discards** them (budget ignored).
- torchcodec/ffmpeg is **fork-unsafe** → spawn dataloader workers, always.
- Community data lies: metadata frame counts can exceed parquet rows
  (zaringleb), 20 datasets aren't 6-DoF, camera names carry no semantics,
  stats keys were broken pre-conversion. Validate loudly, drop loudly.
- CPU bf16 (oneDNN) makes HF nondeterministic vs itself in cached decode
  (1–5 ULP seed, amplified by KV feedback) — never chase bitwise on CPU;
  parity work happens on CUDA with the eager backend.
- SDPA: head_dim > 256 silently falls to the math backend unless KV heads
  are materialized (flash/cudnn cap, mem-efficient×enable_gqa conflict);
  q_len==1 decode is faster eager.
- `pkill -f` from an ssh one-liner matches the ssh session's own cmdline.
- Piping `pyright | tail` eats failures; grep with `head` sends SIGPIPE to
  long-running producers (killed a training run mid-census once).
- uv: `uv add` is inexact, `uv sync` is exact (stale packages mask missing
  extras).
- HF `attention_mask` is Long — cast before boolean mask algebra.
