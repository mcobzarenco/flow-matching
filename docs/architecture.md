# Bijou architecture

Bijou is a vision-language-action model for SO-100/101 arms: a **frozen,
truncated Gemma-4 E2B-IT** encodes camera images + a language instruction
once per observation and exports the K/V of a few of its layers; a
**404M fp32 flow-matching action expert** cross-attends that K/V and
denoises a 50-step action chunk. Per-dataset normalization makes ~1000
miscalibrated community rigs trainable together; the ultimate target is
the owner's physical SO-101.

This document is the deep reference for the model and training system as
they exist, plus the proposed changes we are evaluating (§9). Operational
state (what ran, what scored, machines) lives in `handoff.md`; per-module
contracts live in docstrings; the import DAG and coding rules in
`code-styleguide.md`. Package layout (strict downward-only imports):
`train`/`eval`/`rollout` → `loading` → `data` → `model` → `expert` →
`gemma4`.

```
[instruction][cam_1]..[cam_k][instruction]     chat-templated user turn
      │  frozen truncated E2B: layers 0..14 (bf16), no grad
      ▼
  K/V of GLOBAL prefix layers {4, 9, 14}         PrefixKV, encoded once
      │  cross-attention (query-only, backbone geometry)
      ▼
ActionExpert (404M fp32): 16 layers, each =
  cross-attn(one scheduled stream) → self-attn([state][a_1..a_50]) → MLP
      │  velocity of the chunk at flow time τ
      ▼
Heun integration τ: 1 → 0  →  50-action chunk
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
- `P` — prefix length: the exported prompt tokens the expert
  cross-attends (the `PrefixKV` width)
- `suffix` — expert token count `= 1 + chunk` (the `[state][a_1..a_chunk]`
  sequence; the expert's `S`)
- `chunk` — action-chunk length in timesteps (`chunk_size`, default 50)
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

## 1. Prefix encoder — frozen truncated Gemma-4 E2B

**Why truncated.** Gemma-4 E-series layers ≥15 (E2B, of 35) carry no K/V
weights — the deep half runs query-only against the K/V produced at
layer 13/14. Bijou truncates exactly at that boundary
(`first_kv_shared_layer_idx = 35 − num_kv_shared_layers(20) = 15`), so it
keeps layers 0–14 (~2.1–2.55B params vs 5.2B) and the exported streams
are **bitwise-identical to a full forward**. The expert is then, in
effect, "more KV-shared layers" grafted on with the backbone's own
geometry. `bijou/gemma4/` is a pure-torch reimplementation, bit-exact vs
HF on greedy text+image generation (`verify_parity.py`).

**Exported streams = global-attention prefix layers {4, 9, 14}.** Gemma-4
uses a hybrid 5-period schedule (every 5th layer is FULL/global, the rest
sliding-window-512). Only the global layers are exported because they
have uniform 512 `head_dim`, p-RoPE trained for arbitrary range, and are
never window-truncated — the expert can adopt their geometry exactly.
`encode_prefix` runs the decoder with `kv_stop_layer = max(streams) = 14`:
it caches that layer's K/V and skips its attention/MLP and all deeper
layers (dead compute for a K/V export; ~1/15 of decoder FLOPs saved, more
if the schedule stops lower). `KVCache.update()` is functional (no
in-place writes), so this same path is autograd-transparent when the
trunk is trained (§9.1).

**Prompt = instruction sandwich** `[task][cam_1..N][task]` in one
chat-templated user turn, right-padded across a batch. Under causal
attention this yields instruction-conditioned image K/V *and*
image-conditioned instruction K/V for a few extra tokens. Camera NAMES
are positional slots (sorted); community image/image2 keys carry no
reliable wrist-vs-scene semantics (SmolVLA precedent), so slot order is
the only camera signal.

**Vision geometry** (encoder-free E-series tower, 768 hidden, 16-px
patches, 3×3 spatial pool): a 640×480 frame → resized 624×480 → 39×30
patches → **130 soft tokens** (13×10, one per 48×48-px cell), under the
140-token/camera budget. Prompt ≈ 292 tokens for 2 cameras; padded
batches reach ~452. The acuity probe (§8) found position is *sharpest at
the tower output* (8.4 px linear readout) and degrades through the LM
layers — the pool is not the bottleneck; the text stack's handling of
visual tokens is.

## 2. Action expert — flow-matching decoder

A narrow decoder over the suffix `[state][a_1..a_50]` (`suffix_length =
1 + chunk_size`). Default shape: **hidden 1024, 8 self-attn heads
(head_dim 128), intermediate 4096 (GLU), 8 cross-attn heads, 16 layers,
~404M fp32**. Freshly initialized, never loaded from the backbone.

Each `ExpertLayer` is a Gemma-style sandwich of three sublayers, each
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
real prefix length (`padding_mask.sum(1)`), not the padded batch width —
otherwise a sample's prediction depends on batch-mates' prompt lengths
(measured max|Δ| 0.55 before the fix; batch-1 rollout unaffected).

**State placement (π0 layout).** State enters the expert, not the VLM
prompt: the frozen backbone only ever sees in-distribution (image+text)
inputs, and slow visual context is decoupled from fast proprioception.

**Time conditioning (default: input-additive).** τ → sinusoidal
(geometric periods 4e-3..4, π0's unit-interval choice) → MLP → added to
the action-token embeddings at the input; the state token gets no time;
layers are unconditioned. An alternative per-layer adaRMS scheme is
implemented and under evaluation (§9.2).

Params live ~50% in the MLPs, ~33% in cross-attention (8 heads × 512
over the residual 1024), ~17% in self-attention.

## 3. Flow matching — objective and sampling

lerobot's π0/SmolVLA convention. With ε ~ N(0, I) and clean chunk a:

    x_τ = τ·ε + (1−τ)·a          target  u = ε − a
    τ ~ Beta(1.5, 1) → (0.001, 1]   (mass toward τ=1)

MSE of the expert's velocity against u over the FULL chunk: episode-
boundary chunks carry repeat-last-action targets (lerobot's delta-
timestamps query clamps indices to the episode range, so tail positions
hold the final real action — verified elementwise on v1). Decision
2026-07-29, replacing the earlier masked-out padding: the expert attends
every chunk position (directly under bidirectional self-attention), so
masked padding still shaped predictions invisibly, and "hold the last
action" is the correct post-completion behavior. The scan
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
mean-of-N draws is the single largest known accuracy lever (§9.7).

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
tokenizer (§9.3). lerobot's native dataset-level quantiles are a
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
net; a tighter clip renormalizes most steps and injects moment noise).

**Component learning rates.** `--expert-lr` (always > 0), plus optional
`--text-lr` / `--vision-lr` — omitting a component's lr keeps it frozen
(explicit 0 is rejected). All groups share the cosine shape scaled to
their own peak. Trunk training uses fp32 master weights with a bf16
autocast prefix encode (bf16 updates vanish below bf16 resolution at
~1e-5); the expert stays fp32-with-TF32 outside the autocast region. When
the trunk is live, one `BijouTrainStep` module owns prefix-encode + expert
so a single DDP wrapper (`static_graph`) hooks both; the frozen path keeps
the expert-only wrap, byte-identical. See §9.1.

**In-training probes.** `--eval-samples N` sizes two MAE probes
(eval_chunk_mae on holdout, train_mae on train), drawn exactly as
`bijou.eval --seed` would, sharded and all-reduced, CPU-resident.

**Checkpoint schema** (`loading.py` dataclasses): `expert.safetensors` +
`bijou_config.json` (backbone id, full ExpertConfig, per-dataset +
aggregate stats, train args, step); `optimizer.pt` for lossless
`--resume`; and `backbone.safetensors` (bf16 trunk snapshot) iff the trunk
was trained. Old checkpoints load unchanged (new config fields default at
the parse edge; `ensure_matching_expert_config` backfills defaults into
the saved dict before diffing). `--init-from` = warm start (config-guarded,
optimizer ignored); `--resume` = lossless continuation (optimizer +
scheduler; CLI lr ignored; cosine re-evaluated over the new `--steps`, so
extending re-heats LR — the owner accepts this when reusing moments, else
prefers init-from + warmup).

**Regression gates.** `check.py` (ruff + pyright + pytest, final verdict
line only). The **loss oracle**: a 2-step tiny-backbone CPU run must
reproduce **1.8896 / 1.7237** exactly after any change near the math
(tied to the current tiny-gemma4). gemma4 changes additionally gate on
`verify_parity`. Any new architecture path records its own oracle loudly.

**Step split** (H100, batch 64, measured): encode_prefix 79.3% / expert
fwd 4.6% / bwd 15.4% / opt 0.7%. The frozen backbone forward dominates —
expert width is nearly free wall-clock; the perf wins are prefix-side
(§9.8).

## 6. Inference and deployment

`bijou.rollout` on the physical SO-101 (`docs/rollout_so101.md`): laptop
RTX 3000 Ada (8 GiB) fits the bf16 backbone (4.77 GiB) + bf16 expert; peak
6.29 GiB; ~233 ms/replan warm. `--max-relative-target` is a lerobot per-
tick rate limiter (not a safety system); camera names are positional
prompt slots; task string must match the recorded instruction. Deployment
always fine-tunes on rig data first (zero-shot cross-rig transfer is the
wall, §8) — so the operative metric for any change is fine-tuned-then-
scored rig MAE, not zero-shot.

## 7. Empirical grounding (what shaped the design)

Open-loop chunk MAE (raw degrees, Heun-10, 256 frames, seed 0). Baselines:
state-copy 11.10 train / 10.30 community-holdout / 9.54 owner-rig.

- **Scale dominates architecture.** A matched 4-arm ablation (causal vs
  bidirectional, 140 vs 280 soft tokens, 4-4-8 vs all-deepest streams;
  20k→40k) moved 0.6–1.2 MAE per doubling of steps while no architecture
  variant separated from control where it counts. Kept: causal
  (bidirectional catastrophic cross-rig, 17.1 vs 13.3), 140 tokens
  (280 = 1.9×/step, edge reversed), 4-4-8 (streams0016's rig edge is a
  hint, §9.4).
- **Mainline lineage.** cont45k (v1v2v3, ~27M cumulative samples, frozen)
  scores 6.85 community-holdout / **11.71 owner-rig** (best frozen rig,
  vs copy 9.54 — still loses cross-rig). Episode-level generalization is
  ~free; cross-rig is the wall.
- **The wall is grounding, not action space.** Re-anchor probe: cross-rig
  error is a frame-dependent *level* mis-estimation (chunk shape is
  right); a per-frame constant offset would beat copy, but no single
  per-rig offset does → the model mis-localizes the working point
  *visually* on an unseen rig. This **falsified delta-actions** as the
  fix (§9.9) and pointed at the representation.
- **Acuity probe:** metric position is present at the expert's K/V inputs
  but thin (K4 10.8 → K14 17.3 px in-scene; 25–32 px cross-background)
  and NOT object-centric (task-object motion ≈ 1.9× a background patch's
  at K14). Motivates trunk adaptation (§9.1).
- **τ-diagnostic:** ~1 MAE recoverable integration error in-domain; OOD
  cost concentrates at high τ (initial placement from context);
  fine-tuning roughens mid-τ transport. Motivates adaRMS (§9.2) and
  sample-draws (§9.7).
- **Fine-tuning works and is always done.** From cont45k, rig ft beats
  copy on held-out rig episodes (first honest crossing); a paired ft
  from a +11.5M-sample pretrain beat a shorter one by 0.22 — pretrain
  quality carries downstream, which is why §9 changes are judged by
  fine-tuned rig MAE.

## 8. Directions under evaluation and proposed changes

Each subsection: the change, its justification, status, and the key
design decisions. Full blow-by-blow lives in git history (these subsume
the retired `plan_*.md`).

### 8.1 Trunk unfreezing (IMPLEMENTED; evaluating)

**Change.** `--text-lr` trains E2B text layers 0–14 + the multimodal
projector (embeddings and per-layer-embedding tables stay frozen — few
rows/step, dense Adam waste, cheapest forgetting control); `--vision-lr`
adds the tower (expected to stay off — the acuity probe puts position
sharpest at the tower output, so adaptation is needed *downstream*).
**Justification.** The in-dist-vs-cross-rig gap and the acuity probe both
localize the bottleneck in the text stack's use of visual tokens, not the
expert; π0/SmolVLA both train their trunks. **Numerics/plumbing.** fp32
masters + bf16 autocast; `BijouTrainStep` + single DDP wrap
(static_graph); `backbone.safetensors` rides in the checkpoint; frozen
path stays byte-identical (oracle exact). **Status/finding.** First A/B
(cont45k init, text-lr 2e-5, 15k→resumed 30k) ended ~7.2 community-
holdout — still *above* frozen cont45k's 6.85 in-distribution (feature
drift: the expert was tuned to frozen features). Verdict pending the
decision-relevant test: **paired rig fine-tune from the unfrozen vs the
frozen pretrain** — does the adapted trunk transfer better to the rig?
**Open dials.** trunk-LR grid; freeze-then-thaw if a warm-expert transient
appears; ZeRO-1 / activation checkpointing for larger batch; a full-thaw
budget only if cheap thaws pay.

### 8.2 adaRMS time conditioning (IMPLEMENTED; first arm in flight)

**Change.** `--time-conditioning adarms`: DiT-style per-layer modulation
of the expert by τ. A per-layer zero-init head `SiLU → Linear(hidden →
6·hidden)` yields, per sublayer, a **scale** on the pre-norm output
(`rmsnorm(h)·(1+γ)`) and a **gate** on the residual contribution
(`residual + g·sublayer`); a final-norm scale head too. **No shift β** —
that is a LayerNorm artifact (LayerNorm has a bias to condition); RMSNorm
is bias-free by design, so the additive τ-injection is supplied by the
gate instead, and scale modulates the one thing an RMSNorm output is (a
magnitude). **Identity at init** (γ=gate=0 ⇒ every block is the identity ⇒
zero velocity field), so it is a from-scratch-only architecture (cannot
warm-start additive→adarms; the guard enforces this). **Justification.**
The τ-diagnostic's ~1 MAE integration gap + mid-τ roughening: the marginal
field changes character along τ, and per-layer modulation lets each block
reweight by τ rather than propagating one input-added vector through 16
residual blocks. **Cost.** +~101M (~+25%), few % step time (expert is
~20% of the step). **Pre-registered signature.** Shrinks the Heun-5→30
gap and the mid-τ bump; chunk-MAE −0.1..−0.4; rig zero-shot unchanged
(grounding ≠ conditioning). Param-confounded (win = modulation OR
capacity) until a bottleneck-head follow-up. **Status.** Additive stays
default and byte-identical; first adarms run (bidirectional, E2B-width
1536, --fps 30, 40k) is in flight. Forward-compat: once AR co-training
shares the expert (§8.3), modulation must be MASKED to the flow positions
(per-position gating, not per-sample broadcast).

### 8.3 Autoregressive FAST-token co-training (tokenizer done; model proposed)

**Change.** Add a second training objective: causal next-token prediction
over **FAST** action tokens (DCT + BPE of the chunk, arXiv:2501.09747),
mixed with the flow loss, sharing the backbone. Follows π0.5 / knowledge-
insulation: next-token CE is the gradient source that shapes VLM
representations well; the flow expert stays the fast deployed decoder
(with its K/V **stop-gradient-insulated**, so the two objectives touch
disjoint params and the mixture weight collapses into the component LRs).
**Justification.** π0-FAST matches diffusion π0 at 5× less training compute
and follows language better; here it is primarily a *representation-
shaping* objective for the trunk, targeting the grounding wall (§7).
**Tokenizer (done):** owned DCT+BPE (`bijou/fast/`), fit on 1040 datasets
/ 4.9M chunks → `mcobzarenco/bijou-checkpoints/fast_tokenizer_v1` (vocab
1024, ~52 tok/chunk, recon MAE ~0.44° ≪ model error). Immutable artifact
(a refit changes token semantics); per-dataset quantile normalization
read from `stats.json`; constant-dim guard + normalized clip (parked
joints have ~0 span). **Architecture options** (decision pending the
unfreeze A/B):

| option | trunk-shaping | expert interference | new params | AR decode | note |
|---|---|---|---|---|---|
| A shared expert | K/V cross-attn (needs --text-lr) | real risk | ~3M | ~50–100 ms | smallest diff; duplicate the state token so flow/AR masks don't leak |
| B separate 6-layer decoder | K/V cross-attn | none | ~85–150M | ~50 ms | escape hatch; weakest prior |
| C full VLM (KV-sharing) | native LM circuit (π0.5-faithful) | none by construction | ~5M | ~150–250 ms | primary; deep half runs query-only against the cached layer-14 K/V — prefix never runs it, checkpoint/rollout unchanged |

C's frozen deep half is the risk (late layers are most language-
specialized; suffix tokens carry pad-PLE identity they never trained on).
Resolve by a **thaw dial** measured with cheap CE-convergence probes:
C0 frozen → C1 +deep RMSNorm scales (Lu-et-al. trick, proposed default) →
C2 +top-5 layers → C3 LoRA → C4 full (ZeRO-1). **Decision rule.** Unfreeze
A/B strong ⇒ the K/V pathway suffices ⇒ A is attractive for its cost;
weak ⇒ C tests the actual π0.5 mechanism. **Inference stays flow** (the AR
head is a training-time shaper + an eval diagnostic — `bijou-ar@step`
policy scoring the same frames).

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
a matched arm. (The in-flight adaRMS run already carries width 1536.)

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
