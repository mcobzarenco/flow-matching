# Plan: training with an unfrozen Gemma4 E2B trunk

Status: **IMPLEMENTED 2026-07-28** (`--text-lr` /
`--vision-lr` in bijou.train; snapshot/load in bijou.loading).
Validated on the tiny backbone: flags-off oracle EXACT (1.8896/1.7237),
flags-on grad-flow probe (`outputs/probe_unfreeze_gradflow.py`) all
green (partition exactness incl. stop-layer split, embeddings/PLE/tower
frozen, every trainable param receives finite grads — the DDP
static_graph precondition), checkpoint round-trips (new format, OLD
format byte-identical path, --init-from adapted trunk with flags on and
off), 2-rank gloo DDP smoke. Flags-on probe oracle: 1.5528 (seed 0,
2 dev items, warm out_proj). Note: a FRESH expert's zero-init
action_out_proj blocks trunk gradients on step 1 only (grads populate
as zeros — DDP-safe; flows from step 2). GPU memory/throughput
validation pending on the new box. Goal: continue from the best
checkpoint — `bijou_community_v1v2v3_cont45k_ddp4/step_045000`, on HF
with optimizer — with the text trunk live at a low learning rate:
shortest time to a potentially better checkpoint. Not a matched
ablation round; the LoRA alternative from the original plan is dropped
(hand-rolled adapters inside parity-tested `gemma4/`, strictly less
expressive, and its forgetting-control motivation is weak here — the
expert-only rig ft already costs +0.85 community MAE and we accept it).

## Why now (evidence since the original plan)

The original precondition — "run the flag ablations first" — is met
(4-arm round: scale dominates, no architecture change earned its cost).
Three probes localize the remaining error in the representation/its
use, not the action parameterization:

- Re-anchor probe: cross-rig error = frame-dependent *level*
  mis-estimation; chunk shape is right. Visual grounding, not action
  space.
- τ-diagnostic: OOD cost concentrates at high τ (initial placement from
  context), where the prefix representation is the only input signal.
- Vision acuity probe: position is sharpest at the vision tower's
  output (8.4 px linear readout) and **degrades through the LM layers**
  (K4 10.8 → K9 15.4 → K14 17.3 px; cross-background 25–32 px), and the
  representation is not object-centric (task-object motion barely
  outweighs background motion at K14). The trunk's *text stack* is
  where position dies — exactly what text-layer training addresses.

## 1. What unfreezes (CLI surface)

Two optional flags, both default 0.0 = today's behavior, bitwise:

- `--text-lr LR`: text decoder layers 0–14 (attention, MLPs,
  norms) + the multimodal projector (`embed_vision`), as one param
  group at LR. **Token embeddings and per-layer-embedding tables stay
  frozen** — few rows touched per batch, dense Adam state for a 262k
  vocab is pure waste, and frozen embeddings are the cheapest
  forgetting control. (A future `--unfreeze-embeddings` flag is
  possible; no current need.)
- `--vision-lr LR`: adds the vision tower (~120M params —
  the encoder-free E-series pipeline, not the 400M SigLIP of the
  original draft) + its position table as another param group.
  **Expected to stay 0**: the acuity probe shows the tower's output is
  the *best* positional stage; adaptation is needed downstream of it.
  When 0, the vision tower runs under `no_grad` (soft tokens are
  inputs to the text stack, so autograd never needs the tower's graph
  — saves its activations and any backward through it).

Guards: flags require `--holdout-episodes` unchanged semantics
(nothing new); `--vision-lr > 0` without text unfreezing is
permitted (legitimate arm) but prints loudly. Expert LR/schedule
unchanged; all groups share the cosine schedule + warmup, scaled by
their group LR. Continuing from cont45k means the expert is *warm* —
the original plan's "cold expert backpropagating garbage into a live
trunk" risk mostly evaporates; `--warmup-steps 500` (the owner's
standard init-from recipe) is expected to suffice. A trunk-LR zero-hold
(freeze-then-thaw) is a contingency, not built until a 200-step probe
shows instability.

Suggested starting point: `--text-lr 1e-5` (grid 5e-6..2.5e-5
if budget allows) with `--lr 1e-4`, `--grad-clip 10.0`. The original
plan said 1.0; MEASURED live-trunk norms (10k run, 2026-07-28) are
p50 3.8 / p90 6.4 / max 19.2 — clip 1.0 renormalized 100% of steps by
~4× (moment-estimate noise from the varying rescale, and decoupled
weight decay silently ~4× stronger relative to the gradient) instead
of catching outliers. 10.0 preserves the lineage's never-bites
safety-net semantics. Weight decay: keep 1e-5 on matmul weights, 0 on
norms (param-group filter by name).

## 2. Gradient path (verified, unchanged from original)

The expert consumes exported K/V of global layers {4, 9, 14};
`KVCache.update()` is functional during prefill (stores/returns the
computed tensors, no in-place writes), so autograd flows through
`cache.layers[i].keys/values` as-is. Required changes:

- `BijouModel.encode_prefix` + `data.encode_prefix`: `torch.no_grad`
  becomes conditional (parameter, not global state). Inference/eval
  keep no-grad unconditionally.
- `embed_multimodal`: vision tower call wrapped in `no_grad` when the
  tower is frozen; PLE/embedding lookups stay under grad (cheap) but
  their tables have `requires_grad=False`.
- `kv_stop_layer` (stops prefix encode after the deepest exported
  layer's K/V) is grad-transparent — keep.
- Unit tests (tiny backbone, CPU): nonzero grad on a layer-4 K/V
  projection and on a vision-tower linear iff the respective flag is
  on; zero otherwise; expert grads bitwise-identical with both flags
  off (loss oracle 1.8896 / 1.7237 must hold EXACTLY with flags off;
  flags-on gets its own loudly-recorded oracle values).

## 3. Numerics: fp32 masters + bf16 autocast (decision pinned)

Direct bf16 updates at LR ~1e-5 vanish below bf16 resolution. Choice
(a) from the original plan: when any unfreeze flag is on, load the
trunk fp32 and run prefix encode under `torch.autocast(bf16)`. The
expert stays fp32-with-TF32 *outside* the autocast region (narrower
than the original draft — cleaner attribution, oracle stability); the
expert already casts incoming K/V streams to its own dtype. Frozen
mode (both flags 0) keeps today's bf16 load path — bitwise identical,
regression-gated. gemma4 HF-parity tests untouched (inference path
unchanged).

## 4. Memory budget (per H100-80GB, batch 32/rank, text-only unfreeze)

Trainable ≈ decoder layers + projector ≈ **~1.8B** (verify the exact
count at implementation; embeddings/PLE frozen cut it well below the
original 2.15B estimate).

| item | est. |
|---|---|
| trunk fp32 masters | ~7.2 GB |
| trunk fp32 grads | ~7.2 GB |
| Adam moments ×2 fp32 | ~14.4 GB |
| frozen: vision tower + embeddings/PLE (bf16) | ~1.5 GB |
| expert fp32 (params+grads+Adam) | ~6.5 GB |
| activations (bf16, batch 32, ~450 tok, 15 layers; tower no-grad) | ~6–12 GB |
| **total** | **~43–49 GB** — fits with headroom |

Levers if a bigger batch is wanted: ZeRO-1
(`ZeroRedundancyOptimizer`, −~11 GB/rank), activation checkpointing
per trunk layer (~30% step cost), gradient accumulation. 8-bit Adam
stays last resort (new dependency).

## 5. Distributed training

DDP currently wraps only the expert. With the trunk live, introduce
`BijouTrainStep(nn.Module)` owning the BijouModel: `forward(batch,
noise, tau)` = prefix-encode (with grad) + expert velocity; DDP wraps
that when any unfreeze flag is on (`gradient_as_bucket_view`,
`static_graph=True` — fixed layer usage). Frozen mode keeps the
current expert-only wrap, byte-identical, regression-tested. Find
-unused-parameters stays off (all wrapped params participate:
embeddings/PLE are excluded from the wrap by `requires_grad=False`).

## 6. Checkpoint format (existing checkpoints keep loading — hard requirement)

Current schema is the `loading.py` dataclasses (CheckpointMetadata /
CheckpointInfo / CheckpointTrainArgs). Changes:

- When any trunk param trained: write `backbone.safetensors` (bf16
  cast of the fp32 masters — full truncated-trunk state, ~4.3 GB) next
  to `expert.safetensors`. Metadata gains OPTIONAL fields
  (`text_lr`, `vision_lr`) with defaults so old
  json parses unchanged; train args ride along as they already do.
- `from_checkpoint`: if `backbone.safetensors` exists, load it over
  the HF-resolved truncated backbone; else exactly today's behavior.
  Old checkpoints: no new file, no new fields → byte-identical load
  path. Eval/rollout pick up adapted trunks transparently (laptop
  VRAM unchanged: bf16 trunk is the same size).
- `--init-from` a trunk-trained checkpoint with flags at 0 = freeze
  the adapted trunk (adapt once, iterate experts cheaply). The
  ExpertConfig guard is untouched (unfreeze flags are train args, not
  architecture) — so cont45k init-from with unfreeze flags on is
  allowed by design, which is the whole point.
- `optimizer.pt` grows to ~30–35 GB (fp32 masters + moments). Bump
  `--save-every`, and delete optimizers of non-final steps as the run
  progresses (the owner already prunes these manually).

## 7. Throughput expectation

Today's step split: prefix fwd 79% (of which vision ~1/3), expert
fwd+bwd 20%. Text-only unfreeze adds ≈2× the text-trunk forward as
backward, vision stays forward-only no-grad: expect **~2–2.3× step
time** at the same batch (measure on the new box; the original 2–2.5×
estimate stands). 20k steps DDP4 at batch 64/rank ≈ one overnight.

## 8. Validation ladder (before any long run)

1. Unit grad-flow tests + flags-off oracle EXACT (CPU tiny backbone);
   record flags-on oracle values loudly.
2. Memory probe at target batch on one GPU; report peak.
3. 2-rank gloo smoke for the DDP wrap switch.
4. 200-step run from cont45k init: loss should start near cont45k's
   final (~0.089) and not spike; compare against a frozen 200-step
   control from the same init (loss continuity, grad norms per group).
5. The real run: frozen-continue vs `--text-lr 1e-5`
   continue, same budget from the same init, scored on community
   holdout/train + rig zero-shot + full rig-holdout after a paired rig
   ft. Decision metric: does trunk adaptation move holdout AND
   rig-side numbers (representation win), or only train-side
   (capacity memorization)? Re-run the acuity probe on the adapted
   trunk — K-stage RMSE and the object-centricity ratio are now
   baseline-ed (17.3 px / 1.9×) and should move if the mechanism is
   what we think.

## 9. Risks

- Feature drift under a warm expert: the expert was trained against
  frozen features; the trunk moving underneath it can transiently
  worsen loss. Low trunk LR + shared warmup mitigates; step-4 probe
  catches it.
- Language/vision forgetting: accepted for this deployment (narrow
  instruction distribution); frozen embeddings help; not otherwise
  mitigated. If instruction diversity ever matters, revisit.
- Checkpoint bloat / slower loops: ~35 GB optimizers, prune
  aggressively.
- Silent parity regression: inference path untouched; gemma4 parity
  tests stay as-is; flags-off oracle is the tripwire.

## Order of work

1. Grad-path conditionality + `requires_grad` partitioning + unit
   tests (self-contained).
2. `BijouTrainStep` + DDP switch + param groups + autocast + fp32
   trunk load; flags-off oracle check.
3. Checkpoint write/read (`backbone.safetensors` + optional metadata
   fields); round-trip test incl. an OLD checkpoint from HF.
4. Memory/throughput probe on the next box; then ladder steps 4–5.

Estimate: a focused day of implementation + validation (most of it in
`train.py`/`loading.py`; ~zero changes inside `gemma4/`), then one
overnight for the A/B.
