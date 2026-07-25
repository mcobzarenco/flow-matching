# Plan: training with an unfrozen Gemma4 E2B trunk

Status: **proposal, not implemented**. Design for letting gradients from the
flow-matching loss update the VLM prefix encoder (currently frozen bf16,
layers 0–14 of E2B + vision tower + projector + embeddings, ~2.55B params).

## Why (and why it might not help)

For: the trunk's features are generic web-VLM features; adapting them to
robot imagery/instructions is the single biggest capacity unlock left
(π0 and SmolVLA both train their VLM trunks). The in-dist vs held-out gap
suggests representations, not the expert, are the bottleneck.
Against: 27M frames of robot data can erode language grounding
(catastrophic forgetting), cost triples-ish, and the expert-only recipe is
not yet exhausted (bidirectional/soft-token/stream ablations pending).
Recommendation: run the flag ablations first; unfreeze second.

## 1. Gradient path (the model-specific part)

The expert consumes the prefix as exported K/V of global layers {4, 9, 14}.
Verified: `KVCache.update()` is functional during prefill — it stores and
returns the exact computed tensors (no preallocated buffers, no in-place
writes), so autograd flows through `cache.layers[i].keys/values` as-is.

Required changes:

- `encode_prefix` (bijou/data.py) and `BijouModel.encode_prefix`: make the
  `torch.no_grad()` conditional (`requires_grad` mode flag or a
  `with_grad: bool` parameter). Inference/eval keep no-grad.
- `embed_multimodal` (vision tower + projector + PLE): same conditionality.
- Keep two K/V paths: the cache path (inference, incl. cached decode) and
  the training path. Since prefill-cache storage is already functional,
  the SAME path works for training — no new collection mechanism needed.
  Add a unit test asserting nonzero grads on a layer-4 K/V projection and
  on the token embedding after one training step (tiny backbone), and zero
  grads with the flag off.
- Sliding-window layers slice their stored K/V (`[-window+1:]`) — irrelevant
  for training (streams are global layers; prefill length < window anyway),
  but the slice is also functional, so no hazard.

## 2. What to unfreeze (staged)

- **Stage A (default)**: text layers 0–14 + per-layer-embedding tables +
  multimodal projector. Vision tower (SigLIP, ~400M) stays frozen —
  robot images are natural images; saves ~1/3 of trunk activation memory.
- **Stage B (flag)**: `--vision-lr` > 0 adds the vision tower (+ separate
  param group).
- **Alternative track**: LoRA on q/k/v/o + MLP of the text layers
  (rank 16–64) — ~50M trainable, no optimizer-state explosion, checkpoint
  stays small. Worth having as a cheap arm in the same ablation; the plan
  below is for full fine-tuning, LoRA piggybacks on the same grad-path work.

## 3. Numerics: mixed precision done properly

The trunk is bf16. Direct bf16 updates are unusable at trunk lr (~1e-5
relative updates vanish below bf16's ~4e-3 resolution). Standard fix:
**fp32 master weights in the optimizer, bf16 compute copy** — i.e. either
(a) hold trunk params in fp32 and autocast the forward to bf16, or
(b) keep bf16 params + maintain fp32 masters manually.
(a) is torch-idiomatic: load the trunk fp32 (`dtype=torch.float32` in
`load_model`), wrap prefix encode + expert forward in
`torch.autocast(bf16)`. The expert itself stays fp32-with-TF32 (unchanged
semantics under autocast — revisit later).

## 4. Memory budget (per H100, batch 32/rank)

| item | size |
|---|---|
| trunk fp32 masters (2.15B text-side) | 8.6 GB |
| trunk grads fp32 | 8.6 GB |
| Adam moments fp32 ×2 | 17.2 GB |
| frozen vision tower bf16 | 0.9 GB |
| expert (params+grads+moments fp32) | 6.5 GB |
| activations (bf16 autocast, batch 32, ~450 tok, 15 layers) | ~8–15 GB |
| **total** | **~55–60 GB** ✓ fits, tight |

Levers if it doesn't fit at the desired batch:
1. `ZeroRedundancyOptimizer` (ZeRO-1): shards Adam moments across the 4
   ranks → −13 GB/GPU. Cheap to adopt, first lever to pull.
2. Activation checkpointing per trunk layer (`torch.utils.checkpoint`),
   ~30% step-time cost, −70% trunk activation memory.
3. Batch 16/rank + (new) gradient accumulation ×2.
4. 8-bit Adam for the trunk group (bitsandbytes) — new dependency, last
   resort.

## 5. Optimization recipe

- **Param groups**: expert lr 1e-4 (unchanged); trunk lr `--trunk-lr`
  (default 0.0 = frozen — the single CLI switch; suggest 1e-5 to 2.5e-5),
  same cosine schedule, shared warmup (raise to ~1000 even for fine-tunes —
  a cold expert backpropagating garbage into a live trunk early is the
  main destabilization risk; consider trunk-lr zero-hold for the first
  500 steps: freeze-then-thaw inside one run).
- Grad clip: tighten global clip to 1.0 when the trunk is live (10.0 was
  calibrated for the expert alone).
- Weight decay: keep 1e-5 on matmul weights, 0 on norms/embeddings
  (param-group filter by name).

## 6. Distributed training changes

DDP currently wraps only the expert; with the trunk live, both modules'
grads must be produced inside one wrapped forward for bucket accounting.
Introduce `BijouTrainStep(nn.Module)` owning the BijouModel whose
`forward(batch tensors, noisy_actions, tau)` runs prefix-encode (with
grad) + expert velocity; DDP wraps that when `--trunk-lr > 0`
(`gradient_as_bucket_view`, no buffer broadcast). Frozen mode keeps the
current expert-only wrap — byte-identical behavior, regression-tested.
`static_graph=True` is safe (fixed layer usage) and helps with the large
bucket count. Backbone buffers (RoPE tables) constant as before.

## 7. Checkpoint format (breaking-ish, kept backward compatible)

- New per-checkpoint file `backbone.safetensors` (trunk weights, saved
  bf16 ~4.3 GB; fp32 masters live only inside `optimizer.pt`, which grows
  to ~35 GB — bump `--save-every`, consider keep-last-k pruning).
- `bijou_config.json` gains `"trunk_trained": true` + trunk lr.
- `loading.from_checkpoint`: if `backbone.safetensors` exists, load it
  over the HF backbone after truncation; else current behavior. Old
  checkpoints remain loadable unchanged; eval/rollout pick up trained
  trunks transparently.
- `--resume`/`--init-from` follow existing semantics; `--init-from` a
  trunk-trained checkpoint with `--trunk-lr 0` = freeze the adapted trunk
  (useful: adapt once, iterate experts cheaply afterwards).

## 8. Throughput expectation

Backward through the trunk ≈ 2× its forward; step time roughly 2–2.5×
current (~1.15 s → ~2.4–2.9 s at batch 64 global-256 equivalents, or hold
~1.5 s at batch 32/rank). 20k steps ≈ 14–17 h on the 4× box. Budget one
overnight per arm.

## 9. Validation ladder (before any long run)

1. Unit: grad-flow assertions (tiny backbone) — nonzero trunk grads with
   flag on, zero with flag off; expert grads unchanged bitwise with flag
   off (regression oracle 1.9777).
2. Memory probe: one H100, target batch, report peak.
3. 2-rank gloo smoke (mechanics under DDP wrap change).
4. 200-step 4-GPU run: loss continuity vs frozen baseline at same init
   (first ~20 steps should track closely; divergence pattern sanity).
5. A/B 20k: frozen vs `--trunk-lr 1e-5` from the same `--init-from`,
   scored with `bijou.eval` on in-dist + held-out-rig + (once built)
   held-out episodes. Held-out is where trunk adaptation should pay;
   if it only moves in-dist numbers, it's memorizing with more capacity.

## 10. Risks

- Language/vision forgetting → worse novel-instruction following
  (mitigate: low trunk lr, freeze embeddings, LoRA arm as control).
- Early-training destabilization from cold-expert gradients (mitigate:
  freeze-then-thaw).
- Checkpoint bloat and slower iteration loops.
- The KV-export training path silently regressing inference parity —
  keep the gemma4 HF-parity tests untouched (inference path unchanged).

## Suggested order of work

1. Grad-path conditionality + unit tests (small, self-contained).
2. `BijouTrainStep` + DDP wrap switch + param groups + autocast (+ fp32
   trunk load), regression-oracle check.
3. Checkpoint format + from_checkpoint loading.
4. ZeRO-1 + (if needed) activation checkpointing.
5. Memory/throughput probes, then the A/B.

Estimated implementation: a focused day including validation, most of it
in `train.py` + `loading.py`, ~zero changes inside `gemma4/` (the cache
path already supports it).
