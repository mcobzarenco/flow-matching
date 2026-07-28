# Plan: autoregressive FAST action tokens + flow matching (co-training)

Status: **design agreed, tokenizer implemented, model work not started**
(2026-07-28). Owner intent: a shared backbone trained by a mixture of
autoregressive FAST-token prediction (causal CE) and the existing
flow-matching expert loss, following the π0.5 / knowledge-insulation
finding that next-token CE is the gradient source that shapes VLM
representations well, while the flow expert stays the fast deployed
decoder.

References: FAST (arXiv:2501.09747) — DCT + BPE action tokenization,
~30 tokens per 1s chunk, π0-FAST matches diffusion π0 at 5× less
training compute; π0.5-KI — train the VLM on FAST CE, insulate the flow
expert behind stop-gradient.

## 1. Tokenizer artifact (implemented: `bijou/fast/`)

Owned reimplementation (no trust_remote_code): orthonormal DCT-II
matrix over time per dim → round(×scale) → freq-major flatten → chars →
plain BPE (no ByteLevel). Deviations from the reference and local
measurements (21–25 tokens/chunk, 12–14× compression, 0.37–0.68° recon
MAE on 5 local datasets): see handoff §6 and `tests/test_fast.py`.

**The artifact is a directory, versioned immutably** (a refit changes
token semantics even at identical BPE — models and tokenizer version
are permanently coupled):

    fast_tokenizer_v1/
      fast_config.json      # scale, alphabet, H, D
      bpe.json              # BPE vocabulary/merges
      quantile_stats.json   # per-dataset q01/q99 (fit normalization)
      fit_report.json       # provenance: datasets, chunks, fidelity

Quantile lifecycle (settled 2026-07-28 after two owner corrections:
tokenizer-owned table → checkpoint-owned table → **dataset-owned,
native schema** — each strictly simpler):

- **Newer lerobot already computes q01/q10/q50/q90/q99 natively** in
  `meta/stats.json` (the owner's so101_pick_place_v2 has them); the
  community conversions predate that and carry only
  count/max/mean/min/std. So: BACKFILL the native keys (action +
  state) into the converted datasets — one parquet scan each via the
  fit CLI's `--backfill-stats` mode — and re-upload (3 collection
  repos, stats.json files only). Methodology note (verified in lerobot
  source): native dataset-level quantiles are a count-weighted MEAN of
  per-episode quantiles — quantiles don't compose by averaging, so
  extremes regress toward the median (measured −54° vs exact −120° on
  rig v2) — i.e. native values are WRONG for corpus use and the
  backfill force-corrects them everywhere, rig datasets included.
  mean/std/min/max aggregate correctly (parallel variance) — the flow
  lineage's MEAN_STD normalization is untainted.
- TRAINING/EVAL: quantiles ride items via DatasetStats /
  StatsAttachedDataset — the EXACT mean/std mechanism (parse-edge
  `None` for un-backfilled datasets; AR training fails loudly with
  "backfill" as the remedy; flow-only training unaffected).
- CHECKPOINTS: per_dataset_normalization carries the quantiles for
  free once DatasetStats does; ROLLOUT resolves the rig from the
  checkpoint like --stats-repo-id today. No aggregate fallback for AR.
- FIT: reads quantiles FROM stats.json when present (fit normalization
  ≡ training normalization, same source), computes+warns otherwise;
  the tokenizer dir's `quantile_stats.json` is provenance only.

Fit CLI (`python -m bijou.fast`): fits on FULL corpora (BPE training
measured near-linear, ~0.8s/7.3k chunks → minutes for ~5M chunks);
`--max-chunks` caps for experiments. Run on the 2×H100 box once the
datasets (community ×3 + the two rig sets from the hub) land; upload
the directory to `mcobzarenco/bijou-checkpoints/fast_tokenizer_v1/`.
Chunks are full in-episode windows (stride 5 default); training-time
tokenization of episode-tail chunks (padded actions) is an open detail
— candidate: repeat-last-frame before encoding, mask CE on nothing
(tokens are a joint code for the whole chunk; padding is invisible).

## 2. Architecture options (decision pending r2 evidence)

Common to all: prefix encode unchanged; flow expert unchanged and
bitwise-identical when the AR stream is off; FAST tokens wrapped in
BOA/EOA; ragged token sequences (measured 21–45) padded per batch with
CE label masking; state is NOT fed to the AR stream initially (π0-FAST
feeds 256-bin state as prompt text — a collator-only add-on arm later).

### Option A — shared expert (smallest diff)

Suffix `[s_flow][f_1..f_50][s_ar][t_1..t_k]` through the existing
ActionExpert; new params: token embedding + LM head (~3M).

- **Duplicated state token** is the load-bearing detail: today's mask
  has state attending everything, so one shared state token would leak
  AR content into flow queries transitively (and couple the branches'
  compute). With `s_flow` (today's semantics, τ added to f only) and
  `s_ar` (attends itself; attended by t_i) the two blocks are mutually
  invisible: dropping the AR block reproduces today's expert EXACTLY
  (oracle intact, r2 checkpoints warm-startable), and AR decode
  materializes only `[s_ar][t_*]` — no flow positions, no τ. ~50–100ms
  per chunk through 404M with prefix precomputed.
- Trunk-shaping pathway: cross-attention over exported K/V only (needs
  `--text-lr`). Same interface flow gradients use — r2 is the live test
  of that pathway's sufficiency.
- Risk: **weight interference** — one 404M net serves a τ-conditioned
  vector field and discrete sequence modeling; flow loss is not
  plateaued. Signature: flow probe MAE degrades vs a no-AR control.

### Option B — separate small AR decoder (escape hatch)

`ActionTokenDecoder`: token embedding + state anchor + 6 ExpertLayers
(hidden 768–1024, own cross_attention_schedule over {4,9,14}) + fresh
head; ~85–150M params. Zero interference, no trunk surgery, but no
pretrained sequence prior AND K/V-only trunk pathway — the weakest
theoretical story. Build only if A interferes and C is too heavy.

### Option C — full VLM via KV sharing (π0.5-faithful; primary candidate)

The E-series gift: layers 15–34 carry NO K/V weights — they run
query-only against layer 13/14's K/V, which the 15-layer prefix encode
already caches. Therefore:

- **Prefix never runs the deep half**; only the ~30–60-token suffix
  passes through all 35 layers (FLOPs negligible). `kv_stop_layer`
  stays for flow-only paths.
- **No vocab surgery**: suffix action tokens enter as a fresh
  nn.Embedding, riding the pad-token PLE row — the exact mechanism
  image soft tokens already use — with a fresh 1024-way head on the
  final hidden (~5M new params).
- Memory: deep half loads frozen bf16, ≈ +4 GB/rank (~0.7B non-PLE +
  ~1.3B PLE slices). Checkpoints unchanged: deep half stays HF-identical
  while frozen, so `backbone.safetensors` still stores layers 0–14
  only; rollout keeps loading truncated (flow needs no deep half).
- Gradients: CE at the head → through the deep residual stream → into
  live layers 0–14 via BOTH the suffix hiddens and the prefix's
  layer-13/14 K/V — the native next-token circuit, zero flow-expert
  interference by construction.
- Plumbing exists: `load_model(truncate_layers=None)`, SharedKV path,
  cache continuation — all parity-tested against HF on the full model.
- Inference: AR decode ≈ 30–60 sequential full-depth single-token steps
  (~150–250 ms class) — an eval diagnostic, not the deployed decoder.

**The frozen-deep concern (owner-raised, legitimate)**: late layers are
the most language-specialized (output-oriented, logit-lens); suffix
tokens carry pad-PLE identity the deep half never trained on; and live
layers 0–14 drift under a frozen deep half that cannot co-adapt. The
additive residual stream argues the failure mode is "wasted depth, not
damage" (layer-14 content stays recoverable), and the
frozen-pretrained-transformer line (Lu et al.) + prefix-tuning suggest
norm-scale training + trained input embeddings recover most adaptation.
Resolve by measurement via the **thaw dial**:

| dial | trainable added | memory/rank | note |
|---|---|---|---|
| C0 deep frozen | 0 | 0 | baseline |
| C1 + deep RMSNorm scales + final norm | ~0.3M | ~0 | Lu-et-al. trick; proposed default |
| C2 + top-5 layers | ~175M | ~3 GB | thaws the most language-tuned part |
| C3 + LoRA on deep q/o+MLP | ~30M | ~0.5 GB | LoRA's real niche; hand-rolled adapters in gemma4 |
| C4 deep fully live | ~0.7B | ~11 GB | needs ZeRO-1 / smaller batch; π0-FAST-faithful |

## 3. Losses, insulation, CLI surface

- `L = L_flow + L_CE`, but under full insulation (flow reads DETACHED
  K/V, per π0.5-KI; default on) the objectives touch disjoint
  parameters and the mixture weight collapses into the existing
  component LRs — no λ flag until a non-insulated arm needs one.
- Planned flags (component-lr family): `--fast-tokenizer PATH` (enables
  the AR stream), `--ar-lr` (fresh embed/head — fresh params want more
  than a pretrained trunk's 1e-5; default = expert lr), `--ar-deep
  {frozen,norms,top5,lora,full}` (option C dial), `--insulate-expert`
  (default true). Eval: token CE + exact-decode rate + a
  `bijou-ar@step` policy in bijou.eval decoding chunks for the same
  paired chunk-MAE tables (AR vs flow vs copy on identical frames).

## 4. Pre-registered signatures and decision rules

- r2 (flow-grads-through-K/V, in flight) strong ⇒ pathway proven ⇒ A
  becomes attractive for its cost; r2 weak ⇒ C is the design that
  tests the actual π0.5 mechanism.
- AR CE convergence probes (2–3k steps, small batch, 2×H100 box, after
  tokenizer fit): C0 vs C1 vs C2 vs shallow-exit (head on layer 14 —
  arbitrates whether the deep prior helps at all, i.e. C vs A/B
  premise). CE plateaus high at C0 but drops at C1/C2 ⇒ owner's
  frozen-deep concern confirmed, pick the paying dial.
- Interference check (A only): flow probe MAE vs a no-AR control from
  the same init. Degradation ⇒ escalate to B/C.
- Malformed AR decodes (FastDecodeError rate) logged at eval; >1% at
  convergence ⇒ constrained decoding or BOA/EOA handling bug hunt.

## 5. Order of work

1. Tokenizer fit CLI (done with this plan) → corpus fit on the 2×H100
   box (community + rig from hub) → upload `fast_tokenizer_v1`.
2. Collator: per-item token targets (quantile lookup, BOA/EOA, ragged
   padding + label masks).
3. Option C skeleton behind `--fast-tokenizer`: full-depth load path,
   suffix forward with cache, fresh embed/head, C0/C1 dial; flags-off
   oracle EXACT; new grad-flow probe arm.
4. CE-convergence probes (the §4 ladder) on the 2×H100 box; pick the
   dial; then the co-training run on the 8×A100 box after r2 concludes
   and is scored.
5. bijou.eval AR policy + report integration.
