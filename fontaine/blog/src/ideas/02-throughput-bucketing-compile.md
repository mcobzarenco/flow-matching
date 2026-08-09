# 2. Throughput: bucketed batching + torch.compile on the frozen prefix — `screening` (2a landed 2026-08-05; GPU A/B conditional)

*Tag: `throughput-compile` · idea #2 · [index](../ideas.md)*

- **Hypothesis:** length-bucketed batching + `torch.compile` of the
  prefix encode (79% of step time) buys ≥20% step-time on 1×H100 —
  compounding interest on every later run.
- **2a LANDED (2026-08-05, [post](../posts/2026-08-05-bucketing-impl-sim.md)):**
  `--bucket-by-length` (default OFF) — `LengthBucketedBatchSampler`,
  camera-count keys, oracle-gated (3 CPU oracles bit-exact, gradflow
  green, 6 unit tests). **Sim finding: under the current recipe
  (`--camera-counts 1 2`) padding inflation is only +5.09% → ceiling
  ~3.6% step-time — below the <5% deprioritize line ⇒ NO GPU screen
  for current lineages.** Full-corpus census (3–4-cam datasets in):
  +32.55% → −23.8% padded tokens, ~19% ceiling. Conditional pre-reg
  in the post: first widened-selection run family runs the 1k-step
  A/B before adopting; paired arms must share the flag.
- **Cost remaining:** 2b (compile) — real implementation vs the
  blocker map below; decoupled from bucketing under narrow census
  (shape variance is text-jitter ⇒ pad-to-fixed-length).
- **Falsification (2b):** measured s/step and samples/s on identical
  configs, before/after, on THIS box. If <10% combined, bank the
  numbers and deprioritize.
- **Implementation notes (deep-dive 2026-08-05):** compile blockers
  on the prefix path: `pooled[valid_mask]` dynamic shape
  (vision.py:606), host syncs + `masked_scatter` (masks.py:132,
  model.py:196-204), KVCache `torch.cat` mutation, dense additive
  masks. No prefix attention takes the flash path today (sliding =
  always-masked, global head_dim 512 > fused cap). Bucketing is a
  compile prerequisite. Bonus levers: skip K/V writes for
  non-exported layers when `retain_cache=False`; cache frozen-run
  probe prefix encodes (bit-identical across evals).
- **2026-08-08:** molmo2 perf pass-1
  [pre-reg finalized](../posts/2026-08-08-prereg-molmo2-perf-pass1.md)
  (S-bundle off the perf review): its P3 sync removals are exactly
  the molmo2-side compile blockers (host syncs in model.py/text.py +
  boolean-index `nonzero` in the chunked losses) — 2b's molmo2 prep
  now rides that bundle. Static-shapes question answered in the
  review: keep dynamic (+5.09% padding ceiling stands).
- **2026-08-09:** box ladder
  [FALSIFIED the bundle](../posts/2026-08-09-perfpass1-box-results.md)
  (C −7.3%, P1 cuDNN −10.8% on the true 4×DDP recipe — kernel
  microbenches don't predict end-to-end under comms overlap). The
  P3 sync removals 2b wanted stay bitwise-proven and ride the
  hygiene subset item (`molmo2-perf-pass1-subset-landing`), with no
  speed claim.
