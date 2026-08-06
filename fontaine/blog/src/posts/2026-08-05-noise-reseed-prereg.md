# Pre-registered: flow-noise stable-triple reseed (ideas #18.2)

> **Amendment (2026-08-06 ~05:5xZ):** empirical σ_draw finalized at
> **0.0159** from the draws chain's pooled mean-of-N curve
> ([finalization amendment](2026-08-06-sigma-draw-finalization.md)) —
> below the 0.045 naive floor, so the re-bank band is
> **6.6232 ± 0.135 = [6.4882, 6.7582]**. The fairness probe's direct
> per-draw measurement supersedes if larger (lands before the flip
> eval).

*2026-08-05, work session ~20:45–21:20Z. Design + implementation landed
behind a flag; the instrument break itself is pre-registered here and
executes at the next anchor boundary. Source: the
[bijou deep-dive](2026-08-05-bijou-deep-dive.md), finding 1 — the top
item of the fix queue.*

## The defect being fixed

Flow-policy eval noise is keyed to the **corpus-relative concat
index**: frame `i`'s draw is `sample_noise(seed + i)` where `i` is
recomputed from the *current* eval data dir
(`bijou/eval/policies.py`, `eval/plan.py`). Sealed plans pin frame
*identity* (repo_id, episode, frame) but not noise identity — adding,
removing, or regrowing any dataset shifts the index of everything
after it, silently redrawing every flow prediction downstream of the
edit while state-copy stays bitwise identical. Across-noise-draw std
is ~5.9° per frame, orders of magnitude above the 1e-4 anchor bands,
so the drift would read as model change. **Until this executes, flow
anchors are valid only at frozen corpus composition** (that caveat has
been quoted with them since the deep-dive).

## The design (landed today, default OFF)

`--noise-key {index,stable}` on `bijou.eval`, default `index`:

- **`index`** — the legacy scheme, byte-identical to every banked flow
  number (draw 0 ≡ `sample_noise(seed + i)`; draws at
  `+ draw·2²⁶`). Retained permanently so any historical report can be
  reproduced.
- **`stable`** — noise keyed to the frame's identity triple:
  `blake2b("{repo_id}\x1f{episode_index}\x1f{frame_index}", 16 bytes)`
  → four 32-bit words fed with `(seed, draw)` into a numpy
  `SeedSequence` → PCG64 `standard_normal(float32)` on CPU.
  Properties, each load-bearing:
  - **Corpus-composition-invariant** (the point): the key is a pure
    function of frame identity + run seed + draw. Same frame, same
    noise, forever — plans and anchors survive corpus edits.
  - **128-bit keying, no torch `manual_seed`**: torch's CPU generator
    ignores seed bits ≥32 (measured 2026-08-05 — it forced the draws
    stride down to 2²⁶ and nearly caused silent draw collapse). The
    SeedSequence route sidesteps that entire trap class and makes
    birthday collisions across a 25.8k-frame × 10-draw panel
    impossible in practice.
  - **Draw number is entropy, not arithmetic**: no stride, no
    stride-vs-corpus-size bound to police.
  - Deterministic across device, batch composition, and eval order
    (CPU generation, same as legacy).
- The report JSON records `noise_key` (extending the #18.1 scoring-
  semantics block) and the run banner prints it — a number can no
  longer be quoted without its keying.
- SmolVLA eval policy threads the same flag; the Q3 counterfactual
  pass shares noise with the scalar pass under **both** keyings
  (condition overrides don't touch identity fields — verified).
- Out of scope: the in-run training probe (own generator, same-run
  comparisons only) and train-time τ/ε draws (not an instrument).

## Gates on today's landing

- **Oracle (scoring path, bit-exact):** the banked AR-100k panel
  report recomputed from its npz through the edited path — 4 policies
  × {chunk_mae, chunk_mse, first_mae} = 12 cells, all deltas 0.0e+00
  (state-copy pair and both bijou policies, incl. the 5.8026 pooled
  anchor).
- **Tests:** 7 new (`tests/test_stable_noise.py`): default-path byte
  identity vs history; corpus-index invariance of `stable`;
  determinism + identity sensitivity per slot; 10-draw pairwise
  distinctness; separator anti-aliasing (episode/frame slot swap,
  repo_id digit bleed); N(0,1) distribution check; loud failure on an
  unknown key. Full suite + `check.py` green.

## The pre-registered instrument break (executes at the anchor boundary)

**When:** after the box-batch 40k panel reads land and are posted
(they run under `index` — mid-experiment keying flips are exactly what
this amendment exists to prevent). First eval after that boundary.

**What:** re-bank the flow anchor under `stable` keying — one panel
eval of flow-80k @ heun-30, N=1, panel `curated_v0_k4l2`
(~1.7 h GPU). From then on `stable` is the quoted keying for all new
flow numbers; `index` numbers stay valid as-labeled at frozen corpus.

**Predictions / decision rules:**

1. **Controls (hard):** state-copy and any AR summaries must be
   **bitwise identical** across keyings — neither takes noise. Any
   delta ⇒ the change leaked outside the noise path ⇒ do not adopt,
   investigate.
2. **Flow shift band (primary):** re-keyed chunk_mae is a fresh draw
   of the same noise distribution. Naive band: per-frame across-draw
   std ~5.9° pooled over 17,204 core frames ⇒ σ ≈ 0.045. The draws
   chain landing tonight gives the *empirical* per-draw pooled spread
   from its per-draw dumps; the band is
   `6.6232 ± 3·max(0.045, empirical σ_draw)`. Inside ⇒ adopt and
   re-bank. Outside ⇒ something other than the draw moved — hold the
   flip, diagnose.
3. **Secondary observables** (quoted, not gated): first_mae and
   state-copy-relative margin under the new keying.

**Why not flip now:** zero live comparisons should change keying
mid-flight, and re-banking costs a GPU eval that would contend with
the draws chain. The flag landing today makes the flip a
one-token change + one eval at a boundary we already have to visit.
