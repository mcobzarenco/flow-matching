# fast_tokenizer_v1 review — degenerate BPE vocabulary (findings, 2026-07-30)

Audience: whoever is building/running the AR FAST decoder arms.
Artifact under review: `mcobzarenco/bijou-checkpoints/fast_tokenizer_v1`
(fit 2026-07 on 1,040 datasets / 4,919,596 chunks, stride 5, scale 10,
vocab 1024, chunk 50×6; `fit_report.json` rides in the artifact). Every
number below is measured — the probe snippets are in the appendix.

**TL;DR: v1 is normalized correctly, but its BPE is degenerate — 1019 of
1024 vocab slots are base alphabet and only 5 merges were learned (all of
them zero-runs). Token sequences are ~2× longer than they need to be at
identical reconstruction fidelity. Recommend fitting a `fast_tokenizer_v2`
with percentile-bounded alphabet before AR arms go far; v1 results remain
internally valid but are a weak representation-shaping target, and
token-level metrics will not be comparable across tokenizer versions.**

## Finding 1 — normalization is correct (suspicion cleared)

The corpus fit normalizes every chunk with its dataset's own exact
q01/q99 from `meta/stats.json`, mapped to [−1, 1]
(`bijou/fast/cli.py::dataset_chunks` → `QuantileEntry.normalize`; the
exact quantiles are the reason `ldtools.backfill_quantile_stats` exists).
Evidence from the artifact itself:

- `recon_mae_raw` is reported in raw degrees (corpus-weighted mean
  0.463°) — only meaningful if the normalize/denormalize round trip is
  applied and correct.
- Quantized coefficients probed on 6 sample datasets (359k coefficients):
  **89.0% exact zeros** — the sparsity signature of quantile-normalized,
  DCT-transformed data. Raw joint angles would produce huge dense
  coefficients and would have tripped the explicit alphabet-vs-vocab
  guard in `FastTokenizer.fit_quantized`.

So no, it was not fit on raw angles, and "normalize with p1/p99 before
fitting" is already the design.

## Finding 2 — the alphabet ate the vocabulary

From the fitted `bpe.json` + `fast_config.json`:

| quantity | value |
|---|---|
| BPE vocab | 1024 |
| base alphabet | **1019** (`min_coefficient` −452 … +566) |
| learned merges | **5** — runs of the zero symbol of length 2/4/8/16/32 |

Mechanism: `NORMALIZED_CLIP = 8.0` admits normalized values to ±8, so DCT
coefficients can reach 8·√50·10 ≈ ±566 at scale 10 (the fitted alphabet
max is exactly 566 — the bound is attained). `fit_quantized` derives the
alphabet from the corpus **min/max**, so a handful of worst-case outlier
coefficients across 4.9M chunks dictated a 1019-symbol alphabet, leaving
5 of 1024 slots for merges. The tail symbols are near-dead weight:
probing 359k coefficients, 0.008% exceed |71| and **none** exceed |150|.

The clip itself exists for a good reason — without the constant-dim
guard + clip, a parked joint's jitter amplifies into a measured
7.3e9-symbol alphabet (recorded in `tokenizer.py`). But 8.0 was
calibrated as an input-sanity bound, not an alphabet budget, and the
min/max alphabet derivation transmits worst-case outliers straight into
the vocab.

Net effect on v1 (from `fit_report.json`, chunk-weighted over 1,040
datasets): **53.3 tokens/chunk** mean (dataset p10/p50/p90 =
41.1/52.4/71.7), compression 5.6× vs the paper's ~13× regime. Every
nonzero coefficient costs one full token; zeros get crude run-length
encoding via the 5 merges; nothing else was learned.

## Paired experiment — same data, same eval, alphabet fixed

Refit on local sample chunks (1,731 chunks from 3 six-dof datasets —
small sample, but the comparison is paired: identical chunks, identical
eval, reconstruction error measured against the *unclipped* sample so any
clipping cost is included):

| tokenizer | alphabet | merges | tokens/chunk (p90) | recon MAE / p99 |
|---|---|---|---|---|
| v1 (clip 8) | 1019 | 5 | 67.9 (96) | 0.493° / 2.40° |
| refit, clip 2.0 | 149 | **875** | **28.3** (41) | 0.493° / 2.40° |
| refit, clip 1.5 | 149 | 875 | 28.3 (40) | 0.493° / 2.40° |

**2.4× shorter sequences at reconstruction error identical to the third
decimal.** (This sample is harder than the corpus average — v1 scores
67.9 here vs 53.3 corpus-weighted — so read the ratio, not the absolute;
corpus-level v2 numbers are expectation until the refit runs.)

## Why the AR arms should care

1. **Sequence length / decode latency**: ~2× more AR steps per chunk than
   necessary, and `--ar-max-tokens` sized for v1 lengths (corpus p90 ~72)
   would halve under v2 (probe p90 ~41).
2. **Supervision quality**: with 5 merges, next-token prediction is
   almost entirely single-quantized-coefficient prediction. 875 merges
   means tokens that encode multi-coefficient temporal structure — a
   richer target for the stated purpose of the FAST arm
   (representation shaping, architecture.md §8.3).
3. **Softmax mass**: ~85% of the 1024-way output distribution covers
   base-alphabet tail symbols that occur at ≤0.008% frequency.
4. **Comparability**: token semantics change with any refit — loss,
   perplexity, tokens/chunk and accuracy are **not comparable across
   tokenizer versions**. Arms must record which tokenizer they trained
   with (checkpoints already do) and never mix.

## Recommendation

Fit **`fast_tokenizer_v2`** with the alphabet decoupled from worst-case
outliers. Preferred: derive alphabet bounds from coefficient
*percentiles* at fit time (e.g. cover 99.995% of quantized coefficients;
the encoder already clips out-of-alphabet coefficients and counts them
loudly), keeping `NORMALIZED_CLIP` as a generous input-sanity bound.
Simpler alternative with the same measured effect: lower
`NORMALIZED_CLIP` to ~2. Either lands the alphabet around ~150 symbols
and frees ~870 slots for merges.

Cost: the v1 corpus fit took 32 min CPU-only on the 2×H100 box
(`fit_seconds` 1928); a refit does not disturb training. Immutability
policy already anticipates this — new artifact name, v1 stays published.

Expected v2 (extrapolation from the paired probe, to be re-measured on
the corpus and recorded in its own `fit_report.json`): ~25–30
tokens/chunk mean, recon unchanged, ~870+ merges.

Unrelated loose end noticed in the report: `willnorris/bbox-2` has the
worst per-dataset reconstruction (5.28° MAE vs 0.463° corpus mean) —
worth a one-off look at its quantiles/units, independent of the vocab
issue.

## What does NOT need to change

- Per-dataset q01/q99 quantile normalization (correct, and load-bearing).
- `scale = 10.0` (89% zero coefficients = healthy sparsity; paper calls
  the setting insensitive).
- `vocab_size = 1024` (paper default; the budget just needs to reach the
  merges).
- DCT/BPE machinery, decode error handling, artifact schema.

## Appendix — how each number was measured

All probes run 2026-07-30 against the downloaded artifact
(`outputs/checkpoints/fast_tokenizer_v1`) in the flow-matching venv.

- **Merges/alphabet**: `FastTokenizer.load(...)`; `bpe.get_vocab()`
  filtered to multi-char tokens; symbol composition via
  `ord(ch) + min_coefficient`.
- **Clip→alphabet bound**: 8·√50·10 = 565.7 vs fitted max
  `alphabet_size + min_coefficient − 1` = 566.
- **Coefficient tail / zero fraction**: 359k coefficients from 6 sample
  datasets (`~/w/datasets/community_samples`), normalized with each
  dataset's `stats.json` quantiles, `FastTokenizer.quantize_chunks`,
  thresholds |71|/|150|/|300|/|566|.
- **Corpus tokens/chunk**: `fit_report.json` `per_dataset`,
  chunk-weighted.
- **Paired refit**: 6-dof sample datasets, 50-frame windows stride 5,
  `FastTokenizer.fit(np.clip(normalized, ±clip), scale=10, vocab=1024)`;
  eval on ~150 chunks/dataset, error de-normalized to degrees via
  (q99−q01)/2, measured against unclipped chunks.
