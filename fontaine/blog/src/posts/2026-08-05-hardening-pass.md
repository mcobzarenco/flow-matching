# Ideas #18.1: the cheap instrument-hardening pass — landed

*2026-08-05 ~20:55Z. CPU-only work item executed while both GPU chains
ran (box 4×H100 paired batch, local noise-draw chain). Source: the
[bijou deep-dive](2026-08-05-bijou-deep-dive.md) fix queue, item 1 —
five additive fixes, oracle-gated, no behavior change on any healthy
path.*

## What changed

1. **`--aux-prompt-hash` now pins measurement, not just training**
   (deep-dive finding 4). The pin reaches the in-run probe selection
   (`bijou.train`'s holdout probe `select_datasets` call) and offline
   eval (`bijou.eval` gains the flag). Before: a pinned run whose
   stamp mismatched trained a dataset as unjudged while its probe and
   eval rendered full tags for the same dataset — train and
   instrument silently disagreed on the prompt distribution. The
   train-split probe needed no change (it samples the already-pinned
   training selection).
2. **`resolve_plan` bounds-checks `frame_index`** (finding 6a). A
   planned frame past its episode's last row — the
   truncated/re-encoded-episode trap — now feeds the existing
   fail-loudly path (`episode has N rows`) instead of silently
   scoring the *next* episode's rows via offset arithmetic.
3. **`score_frame` refuses zero-valid frames** (finding 6c). The
   `max(divisor, 1)` guards turned an impossible-today zero-valid
   frame into a perfect 0.0 chunk_mae; combined with 6a that could
   *lower* MAE. Now an assert at the source.
4. **The report JSON records full scoring semantics** (finding 5):
   `exclude`, `aux_prompt_hash`, `sample_steps`, `sample_method`,
   `sample_draws`, `generate`, `condition_override`, `batch_size`,
   `world_size`. An eval is now reproducible — and a Q3
   counterfactual (`condition_override`) *identifiable* — from its
   artifact alone. Keys are additive; existing consumers
   (`sealed_v2_anchor.py`, `flow_vs_ar_paired.py`) read by name and
   are unaffected.
5. **npz dumps gain dataset-local identity**: `episode_index` and
   `frame_index` columns next to the existing concat `index` (which
   is valid only under one exact selection — the flow-eval-noise
   finding made that sharp). Rows stay addressable across corpus
   recompositions; threaded through the multi-GPU shard merge with
   the same index-sorted alignment as every other dump column.

## Gates

- **Oracle (scoring path, bit-exact):** recomputed the banked
  AR-100k panel report
  (`eval__bijou_arb_rcond_100k_ddp4__step_100000__panel_k4l2`) from
  its npz through the *edited* `score_frame`/`summarize`: 4 policies
  × {chunk_mae, chunk_mse, first_mae} = 12 cells, **all deltas
  0.00e+00**, including the 5.8026 pooled anchor.
- **Tests:** 3 new (out-of-range plan frame fails loudly; zero-valid
  frame refused; identity columns survive shard-merge permutation),
  full suite 168 passed; `check.py` green (ruff, pyright, format).
- Resolution of in-range frames is unchanged by construction (same
  arithmetic, plus a check) — the existing
  `test_resolve_maps_to_concat_indices_and_splits_core` pins it.

## Comparability notes

- No current number moves: every fix is an assert, a recorded field,
  or a new column. The next eval's report JSON simply carries more
  provenance.
- `--aux-prompt-hash` on eval changes judged/unjudged rendering *only
  when a pin is passed and mismatches* — no run to date used a pin.
  Future pinned runs must pass the same pin to probe and eval; the
  report field makes compliance auditable.

## Still open in ideas #18 (unchanged)

Flow-noise stable-triple reseed (#18.2 — versioned amendment at an
anchor boundary), Q3 tripwire noise fix (#18.3), resume hardening
(#18.4 — blocks idea #3), rig-rollout safety gate (#18.5 — blocks the
first physical run), parity extension (#18.6), duplicate-content
census (#18.7). Also noting here so it isn't lost: deep-dive finding
**6b** (leakage checker's same-repo-id branch trusts episode
numbering with no count/content check) was *not* part of this pass —
it now rides the #18 queue explicitly.
