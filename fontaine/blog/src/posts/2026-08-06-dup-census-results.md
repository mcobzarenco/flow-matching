# Duplicate-content census: 12.2% of the panel's core frames have byte-exact twins in train (#18.7)

*2026-08-06, work session ~01:4x–02:1xZ. Instruments:
`fontaine/scripts/dup_content_census.py`,
`fontaine/scripts/dup_census_anchor_impact.py`. Reports:
`~/dup_census_report.json`, `~/dup_census_anchor_impact.json`.*

## Why now

The bijou deep-dive flagged it (finding 7): `data.py` dedups exact repo
ids only, while the holdout split is a pure function of
`(repo_id, num_episodes, fraction, split_seed)`. A community fork —
same recordings under a different repo id — gets an **uncorrelated
holdout draw**, so an episode held out in repo X can sit in repo Y's
train side. Every fine holdout delta we are about to read (the
box-batch results land ~03–04Z with a 0.15 band; the E4B adopt band is
`max(3σ_seed, 0.15)`) silently assumed this channel was empty. The
census ran before those reads, as the deep-dive prescribed ("before
trusting fine holdout deltas").

## Reads (declared in the instrument header before any fingerprint was computed)

R1 cross-repo exact-duplicate clusters; R2 (primary) holdout→train
leakage under the panel convention (fps 30, cameras {1,2}, holdout 0.1,
split seed 0); R3 panel-row impact via the k4l2 plan; R4 intra-repo
duplicates; R5 (companion, declared before computing) clean-vs-leaked
anchor split on the two banked panels.

## Results

**R1 — the corpus is heavily forked.** Of 52,507 episodes across 981
repos, **6,935 episodes (2.67M frames) sit in 3,348 cross-repo
byte-exact clusters** — episode-length action *and* state streams
identical to the byte. The quantized tier (1e-3 rounding) adds
nothing: these are pure re-uploads, not re-encodes. Dominant structure:
same-user variants (`samanthalhy/so100_herding_2↔3` 365 shared
clusters, the `shylee/pengrip{C,E,F}` family, `dopaul` chess merges
173+144) and cross-user forks (`Chojins↔bensprenger`
chess_game_001_blue_stereo 306, `Dangvi↔s20000s/soarm100_data` 125,
`lirislab↔roboticshack` guess_who 96+96).

**R2 — the split is breached.** **524 holdout episodes** (of the
panel's ~4,300, across **79 repos**) have a byte-exact duplicate in the
train side of some other selected repo. Zero of them are intra-repo
only — this is entirely the cross-repo fork channel the repo-id dedup
cannot see.

**R3 — panel impact: 2,096 / 17,204 core rows (12.2%) and
1,048 / 8,596 labeled rows score on leaked episodes.**

**R4 —** intra-repo duplicate clusters exist (double-weighting) but
none cross the split on their own.

**R5 — the leak is worth real MAE.** Clean-vs-leaked split of the
banked panels (pooling imported from `box_batch_results.py`; the full
partition reproduces both anchors exactly):

| model | full panel (anchor) | clean core (15,108 fr) | leaked core (2,096 fr) | leaked−clean frame-MAE CI95 |
|---|---|---|---|---|
| AR-100k | 5.8026 / 2.1431 | **5.9761 / 2.1695** | 4.5359 / 1.9532 | [−1.62, −1.20] |
| flow-80k | 6.6232 / 1.9331 | **6.8137 / 1.9714** | 5.2331 / 1.6571 | [−1.76, −1.33] |

Leaked frames score ~1.3–1.6 points better than clean on both models —
far outside frame-sampling noise. **The published anchors are ~0.17–0.19
optimistic in level**; clean-panel anchors are AR-100k
**5.9761 / 2.1695**, flow-80k **6.8137 / 1.9714**.

Honest confound: the fork clusters are concentrated in specific content
(chess games, herding, pengrip), so part of the clean-vs-leaked gap can
be content difficulty rather than memorization. The census certifies
the leak exists and bounds its panel share; it does not causally
attribute the full −1.4 to memorization (that would need a
counterfactual run trained without the twins — not worth a GPU slot
under run-only-what-changes-the-next-decision).

## What this does and does not invalidate

- **Paired within-corpus deltas stand.** Every training run in the
  program (box arms, E4B, mainline) shares the same train corpus, so
  the same 12.2% of panel frames is equally "leaked-to" for every
  model. The box-batch primary read (B−A-s0 paired per-frame), the
  replicate σ_seed, and the draws-chain relative gains are unaffected
  in their comparisons.
- **Absolute generalization claims carry a caveat.** The panel measures
  ~12% memorization-eligible frames; the comm-holdout→rig bridge and
  any "generalizes to X°" statement should quote the clean-core
  column.
- **Anchor convention going forward:** the exclusion list is exact and
  frozen (`dup_census_report.json` → 524 episodes). Re-defining the
  panel (excluding leaked episodes) is a panel change and needs its own
  amendment + re-bank of every anchor — proposed as a queue item, owner
  steer welcome. Until then, results posts report full-panel (anchor
  convention) with the clean-core column alongside.

## Validation

- `--oracle` synthetic suite: planted cross-repo dup across the split
  (leaks, incl. an f64 round-trip donor), train→train dup (no leak),
  intra-repo split-crossing twin (leaks, tagged), single-episode donor
  repo (leaks), 1e-2 noise copy (invisible in every tier),
  quantum-grid re-encode (quantized tier only), constant-action/
  different-state pair (action-only tier only) — all pass.
- Split mirror **proven on real data**: the plan's episode set equals
  the re-derived `holdout_episodes()` output on **all 878 plan repos**
  (and selection count 878 matches the pipeline's own).
- Join content-checked against raw parquet (npz `truth[i,j]` ==
  `action[frame+j]` on sampled rows, both models); core-flag pattern
  asserts per repo; hash-collision guard re-loaded 20 flagged pairs
  with `np.array_equal` — all equal.
- Anchors 5.8026/2.1431 and 6.6232/1.9331 reproduce exactly through
  the partition; zero structural warnings corpus-wide (info.json
  episode counts all match parquet; all episode ids contiguous).
- `check.py` green (212).
