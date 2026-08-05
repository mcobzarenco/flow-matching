# Amendment: sealed plan v2 — the census-flagged repos leave the measure

*2026-08-05 ~17:5xZ. First use of the seal-amendment policy the owner
set today ("we should have measurable outcomes, but if the measure is
incorrect, we should be able to update it", 17:08Z; "can we just
remove those episodes from the sealed plan", 17:20Z). Amendments are
posted and versioned; what stays forbidden is silently editing the
measure after seeing results.*

## What changed

`plans/holdout_curated_v0_k4l2_sealed.json` (v1) pinned episodes from
three repos the [wrap census](2026-08-05-wrap-census.md) showed to be
corrupted — measuring a model against known-broken ground truth is a
measure error, not a model property:

| repo | failure mode | core frames | labeled |
|------|--------------|-------------|---------|
| kevin510/lerobot-cat-toy-placement | systemic ±180° wraps (action+state) | 16 | 8 |
| kevin510/so-100-draw-smiley | systemic ±180° wraps (action+state) | 20 | 10 |
| willnorris/bbox-2 | state-stream glitch, all 6 dims | 16 | 8 |

**v2** (`plans/holdout_curated_v0_k4l2_sealed_v2.json`) = v1 minus
those triplets: core 17,204 → 17,152 (−52 frames, 13 distinct
episodes), labeled 8,596 → 8,570 (−26). Everything else — plan seed,
split, fps/camera filters, per-episode draw counts — is untouched.
**v1 is deprecated** for all future scoring.

## Anchors

- The in-flight v1 baseline score (running as this posts, ETA
  ~18:20Z) still banks — as the *pre-removal instrument record*, and
  because a v2 anchor derives from it exactly: the pooled panel
  chunk_mae is a plain frame mean, so removing repos re-pools from
  the report's per-dataset means and counts with no re-eval.
  (Correction to my 17:4xZ Discord note: the sealed run carries no
  `--dump-predictions`; the recompute uses the per-dataset
  decomposition in the JSON, which is exact for frame-mean metrics —
  not per-frame dumps.)
- A direct v2 eval run will verify the arithmetic at the next free
  GPU boundary (it shares ~99.7% of frames with v1; expected shift is
  small — the census measured the 3 repos' pinned frames as
  high-error, so v2 anchors should come in slightly *below* v1).
- Upstream: the dataset README now documents the three repos +
  the pre-removal revision hash
  (`250f6ed2c45c115b0a9570f43f8b736b8a1ad3f1`,
  [commit a9f652f](https://huggingface.co/datasets/mcobzarenco/community_curated_v0/commit/a9f652f4df65eb096ba9f96d325e27a3a089cb06)),
  pushed on the owner's explicit "you push" (17:20Z).

## Why this matters beyond hygiene

The owner also stated the project's north star today: **a VLA for
their rig**. The community panel is a proxy instrument — it should be
as clean as we can make it, but polishing proxy numbers is not the
goal. This reweights the backlog toward rig-transfer questions (rig
fine-tune lineage, sign/calibration robustness, deployment-class
decode latency) over community-panel micro-optimization; ideas.md
ordering will reflect it.
