# Amendment proposal: panel v2 — leaked and corrupt episodes leave the measure (owner steer wanted)

*2026-08-06, work session ~02:1x–02:4xZ. Instrument:
`fontaine/scripts/panel_v2.py`. Artifacts:
`plans/holdout_curated_v0_k4l2_panel_v2.json` (frozen, self-contained),
`~/panel_v2_anchors.json`. Follows the seal-amendment policy (owner
2026-08-05 17:08Z: "if the measure is incorrect, we should be able to
update it") — posted and versioned, nothing silently edited. **This is
a proposal: no instrument switches to v2 until the owner steers.***

## Why

Two independent censuses have now shown the panel scores frames it
should not:

1. **[Dup-content census](2026-08-06-dup-census-results.md) (#18.7):**
   524 of the panel's holdout episodes have byte-exact action+state
   twins in the train side of some other repo (the cross-repo fork
   channel repo-id dedup can't see). **2,096 / 17,204 core rows
   (12.2%) score on train-leaked content**, and those rows measure
   ~1.3–1.6 better than clean rows on both banked models.
2. **[Wrap census](2026-08-05-wrap-census.md) (#14):** kevin510's two
   repos (systemic ±180° wraps) and willnorris/bbox-2 (state-stream
   glitch) are corrupted ground truth. The *sealed* plan dropped them
   ([sealed v2](2026-08-05-sealed-plan-v2.md)) — but the panel plan
   still scores them (52 core + 26 labeled rows, averaging ~31°
   chunk MAE — wrap-scale garbage, not model error).

A holdout row with a byte-exact train twin measures memorization, and
a corrupted-truth row measures the corruption. Both are measure
errors.

## The definition (frozen)

**Panel v2 = panel v1 minus (a) every row on a census-leaked
(repo, episode), (b) every row of the 3 wrap-census corrupt repos.**

- Strict row-subset in original order — no re-draw, no new episodes.
  Plan seed, split, fps/camera filters, frames-per-episode all
  untouched. Consequence: **every banked per-frame npz re-pools to v2
  exactly, with zero re-evals** — adoption is CPU-only.
- Core 17,204 → **15,056** (−2,096 leaked, −52 corrupt, 0 overlap);
  labeled 8,596 → **7,522** (−1,048, −26).
- The exclusion lists are embedded in the plan file itself (524
  `repo::episode` keys + 3 repos + provenance) — the artifact is
  self-contained and frozen; the plan is schema-identical to v1 and
  drops into `bijou.eval --sample-plan`.

## v2 anchors (derived from the banked npzs, oracle-gated)

| column (chunk_mae / first_mae) | v1 (anchor) | census clean-core | **v2** |
|---|---|---|---|
| AR-100k | 5.8026 / 2.1431 | 5.9761 / 2.1695 | **5.8894 / 2.1396** |
| flow-80k Heun-30 | 6.6232 / 1.9331 | 6.8137 / 1.9714 | **6.7151 / 1.9453** |
| state-copy | 11.7847 / 2.6202 | — | **11.7639 / 2.5851** |
| state-copy-norm | 11.7357 / 2.4426 | — | **11.7451 / 2.4350** |

The two exclusions partially offset in level: removing leaked rows
pushes the level up (~+0.17–0.19, the memorization discount), removing
the corrupt repos' wrap-scale rows pulls it back (~−0.09). Honest
caveat carried over from the census: part of the leaked-vs-clean gap
is content difficulty (forks concentrate in chess/herding/pengrip), so
v2's level is "panel minus measure errors," not a pure de-memorized
number. It is still the right instrument: scoring holdout rows with
byte-exact train twins is wrong regardless of why they score better.

## Transition rules (proposed)

1. **In-flight pre-registered reads finish on v1 as registered** — the
   box-batch results (~04Z), the draws chain + fairness probe, E4B's
   gates, and the SnapFlow distill primary all quoted v1 anchors in
   their pre-regs; swapping the measure mid-flight is exactly what the
   amendment policy forbids. Their results posts quote the v2 column
   alongside (CPU re-pool of the same npzs).
2. **On owner approval, v2 becomes the anchor convention for every NEW
   pre-reg**, and the anchor table above is the new bank.
3. **Bundle the anchor-moving backlog at one boundary.** Two other
   approved-or-pending changes each force a re-bank:
   - the `--noise-key stable` flip (#18.2, implemented + pre-registered,
     due "at the first anchor boundary after the box reads") — needs
     one GPU flow-80k panel re-eval (draw noise changes predictions);
   - the shortest-arc metric proposal (#14, still awaiting owner
     sign-off — CPU re-score, panel effect −0.0528 on v1 AR).
   Doing the v2 switch, the noise flip, and (if approved) shortest-arc
   at the same boundary re-banks the flow anchor once instead of three
   times. The natural boundary: right after the box-batch results post,
   when the flow-80k re-eval can take a quiet GPU slot.

## Owner decision points

1. Adopt v2 as the primary panel convention? (Recommended: yes.)
2. Bundle the noise-key flip at the same boundary? (Recommended: yes —
   it's already pre-registered to fire at the next boundary.)
3. Shortest-arc scoring in the same re-bank, or keep deferring? (No
   recommendation — it changes metric semantics, owner's call from
   the wrap-census post stands.)

## Validation

All hard asserts in `panel_v2.py`, run before any v2 number printed:
leaked-row exclusion counts equal the census's published 2,096/1,048;
corrupt-repo rows equal sealed-v2's 52/26; v1 pooling reproduces both
banked anchors (<5e-4); leaked-only exclusion reproduces the census's
clean-core 5.9761/2.1695 + 6.8137/1.9714 (≤1e-4); state-copy pools
identically from both npzs (cross-npz join consistency, <5e-4);
materialized v2 re-filters to itself (idempotence) and is a strict
ordered subset of v1; synthetic materialization oracle (known
exclusions incl. the overlap case) passes. The row join is the same
code path the census content-verified against raw parquet.
`check.py` green.
