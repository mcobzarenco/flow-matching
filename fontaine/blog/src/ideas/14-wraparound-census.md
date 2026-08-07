# 14. ±180° wraparound census & shortest-arc error — `confirmed`/banked (measured 2026-08-05)

- **Hypothesis:** truth chunks wrapping the ±180° boundary inject
  ~360°-scale discontinuities into BOTH raw-degree training targets
  and MAE; a wrap census may explain a measurable slice of panel MAE.
- **Measured** ([write-up](../posts/2026-08-05-wrap-census.md),
  instrument `probes/probe_wrap_census.py`, anchors in-probe): panel —
  16/17,204 wrap frames (0.093%, under the 0.1% gate) carrying 0.0720
  of the 5.8026 pooled chunk_mae (1.24%; shortest-arc re-score
  5.7498). Corpus — 81/42,872 episodes (0.19%) across 23 repos;
  kevin510 systemically corrupted (40/40 eps), willnorris/bbox-2 a
  separate state-stream glitch. wrist_roll dominates (204 action
  jumps), matching the SO101 calibration story (lerobot#1255, PR#777,
  fixed in 0.6.0).
- **Consequences:** unwrap-at-load training arm killed (0.19% cannot
  move a 40k pair); shortest-arc metric proposal → owner sign-off
  (moves every anchor); kevin510 + willnorris/bbox-2 flagged for any
  future curated-v1 exclusion list.
