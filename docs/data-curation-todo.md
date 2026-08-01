# Data curation TODO (v1v2v3 → community_curated_v1)

Working list for the corpus filter/merge + judge-driven augmentation
(plan agreed 2026-07-31; context: `episode-judging.md`,
`fast-tokenizer-v1-review.md`, ledger comparability rules in
`architecture.md` §7).

- [ ] **1. Provision gcloud box** — CPU-only ~64–88 vCPU, 3 TB SSD,
      on-demand (not spot); `init-vm-cpu.sh`, `hf auth`,
      `~/.anthropic.env`.
- [ ] **2. Download data** — v1/v2/v3 collections + rig datasets; verify
      exact q01/q99 backfill in every `stats.json`.
- [ ] **3. Calibration pilot** — judge sweep 2 eps/dataset on the
      fps30∩6/6 scope (~2.0k episodes, opus + haiku passes); hand-label
      ~75 stratified episodes; build the aggregation/calibration tool
      (majority-vote camera maps, verdict distributions, judge-vs-human
      + cross-model agreement); pick thresholds + cascade.
- [ ] **4. Full judge sweep** — all eligible in-scope episodes (53.2k)
      with the chosen cascade; sidecars merged per dataset.
- [ ] **5. Build `merge_collections` (ldtools)** — single filtered
      collection with per-episode drop manifest; renumber episodes,
      remap `meta/judgments.json`, recompute stats + exact quantiles,
      validate by reload. Filters:
  - [ ] mechanical: dims 6/6, fps=30, length < 50 frames, decode
        failures, stats/metadata guards
  - [ ] trajectory (parquet-only): NaN/inf actions, zero-travel
        episodes, idle fraction ≥ ~80%, length outliers (< 2 s,
        > p99.5)
  - [ ] judge-gated (thresholds from step 3): episode discards,
        dataset-level kill (majority discarded), task_completion=no +
        score ≤ 3, dead camera streams
  - [ ] relabel placeholder/mismatched instructions from
        `suggested_instructions` (bake best into tasks; full list stays
        in sidecars)
- [ ] **6. Ship `mcobzarenco/community_curated_v1`** to the hub with the
      manifest-generated card.
- [ ] **7. Re-baseline the new frame set** — state-copy + AR-unftext-50k
      scored on curated holdout (new indexing = new ledger section;
      nothing comparable across frame sets).
- [ ] **8. A/B** — live-trunk AR recipe, curated vs current fps-30 set,
      matched steps; judged on comm holdout + rig-holdout first_mae.
- [ ] **9. Augmentation arms (bijou-side)** — instruction sampling from
      sidecars; camera-kind prefix annotations with dropout-to-unknown.
- [ ] **10. fast_tokenizer_v2** — percentile-bounded alphabet, fit on
      the curated corpus; coordinate timing with the AR experiments
      (token metrics never cross tokenizer versions).
- [ ] *(parallel, cheap)* idle-trim probe from parquet — quantify
      leading/trailing idle before deciding whether the merge trims.
