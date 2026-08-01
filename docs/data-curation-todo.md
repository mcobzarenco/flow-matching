# Data curation TODO (v1v2v3 → community_curated_v1)

Working list for the corpus filter/merge + judge-driven augmentation
(plan agreed 2026-07-31; context: `episode-judging.md`,
`fast-tokenizer-v1-review.md`, ledger comparability rules in
`architecture.md` §7).

- [ ] **1. Provision gcloud box** — CPU-only ~64–88 vCPU, 3 TB SSD,
      on-demand (not spot); `init-vm-cpu.sh`, `hf auth`,
      `~/.anthropic.env`.
- [ ] **2. Download data** — only the in-scope datasets (dims 6/6 ∧
      fps=30, known from metadata: 991 of 1,242) + rig datasets; verify
      exact q01/q99 backfill in every `stats.json`.
- [ ] **3. Mechanical filter pass** (before any judge spend) — small
      tool over meta + data parquet producing per-dataset episode
      exclusion lists consumed by both the judge sweep and the merge:
  - [ ] scope: dims 6/6, fps=30 (dataset-level)
  - [ ] length < 50 frames; length outliers (< 2 s, > p99.5)
  - [ ] NaN/inf actions, zero-travel episodes, idle fraction ≥ ~80%
  - [ ] **dataset-level survivor floor**: drop datasets with < k
        episodes remaining after the above (k TBD at calibration, ~5–10
        — tiny datasets carry per-dataset stats/holdout overhead for
        negligible frames)
  - [ ] quantify leading/trailing idle while here (trim decision for a
        later filter version, not this one)
- [ ] **4. Calibration pilot** — judge sweep 2 eps/dataset on the
      mechanical survivors (opus + haiku passes); hand-label ~75
      stratified episodes; build the aggregation/calibration tool
      (majority-vote camera maps, verdict distributions, judge-vs-human
      + cross-model agreement); pick thresholds + cascade.
- [ ] **5. Full judge sweep** — mechanical survivors only (≤ 53.2k
      episodes) with the chosen cascade; sidecars merged per dataset;
      decode failures recorded (free integrity census).
- [ ] **6. Build `merge_collections` (ldtools)** — single filtered
      collection from exclusion lists + judge verdicts, with
      per-episode drop manifest; renumber episodes, remap
      `meta/judgments.json`, recompute stats + exact quantiles,
      validate by reload. Judge-gated filters (thresholds from step 4):
  - [ ] episode discards; dataset-level kill (majority discarded)
  - [ ] task_completion=no + score ≤ 3
  - [ ] dead camera streams (majority-vote unknown + dark)
  - [ ] relabel placeholder/mismatched instructions from
        `suggested_instructions` (bake best into tasks; full list stays
        in sidecars)
- [ ] **7. Ship `mcobzarenco/community_curated_v1`** to the hub with the
      manifest-generated card.
- [ ] **8. Re-baseline the new frame set** — state-copy + AR-unftext-50k
      scored on curated holdout (new indexing = new ledger section;
      nothing comparable across frame sets).
- [ ] **9. A/B** — live-trunk AR recipe, curated vs current fps-30 set,
      matched steps; judged on comm holdout + rig-holdout first_mae.
- [ ] **10. Augmentation arms (bijou-side)** — instruction sampling from
      sidecars; camera-kind prefix annotations with dropout-to-unknown;
      **subgoal conditioning**: judges emit per-episode subgoal segments
      (every frame inherits its segment's label — piecewise-constant
      interpolation between annotated boundaries), train-time arm
      conditions the prefix on the current subgoal (prototyped on the
      rig 2026-08-01; segments ride in `meta/judgments.json`).
- [ ] **11. fast_tokenizer_v2** — percentile-bounded alphabet, fit on
      the curated corpus; coordinate timing with the AR experiments
      (token metrics never cross tokenizer versions).
