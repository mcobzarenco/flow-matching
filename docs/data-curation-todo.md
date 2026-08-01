# Data curation TODO (v1v2v3 → community_curated_v1)

Working list for the corpus filter/merge + judge-driven augmentation
(plan agreed 2026-07-31; context: `episode-judging.md`,
`fast-tokenizer-v1-review.md`, ledger comparability rules in
`architecture.md` §7).

- [x] **1. Provision gcloud box** — `curation-1` (us-east1-b):
      n2-standard-32 on-demand, 8×375 GB local-SSD RAID0 = 2.9 TB
      `/data` (1.6/2.9 GB/s measured), 100 GB pd-balanced `/durable`;
      repos + hf auth + anthropic key installed.
- [x] **2. Download data** — in-scope datasets (dims 6/6 ∧ fps=30) +
      rig: 991 + 2 datasets, ~680 GB in `/data/source/{v1,v2,v3,rig}`
      (sequential 8-worker hf download after parallel bursts tripped the
      CDN limiter). q01/q99 verification folded into step 3's tool.
- [ ] *(ON HOLD)* **2b. Rig dataset consolidation** — merge
      `so101_pick_place_{clean,v2}` into one dataset and rename camera
      `front` → `top` (verified 2026-08-01: it is a fixed overhead view;
      the name misleads humans and judges alike). Re-backfill stats +
      exact quantiles after the merge. NOTE: new frame indexing ⇒ rig
      eval numbers start a fresh ledger line; old rig comparisons don't
      carry over. Checkpoint stats tables key on repo_id — rollout
      `--stats-repo-id` must use the new id going forward.
- [ ] **3. Mechanical filter + merge into `curated_v0`** (before any
      judge spend) — `ldtools.filter_collections`: applies the episode
      filters below and writes survivors into ONE combined collection
      (`/data/curated_v0/<user>/<dataset>`): untouched datasets
      hardlinked, datasets with drops rebuilt — episodes renumbered,
      videos REMUXED not re-encoded (keyframe-aligned starts, measured
      130/130; encode-needing files quarantined loudly), stats
      re-aggregated + exact action/state quantiles recomputed. Judges
      then run on `curated_v0`; judge-gated filtering produces the final
      collection later (step 6):
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
  - [ ] materialize judge subgoals into lerobot-native form during the
        merge (sidecar stays the provenance store): SARM-style episode
        columns (`{sparse}_subtask_names/_start_frames/_end_frames` in
        episodes parquet — what the online visualizer renders) and/or
        `language_persistent` rows with `style="subtask"` (per-frame
        column; `active_at(t)` resolver = our piecewise-constant lookup;
        `lerobot.datasets.language` + `annotations/steerable_pipeline`
        writer).
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
