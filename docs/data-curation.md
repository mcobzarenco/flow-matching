# Data curation — the community corpus and how it was built

What we did to the training data, why, and what deliberately remains
open. The shipped artifact is
**`mcobzarenco/community_curated_v0`** on the HF hub (2026-08-02):
**981 datasets · 52,507 episodes · 24.8M frames (~229 h)** of 6-DoF
fps-30 teleoperation in one collection (`<user>/<dataset>`), fully
judge-annotated, with a general-audience README and the raw curation
records archived in-repo (`provenance/curation_archive.tar.gz`).
Companion docs: `episode-judging.md` (how verdicts are produced),
`episode-annotations.md` (how training consumes them), results on the
corpus in `architecture.md` §7.

## Pipeline (all shipped)

1. **Scope.** From the three converted community collections
   (`community_dataset_v{1,2,3}_v3`, 1,242 datasets / 73.9k episodes /
   36.9M frames): keep datasets with 6-dim action *and* state at 30 fps
   → 991 datasets / ~53.2k episodes.
2. **Mechanical episode filters** (`ldtools.filter_collections`; per
   episode, any hit drops it): < 50 frames (one action chunk); < 2 s;
   > p99.5 of the scope (2,367 frames ≈ 79 s — marathon recordings are
   idle/stuck sessions); NaN/Inf actions; zero travel; ≥ 80% idle
   steps. Dataset floor: ≥ 5 surviving episodes. Survivors: **981
   datasets / 52,507 episodes**.
3. **Union build.** Untouched datasets hardlinked; datasets with drops
   rebuilt — episodes renumbered, videos **losslessly remuxed** (never
   re-encoded; keyframe-aligned starts verified), stats re-aggregated
   and **exact** action/state q01/q99 recomputed (lerobot's native
   episode-averaged quantiles regress extremes — measured; the exact
   ones feed FAST-style normalization).
4. **Judge calibration pilot** — 2 episodes/dataset judged
   independently by opus-4-8, haiku-4-5 and opus-5 (paired by
   construction). Decisive numbers: haiku over-discards 2.1× and is
   unusable for camera tags (49% agreement) → no cheap-model cascade;
   opus-5 matches 4-8 on per-frame labels (holding agreement 92.8%,
   progress r 0.89) and instruction quality, but is far more lenient on
   the categorical verdict (1.8% vs 10.3% discards) → **score
   thresholds are the stable filtering signal, the verdict field is
   not**. Opus 5 chosen (same price, newer judge).
5. **Full sweep** — opus-5 over all 52,507 episodes: batch API (flat
   50% off) in double-buffered waves; length-adaptive evidence (one
   sampled timestep per 1.5 s, clipped [5, 20], corpus mean 10.1);
   idempotency keyed `(episode, model, prompt_hash, num_timesteps,
   max_image_dim)` so prompt/model/evidence changes re-judge
   deliberately and nothing re-judges by accident. Outcome:
   **50,592/50,612 judged (99.96%)**; ~20 episodes permanently
   unjudgeable (video/parquet timestamp desync or corrupt packets —
   verified pre-existing in the source recordings, kept in-corpus,
   identifiable by their absent judge records). Total judge spend
   ≈ **$2.9k** measured across all passes.
6. **Materialization** — one pinned `(model, prompt_hash)` selection
   projected into LeRobot-native surfaces per dataset: NaN-masked
   `annotation.{progress,holding,visible_object,visible_gripper}`
   columns, `language_persistent` subtask rows (piecewise-constant
   subgoals, every frame), `language_events` rows on exact firing
   frames (a frame may carry several), `meta/camera_kinds.json`
   majority-vote camera tags, `meta/judge_annotations.json` provenance
   stamp. Sidecar `meta/judgments.json` remains the source of truth.
7. **Verification** — robot-data columns bitwise-identical to pristine
   sources through every rewrite; annotation columns byte-checked
   against sidecars corpus-wide; full decode census (~208k frame
   decodes): 1 hard failure beyond the known unjudgeable residue.

**Rig datasets** (`so101_pick_place_{clean,v2}`) were judged separately
at dense evidence — 48 timesteps/episode (~every 10–20 frames, the
100-image API ceiling) — giving ~5–10× the corpus's per-frame
supervision density for fine-tuning. Aux-loss code must weight by the
NaN mask, not assume uniform density across datasets. Their camera maps
record `front → top` (the "front" camera is a fixed overhead view).

## Decisions of record

- **Soft filtering shipped, destructive merge dropped.** Verdicts and
  scores ride with the data; filtering happens at train time
  (`episodes=[...]`). A judge-gated rebuild (drop + renumber) was
  planned and rejected: renumbering invalidates sidecars/columns and
  forces video re-uploads, for no gain over train-time predicates. A
  hard-filtered `community_curated_v1` remains possible as a separate
  artifact if ever needed.
- **Broken episodes stay in the corpus.** The ~20 undecodable episodes
  are data-quality facts, not upload mistakes; they carry no judge
  records and all-NaN columns. Consumers are expected to be
  substitution-tolerant (catch-and-resample in the dataloader) rather
  than rely on exclusion lists, which go stale — the decode itself is
  the only honest oracle.
- **Rig consolidation dropped** (was: merge clean+v2, rename camera
  `front → top`). The camera-name problem is solved by annotation
  (`camera_kinds.json`) without touching data; a repo merge would have
  reset eval ledger comparability and churned checkpoint stats keys for
  cosmetic benefit.
- **No cheap-judge cascade** (calibration numbers above).
- **Human labeling optional.** The 75-episode stratified worksheet was
  prepared (in the provenance archive) but never labeled; thresholds so
  far rest on cross-model consistency and rig ground truth
  (gripper-aperture channel: holding ≈ 75–85% with an
  open-hover→true bias). Label it if judge-vs-human ever becomes
  load-bearing.
- **Curation infra is disposable.** The gcloud box was deleted after
  archiving; everything needed to re-run lives in git (tooling), the
  hub (data + provenance), and this doc. Re-provisioning is an init
  script + ~1 h of downloads.

## Open items

- **Train-side consumption arms** — instruction sampling from
  `suggested_instructions`, camera-kind prefix tags with
  dropout-to-unknown, subgoal conditioning/prediction, per-frame aux
  targets: tracked in `architecture.md` §8.10/§2.4, with first
  corpus-scale results in §7. Paired attribution of curation vs
  architecture gains still owed there.
- **`fast_tokenizer_v2`** — percentile-bounded alphabet fit on the
  curated corpus; coordinate with the AR experiments (token metrics
  never compare across tokenizer versions).
- **Judge evolution** — any prompt/schema change re-judges by hash; the
  aggregation tool (`bijou.judge.aggregate`) and the calibration
  methodology are the gate before trusting a new judge configuration at
  corpus scale.
