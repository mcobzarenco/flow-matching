# Community datasets → one filtered v3.0 corpus: pipeline plan

Goal: take the union of `HuggingFaceVLA/community_dataset_{v1,v2,v3}`
(disjoint release batches, not format versions), migrate every sub-dataset to
LeRobot **v3.0**, filter episodes with VLM judges, and assemble a single
high-quality corpus for VLA training.

## 0. Scale & ground truth

| collection | sub-datasets | episodes | frames | size | format | embodiments |
|---|---|---|---|---|---|---|
| v1 | 128 | 11,132 | 5.11M | 259 GB | v2.0 + v2.1 mix | SO-100 only |
| v2 | 340 | 6,325 | 5.03M | 264 GB | v2.0 + v2.1 mix | SO-100 only |
| v3 | 791 | 50,622 | 25.97M | 758 GB | per-episode v2.x layout ("v2.1+") | 46+ types, 12 action-dim configs, varying fps |
| **union** | **~1,259** | **~68k** | **~36.1M** | **~1.28 TB** | | |

Facts already established empirically on v1:
- The collection-level cleaning pipeline left **stats keys missing the
  `images.` segment** in every dataset checked (51/51). Assume v1+v2 are all
  affected; verify per dataset.
- Some episodes are internally inconsistent (`length` ≠ video span) — e.g.
  re-record leftovers. These crash `lerobot-edit-dataset`'s video re-encode
  and must be quarantined or deleted, not merged.
- v2.0-format datasets (2 in the local v1 snapshot) cannot be converted by
  lerobot 0.6's v21→v30 script directly; they need the v2.0→v2.1 hop
  (`episodes_stats.jsonl` generation) that only ships in lerobot ≤0.3.x.

## 1. Infrastructure

- **Do everything on a cloud box, not the laptop.** 1.28 TB in + ~1.3 TB
  working copies + final corpus ⇒ provision ≥ **4 TB** volume. Laptop-side
  bandwidth (~0.5–2 MB/s observed) makes local processing a non-starter;
  datacenter pipe pulls 1.28 TB in hours.
- Provision with `init-vm.sh` (H100 box doubles as the judge machine).
- `export HF_TOKEN` on the box (collections are login-gated).
- Everything under `~/corpus/{raw,converted,filtered,merged}` + a
  `manifest.jsonl` (one record per sub-dataset per stage).

## 2. Pipeline stages (manifest-driven, idempotent, resumable)

Each stage reads/writes `manifest.jsonl`; a crashed run resumes by skipping
records already marked done. Failures quarantine the dataset (status +
error), never abort the sweep.

### Stage A — Inventory
One `list_repo_tree` per collection → per-sub-dataset record: path, file
count, byte size, `codebase_version` (fetch each `meta/info.json`, ~2 KB).
Output: the master worklist (~1,259 rows). Also dedupe: if the same
`contributor/name` appears in multiple collections, keep the newest.

### Stage B — Download (bounded disk)
Per sub-dataset: `hf download <collection> --include "<user>/<ds>/**"
--local-dir raw/...` (resumable). Process in batches of ~20 datasets;
delete `raw/` copies after Stage D succeeds so peak disk stays bounded.
Verify: local file count+sizes vs hub tree (script exists from the v1
integrity check).

### Stage C — Format normalization
- `v2.1` → nothing.
- `v2.0` → generate `meta/episodes_stats.jsonl` per episode + drop
  `stats.json` (the v2.0→v2.1 delta). Preferred: run old converter in an
  ephemeral env: `uv run --no-project --with "lerobot==0.3.2" python -m
  lerobot.datasets.v21.convert_dataset_v20_to_v21 ...`; fallback: implement
  the delta ourselves (small, well-understood).
- Anything else (or missing/corrupt info.json) → quarantine.

### Stage D — Convert to v3.0
`uv run python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id
<user>/<ds> --root converted/<user>/<ds> --push-to-hub false` (on a copy;
converter swaps in place and leaves `_old`, which we delete).
Then two post-passes:
1. **stats-key repair**: for every `observation.images.X` feature whose
   stats entry is keyed `observation.X`, re-key (generalization of the
   ZGGZZG fix).
2. **integrity gate**: load with `LeRobotDataset`; check per episode
   `length == dataset_to_index - dataset_from_index == video_span × fps`
   for every camera; episodes failing → record in manifest for Stage F
   deletion (or quarantine dataset if majority broken).

### Stage E — Judge
Two-tier, to balance cost and quality:
1. **First pass: local Gemma 4 12B** on the H100 (bf16, ~10 s/episode as
   measured; ~5 s with 4 frames + budget 70 + short max_new_tokens).
   68k episodes ≈ 90–190 GPU-hours sequential ⇒ **use vLLM** (or 2–4 GPU
   workers) to get continuous batching; target < 24 h wall-clock.
   Output: JSONL of `Judgment` records keyed (collection, dataset, episode).
2. **Adjudication: Claude** (`fmatch/judge_episode.py`, sonnet) only for the
   disagreement/uncertainty band: verdict == "review", parse failures, and
   a random QA sample of keeps/discards. At ~4.3k tokens/episode, budget
   scales with band size (10% band ≈ 30M tokens ≈ $90–100 sonnet input).

**Calibration before the full sweep (mandatory):** the current prompt is
harsh (ZGGZZG episodes all scored 3/10 discard; owner disagreed on one).
Hand-label ~100 episodes sampled across collections/embodiments, tune
prompt + keep-threshold until precision/recall against hand labels is
acceptable. v3's embodiment diversity (mobile, humanoid) needs prompt
generalization — the current text assumes arm teleop. Add per-embodiment
context strings.

### Stage F — Filter
Per dataset: `delete_episodes` for (judge-discards ∪ integrity failures),
via `lerobot-edit-dataset` / `lerobot.datasets.dataset_tools`, writing to
`filtered/`. Datasets with zero surviving episodes are dropped. Keep the
judgment JSONL as provenance.

### Stage G — Group & merge
One physical LeRobotDataset requires identical features, so a single
monolith across 46 embodiments / 12 action dims / mixed fps is impossible
without destructive padding. Plan: **merge per feature-signature group**:
- signature = (robot_type, action/state dim + names, camera count,
  resolution, fps)
- standardize camera names within each group to `camera1..N`
  (scene-view first, wrist second where identifiable by name — else listing
  order; keep the mapping in the manifest)
- merge each group with lerobot's aggregation tooling
  (`lerobot-edit-dataset` merge / `aggregate_datasets`), regenerate stats.
Result: a handful of BIG datasets (the SO-100/101 @30fps group will dominate:
roughly v1+v2 entirely + ~88% of v3). Training can consume several groups via
`--dataset.repo_id` lists (lerobot-train accepts a list and keeps common keys).

### Stage H — Final QA
- Integrity re-check on every merged dataset (the Stage D gate re-run).
- Spot-check in the web visualizer (v3 groups get the Statistics/3D tabs).
- Corpus datasheet: per-group episode/frame/hour counts, provenance mapping,
  judge score distributions, discard rates by collection.

## 3. Cost/time envelope (single H100 box)

| stage | estimate |
|---|---|
| download 1.28 TB | ~4–8 h on a datacenter pipe |
| v2.0 hops + v21→v30 conversion | IO-bound; the 33-ep test converted in ~1 s ⇒ hours for the corpus |
| stats repair + integrity gate | hours |
| judging (Gemma, vLLM-batched) | the long pole: target < 24 h; sequential worst case ~8 days |
| Claude adjudication | $ scales with band; ~$100 per 10% of corpus |
| filter + merge | hours (video re-encode only where files mix keep/discard) |
| **wall-clock** | **a weekend, mostly judge-bound** |

## 4. Risks / open questions

- **Judge validity is the whole ballgame** — a miscalibrated judge silently
  deletes half the corpus or keeps junk. Calibration set + QA sampling is
  non-negotiable; keep judgments as data so thresholds can be re-decided
  without re-judging.
- Heterogeneous v3 datasets may hide surprises (depth cameras, missing
  tasks, odd fps, audio?) — the quarantine path and per-signature grouping
  absorb these, but expect a tail of manual triage.
- `delete_episodes` re-encodes video segments — CPU-heavy at corpus scale;
  measure on a few datasets, consider `--video_backend`/encoder settings.
- Hub rate limits: authenticated + Xet helps; per-dataset `--include`
  downloads are many small requests — batch sensibly.
- Licensing/attribution: community data is Apache-2.0 per collection cards,
  but keep the provenance manifest in the final corpus regardless.
