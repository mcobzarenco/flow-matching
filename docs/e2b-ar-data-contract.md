# Data contract — full-E2B AR decoder with auxiliary targets

Audience: the agent building the decoder-only model on a **full**
(untruncated) Gemma-4 E2B that autoregressively predicts FAST action
tokens plus auxiliary targets (current subgoal, holding state, …),
conditioned on task text, semantically tagged camera images, and
projected robot state. This document specifies what data the model's
loader can rely on, where it lives, its exact semantics, and its known
weaknesses. Statuses are dated; measured numbers say how they were
measured. Model architecture is the reader's domain — this is the data
side of the line, plus one vocabulary note at the end.

## Corpora

| corpus | where | status (2026-08-02) |
|---|---|---|
| community, in-scope (dims 6/6 ∧ fps=30: 991 datasets, 53.2k eligible episodes) | `/data/source/{v1,v2,v3}` on `curation-1` (gcloud); hub mirrors `mcobzarenco/community_dataset_v{1,2,3}_v3` | downloaded; mechanical filter + judge sweep pending |
| `mcobzarenco/community_curated_v1` (single filtered+annotated collection) | hub, eventually | NOT YET BUILT — the merge tool (curation TODO 6) produces it |
| rig | `mcobzarenco/so101_pick_place_{clean,v2}` | v2 has 12 judged episodes with full annotations (the reference for every claim below); a merge + camera rename `front→top` is planned (TODO 2b) — repo id and episode numbering will change |

Until the curated collection ships, code against the interfaces below,
not against corpus-wide annotation availability: today only rig v2 and
the 6 local sample datasets carry sidecars.

## Per-sample inputs

- **Task description**: lerobot `tasks` per episode. Community task
  strings are frequently junk — judged `instruction_quality`
  (good/vague/mismatched/placeholder) plus 2–3 grounded
  `suggested_instructions` per judged episode live in the sidecar;
  recommended trainng-time policy: sample among {original, suggestions}
  when quality is good/vague, suggestions only when
  mismatched/placeholder.
- **Camera images**: `observation.images.*`, 1–4 cameras (corpus mix:
  889×2-cam, 233×3-cam, 100×1-cam, 20×4-cam), mostly 640×480@30fps.
  Decode via `LeRobotDataset`; corrupt frames exist in community data —
  bijou's substitution guard (`StatsAttachedDataset`) is the precedent.
- **Semantic camera names**: judged `camera_kinds` maps each dataset
  camera name → `wrist|top|front|side|unknown` (judged VISUALLY under
  anonymous labels — dataset names carry no signal in the community
  collections, 99.9% anonymized). Per-episode tags flip on ambiguous
  views (measured: oblique overhead ↔ top/front), so consume a
  **per-dataset majority vote** across judged episodes (aggregation
  utility not yet written — trivial over sidecar records). Train with
  dropout-to-`unknown` so deployment tolerates missing tags.
- **Robot state**: `observation.state`, 6-dof. Normalize with the
  dataset's OWN mean/std (per-dataset normalization is load-bearing:
  59–95% of aggregate action variance is between-rig calibration offset —
  architecture.md §4). Stats come from `meta/stats.json`; checkpoints
  must persist the per-dataset table they trained with (see
  `bijou/loading.py` for the schema precedent).

## Action targets — FAST tokens

- **Chunk**: 50 native frames (fps=30 scope ⇒ 1.67 s), from lerobot
  `delta_timestamps`; `action_is_pad` marks repeat-last padding past the
  episode end. Measured on the corpus: 9.78% of frames carry a padded
  tail; mask padded steps out of the loss. Episodes < 50 frames are
  dropped upstream (mechanical filter).
- **Normalization for tokenization**: per-dataset exact q01/q99 from
  `meta/stats.json` mapped to [−1, 1]
  (`bijou.fast.quantile_entry_from_stats`); the exact quantiles were
  backfilled corpus-wide (lerobot's native episode-averaged quantiles
  are wrong — README of lerobot-dataset-tools).
- **Tokenizer**: `mcobzarenco/bijou-checkpoints/fast_tokenizer_v1`
  (vocab 1024, ~53 tok/chunk measured corpus-weighted) — but see
  `fast-tokenizer-v1-review.md`: v1's BPE is degenerate (1019/1024
  alphabet, 5 merges); **v2 with a percentile-bounded alphabet
  (~25–30 tok/chunk expected) is recommended before long training
  runs**, ideally fit on the curated corpus. Token-level metrics never
  compare across tokenizer versions. Decode failures of sampled
  sequences raise `FastDecodeError` — malformed generations are the
  caller's fallback decision (`bijou/fast/codec.py` defines BOA/EOA
  appended after the BPE ids).

## Auxiliary targets — availability and access

Source of truth for ALL judge annotations:
`meta/judgments.json` per dataset (`bijou.judge.store.load_sidecar` →
`JudgmentRecord`; envelope validates at load, payload via
`record.parsed_judgment()` for records whose `prompt_hash` matches).
Records key on `(episode_index, model, prompt_hash)`; multiple judges'
verdicts coexist — the loader must pin `(model, prompt_hash)` in its
config for reproducibility.

| target | semantics | native LeRobot access | coverage |
|---|---|---|---|
| **subgoal** | piecewise-constant text; segments cover every frame of a judged episode | `language_persistent` rows, `style="subtask"` — items expose the column; `lerobot.datasets.language_render.active_at(t, persistent=rows, style="subtask")` resolves the frame's subgoal | every frame of a judged episode |
| **holding** (bool) | gripper physically holds the task object AT the sampled frame; **never interpolate** | `annotation.holding` float32 column: 0/1 at sampled frames, NaN elsewhere — loss mask = `isfinite` | ~10–20 sampled frames per judged episode |
| **progress** (0–1) | task-completion fraction at the sampled frame; dips imply an event | `annotation.progress` float32 column, NaN-masked; works with `delta_timestamps` (e.g. progress at t and t+chunk) | same sampled frames |
| **visibility** | per camera: task object / gripper visible at the sampled frame | `annotation.visible_object` / `annotation.visible_gripper` float32 vectors over cameras (order = the feature's `names`), NaN-masked | same sampled frames |
| **events** | free-text occurrences at a frame (drops, interventions, resets); mistake-marking for BC masking | `language_events` rows, `style="event"`, on the exact firing frame; `emitted_at(t)` resolves them (style registration: import `bijou.judge.materialize`) | rare by design |
| episode quality | verdict/scores/issues — filtering & sample weighting | sidecar only (curation applies discards upstream in the curated corpus) | every judged episode |

All of the above except episode quality are written by
`bijou.judge.materialize` (full-dataset rewrite, idempotent). The
finite-value mask of `annotation.progress` IS the judge's sampled-frame
set: a sampled frame with no event row is a true "no event" negative;
an unsampled frame is unknown. `meta/judge_annotations.json` stamps the
`(model, prompt_hash)` selection the columns were built from — check it
against the loader's pin instead of trusting sidecar joins. Reference
dataset with all surfaces live: `mcobzarenco/so101_pick_place_v2`.

Known label quality (measured on rig, aperture channel as ground truth):
`holding` ≈ 75–85% per-sampled-frame agreement, errors concentrated at
transitions with a systematic open-gripper-hover→true bias; `progress`
monotone on clean episodes; camera tags need majority voting. **Treat
auxiliary labels as weak supervision** — masked losses, modest weights,
and re-measure at calibration (curation TODO 4).

Per-frame scalars ride ordinary feature columns — no sidecar joins in
the loader. The sidecar remains the provenance store and the only home
of episode-level quality fields.

## Suggested target-sequence assembly (informative, not binding)

Per sampled training frame: prefix = [task text ⊕ per-camera semantic
tag + image ⊕ projected state] → decode [aux: current subgoal text,
holding token, …] → [BOA] 50-step FAST tokens [EOA]. Aux-before-actions
matches the stated design ("finish the output sequence with the
actions"); note aux supervision exists only on judged episodes'
sampled frames — the aux segment must be maskable per sample. Subgoal
text via `active_at` is available at EVERY frame of a judged episode
(piecewise-constant), so subgoal conditioning/prediction need not be
restricted to sampled frames; `holding`/`progress` must be.

## Vocabulary note (the one architecture question asked)

Adding FAST tokens to E2B's vocabulary — decision procedure, in order:

1. **Reuse a reserved/unused token block if one is large enough**
   (≥ 1024 + BOA/EOA). π0-FAST does exactly this on PaliGemma's
   reserved `<loc/seg>` blocks. Zero shape changes, checkpoints stay
   load-compatible, no optimizer-state surgery. Check the E2B tokenizer
   for `<unusedNN>`/reserved ids first — historically Gemma tokenizers
   reserve only ~99 `<unused>` slots, which is NOT enough, but verify on
   the actual E2B tokenizer before ruling it out.
2. Otherwise **resize embeddings** (+1026 rows): initialize new rows
   from the embedding mean (+ small noise) rather than random — mean
   init measurably stabilizes early loss on extended vocabs; Gemma ties
   input/output embeddings, so one matrix grows and the LM head follows
   automatically. Consequences to plan for: checkpoint shapes diverge
   from stock E2B (record the mapping in the checkpoint config), and the
   softmax over 262k+1026 ids is unchanged cost-wise.

Either way, `bijou/fast` token ids `[0, vocab)` + BOA/EOA map into the
chosen id block by a constant offset — keep that offset in the
checkpoint config, not in code.

## Invariants the loader must respect

- Frame indexing: judge `until_frame`/`frame` fields are **1-based
  inclusive**; lerobot `frame_index` is 0-based. Off-by-one here
  silently corrupts every aux label.
- Sidecar/`language_persistent` selection by `(model, prompt_hash)` —
  never mix prompt versions in one training run's labels.
- Episode renumbering (curation merge, rig dataset merge) invalidates
  sidecar `episode_index` keys — consume post-merge datasets only after
  their sidecars/columns were remapped (merge tool's job, not yours).
- Eval comparability: any new frame set (curated corpus, merged rig)
  starts its own ledger section (architecture.md §7 rules).
