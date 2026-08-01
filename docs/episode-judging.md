# Episode judging — VLM curation of the training corpus

`bijou/judge` scores individual dataset episodes with vision-language
models, as curation metadata for training. This document records the
approach and the invariants; CLI specifics live in each module's
docstring (`bijou.judge.claude`, `bijou.judge.gemma`, `bijou.judge.sweep`).

## Why judges

The community corpus is crowdsourced and uneven, and its problems are
mostly *not* detectable from trajectories alone. Three per-episode signals
need eyes on pixels:

1. **Demonstration quality** — did a competent, complete demonstration
   happen (vs fumbling, idling, failure, unusable cameras)?
2. **Label quality** — task strings are frequently junk ("test1" on a
   perfectly good demo) or describe something other than what happens.
   A wrong label on a good demo is a *relabeling* opportunity, not a
   discard — the two axes are scored independently.
3. **Camera semantics** — the converted collections anonymize camera
   names (`image`, `image2`, ... in ~99.9% of datasets, measured 2026-07)
   with inconsistent ordering, so wrist/top/front/side identity exists
   only in the pixels. The judge tags each camera visually, with
   `unknown` as the honest fallback — which doubles as a train-time
   dropout target for camera annotations.

Each verdict sees a sparse sample of frames (every camera at ~10 evenly
spaced timesteps) plus statistics computed over the **full** trajectory
from parquet (travel, per-step jumps, tracking error, idle fraction), so
whole-episode claims don't rest on sampled frames. Judging is metadata
work: no GPU needed for the API judge, no video decode beyond the sampled
timesteps.

## The judges

Two judges emit the same strict-JSON verdict from the same evidence
(`bijou/judge/evidence.py`) and the same prompt (`bijou/judge/schema.py`):

- **`bijou.judge.claude`** — Anthropic API. The workhorse: fast to run at
  scale from any machine, no local GPU. Non-deterministic by nature (the
  API guarantees none; newer models reject sampling controls outright), so
  provenance rides in each record instead: model id + prompt hash.
- **`bijou.judge.gemma`** — local Gemma 4 via plain transformers (bf16,
  single device, no quantization backends). Greedy decode makes it the
  reproducible-verdict path, and marginal cost is zero on our own GPUs —
  the intended high-volume judge once calibrated against the API judge.

## The verdict schema — the parser is the schema

`EpisodeJudgment` mirrors the prompt's demanded JSON, and `from_dict`
enforces it exhaustively: required fields, enum membership, integer 1–10
scores (no bools/strings/fractions sneaking through coercion), non-empty
relabels, camera map covering exactly the shown cameras. There is
deliberately **no separate jsonschema document** — the prompt and the
dataclass are already two sources of truth; a third would drift. A verdict
violating the schema is a loud parse failure to be retried, never
backfilled or clamped: a silently-wrong score poisons aggregation, which
is worse than a crash.

Fields, briefly: `overall_score` + `verdict` (keep/review/discard) +
per-aspect scores + `issues`/`summary` judge the demonstration;
`instruction_quality` (good/vague/mismatched/placeholder) judges the
label; `observed_task` + `suggested_instructions` are grounded relabels
usable directly as training instructions (hindsight relabeling when the
stated task isn't what happened); `subgoals` segments the episode into
sequential phases ("reach toward the boat", "grasp and lift", …) with
1-based inclusive `until_frame` boundaries — every frame inherits its
segment's label (`subgoal_at`), boundaries quantized to the sampled
timesteps; `camera_kinds` maps each camera name to
wrist/top/front/side/unknown.

## Prompt identity is a hash, not a version

`PROMPT_HASH` is a short content digest of the system prompt, recorded
with every stored verdict. Editing the prompt changes the hash, which
automatically makes old verdicts non-matching (they remain stored — they
are still valid verdicts *of that prompt*); nothing is bumped by hand,
and forgetting to bump is impossible. Verdicts are comparable iff they
share `(model, prompt_hash)`.

## Storage: journal + per-dataset sidecar

Two layers with different lifetimes:

- **Journal** (`--output` JSONL): the sweep's write-ahead log. One line
  per episode as results stream in, ok and failed alike, crash-safe.
  Operational, machine-local, disposable after merging.
- **Sidecar** (`meta/judgments.json` inside each dataset): the durable
  store. Successful verdicts folded in from the journal (auto-merge at
  run end; `--merge-only` recovers an interrupted run; merging is
  idempotent, writes are atomic). Records keep `EpisodeJudgment.to_dict()`
  verbatim under `"judgment"` plus provenance (model, prompt hash,
  evidence parameters, token usage), keyed by
  `(episode_index, model, prompt_hash)`.

The sidecar lives *inside* the dataset directory on purpose: hub
upload/download carries it, any machine holding the dataset holds its
verdicts, and train-time consumers read it next to the rest of the
metadata with a light import (`bijou.judge.store`). The record envelope
validates at load; the verdict payload validates at consumption
(`parsed_judgment()`), because a sidecar legitimately mixes prompt
versions and each payload obeys its own prompt's schema — consumers pick
records by `prompt_hash` first.

Consequences of the key choice: re-running the same configuration is a
no-op on any machine with the sidecars; switching model re-judges
deliberately (cascades, cross-model calibration — multiple models'
verdicts coexist per episode); prompt edits re-judge automatically.
Failures stay journal-local: retrying costs nothing (evidence gathering
fails before any API spend), so fresh machines retry transient failures
instead of inheriting stale quarantines. Mechanical skips (episodes
shorter than one action chunk) are recomputed at plan time — a pure
function of episode length needs no memory.

Known caveat, recorded where it bites: lerobot's `delete_episodes`
renumbers `episode_index`, so any dataset rewrite must remap or
deliberately drop the sidecar.

## Sweeping and cost discipline

The sweep plans from metadata only (episode lengths, camera counts),
prices the plan before spending (`--dry-run`), and runs episodes in
spawn-isolated worker processes — a decoder crash on a corrupt community
video fails one episode, not the sweep, and the sweep doubles as a
corpus-wide video-integrity census.

Cost scales with images × per-image tokens; measure, don't guess: the
free `count_tokens` endpoint prices real payloads exactly (the in-code
estimate constants were calibrated that way and say so). Levers, in
decreasing order of impact: model choice (opus : sonnet : haiku pricing
spans ~7:1), batch submission (flat 50% off), corpus scoping (e.g.
trainable-dims + dominant-fps datasets only), evidence size
(timesteps × cameras × resolution). Camera count varies per dataset
(1–4), so per-episode cost does too.

## Calibration before trust

Judge verdicts are advisory until calibrated. Before filtering a corpus
on them:

- Hand-label a stratified episode sample; measure judge–human agreement
  and pick thresholds numerically (asymmetric costs: a false discard
  wastes one episode, a false keep pollutes training).
- Measure cross-model agreement (API vs local; expensive vs cheap model)
  on the same episodes — the sidecar keeps coexisting verdicts precisely
  for this. A cheap-model sweep with expensive-model escalation on
  disagreement/review is the intended end state.
- Aggregate `camera_kinds` per dataset by majority vote across episodes:
  single-episode tags flip on ambiguous views (downward-pitched wrist
  cams vs `top`); disagreement across episodes of one dataset is itself
  a signal worth surfacing.
- Since API verdicts are non-deterministic, judge a subsample twice and
  fold run-to-run variance into the agreement analysis.

## Train-time consumption (the point of all this)

Planned uses, in rough order of ambition — the sidecar schema already
carries what each needs:

1. **Filtering**: drop `discard` episodes via the existing
   `LeRobotDataset(episodes=...)` mechanism (same path the holdout split
   uses). Dataset-level aggregates (e.g. most episodes discarded) flag
   whole datasets for exclusion.
2. **Soft weighting**: sampler weights from scores instead of a binary
   cut — keeps data while down-weighting mediocrity.
3. **Camera annotations**: per-dataset camera-kind maps feed the prefix
   collator (e.g. text tags per view), with dropout to `unknown` so the
   policy tolerates missing annotations at deployment.
4. **Instruction augmentation**: sample from `suggested_instructions`
   (plus the original task string) at train time — label diversity and
   hindsight relabels with zero new video.
5. **Subgoal conditioning**: condition the prefix on the current frame's
   subgoal (piecewise-constant between judged boundaries) — finer
   language grounding for long-horizon episodes, and a natural
   inference-time steering hook.

Each is a measured experiment against eval baselines, not a default: the
curation signal earns its place in training the same way any other change
does.
