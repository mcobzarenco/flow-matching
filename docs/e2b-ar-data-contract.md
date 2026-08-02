# Episode annotations — what exists on disk and how to read it

Judge-produced annotations ride *inside* each LeRobot dataset: nothing
here needs the judging stacks, the Anthropic API, or a GPU — a plain
`LeRobotDataset` plus two small JSON files carries everything. This
document is the mechanical consumption contract: what each annotation
means, where it lives, and the exact code that reads it. Every snippet
below was run verbatim against the reference dataset before landing in
this file.

Reference dataset with all surfaces live:
**`mcobzarenco/so101_pick_place_v2`** (12 judged episodes, materialized
2026-08-02). How labels get produced and validated is
`episode-judging.md`'s territory; train-time *usage* policy (weighting,
dropout, curricula) is deliberately out of scope here.

## Provenance model (read this first)

- `meta/judgments.json` (the **sidecar**) is the source of truth: raw
  verdicts keyed by `(episode_index, model, prompt_hash)`. Several
  models' verdicts coexist per episode; a consumer pins one pair.
- Everything per-frame is a **regenerable projection** of one pinned
  selection, materialized into native LeRobot surfaces by
  `python -m bijou.judge.materialize` (full-dataset rewrite,
  idempotent).
- `meta/judge_annotations.json` stamps which selection the projections
  were built from. Check it against your pin and fail loudly on
  mismatch — never assume columns match the sidecar records you happen
  to be reading:

```python
import json
from pathlib import Path
from bijou.judge import PROMPT_HASH

root = Path("~/w/datasets/marius/so101_pick_place_v2").expanduser()
stamp = json.loads((root / "meta/judge_annotations.json").read_text())
assert stamp["prompt_hash"] == PROMPT_HASH, (
    f"columns built at {stamp['prompt_hash']}, loader pins {PROMPT_HASH}"
)
judge_model = stamp["model_filter"] or stamp["models"][0]
```

## Inventory

| annotation | granularity | lives in | read via |
|---|---|---|---|
| verdict, scores, completion, instruction quality, suggested instructions, issues | episode | sidecar | `bijou.judge.store.load_sidecar` |
| camera kinds (majority vote) | dataset | `meta/camera_kinds.json` | `json.loads` |
| subgoal text | every frame (piecewise-constant) | `language_persistent` column, `style="subtask"` | `active_at(t, ...)` |
| events (free text) | exact firing frame | `language_events` column, `style="event"` | `emitted_at(t, ...)` |
| progress ∈ [0,1] | judge-sampled frames only | `annotation.progress` float32 column, NaN elsewhere | item + `isfinite` mask |
| holding ∈ {0,1} | judge-sampled frames only | `annotation.holding` float32 column, NaN elsewhere | item + `isfinite` mask |
| object/gripper visibility per camera | judge-sampled frames only | `annotation.visible_object` / `annotation.visible_gripper` float32 vectors, NaN elsewhere | item + feature `names` |

## 1. Episode-level fields (sidecar)

Verdict/scores for filtering and weighting; `instruction_quality` +
`suggested_instructions` for task-string relabeling. The sidecar is
plain JSON next to the rest of the metadata, so the hub carries it:

```python
from bijou.judge import PROMPT_HASH
from bijou.judge.store import load_sidecar

records = [
    r
    for r in load_sidecar(root)  # [] when the dataset was never judged
    if r.prompt_hash == PROMPT_HASH and r.model == judge_model
]
by_episode = {r.episode_index: r.parsed_judgment() for r in records}

j = by_episode[0]  # EpisodeJudgment — typed, validated on parse
j.verdict.value              # "keep" | "review" | "discard"
j.overall_score              # 1..10 (j.scores.* for the four subscores)
j.task_completion_visible    # yes | partial | no | unclear
j.instruction_quality.value  # good | vague | mismatched | placeholder
j.suggested_instructions     # 2-3 grounded rewrites, always present
j.issues                     # free-text problem list, often empty
```

`parsed_judgment()` validates under the *current* schema — records at
other prompt hashes obey their own prompt's schema, which is why the
hash filter comes first.

## 2. Camera kinds (dataset-level majority vote)

Per-episode tags flip on ambiguous views, so consume the majority-vote
map, not individual verdicts (`python -m bijou.judge.aggregate
--write-camera-maps` produces it). Ties resolve to `"unknown"` and are
flagged — on the reference dataset the overhead camera is a true 6–6
`front`/`top` tie:

```python
kinds = json.loads((root / "meta/camera_kinds.json").read_text())
assert kinds["prompt_hash"] == PROMPT_HASH
{cam: v["kind"] for cam, v in kinds["cameras"].items()}
# {'front': 'unknown', 'wrist': 'wrist'}   <- kind ∈ wrist|top|front|side|unknown
kinds["cameras"]["front"]["tie"]  # True — vote detail kept for consumers
```

Camera keys here are short names (`"front"`, not
`"observation.images.front"`), matching every other annotation surface.

## 3. Per-frame scalars: progress and holding

Ordinary float32 feature columns; **NaN means the judge never saw that
frame**. The finite mask IS the judge's sampled-frame set (~5–20
evenly-spaced frames per judged episode). Never interpolate between
sampled frames — the frames in between were not observed:

```python
import torch
from bijou.judge.materialize import EVENT_STYLE  # registers style="event" (needed once, see §5)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("marius/so101_pick_place_v2", root=str(root))
item = ds[0]

mask = torch.isfinite(item["annotation.progress"])  # supervise only where True
item["annotation.progress"]  # 0.0  (fraction of task complete at this frame)
item["annotation.holding"]   # 0.0 / 1.0: gripper physically holds the object
```

Unjudged episodes carry the columns too (all-NaN), so one code path
serves the whole corpus. The columns compose with `delta_timestamps`
like any other feature — e.g. progress now and one action-chunk ahead:

```python
ds = LeRobotDataset(
    "marius/so101_pick_place_v2",
    root=str(root),
    delta_timestamps={"annotation.progress": [0.0, 50 / 30]},
)
ds[0]["annotation.progress"]  # tensor([0., nan]) — future frame unsampled here
```

## 4. Visibility vectors

Two vectors per frame, one slot per camera; slot order is the feature's
`names` (sorted camera short names — same order everywhere). NaN-masked
like the scalars:

```python
names = ds.meta.features["annotation.visible_object"]["names"]  # ['front', 'wrist']
item["annotation.visible_object"]   # tensor([0., 0.]) — task object visible per camera
item["annotation.visible_gripper"]  # same layout for the gripper
```

## 5. Subgoals and events (language columns)

**Subgoals** are piecewise-constant text covering *every* frame of a
judged episode: `language_persistent` rows (`style="subtask"`) activate
at their segment's first frame and persist until superseded. This is
also what the online dataset visualizer's Annotations tab renders.

**Events** (drops, resets, interventions, progress regressions) are
momentary rows in `language_events` (`style="event"`) stored on the
exact firing frame. The `"event"` style is project-local — import
`bijou.judge.materialize` (anywhere, once) before using lerobot's
resolvers on it, or they reject the unknown style:

```python
from bijou.judge.materialize import EVENT_STYLE  # side effect: registers "event"
from lerobot.datasets.language_render import active_at, emitted_at

t = float(item["timestamp"])
subgoal = active_at(t, persistent=item["language_persistent"], style="subtask")
subgoal["content"]  # 'reach toward the toy boat' — non-None on every judged frame

event = emitted_at(
    t,
    persistent=item["language_persistent"],
    events=item["language_events"],
    style=EVENT_STYLE,
)  # None on frames without an event (the overwhelming majority)
```

Event negatives are only defined on judge-sampled frames: a sampled
frame (finite `annotation.progress`) with no event row is a true "no
event"; an unsampled frame is *unknown*, not negative.

## Coverage (2026-08-02)

| where | sidecar | columns + language + camera map |
|---|---|---|
| rig `so101_pick_place_v2` | 12/50 episodes (opus) | ✅ materialized (the reference) |
| `curated_v0` on curation-1 (981 datasets / 52.5k episodes) | 2 episodes/dataset, opus + haiku (calibration pilot) | ❌ not yet — materialization happens during the curation merge (TODO 6), which rewrites every parquet anyway |
| `mcobzarenco/community_curated_v1` (hub) | ships fully judged + materialized | not yet built |

Until the curated collection ships, corpus-wide code must tolerate
missing sidecars (`load_sidecar` → `[]`), missing `annotation.*`
features, and missing `camera_kinds.json` — presence of
`meta/judge_annotations.json` is the cheap "materialized?" probe.

## Invariants

- **Indexing**: judge fields (`until_frame`, `frame`) are 1-based
  inclusive; lerobot `frame_index` is 0-based. The materializer resolves
  this once — consumers of the columns/language rows never touch judge
  frame numbers. Only sidecar-direct consumers (episode-level fields)
  need care, and those fields carry no frame numbers.
- **Pinning**: one `(model, prompt_hash)` per training run's labels;
  verify via `meta/judge_annotations.json` (§ provenance) rather than
  re-deriving from sidecar records.
- **Renumbering invalidates**: any episode-renumbering rewrite (merge
  tools) must remap sidecars and re-materialize; consume post-merge
  datasets only after that happened (the merge tool's job — check the
  stamp's `written_at` if in doubt).
- **Weak supervision** (measured on rig, gripper-aperture channel as
  ground truth): `holding` ≈ 75–85% per-frame agreement with a
  systematic open-hover→true bias; cross-model (opus/haiku, 18.7k paired
  frames) holding agreement 80.7%, progress MAE 0.15. Mask losses,
  weight modestly, and re-measure before trusting more.
