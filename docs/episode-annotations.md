# Episode annotations from the LLM judges

Datasets that went through the judging pipeline carry a set of
annotations beyond what teleoperation recorded: an episode-quality
verdict, relabeled task instructions, semantic camera tags, subgoal
segments, and sparse per-frame labels (progress, holding, visibility,
events). They ride *inside* the dataset — reading them needs a plain
`LeRobotDataset`, two small JSON files, and no GPU, no API, and none of
the judging stacks. This is the how-to for augmenting a training run
with them: what each annotation means, where it lives, and the code
that reads it.

## How the annotations are produced

An LLM judge (vision-capable) sees, per episode: the stored task
instruction, evenly spaced frames from every camera (one timestep per
~1.5 s of episode, clipped to 5–20, first and last frame always
included), and full-trajectory motor statistics computed from the
parquet. It returns one strict-JSON verdict that is parsed and
validated before storage — frame annotations must cover exactly the
sampled timesteps, subgoal segments must cover exactly the episode,
progress regressions require an explaining event; anything else is
rejected, not repaired. Cameras are shown under anonymous labels (A,
B, …) so recorded names cannot bias the viewpoint call, and are
translated back to dataset names after validation.

Raw verdicts land in a per-dataset sidecar, `meta/judgments.json`,
keyed by `(episode_index, model, prompt_hash, num_timesteps,
max_image_dim)` — the prompt hash is content-derived, and the evidence
fields identify the image selection shown to the judge, so re-judging
under a different configuration coexists rather than overwrites. A separate materialization step projects one pinned selection of
those verdicts into native LeRobot surfaces (feature columns and
language rows), which is what training code consumes; the sidecar
remains the provenance store and the only home of episode-level fields.

## Provenance and pinning

`meta/judge_annotations.json` stamps which `(model, prompt_hash)`
selection the materialized surfaces were built from. The stamp is the
blessed selection, and it ships in-band: consumers of the columns and
language rows need no hash knowledge at all — there is exactly one
materialization at a time. What the stamp is *for*:

- **log it with every training run** (config/W&B), so runs are
  comparable and a re-materialized corpus is visible in the record;
- **filter sidecar reads by its fields** (§ episode-level fields), so
  episode-level labels provably match the per-frame columns;
- optionally **pin a literal** in a training config when a sweep must
  fail loudly if the corpus is re-materialized under a newer prompt
  mid-sweep. (Asserting against the imported `bijou.judge.PROMPT_HASH`
  is only right when trainer and judge run from the same checkout —
  the constant advances with the code, not with your labels.)

```python
import json
from pathlib import Path

root = Path("/path/to/dataset")  # contains meta/, data/, videos/
stamp = json.loads((root / "meta/judge_annotations.json").read_text())
run_config["judge_labels"] = stamp  # provenance travels with the run
judge_model = stamp["model_filter"] or stamp["models"][0]
```

Dataset-wide statistics are independent of all of this by design:
`meta/stats.json` derives from robot data only, and annotation columns
carry no stats entries (nothing normalizes labels; NaN would poison
stats aggregation). Re-judging or re-materializing never changes
normalization — the two only meet at curation time, when judge-gated
episode filtering changes the episode set and the merge tool recomputes
stats for what survives.

## What is available

| annotation | granularity | lives in | read via |
|---|---|---|---|
| verdict, scores, completion, instruction quality, suggested instructions, issues | episode | sidecar | `bijou.judge.store.load_sidecar` |
| camera kinds (majority vote) | dataset | `meta/camera_kinds.json` | `json.loads` |
| subgoal text | every frame (piecewise-constant) | `language_persistent` column, `style="subtask"` | `active_at(t, ...)` |
| events (free text) | exact firing frame | `language_events` column, `style="event"` | read the item's rows directly (a frame may carry several) |
| progress ∈ [0,1] | judge-sampled frames only | `annotation.progress` float32 column, NaN elsewhere | item + `isfinite` mask |
| holding ∈ {0,1} | judge-sampled frames only | `annotation.holding` float32 column, NaN elsewhere | item + `isfinite` mask |
| object/gripper visibility per camera | judge-sampled frames only | `annotation.visible_object` / `annotation.visible_gripper` float32 vectors, NaN elsewhere | item + feature `names` |

## Episode-level fields (sidecar)

Verdict and scores drive filtering and sample weighting;
`instruction_quality` and `suggested_instructions` drive task-string
relabeling and instruction augmentation. The sidecar is plain JSON next
to the rest of the metadata, so hub upload/download carries it:

```python
from bijou.judge.store import load_sidecar

records = [
    r
    for r in load_sidecar(root)  # [] when the dataset was never judged
    if r.prompt_hash == stamp["prompt_hash"] and r.model == judge_model
]
by_episode = {r.episode_index: r.parsed_judgment() for r in records}

j = by_episode[0]  # EpisodeJudgment — typed, validated on parse
j.verdict.value  # "keep" | "review" | "discard"
j.overall_score  # 1..10 (j.scores.* for the four subscores)
j.task_completion_visible  # yes | partial | no | unclear
j.instruction_quality.value  # good | vague | mismatched | placeholder
j.suggested_instructions  # 2-3 grounded rewrites, always present
j.issues  # free-text problem list, often empty
```

`parsed_judgment()` validates under the *current* schema — records at
other prompt hashes obey their own prompt's schema, which is why the
hash filter comes first.

## Camera kinds

Each camera gets a semantic tag: `wrist | top | front | side |
unknown`. Single-episode tags flip on ambiguous views, so the
per-dataset **majority vote** across judged episodes is the consumable
form; ties resolve to `unknown` (which doubles as a natural train-time
dropout target for camera tags) and are flagged so genuine ambiguity
stays visible:

```python
kinds = json.loads((root / "meta/camera_kinds.json").read_text())
assert kinds["prompt_hash"] == stamp["prompt_hash"], (kinds, stamp)
{cam: v["kind"] for cam, v in kinds["cameras"].items()}
# e.g. {'overhead': 'top', 'gripper_cam': 'wrist'}
kinds["cameras"]["overhead"]["tie"]  # True on a genuine split vote
```

Camera keys are short names (`"overhead"`, not
`"observation.images.overhead"`) — the same convention as every other
annotation surface.

## Progress and holding (per-frame scalars)

Ordinary float32 feature columns; **NaN means the judge never saw that
frame**. The finite mask IS the judge's sampled-frame set (5–20 evenly
spaced frames per judged episode). Supervise through the mask and never
interpolate between sampled frames — the frames in between were not
observed:

```python
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id, root=str(root))
item = ds[index]

mask = torch.isfinite(item["annotation.progress"])  # supervise only where True
item["annotation.progress"]  # task-completion fraction at this frame
item["annotation.holding"]  # 0.0 / 1.0: gripper physically holds the object
```

Unjudged episodes carry the columns too (all-NaN), so one code path
serves mixed corpora. The columns compose with `delta_timestamps` like
any other feature — e.g. progress now and one 50-step action chunk
ahead at 30 fps:

```python
ds = LeRobotDataset(
    repo_id,
    root=str(root),
    delta_timestamps={"annotation.progress": [0.0, 50 / 30]},
)
ds[index]["annotation.progress"]  # tensor([p_now, p_next]) — NaN where unsampled
```

## Visibility (per-frame, per-camera)

Two vectors per frame — is the task object / the gripper visible in
each camera — with one slot per camera. Slot order is the feature's
`names` (sorted camera short names); NaN-masked like the scalars.
Single-camera datasets store these as plain scalars (lerobot's feature
convention casts shape-(1,) features to scalar values, not length-1
lists — items yield a 0-d tensor there):

```python
names = ds.meta.features["annotation.visible_object"][
    "names"
]  # e.g. ['overhead', 'wrist']
item["annotation.visible_object"]  # tensor with one 0/1/NaN slot per camera
item["annotation.visible_gripper"]  # same layout for the gripper
```

## Subgoals and events (language columns)

**Subgoals** split an episode into temporal segments ("reach toward the
object", "place it on the target", …) and are piecewise-constant text
covering *every* frame of a judged episode: `language_persistent` rows
(`style="subtask"`) activate at their segment's first frame and persist
until superseded. This is also the form the online dataset visualizer's
Annotations tab renders. Because coverage is every-frame, subgoal
conditioning or prediction is not restricted to sampled frames.

**Events** (drops, resets, interventions, progress regressions) are
momentary rows in `language_events` (`style="event"`) stored on the
exact frame where they fired. **A frame may carry several events**
(measured on the full corpus sweep: 0.15% of sampled frames, 1.5% of
episodes) — read the item's rows directly and treat them as a set. Do
NOT resolve events through `lerobot.datasets.language_render.emitted_at`:
it is a single-row resolver and raises `ValueError: Ambiguous resolver`
on multi-event frames.

```python
from lerobot.datasets.language_render import active_at

t = float(item["timestamp"])
subgoal = active_at(t, persistent=item["language_persistent"], style="subtask")
subgoal["content"]  # non-None on every frame of a judged episode

events = [row["content"] for row in (item["language_events"] or [])]
# [] on frames without an event (the overwhelming majority)
```

Event negatives are only defined on judge-sampled frames: a sampled
frame (finite `annotation.progress`) with no event row is a true "no
event"; an unsampled frame is *unknown*, not negative.

## Invariants

- **Indexing**: judge-side frame numbers are 1-based inclusive; lerobot
  `frame_index` is 0-based. The materializer resolves this once —
  consumers of the columns and language rows never touch judge frame
  numbers. Sidecar-direct consumers read episode-level fields only,
  which carry none.
- **Pinning**: one `(model, prompt_hash)` per training run's labels;
  verify via `meta/judge_annotations.json` rather than re-deriving from
  sidecar records. Multiple judges' verdicts legitimately coexist in a
  sidecar — the stamp says which one the surfaces reflect.
- **Renumbering invalidates**: any episode-renumbering rewrite (dataset
  merges, episode deletion) must remap the sidecar and re-materialize;
  consume post-rewrite datasets only after that happened. The stamp's
  `written_at` postdates the rewrite when it did.
- **Weak supervision**: treat per-frame labels as noisy. Measured
  against a rig dataset's gripper-aperture channel as ground truth,
  `holding` agrees on ~75–85% of sampled frames with a systematic
  open-gripper-hover→true bias; on ~1.9k episodes judged independently
  by two models, holding agreement was 80.7% and progress MAE 0.15
  (r 0.73) over 18.7k paired frames. Mask losses, weight modestly, and
  re-measure against your own ground truth before trusting more.
