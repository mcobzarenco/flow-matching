"""The judge-annotation ARTIFACT contract (leaf module).

The shapes of what the judging pipeline writes to disk and training
reads back: the verdict schema (the parser IS the schema — see
EpisodeJudgment.from_dict), the sidecar record envelope + I/O, the
semantic camera-kind vocabulary, and the project-local lerobot "event"
language style (registered here, at the one module both the writer
— bijou.judge — and the readers — data/aux_text/train — sit above).

Production concerns stay in bijou.judge: SYSTEM_PROMPT and PROMPT_HASH
(how verdicts are made and identified) never moved — consumers compare
hashes found IN artifacts (stamps vs records) and need no knowledge of
the live prompt. bijou.judge.schema/store re-export the moved names, so
judge-side code is unaffected.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from lerobot.datasets.language import (
    EVENT_ONLY_STYLES,
    EXTENDED_STYLES,
    STYLE_REGISTRY,
)

# The project-local lerobot language style event rows are stored under
# (registered via lerobot's documented import-time hook; idempotent
# set-adds, so writer- and reader-side imports coexist).
EVENT_STYLE = "event"
EXTENDED_STYLES.add(EVENT_STYLE)
EVENT_ONLY_STYLES.add(EVENT_STYLE)
STYLE_REGISTRY.add(EVENT_STYLE)

# The per-dataset sidecar the judging pipeline writes verdicts to.
JUDGMENTS_RELPATH = Path("meta") / "judgments.json"

# Progress may wobble by re-estimation between sampled frames; dips beyond
# this require an explaining event (the prompt demands one for ANY dip —
# the tolerance only keeps jitter from failing otherwise-valid verdicts).
PROGRESS_DIP_TOLERANCE = 0.05


class Verdict(StrEnum):
    """Curation decision for an episode."""

    KEEP = "keep"
    REVIEW = "review"
    DISCARD = "discard"


class TaskCompletion(StrEnum):
    """Whether task completion is observable in the sampled frames."""

    YES = "yes"
    PARTIAL = "partial"
    NO = "no"
    UNCLEAR = "unclear"


class InstructionQuality(StrEnum):
    """How well the stored task string describes the demonstration.

    Community task strings are frequently junk ("test1", "Test Boulon") on
    top of perfectly usable demonstrations; this axis is deliberately
    separate from the quality verdict so good demos with bad labels can be
    relabeled instead of discarded.
    """

    GOOD = "good"  # specific and matches what the episode shows
    VAGUE = "vague"  # generic but compatible ("pick up the object")
    MISMATCHED = "mismatched"  # describes something visibly different
    PLACEHOLDER = "placeholder"  # empty/meaningless ("test", "task1", ...)


class CameraKind(StrEnum):
    """Visually judged camera mount/viewpoint category.

    The converted community collections use anonymized camera names
    ("image", "image2", ...) whose ordering is inconsistent across datasets
    (measured: 99.9% of 1,242 datasets), so viewpoint semantics can only
    come from looking at the frames. UNKNOWN is the honest fallback and is
    also useful as a train-time dropout target for camera annotations.
    """

    WRIST = "wrist"
    TOP = "top"
    FRONT = "front"
    SIDE = "side"
    UNKNOWN = "unknown"


# The kind vocabulary as strings — the form the prompt renderers and
# name-derivation helpers consume (single source: the enum above).
CAMERA_KINDS = frozenset(kind.value for kind in CameraKind)


def _score_1_10(data: dict[str, Any], field: str) -> int:
    """1-10 integer score, strictly — what a jsonschema 'integer' + bounds
    would check, without a second schema document to keep in sync.

    Bare int() coercion lets true -> 1, "7" -> 7 and 7.9 -> 7 slide through
    silently; a silently-wrong score poisons downstream aggregation, which
    is worse than a loud parse failure (the sweep records those for
    --retry-failed). Integer-valued floats (7.0) are accepted — JSON Schema
    itself treats them as integers.
    """
    value = data[field]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be an integer, got {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field} must be an integer, got {value!r}")
        value = int(value)
    if not 1 <= value <= 10:
        raise ValueError(f"{field} must be in 1..10, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class Scores:
    """Per-aspect quality sub-scores on a 1-10 scale."""

    visual_quality: int
    smoothness: int
    efficiency: int
    camera_framing: int


def _strict_bool(data: dict[str, Any], field: str) -> bool:
    value = data[field]
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class CameraVisibility:
    """What one camera can see at one sampled frame."""

    task_object: bool
    gripper: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraVisibility:
        return cls(
            task_object=_strict_bool(data, "task_object"),
            gripper=_strict_bool(data, "gripper"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"task_object": self.task_object, "gripper": self.gripper}


@dataclass(frozen=True, slots=True)
class FrameAnnotation:
    """Dense annotations for ONE sampled frame — valid at that frame only.

    Deliberately never interpolated (unlike subgoals): the judge saw this
    exact frame; the frames in between were not observed. Consumers train
    on annotated frames with a mask, or densify with a tracker — never
    lerp.
    """

    frame: int  # 1-based frame number, one of the sampled timesteps
    progress: float  # fraction of the task completed, in [0, 1]
    holding: bool  # gripper physically holds the task object
    visible: dict[str, CameraVisibility]  # keyed by short camera name
    events: tuple[str, ...]  # unusual occurrences at this frame

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameAnnotation:
        frame = data["frame"]
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 1:
            raise ValueError(f"frame must be a positive integer, got {frame!r}")
        progress = data["progress"]
        if isinstance(progress, bool) or not isinstance(progress, int | float):
            raise TypeError(f"progress must be a number, got {progress!r}")
        if not 0.0 <= progress <= 1.0:
            raise ValueError(f"progress must be in [0, 1], got {progress}")
        visible_raw = data["visible"]
        if not isinstance(visible_raw, dict) or not visible_raw:
            raise ValueError("visible must be a non-empty object")
        events_raw = data["events"]
        if not isinstance(events_raw, list):
            raise TypeError(
                f"events must be an array, got {type(events_raw).__name__}",
            )
        events = tuple(str(event).strip() for event in events_raw)
        if not all(events):
            raise ValueError(f"events contains empty entries: {events_raw!r}")
        return cls(
            frame=frame,
            progress=float(progress),
            holding=_strict_bool(data, "holding"),
            visible={
                str(camera): CameraVisibility.from_dict(vis)
                for camera, vis in visible_raw.items()
            },
            events=events,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "progress": self.progress,
            "holding": self.holding,
            "visible": {camera: vis.to_dict() for camera, vis in self.visible.items()},
            "events": list(self.events),
        }


@dataclass(frozen=True, slots=True)
class Subgoal:
    """One temporal segment of an episode.

    Frames (previous segment's ``until_frame``, this ``until_frame``] in
    1-based numbering carry ``subgoal``; the first segment starts at
    frame 1. Boundaries are judge estimates quantized to the sampled
    timesteps — every frame in between inherits its segment's label
    (piecewise-constant interpolation).
    """

    until_frame: int
    subgoal: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Subgoal:
        until_frame = data["until_frame"]
        if isinstance(until_frame, bool) or not isinstance(until_frame, int):
            raise TypeError(f"until_frame must be an integer, got {until_frame!r}")
        if until_frame < 1:
            raise ValueError(f"until_frame must be >= 1, got {until_frame}")
        subgoal = str(data["subgoal"]).strip()
        if not subgoal:
            raise ValueError("subgoal must be a non-empty string")
        return cls(until_frame=until_frame, subgoal=subgoal)

    def to_dict(self) -> dict[str, Any]:
        return {"until_frame": self.until_frame, "subgoal": self.subgoal}


@dataclass(frozen=True, slots=True)
class EpisodeJudgment:
    """Structured verdict returned by a judge model.

    Use ``from_response_text`` for raw model output (tolerates surrounding
    prose or markdown fences) and ``to_json``/``from_json`` for strict
    round-trips.
    """

    overall_score: int
    verdict: Verdict
    task_completion_visible: TaskCompletion
    scores: Scores
    instruction_quality: InstructionQuality
    observed_task: str
    suggested_instructions: tuple[str, ...]
    subgoals: tuple[Subgoal, ...]
    frame_annotations: tuple[FrameAnnotation, ...]
    camera_kinds: dict[str, CameraKind]
    issues: tuple[str, ...]
    summary: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeJudgment:
        try:
            scores = data["scores"]
            if not isinstance(scores, dict):
                raise TypeError(
                    f"scores must be an object, got {type(scores).__name__}",
                )
            camera_kinds = data["camera_kinds"]
            if not isinstance(camera_kinds, dict) or not camera_kinds:
                raise ValueError("camera_kinds must be a non-empty object")
            observed_task = str(data["observed_task"]).strip()
            if not observed_task:
                raise ValueError("observed_task must be a non-empty string")
            suggested = data["suggested_instructions"]
            if not isinstance(suggested, list) or not suggested:
                raise ValueError("suggested_instructions must be a non-empty array")
            instructions = tuple(str(entry).strip() for entry in suggested)
            if not all(instructions):
                raise ValueError(
                    f"suggested_instructions contains empty entries: {suggested!r}",
                )
            subgoals_raw = data["subgoals"]
            if not isinstance(subgoals_raw, list) or not subgoals_raw:
                raise ValueError("subgoals must be a non-empty array")
            subgoals = tuple(Subgoal.from_dict(entry) for entry in subgoals_raw)
            boundaries = [segment.until_frame for segment in subgoals]
            if boundaries != sorted(set(boundaries)):
                raise ValueError(
                    f"subgoal until_frame values must be strictly increasing: {boundaries}",
                )
            annotations_raw = data["frame_annotations"]
            if not isinstance(annotations_raw, list) or not annotations_raw:
                raise ValueError("frame_annotations must be a non-empty array")
            annotations = tuple(
                FrameAnnotation.from_dict(entry) for entry in annotations_raw
            )
            frames = [annotation.frame for annotation in annotations]
            if frames != sorted(set(frames)):
                raise ValueError(
                    f"frame_annotations frames must be strictly increasing: {frames}",
                )
            for previous, current in itertools.pairwise(annotations):
                dip = previous.progress - current.progress
                if dip > PROGRESS_DIP_TOLERANCE and not current.events:
                    raise ValueError(
                        f"progress regresses {previous.progress:.2f} -> "
                        f"{current.progress:.2f} at frame {current.frame} without "
                        "an explaining event",
                    )
            return cls(
                overall_score=_score_1_10(data, "overall_score"),
                verdict=Verdict(data["verdict"]),
                task_completion_visible=TaskCompletion(data["task_completion_visible"]),
                scores=Scores(
                    visual_quality=_score_1_10(scores, "visual_quality"),
                    smoothness=_score_1_10(scores, "smoothness"),
                    efficiency=_score_1_10(scores, "efficiency"),
                    camera_framing=_score_1_10(scores, "camera_framing"),
                ),
                instruction_quality=InstructionQuality(data["instruction_quality"]),
                observed_task=observed_task,
                suggested_instructions=instructions,
                subgoals=subgoals,
                frame_annotations=annotations,
                camera_kinds={
                    str(name): CameraKind(kind) for name, kind in camera_kinds.items()
                },
                issues=tuple(str(issue) for issue in data.get("issues", [])),
                summary=str(data.get("summary", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed judge verdict: {error}") from error

    @classmethod
    def from_json(cls, text: str) -> EpisodeJudgment:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_response_text(cls, text: str) -> EpisodeJudgment:
        """Parse raw model output by extracting the outermost JSON object."""
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in response")
        return cls.from_json(text[start : end + 1])

    def rename_cameras(self, mapping: dict[str, str]) -> EpisodeJudgment:
        """Translate camera keys throughout (anonymous labels -> dataset
        names). Judges answer under anonymous labels so recorded names
        cannot bias the viewpoint call; storage and downstream consumers
        want the dataset's real names. Strict: every camera key in the
        verdict must appear in ``mapping``."""
        seen: set[str] = set(self.camera_kinds)
        for annotation in self.frame_annotations:
            seen |= set(annotation.visible)
        unknown = seen - set(mapping)
        if unknown:
            raise ValueError(f"camera keys {sorted(unknown)} not in {sorted(mapping)}")
        return replace(
            self,
            camera_kinds={
                mapping[label]: kind for label, kind in self.camera_kinds.items()
            },
            frame_annotations=tuple(
                replace(
                    annotation,
                    visible={
                        mapping[label]: vis for label, vis in annotation.visible.items()
                    },
                )
                for annotation in self.frame_annotations
            ),
        )

    def check_cameras(self, expected: list[str]) -> None:
        """Raise if camera_kinds does not cover exactly the shown cameras
        (anonymous labels at parse time, dataset names after
        ``rename_cameras``) — a judge that renamed or dropped a camera
        produced an unusable verdict.
        """
        got, want = set(self.camera_kinds), set(expected)
        if got != want:
            raise ValueError(
                f"camera_kinds keys {sorted(got)} != cameras shown {sorted(want)}",
            )

    def check_frame_annotations(
        self,
        sampled_frames: list[int],
        cameras: list[str],
    ) -> None:
        """Raise unless dense annotations cover exactly the sampled frames,
        each with visibility for exactly the shown cameras — the judge
        annotating frames it never saw (or cameras it renamed) produced an
        unusable verdict."""
        got = [annotation.frame for annotation in self.frame_annotations]
        if got != list(sampled_frames):
            raise ValueError(
                f"frame_annotations frames {got} != sampled frames {list(sampled_frames)}",
            )
        want = set(cameras)
        for annotation in self.frame_annotations:
            if set(annotation.visible) != want:
                raise ValueError(
                    f"frame {annotation.frame} visibility covers "
                    f"{sorted(annotation.visible)} != cameras shown {sorted(want)}",
                )

    def check_subgoals(self, num_frames: int) -> None:
        """Raise unless the segments cover exactly frames 1..num_frames —
        anything else breaks the per-frame lookup downstream."""
        last = self.subgoals[-1].until_frame
        if last != num_frames:
            raise ValueError(
                f"subgoals end at frame {last}, episode has {num_frames} frames",
            )

    def subgoal_at(self, frame: int) -> str:
        """1-based frame number -> its segment's subgoal (every frame
        between annotated boundaries inherits the segment label)."""
        for segment in self.subgoals:
            if frame <= segment.until_frame:
                return segment.subgoal
        raise ValueError(
            f"frame {frame} beyond the last subgoal boundary "
            f"({self.subgoals[-1].until_frame})",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "verdict": self.verdict.value,
            "task_completion_visible": self.task_completion_visible.value,
            "scores": {
                "visual_quality": self.scores.visual_quality,
                "smoothness": self.scores.smoothness,
                "efficiency": self.scores.efficiency,
                "camera_framing": self.scores.camera_framing,
            },
            "instruction_quality": self.instruction_quality.value,
            "observed_task": self.observed_task,
            "suggested_instructions": list(self.suggested_instructions),
            "subgoals": [segment.to_dict() for segment in self.subgoals],
            "frame_annotations": [
                annotation.to_dict() for annotation in self.frame_annotations
            ],
            "camera_kinds": {
                name: kind.value for name, kind in self.camera_kinds.items()
            },
            "issues": list(self.issues),
            "summary": self.summary,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True, slots=True)
class JudgmentRecord:
    """One stored verdict: identity, evidence provenance, and the judgment.

    ``key()`` — (episode_index, model, prompt_hash, num_timesteps,
    max_image_dim) — is what makes verdicts comparable and sweeps
    idempotent. The prompt hash is content-derived (schema.PROMPT_HASH),
    so editing the prompt re-judges automatically and nothing is bumped
    by hand; the two evidence fields identify the image selection (the
    RESOLVED timestep count — fixed and adaptive runs that sample the
    same frames share a key) and resolution, so changing what the judge
    sees re-judges just as deliberately.
    """

    episode_index: int
    model: str
    prompt_hash: str
    judged_at: str  # UTC, "%F %T"
    num_timesteps: int  # sampled timesteps the judge saw
    max_image_dim: int  # px, longer side after downscaling
    usage: dict[str, int]  # input/output token counts
    # Verbatim verdict payload. Deliberately NOT parsed at load: a sidecar
    # legitimately mixes prompt versions (cascades, calibration, schema
    # evolution), and each payload obeys ITS OWN prompt's schema — parsing
    # everything through the current one would brick loading the moment
    # the schema grows a field. Consumers call ``parsed_judgment()`` on
    # the records whose prompt_hash matches the schema they understand.
    judgment: dict[str, Any]

    def key(self) -> tuple[int, str, str, int, int]:
        return (
            self.episode_index,
            self.model,
            self.prompt_hash,
            self.num_timesteps,
            self.max_image_dim,
        )

    def parsed_judgment(self) -> EpisodeJudgment:
        """Validate + parse the payload under the CURRENT schema — call on
        records whose ``prompt_hash`` matches the running code's
        ``schema.PROMPT_HASH`` (older payloads raise, by design)."""
        return EpisodeJudgment.from_dict(self.judgment)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgmentRecord:
        """Parse + validate the record envelope (the payload validates at
        consumption — see ``judgment``/``parsed_judgment``)."""
        try:
            judgment = data["judgment"]
            if not isinstance(judgment, dict):
                raise TypeError(
                    f"judgment must be an object, got {type(judgment).__name__}",
                )
            return cls(
                episode_index=int(data["episode_index"]),
                model=str(data["model"]),
                prompt_hash=str(data["prompt_hash"]),
                judged_at=str(data["judged_at"]),
                num_timesteps=int(data["num_timesteps"]),
                max_image_dim=int(data["max_image_dim"]),
                usage={str(k): int(v) for k, v in data["usage"].items()},
                judgment=judgment,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed judgment record: {error}") from error

    @classmethod
    def from_journal(cls, record: dict[str, Any]) -> JudgmentRecord:
        """Build from a sweep journal ok-line (field names differ: the
        journal says ``episode``/``time``). Episode length, fps, task etc.
        stay journal-only — they are recoverable from the dataset metadata
        itself."""
        return cls.from_dict(
            {
                "episode_index": record["episode"],
                "judged_at": record["time"],
                **record,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "judged_at": self.judged_at,
            "num_timesteps": self.num_timesteps,
            "max_image_dim": self.max_image_dim,
            "usage": dict(self.usage),
            "judgment": dict(self.judgment),
        }


def load_sidecar(dataset_dir: Path) -> list[JudgmentRecord]:
    """Parsed, validated records from a dataset's sidecar ([] when absent).

    A corrupt sidecar is fatal, not empty: treating it as empty would
    re-judge and then overwrite whatever the file still holds.
    """
    path = dataset_dir / JUDGMENTS_RELPATH
    if not path.exists():
        return []
    try:
        judgments = json.loads(path.read_text())["judgments"]
        if not isinstance(judgments, list):
            raise TypeError(
                f"'judgments' must be an array, got {type(judgments).__name__}",
            )
        return [JudgmentRecord.from_dict(record) for record in judgments]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(
            f"corrupt judgments sidecar {path}: {error}; fix or remove it",
        ) from error


def write_sidecar(dataset_dir: Path, records: list[JudgmentRecord]) -> None:
    """Atomically (re)write a dataset's sidecar, sorted for stable diffs.

    NOTE for the future curation rewrite: lerobot's delete_episodes
    renumbers episode_index — any dataset rewrite must remap (or
    deliberately drop) this file, it does not survive renumbering by itself.
    """
    path = dataset_dir / JUDGMENTS_RELPATH
    ordered = sorted(records, key=JudgmentRecord.key)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"judgments": [r.to_dict() for r in ordered]}, indent=1))
    tmp.replace(path)
