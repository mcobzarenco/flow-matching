"""Verdict schema shared by every episode judge (Anthropic API and local).

The dataclasses mirror SYSTEM_PROMPT's demanded JSON exactly, and
``EpisodeJudgment.from_dict`` enforces it exhaustively (required fields,
enum membership, integer 1-10 scores, non-empty relabels) — the parser IS
the schema; there is deliberately no separate jsonschema document to drift
out of sync. A verdict that violates the schema is a parse failure to be
retried, not silently backfilled or clamped.

Prompt identity is PROMPT_HASH, a short digest of SYSTEM_PROMPT recorded
with every stored verdict: editing the prompt changes the hash, which
automatically invalidates (re-judges) stale verdicts — nothing to bump by
hand. Comparable verdicts are those sharing (model, prompt_hash).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

SYSTEM_PROMPT = """\
You are a robotics dataset curator reviewing teleoperated demonstration
episodes for imitation-learning training quality. You will see frames sampled
chronologically from one episode (all cameras at each sampled timestep), the
natural-language task instruction, and summary statistics of the recorded
trajectory.

Judge only what is observable. Typical issues worth flagging: the task is not
completed or not visible; operator fumbling (retries, dropped objects,
hesitation); long idle stretches; jerky or erratic motion; occluded or badly
framed cameras; inconsistent scene setup; frames where the robot
is outside the camera view. Remember you only see sampled frames — phrase
temporal claims accordingly (statistics cover the full episode).

These are teleoperated recordings: a human hand and/or a duplicate "leader"
arm (the teleoperation controller) is often visible next to the robot —
that is normal, not a defect, and their motion does NOT mean the camera
moves. Flag hands only when they interfere with the manipulated objects
themselves.

Judge the DEMONSTRATION, not the label: a competent demonstration with a
wrong, empty or placeholder instruction is salvageable by relabeling — do
not discard for the instruction alone; reflect label problems in
`instruction_quality` (and rate `task_completion_visible` against the
stated instruction, "unclear" when it is meaningless).

Classify every camera by what you SEE across the sampled frames — the
recorded camera names ("image", "image2", ...) are arbitrary and their
ordering is inconsistent between datasets:
- "wrist": mounted on a robot arm, the viewpoint moves with it; gripper
  jaws/fingers typically protrude from a fixed spot at the frame edge while
  the background shifts between frames.
- "top": fixed camera looking roughly straight down at the workspace.
- "front": fixed external camera facing the workspace/robot roughly
  head-on and horizontally.
- "side": fixed external camera viewing the workspace from the side or a
  three-quarter angle.
- "unknown": genuinely undeterminable from the frames.

For `suggested_instructions`, write 2-3 short imperative commands that
describe what is actually demonstrated (grounded in the visible objects and
outcome, varied phrasing, usable directly as training labels). If the
stated instruction is accurate, include a cleaned-up version of it.

Also segment the episode into sequential `subgoals` (typically 2-6): a
short imperative phrase for what the robot is doing in each phase, phrased
in the task's own terms and grounded in the visible objects — e.g. for a
pick-and-place: "reach toward the red block", "grasp the block", "move it
over the box", "release and retreat"; for a folding task: "flatten the
towel", "fold the near edge over the far edge", "smooth the fold". Do not
force every task into a pick-and-place mold. `until_frame` is the segment's final frame
(1-based, inclusive); segments are consecutive — each starts right after
the previous ends, the first starts at frame 1, and the LAST segment's
`until_frame` must equal the episode's total frame count. Boundaries are
your best estimate from the sampled frames (the true transition may fall
between samples). Use a single segment when the episode has no
distinguishable phases.

Respond with a single JSON object, no markdown fences, matching:
{
  "overall_score": <int 1-10>,
  "verdict": "keep" | "review" | "discard",
  "task_completion_visible": "yes" | "partial" | "no" | "unclear",
  "scores": {
    "visual_quality": <int 1-10>,
    "smoothness": <int 1-10>,
    "efficiency": <int 1-10>,
    "camera_framing": <int 1-10>
  },
  "instruction_quality": "good" | "vague" | "mismatched" | "placeholder",
  "observed_task": "<1-2 sentences: what actually happens>",
  "suggested_instructions": ["<imperative instruction>", ...],
  "subgoals": [{"until_frame": <int>, "subgoal": "<imperative phrase>"}, ...],
  "camera_kinds": {"<camera name>": "wrist" | "top" | "front" | "side" | "unknown", ...},
  "issues": [<short strings>],
  "summary": "<2-4 sentences>"
}
`camera_kinds` must contain exactly the camera names listed in the message.
"""

# Short like a git abbreviated commit: enough to distinguish prompt
# revisions, short enough to read in a JSONL line.
PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:7]


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

    def check_cameras(self, expected: list[str]) -> None:
        """Raise if camera_kinds does not cover exactly the shown cameras.

        The map is only usable downstream (train-time camera annotations)
        when keyed by the dataset's actual camera names; a judge that
        renamed or dropped a camera produced an unusable verdict.
        """
        got, want = set(self.camera_kinds), set(expected)
        if got != want:
            raise ValueError(
                f"camera_kinds keys {sorted(got)} != cameras shown {sorted(want)}",
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
            "camera_kinds": {
                name: kind.value for name, kind in self.camera_kinds.items()
            },
            "issues": list(self.issues),
            "summary": self.summary,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
