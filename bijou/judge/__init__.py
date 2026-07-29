"""LLM-as-judge episode curation for LeRobot v3.0 datasets.

Two judges emit the same strict-JSON verdict from sampled frames + the
task instruction + full-trajectory statistics (see docs/episode-judging.md
for the approach):

- ``bijou.judge.claude`` — Anthropic API judge (single episode CLI).
- ``bijou.judge.gemma`` — local Gemma 4 judge (single episode CLI; greedy,
  the reproducible-verdict path).
- ``bijou.judge.sweep`` — resumable parallel sweep over collection roots;
  verdicts land in each dataset's ``meta/judgments.json``.

This package root re-exports only the light, consumer-facing pieces (the
verdict schema and the sidecar store) — importing them must not pull the
anthropic or transformers stacks.
"""

from bijou.judge.schema import (
    PROMPT_HASH,
    SYSTEM_PROMPT,
    CameraKind,
    EpisodeJudgment,
    InstructionQuality,
    Scores,
    TaskCompletion,
    Verdict,
)
from bijou.judge.store import (
    JUDGMENTS_RELPATH,
    judgment_key,
    load_sidecar,
    sidecar_record,
    write_sidecar,
)

__all__ = [
    "JUDGMENTS_RELPATH",
    "PROMPT_HASH",
    "SYSTEM_PROMPT",
    "CameraKind",
    "EpisodeJudgment",
    "InstructionQuality",
    "Scores",
    "TaskCompletion",
    "Verdict",
    "judgment_key",
    "load_sidecar",
    "sidecar_record",
    "write_sidecar",
]
