"""One-episode judge work unit for the sweep's process pool.

Lives outside sweep.py so spawn-based workers can unpickle JudgeTask
regardless of how the sweep was launched (see the bijou.eval.__main__
precedent: definitions must not live in a module that runs as __main__).
Process isolation is the point: an AV1 decoder crash on a corrupt
community video fails one episode, not the sweep.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic, APIError

from bijou.judge.claude import build_user_content, request_verdict
from bijou.judge.evidence import load_episode_summary
from bijou.judge.schema import PROMPT_HASH, EpisodeJudgment

_client_cache: Anthropic | None = None


def _client() -> Anthropic:
    global _client_cache  # noqa: PLW0603 - one API client per worker process
    if _client_cache is None:
        _client_cache = Anthropic(max_retries=5)  # SDK backoff handles 429/529
    return _client_cache


@dataclass(frozen=True, slots=True)
class JudgeTask:
    """One episode to judge (picklable work unit for the process pool)."""

    root: str  # dataset directory
    repo_id: str
    episode: int
    num_timesteps: int
    max_image_dim: int
    model: str
    max_tokens: int


def judge_one(task: JudgeTask) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "dataset": task.repo_id,
        "episode": task.episode,
        "time": time.strftime("%F %T", time.gmtime()),
        "model": task.model,
        "prompt_hash": PROMPT_HASH,
    }
    try:
        summary = load_episode_summary(
            root=Path(task.root),
            repo_id=task.repo_id,
            episode=task.episode,
            num_timesteps=task.num_timesteps,
            max_image_dim=task.max_image_dim,
        )
        content = build_user_content(summary)
        raw, usage = request_verdict(_client(), task.model, task.max_tokens, content)
        judgment = EpisodeJudgment.from_response_text(raw)
        judgment.check_cameras(summary.camera_names)
        record.update(
            status="ok",
            task=summary.task,
            num_frames=summary.num_frames,
            duration_s=round(summary.duration_s, 2),
            fps=summary.fps,
            cameras=summary.camera_names,
            num_timesteps=task.num_timesteps,
            max_image_dim=task.max_image_dim,
            judgment=judgment.to_dict(),
            usage=usage,
        )
    except APIError as error:
        record.update(status="failed", error=f"api: {error}")
    except Exception as error:  # noqa: BLE001 - quarantine and continue the sweep
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    record["seconds"] = round(time.perf_counter() - started, 2)
    return record
