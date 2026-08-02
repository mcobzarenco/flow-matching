"""Batch-mode judging via the Anthropic Message Batches API (flat 50% off).

Same evidence, same prompt, same journal and sidecar semantics as the
synchronous sweep — only the transport differs: requests are uploaded in
bulk (≤ 100k requests / 256 MB per batch upstream; we chunk well under
both), processed server-side within 24 h (typically well under an hour),
and results are fetched and folded into the ordinary journal, so
``merge_journal`` and the idempotency keys need no batch awareness.

Two artifacts per run, both under the journal's directory:

- the journal itself (shared with sync mode) — one line per *finished*
  episode;
- a **manifest** (``<journal stem>.manifest.jsonl``) — one line per
  *submitted* episode, written immediately after each batch create call,
  carrying the batch id plus everything needed to validate that
  episode's result later (camera labels/names, sampled frames, counts).
  The manifest is what makes the flow resumable: a rerun folds pending
  batches it finds there before submitting anything new, so a crashed or
  Ctrl-C'd run never loses paid-for results and never double-submits.
  (A crash inside the create→manifest-append window would orphan one
  batch; the window is milliseconds, and orphans remain listable via the
  API for 29 days.)

Evidence building (video decode → JPEG payloads) runs in the same
spawn-based process pool as sync mode; decode failures journal as
``failed`` without any API spend, exactly like sync.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

from anthropic import Anthropic

from .claude import build_user_content
from .evidence import load_episode_summary
from .schema import PROMPT_HASH, SYSTEM_PROMPT
from .worker import JudgeTask, validated_judgment

# One batch stays well under the API's 256 MB / request-count caps: the
# byte budget dominates (10 timesteps x 4 cams ~ 3 MB of JPEG base64),
# and smaller batches give earlier partial results to fold.
MAX_BATCH_REQUESTS = 1_000
MAX_BATCH_BYTES = 180 * 1024 * 1024
POLL_SECONDS = 60.0


def manifest_path_for(journal: Path) -> Path:
    return journal.with_suffix(".manifest.jsonl")


def custom_id_for(repo_id: str, episode: int, model: str, prompt_hash: str) -> str:
    """Deterministic, charset-safe batch custom_id (the API caps it at 64
    chars and repo ids can exceed that; the manifest owns the mapping
    back)."""
    key = "\x00".join((repo_id, str(episode), model, prompt_hash))
    return hashlib.sha256(key.encode()).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class BatchEntry:
    """One submitted episode: batch identity + the evidence facts needed
    to validate its result without re-decoding the video."""

    custom_id: str
    batch_id: str
    repo_id: str
    episode: int
    model: str
    prompt_hash: str
    num_timesteps: int
    max_image_dim: int
    task: str
    fps: float
    duration_s: float
    num_frames: int
    camera_labels: list[str]
    camera_names: list[str]
    sampled_frames: list[int]

    def journal_key(self) -> tuple[str, int, str, str]:
        return (self.repo_id, self.episode, self.model, self.prompt_hash)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchEntry:
        try:
            return cls(
                custom_id=str(data["custom_id"]),
                batch_id=str(data["batch_id"]),
                repo_id=str(data["repo_id"]),
                episode=int(data["episode"]),
                model=str(data["model"]),
                prompt_hash=str(data["prompt_hash"]),
                num_timesteps=int(data["num_timesteps"]),
                max_image_dim=int(data["max_image_dim"]),
                task=str(data["task"]),
                fps=float(data["fps"]),
                duration_s=float(data["duration_s"]),
                num_frames=int(data["num_frames"]),
                camera_labels=[str(v) for v in data["camera_labels"]],
                camera_names=[str(v) for v in data["camera_names"]],
                sampled_frames=[int(v) for v in data["sampled_frames"]],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed manifest entry: {error}") from error


def load_manifest(path: Path) -> list[BatchEntry]:
    """Every submitted episode across all runs (corrupt manifest is fatal:
    treating it as empty would double-submit paid requests)."""
    if not path.exists():
        return []
    entries: list[BatchEntry] = []
    with path.open() as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                entries.append(BatchEntry.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as error:
                raise SystemExit(
                    f"corrupt batch manifest {path}:{number}: {error}; "
                    "fix or remove it",
                ) from error
    return entries


@dataclass(frozen=True, slots=True)
class BuiltRequest:
    """Evidence turned into an uploadable request (pool work-unit result).

    ``params`` is the Anthropic MessageCreateParams payload; ``meta``
    holds the BatchEntry fields except ``batch_id`` (unknown until
    submission). ``nbytes`` is the JSON-serialized request size, used for
    chunking against the API's batch byte cap.
    """

    custom_id: str
    params: dict[str, Any]
    nbytes: int
    meta: dict[str, Any]


def build_request(task: JudgeTask) -> BuiltRequest:
    """Decode evidence and build one batch request (runs in the pool)."""
    summary = load_episode_summary(
        root=Path(task.root),
        repo_id=task.repo_id,
        episode=task.episode,
        num_timesteps=task.num_timesteps,
        max_image_dim=task.max_image_dim,
    )
    content = build_user_content(summary)
    params: dict[str, Any] = {
        "model": task.model,
        "max_tokens": task.max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
    }
    custom_id = custom_id_for(task.repo_id, task.episode, task.model, PROMPT_HASH)
    return BuiltRequest(
        custom_id=custom_id,
        params=params,
        nbytes=len(json.dumps(params)),
        meta={
            "custom_id": custom_id,
            "repo_id": task.repo_id,
            "episode": task.episode,
            "model": task.model,
            "prompt_hash": PROMPT_HASH,
            "num_timesteps": task.num_timesteps,
            "max_image_dim": task.max_image_dim,
            "task": summary.task,
            "fps": summary.fps,
            "duration_s": round(summary.duration_s, 2),
            "num_frames": summary.num_frames,
            "camera_labels": summary.camera_labels,
            "camera_names": summary.camera_names,
            "sampled_frames": summary.sampled_frames,
        },
    )


def chunk_requests(
    built: list[BuiltRequest],
    *,
    max_requests: int = MAX_BATCH_REQUESTS,
    max_bytes: int = MAX_BATCH_BYTES,
) -> list[list[BuiltRequest]]:
    """Greedy order-preserving chunking under both API caps."""
    chunks: list[list[BuiltRequest]] = []
    current: list[BuiltRequest] = []
    current_bytes = 0
    for request in built:
        if request.nbytes > max_bytes:
            raise ValueError(
                f"episode {request.meta['repo_id']}/{request.meta['episode']} "
                f"serializes to {request.nbytes:,} bytes > {max_bytes:,} "
                "per-batch cap — lower --num-frames or --max-image-dim",
            )
        if current and (
            len(current) >= max_requests or current_bytes + request.nbytes > max_bytes
        ):
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(request)
        current_bytes += request.nbytes
    if current:
        chunks.append(current)
    return chunks


def fold_result(entry: BatchEntry, result: Any) -> dict[str, Any]:
    """One fetched batch result -> one journal record (ok or failed).

    ``result`` is the SDK's MessageBatchResult union (Any: third-party
    boundary); mirrors worker.judge_one's record shape so merge_journal
    and load_journal_done treat both transports identically.
    """
    record: dict[str, Any] = {
        "dataset": entry.repo_id,
        "episode": entry.episode,
        "time": time.strftime("%F %T", time.gmtime()),
        "model": entry.model,
        "prompt_hash": entry.prompt_hash,
        "batch_id": entry.batch_id,
    }
    if result.type != "succeeded":
        error = getattr(result, "error", None)
        detail = f": {error}" if error is not None else ""
        record.update(status="failed", error=f"batch {result.type}{detail}")
        return record
    message = result.message
    raw = "".join(block.text for block in message.content if block.type == "text")
    try:
        judgment = validated_judgment(
            raw,
            camera_labels=entry.camera_labels,
            camera_names=entry.camera_names,
            sampled_frames=entry.sampled_frames,
            num_frames=entry.num_frames,
        )
    except ValueError as error:
        record.update(status="failed", error=str(error))
        return record
    record.update(
        status="ok",
        task=entry.task,
        num_frames=entry.num_frames,
        duration_s=entry.duration_s,
        fps=entry.fps,
        cameras=entry.camera_names,
        num_timesteps=entry.num_timesteps,
        max_image_dim=entry.max_image_dim,
        judgment=judgment.to_dict(),
        usage={
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        },
    )
    return record


def _write_journal_line(journal: IO[str], record: dict[str, Any]) -> None:
    journal.write(json.dumps(record) + "\n")
    journal.flush()


def submit_tasks(
    client: Anthropic,
    tasks: list[JudgeTask],
    *,
    workers: int,
    manifest_path: Path,
    journal: IO[str],
    stats: FoldStats,
) -> list[BatchEntry]:
    """Build evidence in parallel, upload in chunks, extend the manifest.

    Returns submitted entries; evidence failures are journaled as failed
    here (no API spend), matching sync semantics.
    """
    built: list[BuiltRequest] = []
    # spawn (not fork): same rationale as the sync pool — video decoders
    # in forked children of a torch-loaded parent invite corruption.
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(build_request, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                built.append(future.result())
            except Exception as error:  # noqa: BLE001 - quarantine, keep building
                record = {
                    "dataset": task.repo_id,
                    "episode": task.episode,
                    "time": time.strftime("%F %T", time.gmtime()),
                    "model": task.model,
                    "prompt_hash": PROMPT_HASH,
                    "status": "failed",
                    "error": f"evidence: {type(error).__name__}: {error}",
                }
                _write_journal_line(journal, record)
                stats.count(record)
                print(
                    f"EVIDENCE FAILED {task.repo_id} ep {task.episode}: {error}",
                    file=sys.stderr,
                )
            if i % 25 == 0 or i == len(tasks):
                print(f"[build {i}/{len(tasks)}] payloads ready", flush=True)

    # Deterministic submission order (as_completed scrambles it).
    built.sort(key=lambda request: (request.meta["repo_id"], request.meta["episode"]))
    entries: list[BatchEntry] = []
    with manifest_path.open("a") as manifest:
        for chunk in chunk_requests(built):
            batch = client.messages.batches.create(
                requests=[
                    {"custom_id": request.custom_id, "params": request.params}  # type: ignore[typeddict-item]  # params is a plain dict; the SDK accepts it (MessageCreateParamsNonStreaming is a TypedDict)
                    for request in chunk
                ],
            )
            for request in chunk:
                entry = BatchEntry.from_dict({**request.meta, "batch_id": batch.id})
                manifest.write(json.dumps(asdict(entry)) + "\n")
                entries.append(entry)
            manifest.flush()
            megabytes = sum(request.nbytes for request in chunk) / 1024**2
            print(
                f"submitted batch {batch.id}: {len(chunk)} requests "
                f"({megabytes:.0f} MB)",
                flush=True,
            )
    return entries


@dataclass(slots=True)
class FoldStats:
    """Mutable tally across submit/poll phases (token counts feed the
    end-of-run spent estimate)."""

    ok: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def count(self, record: dict[str, Any]) -> None:
        if record["status"] == "ok":
            self.ok += 1
        else:
            self.failed += 1
        usage = record.get("usage")
        if usage:
            self.input_tokens += usage["input_tokens"]
            self.output_tokens += usage["output_tokens"]


def poll_and_fold(
    client: Anthropic,
    pending: list[BatchEntry],
    *,
    journal: IO[str],
    stats: FoldStats,
    poll_seconds: float = POLL_SECONDS,
) -> None:
    """Poll pending batches; fold each finished batch's results into the
    journal, tallying into ``stats``."""
    entries_by_batch: dict[str, dict[str, BatchEntry]] = {}
    for entry in pending:
        entries_by_batch.setdefault(entry.batch_id, {})[entry.custom_id] = entry
    while entries_by_batch:
        for batch_id in sorted(entries_by_batch):
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status != "ended":
                counts = batch.request_counts
                print(
                    f"batch {batch_id}: {batch.processing_status} "
                    f"({counts.processing} processing / {counts.succeeded} ok / "
                    f"{counts.errored} err)",
                    flush=True,
                )
                continue
            entries = entries_by_batch.pop(batch_id)
            folded = 0
            for response in client.messages.batches.results(batch_id):
                entry = entries.get(response.custom_id)
                if entry is None:
                    # Not ours: journal-done on a rerun, or another run
                    # sharing the batch. Fine either way — merge is keyed.
                    continue
                record = fold_result(entry, response.result)
                _write_journal_line(journal, record)
                stats.count(record)
                folded += 1
                if record["status"] == "failed":
                    print(
                        f"FAILED {record['dataset']} ep {record['episode']}: "
                        f"{record['error']}",
                        file=sys.stderr,
                    )
            print(
                f"batch {batch_id}: folded {folded}/{len(entries)} results "
                f"(totals ok={stats.ok} failed={stats.failed})",
                flush=True,
            )
        if entries_by_batch:
            time.sleep(poll_seconds)
