"""Pure-CPU tests for batch-mode plumbing: chunking, manifest round-trip,
and result folding (the SDK result union is stubbed — no API)."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

from bijou.judge.batch import (
    BatchEntry,
    BuiltRequest,
    chunk_requests,
    custom_id_for,
    fold_result,
)
from bijou.judge.schema import PROMPT_HASH


def built(repo_id: str, episode: int, nbytes: int) -> BuiltRequest:
    return BuiltRequest(
        custom_id=custom_id_for(repo_id, episode, "m", PROMPT_HASH, 10, 512),
        params={},
        nbytes=nbytes,
        meta={"repo_id": repo_id, "episode": episode},
    )


def entry(**overrides: object) -> BatchEntry:
    fields: dict[str, object] = {
        "custom_id": custom_id_for("u/d", 3, "claude-opus-4-8", PROMPT_HASH, 2, 512),
        "batch_id": "msgbatch_test",
        "repo_id": "u/d",
        "episode": 3,
        "model": "claude-opus-4-8",
        "prompt_hash": PROMPT_HASH,
        "num_timesteps": 2,
        "max_image_dim": 512,
        "task": "pick the thing",
        "fps": 30.0,
        "duration_s": 2.0,
        "num_frames": 60,
        "camera_labels": ["A"],
        "camera_names": ["front"],
        "sampled_frames": [1, 60],
    }
    fields.update(overrides)
    return BatchEntry.from_dict(fields)


def test_custom_id_deterministic_distinct_and_safe() -> None:
    a = custom_id_for("u/d", 0, "m", "h", 10, 512)
    assert a == custom_id_for("u/d", 0, "m", "h", 10, 512)
    assert a != custom_id_for("u/d", 1, "m", "h", 10, 512)
    assert a != custom_id_for("u/d", 0, "m2", "h", 10, 512)
    # Evidence identity is part of the key: different image selection
    # (timestep count) or resolution re-judges deliberately.
    assert a != custom_id_for("u/d", 0, "m", "h", 15, 512)
    assert a != custom_id_for("u/d", 0, "m", "h", 10, 768)
    assert len(a) == 32 and a.isalnum()


def test_journal_key_carries_evidence_identity() -> None:
    assert entry().journal_key() == (
        "u/d",
        3,
        "claude-opus-4-8",
        PROMPT_HASH,
        2,
        512,
    )
    assert entry(num_timesteps=15).journal_key() != entry().journal_key()


def test_chunking_respects_count_and_bytes_in_order() -> None:
    requests = [built("u/d", i, nbytes=10) for i in range(5)]
    by_count = chunk_requests(requests, max_requests=2, max_bytes=10_000)
    assert [len(chunk) for chunk in by_count] == [2, 2, 1]
    flat = [request.meta["episode"] for chunk in by_count for request in chunk]
    assert flat == [0, 1, 2, 3, 4]

    by_bytes = chunk_requests(requests, max_requests=100, max_bytes=25)
    assert [len(chunk) for chunk in by_bytes] == [2, 2, 1]


def test_manifest_entry_json_round_trip() -> None:
    original = entry()
    assert BatchEntry.from_dict(json.loads(json.dumps(asdict(original)))) == original


def test_fold_result_errored_and_expired() -> None:
    for kind in ("errored", "expired"):
        result = SimpleNamespace(
            type=kind,
            error="overloaded" if kind == "errored" else None,
        )
        record = fold_result(entry(), result)
        assert record["status"] == "failed"
        assert kind in record["error"]
        assert record["dataset"] == "u/d" and record["episode"] == 3
        assert record["prompt_hash"] == PROMPT_HASH


def test_fold_result_succeeded_validates_and_renames_cameras() -> None:
    judgment_json = {
        "overall_score": 7,
        "verdict": "keep",
        "task_completion_visible": "yes",
        "scores": {
            "visual_quality": 7,
            "smoothness": 7,
            "efficiency": 7,
            "camera_framing": 7,
        },
        "instruction_quality": "good",
        "observed_task": "picks the thing",
        "suggested_instructions": ["pick the thing up"],
        "subgoals": [{"until_frame": 60, "subgoal": "do it"}],
        "frame_annotations": [
            {
                "frame": frame,
                "progress": progress,
                "holding": False,
                "visible": {"A": {"task_object": True, "gripper": True}},
                "events": [],
            }
            for frame, progress in ((1, 0.0), (60, 1.0))
        ],
        "camera_kinds": {"A": "front"},
        "issues": [],
        "summary": "fine",
    }
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(judgment_json))],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )
    record = fold_result(entry(), SimpleNamespace(type="succeeded", message=message))
    assert record["status"] == "ok"
    assert record["usage"] == {"input_tokens": 100, "output_tokens": 50}
    # Anonymous label "A" translated to the dataset camera name.
    assert record["judgment"]["camera_kinds"] == {"front": "front"}
    assert record["cameras"] == ["front"]

    # A verdict violating the schema (subgoals not covering the episode)
    # folds as failed, exactly like sync mode.
    bad = dict(judgment_json, subgoals=[{"until_frame": 30, "subgoal": "half"}])
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(bad))],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    record = fold_result(entry(), SimpleNamespace(type="succeeded", message=message))
    assert record["status"] == "failed"
    assert "subgoals end at frame 30" in record["error"]
