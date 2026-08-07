"""Eval aggregation surfaces: per-dataset slices, shard merging, report
rendering.

Pure CPU/synthetic. Covers the per-dataset breakdown added for the
offline report (slice_by_dataset ordering + exact aggregation), the
multi-GPU shard merge (index-sorted, world-size-invariant), and that
render_report emits the collapsible block without touching real model
outputs (matplotlib Agg, tiny tensors).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bijou.eval.metrics import FrameScore, score_frame, slice_by_dataset
from bijou.eval.report import THEMES, ReportSample, ReportTable, render_report
from bijou.eval.sharding import ShardResults, merge_shards


def _score(index: int, repo_id: str, abs_error: float) -> FrameScore:
    # 2 valid steps x 3 motors, error spread evenly: chunk_mae == abs_error.
    return FrameScore(
        index=index,
        repo_id=repo_id,
        n_valid=2,
        abs_error_sum=abs_error * 6,
        squared_error_sum=abs_error**2 * 6,
        first_mae=abs_error,
        per_motor_abs_sum=(abs_error * 2,) * 3,
        inference_seconds=0.01,
    )


def test_slice_by_dataset_orders_and_aggregates() -> None:
    scores = {
        "copy": [
            _score(0, "user/b", 2.0),
            _score(1, "user/a", 4.0),
            _score(2, "user/b", 6.0),
            _score(3, "user/c", 1.0),
        ],
        "bijou": [
            _score(0, "user/b", 1.0),
            _score(1, "user/a", 2.0),
            _score(2, "user/b", 3.0),
            _score(3, "user/c", 0.5),
        ],
    }
    per_dataset = slice_by_dataset(scores)
    # Frame count descending, repo_id breaking the a/c tie.
    assert list(per_dataset) == ["user/b", "user/a", "user/c"]
    assert per_dataset["user/b"].frames == 2
    # Exact valid-weighted means, per policy.
    assert per_dataset["user/b"].chunk_mae == {"copy": 4.0, "bijou": 2.0}
    assert per_dataset["user/a"].chunk_mae == {"copy": 4.0, "bijou": 2.0}
    assert per_dataset["user/c"].chunk_mae == {"copy": 1.0, "bijou": 0.5}


def _shard(
    frame_ids: list[int],
    repo_id: str,
    outcome: str | None,
) -> ShardResults:
    scores = {
        name: [_score(i, repo_id, float(i)) for i in frame_ids]
        for name in ("copy", "bijou")
    }
    return ShardResults(
        scores=scores,
        outcomes=dict.fromkeys(frame_ids, outcome),
        holding_labels={frame_ids[0]: 1.0},
        progress_labels={frame_ids[0]: 0.5},
        event_labels={frame_ids[0]: "none"},
        visible_labels={frame_ids[0]: "object 0; gripper none"},
        sensitivity_deltas=[float(len(frame_ids))],
        report_samples={},
        generations={},
        subgoal_records={},
        dump_predictions={"bijou": [torch.full((2, 3), float(i)) for i in frame_ids]},
        dump_truth=[torch.zeros(2, 3) for _ in frame_ids],
        dump_valid=[torch.ones(2, dtype=torch.bool) for _ in frame_ids],
        dump_repo=[repo_id for _ in frame_ids],
        dump_index=list(frame_ids),
        # Synthetic identity: episode = frame id, frame-in-episode = 10x.
        dump_episode=list(frame_ids),
        dump_frame=[10 * i for i in frame_ids],
        # Per-draw stacks: [draws=4, chunk=2, dim=3], filled with the id.
        dump_draws=[torch.full((4, 2, 3), float(i)) for i in frame_ids],
    )


def test_merge_shards_sorts_and_is_world_size_invariant() -> None:
    # Round-robin sharding of sorted indices [0, 3, 5, 8]: rank 0 gets
    # 0, 5; rank 1 gets 3, 8.
    rank0 = _shard([0, 5], "user/a", "success")
    rank1 = _shard([3, 8], "user/b", None)
    merged = merge_shards([rank0, rank1])

    for frame_scores in merged.scores.values():
        assert [s.index for s in frame_scores] == [0, 3, 5, 8]
    # Paired-comparison contract: every policy covers the same frames in
    # the same order.
    assert [s.index for s in merged.scores["copy"]] == [
        s.index for s in merged.scores["bijou"]
    ]
    assert merged.outcomes == {0: "success", 5: "success", 3: None, 8: None}
    assert merged.holding_labels == {0: 1.0, 3: 1.0}
    assert merged.event_labels == {0: "none", 3: "none"}
    assert merged.visible_labels == {
        0: "object 0; gripper none",
        3: "object 0; gripper none",
    }
    assert sorted(merged.sensitivity_deltas) == [2.0, 2.0]
    # Dump rows follow the same global index order, all fields aligned.
    assert merged.dump_index == [0, 3, 5, 8]
    assert merged.dump_repo == ["user/a", "user/b", "user/a", "user/b"]
    assert merged.dump_episode == [0, 3, 5, 8]
    assert merged.dump_frame == [0, 30, 50, 80]
    assert [float(t[0, 0]) for t in merged.dump_predictions["bijou"]] == [
        0.0,
        3.0,
        5.0,
        8.0,
    ]
    assert [float(t[0, 0, 0]) for t in merged.dump_draws] == [0.0, 3.0, 5.0, 8.0]

    # Gather order must not matter (all_gather_object order is rank
    # order, but nothing downstream may depend on it).
    flipped = merge_shards([rank1, rank0])
    assert [s.index for s in flipped.scores["copy"]] == [0, 3, 5, 8]
    assert flipped.dump_repo == merged.dump_repo

    # World size 1 (the single-process path) gives the same view.
    single = merge_shards([_shard([0, 3, 5, 8], "user/a", "success")])
    assert [s.index for s in single.scores["copy"]] == [0, 3, 5, 8]


def test_merge_shards_tolerates_empty_dump_draws() -> None:
    # --dump-predictions without --dump-draws: dump_draws is [] while the
    # other dump_* fields carry one row per frame. The merge must not
    # apply the row permutation to the empty list (2026-08-06 state-probe
    # arm 1 crashed at merge on exactly this shape).
    import dataclasses

    rank0 = dataclasses.replace(_shard([0, 5], "user/a", "success"), dump_draws=[])
    rank1 = dataclasses.replace(_shard([3, 8], "user/b", None), dump_draws=[])
    merged = merge_shards([rank0, rank1])
    assert merged.dump_draws == []
    assert merged.dump_index == [0, 3, 5, 8]


def test_merge_shards_tolerates_empty_dump_predictions() -> None:
    # --dump-draws without --dump-predictions: the per-policy prediction
    # lists stay [] while dump_index carries one row per frame — the
    # exact mirror of the dump_draws case above (2026-08-06 fairness
    # probe crashed at merge on this shape after scoring all frames).
    import dataclasses

    rank0 = dataclasses.replace(
        _shard([0, 5], "user/a", "success"),
        dump_predictions={"bijou": []},
    )
    rank1 = dataclasses.replace(
        _shard([3, 8], "user/b", None),
        dump_predictions={"bijou": []},
    )
    merged = merge_shards([rank0, rank1])
    assert merged.dump_predictions == {"bijou": []}
    assert merged.dump_index == [0, 3, 5, 8]
    assert [d[0, 0, 0].item() for d in merged.dump_draws] == [0.0, 3.0, 5.0, 8.0]


def test_score_frame_refuses_zero_valid_frames() -> None:
    # Through the max(divisor, 1) guards a zero-valid frame would score
    # a perfect 0.0 — it must fail loudly instead.
    with pytest.raises(AssertionError, match="no valid chunk steps"):
        score_frame(
            index=7,
            repo_id="user/a",
            predicted=torch.zeros(2, 3),
            truth=torch.zeros(2, 3),
            valid=torch.zeros(2, dtype=torch.bool),
            inference_seconds=0.0,
        )


def test_render_report_emits_collapsible_table(tmp_path: Path) -> None:
    chunk, dim = 4, 3
    sample = ReportSample(
        index=0,
        episode=0,
        frame_in_episode=0,
        repo_id="user/b",
        task="pick",
        state=torch.zeros(dim),
        cameras={"top": torch.zeros(3, 8, 8)},
        truth=torch.zeros(chunk, dim),
        valid=torch.ones(chunk, dtype=torch.bool),
        predictions={"policy": torch.zeros(chunk, dim)},
        aux_generated=None,
        aux_label=None,
    )
    path = tmp_path / "report.html"
    render_report(
        path,
        config_lines=["test"],
        summaries=[],
        comparisons=[],
        motor_names=[f"m{i}" for i in range(dim)],
        samples=[sample],
        total_scored=1,
        theme=THEMES["light"],
        extra_tables=[],
        collapsible_tables=[
            ReportTable(
                title="Per-dataset chunk MAE",
                header=["dataset", "frames"],
                rows=[["user/b", "2"]],
            ),
        ],
    )
    document = path.read_text()
    assert "<details><summary>Per-dataset chunk MAE</summary>" in document
    assert "user/b" in document
