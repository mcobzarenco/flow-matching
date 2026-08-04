"""Eval aggregation surfaces: per-dataset slices and report rendering.

Pure CPU/synthetic. Covers the per-dataset breakdown added for the
offline report (slice_by_dataset ordering + exact aggregation) and
that render_report emits the collapsible block without touching real
model outputs (matplotlib Agg, tiny tensors).
"""

from __future__ import annotations

from pathlib import Path

import torch

from bijou.eval.metrics import FrameScore, slice_by_dataset
from bijou.eval.report import THEMES, ReportSample, ReportTable, render_report


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
