"""Per-frame prediction metrics and their aggregation.

All errors are computed over the pad-masked chunk in raw action units
(degrees for SO-100/101 joints). ``FrameScore`` keeps the per-frame numbers
small (no tensors) so tens of thousands of them can be held for reporting;
aggregation is exact (valid-step weighted) rather than a mean of means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class FrameScore:
    """Errors of one policy's chunk prediction on one dataset frame."""

    index: int
    repo_id: str
    n_valid: int
    abs_error_sum: float
    squared_error_sum: float
    first_mae: float
    per_motor_abs_sum: tuple[float, ...]
    inference_seconds: float

    @property
    def chunk_mae(self) -> float:
        return self.abs_error_sum / max(self.n_valid * len(self.per_motor_abs_sum), 1)

    @property
    def chunk_mse(self) -> float:
        return self.squared_error_sum / max(
            self.n_valid * len(self.per_motor_abs_sum),
            1,
        )


def score_frame(
    index: int,
    repo_id: str,
    predicted: Tensor,
    truth: Tensor,
    valid: Tensor,
    inference_seconds: float,
) -> FrameScore:
    """predicted/truth: [chunk, action_dim] raw units; valid: [chunk] bool."""
    n_valid = int(valid.sum())
    # A zero-valid frame would flow through the max(divisor, 1) guards as
    # a perfect 0.0 chunk_mae — refuse to score it at all.
    assert n_valid > 0, f"{repo_id} frame {index}: no valid chunk steps"
    diff = (predicted - truth)[valid]
    return FrameScore(
        index=index,
        repo_id=repo_id,
        n_valid=n_valid,
        abs_error_sum=float(diff.abs().sum()),
        squared_error_sum=float((diff**2).sum()),
        first_mae=float((predicted[0] - truth[0]).abs().mean()),
        per_motor_abs_sum=tuple(diff.abs().sum(dim=0).tolist()),
        inference_seconds=inference_seconds,
    )


@dataclass(frozen=True, slots=True)
class PolicySummary:
    """Exact aggregation of one policy's FrameScores."""

    name: str
    frames: int
    chunk_mae: float
    chunk_mse: float
    first_mae: float
    per_motor_mae: tuple[float, ...]
    mae_p50: float
    mae_p90: float
    seconds_per_frame: float

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.name,
            "frames": self.frames,
            "chunk_mae": self.chunk_mae,
            "chunk_mse": self.chunk_mse,
            "first_mae": self.first_mae,
            "per_motor_mae": list(self.per_motor_mae),
            "mae_p50": self.mae_p50,
            "mae_p90": self.mae_p90,
            "seconds_per_frame": self.seconds_per_frame,
        }


def summarize(name: str, scores: list[FrameScore]) -> PolicySummary:
    dims = len(scores[0].per_motor_abs_sum)
    total_valid = sum(s.n_valid for s in scores)
    per_motor = [
        sum(s.per_motor_abs_sum[d] for s in scores) / max(total_valid, 1)
        for d in range(dims)
    ]
    frame_maes = torch.tensor([s.chunk_mae for s in scores])
    return PolicySummary(
        name=name,
        frames=len(scores),
        chunk_mae=sum(s.abs_error_sum for s in scores) / max(total_valid * dims, 1),
        chunk_mse=sum(s.squared_error_sum for s in scores) / max(total_valid * dims, 1),
        first_mae=sum(s.first_mae for s in scores) / len(scores),
        per_motor_mae=tuple(per_motor),
        mae_p50=float(frame_maes.quantile(0.5)),
        mae_p90=float(frame_maes.quantile(0.9)),
        seconds_per_frame=sum(s.inference_seconds for s in scores) / len(scores),
    )


@dataclass(frozen=True, slots=True)
class DatasetSlice:
    """One dataset's share of an eval: frame count plus each policy's
    chunk MAE over exactly those frames (policy name -> MAE; the policy
    set is genuinely dynamic — it depends on the CLI flags — so a dict
    is the right shape there)."""

    frames: int
    chunk_mae: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {"frames": self.frames, "chunk_mae": self.chunk_mae}


def slice_by_dataset(scores: dict[str, list[FrameScore]]) -> dict[str, DatasetSlice]:
    """Per-dataset breakdown across policies: repo_id -> DatasetSlice.

    Every policy must have scored the same frames (the runner guarantees
    this — same contract as ``compare_paired``). Iteration order is frame
    count descending, repo_id tie-break, so renderers can emit rows
    without re-sorting. Small-n caveat travels with the numbers: a
    1024-frame eval leaves 1-3 frames on most of a 900-dataset corpus —
    those rows are noise-dominated.
    """
    per_policy: dict[str, dict[str, list[FrameScore]]] = {}
    for name, frame_scores in scores.items():
        groups: dict[str, list[FrameScore]] = {}
        for score in frame_scores:
            groups.setdefault(score.repo_id, []).append(score)
        per_policy[name] = groups
    reference = next(iter(per_policy.values()))
    ordered = sorted(reference.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    return {
        repo_id: DatasetSlice(
            frames=len(subset),
            chunk_mae={
                name: summarize(name, per_policy[name][repo_id]).chunk_mae
                for name in per_policy
            },
        )
        for repo_id, subset in ordered
    }


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Per-frame paired deltas of a policy against a reference policy."""

    policy: str
    reference: str
    frames: int
    mean_delta: float
    win_rate: float
    delta_p50: float

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "reference": self.reference,
            "frames": self.frames,
            "mean_delta_mae": self.mean_delta,
            "win_rate": self.win_rate,
            "delta_mae_p50": self.delta_p50,
        }


def compare_paired(
    name: str,
    scores: list[FrameScore],
    reference_name: str,
    reference: list[FrameScore],
) -> PairedComparison:
    """Frame-by-frame MAE deltas (policy - reference); negative = better.

    Both lists must cover the same frames in the same order (the runner
    guarantees this by scoring every policy on the same fetched items).
    """
    assert len(scores) == len(reference)
    deltas: list[float] = []
    wins = 0
    for ours, theirs in zip(scores, reference, strict=True):
        assert ours.index == theirs.index
        delta = ours.chunk_mae - theirs.chunk_mae
        deltas.append(delta)
        wins += delta < 0
    delta_tensor = torch.tensor(deltas)
    return PairedComparison(
        policy=name,
        reference=reference_name,
        frames=len(deltas),
        mean_delta=float(delta_tensor.mean()),
        win_rate=wins / len(deltas),
        delta_p50=float(delta_tensor.quantile(0.5)),
    )


def format_table(rows: list[list[str]], header: list[str]) -> str:
    """Space-aligned table; numeric-looking cells right-aligned."""

    def is_number(cell: str) -> bool:
        try:
            float(cell)
        except ValueError:
            return False
        return True

    table = [header, *rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = []
    for row in table:
        cells = [
            cell.rjust(widths[i]) if is_number(cell) else cell.ljust(widths[i])
            for i, cell in enumerate(row)
        ]
        lines.append("  ".join(cells).rstrip())
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def is_finite_summary(summary: PolicySummary) -> bool:
    return math.isfinite(summary.chunk_mae) and math.isfinite(summary.chunk_mse)
