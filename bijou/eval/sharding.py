"""Multi-GPU eval sharding: one rank's results, and their merge.

The eval runner shards the sampled frame indices round-robin across
ranks (``indices[rank::world_size]`` — indices are sorted, so each rank
gets a near-equal spread across datasets), scores its shard, and
gathers one ``ShardResults`` per rank to rank 0 via
``all_gather_object``. ``merge_shards`` restores the single-process
view: every per-policy score list re-sorted by global frame index, so
the paired-comparison contract (same frames, same order, across
policies) holds for any world size. The single-process path uses the
same merge on a one-element list — one downstream code path.

Determinism note: results reproduce exactly at fixed ``(seed,
world_size, batch_size)``. Across world sizes they match only to
batch-composition numerics (measured on the rcond-100k 1024-frame
holdout eval, 1 vs 4 GPUs: bijou chunk MAE 5.328 vs 5.315, 0.25%;
state-copy baselines match exactly). Root-caused via
``outputs/probe_batch_invariance.py`` (2026-08-04): decode SEMANTICS
are exactly batch-invariant — cpu/fp32 and cuda/fp32 reproduce the
same actions under four batch groupings including ragged padding and
the round-robin interleave — but under the production bf16 backbone a
batch-shape-dependent kernel path flips near-tie greedy argmaxes
(observed even at identical padding width), and one flipped token
cascades autoregressively (max 4.34 raw units on the probe row). The
narrated pass additionally has a designed composition dependence (the
lockstep value phase's repeat-terminators, see ar_backbone.py) —
hence its slightly larger Q3/paired drift. Flow decoders additionally
consume noise draws in shard order — same caveat as training's
per-rank RNG streams.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ..aux_text import AuxGeneration
from .metrics import FrameScore
from .report import ReportSample


@dataclass(frozen=True, slots=True)
class ShardResults:
    """Everything one rank produced while scoring its shard.

    ``outcomes``/``holding_labels``/``progress_labels`` are keyed by
    global frame index (an outcome of None = frame has no requestable
    outcome label). ``generations`` holds the narrated pass's per-frame
    output ({} when no narrated pass ran). The ``dump_*`` fields carry
    the --dump-predictions payload row-aligned with ``dump_index``
    ([] when dumping is off).
    """

    scores: dict[str, list[FrameScore]]
    outcomes: dict[int, str | None]
    holding_labels: dict[int, float]
    progress_labels: dict[int, float]
    # Display-form label strings (aux_text.label_values semantics):
    # event carries the explicit "none" on judge-sampled no-event
    # frames; visible is the positional "object …; gripper …" line.
    event_labels: dict[int, str]
    visible_labels: dict[int, str]
    sensitivity_deltas: list[float]
    report_samples: dict[int, ReportSample]
    generations: dict[int, AuxGeneration]
    dump_predictions: dict[str, list[Tensor]]
    dump_truth: list[Tensor]
    dump_valid: list[Tensor]
    dump_repo: list[str]
    dump_index: list[int]
    # Dataset-local identity (episode index, frame within episode):
    # dump_index is a CONCAT index, valid only under this eval's exact
    # selection — these columns keep npz rows addressable when the
    # corpus composition changes.
    dump_episode: list[int]
    dump_frame: list[int]
    # --dump-draws: the bijou policy's per-frame [draws, chunk, dim]
    # pre-average stacks, row-aligned with dump_index ([] when off).
    dump_draws: list[Tensor]


def merge_shards(shards: list[ShardResults]) -> ShardResults:
    """Merge per-rank results; score lists and dump rows sorted by index.

    Sorting makes the output independent of world size and of gather
    order, so downstream aggregation, JSON payloads and .npz dumps are
    stable across single- and multi-GPU invocations of the same eval.
    """
    scores: dict[str, list[FrameScore]] = {}
    dump_predictions: dict[str, list[Tensor]] = {}
    for shard in shards:
        for name, frame_scores in shard.scores.items():
            scores.setdefault(name, []).extend(frame_scores)
        for name, chunks in shard.dump_predictions.items():
            dump_predictions.setdefault(name, []).extend(chunks)
    for frame_scores in scores.values():
        frame_scores.sort(key=lambda score: score.index)

    dump_index = [index for shard in shards for index in shard.dump_index]
    order = sorted(range(len(dump_index)), key=dump_index.__getitem__)
    # --dump-predictions without --dump-draws: dump_draws stays [] while
    # the other dump_* fields carry one row per dumped frame.
    dump_draws = [d for shard in shards for d in shard.dump_draws]

    def permuted[T](rows: list[T]) -> list[T]:
        return [rows[i] for i in order]

    return ShardResults(
        scores=scores,
        outcomes={k: v for shard in shards for k, v in shard.outcomes.items()},
        holding_labels={
            k: v for shard in shards for k, v in shard.holding_labels.items()
        },
        progress_labels={
            k: v for shard in shards for k, v in shard.progress_labels.items()
        },
        event_labels={k: v for shard in shards for k, v in shard.event_labels.items()},
        visible_labels={
            k: v for shard in shards for k, v in shard.visible_labels.items()
        },
        sensitivity_deltas=[
            delta for shard in shards for delta in shard.sensitivity_deltas
        ],
        report_samples={
            k: v for shard in shards for k, v in shard.report_samples.items()
        },
        generations={k: v for shard in shards for k, v in shard.generations.items()},
        dump_predictions={
            name: permuted(chunks) for name, chunks in dump_predictions.items()
        },
        dump_truth=permuted([t for shard in shards for t in shard.dump_truth]),
        dump_valid=permuted([t for shard in shards for t in shard.dump_valid]),
        dump_repo=permuted([r for shard in shards for r in shard.dump_repo]),
        dump_index=permuted(dump_index),
        dump_episode=permuted([e for shard in shards for e in shard.dump_episode]),
        dump_frame=permuted([f for shard in shards for f in shard.dump_frame]),
        dump_draws=permuted(dump_draws) if dump_draws else [],
    )
