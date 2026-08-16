"""Shared operation bodies for families whose action decoder is an
:class:`~bijou.modelling.decoders.ar_suffix.ARSuffixDecoder` (the
discrete suffix role): the two-phase CE objective composition, the
block decode/scoring adapters, and the narrated instruments.

Free functions, not a base class (families share code through
composition): each takes the (backbone, decoder) pair plus an
already-encoded memory — the FAMILY owns encode and the precision
policy around these calls; these functions never open autocast regions
or grad modes of their own."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

from ..modelling.aux_text import AuxField
from ..modelling.decoders.ar_suffix import (
    ARSuffixDecoder,
    PrefixMemory,
    ar_backbone_counts,
    ar_backbone_loss_sums,
)
from ..modelling.interface import (
    ActionCaptureStep,
    ARSampling,
    CollatedBatch,
    ValueCandidate,
)
from ..vla import ARPrediction, Loss, LossReport, NarratedPrediction


def ar_loss_counts(
    decoder: ARSuffixDecoder[Any, Any],
    batch: CollatedBatch[Any],
) -> dict[str, Tensor]:
    """The suffix objective's per-component counts (data-only, no
    forward): CE action-target count under ``"action_ar"``, plus the
    narration value-line target count under ``"narration"`` iff the batch carries an
    aux-augmented suffix (run-constant by collation)."""
    action_count, aux_count = ar_backbone_counts(decoder, batch)
    counts = {"action_ar": action_count}
    if aux_count is not None:
        counts["narration"] = aux_count
    return counts


def ar_suffix_report[B: nn.Module, M: PrefixMemory](
    backbone: B,
    decoder: ARSuffixDecoder[B, M],
    memory: M,
    batch: CollatedBatch[Any],
    *,
    counts: dict[str, Tensor],
    narration_weight: float,
) -> LossReport:
    """The suffix CE objective in two-phase form: sum-form losses over
    the ALL-REDUCED ``counts`` — objective = action_sum·W/action_count
    (+ narration_weight·aux_sum·W/max(aux_count, 1) when the run trains
    value lines), whose DDP mean is the global token-weighted mean."""
    action_sum, action_count, aux_sum, aux_count = ar_backbone_loss_sums(
        backbone,
        decoder,
        memory,
        batch,
    )
    world = dist.get_world_size() if dist.is_initialized() else 1
    objective = action_sum * world / counts["action_ar"]
    components = {"action_ar": Loss(sum=action_sum, count=action_count)}
    if aux_sum is not None:
        assert aux_count is not None  # the sums contract: aux rides as a pair
        objective = objective + narration_weight * (
            aux_sum * world / counts["narration"].clamp(min=1)
        )
        components["narration"] = Loss(sum=aux_sum, count=aux_count)
    return LossReport(objective=objective, components=components)


def ar_block_prediction[B: nn.Module, M: PrefixMemory](
    backbone: B,
    decoder: ARSuffixDecoder[B, M],
    memory: M,
    batch: CollatedBatch[Any],
    *,
    sampling: ARSampling | None,
    capture: list[ActionCaptureStep] | None,
) -> ARPrediction:
    """The action-block decode (``generate=()`` — never any text; the
    empty-request generations are discarded, not surfaced)."""
    actions, _ = decoder.predict_chunk(
        backbone,
        memory,
        batch,
        generate=(),
        sampling=sampling,
        action_capture=capture,
    )
    return ARPrediction(actions=actions)


def ar_block_logits[B: nn.Module, M: PrefixMemory](
    backbone: B,
    decoder: ARSuffixDecoder[B, M],
    memory: M,
    action_ids: Tensor,
) -> Tensor:
    """Teacher-forced block logits for a rectangular id batch — the
    tensor adapter over the decoder's per-row surface (one prefill
    plus one suffix forward; CONSUMES the memory's cache).

    Shapes:
      - ``action_ids``: [B, S] long — block-relative ids
      - returns: [B, S, vocab_total] float32
    """
    rows: list[list[int] | None] = [[int(token) for token in row] for row in action_ids]
    outputs = decoder.teacher_forced_block_logits(backbone, memory, rows)
    logits = [row_logits for row_logits in outputs if row_logits is not None]
    assert len(logits) == len(outputs)  # tensor form has no filler rows
    return torch.stack(logits)


def narrated_prediction[B: nn.Module, M: PrefixMemory](
    backbone: B,
    decoder: ARSuffixDecoder[B, M],
    memory: M,
    batch: CollatedBatch[Any],
    *,
    generate: tuple[AuxField, ...],
) -> NarratedPrediction:
    """The narrated pass: value lines then actions, one decode. The
    decoder validates the request against its TRAINED fields and
    template order; an empty request is refused here (action-only
    inference is the block decode)."""
    if not generate:
        raise ValueError(
            "predict_narrated needs a non-empty generate request — "
            "action-only inference is predict/predict_ar",
        )
    actions, generations = decoder.predict_chunk(
        backbone,
        memory,
        batch,
        generate=generate,
    )
    return NarratedPrediction(actions=actions, generations=generations)


def value_candidates[B: nn.Module, M: PrefixMemory](
    backbone: B,
    decoder: ARSuffixDecoder[B, M],
    memory: M,
    batch: CollatedBatch[Any],
    *,
    field: AuxField,
    generate: tuple[AuxField, ...],
    draws: int,
    sampling_for_draw: Callable[[int], ARSampling],
) -> tuple[NarratedPrediction, list[list[ValueCandidate]]]:
    """The subgoal-draws instrument over ONE prefill: (a) the full
    narrated pass — op-identical to :func:`narrated_prediction` so the
    draws=0 limit stays bit-exact against it — and (b) ``draws + 1``
    text-only decodes of ``field`` against the restored prefix cache:
    candidate 0 greedy, candidates 1..draws temperature-sampled under
    ``sampling_for_draw(draw)``. The greedy candidate's text must equal
    the full pass's parsed field value — same restored cache state,
    same ops — and a mismatch is a broken instrument, not a warning."""
    if draws < 0:
        raise ValueError(f"draws must be >= 0, got {draws}")
    snapshot = decoder.cache_snapshot(memory)
    actions, generations = decoder.predict_chunk(
        backbone,
        memory,
        batch,
        generate=generate,
    )
    rows: list[list[ValueCandidate]] = [[] for _ in range(batch.state.shape[0])]
    for draw in range(draws + 1):
        decoder.cache_restore(memory, snapshot)
        sampling = None if draw == 0 else sampling_for_draw(draw)
        for row, candidate in enumerate(
            decoder.decode_value_line(
                backbone,
                memory,
                field=field,
                sampling=sampling,
            ),
        ):
            rows[row].append(candidate)
    for row, (generation, row_candidates) in enumerate(
        zip(generations, rows, strict=True),
    ):
        parsed = getattr(generation, field.value)
        if parsed is not None and not isinstance(parsed, str):
            raise TypeError(
                f"{field.value!r} parses to {type(parsed).__name__}, not "
                "text — candidate decoding covers free-text fields only",
            )
        full_value = parsed or ""
        if full_value != row_candidates[0].text:
            raise SystemExit(
                f"candidate-0 drift on batch row {row}: text-only greedy "
                f"decode {row_candidates[0].text!r} != full-pass value "
                f"{full_value!r} — the shared-prefill contract broke, stop",
            )
    return (
        NarratedPrediction(actions=actions, generations=generations),
        rows,
    )
