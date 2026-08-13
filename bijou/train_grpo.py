"""Token-GRPO gradient step (design memo 2026-08-13 §2/§8 item 2):
advantage-weighted clipped token-level CE over the action block of a
SAMPLED suffix, DAPO clip-higher bounds, ratio against the rollout's
recorded per-token logprobs (the TokenRow surface, §8 item 1).

The training forward is the SFT teacher-forced path verbatim —
:func:`~bijou.decoders.ar_backbone.suffix_targets` builds the same
``[opener][BOA][a_1..a_T]`` scaffold and the decoder forward consumes
the memory's prefix cache exactly as :func:`ar_backbone_losses` does —
except the suffix carries the ROLLOUT-SAMPLED ids and the objective is
the PPO/GRPO clipped surrogate instead of CE against dataset targets.
Only sampled DECISIONS are trained: the opener predicts constants, BOA
was forced by the decode scaffold (its identity is not a decision),
and PAD padding never carries loss — position ``t`` of a row's
recorded ids is scored, nothing else.

Grammar-mask contract (the item-2 oracle): the decode's action mask at
step ``t`` is budget arithmetic over the id prefix — a pure function
of the recorded ids — so :func:`grammar_masks_from_ids` recomputes it
trainer-side and tests/test_grpo_step.py pins bit-equality against the
rollout's recorded packbits surface. The train-time distribution is
then the SAME masked softmax the decode sampled from (illegal columns
at −inf, temperature-scaled, fp32), so at an unchanged policy every
ratio is 1 to reduction-shape noise only (the §8 amended bound: the
one-shot teacher-forced forward vs the decode's incremental cache
feeding — logprobs within 1e-5, not bitwise).

Loss forms mirror the SFT pair: :func:`grpo_objective_sums` is the
sum-form half (caller owns normalization — chunked backward divides by
FULL-batch token counts), :func:`grpo_loss` the single-batch
token-weighted mean. Mask-multiply, never boolean-index, on the live
path (the ar_backbone_loss_sums discipline); stats are detached.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import Tensor, nn

from .decoders.ar_backbone import ARSuffixDecoder, suffix_targets
from .interface import CollatedBatch, ObservationMemory


@dataclasses.dataclass(frozen=True, slots=True)
class GRPOConfig:
    """Frozen surrogate constants (memo §2): DAPO clip-higher
    ``[0.8, 1.28]`` on the absolute ratio, rollout temperature 1.0.
    ``temperature`` must equal the temperature the rows were sampled
    at — the ratio is π_new/π_old under the SAME masked softmax, and
    the rollout recorded its own (TokenRow.temperature; the collator
    guards the match, this module trusts its caller)."""

    clip_low: float = 0.8
    clip_high: float = 1.28
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.clip_low <= 1.0 <= self.clip_high):
            raise ValueError(
                f"clip bounds [{self.clip_low}, {self.clip_high}] must "
                "bracket 1 with a positive lower bound — the surrogate "
                "is anchored at ratio 1",
            )
        if self.temperature <= 0.0:
            raise ValueError(f"temperature {self.temperature} must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class GRPOStats:
    """Detached per-step logging facts. ``approx_kl`` is the k3
    estimator of KL(π_old ‖ π_new) over trained tokens — ratio − 1 −
    log ratio, non-negative, 0 at an unchanged policy; the drift
    tripwire input (§7). The anchor KL (§2, vs frozen er60k) needs a
    reference forward and rides the loop harness, not the step."""

    tokens: int
    mean_ratio: float
    min_ratio: float
    max_ratio: float
    clip_fraction: float
    approx_kl: float


def grammar_masks_from_ids(
    decoder: ARSuffixDecoder[Any],
    ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """(grammar masks [B, T, vocab_total] bool, decision positions
    [B, T] bool) recomputed from PAD-padded CODEC ids alone — the
    trainer's half of "train-time grammar mask == rollout mask". The
    decode's mask at step ``t`` depends only on the remaining symbol
    budget, i.e. on the id prefix: legal ids are the non-special
    tokens whose symbol expansion fits, PAD exactly when the chunk is
    consumed. Decision positions are the steps with budget remaining —
    everything a rollout row records, nothing more; trailing PAD
    padding sits at remaining 0 and is excluded. Loud on rows that do
    not consume the chunk exactly or carry non-PAD ids past
    consumption (corrupt rows, never a real decode's output)."""
    lengths = decoder.symbol_lengths.to(ids.device)
    pad = decoder.codec.pad
    total = decoder.config.chunk_size * decoder.config.action_dim
    step_lengths = lengths[ids]  # [B, T]; specials (PAD) expand to 0
    remaining = total - (step_lengths.cumsum(dim=1) - step_lengths)
    decisions = remaining > 0
    consumed = step_lengths.sum(dim=1)
    if not bool((consumed == total).all()):
        raise ValueError(
            f"sampled rows must consume the chunk exactly ({total} "
            f"symbols) — got per-row totals {consumed.tolist()}; these "
            "are not a real decode's ids",
        )
    if not bool((ids[~decisions] == pad).all()):
        raise ValueError(
            "non-PAD ids past chunk consumption — corrupt rows (a real "
            "decode emits only PAD once its budget is spent)",
        )
    allowed = (lengths > 0)[None, None, :] & (
        lengths[None, None, :] <= remaining[..., None]
    )
    allowed[..., pad] = remaining == 0
    return allowed, decisions


def sampled_token_logprobs[B: nn.Module](
    backbone: B,
    decoder: ARSuffixDecoder[B],
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
    *,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """(per-token chosen logprobs [B, T] fp32 WITH graph, decision
    positions [B, T] bool) for a batch whose ``action_tokens`` carry a
    rollout's SAMPLED ids (``[BOA, a_1..a_T]``, PAD-padded — the
    collator convention): one teacher-forced decoder forward over the
    SFT scaffold, block columns reduced under the recomputed grammar
    mask — ``log_softmax(block_logits / temperature)`` with illegal
    columns at −inf, exactly the distribution the decode sampled from
    (token_rows_from_capture's reduction). CONSUMES the memory's
    cache, like every decoder forward. Non-decision positions gather
    PAD's logprob (0 — the only legal id there) and must be excluded
    by the caller via the returned mask."""
    if batch.suffix_tokens is not None:
        raise SystemExit(
            "GRPO trains the action block of a SAMPLED stream — aux "
            "suffixes (suffix_tokens) are SFT-only; build the replay "
            "batch with action_tokens",
        )
    tokens = batch.action_tokens
    if tokens is None:
        raise SystemExit(
            "GRPO batch carries no action_tokens — the replay collator "
            "must supply the rollout's sampled ids ([BOA, a_1..a_T])",
        )
    codec = decoder.codec
    if not bool((tokens[:, 0] == codec.boa).all()):
        raise ValueError(
            "action_tokens must open with BOA (the collator convention "
            "— ActionCodec.encode's [BOA, t_1..t_k]); got leading ids "
            f"{tokens[:, 0].tolist()}",
        )
    full, _, _ = suffix_targets(decoder, batch)
    logits = decoder(backbone, memory, full[:, :-1])
    # Position j predicts full[:, j + 1]: with k opener ids and BOA at
    # full index k, sampled id a_t (full index k + t) is predicted at
    # j = k + t − 1 — columns [k, k + T) for the T = width − 1 ids.
    k = len(decoder.opener_ids)
    ids = tokens[:, 1:]
    base = decoder.config.block_base
    block = logits[:, k : k + ids.shape[1], base : base + decoder.config.vocab_total]
    allowed, decisions = grammar_masks_from_ids(decoder, ids)
    logprobs = (
        (block.float() / temperature)
        .masked_fill(~allowed, float("-inf"))
        .log_softmax(-1)
    )
    return logprobs.gather(-1, ids[..., None]).squeeze(-1), decisions


def grpo_objective_sums(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    decisions: Tensor,
    config: GRPOConfig,
) -> tuple[Tensor, Tensor, GRPOStats]:
    """Sum-form clipped surrogate: (objective SUM with graph, decision
    count, detached stats). Per token, ratio ``r = exp(new − old)``,
    objective ``min(r·A, clamp(r, clip_low, clip_high)·A)`` with the
    row's advantage broadcast to its tokens (memo §2); the caller
    normalizes (−sum/count is the token-weighted mean; chunked
    backward divides per-chunk sums by the FULL-batch count). At an
    unchanged policy ``exp(x − x.detach())`` is exactly 1 with
    gradient ∇new, so the surrogate's gradient IS advantage-weighted
    CE's (the item-2 oracle, bit-exact — torch.minimum splits tie
    gradients evenly and the halves resum exactly); zero advantage
    zeroes every gradient identically."""
    if new_logprobs.shape != old_logprobs.shape or advantages.shape != (
        new_logprobs.shape[0],
    ):
        raise ValueError(
            f"shape mismatch: new {tuple(new_logprobs.shape)}, old "
            f"{tuple(old_logprobs.shape)}, advantages "
            f"{tuple(advantages.shape)} — old must match new [B, T], "
            "advantages one scalar per row [B]",
        )
    # .to(new_logprobs) carries device AND dtype: callers hand rollout
    # records / group z-scores as fresh CPU tensors while the training
    # forward lives on the GPU (measured live: the R0 first gradient
    # step, 2026-08-13 15:51Z — cuda/cpu crash the CPU oracles cannot
    # see).
    old_logprobs = old_logprobs.to(new_logprobs)
    advantage = advantages.to(new_logprobs)[:, None]
    if not bool(torch.isfinite(old_logprobs[decisions]).all()):
        raise ValueError(
            "non-finite rollout logprobs at decision positions — "
            "corrupt rows (every recorded logprob is a finite masked "
            "log-softmax value)",
        )
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantage
    clipped = ratio.clamp(config.clip_low, config.clip_high) * advantage
    trained = decisions.to(new_logprobs.dtype)
    objective_sum = (torch.minimum(unclipped, clipped) * trained).sum()
    count = decisions.sum()
    with torch.no_grad():
        tokens = int(count)
        denominator = count.clamp(min=1).to(new_logprobs.dtype)
        mean_ratio = float((ratio * trained).sum() / denominator)
        anchored = ratio.masked_fill(~decisions, 1.0)
        outside = (ratio < config.clip_low) | (ratio > config.clip_high)
        clip_fraction = float((outside & decisions).sum() / denominator)
        # k3 = r − 1 − log r; log ratio is the logprob difference the
        # ratio was exponentiated from (no log(exp) round trip).
        k3 = ratio - 1.0 - (new_logprobs - old_logprobs)
        stats = GRPOStats(
            tokens=tokens,
            mean_ratio=mean_ratio,
            min_ratio=float(anchored.amin()),
            max_ratio=float(anchored.amax()),
            clip_fraction=clip_fraction,
            approx_kl=float((k3 * trained).sum() / denominator),
        )
    return objective_sum, count, stats


def grpo_loss[B: nn.Module](
    backbone: B,
    decoder: ARSuffixDecoder[B],
    memory: ObservationMemory,
    batch: CollatedBatch[Any],
    *,
    old_logprobs: Tensor,
    advantages: Tensor,
    config: GRPOConfig,
) -> tuple[Tensor, GRPOStats]:
    """Single-batch GRPO step objective: (scalar loss with graph —
    the NEGATED token-weighted mean surrogate — detached stats).
    ``old_logprobs`` [B, T] are the rollout's recorded per-token
    logprobs padded to the batch width (padding values never
    contribute — mask-multiplied out); ``advantages`` [B] the group
    z-scores, one per row, broadcast to the row's tokens."""
    new_logprobs, decisions = sampled_token_logprobs(
        backbone,
        decoder,
        memory,
        batch,
        temperature=config.temperature,
    )
    objective_sum, count, stats = grpo_objective_sums(
        new_logprobs,
        old_logprobs,
        advantages,
        decisions,
        config,
    )
    return -(objective_sum / count.clamp(min=1)), stats
