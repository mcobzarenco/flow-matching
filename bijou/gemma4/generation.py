"""Autoregressive decoding for the pure-torch Gemma4 model.

Greedy decoding matches HF `generate(do_sample=False)` exactly: the next token
is the float32 argmax of the last position's (softcapped) logits. Sampling
with temperature/top-k/top-p is provided for completeness with an explicit
`torch.Generator` for reproducibility (HF's RNG stream is not replicated).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .cache import KVCache
from .model import Gemma4Model


@dataclass(frozen=True, slots=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int | None = 64
    top_p: float | None = 0.95


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """``sequences`` includes the prompt; ``step_logits`` holds the pre-argmax
    float32 logits of each generated position (for parity checking)."""

    sequences: Tensor
    step_logits: tuple[Tensor, ...]


def _sample(
    logits: Tensor,
    params: SamplingParams,
    generator: torch.Generator | None,
) -> Tensor:
    logits = logits / params.temperature
    if params.top_k is not None:
        kth = torch.topk(logits, params.top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if params.top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=False, dim=-1)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative <= 1 - params.top_p
        mask = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(mask, -float("inf"))
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model: Gemma4Model,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    pixel_values: Tensor | None = None,
    image_position_ids: Tensor | None = None,
    eos_token_ids: tuple[int, ...] | None = None,
    sampling: SamplingParams | None = None,
    generator: torch.Generator | None = None,
) -> GenerationResult:
    """Batch-size-1 decoding with a KV cache (greedy unless ``sampling``)."""
    if input_ids.shape[0] != 1:
        raise ValueError("generate() currently supports batch size 1")
    if eos_token_ids is None:
        eos_token_ids = model.config.text.eos_token_ids

    cache = KVCache(model.config.text)
    output = model(
        input_ids,
        pixel_values=pixel_values,
        image_position_ids=image_position_ids,
        cache=cache,
        logits_to_keep=1,
    )

    tokens = input_ids
    step_logits: list[Tensor] = []
    for _ in range(max_new_tokens):
        last_logits = output.logits[:, -1, :].float()
        step_logits.append(last_logits)
        if sampling is None:
            next_token = last_logits.argmax(dim=-1)
        else:
            next_token = _sample(last_logits, sampling, generator)
        tokens = torch.cat([tokens, next_token[:, None]], dim=-1)
        if int(next_token.item()) in eos_token_ids:
            break
        output = model(next_token[:, None], cache=cache, logits_to_keep=1)

    return GenerationResult(sequences=tokens, step_logits=tuple(step_logits))
