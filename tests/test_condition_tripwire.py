"""Conditioning-tripwire noise reuse.

The tripwire compares an outcome-overridden decode against the scalar
pass's predictions; for a flow decoder a fresh draw floors mean |Δ| at
the sampling variance even when the model is conditioning-blind — the
exact state the alarm exists to catch. The gemma boundary
(``models.gemma_flow.decode_chunk``) therefore returns the noise it
integrated, and the tripwire's override decode reuses it per item.
These tests pin the two guarantees that rests on: (1) the boundary's
fallback draw is bit-exact the one ``sample_actions`` would make (same
actions, same generator consumption), and (2) reusing the returned
noise makes an otherwise-identical re-decode bitwise equal, so a
conditioning-blind model's Δ is exactly zero.
"""

from __future__ import annotations

from typing import Any

import torch
from test_flow_decoder import ACTION_DIM, CHUNK, build, fabricate
from test_snapflow_distill import flow_batch, randomize

from bijou.modelling.decoders.flow import FlowDecoder, TimeConditioning
from bijou.modelling.encoders.gemma4 import GemmaMemory
from bijou.modelling.interface import CollatedBatch
from bijou.models.gemma_flow import decode_chunk


def _decoder_memory_batch() -> tuple[
    FlowDecoder,
    GemmaMemory,
    CollatedBatch[Any],
]:
    decoder = build(TimeConditioning.ADARMS)
    randomize(decoder)
    memory, _, _, _ = fabricate()
    return decoder, memory, flow_batch()


def test_generator_draw_is_the_documented_randn() -> None:
    """decode_chunk's fallback draw must be exactly one randn of
    [B, chunk, action_dim] in the normalized state's dtype — the same
    call sample_actions makes, so the boundary's draw shifts neither
    the decode nor the generator's downstream draws."""
    decoder, memory, batch = _decoder_memory_batch()
    g_used = torch.Generator().manual_seed(11)
    actions, drawn = decode_chunk(decoder, memory, batch, generator=g_used, num_steps=3)

    g_manual = torch.Generator().manual_seed(11)
    expected_noise = torch.randn(
        batch.state.shape[0],
        CHUNK,
        ACTION_DIM,
        dtype=batch.state.dtype,
        generator=g_manual,
    )
    assert torch.equal(drawn, expected_noise)
    assert torch.equal(g_used.get_state(), g_manual.get_state())

    via_noise, _ = decode_chunk(
        decoder,
        memory,
        batch,
        noise=expected_noise,
        num_steps=3,
    )
    assert torch.equal(via_noise, actions)


def test_supplied_noise_is_returned_verbatim() -> None:
    decoder, memory, batch = _decoder_memory_batch()
    noise = torch.randn(
        batch.state.shape[0],
        CHUNK,
        ACTION_DIM,
        generator=torch.Generator().manual_seed(5),
    )
    _, returned = decode_chunk(decoder, memory, batch, noise=noise, num_steps=3)
    assert returned is noise


def test_reused_noise_zeroes_the_blind_model_delta() -> None:
    """The tripwire's shape: a scalar-pass decode with an advancing
    generator, then a per-row re-decode of a subset. With the captured
    noise the re-decode is BITWISE the scalar pass (Δ exactly 0 for a
    model the override cannot reach); without it, the same re-decode
    lands on fresh noise and a spurious nonzero Δ."""
    decoder, memory, batch = _decoder_memory_batch()
    generator = torch.Generator().manual_seed(11)
    scalar, scalar_noise = decode_chunk(
        decoder,
        memory,
        batch,
        generator=generator,
        num_steps=3,
    )

    reused, _ = decode_chunk(
        decoder,
        memory,
        batch,
        noise=scalar_noise,
        num_steps=3,
    )
    assert torch.equal(reused, scalar)

    fresh, _ = decode_chunk(decoder, memory, batch, generator=generator, num_steps=3)
    assert not torch.equal(fresh, scalar)
