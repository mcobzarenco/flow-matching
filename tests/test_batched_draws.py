"""Batched noise-draw ensembling (main's 2ee2be5, reviewed 2026-08-07):
the draws-major tilers and the seq-vs-batched equivalence oracle.

Pure CPU/synthetic on the tiny FlowDecoder fixture. The load-bearing
guarantees: (a) tile_memory/tile_stats are draws-major — row d·B+i is
(draw d, item i), the collapse_draws/--dump-draws layout; (b) one
solver call at draws x B reproduces the sequential per-draw integration
(same noise, same memory) at fp-noise tolerance, INCLUDING per-sample
RoPE position bases derived from a tiled padding mask with unequal
real lengths; (c) memories the tilers cannot represent (a live KV
cache, un-projected residual taps) are refused loudly, never tiled
inconsistently.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from test_flow_decoder import (
    ACTION_DIM,
    BATCH,
    CHUNK,
    PREFIX_LEN,
    STATE_DIM,
    build,
)
from test_flow_decoder import (
    fabricate as fabricate_unpadded,
)

from bijou.decoders.flow import FlowDecoder, TimeConditioning
from bijou.eval.policies import tile_memory, tile_stats
from bijou.interface import NormStats, ObservationMemory
from bijou.model import SamplingMethod

DRAWS = 3


def fabricate_padded() -> tuple[ObservationMemory, torch.Tensor]:
    """A padded two-item memory with UNEQUAL real lengths (5 vs 3), so
    the decoder's per-sample cross-attention position bases differ by
    row — the part of the memory contract a whole-batch tile could
    silently break."""
    memory, state, _, _ = fabricate_unpadded()
    mask = torch.tensor(
        [[True] * PREFIX_LEN, [True] * 3 + [False] * (PREFIX_LEN - 3)],
    )
    return dataclasses.replace(memory, padding_mask=mask), state


def randomized_decoder() -> FlowDecoder:
    """The equivalence oracle needs a non-trivial velocity field — at
    true init the output projection is zero and every comparison is
    vacuously 0 == 0."""
    decoder = build(TimeConditioning.ADARMS)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                )
                * 0.2,
            )
    return decoder


def test_tile_memory_is_draws_major() -> None:
    memory, _ = fabricate_padded()
    tiled = tile_memory(memory, DRAWS)
    stream, tiled_stream = memory.streams["kv0"], tiled.streams["kv0"]
    mask, tiled_mask = memory.padding_mask, tiled.padding_mask
    assert tiled_stream.key.shape[0] == DRAWS * BATCH
    assert tiled.length == memory.length
    assert mask is not None and tiled_mask is not None
    for draw in range(DRAWS):
        for item in range(BATCH):
            row = draw * BATCH + item
            assert torch.equal(tiled_stream.key[row], stream.key[item])
            assert torch.equal(tiled_stream.value[row], stream.value[item])
            assert torch.equal(tiled_mask[row], mask[item])


def test_tile_stats_tiles_exactly_the_predict_chunk_fields() -> None:
    generator = torch.Generator().manual_seed(2)

    def stats() -> NormStats:
        return NormStats(
            mean=torch.randn(BATCH, ACTION_DIM, generator=generator),
            std=torch.rand(BATCH, ACTION_DIM, generator=generator) + 0.5,
            q01=torch.randn(BATCH, ACTION_DIM, generator=generator),
            q99=torch.randn(BATCH, ACTION_DIM, generator=generator),
        )

    # A structural stand-in is enough: tile_stats replaces exactly
    # state and the two stats, so any dataclass carrying them
    # exercises the whole function.
    batch = dataclasses.make_dataclass(
        "FakeBatch",
        ["state", "action_stats", "state_stats"],
    )(torch.randn(BATCH, STATE_DIM, generator=generator), stats(), stats())
    tiled = tile_stats(batch, DRAWS)
    q99, tiled_q99 = batch.action_stats.q99, tiled.action_stats.q99
    assert q99 is not None and tiled_q99 is not None
    for draw in range(DRAWS):
        rows = slice(draw * BATCH, (draw + 1) * BATCH)
        assert torch.equal(tiled.state[rows], batch.state)
        assert torch.equal(tiled.action_stats.mean[rows], batch.action_stats.mean)
        assert torch.equal(tiled_q99[rows], q99)
        assert torch.equal(tiled.state_stats.std[rows], batch.state_stats.std)


def test_tile_memory_refuses_cache_and_residuals() -> None:
    memory, _ = fabricate_padded()
    with pytest.raises(ValueError, match="cache"):
        tile_memory(dataclasses.replace(memory, cache=object()), DRAWS)
    with pytest.raises(ValueError, match="residual"):
        tile_memory(
            dataclasses.replace(
                memory,
                residuals={"res2": torch.zeros(BATCH, PREFIX_LEN, 8)},
            ),
            DRAWS,
        )


def test_batched_draws_match_sequential() -> None:
    """THE equivalence oracle (mirrors the owner-side fp32 probe,
    9.2e-5 deg on real checkpoints): integrating DRAWS noise draws in
    one solver call at draws x B against a tiled memory reproduces the
    per-draw sequential integration, on a padded memory whose rows
    have different real lengths."""
    decoder = randomized_decoder()
    memory, state = fabricate_padded()
    generator = torch.Generator().manual_seed(3)
    noise = torch.randn(
        DRAWS,
        BATCH,
        CHUNK,
        ACTION_DIM,
        generator=generator,
    )
    sequential = torch.stack(
        [
            decoder.sample_actions(
                memory,
                state,
                noise=noise[draw],
                num_steps=4,
                method=SamplingMethod.HEUN,
            )
            for draw in range(DRAWS)
        ],
    )
    batched = decoder.sample_actions(
        tile_memory(memory, DRAWS),
        state.repeat(DRAWS, 1),
        noise=noise.reshape(DRAWS * BATCH, CHUNK, ACTION_DIM),
        num_steps=4,
        method=SamplingMethod.HEUN,
    ).reshape(DRAWS, BATCH, CHUNK, ACTION_DIM)
    torch.testing.assert_close(batched, sequential, rtol=1e-5, atol=1e-5)
