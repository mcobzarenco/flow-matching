"""Chunked backward (``--backward-chunks``): the pre-registered memory
fallback for runs whose loader batch doesn't fit (E4B screen pre-reg,
2026-08-05).

Covers the exactness contract at every altitude: the sum-form losses
reconstruct the mean-form objectives componentwise (all three decoder
families); the forward-free count helpers agree with the sums; the
collate-time splitter preserves per-step sample composition; and — the
pre-registered oracle — accumulating per-chunk sum-form gradients
normalized by FULL-batch counts reproduces the unchunked gradient at
fp-noise tolerance, on an aux batch whose chunks carry UNEQUAL aux
counts (the case a naive mean-of-chunk-means gets wrong). Memory is
held bit-identical across the comparison (sliced from one prefill);
the padding-width fp realization of per-chunk collation is measured
separately (diagnostic 2026-08-05: same math, gradient rel ~2e-4 on
the saturated random tiny fixture, forward identical to 1e-6).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import torch
from test_ar_backbone import (
    BATCH,
    FakeInputs,
    batch,
    build,
    codec,
    encode_memory,
)
from test_ar_backbone_aux import build_with_aux
from test_flow_decoder import (
    build as build_flow,
)
from test_flow_decoder import (
    fabricate,
)

from bijou.aux_text import assemble_suffix
from bijou.decoders.ar_backbone import (
    ar_backbone_counts,
    ar_backbone_loss_sums,
    ar_backbone_losses,
)
from bijou.decoders.flow import (
    TimeConditioning,
    flow_matching_loss,
    flow_matching_loss_sums,
)
from bijou.gemma4.cache import KVCache
from bijou.interface import CollatedBatch, MemoryStream, NormStats, ObservationMemory
from bijou.train import ChunkedBatch, ChunkingCollator


def slice_batch(sample: CollatedBatch[Any], i: int) -> CollatedBatch[Any]:
    def stats(s: NormStats) -> NormStats:
        return dataclasses.replace(
            s,
            mean=s.mean[i : i + 1],
            std=s.std[i : i + 1],
            q01=None if s.q01 is None else s.q01[i : i + 1],
            q99=None if s.q99 is None else s.q99[i : i + 1],
        )

    return dataclasses.replace(
        sample,
        state=sample.state[i : i + 1],
        actions=sample.actions[i : i + 1],
        action_is_pad=sample.action_is_pad[i : i + 1],
        action_stats=stats(sample.action_stats),
        state_stats=stats(sample.state_stats),
        action_tokens=(
            None if sample.action_tokens is None else sample.action_tokens[i : i + 1]
        ),
        suffix_tokens=(
            None if sample.suffix_tokens is None else sample.suffix_tokens[i : i + 1]
        ),
        suffix_is_aux=(
            None if sample.suffix_is_aux is None else sample.suffix_is_aux[i : i + 1]
        ),
    )


def slice_memory(memory: ObservationMemory, i: int) -> ObservationMemory:
    """Row i of a batched memory, cache included — the SAME prefix
    numerics as the batch prefill (bit-identical slices), so gradient
    comparisons isolate the chunk decomposition itself."""
    cache = memory.cache
    if cache is not None:
        assert isinstance(cache, KVCache)  # opaque at the seam; Gemma here
        sliced = copy.copy(cache)
        sliced.layers = []
        for layer in cache.layers:
            layer_copy = copy.copy(layer)
            if layer_copy.keys is not None and layer_copy.values is not None:
                layer_copy.keys = layer_copy.keys[i : i + 1]
                layer_copy.values = layer_copy.values[i : i + 1]
            sliced.layers.append(layer_copy)
        cache = sliced
    return ObservationMemory(
        streams={
            name: MemoryStream(s.key[i : i + 1], s.value[i : i + 1])
            for name, s in memory.streams.items()
        },
        length=memory.length,
        padding_mask=(
            None if memory.padding_mask is None else memory.padding_mask[i : i + 1]
        ),
        cache=cache,
    )


def decoder_grads(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def aux_sample(loaded: Any, decoder: Any) -> CollatedBatch[Any]:
    """The aux-on fixture batch: row 0 carries aux value lines, row 1
    none — chunks of 1 then have UNEQUAL aux (and action) counts, the
    exact case where normalizing by full-batch counts differs from a
    mean of chunk means."""
    sample = batch(loaded)
    tokens = sample.action_tokens
    assert tokens is not None
    aux_ids = [[ord(c) for c in "yes\n30%\n"], []]
    suffix, is_aux = assemble_suffix(
        aux_ids,
        tokens,
        block_base=decoder.config.block_base,
        codec_pad=loaded.pad,
    )
    return dataclasses.replace(sample, suffix_tokens=suffix, suffix_is_aux=is_aux)


def test_chunking_collator_splits_evenly_and_preserves_order() -> None:
    def identity(items: list[Any]) -> list[Any]:
        return list(items)  # a stand-in collator

    chunked = ChunkingCollator(identity, 4)(list(range(12)))  # type: ignore[arg-type]
    assert isinstance(chunked, ChunkedBatch)
    assert [len(c) for c in chunked.chunks] == [3, 3, 3, 3]  # type: ignore[arg-type]
    assert [x for c in chunked.chunks for x in c] == list(range(12))  # type: ignore[misc]
    # A short straggler batch splits near-evenly, no empty chunks.
    straggler = ChunkingCollator(identity, 4)(list(range(5)))  # type: ignore[arg-type]
    assert [len(c) for c in straggler.chunks] == [1, 1, 1, 2]  # type: ignore[arg-type]
    assert [x for c in straggler.chunks for x in c] == list(range(5))  # type: ignore[misc]


def test_ar_backbone_sums_reconstruct_mean_losses() -> None:
    backbone, decoder, loaded = build()
    sample = batch(loaded)
    total, _action, aux_sum, aux_count = ar_backbone_losses(
        backbone,
        decoder,
        encode_memory(backbone),
        sample,
    )
    assert aux_sum is None and aux_count is None
    action_sum, action_count, sums_aux, sums_aux_count = ar_backbone_loss_sums(
        backbone,
        decoder,
        encode_memory(backbone),
        sample,
    )
    assert sums_aux is None and sums_aux_count is None
    assert torch.allclose(action_sum / action_count, total, atol=1e-6)
    helper_action, helper_aux = ar_backbone_counts(decoder, sample)
    assert torch.equal(helper_action, action_count)
    assert helper_aux is None


def test_ar_backbone_aux_sums_reconstruct_components() -> None:
    backbone, decoder = build_with_aux()
    loaded = codec()
    sample = aux_sample(loaded, decoder)
    total, action, aux_sum, aux_count = ar_backbone_losses(
        backbone,
        decoder,
        encode_memory(backbone),
        sample,
    )
    assert aux_sum is not None and aux_count is not None
    action_sum, action_count, sums_aux, sums_aux_count = ar_backbone_loss_sums(
        backbone,
        decoder,
        encode_memory(backbone),
        sample,
    )
    assert sums_aux is not None and sums_aux_count is not None
    assert torch.allclose(action_sum / action_count, action, atol=1e-6)
    assert torch.allclose(sums_aux, aux_sum, atol=1e-6)
    assert torch.equal(sums_aux_count, aux_count)
    reconstructed = action_sum / action_count + decoder.aux_loss_weight * (
        sums_aux / sums_aux_count.clamp(min=1)
    )
    assert torch.allclose(reconstructed, total, atol=1e-6)
    helper_action, helper_aux = ar_backbone_counts(decoder, sample)
    assert torch.equal(helper_action, action_count)
    assert helper_aux is not None and torch.equal(helper_aux, aux_count)


def test_chunked_gradient_matches_unchunked_ar_backbone_aux() -> None:
    """THE pre-registered oracle (E4B screen pre-reg): per-chunk
    sum-form backwards over full-batch normalizers accumulate the
    unchunked gradient at fp-noise tolerance. Aux-on, chunk aux counts
    unequal (8 vs 0) — a mean-of-chunk-means is wrong here by
    construction; the full-count normalization is not."""
    backbone, decoder = build_with_aux()
    loaded = codec()
    sample = aux_sample(loaded, decoder)
    rows = [slice_batch(sample, i) for i in range(BATCH)]
    counts = [ar_backbone_counts(decoder, row) for row in rows]
    aux_counts = [c[1] for c in counts]
    assert aux_counts[0] is not None and aux_counts[1] is not None
    assert int(aux_counts[0]) != int(aux_counts[1])  # the unequal case
    action_norm = torch.stack([c[0] for c in counts]).sum()
    aux_norm = torch.stack([aux_counts[0], aux_counts[1]]).sum()

    total, _, _, _ = ar_backbone_losses(
        backbone,
        decoder,
        encode_memory(backbone),
        sample,
    )
    total.backward()
    reference = decoder_grads(decoder)
    decoder.zero_grad()

    chunked_total = torch.zeros(())
    for i, row in enumerate(rows):
        action_sum, _, aux_sum, _ = ar_backbone_loss_sums(
            backbone,
            decoder,
            slice_memory(encode_memory(backbone), i),
            row,
        )
        assert aux_sum is not None
        share = action_sum / action_norm + decoder.aux_loss_weight * (
            aux_sum / aux_norm.clamp(min=1)
        )
        share.backward()
        chunked_total = chunked_total + share.detach()
    accumulated = decoder_grads(decoder)

    assert torch.allclose(chunked_total, total.detach(), atol=1e-6)
    assert set(accumulated) == set(reference)
    for name, grad in reference.items():
        scale = float(grad.norm()) or 1.0
        rel = float((accumulated[name] - grad).norm()) / scale
        # Bound calibrated cross-hardware (2026-08-07): <1e-5 on the
        # H100 box, 1.0004e-4 on the owner's RTX 3000 Ada (same math,
        # different kernels/reduction order — the module docstring's
        # padding-width diagnostic measures ~2e-4 for same-math fp
        # realizations). The guarded failure mode (mean-of-chunk-means
        # normalization) shows rel ≫ 1e-2, so 5e-4 stays a sharp oracle.
        assert rel < 5e-4, f"{name}: chunked-vs-unchunked gradient rel {rel}"


def test_ar_fast_sums_reconstruct_mean() -> None:
    from test_ar_fast import batch as fast_batch
    from test_ar_fast import build as build_fast
    from test_ar_fast import memory as fast_memory

    from bijou.decoders.ar_fast import (
        ar_fast_counts,
        ar_fast_loss,
        ar_fast_loss_sums,
    )

    decoder, loaded = build_fast()
    sample = fast_batch(loaded)
    mean = ar_fast_loss(decoder, fast_memory(), sample)
    loss_sum, count = ar_fast_loss_sums(decoder, fast_memory(), sample)
    assert torch.allclose(loss_sum / count, mean, atol=1e-6)
    assert torch.equal(ar_fast_counts(decoder, sample), count)


def test_flow_sums_reconstruct_mean() -> None:
    decoder = build_flow(TimeConditioning.ADARMS)
    memory, state, actions, _ = fabricate()
    stats = NormStats(
        mean=torch.zeros(actions.shape[0], actions.shape[-1]),
        std=torch.ones(actions.shape[0], actions.shape[-1]),
        q01=None,
        q99=None,
    )
    sample = CollatedBatch(
        encoder_inputs=FakeInputs(),
        state=state,
        actions=actions,
        action_is_pad=torch.zeros(actions.shape[:2], dtype=torch.bool),
        action_stats=stats,
        state_stats=stats,
        action_tokens=None,
        suffix_tokens=None,
        suffix_is_aux=None,
    )
    # Same RNG consumption (identical shapes) => identical noise/tau
    # draws => the identity is exact, not just distributional.
    torch.manual_seed(11)
    mean = flow_matching_loss(decoder, memory, sample)
    torch.manual_seed(11)
    loss_sum, count = flow_matching_loss_sums(decoder, memory, sample)
    assert int(count) == actions.numel()
    assert torch.allclose(loss_sum / count, mean, atol=1e-6)


def test_chunked_batch_transfer_surface() -> None:
    _, _, loaded = build()
    sample = batch(loaded)
    chunked = ChunkedBatch(
        (slice_batch(sample, 0), slice_batch(sample, 1)),
    )
    assert len(chunked.all_tensors()) == 2 * len(slice_batch(sample, 0).all_tensors())
    moved = chunked.to("cpu")
    assert isinstance(moved, ChunkedBatch)
    assert len(moved.chunks) == 2
