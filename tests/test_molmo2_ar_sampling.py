"""Molmo2 arm of the AR sampled-draws instrument (ideas #19), pure CPU.

The sampled-draws pre-registration names two arms — AR-100k (gemma4
trunk) and molmo2 AR 40k — and quotes its mechanics as oracle-pinned.
`tests/test_ar_sampling.py` pins them on the gemma trunk only; these
tests pin the trunk-specific halves on the Molmo2 fixture, because the
molmo2 arm runs the SAME shared suffix decode over a DIFFERENT cache
implementation (`Molmo2KVCache`): (1) the T→0 limit recovers the
molmo2 greedy decode exactly; (2) hot sampling is grammar-valid,
key-deterministic, and draw-distinct; (3) the by-reference
cache_snapshot/restore that shares one prefill across draws is exact
over the molmo2 cache — sound only while its `update()` rebinds and
never writes in place, pinned directly in (4); (5) the eval loop's
model-level entry (`ar_predict_sampled`) dispatches the molmo2 trunk
to the same decode. Pure-sampler math (mask compliance, batch
composition, RNG keying) is trunk-free and stays pinned in
test_ar_sampling. Fixture family: tests/test_molmo2_ar.

Also pinned here (2026-08-08): the narrated/[generate|…] pass on the
molmo2 concrete — (6) BijouPolicy.aux_fields gates on the family
scaffold ARSuffixDecoder, not the Gemma concrete (the isinstance that
silently dropped the narrated pass, and with it the report's
accuracy-by-field table, from the banked molmo2 40k panel); (7) the
decode entry the narrated pass calls (predict_chunk generate=…)
elicits value lines and a grammar-valid chunk on the molmo2 trunk."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_aux_text import CharTokenizer
from test_molmo2_ar import (
    BATCH,
    batch,
    build_decoder,
    build_encoder,
    codec,
    decoder_config,
    encode_memory,
    text_config,
    tiny_inputs,
)

from bijou.aux_text import (
    AUX_TEMPLATE_VERSION,
    AuxDecodeConfig,
    AuxField,
    build_aux_runtime,
)
from bijou.decoders.ar_backbone import ARSampling
from bijou.decoders.ar_molmo2 import Molmo2ARDecoder
from bijou.eval.policies import BijouPolicy, stable_sample_rng
from bijou.fast.codec import FastActionCodec
from bijou.model import BijouModel
from bijou.molmo2.cache import Molmo2KVCache
from bijou.molmo2.model import Molmo2Model, load_model
from bijou.molmo2.testing import write_tiny_text_checkpoint


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_tiny_text_checkpoint(
        tmp_path_factory.mktemp("molmo2-ar-sampling") / "tiny-molmo2",
    )


@pytest.fixture(scope="module")
def model(tiny_checkpoint: Path) -> Molmo2Model:
    return load_model(str(tiny_checkpoint), dtype=torch.float32)


def rngs(draw: int, *, seed: int = 0) -> tuple[np.random.Generator, ...]:
    """One keyed stream per fixture row, the way eval builds them."""
    return tuple(
        stable_sample_rng(seed, "fixture/repo", 0, row, draw) for row in range(BATCH)
    )


def test_low_temperature_limit_recovers_greedy(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    """Greedy rows and sampled rows share one decision point on the
    molmo2 trunk too — banked molmo2 greedy numbers stand next to
    sampled rows."""
    decoder, loaded = build_decoder(model)
    encoder = build_encoder(tiny_checkpoint)
    sample = batch(loaded, tiny_inputs())
    greedy = decoder.predict_chunk(model, encode_memory(encoder, model), sample)
    cold = decoder.predict_chunk(
        model,
        encode_memory(encoder, model),
        sample,
        sampling=ARSampling(temperature=1e-4, rngs=rngs(0)),
    )
    assert torch.equal(cold.actions, greedy.actions)


def test_sampled_decode_valid_deterministic_and_draw_distinct(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    """Hot sampling stays inside the grammar (full valid chunks decode)
    and is exactly reproducible under its keys; a different draw index
    yields a different chunk."""
    decoder, loaded = build_decoder(model)
    encoder = build_encoder(tiny_checkpoint)
    sample = batch(loaded, tiny_inputs())
    hot = lambda draw: decoder.predict_chunk(
        model,
        encode_memory(encoder, model),
        sample,
        sampling=ARSampling(temperature=2.0, rngs=rngs(draw)),
    )
    first, again, other = hot(0), hot(0), hot(1)
    assert first.actions.shape == (BATCH, loaded.time_horizon, loaded.action_dim)
    assert bool(torch.isfinite(first.actions).all())
    assert torch.equal(first.actions, again.actions)
    assert not torch.equal(first.actions, other.actions)


def test_cache_snapshot_restore_shares_one_prefill_exactly(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    """The instrument's per-draw loop over the MOLMO2 cache: sampled
    decode (consuming the cache), restore, same sampled decode again —
    both must equal a fresh-encode sampled decode bit-for-bit. Valid
    while Molmo2KVCache stays append-only."""
    decoder, loaded = build_decoder(model)
    encoder = build_encoder(tiny_checkpoint)
    sample = batch(loaded, tiny_inputs())
    memory = encode_memory(encoder, model)
    sampling = lambda: ARSampling(temperature=2.0, rngs=rngs(0))
    snapshot = decoder.cache_snapshot(memory)
    first = decoder.predict_chunk(model, memory, sample, sampling=sampling())
    decoder.cache_restore(memory, snapshot)
    second = decoder.predict_chunk(model, memory, sample, sampling=sampling())
    fresh = decoder.predict_chunk(
        model,
        encode_memory(encoder, model),
        sample,
        sampling=sampling(),
    )
    assert torch.equal(first.actions, second.actions)
    assert torch.equal(first.actions, fresh.actions)


def test_molmo2_cache_update_rebinds_never_writes() -> None:
    """cache_snapshot captures by REFERENCE; that is sound only while
    Molmo2KVCache.update rebinds layer tensors to new storage and never
    writes into stored ones. A future in-place (preallocated) cache
    must fail here, not corrupt draws 2..N silently."""
    cache = Molmo2KVCache(num_layers=1)
    generator = torch.Generator().manual_seed(4)
    prefill = torch.randn(1, 2, 5, 3, generator=generator)
    cache.update(0, prefill.clone(), prefill.clone())
    cache.advance(5)
    held_keys, held_values = cache.layers[0].keys, cache.layers[0].values
    assert held_keys is not None and held_values is not None
    frozen = held_keys.clone()
    suffix = torch.randn(1, 2, 1, 3, generator=generator)
    cache.update(0, suffix.clone(), suffix.clone())
    assert cache.layers[0].keys is not held_keys
    assert cache.layers[0].values is not held_values
    assert torch.equal(held_keys, frozen)


def test_ar_predict_sampled_dispatches_molmo2(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    """The eval loop's per-draw call reaches the molmo2 trunk: the
    model-level entry equals the decoder-level sampled decode."""
    decoder, loaded = build_decoder(model)
    encoder = build_encoder(tiny_checkpoint)
    bijou = BijouModel(backbone=model, encoder=encoder, decoder=decoder)
    sample = batch(loaded, tiny_inputs())
    via_model = bijou.ar_predict_sampled(
        bijou.encode(sample.encoder_inputs, with_grad=False),
        sample,
        sampling=ARSampling(temperature=2.0, rngs=rngs(0)),
    )
    direct = decoder.predict_chunk(
        model,
        encode_memory(encoder, model),
        sample,
        sampling=ARSampling(temperature=2.0, rngs=rngs(0)),
    )
    assert torch.equal(via_model.actions, direct.actions)


def test_bf16_mounted_trunk_decodes_with_fp32_patch(
    tiny_checkpoint: Path,
) -> None:
    """The production mount (loading.py casts the trunk bf16; the
    trainable FAST patch stays fp32): the suffix decode must not let
    torch.where promote mixed-dtype embeds to fp32 — the first
    attention matmul rejects fp32 hidden states against bf16 weights
    (the 40k endpoint eval crash, 2026-08-08)."""
    bf16 = load_model(str(tiny_checkpoint), dtype=torch.bfloat16)
    decoder, loaded = build_decoder(bf16)
    encoder = build_encoder(tiny_checkpoint)
    sample = batch(loaded, tiny_inputs())
    prediction = decoder.predict_chunk(
        bf16,
        encode_memory(encoder, bf16),
        sample,
    )
    assert prediction.actions.shape == (BATCH, loaded.time_horizon, loaded.action_dim)
    assert bool(torch.isfinite(prediction.actions).all())


def aux_config() -> AuxDecodeConfig:
    return AuxDecodeConfig(
        template_version=AUX_TEMPLATE_VERSION,
        fields=(AuxField.SUBGOAL, AuxField.HOLDING, AuxField.PROGRESS),
        prompt_hash="9b796de",
        judge_model="claude-opus-4-8",
    )


def build_aux_decoder(model: Molmo2Model) -> tuple[Molmo2ARDecoder, FastActionCodec]:
    """An aux-CAPABLE molmo2 decoder — build_decoder plus the trained
    aux record + runtime (test_ar_backbone_aux.build_with_aux's shape
    on this trunk)."""
    loaded = codec()
    torch.manual_seed(0)
    decoder = Molmo2ARDecoder(
        dataclasses.replace(decoder_config(loaded), aux=aux_config()),
        text_config().text,
        loaded,
        tokenizer=CharTokenizer(),
        aux_runtime=build_aux_runtime(aux_config(), CharTokenizer()),
        device="cpu",
        dtype=torch.float32,
    )
    decoder.init_tables_from_backbone(model)
    return decoder, loaded


def test_bijou_policy_aux_fields_sees_the_molmo2_concrete(
    model: Molmo2Model,
) -> None:
    """Regression (2026-08-08): BijouPolicy.aux_fields gated on the
    Gemma concrete (ARBackboneDecoder), so an aux-trained molmo2
    checkpoint reported NO trained fields and the eval's narrated pass
    — the sole producer of the report's accuracy-by-field table —
    silently never rode (the banked molmo2 40k panel is missing the
    table for exactly this reason). The gate is the family scaffold,
    ARSuffixDecoder."""
    aux_fields = BijouPolicy.aux_fields.fget
    assert aux_fields is not None
    decoder, _ = build_aux_decoder(model)
    stub = SimpleNamespace(model=SimpleNamespace(decoder=decoder))
    assert aux_fields(stub) == aux_config().fields
    # Aux-less molmo2 stays narration-free — same property, same trunk.
    bare, _ = build_decoder(model)
    stub = SimpleNamespace(model=SimpleNamespace(decoder=bare))
    assert aux_fields(stub) == ()


def test_narrated_generate_decodes_fields_on_the_molmo2_trunk(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    """The decode entry the narrated pass calls (predict_chunk with the
    checkpoint's full trained request — what the fixed eval runs at the
    60k endpoint panel): per-field value lines then a full grammar-valid
    chunk, on the molmo2 cache."""
    decoder, loaded = build_aux_decoder(model)
    encoder = build_encoder(tiny_checkpoint)
    request = aux_config().fields
    prediction = decoder.predict_chunk(
        model,
        encode_memory(encoder, model),
        batch(loaded, tiny_inputs()),
        generate=request,
    )
    generations = prediction.generations
    assert generations is not None
    assert len(generations) == BATCH
    assert prediction.actions.shape == (BATCH, loaded.time_horizon, loaded.action_dim)
    assert bool(torch.isfinite(prediction.actions).all())
    for generation in generations:
        # holding is candidate-constrained (yes/no) — always parses.
        assert generation.holding is not None
        assert "holding: " in generation.text
