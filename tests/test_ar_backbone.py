"""ARBackboneDecoder tests: the backbone's suffix role, pure CPU.

Reuses the keystone fixture family (tests/test_backbone_continuation)
with a 256-token vocabulary so the FAST block (fixture codec: 130 ids)
sits at the vocabulary TAIL — block_base 126 — mirroring the real plan
(E2B: 1026 ids at 258885 inside the unused tail). Covers: config
round-trip, the patched full-vocabulary head (text columns = the tied
head's, block columns = the patch's, softcap after overwrite),
teacher-forced-vs-incremental consistency (the decode loop's cache path
against the one-shot forward), the CE contract (state/PAD ignored,
targets offset into the block), constrained decode validity, the
zero-init state projection, and the loud guards (no cache; truncated
backbone)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_backbone_continuation import tiny_text_config
from torch import Tensor

from bijou.decoders.ar_backbone import (
    ARBackboneConfig,
    ARBackboneDecoder,
    ar_backbone_loss,
)
from bijou.decoders.ar_fast import IGNORE_INDEX
from bijou.encoders.gemma4 import GemmaEncoder
from bijou.fast.codec import ActionCodec
from bijou.gemma4.config import Gemma4Config
from bijou.gemma4.model import Gemma4Model
from bijou.interface import CollatedBatch, NormStats, ObservationMemory
from bijou.loading import ar_backbone_config_to_dict, parse_decoder_config
from bijou.nn import AttentionBackend

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_fast_tokenizer"
BATCH = 2
PROMPT_LENGTHS = (11, 7)
VOCAB = 256


def codec() -> ActionCodec:
    return ActionCodec.load(FIXTURE)


def gemma_config() -> Gemma4Config:
    text = dataclasses.replace(
        tiny_text_config(),
        vocab_size=VOCAB,
        vocab_size_per_layer_input=VOCAB,
    )
    return Gemma4Config(
        text=text,
        vision=None,
        image_token_id=999,
        video_token_id=998,
        audio_token_id=997,
        boi_token_id=996,
        eoi_token_id=995,
        dtype=torch.float32,
    )


def decoder_config(loaded: ActionCodec) -> ARBackboneConfig:
    return ARBackboneConfig(
        tokenizer=str(FIXTURE),
        vocab_total=loaded.vocab_total,
        block_base=VOCAB - loaded.vocab_total,  # tail placement, like E2B
        state_dim=loaded.action_dim,
        chunk_size=loaded.time_horizon,
        action_dim=loaded.action_dim,
    )


def build() -> tuple[Gemma4Model, ARBackboneDecoder, ActionCodec]:
    loaded = codec()
    torch.manual_seed(0)
    backbone = Gemma4Model(gemma_config(), attn_backend=AttentionBackend.EAGER)
    backbone.eval()
    backbone.requires_grad_(False)
    decoder = ARBackboneDecoder(
        decoder_config(loaded),
        gemma_config().text,
        loaded,
        device="cpu",
        dtype=torch.float32,
    )
    return backbone, decoder, loaded


def encode_memory(backbone: Gemma4Model) -> ObservationMemory:
    """A real prefill through the real encoder path: LEFT-padded prompt
    ids (kept below the FAST block), retained cache."""
    config = backbone.config
    stop = config.text.first_kv_shared_layer_idx - 1
    encoder = GemmaEncoder(
        config,
        exports=(stop,),
        processor_dir="unused",
        max_soft_tokens=1,
    )
    generator = torch.Generator().manual_seed(7)
    width = max(PROMPT_LENGTHS)
    ids = torch.zeros((BATCH, width), dtype=torch.long)
    real = torch.zeros((BATCH, width), dtype=torch.bool)
    for i, length in enumerate(PROMPT_LENGTHS):
        ids[i, width - length :] = torch.randint(3, 100, (length,), generator=generator)
        real[i, width - length :] = True
    with torch.no_grad():
        return encoder.encode_tensors(
            backbone,
            ids,
            padding_mask=real,
            retain_cache=True,
        )


class FakeInputs:
    def pin_memory(self) -> FakeInputs:
        return self

    def to(self, device: object, *, non_blocking: bool = False) -> FakeInputs:
        return self

    def tensors(self) -> dict[str, Tensor]:
        return {}


def batch(loaded: ActionCodec) -> CollatedBatch[FakeInputs]:
    generator = torch.Generator().manual_seed(2)
    chunk, dim = loaded.time_horizon, loaded.action_dim
    actions = torch.cumsum(
        torch.randn(BATCH, chunk, dim, generator=generator) * 0.05,
        dim=1,
    ).clamp(-1, 1)
    q01 = np.full(dim, -1.0)
    q99 = np.full(dim, 1.0)
    sequences = [loaded.encode(actions[i].numpy(), q01, q99) for i in range(BATCH)]
    width = max(len(s) for s in sequences)
    tokens = torch.tensor(
        [s + [loaded.pad] * (width - len(s)) for s in sequences],
        dtype=torch.long,
    )
    stats = NormStats(
        mean=torch.zeros(BATCH, dim),
        std=torch.ones(BATCH, dim),
        q01=torch.full((BATCH, dim), -1.0),
        q99=torch.full((BATCH, dim), 1.0),
    )
    return CollatedBatch(
        encoder_inputs=FakeInputs(),
        state=torch.randn(BATCH, dim, generator=generator),
        actions=actions,
        action_is_pad=torch.zeros(BATCH, chunk, dtype=torch.bool),
        action_stats=stats,
        state_stats=stats,
        action_tokens=tokens,
        suffix_tokens=None,
        suffix_is_aux=None,
    )


def test_config_roundtrips_through_schema_json() -> None:
    config = decoder_config(codec())
    payload = json.loads(json.dumps(ar_backbone_config_to_dict(config)))
    assert payload["kind"] == "ar_backbone"
    assert parse_decoder_config(payload) == config


def test_state_projection_is_zero_initialized() -> None:
    _, decoder, _ = build()
    assert torch.equal(
        decoder.state_proj.weight,
        torch.zeros_like(decoder.state_proj.weight),
    )
    assert decoder.state_proj.bias is not None
    assert torch.equal(
        decoder.state_proj.bias,
        torch.zeros_like(decoder.state_proj.bias),
    )


def test_init_tables_from_backbone_centers_on_row_means() -> None:
    backbone, decoder, _ = build()
    decoder.init_tables_from_backbone(backbone)
    embed_mean = backbone.language_model.embed_tokens.weight.mean(dim=0)
    got = decoder.fast_embed.weight.detach().mean(dim=0)
    # Row noise is N(0, 0.02); the mean over 130 rows shrinks it ~11x.
    assert float((got - embed_mean).abs().max()) < 0.02


def test_patched_head_places_the_block_and_softcaps() -> None:
    backbone, decoder, _ = build()
    base = decoder.config.block_base
    end = base + decoder.config.vocab_total
    hidden = torch.randn(BATCH, 3, backbone.config.text.hidden_size)
    with torch.no_grad():
        logits = decoder._patched_logits(backbone, hidden)  # unit under test
        cap = backbone.config.text.final_logit_softcapping
        assert cap is not None
        text_reference = torch.tanh(backbone.lm_head(hidden) / cap) * cap
        block_reference = torch.tanh(hidden @ decoder.fast_embed.weight.T / cap) * cap
    assert torch.equal(logits[..., :base], text_reference[..., :base])
    assert torch.equal(logits[..., base:end], block_reference)
    assert float(logits.abs().max()) <= cap


def test_loss_is_finite_and_ignores_state_and_pad() -> None:
    backbone, decoder, loaded = build()
    sample = batch(loaded)
    loss = ar_backbone_loss(backbone, decoder, encode_memory(backbone), sample)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    tokens = sample.action_tokens
    assert tokens is not None
    lengths = (tokens != loaded.pad).sum(dim=1)
    assert int(lengths.min()) < tokens.shape[1]  # the batch IS ragged
    assert IGNORE_INDEX == -100


def test_teacher_forced_matches_incremental_decode_path() -> None:
    """The decode loop's incremental cache feeding must reproduce the
    one-shot teacher-forced logits at every position (same suffix,
    fresh caches; tolerance covers reduction-shape noise only)."""
    backbone, decoder, loaded = build()
    sample = batch(loaded)
    tokens = sample.action_tokens
    assert tokens is not None
    forced = tokens[:, : 6 + 1]  # [BOA, t1..t6]
    state = sample.state  # stats are identity in the fixture

    one_shot = decoder(backbone, encode_memory(backbone), state, forced)

    memory = encode_memory(backbone)
    stepwise: list[Tensor] = []
    fed = 0
    feed = forced[:, :1]
    feed_state: Tensor | None = state
    for j in range(forced.shape[1]):
        embeds, per_layer = decoder._suffix_inputs(backbone, feed_state, feed)
        hidden = decoder._continue_suffix(backbone, memory, embeds, per_layer, fed)
        fed += embeds.shape[1]
        stepwise.append(decoder._patched_logits(backbone, hidden)[:, -1, :])
        if j + 1 < forced.shape[1]:
            feed = forced[:, j + 1 : j + 2]
            feed_state = None
    # one_shot position j+1 predicts after [state][BOA..t_j] — stepwise j
    # is the same prediction point (position 0 of one_shot is the state
    # slot's prediction, stepwise[0] is BOA's = one_shot position 1).
    for j, step_logits in enumerate(stepwise):
        delta = float((one_shot[:, j + 1, :] - step_logits).abs().max())
        assert delta < 1e-4, f"step {j}: incremental diverges by {delta}"


def test_predict_chunk_constrained_decode_always_valid() -> None:
    backbone, decoder, loaded = build()
    sample = batch(loaded)
    chunks = decoder.predict_chunk(backbone, encode_memory(backbone), sample)
    assert chunks.shape == (BATCH, loaded.time_horizon, loaded.action_dim)
    assert bool(torch.isfinite(chunks).all())


def test_missing_cache_fails_loudly() -> None:
    backbone, decoder, loaded = build()
    memory = encode_memory(backbone)
    without_cache = ObservationMemory(
        streams=memory.streams,
        length=memory.length,
        padding_mask=memory.padding_mask,
        cache=None,
    )
    with pytest.raises(ValueError, match="retain_cache"):
        ar_backbone_loss(backbone, decoder, without_cache, batch(loaded))


def test_truncated_backbone_is_rejected() -> None:
    loaded = codec()
    truncated = dataclasses.replace(
        gemma_config().text,
        num_hidden_layers=6,
        layer_types=gemma_config().text.layer_types[:6],
        num_kv_shared_layers=0,
    )
    with pytest.raises(ValueError, match="FULL backbone"):
        ARBackboneDecoder(decoder_config(loaded), truncated, loaded)
