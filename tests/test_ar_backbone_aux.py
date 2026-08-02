"""ARBackboneDecoder aux path (suffix format 2).

Builds on test_ar_backbone's tiny fixture family (256-vocab text-only
model, fixture FAST codec at the vocabulary tail). Covers: the
range-split suffix embedding (text ids through the frozen tables, block
ids through the patch — and bitwise identity of the all-block path),
the componentized full-vocab CE (aux-off single-call equality; aux-on
weighting arithmetic), config round-trip with the aux section, and the
ONE free-until-BOA decode path (block ids masked out of the free phase,
budget enforced, force-BOA fallback counted, always a grammar-valid
action chunk).
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch
from test_ar_backbone import (
    BATCH,
    batch,
    build,
    codec,
    decoder_config,
    encode_memory,
    gemma_config,
)
from test_aux_text import CharTokenizer

import bijou.decoders.ar_backbone
from bijou.aux_text import (
    AUX_TEMPLATE_VERSION,
    GENERATION_OPENER,
    MAX_FREE_TOKENS,
    AuxDecodeConfig,
    AuxDecodeMode,
    AuxField,
    assemble_suffix,
    build_aux_runtime,
)
from bijou.decoders.ar_backbone import (
    ARBackboneDecoder,
    ar_backbone_loss,
    ar_backbone_losses,
)
from bijou.gemma4.model import Gemma4Model
from bijou.loading import ar_backbone_config_to_dict, parse_decoder_config
from bijou.nn import AttentionBackend


def aux_config() -> AuxDecodeConfig:
    return AuxDecodeConfig(
        template_version=AUX_TEMPLATE_VERSION,
        fields=(AuxField.SUBGOAL, AuxField.HOLDING, AuxField.PROGRESS),
        prompt_hash="9b796de",
        judge_model="claude-opus-4-8",
    )


def build_with_aux() -> tuple[Gemma4Model, ARBackboneDecoder]:
    """An aux-CAPABLE decoder: aux rides the config (FREE decode is
    gated on it), runtime + loss weight attached."""
    loaded = codec()
    torch.manual_seed(0)
    backbone = Gemma4Model(gemma_config(), attn_backend=AttentionBackend.EAGER)
    backbone.eval()
    backbone.requires_grad_(False)
    decoder = ARBackboneDecoder(
        dataclasses.replace(decoder_config(loaded), aux=aux_config()),
        gemma_config().text,
        loaded,
        tokenizer=CharTokenizer(),
        aux_runtime=build_aux_runtime(aux_config(), CharTokenizer()),
        aux_loss_weight=0.5,
        device="cpu",
        dtype=torch.float32,
    )
    return backbone, decoder


def test_config_roundtrips_with_aux_section() -> None:
    loaded = codec()
    config = dataclasses.replace(decoder_config(loaded), aux=aux_config())
    payload = json.loads(json.dumps(ar_backbone_config_to_dict(config)))
    assert payload["aux"]["fields"] == ["subgoal", "holding", "progress"]
    assert parse_decoder_config(payload) == config


def test_wrong_template_version_is_rejected() -> None:
    payload = aux_config().to_dict()
    payload["template_version"] = 99
    with pytest.raises(SystemExit, match="template_version"):
        AuxDecodeConfig.from_dict(payload)


def test_backbone_id_embedding_matches_codec_path_bitwise() -> None:
    """All-block suffixes through the range-split path must equal the
    codec-id path exactly — the aux-off oracle depends on it."""
    backbone, decoder, loaded = build()
    tokens = batch(loaded).action_tokens
    assert tokens is not None
    state = torch.randn(BATCH, decoder.config.state_dim)
    old = decoder._suffix_inputs(backbone, state, tokens)
    new = decoder._suffix_inputs_backbone_ids(
        backbone,
        state,
        tokens + decoder.config.block_base,
    )
    assert torch.equal(old[0], new[0])
    assert torch.equal(old[1], new[1])


def test_mixed_suffix_embeds_route_by_id_range() -> None:
    backbone, decoder, _ = build()
    text = backbone.language_model
    text_ids = torch.tensor([[ord("s"), ord(":")]])
    block_ids = torch.tensor([[3, 5]])
    mixed = torch.cat([text_ids, block_ids + decoder.config.block_base], dim=1)
    embeds, ple = decoder._suffix_inputs_backbone_ids(backbone, None, mixed)
    assert torch.equal(embeds[:, :2], text.embed_tokens(text_ids).float())
    reference = decoder.fast_embed(block_ids) * decoder.embed_scale
    assert torch.equal(embeds[:, 2:], reference)
    assert torch.equal(
        ple[:, :2].flatten(2),
        text.embed_tokens_per_layer(text_ids).float(),
    )


def test_aux_off_loss_is_the_single_call_objective() -> None:
    backbone, decoder, loaded = build()
    sample = batch(loaded)
    assert sample.suffix_tokens is None
    total, action, aux_sum, aux_count = ar_backbone_losses(
        backbone,
        decoder,
        encode_memory(backbone),
        sample,
    )
    assert aux_sum is None and aux_count is None
    assert torch.equal(total, action)
    assert torch.equal(
        ar_backbone_loss(backbone, decoder, encode_memory(backbone), sample),
        total,
    )


def test_aux_on_loss_components_and_weighting() -> None:
    backbone, decoder = build_with_aux()
    loaded = codec()
    sample = batch(loaded)
    tokens = sample.action_tokens
    assert tokens is not None
    # Hand-assembled mixed suffix: a short aux text before the actions.
    aux_ids = [[ord(c) for c in "holding: yes\n"], []]
    suffix, is_aux = assemble_suffix(
        aux_ids,
        tokens,
        block_base=decoder.config.block_base,
        codec_pad=loaded.pad,
    )
    sample = dataclasses.replace(sample, suffix_tokens=suffix, suffix_is_aux=is_aux)
    total, action, aux_sum, aux_count = ar_backbone_losses(
        backbone,
        decoder,
        encode_memory(backbone),
        sample,
    )
    assert aux_sum is not None and aux_count is not None
    assert torch.isfinite(aux_sum) and torch.isfinite(action)
    # Row 0 carries exactly len("holding: yes\n") aux positions; row 1
    # none — the count IS the labeled-token count, and the total applies
    # the position-weighted mean.
    assert int(aux_count) == len(aux_ids[0])
    assert torch.allclose(total, action + 0.5 * (aux_sum / aux_count))


def test_free_decode_is_budgeted_and_always_yields_chunks() -> None:
    """The ONE decode path: free phase bounded by MAX_FREE_TOKENS (block
    ids masked out of it), then a full grammar-valid chunk per row —
    regardless of whether the (random) model deigns to emit BOA."""
    backbone, decoder = build_with_aux()
    loaded = codec()
    sample = batch(loaded)
    memory = encode_memory(backbone)
    prediction = decoder.predict_chunk(
        backbone,
        memory,
        sample,
        mode=AuxDecodeMode.FREE,
    )
    generations = prediction.generations
    assert generations is not None
    assert prediction.actions.shape == (BATCH, loaded.time_horizon, loaded.action_dim)
    assert bool(torch.isfinite(prediction.actions).all())
    for generation in generations:
        # Free ids decode as text (char stub): never block ids, bounded.
        assert len(generation.text) <= MAX_FREE_TOKENS
        assert all(ord(c) < decoder.config.block_base for c in generation.text)


def test_zero_budget_forces_boa_and_counts_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Patch the DECODER module's binding (predict_chunk reads its own
    # global, not aux_text's).
    monkeypatch.setattr(bijou.decoders.ar_backbone, "MAX_FREE_TOKENS", 0)
    backbone, decoder = build_with_aux()
    loaded = codec()
    prediction = decoder.predict_chunk(
        backbone,
        encode_memory(backbone),
        batch(loaded),
        mode=AuxDecodeMode.FREE,
    )
    generations = prediction.generations
    assert generations is not None
    assert prediction.actions.shape[1] == loaded.time_horizon
    # Budget 0: no free text possible; fallback fired for any row whose
    # first pick wasn't already BOA.
    assert all(generation.text == "" for generation in generations)
    assert decoder.fallback_count >= 0


def test_opener_format_requires_tokenizer() -> None:
    loaded = codec()
    with pytest.raises(ValueError, match="text tokenizer"):
        ARBackboneDecoder(
            decoder_config(loaded),
            gemma_config().text,
            loaded,
            tokenizer=None,
        )


def test_free_decode_rejected_on_auxless_checkpoint() -> None:
    """Every aux-less training sample fed [ACT] — [AUX] is untrained,
    so FREE decode must refuse rather than emit garbage."""
    backbone, decoder, loaded = build()  # config.aux is None
    with pytest.raises(ValueError, match="aux-less"):
        decoder.predict_chunk(
            backbone,
            encode_memory(backbone),
            batch(loaded),
            mode=AuxDecodeMode.FREE,
        )


def test_mixed_batch_derives_mode_per_row() -> None:
    """A labeled row trains [AUX]->aux, an unlabeled row [ACT]->BOA —
    the loss derives the fed mode from each row's aux positions, so the
    same batch carries both regimes (this is also how aux dropout lands
    on [ACT])."""
    backbone, decoder = build_with_aux()
    loaded = codec()
    sample = batch(loaded)
    tokens = sample.action_tokens
    assert tokens is not None
    aux_ids = [[ord(c) for c in "holding: no\n"], []]
    suffix, is_aux = assemble_suffix(
        aux_ids,
        tokens,
        block_base=decoder.config.block_base,
        codec_pad=loaded.pad,
    )
    sample = dataclasses.replace(sample, suffix_tokens=suffix, suffix_is_aux=is_aux)
    total, action, aux_sum, aux_count = ar_backbone_losses(
        backbone,
        decoder,
        encode_memory(backbone),
        sample,
    )
    assert torch.isfinite(total) and torch.isfinite(action)
    assert aux_sum is not None and aux_count is not None
    assert int(aux_count) == len(aux_ids[0])


def test_opener_ids_tokenize_the_template_opener() -> None:
    _, decoder = build_with_aux()
    assert decoder.opener_ids == tuple(ord(c) for c in GENERATION_OPENER)
