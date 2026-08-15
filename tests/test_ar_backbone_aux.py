"""GemmaARDecoder aux path (suffix format 5, headerless values).

Builds on test_ar_backbone's tiny fixture family (256-vocab text-only
model, fixture FAST codec at the vocabulary tail). Covers: the
range-split suffix embedding (text ids through the frozen tables, block
ids through the patch — and bitwise identity of the all-block path),
the componentized full-vocab CE (aux-off single-call equality; aux-on
weighting arithmetic), config round-trip with the aux section, and the
request-scaffolded decode (value phases budgeted per field, terminator
fallback counted, BOA forced, always a grammar-valid action chunk).
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

import bijou.modelling.decoders.ar_suffix
from bijou.loading import ar_backbone_config_to_dict, parse_decoder_config
from bijou.modelling.aux_text import (
    AUX_TEMPLATE_VERSION,
    GENERATION_OPENER,
    VALUE_BUDGETS,
    AuxDecodeConfig,
    AuxField,
    assemble_suffix,
    build_aux_runtime,
)
from bijou.modelling.decoders.ar_gemma import GemmaARDecoder
from bijou.modelling.decoders.ar_suffix import ar_backbone_loss, ar_backbone_losses
from bijou.modelling.gemma4.model import Gemma4Model
from bijou.modelling.nn import AttentionBackend


def aux_config() -> AuxDecodeConfig:
    return AuxDecodeConfig(
        template_version=AUX_TEMPLATE_VERSION,
        fields=(AuxField.SUBGOAL, AuxField.HOLDING, AuxField.PROGRESS),
        prompt_hash="9b796de",
        judge_model="claude-opus-4-8",
    )


def build_with_aux() -> tuple[Gemma4Model, GemmaARDecoder]:
    """An aux-CAPABLE decoder: aux rides the config (non-empty generate
    is gated on it), runtime + loss weight attached."""
    loaded = codec()
    torch.manual_seed(0)
    backbone = Gemma4Model(gemma_config(), attn_backend=AttentionBackend.EAGER)
    backbone.eval()
    backbone.requires_grad_(False)
    decoder = GemmaARDecoder(
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


def test_mixed_suffix_embeds_route_by_id_range() -> None:
    backbone, decoder, _ = build()
    text = backbone.language_model
    text_ids = torch.tensor([[ord("s"), ord(":")]])
    block_ids = torch.tensor([[3, 5]])
    mixed = torch.cat([text_ids, block_ids + decoder.config.block_base], dim=1)
    embeds, ple = decoder._suffix_inputs_backbone_ids(backbone, mixed)
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
    # Hand-assembled mixed suffix: headerless value lines (holding then
    # progress, the request order) before the actions.
    aux_ids = [[ord(c) for c in "yes\n30%\n"], []]
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
    # Row 0 carries exactly len("yes\n30%\n") aux positions; row 1
    # none — the count IS the labeled-token count, and the total applies
    # the position-weighted mean.
    assert int(aux_count) == len(aux_ids[0])
    assert torch.allclose(total, action + 0.5 * (aux_sum / aux_count))


def test_request_decode_is_budgeted_and_always_yields_chunks() -> None:
    """The request-scaffolded decode: per-field value phases bounded by
    VALUE_BUDGETS (block ids masked out), then BOA forced and a full
    grammar-valid chunk per row — regardless of whether the (random)
    model deigns to emit terminators."""
    backbone, decoder = build_with_aux()
    loaded = codec()
    sample = batch(loaded)
    memory = encode_memory(backbone)
    request = (AuxField.SUBGOAL, AuxField.HOLDING)
    prediction = decoder.predict_chunk(
        backbone,
        memory,
        sample,
        generate=request,
    )
    generations = prediction.generations
    assert generations is not None
    assert prediction.actions.shape == (BATCH, loaded.time_horizon, loaded.action_dim)
    assert bool(torch.isfinite(prediction.actions).all())
    budget = sum(VALUE_BUDGETS[f] for f in request)
    for generation in generations:
        # Values decode as text (char stub): never block ids, bounded;
        # holding is candidate-constrained to yes/no.
        assert generation.holding is not None
        raw_values = [generation.subgoal or "", "yes" if generation.holding else "no"]
        assert sum(len(v) for v in raw_values) <= budget
        assert all(ord(c) < decoder.config.block_base for v in raw_values for c in v)
        # Display text re-attaches field names for reports.
        assert "holding: " in generation.text


def test_zero_budget_forces_terminator_and_counts_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Patch the DECODER module's binding (predict_chunk reads its own
    # global, not aux_text's). SUBGOAL is a free-text field — budgeted;
    # constrained fields (holding) never consult the budget (their
    # candidate + terminator are forced).
    monkeypatch.setattr(
        bijou.modelling.decoders.ar_suffix,
        "VALUE_BUDGETS",
        dict.fromkeys(AuxField, 0),
    )
    backbone, decoder = build_with_aux()
    loaded = codec()
    prediction = decoder.predict_chunk(
        backbone,
        encode_memory(backbone),
        batch(loaded),
        generate=(AuxField.SUBGOAL,),
    )
    generations = prediction.generations
    assert generations is not None
    assert prediction.actions.shape[1] == loaded.time_horizon
    # Budget 0: no value text possible; the terminator was forced on
    # every row (a random model never argmaxes \n against 250 text ids)
    # and the empty line parses as a MISSING subgoal.
    assert all(generation.subgoal is None for generation in generations)
    assert decoder.fallback_count == BATCH


def test_construction_requires_tokenizer() -> None:
    loaded = codec()
    with pytest.raises(ValueError, match="text tokenizer"):
        GemmaARDecoder(
            decoder_config(loaded),
            gemma_config().text,
            loaded,
            tokenizer=None,
        )


def test_out_of_order_generate_is_rejected() -> None:
    backbone, decoder = build_with_aux()
    with pytest.raises(ValueError, match="template order"):
        decoder.predict_chunk(
            backbone,
            encode_memory(backbone),
            batch(codec()),
            generate=(AuxField.HOLDING, AuxField.SUBGOAL),
        )


def test_mixed_batch_supervises_aux_per_row() -> None:
    """A labeled row trains value lines then BOA, an unlabeled row BOA
    directly — the loss reads each row's suffix as assembled (this is
    also how request dropout lands on [generate|actions])."""
    backbone, decoder = build_with_aux()
    loaded = codec()
    sample = batch(loaded)
    tokens = sample.action_tokens
    assert tokens is not None
    aux_ids = [[ord(c) for c in "no\n"], []]
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
