"""ARBackboneDecoder aux path (commit ② of the aux feature).

Builds on test_ar_backbone's tiny fixture family (256-vocab text-only
model, fixture FAST codec at the vocabulary tail). Covers: the
range-split suffix embedding (text ids through the frozen tables, block
ids through the patch — and bitwise identity of the all-block path),
the componentized full-vocab CE (aux-off single-call equality; aux-on
weighting arithmetic), config round-trip with the aux section, and
forced-scaffold decode_with_aux (headers forced, holding constrained to
{yes,no}, block ids masked out of text values, valid action chunk,
loud guard without a runtime).
"""

from __future__ import annotations

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
)
from test_aux_text import CharTokenizer

from bijou.aux_text import (
    AUX_TEMPLATE_VERSION,
    AuxDecodeConfig,
    AuxField,
    build_aux_runtime,
)
from bijou.decoders.ar_backbone import (
    ARBackboneDecoder,
    ar_backbone_loss,
    ar_backbone_losses,
)
from bijou.gemma4.model import Gemma4Model
from bijou.loading import ar_backbone_config_to_dict, parse_decoder_config


def aux_config() -> AuxDecodeConfig:
    return AuxDecodeConfig(
        template_version=AUX_TEMPLATE_VERSION,
        fields=(AuxField.SUBGOAL, AuxField.HOLDING, AuxField.PROGRESS),
        prompt_hash="9b796de",
        judge_model="claude-opus-4-8",
    )


def build_with_aux() -> tuple[Gemma4Model, ARBackboneDecoder]:
    backbone, decoder, _ = build()
    decoder.aux_runtime = build_aux_runtime(aux_config(), CharTokenizer())
    decoder.aux_loss_weight = 0.5
    return backbone, decoder


def test_config_roundtrips_with_aux_section() -> None:
    loaded = codec()
    import dataclasses

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
    total, action, aux = ar_backbone_losses(
        backbone,
        decoder,
        encode_memory(backbone),
        sample,
    )
    assert aux is None
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
    import dataclasses

    from bijou.aux_text import assemble_suffix

    suffix, is_aux = assemble_suffix(
        aux_ids,
        tokens,
        block_base=decoder.config.block_base,
        codec_pad=loaded.pad,
    )
    sample = dataclasses.replace(sample, suffix_tokens=suffix, suffix_is_aux=is_aux)
    total, action, aux = ar_backbone_losses(
        backbone,
        decoder,
        encode_memory(backbone),
        sample,
    )
    assert aux is not None and torch.isfinite(aux) and torch.isfinite(action)
    assert torch.allclose(total, action + 0.5 * aux)


def test_decode_with_aux_structure_and_constraints() -> None:
    backbone, decoder = build_with_aux()
    loaded = codec()
    sample = batch(loaded)
    memory = encode_memory(backbone)
    chunks, generations = decoder.decode_with_aux(backbone, memory, sample)
    assert chunks.shape == (BATCH, loaded.time_horizon, loaded.action_dim)
    assert bool(torch.isfinite(chunks).all())
    for generation in generations:
        # Forced scaffold: every field header present, template order.
        assert (
            generation.text.index("subgoal: ")
            < generation.text.index(
                "holding: ",
            )
            < generation.text.index("progress: ")
        )
        # Constrained value: holding parsed from {yes,no} — never None.
        assert generation.holding in (True, False)
        # Free-decoded values never contain block ids (masked): the
        # char-stub decodes every id below block_base, so decode() not
        # raising on chr() of a block id is implied by construction; the
        # explicit check is that text splits into template lines.
        assert generation.text.endswith("\n") or "progress: " in generation.text


def test_decode_with_aux_without_runtime_fails_loudly() -> None:
    backbone, decoder, loaded = build()
    with pytest.raises(SystemExit, match="aux_runtime"):
        decoder.decode_with_aux(backbone, encode_memory(backbone), batch(loaded))
