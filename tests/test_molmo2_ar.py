"""Molmo2 AR decoder arm — CPU oracles on the tiny fixture.

The gate set mirrors arm B's pre-launch oracles plus the Gemma AR
keystone: (1) prefill-then-continue ≡ monolithic multimodal forward
(the cache invariant the whole suffix role rests on, proven under left
padding), (2) the appended-head layout (frozen base columns, dtype-min
gap for the image specials, fresh FAST rows), (3) the frozen-original-
vocab split — gradients reach exactly the trainable set, (4) row-mean
table init, (5) teacher-forced ≡ incremental decode, (6) a full
request-scaffolded predict_chunk round trip, (7) schema round trip.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_aux_text import CharTokenizer
from test_molmo2_model import image_type_mask_of, tiny_ids, tiny_vision_inputs

from bijou.loading import decoder_schema_dict, parse_decoder_config
from bijou.modelling.aux_text import SUFFIX_FORMAT
from bijou.modelling.codecs import FastActionCodec
from bijou.modelling.decoders.ar_molmo2 import MOLMO2_GENERATION_OPENER, Molmo2ARDecoder
from bijou.modelling.decoders.ar_suffix import ARDecoderConfig, suffix_targets
from bijou.modelling.encoders.molmo2 import Molmo2Encoder, Molmo2Inputs, Molmo2Memory
from bijou.modelling.interface import CollatedBatch, NormStats
from bijou.modelling.molmo2.config import Molmo2Config
from bijou.modelling.molmo2.model import Molmo2Model, build_multimodal_mask, load_model
from bijou.modelling.molmo2.testing import tiny_config_json, write_tiny_text_checkpoint
from bijou.models.molmo2_ar import Molmo2ARVLA
from bijou.models.objectives import ARObjective
from bijou.models.serving import ARServing

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_fast_tokenizer"
BATCH = 2


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_tiny_text_checkpoint(
        tmp_path_factory.mktemp("molmo2-ar") / "tiny-molmo2",
    )


@pytest.fixture(scope="module")
def model(tiny_checkpoint: Path) -> Molmo2Model:
    return load_model(str(tiny_checkpoint), dtype=torch.float32)


def codec() -> FastActionCodec:
    return FastActionCodec.load(FIXTURE)


def text_config() -> Molmo2Config:
    return Molmo2Config.from_dict(tiny_config_json())


def decoder_config(loaded: FastActionCodec) -> ARDecoderConfig:
    text = text_config().text
    return ARDecoderConfig(
        tokenizer=str(FIXTURE),
        vocab_total=loaded.vocab_total,
        block_base=text.fast_block_base,  # 512 + 8 = 520: the extension anchor
        chunk_size=loaded.time_horizon,
        action_dim=loaded.action_dim,
        suffix_format=SUFFIX_FORMAT,
        aux=None,
    )


def build_decoder(model: Molmo2Model) -> tuple[Molmo2ARDecoder, FastActionCodec]:
    loaded = codec()
    torch.manual_seed(0)
    decoder = Molmo2ARDecoder(
        decoder_config(loaded),
        text_config().text,
        loaded,
        tokenizer=CharTokenizer(),
        device="cpu",
        dtype=torch.float32,
    )
    decoder.init_tables_from_backbone(model)
    return decoder, loaded


def build_encoder(checkpoint: Path) -> Molmo2Encoder:
    torch.manual_seed(1)
    return Molmo2Encoder(
        str(checkpoint),
        max_crops=1,
        state_dim=codec().action_dim,
        hidden_size=text_config().text.hidden_size,
    )


def tiny_inputs() -> Molmo2Inputs:
    """The WP4 fixture batch (row 1 left-padded) as collated inputs."""
    input_ids, attention_mask = tiny_ids()
    image = image_type_mask_of(input_ids, attention_mask)
    config = text_config()
    patch_counts = [int((row == config.image_patch_id).sum()) for row in input_ids]
    crops, pooled_idx = tiny_vision_inputs(patch_counts)
    torch.manual_seed(3)
    return Molmo2Inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_type_mask=image,
        crops=crops,
        pooled_patches_idx=pooled_idx,
        state=torch.randn(BATCH, codec().action_dim),
        state_slot=-3,
        has_padding=True,
    )


def encode_memory(
    encoder: Molmo2Encoder,
    model: Molmo2Model,
    *,
    with_grad: bool = False,
) -> Molmo2Memory:
    return encoder.encode(
        model,
        tiny_inputs(),
        with_grad=with_grad,
    )


def batch(loaded: FastActionCodec, inputs: Molmo2Inputs) -> CollatedBatch[Molmo2Inputs]:
    generator = torch.Generator().manual_seed(2)
    chunk, dim = loaded.time_horizon, loaded.action_dim
    actions = torch.cumsum(
        torch.randn(BATCH, chunk, dim, generator=generator) * 0.05,
        dim=1,
    ).clamp(-1, 1)
    q01 = torch.full((dim,), -1.0).numpy()
    q99 = torch.full((dim,), 1.0).numpy()
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
        encoder_inputs=inputs,
        state=inputs.state,
        actions=actions,
        action_is_pad=torch.zeros(BATCH, chunk, dtype=torch.bool),
        action_stats=stats,
        state_stats=stats,
        action_tokens=tokens,
        suffix_tokens=None,
        suffix_is_aux=None,
    )


def test_config_roundtrips_and_anchor_guard(model: Molmo2Model) -> None:
    import dataclasses
    import json

    config = decoder_config(codec())
    payload = json.loads(json.dumps(decoder_schema_dict_of(model)))
    assert payload["kind"] == "ar_backbone"
    assert parse_decoder_config(payload) == config
    # A block base off the extension anchor would shear every action
    # target off its embedding row — refused at construction.
    with pytest.raises(ValueError, match="extension anchor"):
        Molmo2ARDecoder(
            dataclasses.replace(config, block_base=512),
            text_config().text,
            codec(),
            tokenizer=CharTokenizer(),
        )


def decoder_schema_dict_of(model: Molmo2Model) -> dict[str, object]:
    decoder, _ = build_decoder(model)
    return decoder_schema_dict(decoder)


def test_appended_head_layout(model: Molmo2Model) -> None:
    """Frozen base columns exactly the shipped head's, dtype-min gap for
    the image specials, fresh FAST rows — no softcap (Qwen3 has none)."""
    decoder, _ = build_decoder(model)
    base_vocab = text_config().text.vocab_size
    block_base = decoder.config.block_base
    hidden = torch.randn(BATCH, 3, text_config().text.hidden_size)
    lm_head = model.text.lm_head
    assert lm_head is not None
    with torch.no_grad():
        logits = decoder._logits(model, hidden)
        text_reference = lm_head(hidden)
        block_reference = decoder.fast_head(hidden)
    assert logits.shape[-1] == block_base + decoder.config.vocab_total
    assert torch.equal(logits[..., :base_vocab], text_reference)
    assert torch.equal(logits[..., block_base:], block_reference)
    gap = logits[..., base_vocab:block_base]
    assert bool((gap == torch.finfo(logits.dtype).min).all())
    # The gap can never win an argmax.
    assert bool((logits.argmax(dim=-1) != base_vocab).all())


def test_init_tables_center_on_row_means(model: Molmo2Model) -> None:
    decoder, _ = build_decoder(model)
    lm_head = model.text.lm_head
    assert lm_head is not None
    embed_mean = model.text.transformer.wte.embedding.mean(dim=0)
    head_mean = lm_head.weight.mean(dim=0)
    got_embed = decoder.fast_embed.weight.detach().mean(dim=0)
    got_head = decoder.fast_head.weight.detach().mean(dim=0)
    # Row noise is N(0, 0.02); the mean over 130 rows shrinks it ~11x.
    assert float((got_embed - embed_mean).abs().max()) < 0.02
    assert float((got_head - head_mean).abs().max()) < 0.02


def test_prefill_continue_matches_monolithic(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    """The keystone: prompt prefill (multimodal mask, left padding, state
    splice) + suffix continuation through the cache must reproduce the
    single-shot forward over [prompt + suffix] with the suffix rows
    text-typed."""
    decoder, loaded = build_decoder(model)
    encoder = build_encoder(tiny_checkpoint)
    inputs = tiny_inputs()

    suffix = torch.tensor(
        [
            [7, decoder.config.block_base + loaded.boa, decoder.config.block_base + 3],
            [8, decoder.config.block_base + loaded.boa, decoder.config.block_base + 5],
        ],
        dtype=torch.long,
    )
    memory = encode_memory(encoder, model)
    with torch.no_grad():
        continued = decoder(model, memory, suffix)

    # Monolithic reference: the encoder's exact embedding path + the
    # decoder's suffix routing, one forward, extended multimodal mask.
    with torch.no_grad():
        prompt_embeds = model.build_input_embeddings(
            inputs.input_ids,
            crops=inputs.crops,
            pooled_patches_idx=inputs.pooled_patches_idx,
        )
        prompt_embeds[:, inputs.state_slot, :] = encoder.state_proj(inputs.state)
        is_text = (suffix < decoder.config.block_base)[..., None]
        suffix_embeds = torch.where(
            is_text,
            model.text.transformer.wte(
                suffix.clamp(max=decoder.config.block_base - 1),
            ),
            decoder.fast_embed((suffix - decoder.config.block_base).clamp(min=0)),
        )
        full_embeds = torch.cat([prompt_embeds, suffix_embeds], dim=1)
        full_real = torch.cat(
            [
                inputs.attention_mask,
                torch.ones(BATCH, suffix.shape[1], dtype=torch.long),
            ],
            dim=1,
        )
        full_image = torch.cat(
            [
                inputs.image_type_mask,
                torch.zeros(BATCH, suffix.shape[1], dtype=torch.bool),
            ],
            dim=1,
        )
        positions = (full_real.cumsum(-1) - 1).clamp(min=0)
        mask = build_multimodal_mask(
            image_type_mask=full_image,
            padding_mask=full_real,
            dtype=full_embeds.dtype,
            device=full_embeds.device,
        )
        hidden = model.text.transformer(
            inputs_embeds=full_embeds,
            position_ids=positions,
            attention_mask=mask,
        )
        reference = decoder._logits(model, hidden)[:, -suffix.shape[1] :]

    torch.testing.assert_close(continued, reference, rtol=1e-4, atol=1e-4)


def test_teacher_forced_matches_incremental(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    decoder, loaded = build_decoder(model)
    encoder = build_encoder(tiny_checkpoint)
    sample = batch(loaded, tiny_inputs())
    tokens = sample.action_tokens
    assert tokens is not None
    forced = tokens[:, :5] + decoder.config.block_base

    with torch.no_grad():
        one_shot = decoder(model, encode_memory(encoder, model), forced)
        memory = encode_memory(encoder, model)
        fed = 0
        for j in range(forced.shape[1]):
            step_logits, fed = decoder._step(model, memory, forced[:, j : j + 1], fed)
            torch.testing.assert_close(
                step_logits,
                one_shot[:, j].float(),
                rtol=1e-4,
                atol=1e-4,
            )


def test_loss_targets_and_frozen_split(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    """The CE contract (opener constants IGNOREd except the last, PAD
    ignored, block targets offset) and the 18:1xZ freezing split:
    gradients reach fast_embed + fast_head + every decoder block + ln_f
    + state_proj and DO NOT reach wte or the shipped lm_head."""
    decoder, loaded = build_decoder(model)
    encoder = build_encoder(tiny_checkpoint)
    sample = batch(loaded, tiny_inputs())

    full, targets, is_aux = suffix_targets(decoder, sample)
    assert is_aux is None
    opener = decoder.opener_ids
    assert tuple(full[0, : len(opener)].tolist()) == opener
    assert (targets[:, : len(opener) - 1] == -100).all()
    pad_backbone = decoder.config.block_base + loaded.pad
    assert not bool((targets == pad_backbone).any())

    # The trainable set: decoder tables (already trainable) + the text
    # group + prompt state_proj (train.py's unfreeze path) — exercised
    # through the family's own forward (live-trunk encode + suffix CE).
    vla = Molmo2ARVLA(
        model,
        encoder,
        decoder,
        objective=ARObjective(aux_loss_weight=1.0),
        serving=ARServing(),
    )
    for parameter in encoder.param_groups(model)["text"]:
        parameter.requires_grad_(True)
    loss = vla(sample, counts=vla.loss_counts(sample)).objective
    assert torch.isfinite(loss)
    loss.backward()

    assert decoder.fast_embed.weight.grad is not None
    assert decoder.fast_head.weight.grad is not None
    assert float(decoder.fast_head.weight.grad.abs().sum()) > 0
    assert encoder.state_proj.weight.grad is not None
    transformer = model.text.transformer
    for i, block in enumerate(transformer.blocks):
        for name, parameter in block.named_parameters():
            assert parameter.grad is not None, f"block {i} {name} missing grad"
    for name, parameter in transformer.ln_f.named_parameters():
        assert parameter.grad is not None, f"ln_f {name} missing grad"
    # The frozen original vocabulary: both sides.
    assert transformer.wte.embedding.grad is None
    assert transformer.wte.new_embedding.grad is None
    lm_head = model.text.lm_head
    assert lm_head is not None
    assert lm_head.weight.grad is None
    # Reset for other module-scope tests.
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)


def test_left_pad_invariance_of_the_suffix_loss(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    """Row 1 padded vs solo-unpadded: the suffix CE must not see the
    padding (neither through the prompt cache nor the continuation
    mask)."""
    decoder, loaded = build_decoder(model)
    encoder = build_encoder(tiny_checkpoint)
    inputs = tiny_inputs()
    sample = batch(loaded, inputs)
    suffix = torch.tensor(
        [[decoder.config.block_base + loaded.boa, decoder.config.block_base + 2]],
    )

    real = inputs.attention_mask[1].bool()
    config = text_config()
    patch_counts = [int((inputs.input_ids[1] == config.image_patch_id).sum())]
    padded_inputs = Molmo2Inputs(
        input_ids=inputs.input_ids[1:],
        attention_mask=inputs.attention_mask[1:],
        image_type_mask=inputs.image_type_mask[1:],
        crops=inputs.crops[1:],
        pooled_patches_idx=inputs.pooled_patches_idx[1:],
        state=inputs.state[1:],
        state_slot=-3,
        has_padding=True,
    )
    solo_inputs = Molmo2Inputs(
        input_ids=inputs.input_ids[1:, real],
        attention_mask=inputs.attention_mask[1:, real],
        image_type_mask=inputs.image_type_mask[1:, real],
        crops=inputs.crops[1:],
        pooled_patches_idx=inputs.pooled_patches_idx[1:],
        state=inputs.state[1:],
        state_slot=-3,
        has_padding=False,
    )
    del sample, patch_counts
    with torch.no_grad():
        padded_memory = encoder.encode(
            model,
            padded_inputs,
            with_grad=False,
        )
        padded_logits = decoder(model, padded_memory, suffix)
        solo_memory = encoder.encode(
            model,
            solo_inputs,
            with_grad=False,
        )
        solo_logits = decoder(model, solo_memory, suffix)
    torch.testing.assert_close(padded_logits, solo_logits, rtol=1e-4, atol=1e-4)


def test_predict_chunk_decodes_a_valid_chunk(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    decoder, loaded = build_decoder(model)
    encoder = build_encoder(tiny_checkpoint)
    sample = batch(loaded, tiny_inputs())
    memory = encode_memory(encoder, model)
    prediction = decoder.predict_chunk(model, memory, sample)
    assert prediction.actions.shape == (
        BATCH,
        loaded.time_horizon,
        loaded.action_dim,
    )
    assert torch.isfinite(prediction.actions).all()
    assert prediction.generations is not None
    assert len(prediction.generations) == BATCH
    assert prediction.noise is None


def test_generation_opener_is_chatml() -> None:
    assert MOLMO2_GENERATION_OPENER == "<|im_start|>assistant\n"
    tokenizer = CharTokenizer()
    ids = tokenizer.encode(MOLMO2_GENERATION_OPENER, add_special_tokens=False)
    assert ids[-1] == ord("\n")
