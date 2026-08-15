"""MolmoAct2 AR decoder arm — CPU oracles on a tiny widened-vocab trunk
(docs/molmoact2-retirement.md phase 2).

What this suite pins, mirroring the Molmo2 arm's gate set where it
applies and the release facts where it doesn't:

1. construction guards — format 6 only, aux None only, the codec's
   below-block special offsets, base-matrix coverage, and the trunk-
   tokenizer anchor verification (the id-space seam made loud);
2. the EMPTY opener + suffix_targets: BOA (= ``<action_start>``) is fed
   never predicted, position 0 predicts the first bin, ``<action_end>``
   (the pad offset) is never a CE target — and the max(…, 0) clamp
   keeps the empty-opener mask from eating every target but the last;
3. prefill-then-continue ≡ monolithic forward through the real
   Molmo2KVCache (the cache invariant, text-only prompt);
4. teacher-forced ≡ incremental decode;
5. grammar-masked predict_chunk: budget consumed exactly, only
   reachable bins emitted, capture surface block-relative [B, 2048]
   with BACKBONE-id chosen, decode(bins) round-trips to the returned
   chunk, and ``teacher_forced_block_logits`` runs on the parameterless
   decoder (the trunk-device fix);
6. keyed sampling determinism (the GRPO rollout draw);
7. schema round trip through the shared ar_backbone section.

The trunk is random (tiny), so decodes exercise MECHANICS, not quality;
byte-parity against the release runs on the box fixture
(tests/fixtures/molmoact2_discrete), not here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from bijou.convert_legacy import CheckpointMetadata, convert
from bijou.data import DatasetStats
from bijou.fast.molmoact2 import MolmoAct2FastTokenizer
from bijou.loading import (
    BackboneConfig,
    BackboneDepth,
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
    ar_backbone_config_to_dict,
    build_molmoact2_ar_decoder,
    decoder_schema_dict,
    load_vla,
    molmoact2_ar_config_from_flow_section,
    parse_decoder_config,
)
from bijou.modelling.codecs import FastActionCodec, MolmoAct2ActionCodec
from bijou.modelling.decoders.ar_molmoact2 import MolmoAct2ARDecoder
from bijou.modelling.decoders.ar_suffix import (
    IGNORE_INDEX,
    MOLMOACT2_SUFFIX_FORMAT,
    ARDecoderConfig,
    suffix_targets,
)
from bijou.modelling.encoders.molmoact2 import MOLMOACT2_PROMPT_FORMAT, MolmoAct2Encoder
from bijou.modelling.interface import (
    ActionCaptureStep,
    ARSampling,
    CollatedBatch,
    NormStats,
    ObservationMemory,
)
from bijou.modelling.molmo2.cache import Molmo2KVCache
from bijou.modelling.molmo2.config import Molmo2Config
from bijou.modelling.molmo2.model import Molmo2Model, build_multimodal_mask, load_model
from bijou.modelling.molmo2.testing import tiny_config_json, write_tiny_text_checkpoint
from bijou.modelling.molmo2.tokenizer import Molmo2TextTokenizer
from bijou.models.molmoact2_ar import MolmoAct2ARVLA

FAST_FIXTURE = Path(__file__).parent / "fixtures" / "molmoact2_fast_tokenizer"
BATCH = 2
T, D = 30, 6
# The tiny layout mirrors the release's ANCHORING (specials directly
# below the block, block in-base) at a vocabulary the tiny trunk can
# host: <action_start>=138, <action_end>=139, <action_0>=140 — the
# release's 151932/151933/151934 shifted, same arithmetic.
BLOCK_BASE = 140
VOCAB_SIZE = 2200  # ≥ BLOCK_BASE + 2048 = 2188


def molmoact2_config_json() -> dict[str, object]:
    config = json.loads(json.dumps(tiny_config_json()))  # deep copy
    config["text_config"]["vocab_size"] = VOCAB_SIZE
    return config


def write_tiny_molmoact2_trunk(directory: Path) -> Path:
    """A loadable tiny MolmoAct2-flavored trunk: widened vocab hosting
    the real 2048-wide action block IN-BASE at the release anchoring
    (specials directly below <action_0>), plus a WordLevel tokenizer
    carrying the anchor tokens. Plain function so sibling suites
    (test_molmoact2_objectives) build their own copies."""
    written = write_tiny_text_checkpoint(
        directory,
        config_json=molmoact2_config_json(),
    )
    vocab = {
        "<unk>": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<action_output>": 3,
        "hello": 4,
        "<action_start>": BLOCK_BASE - 2,
        "<action_end>": BLOCK_BASE - 1,
        "<action_0>": BLOCK_BASE,
    }
    Tokenizer(WordLevel(vocab, unk_token="<unk>")).save(
        str(written / "tokenizer.json"),
    )
    return written


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_tiny_molmoact2_trunk(
        tmp_path_factory.mktemp("molmoact2-ar") / "tiny-molmoact2",
    )


@pytest.fixture(scope="module")
def model(tiny_checkpoint: Path) -> Molmo2Model:
    return load_model(str(tiny_checkpoint), dtype=torch.float32)


def codec() -> MolmoAct2ActionCodec:
    return MolmoAct2ActionCodec(
        MolmoAct2FastTokenizer.load(FAST_FIXTURE),
        time_horizon=T,
        action_dim=D,
    )


def text_config() -> Molmo2Config:
    return Molmo2Config.from_dict(molmoact2_config_json())


def decoder_config() -> ARDecoderConfig:
    return ARDecoderConfig(
        tokenizer=str(FAST_FIXTURE),
        vocab_total=2048,
        block_base=BLOCK_BASE,
        chunk_size=T,
        action_dim=D,
        suffix_format=MOLMOACT2_SUFFIX_FORMAT,
        aux=None,
    )


def build_decoder(tiny_checkpoint: Path) -> MolmoAct2ARDecoder:
    return MolmoAct2ARDecoder(
        decoder_config(),
        text_config().text,
        codec(),
        tokenizer=Molmo2TextTokenizer(str(tiny_checkpoint)),
    )


# Rig-flavored quantile rows: non-trivial, per-dimension distinct.
Q01 = np.array([-92.3, -104.1, -3.7, -88.0, -45.5, 2.1])
Q99 = np.array([88.9, 102.6, 178.2, 91.4, 47.0, 97.3])


class FakeInputs:
    """BatchInputs protocol filler — the suffix decoder never reads
    encoder inputs (the memory is encoded separately in these tests)."""

    def pin_memory(self) -> FakeInputs:
        return self

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> FakeInputs:
        return self

    def tensors(self) -> dict[str, torch.Tensor]:
        return {}


def encode_memory(
    model: Molmo2Model,
    *,
    padded: bool = False,
) -> ObservationMemory:
    """A text-only prefill with a retained cache — the decoder is
    prompt-agnostic (it continues a Molmo2KVCache); the MolmoAct2
    PROMPT's own bytes are the encoder suite's contract, not this
    one's. Row 1 is left-padded by 2 when ``padded``."""
    generator = torch.Generator().manual_seed(11)
    prompt = torch.randint(1, 100, (BATCH, 9), generator=generator)
    real = torch.ones(BATCH, 9, dtype=torch.long)
    if padded:
        prompt[1, :2] = 0
        real[1, :2] = 0
    transformer = model.text.transformer
    embeds = transformer.wte(prompt)
    mask = build_multimodal_mask(
        image_type_mask=torch.zeros(BATCH, 9, dtype=torch.bool),
        padding_mask=real if padded else None,
        dtype=embeds.dtype,
        device=embeds.device,
    )
    positions = (real.cumsum(-1) - 1).clamp(min=0) if padded else None
    cache = Molmo2KVCache(len(transformer.blocks))
    with torch.no_grad():
        transformer(
            inputs_embeds=embeds,
            position_ids=positions,
            attention_mask=mask,
            cache=cache,
        )
    return ObservationMemory(
        streams={},
        length=9,
        padding_mask=real if padded else None,
        cache=cache,
    )


def action_rows() -> tuple[torch.Tensor, list[list[int]]]:
    """[BATCH, T, D] raw chunks (seed-pinned, hole-free — encode would
    raise loudly otherwise) + their codec-relative token rows."""
    generator = torch.Generator().manual_seed(5)
    normalized = torch.cumsum(
        torch.randn(BATCH, T, D, generator=generator) * 0.04,
        dim=1,
    ).clamp(-0.9, 0.9)
    stats_q01 = torch.tensor(Q01, dtype=torch.float32)
    stats_q99 = torch.tensor(Q99, dtype=torch.float32)
    raw = (normalized + 1.0) * (stats_q99 - stats_q01) / 2.0 + stats_q01
    loaded = codec()
    sequences = [loaded.encode(raw[i].double().numpy(), Q01, Q99) for i in range(BATCH)]
    return raw, sequences


def batch(raw: torch.Tensor, sequences: list[list[int]]) -> CollatedBatch[FakeInputs]:
    width = max(len(s) for s in sequences)
    tokens = torch.tensor(
        [s + [codec().pad] * (width - len(s)) for s in sequences],
        dtype=torch.long,
    )
    stats = NormStats(
        mean=torch.zeros(BATCH, D),
        std=torch.ones(BATCH, D),
        q01=torch.tensor(Q01, dtype=torch.float32).expand(BATCH, D).clone(),
        q99=torch.tensor(Q99, dtype=torch.float32).expand(BATCH, D).clone(),
    )
    return CollatedBatch(
        encoder_inputs=FakeInputs(),
        state=torch.zeros(BATCH, D),
        actions=raw.float(),
        action_is_pad=torch.zeros(BATCH, T, dtype=torch.bool),
        action_stats=stats,
        state_stats=stats,
        action_tokens=tokens,
        suffix_tokens=None,
        suffix_is_aux=None,
    )


def test_construction_guards(tiny_checkpoint: Path) -> None:
    config = decoder_config()
    text = text_config().text
    tokenizer = Molmo2TextTokenizer(str(tiny_checkpoint))
    with pytest.raises(ValueError, match="not the MolmoAct2 release emission"):
        MolmoAct2ARDecoder(
            dataclasses.replace(config, suffix_format=5),
            text,
            codec(),
            tokenizer=tokenizer,
        )
    # A fitted-family codec (specials INSIDE the block): geometry made
    # to pass the scaffold checks so the offsets guard itself fires.
    fitted = FastActionCodec.load(
        Path(__file__).parent / "fixtures" / "tiny_fast_tokenizer",
    )
    with pytest.raises(ValueError, match="below-block layout"):
        MolmoAct2ARDecoder(
            dataclasses.replace(
                config,
                vocab_total=fitted.vocab_total,
                chunk_size=fitted.time_horizon,
                action_dim=fitted.action_dim,
            ),
            text,
            fitted,
            tokenizer=tokenizer,
        )
    # A block base off the tokenizer's anchors: the arithmetic-vs-
    # vocabulary disagreement is exactly the seam bug class.
    with pytest.raises(ValueError, match="disagrees with block_base"):
        MolmoAct2ARDecoder(
            dataclasses.replace(config, block_base=BLOCK_BASE + 2),
            text,
            codec(),
            tokenizer=tokenizer,
        )
    # A block straddling out of the base matrices.
    shallow = dataclasses.replace(text, vocab_size=BLOCK_BASE + 100)
    with pytest.raises(ValueError, match="does not sit inside the base matrices"):
        MolmoAct2ARDecoder(config, shallow, codec(), tokenizer=tokenizer)


def test_empty_opener_and_suffix_targets(tiny_checkpoint: Path) -> None:
    decoder = build_decoder(tiny_checkpoint)
    assert decoder.opener_ids == ()
    raw, sequences = action_rows()
    sample = batch(raw, sequences)
    full, targets, is_aux = suffix_targets(decoder, sample)
    assert is_aux is None
    tokens = sample.action_tokens
    assert tokens is not None
    # Backbone-id suffix: BOA ≡ <action_start> at position 0 of every
    # row, bins at block_base + bin, filler at <action_end>.
    assert bool((full[:, 0] == BLOCK_BASE - 2).all())
    assert torch.equal(full, tokens + BLOCK_BASE)
    # Position 0 (input: BOA) predicts the FIRST BIN — the empty-opener
    # clamp must not eat it (the raw -1 slice masked all but the last).
    for row, sequence in enumerate(sequences):
        assert int(targets[row, 0]) == BLOCK_BASE + sequence[1]
    # <action_end> (the pad target) is never trained; everything else is.
    pad_positions = full[:, 1:] == BLOCK_BASE - 1
    assert bool((targets[pad_positions] == IGNORE_INDEX).all())
    assert bool((targets[~pad_positions] != IGNORE_INDEX).all())


def test_prefill_continue_matches_monolithic(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    """The cache keystone under LEFT PADDING: prefill + suffix
    continuation ≡ one forward over [prompt + suffix], suffix rows
    text-typed causal."""
    decoder = build_decoder(tiny_checkpoint)
    suffix = torch.tensor(
        [
            [BLOCK_BASE - 2, BLOCK_BASE + 3, BLOCK_BASE + 7],
            [BLOCK_BASE - 2, BLOCK_BASE + 5, BLOCK_BASE - 1],
        ],
        dtype=torch.long,
    )
    memory = encode_memory(model, padded=True)
    with torch.no_grad():
        continued = decoder(model, memory, suffix)

    generator = torch.Generator().manual_seed(11)
    prompt = torch.randint(1, 100, (BATCH, 9), generator=generator)
    real = torch.ones(BATCH, 9, dtype=torch.long)
    prompt[1, :2] = 0
    real[1, :2] = 0
    transformer = model.text.transformer
    with torch.no_grad():
        full_embeds = transformer.wte(torch.cat([prompt, suffix], dim=1))
        full_real = torch.cat(
            [real, torch.ones(BATCH, suffix.shape[1], dtype=torch.long)],
            dim=1,
        )
        positions = (full_real.cumsum(-1) - 1).clamp(min=0)
        mask = build_multimodal_mask(
            image_type_mask=torch.zeros_like(full_real, dtype=torch.bool),
            padding_mask=full_real,
            dtype=full_embeds.dtype,
            device=full_embeds.device,
        )
        hidden = transformer(
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
    decoder = build_decoder(tiny_checkpoint)
    raw, sequences = action_rows()
    forced = (batch(raw, sequences).action_tokens + BLOCK_BASE)[:, :6]  # type: ignore[operator]  # action_tokens is set two lines up
    with torch.no_grad():
        one_shot = decoder(model, encode_memory(model), forced)
        memory = encode_memory(model)
        fed = 0
        for j in range(forced.shape[1]):
            step_logits, fed = decoder._step(model, memory, forced[:, j : j + 1], fed)
            torch.testing.assert_close(
                step_logits,
                one_shot[:, j].float(),
                rtol=1e-4,
                atol=1e-4,
            )


def test_teacher_forced_block_logits_on_parameterless_decoder(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    """The mcselect/GRPO replay surface: runs with ZERO decoder
    parameters (the trunk-device fix), block slice [len, 2048], row t
    predicts ids[t] — verified against the plain forward."""
    decoder = build_decoder(tiny_checkpoint)
    assert len(list(decoder.parameters())) == 0
    _, sequences = action_rows()
    bins = [sequence[1:8] for sequence in sequences]  # body only, no boa
    with torch.no_grad():
        rows = decoder.teacher_forced_block_logits(
            model,
            encode_memory(model),
            [bins[0], None],
        )
        assert rows[1] is None
        first = rows[0]
        assert first is not None
        assert first.shape == (len(bins[0]), 2048)
        # Alignment oracle: the same suffix through forward() — the
        # memory's cache is B=2, so the reference feeds both rows (row 1
        # is filler) and reads row 0.
        seq = torch.tensor(
            [[BLOCK_BASE - 2, *(BLOCK_BASE + b for b in bins[0])]] * BATCH,
            dtype=torch.long,
        )
        reference = decoder(model, encode_memory(model), seq)[
            0,
            : len(bins[0]),
            BLOCK_BASE : BLOCK_BASE + 2048,
        ].float()
    torch.testing.assert_close(first, reference, rtol=1e-4, atol=1e-4)


def test_predict_chunk_masked_decode_and_capture(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    """The serving/RL decode on a random trunk: exact budget
    consumption, only reachable bins, block-relative [B, 2048] capture
    with backbone-id chosen, and the executed chunk ≡ codec.decode of
    the captured bins (the scaffold's own decode tail)."""
    decoder = build_decoder(tiny_checkpoint)
    loaded = codec()
    raw, sequences = action_rows()
    sample = batch(raw, sequences)
    capture: list[ActionCaptureStep] = []
    with torch.no_grad():
        prediction = decoder.predict_chunk(
            model,
            encode_memory(model),
            sample,
            action_capture=capture,
        )
    assert prediction.actions.shape == (BATCH, T, D)
    assert len(capture) > 0
    lengths = loaded.symbol_lengths
    bins_per_row: list[list[int]] = [[] for _ in range(BATCH)]
    budgets = [T * D] * BATCH
    for step in capture:
        assert step.block_logits.shape == (BATCH, 2048)
        assert step.allowed.shape == (BATCH, 2048)
        for row in range(BATCH):
            if not bool(step.active[row]):
                continue
            backbone_id = int(step.chosen[row])
            bin_id = backbone_id - BLOCK_BASE
            assert 0 <= bin_id < 1005  # reachable bins only
            assert bool(step.allowed[row, bin_id])
            bins_per_row[row].append(bin_id)
            budgets[row] -= int(lengths[bin_id])
    assert budgets == [0, 0]  # exact fill, every row
    for row in range(BATCH):
        expected = torch.from_numpy(
            loaded.decode(bins_per_row[row], Q01, Q99),
        ).float()
        torch.testing.assert_close(prediction.actions[row], expected)


def test_sampled_decode_is_keyed_deterministic(
    model: Molmo2Model,
    tiny_checkpoint: Path,
) -> None:
    decoder = build_decoder(tiny_checkpoint)
    raw, sequences = action_rows()
    sample = batch(raw, sequences)

    def draw() -> torch.Tensor:
        sampling = ARSampling(
            temperature=1.3,
            rngs=tuple(np.random.default_rng(1000 + row) for row in range(BATCH)),
        )
        with torch.no_grad():
            return decoder.predict_chunk(
                model,
                encode_memory(model),
                sample,
                sampling=sampling,
            ).actions

    assert torch.equal(draw(), draw())


def test_schema_roundtrip(tiny_checkpoint: Path) -> None:
    decoder = build_decoder(tiny_checkpoint)
    payload = json.loads(json.dumps(decoder_schema_dict(decoder)))
    assert payload["kind"] == "ar_backbone"
    assert payload["suffix_format"] == MOLMOACT2_SUFFIX_FORMAT
    assert parse_decoder_config(payload) == decoder_config()


# --- loading arm (from_checkpoint + the release-read helper) ---


def prompt_config(*, action_mode: str = "both") -> MolmoAct2PromptConfig:
    return MolmoAct2PromptConfig(
        format=MOLMOACT2_PROMPT_FORMAT,
        norm_tag="tiny",
        setup_type="tabletop",
        control_mode="joint",
        num_state_tokens=32,
        state_dim=D,
        action_mode=action_mode,
        n_obs_steps=1,
        camera_keys=("top",),
        narration=False,
    )


def tiny_stats() -> DatasetStats:
    return DatasetStats(
        action_mean=(0.0,) * D,
        action_std=(1.0,) * D,
        state_mean=(0.0,) * D,
        state_std=(1.0,) * D,
        action_q01=tuple(Q01.tolist()),
        action_q99=tuple(Q99.tolist()),
        state_q01=(-1.0,) * D,
        state_q99=(1.0,) * D,
    )


def write_ar_checkpoint(
    directory: Path,
    tiny_checkpoint: Path,
    *,
    action_mode: str = "both",
    suffix_format: int = MOLMOACT2_SUFFIX_FORMAT,
) -> Path:
    """A POSSIBLE ar-only LEGACY molmoact2 checkpoint dir: the format-3
    metadata phase-3 training wrote (recorded objective 'ar', the
    train-written discrete layout) — no expert/prompt weight files
    (both roles parameterless), no backbone.safetensors (frozen
    trunk). The converter's input."""
    directory.mkdir(parents=True, exist_ok=True)
    section = ar_backbone_config_to_dict(decoder_config())
    section["suffix_format"] = suffix_format
    metadata = CheckpointMetadata(
        backbone=BackboneConfig(id=str(tiny_checkpoint), depth=BackboneDepth.FULL),
        prompt=prompt_config(action_mode=action_mode),
        decoder=section,
        normalization=tiny_stats(),
        per_dataset_normalization={"marius/rig": tiny_stats()},
        train_args={
            "decoder": "ar_backbone",
            "objective": "ar",
            "decoder_hidden": 64,
            "decoder_heads": 2,
            "decoder_intermediate": 128,
            "decoder_cross_heads": 2,
            "stream_counts": [],
            "self_attention_mode": "bidirectional",
            "chunk_size": T,
            "max_soft_tokens": 140,
            "max_crops": 1,
            "time_conditioning": "additive",
            "target_time_embed": False,
            "fast_tokenizer": str(FAST_FIXTURE),
            "joint_ce": False,
        },
        step=0,
    )
    (directory / "bijou_config.json").write_text(
        json.dumps(metadata.to_json_dict()),
    )
    return directory


def test_converted_ar_checkpoint_roundtrip(
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    checkpoint = write_ar_checkpoint(tmp_path / "ar_only", tiny_checkpoint)
    converted = tmp_path / "converted"
    convert(checkpoint, converted)
    model = load_vla(converted, device="cpu", dtype=torch.float32)
    assert isinstance(model, MolmoAct2ARVLA)
    assert isinstance(model.encoder, MolmoAct2Encoder)
    assert model.ar_decoder.config == decoder_config()
    assert model.ar_decoder.opener_ids == ()
    assert len(list(model.ar_decoder.parameters())) == 0


def test_converted_ar_checkpoint_refusals(
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    # 'continuous' (the rig-ft class) refused by name at the family
    # load — the shared builder's guard, before any weight touches.
    continuous = write_ar_checkpoint(
        tmp_path / "continuous",
        tiny_checkpoint,
        action_mode="continuous",
    )
    converted_continuous = tmp_path / "continuous_vla"
    convert(continuous, converted_continuous)
    with pytest.raises(SystemExit, match="never trained the discrete head"):
        load_vla(converted_continuous, device="cpu", dtype=torch.float32)
    # A stray expert file on a parameterless decoder = format confusion,
    # refused at CONVERSION (mirrors the retired legacy loader's guard).
    stray = write_ar_checkpoint(tmp_path / "stray", tiny_checkpoint)
    (stray / "expert.safetensors").write_bytes(b"")
    with pytest.raises(SystemExit, match="owns no parameters"):
        convert(stray, tmp_path / "stray_vla")
    # A value-line (format-5) section under the molmoact2 prompt:
    # conversion carries the section verbatim; the decoder constructor
    # refuses it at the family load.
    mixed = write_ar_checkpoint(
        tmp_path / "mixed",
        tiny_checkpoint,
        suffix_format=5,
    )
    converted_mixed = tmp_path / "mixed_vla"
    convert(mixed, converted_mixed)
    with pytest.raises(SystemExit, match="format-5 ar_backbone section"):
        load_vla(converted_mixed, device="cpu", dtype=torch.float32)


def release_flow_section(*, n_action_steps: int = T) -> MolmoFlowDecoderConfig:
    """The release-shaped molmo_flow section fields the AR read consumes
    (geometry); expert shape values are plausible fillers."""
    return MolmoFlowDecoderConfig(
        max_horizon=T,
        max_action_dim=32,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        mlp_ratio=4.0,
        ffn_multiple_of=64,
        timestep_embed_dim=32,
        context_layer_norm=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        rope=True,
        causal_attn=False,
        llm_kv_dim=32,
        num_flow_steps=10,
        mask_action_dim_padding=True,
        action_dim=D,
        action_horizon=T,
        n_action_steps=n_action_steps,
        normalization="q01q99",
        time_offset=0.001,
        time_scale=0.999,
        beta_alpha=1.0,
        beta_beta=1.5,
    )


def test_release_read_derives_the_ar_config(tiny_checkpoint: Path) -> None:
    """The AR read of a RELEASE-class checkpoint (no format-6 section):
    geometry from the flow section, block_base from the trunk
    tokenizer's own <action_0>, block width from the artifact — and
    the derived config builds a working decoder through the shared
    builder."""
    config = molmoact2_ar_config_from_flow_section(
        release_flow_section(),
        prompt_config(),
        str(tiny_checkpoint),
        fast_tokenizer=str(FAST_FIXTURE),
    )
    assert config == decoder_config()
    decoder = build_molmoact2_ar_decoder(
        config,
        prompt_config(),
        text_config().text,
        str(tiny_checkpoint),
    )
    assert decoder.opener_ids == ()
    # Non-identity output tails have no first-class consumer.
    with pytest.raises(SystemExit, match="non-identity discrete output tail"):
        molmoact2_ar_config_from_flow_section(
            release_flow_section(n_action_steps=T - 5),
            prompt_config(),
            str(tiny_checkpoint),
            fast_tokenizer=str(FAST_FIXTURE),
        )
    # The builder holds the action-mode line for direct callers too.
    with pytest.raises(SystemExit, match="never trained the discrete head"):
        build_molmoact2_ar_decoder(
            config,
            prompt_config(action_mode="continuous"),
            text_config().text,
            str(tiny_checkpoint),
        )
