"""Checkpoint-schema tests: format-3 sectioned configs, the format-2 and
format-1 read-side synthesizers, and the --init-from config guard across
formats.

Offline by construction: the backbone architecture comes from
``Gemma4Config.e2b()`` (built in code, matching google/gemma-4-e2b-it), and the
legacy fixture is a real pre-format-2 ``bijou_config.json`` (the adaRMS
rig fine-tune's, per-dataset table trimmed).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from bijou.data import DatasetStats
from bijou.decoders.flow import ExpertConfig, FlowDecoder
from bijou.gemma4.config import Gemma4Config
from bijou.loading import (
    BackboneConfig,
    BackboneDepth,
    CheckpointMetadata,
    CheckpointTrainArgs,
    FlowDecoderConfig,
    GemmaPromptConfig,
    Molmo2PromptConfig,
    checkpoint_sections,
    expert_config_from_architecture,
    expert_config_from_train_args,
    flow_decoder_config_from_expert,
    parse_decoder_config,
    parse_prompt_config,
)
from bijou.train import ensure_matching_decoder_config

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_bijou_config.json"


def legacy_meta() -> dict:
    return json.loads(FIXTURE.read_text())


def legacy_expert_config() -> ExpertConfig:
    meta = legacy_meta()
    return expert_config_from_train_args(
        Gemma4Config.e2b(),
        CheckpointTrainArgs.from_dict(meta["train_args"]),
        action_dim=len(meta["normalization"]["action"]["mean"]),
        state_dim=len(meta["normalization"]["observation.state"]["mean"]),
    )


def test_train_args_read_both_key_spellings() -> None:
    """New checkpoints record decoder_*; historical ones expert_* — both
    load, and mixed sources agree on the same values."""
    legacy_args = legacy_meta()["train_args"]
    renamed = {
        **{k: v for k, v in legacy_args.items() if not k.startswith("expert_")},
        "decoder_hidden": legacy_args["expert_hidden"],
        "decoder_heads": legacy_args["expert_heads"],
        "decoder_intermediate": legacy_args["expert_intermediate"],
        "decoder_cross_heads": legacy_args["expert_cross_heads"],
    }
    assert CheckpointTrainArgs.from_dict(renamed) == CheckpointTrainArgs.from_dict(
        legacy_args,
    )


def test_legacy_synthesizer_reproduces_recorded_expert_config() -> None:
    """The synthesized config must equal the expert_config the format-1
    checkpoint actually recorded (same normalization ensure_matching
    applies: json round-trip stringifies enums)."""
    recorded = legacy_meta()["expert_config"]
    # Same back-compat normalization ensure_matching applies: fields
    # added to ExpertConfig after format-1 checkpoints were written are
    # absent from their serialized configs and default-filled on read.
    recorded.setdefault("target_time_embed", False)
    recorded.setdefault("residual_streams", False)
    recorded.setdefault("residual_stream_dim", None)
    recorded.setdefault("cross_attention_kv_heads", None)
    synthesized = json.loads(
        json.dumps(dataclasses.asdict(legacy_expert_config()), default=str),
    )
    assert synthesized == recorded


def test_prompt_decoder_bridge_roundtrips_expert_config() -> None:
    expert_config = legacy_expert_config()
    prompt = GemmaPromptConfig(
        exports=expert_config.streams,
        max_soft_tokens=140,
        format=3,
        state_dim=6,
        condition_fields=(),
        generate_bracket=False,
    )
    decoder = flow_decoder_config_from_expert(expert_config)
    assert decoder.schedule[:5] == ("kv4", "kv4", "kv4", "kv4", "kv9")
    rebuilt = expert_config_from_architecture(prompt, decoder, Gemma4Config.e2b())
    assert rebuilt == expert_config


def test_config_dicts_roundtrip_through_json() -> None:
    expert_config = legacy_expert_config()
    prompt = GemmaPromptConfig(
        exports=(4, 9, 14),
        max_soft_tokens=140,
        format=3,
        state_dim=6,
        condition_fields=(),
        generate_bracket=True,  # the non-default: must survive the trip
    )
    decoder = flow_decoder_config_from_expert(expert_config)
    prompt_parsed = parse_prompt_config(json.loads(json.dumps(prompt.to_dict())))
    decoder_parsed = parse_decoder_config(json.loads(json.dumps(decoder.to_dict())))
    assert prompt_parsed == prompt
    assert decoder_parsed == decoder


def test_prompt_config_bracket_defaults_false_on_old_checkpoints() -> None:
    """Checkpoints written before 2026-08-04 carry no generate_bracket
    key: those flow prompts had no bracket (ar_backbone consumers OR
    the parsed value with their decoder kind, where it is implied)."""
    payload = GemmaPromptConfig(
        exports=(4, 9, 14),
        max_soft_tokens=140,
        format=3,
        state_dim=6,
        condition_fields=(),
        generate_bracket=True,
    ).to_dict()
    del payload["generate_bracket"]
    assert parse_prompt_config(payload).generate_bracket is False


def test_unknown_stream_and_unused_export_fail_loudly() -> None:
    expert_config = legacy_expert_config()
    decoder = flow_decoder_config_from_expert(expert_config)
    prompt = GemmaPromptConfig(
        exports=(4, 9),  # kv14 missing => schedule references unknown stream
        max_soft_tokens=140,
        format=3,
        state_dim=6,
        condition_fields=(),
        generate_bracket=False,
    )
    with pytest.raises(SystemExit, match="unknown stream"):
        expert_config_from_architecture(prompt, decoder, Gemma4Config.e2b())

    schedule_without_kv9 = tuple(
        "kv4" if name == "kv9" else name for name in decoder.schedule
    )
    decoder_unused = dataclasses.replace(decoder, schedule=schedule_without_kv9)
    prompt_full = GemmaPromptConfig(
        exports=(4, 9, 14),
        max_soft_tokens=140,
        format=3,
        state_dim=6,
        condition_fields=(),
        generate_bracket=False,
    )
    with pytest.raises(SystemExit, match="not consumed"):
        expert_config_from_architecture(prompt_full, decoder_unused, Gemma4Config.e2b())


def tiny_stats(dim: int = 6) -> DatasetStats:
    return DatasetStats(
        action_mean=(0.0,) * dim,
        action_std=(1.0,) * dim,
        state_mean=(0.0,) * dim,
        state_std=(1.0,) * dim,
        action_q01=(-1.0,) * dim,
        action_q99=(1.0,) * dim,
        state_q01=(-1.0,) * dim,
        state_q99=(1.0,) * dim,
    )


def format3_meta() -> dict:
    expert_config = legacy_expert_config()
    metadata = CheckpointMetadata(
        backbone=BackboneConfig(
            id="google/gemma-4-e2b-it",
            depth=BackboneDepth.PREFIX,
        ),
        prompt=GemmaPromptConfig(
            exports=expert_config.streams,
            max_soft_tokens=140,
            format=3,
            state_dim=6,
            condition_fields=(),
            generate_bracket=False,
        ),
        decoder=flow_decoder_config_from_expert(expert_config).to_dict(),
        normalization=tiny_stats(),
        per_dataset_normalization={"marius/rig": tiny_stats()},
        train_args=legacy_meta()["train_args"],
        step=5000,
    )
    return json.loads(json.dumps(metadata.to_json_dict(), default=str))


def format2_meta() -> dict:
    """A format-2 payload as historical checkpoints carry it (the backbone
    id lives inside the encoder section) — hand-built because the write
    side moved to format 3."""
    meta3 = format3_meta()
    return {
        "format": 2,
        "encoder": {
            "kind": "gemma4",
            "backbone": meta3["backbone"]["id"],
            "exports": meta3["prompt"]["exports"],
            "max_soft_tokens": meta3["prompt"]["max_soft_tokens"],
        },
        "decoder": meta3["decoder"],
        "step": meta3["step"],
        "train_args": meta3["train_args"],
        "normalization": meta3["normalization"],
        "per_dataset_normalization": meta3["per_dataset_normalization"],
    }


def test_metadata_writes_format3_and_reads_back() -> None:
    meta = format3_meta()
    assert meta["format"] == 3
    assert meta["backbone"] == {
        "id": "google/gemma-4-e2b-it",
        "depth": "prefix",
    }
    assert meta["prompt"]["kind"] == "gemma4"
    assert meta["decoder"]["kind"] == "flow"
    sections = checkpoint_sections(meta)
    assert sections.backbone.depth is BackboneDepth.PREFIX
    assert isinstance(sections.prompt, GemmaPromptConfig)
    assert isinstance(sections.decoder, FlowDecoderConfig)
    rebuilt = expert_config_from_architecture(
        sections.prompt,
        sections.decoder,
        Gemma4Config.e2b(),
    )
    assert rebuilt == legacy_expert_config()


def test_pre3_prompt_sections_are_refused() -> None:
    """Schema-format-2 checkpoints carry pre-format-3 prompts (no
    [generate|…], no soft state token) whose parameter set this code
    no longer implements — parsing refuses them loudly (no back-compat,
    2026-08-03). Schema format 1 has no prompt section at all: its
    metadata still parses (sections None), and the model LOAD refuses
    it instead."""
    with pytest.raises(SystemExit, match="no back-compat"):
        checkpoint_sections(format2_meta())

    from_format1 = checkpoint_sections(legacy_meta())
    from_format3 = checkpoint_sections(format3_meta())
    assert from_format1.backbone == from_format3.backbone
    assert from_format1.prompt is None
    assert from_format1.decoder is None


def test_molmo2_prompt_config_round_trips() -> None:
    """The molmo2 prompt section (WP5, landed with the AR decoder arm)
    parses to its own config class — never misread as a Gemma section —
    and survives a to_dict/from_dict round trip."""
    config = Molmo2PromptConfig(
        max_crops=1,
        format=1,
        state_dim=6,
        condition_fields=("subgoal", "outcome"),
        generate_bracket=True,
    )
    parsed = parse_prompt_config(config.to_dict())
    assert isinstance(parsed, Molmo2PromptConfig)
    assert parsed == config


def meta_decoder(config: ExpertConfig) -> FlowDecoder:
    """An e2b-sized decoder on the meta device: no allocation, and the
    config guard only reads ``.config``."""
    return FlowDecoder(config, device="meta")


def test_ensure_matching_decoder_config_both_formats(tmp_path: Path) -> None:
    expert_config = legacy_expert_config()
    decoder = meta_decoder(expert_config)
    mismatched = meta_decoder(
        dataclasses.replace(expert_config, hidden_size=1024),
    )

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "bijou_config.json").write_text(json.dumps(legacy_meta()))
    ensure_matching_decoder_config(decoder, legacy_dir)
    with pytest.raises(SystemExit, match="decoder config mismatch"):
        ensure_matching_decoder_config(mismatched, legacy_dir)

    format2_dir = tmp_path / "format2"
    format2_dir.mkdir()
    (format2_dir / "bijou_config.json").write_text(json.dumps(format2_meta()))
    ensure_matching_decoder_config(decoder, format2_dir)
    with pytest.raises(SystemExit, match="decoder config mismatch"):
        ensure_matching_decoder_config(mismatched, format2_dir)
