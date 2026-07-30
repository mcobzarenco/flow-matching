"""Checkpoint-schema tests: format-2 tagged configs, the format-1 legacy
synthesizer, and the --init-from config guard across both formats.

Offline by construction: the backbone architecture comes from
``e2b_config()`` (built in code, matching google/gemma-4-e2b-it), and the
legacy fixture is a real pre-format-2 ``bijou_config.json`` (the adaRMS
rig fine-tune's, per-dataset table trimmed).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from bijou.data import DatasetStats
from bijou.expert import ExpertConfig
from bijou.gemma4.config import e2b_config
from bijou.loading import (
    CheckpointMetadata,
    CheckpointTrainArgs,
    GemmaEncoderConfig,
    expert_config_from_architecture,
    expert_config_from_train_args,
    flow_decoder_config_from_expert,
    parse_decoder_config,
    parse_encoder_config,
)
from bijou.train import ensure_matching_expert_config

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_bijou_config.json"


def legacy_meta() -> dict:
    return json.loads(FIXTURE.read_text())


def legacy_expert_config() -> ExpertConfig:
    meta = legacy_meta()
    return expert_config_from_train_args(
        e2b_config(),
        CheckpointTrainArgs.from_dict(meta["train_args"]),
        action_dim=len(meta["normalization"]["action"]["mean"]),
        state_dim=len(meta["normalization"]["observation.state"]["mean"]),
    )


def test_legacy_synthesizer_reproduces_recorded_expert_config() -> None:
    """The synthesized config must equal the expert_config the format-1
    checkpoint actually recorded (same normalization ensure_matching
    applies: json round-trip stringifies enums)."""
    recorded = legacy_meta()["expert_config"]
    synthesized = json.loads(
        json.dumps(dataclasses.asdict(legacy_expert_config()), default=str),
    )
    assert synthesized == recorded


def test_format2_bridge_roundtrips_expert_config() -> None:
    expert_config = legacy_expert_config()
    encoder = GemmaEncoderConfig(
        backbone="google/gemma-4-e2b-it",
        exports=expert_config.streams,
        max_soft_tokens=140,
    )
    decoder = flow_decoder_config_from_expert(expert_config)
    assert decoder.schedule[:5] == ("kv4", "kv4", "kv4", "kv4", "kv9")
    rebuilt = expert_config_from_architecture(encoder, decoder, e2b_config())
    assert rebuilt == expert_config


def test_config_dicts_roundtrip_through_json() -> None:
    expert_config = legacy_expert_config()
    encoder = GemmaEncoderConfig(
        backbone="google/gemma-4-e2b-it",
        exports=(4, 9, 14),
        max_soft_tokens=140,
    )
    decoder = flow_decoder_config_from_expert(expert_config)
    encoder_parsed = parse_encoder_config(json.loads(json.dumps(encoder.to_dict())))
    decoder_parsed = parse_decoder_config(json.loads(json.dumps(decoder.to_dict())))
    assert encoder_parsed == encoder
    assert decoder_parsed == decoder


def test_unknown_stream_and_unused_export_fail_loudly() -> None:
    expert_config = legacy_expert_config()
    decoder = flow_decoder_config_from_expert(expert_config)
    encoder = GemmaEncoderConfig(
        backbone="google/gemma-4-e2b-it",
        exports=(4, 9),  # kv14 missing => schedule references unknown stream
        max_soft_tokens=140,
    )
    with pytest.raises(SystemExit, match="unknown stream"):
        expert_config_from_architecture(encoder, decoder, e2b_config())

    schedule_without_kv9 = tuple(
        "kv4" if name == "kv9" else name for name in decoder.schedule
    )
    decoder_unused = dataclasses.replace(decoder, schedule=schedule_without_kv9)
    encoder_full = GemmaEncoderConfig(
        backbone="google/gemma-4-e2b-it",
        exports=(4, 9, 14),
        max_soft_tokens=140,
    )
    with pytest.raises(SystemExit, match="not consumed"):
        expert_config_from_architecture(encoder_full, decoder_unused, e2b_config())


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


def format2_meta() -> dict:
    metadata = CheckpointMetadata(
        backbone="google/gemma-4-e2b-it",
        expert_config=legacy_expert_config(),
        max_soft_tokens=140,
        normalization=tiny_stats(),
        per_dataset_normalization={"marius/rig": tiny_stats()},
        train_args=legacy_meta()["train_args"],
        step=5000,
    )
    return json.loads(json.dumps(metadata.to_json_dict(), default=str))


def test_metadata_writes_format2_and_reads_back() -> None:
    meta = format2_meta()
    assert meta["format"] == 2
    assert meta["encoder"]["kind"] == "gemma4"
    assert meta["decoder"]["kind"] == "flow"
    rebuilt = expert_config_from_architecture(
        parse_encoder_config(meta["encoder"]),
        parse_decoder_config(meta["decoder"]),
        e2b_config(),
    )
    assert rebuilt == legacy_expert_config()


def test_ensure_matching_expert_config_both_formats(tmp_path: Path) -> None:
    expert_config = legacy_expert_config()
    mismatched = dataclasses.replace(expert_config, hidden_size=1024)

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "bijou_config.json").write_text(json.dumps(legacy_meta()))
    ensure_matching_expert_config(expert_config, legacy_dir)
    with pytest.raises(SystemExit, match="expert config mismatch"):
        ensure_matching_expert_config(mismatched, legacy_dir)

    format2_dir = tmp_path / "format2"
    format2_dir.mkdir()
    (format2_dir / "bijou_config.json").write_text(json.dumps(format2_meta()))
    ensure_matching_expert_config(expert_config, format2_dir)
    with pytest.raises(SystemExit, match="expert config mismatch"):
        ensure_matching_expert_config(mismatched, format2_dir)
