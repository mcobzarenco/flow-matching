"""Phase-3 gates: the VLA checkpoint toolkit (schema round-trip,
atomic writes, the hard-link rule and its copy fallback,
self-containment validation) and bijou.convert_legacy on a fabricated
format-3 gemma-flow directory (family inference, weight-file linking,
idempotence, --replace-stats). The molmoact2 conversion branches gate
on the REAL tiny fixture in the phase-4 parity suite."""

import errno
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import bijou.checkpoint
from bijou.checkpoint import (
    VLAMetadata,
    link_or_copy,
    read_metadata,
    validate_checkpoint,
    write_checkpoint,
)
from bijou.convert_legacy import convert
from bijou.data import DatasetStats
from bijou.loading import (
    PROMPT_FORMAT,
    FlowDecoderSection,
    GemmaPromptConfig,
)
from bijou.modelling.decoders.flow import SelfAttentionMode, TimeConditioning
from bijou.modelling.interface import SamplingMethod  # noqa: F401 — asserts import path
from bijou.vla import VLAFamily

ACTION_DIM = 6


def stats() -> DatasetStats:
    return DatasetStats.from_state_dict(
        {
            "action": {
                "mean": [0.0] * ACTION_DIM,
                "std": [1.0] * ACTION_DIM,
                "q01": [-1.0] * ACTION_DIM,
                "q99": [1.0] * ACTION_DIM,
            },
            "observation.state": {
                "mean": [0.0] * ACTION_DIM,
                "std": [1.0] * ACTION_DIM,
            },
        },
    )


def metadata(**overrides: object) -> VLAMetadata:
    fields: dict = {
        "family": VLAFamily.GEMMA_FLOW,
        "chunk_size": 5,
        "action_dim": ACTION_DIM,
        "backbone_id": "some/backbone",
        "backbone_depth": "prefix",
        "backbone_trained": False,
        "objective": {"kind": "flow"},
        "serving": {"kind": "flow", "num_steps": 5, "method": "heun"},
        "components": {
            "prompt": {"config": {"kind": "gemma4"}, "weights": True},
            "flow_decoder": {"config": {"kind": "flow"}, "weights": True},
        },
        "artifacts": {},
        "stats": stats(),
        "per_dataset_stats": {"repo/a": stats()},
        "train_args": {"seed": 0},
        "step": 2,
        "stats_note": None,
    }
    fields.update(overrides)
    return VLAMetadata(**fields)


def snapshot_dir(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    save_file({"w": torch.zeros(2)}, str(snapshot / "model.safetensors"))
    return snapshot


def component_states() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "prompt": {"state_proj.weight": torch.zeros(2, 2)},
        "flow_decoder": {"proj.weight": torch.ones(2, 2)},
    }


def test_metadata_round_trip() -> None:
    meta = metadata()
    assert VLAMetadata.from_json_dict(meta.to_json_dict()) == meta


def test_metadata_rejects_wrong_schema_version() -> None:
    payload = metadata().to_json_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        VLAMetadata.from_json_dict(payload)


def test_write_checkpoint_pristine_backbone_hard_links(tmp_path: Path) -> None:
    snapshot = snapshot_dir(tmp_path)
    target = tmp_path / "ckpt"
    write_checkpoint(
        target,
        metadata=metadata(),
        components=component_states(),
        backbone=snapshot,
    )
    meta = validate_checkpoint(target)
    assert meta.family is VLAFamily.GEMMA_FLOW
    linked = target / "backbone" / "model.safetensors"
    assert linked.exists()
    assert linked.samefile(snapshot / "model.safetensors")
    assert not (target.parent / "ckpt.tmp").exists()


def test_write_checkpoint_refuses_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "ckpt"
    target.mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        write_checkpoint(
            target,
            metadata=metadata(),
            components=component_states(),
            backbone=snapshot_dir(tmp_path),
        )


def test_write_checkpoint_refuses_component_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="component mismatch"):
        write_checkpoint(
            tmp_path / "ckpt",
            metadata=metadata(),
            components={"prompt": {"w": torch.zeros(1)}},  # flow_decoder missing
            backbone=snapshot_dir(tmp_path),
        )


def test_write_checkpoint_refuses_backbone_form_contradiction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contradicts"):
        write_checkpoint(
            tmp_path / "ckpt",
            metadata=metadata(backbone_trained=True),
            components=component_states(),
            backbone=snapshot_dir(tmp_path),  # a directory, but trained=True
        )


def test_validate_rejects_stray_weight_file(tmp_path: Path) -> None:
    target = tmp_path / "ckpt"
    write_checkpoint(
        target,
        metadata=metadata(),
        components=component_states(),
        backbone=snapshot_dir(tmp_path),
    )
    save_file({"x": torch.zeros(1)}, str(target / "stray.safetensors"))
    with pytest.raises(SystemExit, match="undeclared weight files"):
        validate_checkpoint(target)


def test_read_metadata_names_the_converter_on_legacy_dirs(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "bijou_config.json").write_text("{}")
    with pytest.raises(SystemExit, match="convert_legacy"):
        read_metadata(legacy)


def test_link_falls_back_to_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "a.bin"
    source.write_bytes(b"payload")
    destination = tmp_path / "b.bin"

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(bijou.checkpoint.os, "link", refuse)
    link_or_copy(source, destination)
    assert destination.read_bytes() == b"payload"
    assert not destination.samefile(source)
    assert "copying in full" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# convert_legacy on a fabricated format-3 gemma-flow directory


def legacy_checkpoint(tmp_path: Path) -> Path:
    snapshot = snapshot_dir(tmp_path)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    prompt = GemmaPromptConfig(
        exports=(3,),
        max_soft_tokens=8,
        format=PROMPT_FORMAT,
        state_dim=ACTION_DIM,
        condition_fields=(),
        generate_bracket=False,
    )
    decoder = FlowDecoderSection(
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        self_attention_mode=next(iter(SelfAttentionMode)),
        self_attention_rope_theta=10000.0,
        cross_attention_heads=2,
        schedule=("kv3",),
        action_dim=ACTION_DIM,
        state_dim=ACTION_DIM,
        chunk_size=5,
        time_embed_dim=8,
        time_conditioning=TimeConditioning.ADDITIVE,
    )
    config = {
        "format": 3,
        "backbone": {"id": str(snapshot), "depth": "prefix"},
        "prompt": prompt.to_dict(),
        "decoder": decoder.to_dict(),
        "step": 2,
        "train_args": {
            "decoder": "flow",
            "decoder_hidden": 8,
            "decoder_heads": 2,
            "decoder_intermediate": 16,
            "decoder_cross_heads": 2,
            "stream_counts": [1],
            "self_attention_mode": next(iter(SelfAttentionMode)).value,
            "chunk_size": 5,
            "max_soft_tokens": 8,
            "seed": 0,
        },
        "normalization": stats().state_dict(),
        "per_dataset_normalization": {"repo/a": stats().state_dict()},
    }
    (legacy / "bijou_config.json").write_text(json.dumps(config))
    save_file({"proj.weight": torch.ones(2, 2)}, str(legacy / "expert.safetensors"))
    save_file(
        {"state_proj.weight": torch.zeros(2, 2)},
        str(legacy / "prompt.safetensors"),
    )
    (legacy / "optimizer.pt").write_bytes(b"opt")
    return legacy


def test_convert_gemma_flow(tmp_path: Path) -> None:
    legacy = legacy_checkpoint(tmp_path)
    converted = tmp_path / "converted"
    meta = convert(legacy, converted)
    assert meta.family is VLAFamily.GEMMA_FLOW
    assert meta.chunk_size == 5
    assert meta.action_dim == ACTION_DIM
    assert meta.backbone_trained is False
    assert meta.serving == {"kind": "flow", "num_steps": 5, "method": "heun"}
    assert meta.components["flow_decoder"]["config"]["kind"] == "flow"
    validate_checkpoint(converted)
    # weight files are hard links to the source, never rewrites
    assert (converted / "flow_decoder.safetensors").samefile(
        legacy / "expert.safetensors",
    )
    assert (converted / "prompt.safetensors").samefile(legacy / "prompt.safetensors")
    assert (converted / "optimizer.pt").exists()
    assert (converted / "backbone" / "model.safetensors").exists()


def test_convert_is_idempotent(tmp_path: Path) -> None:
    legacy = legacy_checkpoint(tmp_path)
    first = tmp_path / "one"
    second = tmp_path / "two"
    convert(legacy, first)
    convert(legacy, second)
    assert (first / "metadata.json").read_bytes() == (
        second / "metadata.json"
    ).read_bytes()
    for name in ("flow_decoder.safetensors", "prompt.safetensors"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_convert_replace_stats(tmp_path: Path) -> None:
    legacy = legacy_checkpoint(tmp_path)
    replacement = stats().state_dict()
    replacement["action"]["mean"] = [42.0] * ACTION_DIM
    table = tmp_path / "table.json"
    table.write_text(json.dumps(replacement))
    meta = convert(legacy, tmp_path / "converted", replace_stats=table)
    assert meta.stats.action_mean == (42.0,) * ACTION_DIM
    assert meta.stats_note is not None
    assert "REPLACED" in meta.stats_note
    # per-dataset tables stay untouched
    assert meta.per_dataset_stats["repo/a"].action_mean == (0.0,) * ACTION_DIM


def test_convert_refuses_pre_format_3(tmp_path: Path) -> None:
    legacy = legacy_checkpoint(tmp_path)
    config = json.loads((legacy / "bijou_config.json").read_text())
    config["format"] = 2
    (legacy / "bijou_config.json").write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="format 2"):
        convert(legacy, tmp_path / "converted")
