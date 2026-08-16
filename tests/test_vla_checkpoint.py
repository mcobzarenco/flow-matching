"""The VLA checkpoint toolkit at schema 2 (metadata round-trip, atomic
writes, per-part backbone files with the hard-link dedup, the tokenizer
manifest, self-containment validation) and bijou.convert_legacy on a
fabricated format-3 gemma-flow directory (family inference, the trunk
import with its partition audit, weight-file linking, idempotence,
--replace-stats, the schema-1 refusal). The molmoact2 conversion
branches gate on the tiny fixture in test_convert_molmoact2 and the
train suites."""

import errno
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from vla_fixtures import write_gemma_flow_legacy, write_gemma_trunk

import bijou.checkpoint
from bijou.checkpoint import (
    BACKBONE_TEXT_FILENAME,
    BACKBONE_VISION_FILENAME,
    GEMMA_TOKENIZER_FILES,
    MOLMO_TOKENIZER_FILES,
    VLAMetadata,
    link_or_copy,
    read_metadata,
    tokenizer_manifest,
    validate_checkpoint,
    write_checkpoint,
)
from bijou.convert_legacy import convert
from bijou.data import DatasetStats
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
        "backbone_config": {"model_type": "gemma4", "text_config": {"k": 1}},
        "backbone_text_trained": False,
        "backbone_vision_trained": False,
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


def tokenizer_source(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "tokenizer_source"
    if not source.exists():
        source.mkdir()
        for name in GEMMA_TOKENIZER_FILES:
            (source / name).write_text("{}")
    return {name: source / name for name in GEMMA_TOKENIZER_FILES}


def component_states() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "prompt": {"state_proj.weight": torch.zeros(2, 2)},
        "flow_decoder": {"proj.weight": torch.ones(2, 2)},
    }


def backbone_parts() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return (
        {"language_model.embed_tokens.weight": torch.zeros(4, 2)},
        {"vision_tower.patch.weight": torch.ones(2, 2)},
    )


def write_tiny_checkpoint(tmp_path: Path, **metadata_overrides: object) -> Path:
    target = tmp_path / "ckpt"
    text, vision = backbone_parts()
    write_checkpoint(
        target,
        metadata=metadata(**metadata_overrides),
        components=component_states(),
        backbone_text=text,
        backbone_vision=vision,
        tokenizer_files=tokenizer_source(tmp_path),
    )
    return target


def test_metadata_round_trip() -> None:
    meta = metadata()
    assert VLAMetadata.from_json_dict(meta.to_json_dict()) == meta


def test_metadata_refuses_schema_1_with_reconvert_pointer() -> None:
    payload = metadata().to_json_dict()
    payload["schema_version"] = 1
    with pytest.raises(SystemExit, match="re-convert"):
        VLAMetadata.from_json_dict(payload)


def test_metadata_rejects_unknown_schema_version() -> None:
    payload = metadata().to_json_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        VLAMetadata.from_json_dict(payload)


def test_tokenizer_manifest_is_per_trunk() -> None:
    assert tokenizer_manifest(VLAFamily.GEMMA_FLOW) == GEMMA_TOKENIZER_FILES
    assert tokenizer_manifest(VLAFamily.GEMMA_AR) == GEMMA_TOKENIZER_FILES
    for family in (
        VLAFamily.MOLMO2_AR,
        VLAFamily.MOLMOACT2_FLOW,
        VLAFamily.MOLMOACT2_AR,
        VLAFamily.MOLMOACT2_JOINT,
    ):
        assert tokenizer_manifest(family) == MOLMO_TOKENIZER_FILES


def test_write_checkpoint_serializes_parts_and_tokenizer(tmp_path: Path) -> None:
    target = write_tiny_checkpoint(tmp_path)
    meta = validate_checkpoint(target)
    assert meta.family is VLAFamily.GEMMA_FLOW
    text, vision = backbone_parts()
    assert torch.equal(
        load_file(str(target / BACKBONE_TEXT_FILENAME))[
            "language_model.embed_tokens.weight"
        ],
        text["language_model.embed_tokens.weight"],
    )
    assert torch.equal(
        load_file(str(target / BACKBONE_VISION_FILENAME))["vision_tower.patch.weight"],
        vision["vision_tower.patch.weight"],
    )
    for name in GEMMA_TOKENIZER_FILES:
        linked = target / "tokenizer" / name
        assert (
            linked.stat().st_ino == (tmp_path / "tokenizer_source" / name).stat().st_ino
        )
    assert not (target.parent / "ckpt.tmp").exists()


def test_write_checkpoint_links_part_files(tmp_path: Path) -> None:
    """The frozen-part dedup: a Path-form part hard-links its source —
    conversion links the import, training saves link the previous."""
    first = write_tiny_checkpoint(tmp_path)
    second = tmp_path / "ckpt2"
    write_checkpoint(
        second,
        metadata=metadata(),
        components=component_states(),
        backbone_text=first / BACKBONE_TEXT_FILENAME,
        backbone_vision=first / BACKBONE_VISION_FILENAME,
        tokenizer_files=tokenizer_source(tmp_path),
    )
    validate_checkpoint(second)
    for filename in (BACKBONE_TEXT_FILENAME, BACKBONE_VISION_FILENAME):
        assert (second / filename).stat().st_ino == (first / filename).stat().st_ino


def test_write_checkpoint_refuses_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "ckpt"
    target.mkdir()
    text, vision = backbone_parts()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        write_checkpoint(
            target,
            metadata=metadata(),
            components=component_states(),
            backbone_text=text,
            backbone_vision=vision,
            tokenizer_files=tokenizer_source(tmp_path),
        )


def test_write_checkpoint_refuses_component_mismatch(tmp_path: Path) -> None:
    text, vision = backbone_parts()
    with pytest.raises(ValueError, match="component mismatch"):
        write_checkpoint(
            tmp_path / "ckpt",
            metadata=metadata(),
            components={"prompt": {"w": torch.zeros(1)}},  # flow_decoder missing
            backbone_text=text,
            backbone_vision=vision,
            tokenizer_files=tokenizer_source(tmp_path),
        )


def test_validate_rejects_stray_weight_file(tmp_path: Path) -> None:
    target = write_tiny_checkpoint(tmp_path)
    save_file({"x": torch.zeros(1)}, str(target / "stray.safetensors"))
    with pytest.raises(SystemExit, match="undeclared weight files"):
        validate_checkpoint(target)


def test_validate_rejects_missing_tokenizer_manifest(tmp_path: Path) -> None:
    target = write_tiny_checkpoint(tmp_path)
    (target / "tokenizer" / "chat_template.jinja").unlink()
    with pytest.raises(SystemExit, match=r"chat_template\.jinja"):
        validate_checkpoint(target)


def test_validate_rejects_missing_part_file(tmp_path: Path) -> None:
    target = write_tiny_checkpoint(tmp_path)
    (target / BACKBONE_VISION_FILENAME).unlink()
    with pytest.raises(SystemExit, match="backbone_vision"):
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


@pytest.fixture(scope="module")
def gemma_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("convert-gemma")
    trunk = write_gemma_trunk(root / "trunk")
    # The gemma tokenizer manifest: the hermetic trunk carries stub
    # processor files (vla_fixtures) so conversion is self-contained.
    legacy = write_gemma_flow_legacy(root / "legacy", trunk)
    (legacy / "optimizer.pt").write_bytes(b"opt")
    return trunk, legacy


def test_convert_gemma_flow(gemma_pair: tuple[Path, Path], tmp_path: Path) -> None:
    trunk, legacy = gemma_pair
    converted = tmp_path / "converted"
    meta = convert(legacy, converted)
    assert meta.family is VLAFamily.GEMMA_FLOW
    assert meta.chunk_size == 10
    assert meta.action_dim == ACTION_DIM
    # Pristine trunk: imported wholesale, both flags False, the
    # artifact's config carried VERBATIM.
    assert meta.backbone_text_trained is False
    assert meta.backbone_vision_trained is False
    assert meta.backbone_config == json.loads((trunk / "config.json").read_text())
    assert meta.serving == {"kind": "flow", "num_steps": 5, "method": "heun"}
    assert meta.components["flow_decoder"]["config"]["kind"] == "flow"
    validate_checkpoint(converted)
    # Component weight files are hard links to the source, never rewrites.
    assert (converted / "flow_decoder.safetensors").samefile(
        legacy / "expert.safetensors",
    )
    assert (converted / "prompt.safetensors").samefile(legacy / "prompt.safetensors")
    assert (converted / "optimizer.pt").exists()
    # The trunk import: per-part files carrying our key names, and the
    # tokenizer manifest linked from the artifact.
    text = load_file(str(converted / BACKBONE_TEXT_FILENAME), device="cpu")
    vision = load_file(str(converted / BACKBONE_VISION_FILENAME), device="cpu")
    assert all(key.startswith(("language_model.", "embed_vision.")) for key in text)
    assert all(key.startswith("vision_tower.") for key in vision)
    source = load_file(str(trunk / "model.safetensors"), device="cpu")
    key = "language_model.embed_tokens.weight"
    assert torch.equal(text[key], source[f"model.{key}"])
    assert (converted / "tokenizer" / "tokenizer.json").samefile(
        trunk / "tokenizer.json",
    )


def test_convert_trained_trunk_partitions_the_legacy_file(
    gemma_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """A legacy backbone.safetensors splits into the per-part files with
    both flags conservatively True (the legacy format recorded one
    fact for the whole trunk)."""
    import shutil

    _trunk, legacy = gemma_pair
    trained = tmp_path / "legacy_trained"
    shutil.copytree(legacy, trained)
    snapshot = {
        "language_model.embed_tokens.weight": torch.randn(4, 2),
        "vision_tower.patch.weight": torch.randn(2, 2),
    }
    save_file(snapshot, str(trained / "backbone.safetensors"))
    meta = convert(trained, tmp_path / "converted")
    assert meta.backbone_text_trained is True
    assert meta.backbone_vision_trained is True
    text = load_file(str(tmp_path / "converted" / BACKBONE_TEXT_FILENAME))
    vision = load_file(str(tmp_path / "converted" / BACKBONE_VISION_FILENAME))
    assert torch.equal(
        text["language_model.embed_tokens.weight"],
        snapshot["language_model.embed_tokens.weight"],
    )
    assert torch.equal(
        vision["vision_tower.patch.weight"],
        snapshot["vision_tower.patch.weight"],
    )


def test_convert_is_idempotent(gemma_pair: tuple[Path, Path], tmp_path: Path) -> None:
    _, legacy = gemma_pair
    first = tmp_path / "one"
    second = tmp_path / "two"
    convert(legacy, first)
    convert(legacy, second)
    assert (first / "metadata.json").read_bytes() == (
        second / "metadata.json"
    ).read_bytes()
    for name in (
        "flow_decoder.safetensors",
        "prompt.safetensors",
        BACKBONE_TEXT_FILENAME,
        BACKBONE_VISION_FILENAME,
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_convert_replace_stats(gemma_pair: tuple[Path, Path], tmp_path: Path) -> None:
    _, legacy = gemma_pair
    replacement = stats().state_dict()
    replacement["action"]["mean"] = [42.0] * ACTION_DIM
    table = tmp_path / "table.json"
    table.write_text(json.dumps(replacement))
    meta = convert(legacy, tmp_path / "converted", replace_stats=table)
    assert meta.stats.action_mean == (42.0,) * ACTION_DIM
    assert meta.stats_note is not None
    assert "REPLACED" in meta.stats_note


def test_convert_refuses_pre_format_3(
    gemma_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    import shutil

    _, legacy = gemma_pair
    old = tmp_path / "legacy_old"
    shutil.copytree(legacy, old)
    config = json.loads((old / "bijou_config.json").read_text())
    config["format"] = 2
    (old / "bijou_config.json").write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="format 2"):
        convert(old, tmp_path / "converted")


def test_convert_refuses_vla_directories(
    gemma_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """A VLA-format source (schema 1 OR 2) is not a legacy checkpoint —
    schema-1 artifacts re-convert from their original sources."""
    _, legacy = gemma_pair
    converted = tmp_path / "converted"
    convert(legacy, converted)
    with pytest.raises(SystemExit, match="re-convert"):
        convert(converted, tmp_path / "again")
