"""bijou.convert_v1 — the schema-1 → schema-2 VLA checkpoint upgrade:
both trunk arms (a trained ``backbone.safetensors`` partitions through
the audited splitters; a pristine ``backbone/`` mirror imports from
itself, no cache resolution), verbatim metadata carry, component
weight-file linking, idempotence, and the refusal fence (schema 2,
unknown schemas, legacy ``bijou_config.json`` directories)."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from vla_fixtures import write_gemma_trunk

from bijou.checkpoint import (
    BACKBONE_TEXT_FILENAME,
    BACKBONE_VISION_FILENAME,
    GEMMA_TOKENIZER_FILES,
)
from bijou.convert_v1 import convert

ACTION_DIM = 6


def stats_dict() -> dict:
    return {
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
    }


def v1_metadata(*, family: str, backbone_id: str, trained: bool) -> dict:
    return {
        "schema_version": 1,
        "family": family,
        "spec": {"chunk_size": 5, "action_dim": ACTION_DIM},
        "backbone": {"id": backbone_id, "depth": "prefix", "trained": trained},
        "objective": {"kind": "flow"},
        "serving": {"kind": "flow", "num_steps": 5, "method": "heun"},
        "components": {
            "prompt": {"config": {"kind": "gemma4"}, "weights": True},
            "flow_decoder": {"config": {"kind": "flow"}, "weights": True},
        },
        "artifacts": {"probe": "some/artifact"},
        "stats": stats_dict(),
        "per_dataset_stats": {"repo/a": stats_dict()},
        "train_args": {"seed": 7},
        "step": 42,
    }


def write_v1_components(directory: Path) -> dict[str, dict[str, torch.Tensor]]:
    states = {
        "prompt": {"state_proj.weight": torch.full((2, 2), 3.0)},
        "flow_decoder": {"proj.weight": torch.full((2, 2), 5.0)},
    }
    for name, state in states.items():
        save_file(state, str(directory / f"{name}.safetensors"))
    return states


@pytest.fixture(scope="module")
def gemma_trunk(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_gemma_trunk(tmp_path_factory.mktemp("convert-v1") / "trunk")


def write_v1_trained_gemma(directory: Path, trunk: Path) -> Path:
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps(
            v1_metadata(family="gemma_flow", backbone_id=str(trunk), trained=True),
        ),
    )
    write_v1_components(directory)
    save_file(
        {
            "language_model.embed_tokens.weight": torch.randn(4, 2),
            "vision_tower.patch.weight": torch.randn(2, 2),
        },
        str(directory / "backbone.safetensors"),
    )
    return directory


def test_trained_gemma_upgrades(gemma_trunk: Path, tmp_path: Path) -> None:
    source = write_v1_trained_gemma(tmp_path / "v1", gemma_trunk)
    meta = convert(source, tmp_path / "v2")
    assert meta.backbone_text_trained is True
    assert meta.backbone_vision_trained is True
    # The single v1 file partitioned into the per-part files.
    v1_state = load_file(str(source / "backbone.safetensors"))
    text = load_file(str(tmp_path / "v2" / BACKBONE_TEXT_FILENAME))
    vision = load_file(str(tmp_path / "v2" / BACKBONE_VISION_FILENAME))
    assert torch.equal(
        text["language_model.embed_tokens.weight"],
        v1_state["language_model.embed_tokens.weight"],
    )
    assert torch.equal(
        vision["vision_tower.patch.weight"],
        v1_state["vision_tower.patch.weight"],
    )
    # Component files hard-link (never rewritten).
    for name in ("prompt", "flow_decoder"):
        assert (tmp_path / "v2" / f"{name}.safetensors").stat().st_nlink >= 2
    # config.json verbatim from the artifact; tokenizer/ carried.
    trunk_config = json.loads((gemma_trunk / "config.json").read_text())
    assert meta.backbone_config == trunk_config
    for name in GEMMA_TOKENIZER_FILES:
        assert (tmp_path / "v2" / "tokenizer" / name).is_file()
    # Everything else carries verbatim.
    assert meta.step == 42
    assert meta.train_args == {"seed": 7}
    assert meta.artifacts == {"probe": "some/artifact"}
    assert meta.objective == {"kind": "flow"}
    assert meta.serving == {"kind": "flow", "num_steps": 5, "method": "heun"}


def test_pristine_imports_from_its_own_mirror(
    gemma_trunk: Path,
    tmp_path: Path,
) -> None:
    """The pristine arm is self-contained: the backbone/ mirror supplies
    weights, config.json AND tokenizer files — a backbone id that
    resolves nowhere proves no cache lookup happens."""
    source = tmp_path / "v1"
    source.mkdir()
    (source / "metadata.json").write_text(
        json.dumps(
            v1_metadata(
                family="gemma_flow",
                backbone_id="nonexistent/never-cached",
                trained=False,
            ),
        ),
    )
    write_v1_components(source)
    shutil.copytree(gemma_trunk, source / "backbone")
    meta = convert(source, tmp_path / "v2")
    assert meta.backbone_text_trained is False
    assert meta.backbone_vision_trained is False
    text = load_file(str(tmp_path / "v2" / BACKBONE_TEXT_FILENAME))
    assert len(text) > 0
    assert meta.backbone_config == json.loads(
        (gemma_trunk / "config.json").read_text(),
    )


def test_trained_molmo_strips_part_prefixes(tmp_path: Path) -> None:
    """Molmo trunks partition by the wrapper's text./vision. prefixes,
    stripped to the part files' submodule-level keys."""
    snapshot = tmp_path / "artifact"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"model_type": "molmoact2"}))
    (snapshot / "tokenizer.json").write_text("{}")
    source = tmp_path / "v1"
    source.mkdir()
    meta_dict = v1_metadata(
        family="molmoact2_flow",
        backbone_id=str(snapshot),
        trained=True,
    )
    meta_dict["components"]["prompt"]["weights"] = False
    (source / "metadata.json").write_text(json.dumps(meta_dict))
    save_file(
        {"proj.weight": torch.full((2, 2), 5.0)},
        str(source / "flow_decoder.safetensors"),
    )
    save_file(
        {
            "text.lm_head.weight": torch.randn(4, 2),
            "vision.image_vit.patch.weight": torch.randn(2, 2),
        },
        str(source / "backbone.safetensors"),
    )
    meta = convert(source, tmp_path / "v2")
    text = load_file(str(tmp_path / "v2" / BACKBONE_TEXT_FILENAME))
    vision = load_file(str(tmp_path / "v2" / BACKBONE_VISION_FILENAME))
    assert set(text) == {"lm_head.weight"}
    assert set(vision) == {"image_vit.patch.weight"}
    assert meta.backbone_config == {"model_type": "molmoact2"}


def test_pre_rename_ar_objective_translates(
    gemma_trunk: Path,
    tmp_path: Path,
) -> None:
    """A v1-era AR objective recorded aux_loss_weight (pre-32149df);
    the upgrade renames it so the trained mix survives — the current
    parser would silently default narration_weight to 1.0."""
    source = write_v1_trained_gemma(tmp_path / "v1", gemma_trunk)
    payload = json.loads((source / "metadata.json").read_text())
    payload["family"] = "gemma_ar"
    payload["objective"] = {"kind": "ar", "aux_loss_weight": 0.5}
    payload["serving"] = {"kind": "ar"}
    (source / "metadata.json").write_text(json.dumps(payload))
    meta = convert(source, tmp_path / "v2")
    assert meta.objective == {"kind": "ar", "narration_weight": 0.5}


def test_optimizer_carried(gemma_trunk: Path, tmp_path: Path) -> None:
    source = write_v1_trained_gemma(tmp_path / "v1", gemma_trunk)
    (source / "optimizer.pt").write_bytes(b"opt-state")
    convert(source, tmp_path / "v2")
    assert (tmp_path / "v2" / "optimizer.pt").read_bytes() == b"opt-state"


def test_upgrade_is_idempotent(gemma_trunk: Path, tmp_path: Path) -> None:
    source = write_v1_trained_gemma(tmp_path / "v1", gemma_trunk)
    convert(source, tmp_path / "one")
    convert(source, tmp_path / "two")
    for name in (
        "metadata.json",
        "prompt.safetensors",
        "flow_decoder.safetensors",
        BACKBONE_TEXT_FILENAME,
        BACKBONE_VISION_FILENAME,
    ):
        assert (tmp_path / "one" / name).read_bytes() == (
            tmp_path / "two" / name
        ).read_bytes()


def test_refuses_schema_2_output(gemma_trunk: Path, tmp_path: Path) -> None:
    source = write_v1_trained_gemma(tmp_path / "v1", gemma_trunk)
    convert(source, tmp_path / "v2")
    with pytest.raises(SystemExit, match="already current"):
        convert(tmp_path / "v2", tmp_path / "again")


def test_refuses_unknown_schema(gemma_trunk: Path, tmp_path: Path) -> None:
    source = write_v1_trained_gemma(tmp_path / "v1", gemma_trunk)
    payload = json.loads((source / "metadata.json").read_text())
    payload["schema_version"] = 99
    (source / "metadata.json").write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="schema_version 99"):
        convert(source, tmp_path / "v2")


def test_refuses_legacy_directories(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "bijou_config.json").write_text("{}")
    with pytest.raises(SystemExit, match="convert_legacy"):
        convert(legacy, tmp_path / "v2")
