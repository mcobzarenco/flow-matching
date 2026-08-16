"""The per-trunk backbone importers and from-files loaders (checkpoint
schema v2's translate-once contract): the import audit proves the key
PARTITION — text + vision (+ expert on MolmoAct2) + known-skipped
exactly cover every source shard key, refusing unclassified tensors by
name — and the from-files loaders reproduce the dir-glob HF mounts
bitwise from the partitioned files (extraction is key-filtering, never
value change)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from vla_fixtures import write_gemma_trunk

from bijou.modelling.gemma4.loading import (
    import_backbone_state as import_gemma_state,
)
from bijou.modelling.gemma4.loading import (
    load_config as load_gemma_config,
)
from bijou.modelling.gemma4.loading import (
    load_model as load_gemma_model,
)
from bijou.modelling.gemma4.loading import (
    load_model_from_files as load_gemma_from_files,
)
from bijou.modelling.molmo2.loading import (
    import_backbone_state as import_molmo2_state,
)
from bijou.modelling.molmo2.loading import load_config as load_molmo2_config
from bijou.modelling.molmo2.model import load_model as load_molmo2_model
from bijou.modelling.molmo2.model import (
    load_model_from_files as load_molmo2_from_files,
)
from bijou.modelling.molmo2.testing import write_tiny_text_checkpoint


@pytest.fixture(scope="module")
def gemma_trunk(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_gemma_trunk(tmp_path_factory.mktemp("import-gemma") / "trunk")


@pytest.fixture(scope="module")
def molmo2_trunk(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_tiny_text_checkpoint(
        tmp_path_factory.mktemp("import-molmo2") / "trunk",
    )


# ---------------------------------------------------------------------------
# The Gemma partition and its audit


def test_gemma_import_partitions_by_lr_group(gemma_trunk: Path) -> None:
    """text = language_model.* + embed_vision.* (the multimodal projector
    trains under the TEXT lr); vision = vision_tower.* — exactly the
    backbone_vision LR group's members."""
    imported = import_gemma_state(gemma_trunk)
    assert all(
        key.startswith(("language_model.", "embed_vision.")) for key in imported.text
    )
    assert len(imported.vision) > 0
    assert all(key.startswith("vision_tower.") for key in imported.vision)
    assert any(key.startswith("embed_vision.") for key in imported.text)
    # Bytes verbatim: spot-check one tensor against the source read.
    model_file = gemma_trunk / "model.safetensors"
    from safetensors.torch import load_file

    source = load_file(str(model_file), device="cpu")
    key, tensor = next(iter(imported.text.items()))
    assert torch.equal(tensor, source[f"model.{key}"])
    assert tensor.dtype == source[f"model.{key}"].dtype


def test_gemma_import_skips_known_quirks(
    gemma_trunk: Path,
    tmp_path: Path,
) -> None:
    """Audio towers, KV-shared duplicate K/V weights and a tied lm_head
    are known-skipped — recorded by name, never imported."""
    trunk = tmp_path / "trunk"
    shutil.copytree(gemma_trunk, trunk)
    config = load_gemma_config(trunk)
    shared_layer = config.text.first_kv_shared_layer_idx  # a KV-shared index
    extras = {
        "model.audio_tower.conv.weight": torch.zeros(2),
        "model.embed_audio.weight": torch.zeros(2),
        f"model.language_model.layers.{shared_layer}.self_attn.k_proj.weight": (
            torch.zeros(2)
        ),
        "lm_head.weight": torch.zeros(2),
    }
    save_file(extras, str(trunk / "extra.safetensors"))
    imported = import_gemma_state(trunk)
    assert set(imported.skipped) == set(extras)
    for key in extras:
        assert key.removeprefix("model.") not in imported.text
        assert key.removeprefix("model.") not in imported.vision


def test_gemma_import_refuses_unclassified_keys(
    gemma_trunk: Path,
    tmp_path: Path,
) -> None:
    trunk = tmp_path / "trunk"
    shutil.copytree(gemma_trunk, trunk)
    save_file(
        {"model.mystery_tower.weight": torch.zeros(2)},
        str(trunk / "extra.safetensors"),
    )
    with pytest.raises(SystemExit, match="mystery_tower"):
        import_gemma_state(trunk)


def test_gemma_from_files_matches_dir_glob(
    gemma_trunk: Path,
    tmp_path: Path,
) -> None:
    """The from-files mount == the HF dir-glob mount, bitwise, at both
    depths (FULL, and PREFIX with the packed-PLE slicing)."""
    config = load_gemma_config(gemma_trunk)
    imported = import_gemma_state(gemma_trunk)
    text_file = tmp_path / "backbone_text.safetensors"
    vision_file = tmp_path / "backbone_vision.safetensors"
    save_file(
        {k: v.contiguous() for k, v in imported.text.items()},
        str(text_file),
    )
    save_file(
        {k: v.contiguous() for k, v in imported.vision.items()},
        str(vision_file),
    )
    prefix = config.text.first_kv_shared_layer_idx
    for truncate in (None, prefix):
        reference = load_gemma_model(
            gemma_trunk,
            device="cpu",
            truncate_layers=truncate,
        )
        from_files = load_gemma_from_files(
            config,
            text_file=text_file,
            vision_file=vision_file,
            device="cpu",
            truncate_layers=truncate,
        )
        reference_state = reference.state_dict()
        from_files_state = from_files.state_dict()
        assert set(reference_state) == set(from_files_state)
        for name, tensor in reference_state.items():
            assert tensor.dtype == from_files_state[name].dtype, name
            assert torch.equal(tensor, from_files_state[name]), name


# ---------------------------------------------------------------------------
# The Molmo2 / MolmoAct2 partition and its audit


def test_molmo2_import_partitions_by_role(
    molmo2_trunk: Path,
    tmp_path: Path,
) -> None:
    """text = transformer.* + the untied lm_head; vision = tower +
    connector (the backbone_vision LR group); expert = the MolmoAct2
    action expert; the persisted RoPE table is known-skipped."""
    trunk = tmp_path / "trunk"
    shutil.copytree(molmo2_trunk, trunk)
    save_file(
        {
            "model.action_expert.blocks.0.attn.weight": torch.zeros(2),
            "model.transformer.rotary_emb.inv_freq": torch.zeros(2),
        },
        str(trunk / "extra.safetensors"),
    )
    imported = import_molmo2_state(trunk)
    assert "lm_head.weight" in imported.text
    assert all(
        key == "lm_head.weight" or key.startswith("transformer.")
        for key in imported.text
    )
    assert "transformer.rotary_emb.inv_freq" not in imported.text
    assert len(imported.vision) > 0
    assert set(imported.expert) == {"blocks.0.attn.weight"}
    assert imported.skipped == ("model.transformer.rotary_emb.inv_freq",)


def test_molmo2_import_refuses_unclassified_keys(
    molmo2_trunk: Path,
    tmp_path: Path,
) -> None:
    trunk = tmp_path / "trunk"
    shutil.copytree(molmo2_trunk, trunk)
    save_file(
        {"model.depth_tower.weight": torch.zeros(2), "loose.weight": torch.zeros(2)},
        str(trunk / "extra.safetensors"),
    )
    with pytest.raises(SystemExit, match="depth_tower") as excinfo:
        import_molmo2_state(trunk)
    assert "loose.weight" in str(excinfo.value)


def test_molmo2_from_files_matches_dir_glob(
    molmo2_trunk: Path,
    tmp_path: Path,
) -> None:
    config = load_molmo2_config(molmo2_trunk)
    imported = import_molmo2_state(molmo2_trunk)
    text_file = tmp_path / "backbone_text.safetensors"
    vision_file = tmp_path / "backbone_vision.safetensors"
    save_file(
        {k: v.contiguous() for k, v in imported.text.items()},
        str(text_file),
    )
    save_file(
        {k: v.contiguous() for k, v in imported.vision.items()},
        str(vision_file),
    )
    reference = load_molmo2_model(molmo2_trunk, device="cpu", dtype=torch.bfloat16)
    from_files = load_molmo2_from_files(
        config,
        text_file=text_file,
        vision_file=vision_file,
        device="cpu",
        dtype=torch.bfloat16,
    )
    reference_state = reference.state_dict()
    from_files_state = from_files.state_dict()
    assert set(reference_state) == set(from_files_state)
    for name, tensor in reference_state.items():
        assert tensor.dtype == from_files_state[name].dtype, name
        assert torch.equal(tensor, from_files_state[name]), name
