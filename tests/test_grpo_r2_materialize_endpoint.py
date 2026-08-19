"""GRPO endpoint materializer oracle (the boundary launcher's CPU seam).

The loop banks trainable-only overlays (``step_NNNN.pt``); the A3.4
boundary legs serve a self-contained VLA dir through the anchors' path.
``grpo_r2_materialize_endpoint`` is the file-level bridge — this suite
pins it on the tiny VLA fixture:

1. round-trip: overlay tensors land (cast to the base dtype), every
   untouched tensor and file is bit-identical to the base, the result
   validates as a checkpoint, provenance sidecar recorded;
2. refusals: existing destination, off-surface keys (non-``text.``),
   keys absent from the base, shape mismatches, non-loop payloads —
   each loud, none silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from bijou.checkpoint import validate_checkpoint
from bijou.testing import write_tiny_molmoact2_release
from fontaine.scripts.grpo_r2_materialize_endpoint import materialize


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("grpo-materialize") / "tiny"
    write_tiny_molmoact2_release(root)
    return root / "checkpoint_vla"


def write_overlay(
    path: Path,
    trainable: dict[str, torch.Tensor],
    step: int = 10,
    **extra: object,
) -> Path:
    payload: dict[str, object] = {"step": step, "trainable": trainable}
    payload.update(extra)
    torch.save(payload, path)
    return path


def base_text_keys(base: Path) -> dict[str, torch.Tensor]:
    return load_file(str(base / "backbone_text.safetensors"))


def test_round_trip(tiny_base: Path, tmp_path: Path) -> None:
    base_text = base_text_keys(tiny_base)
    # Two real keys moved (fp32, as the loop trains them), rest untouched.
    moved = dict(list(base_text.items())[:2])
    trainable = {f"text.{key}": (tensor.float() + 1.0) for key, tensor in moved.items()}
    overlay = write_overlay(tmp_path / "step_0010.pt", trainable)
    out = tmp_path / "endpoint_step_0010"

    record = materialize(tiny_base, overlay, out)

    validate_checkpoint(out)
    merged = load_file(str(out / "backbone_text.safetensors"))
    assert set(merged) == set(base_text)
    for key, original in base_text.items():
        if f"text.{key}" in trainable:
            expected = trainable[f"text.{key}"].to(original.dtype)
            assert torch.equal(merged[key], expected)
            assert merged[key].dtype == original.dtype
            assert not torch.equal(merged[key], original)
        else:
            assert torch.equal(merged[key], original)
    # Untouched part files are the base's bytes (hard-link or copy).
    assert (out / "backbone_vision.safetensors").read_bytes() == (
        tiny_base / "backbone_vision.safetensors"
    ).read_bytes()
    assert record["grpo_step"] == 10
    assert record["replaced_keys"] == 2
    import json

    sidecar = json.loads((out / "grpo_overlay.json").read_text())
    assert sidecar["base"] == str(tiny_base)
    assert sidecar["grpo_step"] == 10


def test_refuses_existing_destination(tiny_base: Path, tmp_path: Path) -> None:
    key = next(iter(base_text_keys(tiny_base)))
    overlay = write_overlay(
        tmp_path / "o.pt",
        {f"text.{key}": base_text_keys(tiny_base)[key].float()},
    )
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        materialize(tiny_base, overlay, out)


def test_refuses_off_surface_keys(tiny_base: Path, tmp_path: Path) -> None:
    overlay = write_overlay(
        tmp_path / "o.pt",
        {"vision.blocks.0.weight": torch.zeros(2)},
    )
    with pytest.raises(SystemExit, match="outside the text surface"):
        materialize(tiny_base, overlay, tmp_path / "out")


def test_refuses_unknown_base_key(tiny_base: Path, tmp_path: Path) -> None:
    overlay = write_overlay(
        tmp_path / "o.pt",
        {"text.no_such.tensor": torch.zeros(2)},
    )
    with pytest.raises(SystemExit, match="no tensor"):
        materialize(tiny_base, overlay, tmp_path / "out")


def test_refuses_shape_mismatch(tiny_base: Path, tmp_path: Path) -> None:
    key = next(iter(base_text_keys(tiny_base)))
    overlay = write_overlay(
        tmp_path / "o.pt",
        {f"text.{key}": torch.zeros(1, 2, 3)},
    )
    with pytest.raises(SystemExit, match="shape"):
        materialize(tiny_base, overlay, tmp_path / "out")


def test_refuses_non_loop_payload(tiny_base: Path, tmp_path: Path) -> None:
    path = tmp_path / "not_a_loop_save.pt"
    torch.save({"weights": {}}, path)
    with pytest.raises(SystemExit, match="not a GRPO loop save"):
        materialize(tiny_base, path, tmp_path / "out")

    no_step = tmp_path / "no_step.pt"
    torch.save({"trainable": {"text.x": torch.zeros(1)}}, no_step)
    with pytest.raises(SystemExit, match="no 'step'"):
        materialize(tiny_base, no_step, tmp_path / "out")
