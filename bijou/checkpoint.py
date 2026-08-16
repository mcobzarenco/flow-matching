"""The VLA checkpoint format — schema and IO toolkit.

A VLA checkpoint is a SELF-CONTAINED directory (schema 2 — the full
import: no HF-layout knowledge in any load path):

```
checkpoint/
  metadata.json               # VLAMetadata (schema_version 2, family, …)
  backbone_text.safetensors   # text stack (+ untied lm_head), OUR keys
  backbone_vision.safetensors # vision tower + connector — exactly the
                              #   backbone_vision LR group's members
  <component>.safetensors     # one per family-declared component
  tokenizer/                  # the per-trunk consumed artifact files
  optimizer.pt                # optional (run-seeding checkpoints)
```

Key translation from released HF layouts happens ONCE, at import
(``bijou.convert_legacy``, ``bijou.convert_molmoact2``, the trunk
importers in ``bijou.modelling.*.loading``); every load of these files
is a plain strict ``load_state_dict``. The backbone part files are
ALWAYS present. Presence is not a signal: ``backbone.text_trained`` /
``backbone.vision_trained`` in the metadata are the explicit facts —
a frozen part's file is byte-identical to its parent's and is
HARD-LINKED from it (conversion links the import; every training save
links the previous save while the part stays frozen), so dedup lives
inside our world and the HF cache is deletable.

``metadata.json``'s ``backbone.config`` carries the source artifact's
``config.json`` contents VERBATIM — families parse it at load with the
existing ``Gemma4Config.from_dict``/``Molmo2Config.from_dict``, so no
second config parser can drift. ``backbone.id`` is provenance;
``backbone.depth`` records the depth the part files were saved at
(``full``, or the gemma ``prefix`` mount — what the text file's layer
set contains).

``tokenizer/`` carries exactly the files the trunk's collators and
decoders read (:func:`tokenizer_manifest`): Molmo-trunk families read
``tokenizer.json`` alone; Gemma families read the transformers-facing
set (tokenizer.json, tokenizer_config.json, processor_config.json,
chat_template.jinja — verified sufficient for
``AutoProcessor``/``AutoTokenizer.from_pretrained`` on a bare
directory).

Transfer caveat: local ``rsync`` needs ``-H`` to preserve hard links;
without it the copy is correct but fully materialized.

The metadata's ``objective`` and ``serving`` values are TAGGED DICTS at
this layer — families parse them into their typed payloads; the schema
stays family-agnostic. ``components`` maps each component name to
``{"config": {…}, "weights": bool}``: a component with ``weights``
owns ``<name>.safetensors``; a parameterless component (a decoder
whose trainable surface is the trunk itself) records its config with
``weights: false`` and no file — explicit, never inferred from file
presence. The family's ``checkpoint_components()`` declares the same
names on the write side.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import save_file
from torch import Tensor

from .data import DatasetStats
from .sections import BackboneFiles
from .vla import VLAFamily

SCHEMA_VERSION = 2
METADATA_FILENAME = "metadata.json"
BACKBONE_TEXT_FILENAME = "backbone_text.safetensors"
BACKBONE_VISION_FILENAME = "backbone_vision.safetensors"
TOKENIZER_DIRNAME = "tokenizer"

# The transformers-facing artifact set the Gemma prompt path reads
# (AutoProcessor/AutoTokenizer.from_pretrained on the bare directory —
# verified sufficient standalone, no config.json needed).
GEMMA_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
)
# The Molmo trunks tokenize natively through the tokenizers backend.
MOLMO_TOKENIZER_FILES = ("tokenizer.json",)


def tokenizer_manifest(family: VLAFamily) -> tuple[str, ...]:
    """The tokenizer/ files a family's collators and decoders consume —
    the required minimum a checkpoint must carry (extra files are
    tolerated; missing ones fail validation)."""
    match family:
        case VLAFamily.GEMMA_FLOW | VLAFamily.GEMMA_AR:
            return GEMMA_TOKENIZER_FILES
        case (
            VLAFamily.MOLMO2_AR
            | VLAFamily.MOLMOACT2_FLOW
            | VLAFamily.MOLMOACT2_AR
            | VLAFamily.MOLMOACT2_JOINT
        ):
            return MOLMO_TOKENIZER_FILES


@dataclass(frozen=True, slots=True)
class VLAMetadata:
    """``metadata.json``, both directions (validating parse on the way
    in, explicit serialization on the way out).

    ``backbone_config`` is the source artifact's ``config.json``
    contents verbatim (parsed at load with the trunk's own
    ``from_dict``); ``backbone_text_trained``/``backbone_vision_trained``
    are the explicit per-part facts (True = the part's weights differ
    from the pristine artifact — trained in some run, this one or an
    ancestor). ``train_args`` is the run's full CLI record, verbatim
    provenance; ``stats`` is the count-weighted aggregate normalization
    table and ``per_dataset_stats`` the per-repo tables (keyed by repo
    id — genuinely dynamic); ``stats_note`` records a stats REPLACEMENT
    (the ``--replace-stats`` conversion op) so a substituted table is
    never mistaken for a trained-with one."""

    family: VLAFamily
    chunk_size: int
    action_dim: int
    backbone_id: str
    backbone_depth: str
    backbone_config: dict[str, Any]
    backbone_text_trained: bool
    backbone_vision_trained: bool
    objective: dict[str, Any]
    serving: dict[str, Any]
    # name → {"config": {…}, "weights": bool} (module docstring)
    components: dict[str, dict[str, Any]]
    artifacts: dict[str, str]
    stats: DatasetStats
    per_dataset_stats: dict[str, DatasetStats]
    train_args: dict[str, Any]
    step: int
    stats_note: str | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "family": self.family.value,
            "spec": {"chunk_size": self.chunk_size, "action_dim": self.action_dim},
            "backbone": {
                "id": self.backbone_id,
                "depth": self.backbone_depth,
                "config": self.backbone_config,
                "text_trained": self.backbone_text_trained,
                "vision_trained": self.backbone_vision_trained,
            },
            "objective": self.objective,
            "serving": self.serving,
            "components": self.components,
            "artifacts": self.artifacts,
            "stats": self.stats.state_dict(),
            "per_dataset_stats": {
                repo_id: stats.state_dict()
                for repo_id, stats in sorted(self.per_dataset_stats.items())
            },
            "train_args": self.train_args,
            "step": self.step,
            **({"stats_note": self.stats_note} if self.stats_note is not None else {}),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> VLAMetadata:
        version = data.get("schema_version")
        if version == 1:
            raise SystemExit(
                "checkpoint schema_version 1 — the snapshot-mirror era; "
                "re-convert from the original source (bijou.convert_legacy "
                "on the legacy directory, or bijou.convert_molmoact2 on "
                "the HF release)",
            )
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema_version {version!r} != supported "
                f"{SCHEMA_VERSION} — not a VLA checkpoint (legacy "
                "bijou_config.json directories convert via "
                "bijou.convert_legacy)",
            )
        backbone = data["backbone"]
        spec = data["spec"]
        for name, record in data["components"].items():
            if set(record) != {"config", "weights"}:
                raise ValueError(
                    f"component {name!r} must be {{'config', 'weights'}}, "
                    f"got keys {sorted(record)}",
                )
        return cls(
            family=VLAFamily(data["family"]),
            chunk_size=int(spec["chunk_size"]),
            action_dim=int(spec["action_dim"]),
            backbone_id=str(backbone["id"]),
            backbone_depth=str(backbone["depth"]),
            backbone_config=dict(backbone["config"]),
            backbone_text_trained=bool(backbone["text_trained"]),
            backbone_vision_trained=bool(backbone["vision_trained"]),
            objective=dict(data["objective"]),
            serving=dict(data["serving"]),
            components={k: dict(v) for k, v in data["components"].items()},
            artifacts={k: str(v) for k, v in data["artifacts"].items()},
            stats=DatasetStats.from_state_dict(data["stats"]),
            per_dataset_stats={
                repo_id: DatasetStats.from_state_dict(table)
                for repo_id, table in data["per_dataset_stats"].items()
            },
            train_args=dict(data["train_args"]),
            step=int(data["step"]),
            stats_note=data.get("stats_note"),
        )


def read_metadata(checkpoint: Path) -> VLAMetadata:
    """Parse a checkpoint directory's ``metadata.json`` (loud on legacy
    directories: they carry ``bijou_config.json`` instead)."""
    path = checkpoint / METADATA_FILENAME
    if not path.exists():
        legacy = checkpoint / "bijou_config.json"
        if legacy.exists():
            raise SystemExit(
                f"{checkpoint} is a LEGACY checkpoint (bijou_config.json) — "
                "convert it first: python -m bijou.convert_legacy "
                f"{checkpoint} <new-dir>",
            )
        raise SystemExit(f"{checkpoint} has no {METADATA_FILENAME}")
    return VLAMetadata.from_json_dict(json.loads(path.read_text()))


def backbone_files(checkpoint: Path) -> BackboneFiles:
    """The checkpoint's per-part trunk weight files (the from-files
    loaders' input — plain strict loads of OUR key names)."""
    return BackboneFiles(
        text=checkpoint / BACKBONE_TEXT_FILENAME,
        vision=checkpoint / BACKBONE_VISION_FILENAME,
    )


def tokenizer_directory(checkpoint: Path) -> Path:
    """The checkpoint's tokenizer/ directory — the collators' and
    decoders' processor/tokenizer source (self-containment: loading
    never touches the hub or the HF cache)."""
    return checkpoint / TOKENIZER_DIRNAME


def link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link ``source`` to ``destination``; fall back to a full copy
    (loudly) when the link is impossible (cross-device, or a filesystem
    without hard links). ``source`` is resolved first so links point at
    the real blob, not at an HF-cache snapshot symlink."""
    real = source.resolve()
    try:
        os.link(real, destination)
    except OSError as error:
        print(
            f"[checkpoint] hard link {real} -> {destination} failed "
            f"({error.strerror}); copying in full",
            flush=True,
        )
        shutil.copy2(real, destination)


def validate_checkpoint(checkpoint: Path) -> VLAMetadata:
    """The self-containment check: metadata parses, both backbone part
    files are present, every declared component has its weight file,
    the tokenizer/ directory carries the family's manifest, and no
    undeclared weight file is present (a stray file is a wiring bug,
    never ignorable)."""
    metadata = read_metadata(checkpoint)
    for filename in (BACKBONE_TEXT_FILENAME, BACKBONE_VISION_FILENAME):
        if not (checkpoint / filename).is_file():
            raise SystemExit(f"{checkpoint}: no {filename} — not self-contained")
    missing = [
        name
        for name, record in metadata.components.items()
        if record["weights"] and not (checkpoint / f"{name}.safetensors").exists()
    ]
    if missing:
        raise SystemExit(
            f"{checkpoint}: declared components missing weight files: {missing}",
        )
    tokenizer_dir = checkpoint / TOKENIZER_DIRNAME
    absent = [
        name
        for name in tokenizer_manifest(metadata.family)
        if not (tokenizer_dir / name).is_file()
    ]
    if absent:
        raise SystemExit(
            f"{checkpoint}: tokenizer/ is missing {absent} — the "
            f"{metadata.family.value} prompt path reads these files; the "
            "directory is not self-contained",
        )
    declared = {
        f"{name}.safetensors"
        for name, record in metadata.components.items()
        if record["weights"]
    }
    declared.add(BACKBONE_TEXT_FILENAME)
    declared.add(BACKBONE_VISION_FILENAME)
    stray = [p.name for p in checkpoint.glob("*.safetensors") if p.name not in declared]
    if stray:
        raise SystemExit(
            f"{checkpoint}: undeclared weight files {stray} — every "
            "*.safetensors must be a declared component (or a backbone "
            "part file)",
        )
    return metadata


def write_checkpoint(
    directory: Path,
    *,
    metadata: VLAMetadata,
    components: dict[str, dict[str, Tensor]],
    backbone_text: dict[str, Tensor] | Path,
    backbone_vision: dict[str, Tensor] | Path,
    tokenizer_files: dict[str, Path],
    component_files: dict[str, Path] | None = None,
    optimizer: Path | None = None,
) -> None:
    """Materialize a checkpoint atomically: stage into ``<dir>.tmp``,
    validate self-containment, rename into place (refusing an existing
    target — checkpoints are immutable once published).

    ``components`` are state dicts to serialize; ``component_files``
    optionally maps a component to an EXISTING safetensors file to
    hard-link instead (the conversion path — content-identical files
    never rewrite). Together they must cover exactly the metadata's
    ``weights: true`` components. ``backbone_text``/``backbone_vision``
    are each a state dict to serialize (a part trained this run, or a
    fresh run's first serialization of a frozen part) or an existing
    per-part FILE to hard-link (the frozen-part dedup: conversion links
    the import, training saves link the previous save).
    ``tokenizer_files`` maps tokenizer/ names to source files to link —
    at least the family's :func:`tokenizer_manifest`."""
    if directory.exists():
        raise SystemExit(f"refusing to overwrite existing checkpoint {directory}")
    weighted = {
        name for name, record in metadata.components.items() if record["weights"]
    }
    if set(components) | set(component_files or {}) != weighted:
        raise ValueError(
            f"component mismatch: weights for "
            f"{sorted(set(components) | set(component_files or {}))} vs "
            f"metadata declaring {sorted(weighted)} (weights: true)",
        )
    staging = directory.parent / (directory.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / METADATA_FILENAME).write_text(
        json.dumps(metadata.to_json_dict(), indent=2) + "\n",
    )
    for name, state in components.items():
        save_file(state, str(staging / f"{name}.safetensors"))
    for name, source in (component_files or {}).items():
        link_or_copy(source, staging / f"{name}.safetensors")
    for filename, part in (
        (BACKBONE_TEXT_FILENAME, backbone_text),
        (BACKBONE_VISION_FILENAME, backbone_vision),
    ):
        if isinstance(part, dict):
            save_file(part, str(staging / filename))
        else:
            link_or_copy(part, staging / filename)
    tokenizer_dir = staging / TOKENIZER_DIRNAME
    tokenizer_dir.mkdir()
    for name, source in tokenizer_files.items():
        link_or_copy(source, tokenizer_dir / name)
    if optimizer is not None:
        link_or_copy(optimizer, staging / "optimizer.pt")
    validate_checkpoint(staging)
    staging.rename(directory)
