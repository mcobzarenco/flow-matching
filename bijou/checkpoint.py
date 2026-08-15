"""The VLA checkpoint format — schema and IO toolkit.

A VLA checkpoint is a SELF-CONTAINED directory:

```
checkpoint/
  metadata.json             # VLAMetadata (schema_version, family, …)
  backbone.safetensors      # trained trunk state (backbone.trained)
  backbone/                 # OR: hard-linked mirror of the pristine
                            #   artifact snapshot (config, tokenizer,
                            #   weight shards — loadable as a local dir)
  <component>.safetensors   # one per family-declared component
  optimizer.pt              # optional (run-seeding checkpoints)
```

Self-containment without disk cost: a pristine trunk is materialized
by hard-linking every file of the resolved artifact snapshot into
``backbone/`` (same filesystem: ~zero bytes; cross-device falls back
to a loud copy). Loading a checkpoint never touches the hub or the HF
cache — ``backbone/`` IS a local model directory. Presence is not a
signal: ``backbone.trained`` in the metadata is the explicit fact.

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
from .vla import VLAFamily

SCHEMA_VERSION = 1
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True, slots=True)
class VLAMetadata:
    """``metadata.json``, both directions (validating parse on the way
    in, explicit serialization on the way out).

    ``train_args`` is the run's full CLI record, verbatim provenance;
    ``stats`` is the count-weighted aggregate normalization table and
    ``per_dataset_stats`` the per-repo tables (keyed by repo id —
    genuinely dynamic); ``stats_note`` records a stats REPLACEMENT
    (the ``--replace-stats`` conversion op) so a substituted table is
    never mistaken for a trained-with one."""

    family: VLAFamily
    chunk_size: int
    action_dim: int
    backbone_id: str
    backbone_depth: str
    backbone_trained: bool
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
                "trained": self.backbone_trained,
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
            backbone_trained=bool(backbone["trained"]),
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


def mirror_backbone_snapshot(snapshot: Path, destination: Path) -> None:
    """Hard-link every file of a pristine artifact snapshot directory
    into ``destination`` (recursively) — the checkpoint's ``backbone/``
    is then loadable as a local model directory with ~zero disk cost."""
    if not snapshot.is_dir():
        raise ValueError(
            f"pristine backbone source {snapshot} is not a directory — "
            "expected the resolved artifact snapshot",
        )
    destination.mkdir(parents=True)
    for path in sorted(snapshot.rglob("*")):
        relative = path.relative_to(snapshot)
        if path.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            link_or_copy(path, destination / relative)


def validate_checkpoint(checkpoint: Path) -> VLAMetadata:
    """The self-containment check: metadata parses, every declared
    component has its weight file, the backbone is materialized in the
    form the metadata claims, and no undeclared weight file is present
    (a stray file is a wiring bug, never ignorable)."""
    metadata = read_metadata(checkpoint)
    missing = [
        name
        for name, record in metadata.components.items()
        if record["weights"] and not (checkpoint / f"{name}.safetensors").exists()
    ]
    if missing:
        raise SystemExit(
            f"{checkpoint}: declared components missing weight files: {missing}",
        )
    if metadata.backbone_trained:
        if not (checkpoint / "backbone.safetensors").is_file():
            raise SystemExit(
                f"{checkpoint}: backbone.trained but no backbone.safetensors",
            )
    elif not (checkpoint / "backbone").is_dir():
        raise SystemExit(
            f"{checkpoint}: pristine backbone but no backbone/ snapshot "
            "mirror — the directory is not self-contained",
        )
    declared = {
        f"{name}.safetensors"
        for name, record in metadata.components.items()
        if record["weights"]
    }
    if metadata.backbone_trained:
        declared.add("backbone.safetensors")
    stray = [p.name for p in checkpoint.glob("*.safetensors") if p.name not in declared]
    if stray:
        raise SystemExit(
            f"{checkpoint}: undeclared weight files {stray} — every "
            "*.safetensors must be a declared component (or the trained "
            "backbone)",
        )
    return metadata


def write_checkpoint(
    directory: Path,
    *,
    metadata: VLAMetadata,
    components: dict[str, dict[str, Tensor]],
    backbone: Path | dict[str, Tensor],
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
    ``weights: true`` components. ``backbone`` is a trained state dict
    or an existing trained-state FILE to link (``backbone_trained``
    must be True), or the pristine snapshot DIRECTORY to mirror (must
    be False) — the metadata flag and the argument form are
    cross-checked, loudly."""
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
    backbone_is_trained_form = isinstance(backbone, dict) or backbone.is_file()
    if backbone_is_trained_form != metadata.backbone_trained:
        raise ValueError(
            "backbone argument form contradicts metadata.backbone_trained "
            f"({metadata.backbone_trained}): a trained backbone is a state "
            "dict or safetensors file, a pristine one is the snapshot "
            "directory",
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
    if isinstance(backbone, dict):
        save_file(backbone, str(staging / "backbone.safetensors"))
    elif backbone.is_file():
        link_or_copy(backbone, staging / "backbone.safetensors")
    else:
        mirror_backbone_snapshot(backbone, staging / "backbone")
    if optimizer is not None:
        link_or_copy(optimizer, staging / "optimizer.pt")
    validate_checkpoint(staging)
    staging.rename(directory)
