"""Upgrade a schema-1 VLA checkpoint directory to schema 2.

Usage::

    uv run python -m bijou.convert_v1 <v1-dir> <new-dir>

Schema 1 (the snapshot-mirror era) recorded ONE trunk: a trained
``backbone.safetensors`` (the mounted model's state dict — ``text.``/
``vision.``-prefixed on Molmo trunks, ``vision_tower.*`` + rest on
Gemma) or a pristine ``backbone/`` hard-linked mirror of the HF
artifact snapshot, with a single ``backbone.trained`` fact and no
carried config or tokenizer files. Schema 2 is the full import:
per-part ``backbone_text``/``backbone_vision`` files in OUR key names,
the artifact's ``config.json`` verbatim in the metadata, per-part
trained flags, and ``tokenizer/``.

This module is the ONE home of the schema-1 LAYOUT going forward: the
frozen v1 metadata reader plus the upgrade. It exists because
train-written v1 checkpoints (a v1-era ``bijou.train`` save) have NO
original source to re-convert from — ``bijou.convert_legacy`` covers
``bijou_config.json`` directories and ``bijou.convert_molmoact2``
covers HF releases, but a v1 training save could previously only load
from a git worktree pinned before the schema-2 flip (the
``step_002000`` joint probes ran from one for three days).

The trunk translation reuses the same audited machinery as the other
importers: a trained ``backbone.safetensors`` partitions through
``split_gemma_backbone_state``/``split_molmo2_backbone_state`` (both
per-part trained flags carry the single v1 fact conservatively True); a
pristine trunk imports from the checkpoint's own ``backbone/`` mirror
(self-contained — no HF cache needed). ``config.json`` and the
tokenizer manifest come from the mirror when present, otherwise from
the locally-cached artifact (``resolve_artifact_snapshot`` — never a
download). Component weight files and ``optimizer.pt`` are hard-linked,
never rewritten; everything else in the metadata (components,
objective, serving, stats, train_args) carries VERBATIM — with ONE
recorded translation: a pre-rename AR objective's ``aux_loss_weight``
becomes ``narration_weight`` (the 32149df rename; the current parser
would otherwise silently default the trained mix to 1.0). Upgrading is
deterministic: re-run against the same source it produces
content-identical output.

Comparability caveat (SEPARATE from schema): upgrading a checkpoint
does not reproduce the eval substrate its banked reads used. Sim reads
pinned to the stand-ins substrate (e.g. the joint-probe legs'
``clutter_appearance='standins'``) stay comparable only if the eval
passes that flag explicitly — a flag choice at eval time, not anything
this importer can carry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from .checkpoint import (
    METADATA_FILENAME,
    VLAMetadata,
    tokenizer_manifest,
    validate_checkpoint,
    write_checkpoint,
)
from .convert_legacy import resolve_artifact_snapshot
from .data import DatasetStats
from .modelling.gemma4.loading import (
    import_backbone_state as import_gemma_backbone_state,
)
from .modelling.molmo2.loading import (
    import_backbone_state as import_molmo2_backbone_state,
)
from .sections import (
    split_gemma_backbone_state,
    split_molmo2_backbone_state,
)
from .vla import VLAFamily

V1_SCHEMA_VERSION = 1

GEMMA_FAMILIES = frozenset({VLAFamily.GEMMA_FLOW, VLAFamily.GEMMA_AR})


def read_v1_metadata(source: Path) -> dict[str, Any]:
    """Parse a v1 directory's ``metadata.json``, refusing everything
    that is not schema 1 (loud, with the right converter named)."""
    path = source / METADATA_FILENAME
    if not path.exists():
        if (source / "bijou_config.json").exists():
            raise SystemExit(
                f"{source} is a LEGACY checkpoint (bijou_config.json), "
                "not a v1 VLA directory — convert it with "
                "bijou.convert_legacy",
            )
        raise SystemExit(f"{source} has no {METADATA_FILENAME}")
    data = json.loads(path.read_text())
    version = data.get("schema_version")
    if version != V1_SCHEMA_VERSION:
        raise SystemExit(
            f"{source}: schema_version {version!r} — this importer "
            f"upgrades schema {V1_SCHEMA_VERSION} only (schema 2 is "
            "already current; anything else is not a VLA checkpoint)",
        )
    backbone = data["backbone"]
    if set(backbone) != {"id", "depth", "trained"}:
        raise SystemExit(
            f"{source}: v1 backbone section must be {{'id', 'depth', "
            f"'trained'}}, got keys {sorted(backbone)}",
        )
    return data


def convert(source: Path, destination: Path) -> VLAMetadata:
    """Upgrade ``source`` (schema 1) into ``destination`` (schema 2);
    returns the written metadata. Loud about every inference."""
    data = read_v1_metadata(source)
    family = VLAFamily(data["family"])
    backbone = data["backbone"]
    trained = bool(backbone["trained"])
    gemma_trunk = family in GEMMA_FAMILIES

    # --- the trunk: partition the trained file, or import the mirror ---
    mirror = source / "backbone"
    backbone_file = source / "backbone.safetensors"
    if trained:
        if not backbone_file.is_file():
            raise SystemExit(
                f"{source}: backbone.trained but no backbone.safetensors",
            )
        v1_state = load_file(str(backbone_file), device="cpu")
        if gemma_trunk:
            backbone_text, backbone_vision = split_gemma_backbone_state(v1_state)
        else:
            backbone_text, backbone_vision = split_molmo2_backbone_state(v1_state)
        print(
            f"[convert_v1] trained trunk: {backbone_file.name} "
            f"partitioned into text ({len(backbone_text)}) + vision "
            f"({len(backbone_vision)}) tensors (both flags True)",
        )
    else:
        if not mirror.is_dir():
            raise SystemExit(
                f"{source}: pristine backbone but no backbone/ snapshot "
                "mirror — the v1 directory is not self-contained",
            )
        if gemma_trunk:
            imported = import_gemma_backbone_state(mirror)
            backbone_text, backbone_vision = imported.text, imported.vision
            print(
                f"[convert_v1] pristine trunk imported from its own "
                f"mirror: text ({len(backbone_text)}) + vision "
                f"({len(backbone_vision)}) tensors, skipped "
                f"{len(imported.skipped)}",
            )
        else:
            imported_molmo = import_molmo2_backbone_state(mirror)
            backbone_text, backbone_vision = imported_molmo.text, imported_molmo.vision
            print(
                f"[convert_v1] pristine trunk imported from its own "
                f"mirror: text ({len(backbone_text)}) + vision "
                f"({len(backbone_vision)}) tensors, expert "
                f"({len(imported_molmo.expert)}) left to the decoder "
                f"file, skipped {len(imported_molmo.skipped)}",
            )

    # --- config.json + tokenizer files: the mirror when present (the
    # pristine arm's self-containment), the cached artifact otherwise ---
    if mirror.is_dir():
        snapshot = mirror
    else:
        snapshot = resolve_artifact_snapshot(str(backbone["id"]))
        print(f"[convert_v1] artifact snapshot for config/tokenizer: {snapshot}")
    backbone_config = json.loads((snapshot / "config.json").read_text())
    tokenizer_files: dict[str, Path] = {}
    for name in tokenizer_manifest(family):
        path = snapshot / name
        if not path.is_file():
            raise SystemExit(
                f"{snapshot} has no {name} — the {family.value} prompt "
                "path reads it; cannot produce a self-contained checkpoint",
            )
        tokenizer_files[name] = path

    # --- everything else carries verbatim (one recorded rename: the
    # AR objective's aux_loss_weight became narration_weight in 32149df;
    # pre-rename recordings translate so the recorded value survives —
    # current parse_ar_objective would silently default 1.0 otherwise) ---
    objective = dict(data["objective"])
    if objective.get("kind") == "ar" and "aux_loss_weight" in objective:
        weight = objective.pop("aux_loss_weight")
        objective.setdefault("narration_weight", float(weight))
        print(
            f"[convert_v1] objective aux_loss_weight={weight} -> "
            "narration_weight (the pre-rename recording)",
        )
    spec = data["spec"]
    components = {k: dict(v) for k, v in data["components"].items()}
    component_files = {
        name: source / f"{name}.safetensors"
        for name, record in components.items()
        if record["weights"]
    }
    metadata = VLAMetadata(
        family=family,
        chunk_size=int(spec["chunk_size"]),
        action_dim=int(spec["action_dim"]),
        backbone_id=str(backbone["id"]),
        backbone_depth=str(backbone["depth"]),
        backbone_config=backbone_config,
        # The v1 format recorded ONE trained fact for the whole trunk —
        # both per-part flags carry it conservatively.
        backbone_text_trained=trained,
        backbone_vision_trained=trained,
        objective=objective,
        serving=dict(data["serving"]),
        components=components,
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
    optimizer = source / "optimizer.pt"
    write_checkpoint(
        destination,
        metadata=metadata,
        components={},
        component_files=component_files,
        backbone_text=backbone_text,
        backbone_vision=backbone_vision,
        tokenizer_files=tokenizer_files,
        optimizer=optimizer if optimizer.exists() else None,
    )
    validate_checkpoint(destination)
    print(
        f"[convert_v1] {source} -> {destination}: family={family.value} "
        f"chunk={metadata.chunk_size} action_dim={metadata.action_dim} "
        f"backbone text_trained={trained} vision_trained={trained} "
        f"step={metadata.step}",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="schema-1 VLA checkpoint directory")
    parser.add_argument(
        "destination",
        type=Path,
        help="schema-2 VLA checkpoint directory to create",
    )
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    sys.exit(main())
