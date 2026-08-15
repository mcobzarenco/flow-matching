"""Convert a legacy bijou checkpoint (``bijou_config.json`` format 3)
to the VLA checkpoint format (``bijou/checkpoint.py``).

Usage::

    uv run python -m bijou.convert_legacy <legacy-dir> <new-dir> \\
        [--replace-stats <state-dict.json>]

The conversion is a metadata re-expression plus hard links — weight
files are never rewritten (content-identical by construction), so a
conversion is cheap and, re-run against the same source, produces
content-identical output (the idempotence gate).

Family inference is the (prompt kind, decoder kind, objective) triple:

- gemma prompt + flow decoder → ``gemma_flow``
- gemma prompt + ar_backbone → ``gemma_ar``
- molmo2 prompt + ar_backbone → ``molmo2_ar``
- molmoact2 prompt + molmo_flow + objective flow/ar/joint →
  ``molmoact2_flow`` / ``molmoact2_ar`` / ``molmoact2_joint``

Component configs are carried VERBATIM as the legacy tagged section
dicts — families parse them with the same section machinery either
way, so conversion cannot drift architecture. ``--replace-stats``
substitutes the AGGREGATE normalization table (a first-class operation
for checkpoints whose baked q01/q99 must be corrected before reuse)
and records a ``stats_note`` so a substituted table is never mistaken
for a trained-with one.

Formats 1 and 2 are refused: every checkpoint on the conversion
inventory is format 3; a genuinely needed older directory loads
through the legacy reader at an older git tag first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .checkpoint import VLAMetadata, validate_checkpoint, write_checkpoint
from .data import DatasetStats
from .loading import (
    ARDecoderConfig,
    CheckpointTrainArgs,
    FlowDecoderSection,
    GemmaPromptConfig,
    Molmo2PromptConfig,
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
    checkpoint_sections,
)
from .vla import VLAFamily


def infer_family(
    prompt: GemmaPromptConfig | Molmo2PromptConfig | MolmoAct2PromptConfig,
    decoder: FlowDecoderSection | ARDecoderConfig | MolmoFlowDecoderConfig,
    objective: str,
) -> VLAFamily:
    match prompt, decoder:
        case GemmaPromptConfig(), FlowDecoderSection():
            return VLAFamily.GEMMA_FLOW
        case GemmaPromptConfig(), ARDecoderConfig():
            return VLAFamily.GEMMA_AR
        case Molmo2PromptConfig(), ARDecoderConfig():
            return VLAFamily.MOLMO2_AR
        case MolmoAct2PromptConfig(), MolmoFlowDecoderConfig():
            match objective:
                case "flow":
                    return VLAFamily.MOLMOACT2_FLOW
                case "ar":
                    return VLAFamily.MOLMOACT2_AR
                case "joint":
                    return VLAFamily.MOLMOACT2_JOINT
                case _:
                    raise SystemExit(f"unknown recorded objective {objective!r}")
        case _:
            raise SystemExit(
                f"no family for prompt {type(prompt).__name__} + decoder "
                f"{type(decoder).__name__} — not a convertible combination",
            )


def convert(
    source: Path,
    destination: Path,
    *,
    replace_stats: Path | None = None,
) -> VLAMetadata:
    """Convert ``source`` (legacy) into ``destination`` (VLA format);
    returns the written metadata. Loud about every inference."""
    config_path = source / "bijou_config.json"
    if not config_path.exists():
        raise SystemExit(f"{source}: no bijou_config.json — not a legacy checkpoint")
    meta = json.loads(config_path.read_text())
    fmt = int(meta.get("format", 1))
    if fmt < 3:
        raise SystemExit(
            f"{source}: legacy format {fmt} — only format 3 converts (see "
            "the module docstring for the escape hatch)",
        )
    sections = checkpoint_sections(meta)
    assert sections.prompt is not None and sections.decoder is not None  # format 3
    train_args = CheckpointTrainArgs.from_dict(meta["train_args"])
    family = infer_family(sections.prompt, sections.decoder, train_args.objective)
    stats = DatasetStats.from_state_dict(meta["normalization"])
    stats_note = None
    if replace_stats is not None:
        stats = DatasetStats.from_state_dict(json.loads(replace_stats.read_text()))
        stats_note = f"aggregate stats REPLACED at conversion from {replace_stats}"
        print(f"[convert] {stats_note}")
    action_dim = len(stats.action_mean)

    # --- components: legacy section dicts verbatim; weight files linked ---
    prompt_dict = sections.prompt.to_dict()
    decoder_dict = dict(meta["decoder"])
    components: dict[str, dict[str, Any]] = {}
    component_files: dict[str, Path] = {}
    has_prompt_weights = (source / "prompt.safetensors").exists()
    has_expert_weights = (source / "expert.safetensors").exists()
    match family:
        case VLAFamily.GEMMA_FLOW | VLAFamily.MOLMOACT2_FLOW:
            components["prompt"] = {
                "config": prompt_dict,
                "weights": has_prompt_weights,
            }
            components["flow_decoder"] = {"config": decoder_dict, "weights": True}
            component_files["flow_decoder"] = source / "expert.safetensors"
        case VLAFamily.GEMMA_AR | VLAFamily.MOLMO2_AR:
            components["prompt"] = {
                "config": prompt_dict,
                "weights": has_prompt_weights,
            }
            components["ar_decoder"] = {"config": decoder_dict, "weights": True}
            component_files["ar_decoder"] = source / "expert.safetensors"
        case VLAFamily.MOLMOACT2_AR:
            # The discrete decoder owns zero parameters; the recorded
            # flow SECTION is still the geometry record the ar config
            # derives from (molmoact2_ar_config_from_flow_section).
            components["prompt"] = {
                "config": prompt_dict,
                "weights": has_prompt_weights,
            }
            components["ar_decoder"] = {"config": decoder_dict, "weights": False}
            if has_expert_weights:
                # objective=ar runs may still carry inherited expert
                # weights (stage-2 provenance); keep them as a flow
                # component so nothing is dropped silently.
                components["flow_decoder"] = {"config": decoder_dict, "weights": True}
                component_files["flow_decoder"] = source / "expert.safetensors"
                print(
                    "[convert] objective=ar checkpoint carries expert "
                    "weights — kept as flow_decoder (inherited section)",
                )
        case VLAFamily.MOLMOACT2_JOINT:
            components["prompt"] = {
                "config": prompt_dict,
                "weights": has_prompt_weights,
            }
            components["flow_decoder"] = {"config": decoder_dict, "weights": True}
            component_files["flow_decoder"] = source / "expert.safetensors"
            joint_ce = meta.get("joint_ce")
            if joint_ce is None:
                raise SystemExit(
                    f"{source}: objective=joint but no joint_ce section",
                )
            components["ar_decoder"] = {"config": dict(joint_ce), "weights": False}
    if has_prompt_weights:
        component_files["prompt"] = source / "prompt.safetensors"

    # --- objective / serving ---
    raw_args: dict[str, Any] = dict(meta["train_args"])
    match family:
        case VLAFamily.GEMMA_FLOW | VLAFamily.MOLMOACT2_FLOW:
            objective: dict[str, Any] = {"kind": "flow"}
        case VLAFamily.GEMMA_AR | VLAFamily.MOLMO2_AR:
            objective = {
                "kind": "ar",
                "aux_loss_weight": float(raw_args.get("aux_loss_weight", 1.0)),
            }
        case VLAFamily.MOLMOACT2_AR:
            objective = {"kind": "ar"}
        case VLAFamily.MOLMOACT2_JOINT:
            objective = {
                "kind": "joint",
                "ce_weight": float(raw_args.get("joint_ce_weight", 1.0)),
                "insulate_flow": bool(raw_args.get("insulate_expert", False)),
            }
    match family:
        case VLAFamily.GEMMA_FLOW:
            # The historical Gemma serving operating point.
            serving: dict[str, Any] = {"kind": "flow", "num_steps": 5, "method": "heun"}
        case VLAFamily.MOLMOACT2_FLOW | VLAFamily.MOLMOACT2_JOINT:
            assert isinstance(sections.decoder, MolmoFlowDecoderConfig)
            serving = {
                "kind": "flow",
                "num_steps": sections.decoder.num_flow_steps,
                "method": "euler",
            }
        case VLAFamily.GEMMA_AR | VLAFamily.MOLMO2_AR | VLAFamily.MOLMOACT2_AR:
            serving = {"kind": "ar"}

    artifacts = {}
    if train_args.fast_tokenizer is not None:
        artifacts["fast_tokenizer"] = train_args.fast_tokenizer

    backbone_file = source / "backbone.safetensors"
    backbone_trained = backbone_file.exists()
    backbone: Path
    if backbone_trained:
        backbone = backbone_file
    else:
        snapshot = resolve_pristine_snapshot(sections.backbone.id)
        backbone = snapshot
    metadata = VLAMetadata(
        family=family,
        chunk_size=train_args.chunk_size,
        action_dim=action_dim,
        backbone_id=sections.backbone.id,
        backbone_depth=sections.backbone.depth.value,
        backbone_trained=backbone_trained,
        objective=objective,
        serving=serving,
        components=components,
        artifacts=artifacts,
        stats=stats,
        per_dataset_stats={
            repo_id: DatasetStats.from_state_dict(table)
            for repo_id, table in meta.get("per_dataset_normalization", {}).items()
        },
        train_args=raw_args,
        step=int(meta["step"]),
        stats_note=stats_note,
    )
    optimizer = source / "optimizer.pt"
    write_checkpoint(
        destination,
        metadata=metadata,
        components={},
        component_files=component_files,
        backbone=backbone,
        optimizer=optimizer if optimizer.exists() else None,
    )
    validate_checkpoint(destination)
    print(
        f"[convert] {source} -> {destination}: family={family.value} "
        f"chunk={metadata.chunk_size} action_dim={action_dim} "
        f"backbone_trained={backbone_trained} step={metadata.step}",
    )
    return metadata


def resolve_pristine_snapshot(backbone_id: str) -> Path:
    """The pristine trunk's snapshot directory: a recorded local path is
    used directly; a hub id resolves through the LOCAL cache only —
    conversion never downloads (the machine converting a checkpoint
    already trained/evaluated with the artifact)."""
    local = Path(backbone_id).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import (
        snapshot_download,
    )

    try:
        return Path(
            snapshot_download(backbone_id, local_files_only=True),
        )
    except Exception as error:
        raise SystemExit(
            f"pristine backbone {backbone_id!r} is not in the local HF "
            f"cache ({error}) — fetch it once (huggingface-cli download "
            f"{backbone_id}) and re-run",
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="legacy checkpoint directory")
    parser.add_argument(
        "destination",
        type=Path,
        help="VLA checkpoint directory to create",
    )
    parser.add_argument(
        "--replace-stats",
        type=Path,
        default=None,
        help="JSON file with a DatasetStats state dict replacing the "
        "AGGREGATE normalization table (recorded in stats_note)",
    )
    args = parser.parse_args()
    convert(args.source, args.destination, replace_stats=args.replace_stats)


if __name__ == "__main__":
    sys.exit(main())
