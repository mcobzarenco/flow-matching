"""Checkpoint capture and write for training runs.

Serialization splits at the device boundary: :func:`build_vla_metadata`
and :func:`capture_checkpoint_tensors` run on the main thread at the
save boundary (cheap and pure -- the copies are the boundary's values),
while :func:`write_checkpoint` is CPU+disk only and therefore safe on
the async saver's background thread (``bijou.async_save``);
:func:`save_checkpoint` is the synchronous capture+write entry.
Checkpoints are the VLA format (``bijou/checkpoint.py``):
``metadata.json`` + per-component safetensors + the always-present
backbone (trained bf16 snapshot, the inherited adapted file, or a
hard-linked pristine mirror).

Also here: the payload derivations the metadata records --
:func:`build_objective` (the family's objective payload from the
resolved run flags) with the ``objective_to_json``/``serving_to_json``
write-side encoders -- and the MEAN_STD :class:`Normalizer`, whose
count-weighted aggregate rides checkpoints as the stats fallback for
rigs without their own stats.
"""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ..checkpoint import VLAMetadata
from ..checkpoint import write_checkpoint as write_vla_checkpoint
from ..data import DatasetStats
from ..modelling.encoders.gemma4 import PROMPT_FORMAT
from ..modelling.encoders.molmo2 import MOLMO2_PROMPT_FORMAT
from ..modelling.encoders.molmoact2 import MolmoAct2Encoder
from ..modelling.gemma4.config import Gemma4Config
from ..models.gemma_ar import GemmaARVLA
from ..models.gemma_flow import GemmaFlowVLA
from ..models.molmo2_ar import Molmo2ARVLA
from ..models.molmoact2_ar import MolmoAct2ARVLA
from ..models.molmoact2_flow import MolmoAct2FlowVLA
from ..models.molmoact2_joint import JointObjective, MolmoAct2JointVLA
from ..models.objectives import ARObjective, FlowObjective, SnapflowObjective
from ..models.serving import ARServing, FlowServing
from ..sections import (
    BACKBONE_UNSAVED_KEYS,
    BackboneDepth,
    GemmaPromptConfig,
    Molmo2PromptConfig,
    decoder_schema_dict,
)
from ..vla import VLA
from .args import AR_SUFFIX_FAMILIES, TrainArgs, train_args_record

#: The closed union of trainable family classes — what main() builds
#: and the load/save helpers consume (attribute access needs the
#: concrete union; generic consumers take VLA[Any]).
type TrainableVLA = (
    GemmaFlowVLA
    | GemmaARVLA
    | Molmo2ARVLA
    | MolmoAct2FlowVLA
    | MolmoAct2ARVLA
    | MolmoAct2JointVLA
)


def build_objective(
    args: TrainArgs,
) -> FlowObjective | SnapflowObjective | ARObjective | JointObjective:
    """The family's objective payload from the resolved run flags — the
    constructor value that selects what receives gradients (D3/D4).
    Recorded verbatim in the checkpoint metadata."""
    match args.family:
        case "gemma_flow":
            if args.distill == "snapflow":
                assert args.snapflow_alpha is not None  # __post_init__ paired
                assert args.snapflow_shortcut_weight is not None
                return SnapflowObjective(
                    alpha=args.snapflow_alpha,
                    shortcut_weight=args.snapflow_shortcut_weight,
                )
            return FlowObjective()
        case "gemma_ar" | "molmo2_ar":
            return ARObjective(aux_loss_weight=args.aux_loss_weight)
        case "molmoact2_flow":
            return FlowObjective()
        case "molmoact2_ar":
            # The format-6 emission has no aux fields; the weight is the
            # payload's inert unit value (parse_ar_objective's default).
            return ARObjective(aux_loss_weight=1.0)
        case "molmoact2_joint":
            return JointObjective(
                ce_weight=args.joint_ce_weight,
                insulate_flow=args.insulate_flow,
            )
        case _:
            raise AssertionError(f"unhandled family {args.family!r}")


def objective_to_json(
    objective: FlowObjective | SnapflowObjective | ARObjective | JointObjective,
) -> dict[str, Any]:
    """The objective payload as the metadata's tagged dict (the write
    side of the families' ``parse_*_objective`` readers)."""
    match objective:
        case FlowObjective():
            return {"kind": "flow"}
        case SnapflowObjective():
            return {
                "kind": "snapflow",
                "alpha": objective.alpha,
                "shortcut_weight": objective.shortcut_weight,
            }
        case ARObjective():
            return {"kind": "ar", "aux_loss_weight": objective.aux_loss_weight}
        case JointObjective():
            return {
                "kind": "joint",
                "ce_weight": objective.ce_weight,
                "insulate_flow": objective.insulate_flow,
            }


def serving_to_json(serving: FlowServing | ARServing) -> dict[str, Any]:
    """The serving operating point as the metadata's tagged dict (the
    write side of ``FlowServing.from_dict``/``ARServing.from_dict``)."""
    match serving:
        case FlowServing():
            return {
                "kind": "flow",
                "num_steps": serving.num_steps,
                "method": serving.method.value,
            }
        case ARServing():
            return {"kind": "ar"}


class Normalizer:
    """MEAN_STD normalization from LeRobot dataset stats.

    ``mean``/``std``: [dim] (action_dim or state_dim); normalize/unnormalize
    broadcast over any leading axes ([*, dim] -> [*, dim])."""

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean
        self.std = std

    @classmethod
    def from_stats(
        cls,
        stats: dict[str, dict[str, Any]],
        key: str,
        device: torch.device,
    ) -> Normalizer:
        mean = torch.as_tensor(stats[key]["mean"], dtype=torch.float32, device=device)
        std = torch.as_tensor(stats[key]["std"], dtype=torch.float32, device=device)
        return cls(mean, std)

    @classmethod
    def from_aggregated_stats(
        cls,
        stats_list: list[dict[str, dict[str, Any]]],
        key: str,
        device: torch.device,
    ) -> Normalizer:
        """Count-weighted aggregation across datasets: the exact combined
        mean, and std via E[x²] composition (all in float64 before rounding
        to float32)."""
        counts = torch.tensor(
            [float(s[key]["count"][0]) for s in stats_list],
            dtype=torch.float64,
        )
        means = torch.stack(
            [torch.as_tensor(s[key]["mean"], dtype=torch.float64) for s in stats_list],
        )
        stds = torch.stack(
            [torch.as_tensor(s[key]["std"], dtype=torch.float64) for s in stats_list],
        )
        total = counts.sum()
        weights = (counts / total)[:, None]
        mean = (weights * means).sum(dim=0)
        second_moment = (weights * (stds.pow(2) + means.pow(2))).sum(dim=0)
        std = (second_moment - mean.pow(2)).clamp(min=0).sqrt()
        return cls(
            mean.to(dtype=torch.float32, device=device),
            std.to(dtype=torch.float32, device=device),
        )

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / (self.std + 1e-8)

    def unnormalize(self, x: Tensor) -> Tensor:
        return x * (self.std + 1e-8) + self.mean

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


@dataclass(frozen=True, slots=True)
class Normalizers:
    action: Normalizer
    state: Normalizer


@dataclass(frozen=True, slots=True)
class TrainState:
    """optimizer.pt payload — everything --resume needs beyond the weights.
    Stored as a plain dict on disk (torch.load(weights_only=True) rejects
    custom classes), so the payload methods are the (de)serialization edge.
    """

    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    step: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "step": self.step,
        }

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> TrainState:
        return cls(
            optimizer=data["optimizer"],
            scheduler=data["scheduler"],
            step=int(data["step"]),
        )


def aggregate_stats(normalizers: Normalizers) -> DatasetStats:
    """The count-weighted aggregate stats as a DatasetStats (the checkpoint
    fallback entry for rigs without their own stats)."""
    return DatasetStats(
        action_mean=tuple(normalizers.action.mean.tolist()),
        action_std=tuple(normalizers.action.std.tolist()),
        state_mean=tuple(normalizers.state.mean.tolist()),
        state_std=tuple(normalizers.state.std.tolist()),
        # Quantiles do not compose across datasets (a mean of per-dataset
        # quantiles regresses extremes — the exact bug the corpus backfill
        # fixed), so the aggregate fallback honestly carries none.
        action_q01=None,
        action_q99=None,
        state_q01=None,
        state_q99=None,
    )


@dataclass(frozen=True, slots=True)
class CheckpointTensors:
    """The model-side CPU snapshot of one checkpoint — everything the
    writer needs that came off the device. Captured on the main thread at
    the save boundary (the copies are the boundary's values); consumed by
    ``write_checkpoint`` on either the main thread (sync path) or the
    async saver's background thread.

    ``components`` are the family's ``checkpoint_components()`` state
    dicts; ``backbone`` is the toolkit's backbone argument — a bf16
    snapshot dict when this run trains the trunk, the inherited
    ``backbone.safetensors`` FILE when a frozen run inherited an adapted
    trunk, or the pristine snapshot DIRECTORY to hard-link-mirror."""

    components: dict[str, dict[str, Tensor]]
    backbone: dict[str, Tensor] | Path


def trained_backbone_snapshot(backbone: nn.Module) -> dict[str, Tensor]:
    """The trained-trunk state for ``backbone.safetensors``: parameters
    cast bf16 (the fp32 masters' precision beyond bf16 lives only in
    optimizer.pt), buffers at native dtype (RoPE inv_freq tables are fp32
    by design — bf16 would corrupt them). The copy+cast happens
    HOST-side: a device-side cast would transiently allocate ~4.3 GB of
    VRAM, an OOM at the ~79 GB/80 GB occupancy measured for the
    live-backbone DDP config (A100, batch 32)."""
    parameter_names = {name for name, _ in backbone.named_parameters()}
    return {
        name: (
            tensor.detach().cpu().to(torch.bfloat16)
            if name in parameter_names
            else tensor.detach().cpu()
        ).contiguous()
        for name, tensor in backbone.state_dict().items()
        if name not in BACKBONE_UNSAVED_KEYS
    }


def capture_checkpoint_tensors(
    model: VLA[Any],
    backbone: nn.Module,
    *,
    args: TrainArgs,
    adapted_backbone_source: Path | None,
    pristine_trunk_dir: Path,
) -> CheckpointTensors:
    """Device->CPU copies of every tensor the checkpoint serializes.
    ``copy=True`` even for CPU runs: the snapshot must not alias live
    parameters the next optimizer step mutates. The backbone form
    follows the D9 invariant: trained this run → bf16 snapshot dict;
    inherited adapted (frozen) → the source file to link; pristine →
    the mounted trunk directory to mirror."""
    components = {
        name: {
            key: tensor.detach().to("cpu", copy=True).contiguous()
            for key, tensor in module.state_dict().items()
        }
        for name, module in model.checkpoint_components().items()
    }
    backbone_form: dict[str, Tensor] | Path
    if args.backbone_trained:
        backbone_form = trained_backbone_snapshot(backbone)
    elif adapted_backbone_source is not None:
        if not adapted_backbone_source.is_file():
            raise FileNotFoundError(
                f"inherited adapted-backbone source {adapted_backbone_source} "
                "disappeared — every checkpoint of this run must carry it",
            )
        backbone_form = adapted_backbone_source
    else:
        backbone_form = pristine_trunk_dir
    return CheckpointTensors(components=components, backbone=backbone_form)


def write_checkpoint(
    checkpoint_dir: Path,
    *,
    metadata: VLAMetadata,
    tensors: CheckpointTensors,
    train_state_payload: dict[str, Any],
) -> Path:
    """Serialize one checkpoint through the VLA toolkit
    (:func:`bijou.checkpoint.write_checkpoint`): everything lands in a
    sibling ``.tmp`` directory first, is validated for self-containment,
    and a single rename publishes it — a crash mid-write leaves every
    earlier checkpoint intact and no half-written ``step_*`` directory a
    resume/eval could mistake for a real one (the ``.tmp`` debris is
    evidence, clobbered by the next attempt). Pure CPU+disk — safe on
    the async saver's background thread. Re-saving the same step
    (a resume re-hitting a boundary after a crash) replaces the old
    directory wholesale."""
    if checkpoint_dir.exists():
        print(
            f"replacing existing {checkpoint_dir} (same-boundary re-save)",
            flush=True,
        )
        shutil.rmtree(checkpoint_dir)
    # The toolkit links an EXISTING optimizer file into the staging dir;
    # serialize the payload beside the target first, then drop the
    # original once the link is placed (the inode survives in the
    # checkpoint).
    optimizer_scratch = checkpoint_dir.parent / (checkpoint_dir.name + ".optimizer.pt")
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_state_payload, optimizer_scratch)
    try:
        write_vla_checkpoint(
            checkpoint_dir,
            metadata=metadata,
            components=tensors.components,
            backbone=tensors.backbone,
            optimizer=optimizer_scratch,
        )
    finally:
        optimizer_scratch.unlink(missing_ok=True)
    return checkpoint_dir


def save_checkpoint(
    model: TrainableVLA,
    backbone: nn.Module,
    *,
    args: TrainArgs,
    normalizers: Normalizers,
    per_dataset_stats: dict[str, DatasetStats],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    adapted_backbone_source: Path | None,
    pristine_trunk_dir: Path,
) -> Path:
    """Write one self-contained checkpoint directory (the synchronous
    entry: capture + write inline; the async path drives the same capture
    and writer from bijou.async_save).

    Invariant (D9): the backbone is ALWAYS materialized —
    ``backbone.safetensors`` when the trunk state differs from pristine
    (trained this run, or inherited from an adapted checkpoint with the
    unfreeze flags off — ``adapted_backbone_source``), else a
    hard-linked ``backbone/`` mirror of the pristine snapshot;
    ``metadata.backbone.trained`` records the fact explicitly."""
    train_state = TrainState(
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        step=step,
    )
    return write_checkpoint(
        args.save_dir / f"step_{step:06d}",
        metadata=build_vla_metadata(
            model,
            args=args,
            normalizers=normalizers,
            per_dataset_stats=per_dataset_stats,
            step=step,
            adapted_backbone_source=adapted_backbone_source,
        ),
        tensors=capture_checkpoint_tensors(
            model,
            backbone,
            args=args,
            adapted_backbone_source=adapted_backbone_source,
            pristine_trunk_dir=pristine_trunk_dir,
        ),
        train_state_payload=train_state.to_payload(),
    )


def build_vla_metadata(
    model: TrainableVLA,
    *,
    args: TrainArgs,
    normalizers: Normalizers,
    per_dataset_stats: dict[str, DatasetStats],
    step: int,
    adapted_backbone_source: Path | None,
) -> VLAMetadata:
    """The checkpoint's ``metadata.json`` record. Cheap and pure — runs
    at capture time so the async writer holds no model references.

    Component records (name → config + weights flag) come from the
    concrete family: WEIGHTED components mirror
    ``checkpoint_components()`` exactly (the toolkit cross-checks);
    parameterless components (the molmoact2 prompt side and discrete
    rider) record their configs with ``weights: false`` — the metadata
    is where parameterless configs live."""
    components: dict[str, dict[str, Any]]
    depth = BackboneDepth.FULL
    if isinstance(model, GemmaFlowVLA | GemmaARVLA):
        encoder = model.encoder
        prompt_dict = GemmaPromptConfig(
            exports=encoder.exports,
            max_soft_tokens=args.max_soft_tokens,
            format=PROMPT_FORMAT,
            state_dim=encoder.state_dim,
            condition_fields=tuple(args.condition_fields or ()),
            generate_bracket=(
                args.family in AR_SUFFIX_FAMILIES or args.prompt_generate_bracket
            ),
        ).to_dict()
        # Structural fact of the built model: a truncated backbone has
        # its KV-shared region cut away (truncated_config), a full one
        # keeps it — no plumbing to drift.
        backbone_config = model.backbone.config
        assert isinstance(backbone_config, Gemma4Config)
        depth = (
            BackboneDepth.FULL
            if backbone_config.text.num_kv_shared_layers > 0
            else BackboneDepth.PREFIX
        )
        if isinstance(model, GemmaFlowVLA):
            components = {
                "prompt": {"config": prompt_dict, "weights": True},
                "flow_decoder": {
                    "config": decoder_schema_dict(model.flow_decoder),
                    "weights": True,
                },
            }
        else:
            components = {
                "prompt": {"config": prompt_dict, "weights": True},
                "ar_decoder": {
                    "config": decoder_schema_dict(model.ar_decoder),
                    "weights": True,
                },
            }
    elif isinstance(model, Molmo2ARVLA):
        encoder = model.encoder
        prompt_dict = Molmo2PromptConfig(
            max_crops=encoder.max_crops,
            format=MOLMO2_PROMPT_FORMAT,
            state_dim=encoder.state_dim,
            condition_fields=tuple(args.condition_fields or ()),
            generate_bracket=True,  # the AR-suffix families always render it
        ).to_dict()
        components = {
            "prompt": {"config": prompt_dict, "weights": True},
            "ar_decoder": {
                "config": decoder_schema_dict(model.ar_decoder),
                "weights": True,
            },
        }
    else:
        encoder = model.encoder
        if encoder.prompt_schema is None:
            raise ValueError(
                "MolmoAct2Encoder has no stashed prompt schema — was the "
                "model built outside the train path?",
            )
        components = {
            # The encoder owns zero parameters; its config rides the
            # metadata, never a file.
            "prompt": {"config": dict(encoder.prompt_schema), "weights": False},
        }
        if isinstance(model, MolmoAct2FlowVLA | MolmoAct2JointVLA):
            components["flow_decoder"] = {
                "config": decoder_schema_dict(model.flow_decoder),
                "weights": True,
            }
        if isinstance(model, MolmoAct2ARVLA | MolmoAct2JointVLA):
            components["ar_decoder"] = {
                "config": decoder_schema_dict(model.ar_decoder),
                "weights": False,
            }
    weighted = {name for name, record in components.items() if record["weights"]}
    declared = set(model.checkpoint_components())
    if weighted != declared:
        raise SystemExit(
            f"metadata weighted components {sorted(weighted)} != the "
            f"family's checkpoint_components() {sorted(declared)} — the "
            "write side drifted from the family's declaration",
        )

    normalization = aggregate_stats(normalizers)
    if isinstance(model, MolmoAct2FlowVLA | MolmoAct2JointVLA):
        # The load-bearing tables (the merged scheme) ride the
        # normalization row: the run aggregate honestly carries no
        # quantiles (aggregate_stats), but molmo_flow NORMALIZED with the
        # source checkpoint's merged tables — the written row must carry
        # the tables in use or the descendant checkpoint loses its clamp.
        flow_decoder = model.flow_decoder
        runtime = flow_decoder.runtime
        assert runtime is not None  # configure() ran at build
        encoder = model.encoder
        assert isinstance(encoder, MolmoAct2Encoder)
        assert encoder.state_table is not None
        normalization = dataclasses.replace(
            normalization,
            action_q01=tuple(
                flow_decoder.action_q01[: runtime.action_dim].tolist(),
            ),
            action_q99=tuple(
                flow_decoder.action_q99[: runtime.action_dim].tolist(),
            ),
            state_q01=encoder.state_table[0],
            state_q99=encoder.state_table[1],
        )
    elif isinstance(model, MolmoAct2ARVLA):
        # The ar family has no flow decoder to read tables off — the
        # collator tokenized under the encoder-stashed merged tables;
        # the written row carries THE tables in use (same invariant as
        # the flow branch, different source of truth).
        encoder = model.encoder
        assert isinstance(encoder, MolmoAct2Encoder)
        assert encoder.state_table is not None
        assert encoder.action_table is not None
        normalization = dataclasses.replace(
            normalization,
            action_q01=encoder.action_table[0],
            action_q99=encoder.action_table[1],
            state_q01=encoder.state_table[0],
            state_q99=encoder.state_table[1],
        )

    artifacts: dict[str, str] = {}
    if args.fast_tokenizer is not None:
        artifacts["fast_tokenizer"] = args.fast_tokenizer

    spec = model.spec
    assert spec.family.value == args.family  # construction routed correctly
    return VLAMetadata(
        family=spec.family,
        chunk_size=spec.chunk_size,
        action_dim=spec.action_dim,
        backbone_id=args.backbone,
        backbone_depth=depth.value,
        backbone_trained=(args.backbone_trained or adapted_backbone_source is not None),
        objective=objective_to_json(build_objective(args)),
        serving=serving_to_json(model.serving),
        components=components,
        artifacts=artifacts,
        stats=normalization,
        per_dataset_stats=per_dataset_stats,
        train_args=train_args_record(args),
        step=step,
        stats_note=None,
    )
