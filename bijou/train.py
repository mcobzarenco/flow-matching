"""Train Bijou (flow-matching action expert) on a LeRobot v3 dataset.

The frozen truncated backbone encodes the multimodal prefix per batch
(no grad); the expert is optimized with the π0/SmolVLA flow-matching recipe:
τ ~ Beta(1.5, 1) (scaled into (0, 1)), x_τ = τ·ε + (1−τ)·actions, MSE against
the velocity target ε − actions, over the full chunk — episode-boundary
chunks carry repeat-last-action targets (hold still after task completion)
rather than masked padding. Actions and state are MEAN_STD-normalized
**per dataset** (each sample uses its own dataset's stats, the π0/SmolVLA
convention): 59–95% of the
aggregate action variance across the community collections is between-dataset
rig offsets that images cannot see, and normalizing them away is what makes
the state→action identity learnable (measured: aggregate normalization left
the trained model behind the state-copy baseline). Checkpoints store the
per-dataset stats table plus a count-weighted aggregate as a fallback for
rigs without stats; inference must unnormalize with the deployment rig's
stats.

The prompt is the instruction sandwich discussed in the design:
``[instruction][cam_1]...[cam_N][instruction]`` inside a user chat turn,
giving instruction-conditioned image KV and image-conditioned instruction KV
under causal attention.

Training data is selected with ``--train-data`` via the shared selection
pipeline in ``bijou.data`` (also used by ``bijou.eval``): collection roots
and dataset dirs, loud drops for incompatible/corrupt datasets, per-dataset
stats attachment, chat-templated prompt collation.

Usage::

    uv run python -m bijou.train \
        --train-data ~/datasets/mcobzarenco/community_dataset_v2_v3 \
        --device cuda --steps 5000

    # Multi-GPU: one full replica + optimizer per GPU (DDP over the expert
    # only; the frozen backbone is never synced). --batch-size and
    # --num-workers are PER RANK; logged loss and eval MAE are all-reduced
    # across ranks (the eval set is sharded); logging/checkpoints happen on
    # rank 0. Without torchrun the script runs exactly as before.
    MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 \
    uv run torchrun --standalone --nproc-per-node=4 -m bijou.train \
        --train-data ~/datasets/mcobzarenco/community_dataset_v2_v3 \
        --device cuda --steps 20000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, override

import matplotlib
import matplotlib.pyplot as plt
import torch
import wandb
from safetensors.torch import load_file, save_file
from torch import Tensor

from .data import (
    DatasetStats,
    EpisodeSplit,
    encode_prefix,
    select_datasets,
    worker_init,
)
from .encoders.gemma4 import GemmaInputs, GemmaInputsCollator
from .expert import ExpertConfig, SelfAttentionMode, TimeConditioning
from .gemma4.loading import load_config, resolve_checkpoint_dir
from .gemma4.text import DecoderLayer
from .interface import CollatedBatch, Collator, EncodedPrefix
from .loading import (
    CheckpointMetadata,
    backbone_snapshot,
    default_expert_config,
    flow_decoder_config_from_expert,
    from_backbone,
    load_adapted_backbone,
)
from .model import BijouModel

DEFAULT_BACKBONE = "google/gemma-4-e2b-it"
# Rows in the wandb probe table (each costs camera images + a matplotlib
# figure per eval): a spot check, deliberately not scaled to the probe size.
EVAL_TABLE_ROWS = 32


@dataclass(frozen=True, slots=True)
class TrainArgs:
    train_data: tuple[Path, ...]
    exclude: tuple[str, ...]
    fps: tuple[float, ...] | None
    holdout_episodes: float
    split_seed: int
    backbone: str
    save_dir: Path
    init_from: Path | None
    resume: Path | None
    instruction: str | None
    cameras: tuple[str, ...] | None
    max_cameras: int | None
    max_soft_tokens: int
    stream_counts: tuple[int, ...]
    self_attention_mode: str
    time_conditioning: str
    expert_hidden: int
    expert_heads: int
    expert_intermediate: int
    expert_cross_heads: int
    chunk_size: int
    batch_size: int
    steps: int
    expert_lr: float
    text_lr: float | None
    vision_lr: float | None
    warmup_steps: int
    weight_decay: float
    grad_clip: float

    log_every: int
    eval_every: int
    save_every: int
    num_workers: int
    prefetch_factor: int
    video_decoder_cache: int
    device: str
    seed: int
    eval_samples: int | None
    eval_seed: int
    wandb_project: str | None
    wandb_run_name: str | None

    @property
    def trunk_trained(self) -> bool:
        return self.text_lr is not None or self.vision_lr is not None


class DevicePrefetcher:
    """One-batch-lookahead host->device transfer on a side CUDA stream.

    Centralizes every H2D copy of the training pipeline: the loop receives
    batches already device-resident, and the copy of batch N+1 overlaps the
    compute of batch N (dataloader workers cannot produce CUDA tensors —
    each would need its own CUDA context — so this is the closest torch gets
    to "the loader hands you device batches"). Degrades to plain synchronous
    transfers on non-CUDA devices.
    """

    def __init__(
        self,
        loader: Iterable[CollatedBatch[GemmaInputs]],
        device: torch.device,
    ) -> None:
        self.loader = loader
        self.device = device

    def __iter__(self) -> Iterator[CollatedBatch[GemmaInputs]]:
        if self.device.type != "cuda":
            for batch in self.loader:
                yield batch.to(self.device)
            return

        stream = torch.cuda.Stream(self.device)
        compute_stream = torch.cuda.current_stream(self.device)
        batches = iter(self.loader)

        def preload() -> CollatedBatch[GemmaInputs] | None:
            cpu_batch = next(batches, None)
            if cpu_batch is None:
                return None
            with torch.cuda.stream(stream):
                return cpu_batch.to(self.device, non_blocking=True)

        batch = preload()
        while batch is not None:
            compute_stream.wait_stream(stream)
            # The tensors were allocated on the side stream; tell the caching
            # allocator they are consumed on the compute stream.
            for tensor in batch.all_tensors():
                tensor.record_stream(compute_stream)
            yield batch  # consumer enqueues the step's compute, then returns
            batch = preload()  # blocks on workers while the GPU crunches


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


def flow_matching_loss(
    velocity_model: torch.nn.Module,
    prefix: EncodedPrefix,
    batch: CollatedBatch[Any],
) -> Tensor:
    """``batch`` must already be device-resident; no transfers happen here.
    Actions/state are normalized with each sample's own dataset stats.

    Episode-boundary chunks train on the full ``chunk`` length with
    repeat-last-action targets: lerobot's delta-timestamps query clamps
    indices to the episode range (dataset_reader._get_query_indices), so
    positions past the end already hold the final real action — the
    desired "reach and hold" target — and ``action_is_pad`` is deliberately
    ignored here (decision 2026-07-29: full-chunk targets over masking;
    the expert attends every position, so masked-out padding was still
    silently shaping predictions). Eval stays real-steps-only.

    ``velocity_model`` is the expert (or its DDP wrapper under torchrun —
    training forwards must go through the wrapper for gradient hooks);
    call convention (prefix, state, noisy_actions, tau) is shared by
    ActionExpert, BijouModel and DDP(ActionExpert).

    Shapes (batch fields in CollatedBatch's docstring):
      - prefix.streams[layer]: (key, value), each [B, kv_heads, P, head_dim]
      - velocity/target: [B, chunk, action_dim]; tau: [B]
      - returns: scalar loss
    """
    actions = (
        batch.actions - batch.action_stats.mean[:, None, :]
    ) / batch.action_stats.std[:, None, :]
    state = (batch.state - batch.state_stats.mean) / batch.state_stats.std

    noise = torch.randn_like(actions)
    # π0's time distribution: Beta(1.5, 1) squeezed into (0, 1).
    tau = (
        torch.distributions.Beta(1.5, 1.0)
        .sample((actions.shape[0],))
        .to(actions.device)
    )
    tau = tau * 0.999 + 0.001
    tau_ = tau[:, None, None]
    noisy_actions = tau_ * noise + (1 - tau_) * actions
    target = noise - actions

    velocity = velocity_model(prefix, state, noisy_actions, tau)
    return (velocity.float() - target.float()).pow(2).mean()


def trainable_text_parameters(model: BijouModel) -> Iterator[torch.nn.Parameter]:
    """The text-trunk parameters that PARTICIPATE in a prefix encode, and
    only those — the set must be exact because DDP requires every
    grad-enabled parameter to receive gradients each step.

    Participation mirrors ``kv_stop_layer`` (= the deepest exported
    stream): layers below it run fully; the stop layer runs only its
    input layernorm and K/V projections (``TextAttention.project_kv``;
    its v_norm is scale-less — no parameters); deeper layers, the final
    norm and the LM head never run. Token embeddings and the PLE tables
    stay frozen BY DESIGN (few rows touched per batch, dense Adam state
    for a 262k vocab is waste, and frozen embeddings are the cheapest
    forgetting control) — but the PLE *projection* path runs for every
    consumed layer slice, so it trains. The multimodal projector
    (embed_vision) is the vision->text interface and belongs to the text
    group regardless of whether the vision tower itself is unfrozen.
    """
    text = model.backbone.language_model
    stop_layer = max(model.expert.config.streams)
    for idx, layer in enumerate(text.layers):
        # ModuleList iteration erases the element type (torch types
        # Module.__getattr__ as Tensor | Module): narrow before access.
        assert isinstance(layer, DecoderLayer)
        if idx < stop_layer:
            yield from layer.parameters()
        elif idx == stop_layer:
            yield from layer.input_layernorm.parameters()
            attention = layer.self_attn
            assert attention.k_proj is not None
            assert attention.k_norm is not None
            yield from attention.k_proj.parameters()
            if attention.v_proj is not None:
                yield from attention.v_proj.parameters()
            yield from attention.k_norm.parameters()
    yield from text.per_layer_model_projection.parameters()
    yield from text.per_layer_projection_norm.parameters()
    assert model.backbone.embed_vision is not None
    yield from model.backbone.embed_vision.parameters()


@dataclass(frozen=True, slots=True)
class TrunkParameterCounts:
    """Trainable trunk parameters enabled by :func:`unfreeze_backbone`,
    by subsystem (0 = that subsystem stayed frozen)."""

    text: int
    vision: int


def unfreeze_backbone(model: BijouModel, args: TrainArgs) -> TrunkParameterCounts:
    """Flip ``requires_grad`` on the requested trunk subsets; everything
    else stays frozen (``load_model`` freezes the whole backbone).

    Freezing by ``requires_grad`` alone is sufficient for efficiency too:
    token embeddings, PLE tables and (when frozen) the vision tower feed
    the decoder grad-free inputs, so autograd never builds their graphs —
    no activation cost, no backward — without any code-path changes
    inside gemma4.
    """
    text = 0
    vision = 0
    if args.text_lr is not None:
        for parameter in trainable_text_parameters(model):
            parameter.requires_grad_(True)
            text += parameter.numel()
    if args.vision_lr is not None:
        if model.backbone.vision_tower is None:
            raise SystemExit(
                f"--vision-lr {args.vision_lr} but the "
                f"backbone ({args.backbone}) has no vision tower — drop the "
                "flag or use a multimodal backbone",
            )
        for parameter in model.backbone.vision_tower.parameters():
            parameter.requires_grad_(True)
            vision += parameter.numel()
    return TrunkParameterCounts(text=text, vision=vision)


def decay_split(
    parameters: Iterable[torch.nn.Parameter],
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """(decayed, undecayed): weight decay belongs on matmul weights, not on
    norm scales — the standard ndim heuristic (norm weights and biases are
    1-D)."""
    decayed: list[torch.nn.Parameter] = []
    undecayed: list[torch.nn.Parameter] = []
    for parameter in parameters:
        (decayed if parameter.dim() >= 2 else undecayed).append(parameter)
    return decayed, undecayed


class BijouTrainStep(torch.nn.Module):
    """One training forward with a LIVE trunk: prefix encode (with grad,
    bf16 autocast on CUDA) + expert velocity + loss, as a single module so
    a single DDP wrapper hooks gradients of both the trunk and the expert.

    The frozen path deliberately does NOT use this class — it keeps the
    original no-grad encode + DDP(expert) wiring, byte-identical to every
    run before the unfreeze flags existed (and to the CPU loss oracle).

    The trunk runs under bf16 autocast while its master weights are fp32
    (direct bf16 updates vanish below bf16 resolution at trunk-scale
    learning rates). The expert runs OUTSIDE the autocast region, fp32
    with TF32 matmuls, exactly as in the frozen path; it already casts
    the (autocast-bf16) K/V streams to its own dtype.
    """

    def __init__(self, model: BijouModel) -> None:
        super().__init__()
        self.model = model

    @override
    def forward(self, batch: CollatedBatch[GemmaInputs]) -> Tensor:
        """Batch (shapes in CollatedBatch's docstring) -> scalar loss."""
        inputs = batch.encoder_inputs
        device_type = inputs.input_ids.device.type
        with torch.autocast(device_type, torch.bfloat16, enabled=device_type == "cuda"):
            prefix = self.model.encode_prefix(
                inputs.input_ids,
                pixel_values=inputs.pixel_values,
                image_position_ids=inputs.image_position_ids,
                padding_mask=inputs.attention_mask if inputs.has_padding else None,
            )
        return flow_matching_loss(self.model.expert, prefix, batch)


def _chunk_plot(
    predicted: Tensor,
    truth: Tensor,
    valid: Tensor,
    state: Tensor,
    action_names: list[str],
) -> Any:
    """Per-joint curves over the action chunk: ground truth, the model's
    prediction, and the trivial state-copy baseline (hold current joint
    positions — the minimum bar a learned policy must clear). Returns a
    matplotlib figure (caller logs and closes it).

    Shapes: predicted/truth [chunk, action_dim]; valid [chunk] (bool);
    state [state_dim] — one sample, CPU-resident."""
    dims = predicted.shape[-1]
    ncols = 3
    nrows = (dims + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.5 * nrows),
        squeeze=False,
    )
    steps = range(predicted.shape[0])
    n_valid = int(valid.sum())
    for dim in range(dims):
        ax = axes[dim // ncols][dim % ncols]
        ax.plot(
            steps[:n_valid],
            truth[:n_valid, dim].tolist(),
            label="truth",
            color="black",
            linewidth=1.8,
        )
        ax.plot(
            steps[:n_valid],
            [float(state[dim])] * n_valid,
            label="state-copy",
            color="tab:blue",
            linestyle="--",
            linewidth=1.2,
        )
        ax.plot(
            steps[:n_valid],
            predicted[:n_valid, dim].tolist(),
            label="predicted",
            color="tab:orange",
            linestyle="--",
            linewidth=1.2,
        )
        name = action_names[dim] if dim < len(action_names) else f"dim {dim}"
        ax.set_title(name, fontsize=9)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    return fig


@dataclass(frozen=True, slots=True)
class RichRow:
    """One probe sample's prediction, kept (CPU-side) for the wandb table."""

    sampled: Tensor
    truth: Tensor
    valid: Tensor
    state: Tensor


@dataclass(frozen=True, slots=True)
class ProbeSet:
    """This rank's shard of a seeded MAE probe, CPU-resident between evals.

    ``total`` is the global probe size across ranks. ``rich_items`` are raw
    items kept for wandb rich logging at ``rich_positions`` (positions in
    this shard's streaming order, strided across the shard so the table
    spans the concatenated datasets instead of the earliest ones).
    """

    total: int
    batches: list[CollatedBatch[GemmaInputs]]
    rich_items: list[dict[str, Any]]
    rich_positions: tuple[int, ...]


def build_probe_set(
    dataset: torch.utils.data.ConcatDataset[dict[str, Any]],
    collator: Collator[GemmaInputs],
    num_samples: int,
    seed: int,
    rank: int,
    world_size: int,
    batch_size: int,
    *,
    keep_rich: bool,
) -> ProbeSet:
    """Draw, fetch and collate one probe set's shard for this rank.

    The frame draw is exactly bijou.eval's: seeded sampling without
    replacement over the same selection scores the same frames. The sorted
    draw is striped round-robin across ranks, so every shard spreads evenly
    over the concatenated datasets.
    """
    num = min(num_samples, len(dataset))
    indices = sorted(random.Random(seed).sample(range(len(dataset)), num))
    shard = indices[rank::world_size]
    items = [dataset[i] for i in shard]
    batches = [
        collator(items[i : i + batch_size]) for i in range(0, len(items), batch_size)
    ]
    rich_positions: tuple[int, ...] = ()
    rich_items: list[dict[str, Any]] = []
    if keep_rich and items:
        stride = max(len(items) // EVAL_TABLE_ROWS, 1)
        rich_positions = tuple(range(0, len(items), stride))[:EVAL_TABLE_ROWS]
        rich_items = [items[p] for p in rich_positions]
    return ProbeSet(
        total=num,
        batches=batches,
        rich_items=rich_items,
        rich_positions=rich_positions,
    )


@torch.no_grad()
def validate(
    model: BijouModel,
    probe: ProbeSet,
    device: torch.device,
    seed: int,
    *,
    distributed: bool = False,
    wandb_run: Any = None,
    collator: Collator[GemmaInputs] | None = None,
    action_names: list[str] | None = None,
    step: int = 0,
    table_key: str = "eval/samples",
) -> float:
    """Sampled-chunk MAE in raw action units over this rank's shard of the
    probe set; with ``distributed`` the sums all-reduce to the global value
    (collective — every rank must call this at the same step). Batches
    arrive CPU-resident and visit the device one at a time, and the prefix
    is re-encoded per eval, so probe size costs host RAM, not GPU memory.
    The valid-element-weighted aggregation is exactly bijou.eval's
    chunk_mae. Normalization is per dataset (each sample's own stats,
    matching training). With a wandb run and a probe carrying rich items,
    also logs a table under ``table_key``: camera images, task, state,
    per-joint predicted-vs-truth plots."""
    totals = torch.zeros(2, device=device)  # [abs-error sum, valid elements]
    rich_rows: list[RichRow] = []
    wanted = iter(probe.rich_positions)
    next_rich = next(wanted, None)
    base = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for cpu_batch in probe.batches:
        batch = cpu_batch.to(device)
        prefix = encode_prefix(model, batch)
        state = (batch.state - batch.state_stats.mean) / batch.state_stats.std
        # Eval is a measurement: Heun-10 keeps integration error well below
        # model error (0.018 vs 0.05 mean deviation at the Heun-5 deployment
        # default; ~1-2s extra per eval, off the training path).
        sampled = model.sample_actions(prefix, state, num_steps=10, generator=generator)
        sampled = (
            sampled.float() * batch.action_stats.std[:, None, :]
            + batch.action_stats.mean[:, None, :]
        )
        truth = batch.actions.float()
        valid = ~batch.action_is_pad
        error = (sampled - truth).abs()
        totals[0] += error[valid].sum()
        totals[1] += valid.sum() * error.shape[-1]
        # Batches stream in shard order: pick off the rich positions
        # (matching probe.rich_items one-to-one) as they pass.
        while next_rich is not None and next_rich < base + sampled.shape[0]:
            i = next_rich - base
            rich_rows.append(
                RichRow(
                    sampled=sampled[i].cpu(),
                    truth=truth[i].cpu(),
                    valid=valid[i].cpu(),
                    state=batch.state[i].cpu(),
                ),
            )
            next_rich = next(wanted, None)
        base += sampled.shape[0]
    if distributed:
        torch.distributed.all_reduce(totals)
    mae = float(totals[0] / totals[1].clamp(min=1))

    if wandb_run is not None and probe.rich_items and collator is not None:
        # Cameras vary per sample across mixed datasets: generic positional
        # columns, padded with None where a sample has fewer cameras.
        per_item_cameras = [collator.cameras_of(item) for item in probe.rich_items]
        n_slots = max(len(cams) for cams in per_item_cameras)
        columns: list[Any] = [
            "sample",
            *(f"camera_{i}" for i in range(n_slots)),
            "task",
            "state",
            "chunk_mae",
            "pred_vs_truth",
        ]
        table = wandb.Table(columns=columns)
        for i, (item, row) in enumerate(zip(probe.rich_items, rich_rows, strict=True)):
            cams = per_item_cameras[i]
            images: list[Any] = [
                wandb.Image(
                    (item[camera].clamp(0, 1) * 255)
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .numpy(),
                    caption=camera.removeprefix("observation.images."),
                )
                for camera in cams
            ]
            images += [None] * (n_slots - len(cams))
            figure = _chunk_plot(
                row.sampled,
                row.truth,
                row.valid,
                row.state,
                action_names or [],
            )
            state_str = ", ".join(
                f"{x:.1f}" for x in item["observation.state"].tolist()
            )
            table.add_data(
                i,
                *images,
                str(item["task"]),
                state_str,
                float((row.sampled - row.truth).abs()[row.valid].mean()),
                wandb.Image(figure),
            )
            plt.close(figure)
        wandb_run.log({table_key: table}, step=step)
    return mae


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


def save_checkpoint(
    model: BijouModel,
    args: TrainArgs,
    normalizers: Normalizers,
    per_dataset_stats: dict[str, DatasetStats],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
) -> Path:
    checkpoint_dir = args.save_dir / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.expert.state_dict(), str(checkpoint_dir / "expert.safetensors"))
    if args.trunk_trained:
        # Adapted trunks ride along; from_checkpoint/--init-from detect the
        # file by presence. Frozen runs write exactly the historical layout.
        save_file(
            backbone_snapshot(model),
            str(checkpoint_dir / "backbone.safetensors"),
        )
    # Adam moments etc. (~2x expert params) make --resume a lossless
    # continuation; --init-from ignores this file.
    train_state = TrainState(
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        step=step,
    )
    torch.save(train_state.to_payload(), checkpoint_dir / "optimizer.pt")
    metadata = CheckpointMetadata(
        backbone=args.backbone,
        expert_config=model.expert.config,
        max_soft_tokens=args.max_soft_tokens,
        normalization=aggregate_stats(normalizers),
        per_dataset_normalization=per_dataset_stats,
        train_args={
            k: str(v) if isinstance(v, Path) else v
            for k, v in dataclasses.asdict(args).items()
        },
        step=step,
    )
    (checkpoint_dir / "bijou_config.json").write_text(
        json.dumps(metadata.to_json_dict(), indent=2, default=str),
    )
    return checkpoint_dir


def ensure_matching_expert_config(
    expert_config: ExpertConfig,
    checkpoint: Path,
) -> None:
    """Loud, early failure when a checkpoint's expert differs from the CLI's
    (strict state-dict loading would also fail, but with worse diagnostics
    — and silently NOT fail for same-shape config differences like the
    cross-attention schedule). Handles both checkpoint formats: format 2
    compares decoder configs; format 1 compares the historical serialized
    expert_config."""
    meta = json.loads((checkpoint / "bijou_config.json").read_text())
    if "decoder" in meta:
        saved = meta["decoder"]
        current = flow_decoder_config_from_expert(expert_config).to_dict()
    else:
        saved = meta["expert_config"]
        # Back-compat: fields added to ExpertConfig after a checkpoint was
        # written are absent from its serialized config; fill their defaults
        # so an unchanged run still matches. A pre-adaRMS checkpoint is
        # additive.
        saved.setdefault("time_conditioning", TimeConditioning.ADDITIVE.value)
        current = json.loads(
            json.dumps(dataclasses.asdict(expert_config), default=str),
        )
    if current != saved:
        raise SystemExit(
            f"expert config mismatch vs {checkpoint}:\n"
            f"  checkpoint: {json.dumps(saved, sort_keys=True)}\n"
            f"  cli:        {json.dumps(current, sort_keys=True)}",
        )


def lr_lambda(step: int, args: TrainArgs) -> float:
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(
        prog="python -m bijou.train",
        description="Train the Bijou action expert on LeRobot v3 datasets "
        "(dataset directories and/or collection roots). Runs on a single "
        "GPU by default and data-parallel under torchrun; checkpoints "
        "carry everything bijou.eval and bijou.rollout need.",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        nargs="+",
        required=True,
        help="dataset directories and/or collection roots, mixed freely",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="fnmatch patterns against <user>/<dataset> repo ids to skip",
    )
    parser.add_argument(
        "--fps",
        type=float,
        nargs="+",
        default=None,
        help="keep only datasets recorded at one of these frame rates "
        "(e.g. --fps 30); default keeps every fps, the historical "
        "behavior. Filtering changes the concatenated frame indexing, so "
        "eval numbers are only comparable between runs with the same "
        "filter",
    )
    parser.add_argument(
        "--holdout-episodes",
        type=float,
        default=0.0,
        help="fraction of each dataset's episodes to exclude from training; "
        "score them with bijou.eval --episodes holdout (same fraction and "
        "--split-seed)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="episode-holdout seed, independent of --seed so restarts "
        "never shift the split",
    )
    parser.add_argument(
        "--backbone",
        default=DEFAULT_BACKBONE,
        help="backbone HF model id or local checkpoint path",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/train/bijou_dev"),
        help="output directory: step_NNNNNN/ checkpoints, train_log.jsonl, wandb files",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="override the per-frame task string from the dataset",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="only use these camera keys when present (full keys or "
        "suffixes; default: all cameras of each sample, sorted)",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=None,
        help="cap cameras per sample (applied after --cameras filtering)",
    )
    parser.add_argument(
        "--max-soft-tokens",
        type=int,
        default=140,
        help="vision soft-token budget per camera in the prompt",
    )
    parser.add_argument(
        "--stream-counts",
        type=int,
        nargs="*",
        default=[4, 4, 7],
        help="expert cross-attention layers per backbone KV stream, "
        "shallow to deep (0 skips a stream)",
    )
    parser.add_argument(
        "--self-attention-mode",
        choices=["causal_actions", "bidirectional"],
        default="causal_actions",
        help="expert self-attention over the action chunk",
    )
    parser.add_argument(
        "--time-conditioning",
        choices=[m.value for m in TimeConditioning],
        default=TimeConditioning.ADDITIVE.value,
        help="how flow time τ conditions the expert: 'additive' (π0-style "
        "input add, the default) or 'adarms' (DiT-style per-layer scale/"
        "gate, identity at init). adarms changes the architecture — a fresh "
        "expert only (cannot --init-from an additive checkpoint)",
    )
    parser.add_argument(
        "--expert-hidden",
        type=int,
        default=768,
        help="expert hidden size",
    )
    parser.add_argument(
        "--expert-heads",
        type=int,
        default=6,
        help="expert self-attention heads",
    )
    parser.add_argument(
        "--expert-intermediate",
        type=int,
        default=3072,
        help="expert MLP intermediate size",
    )
    parser.add_argument(
        "--expert-cross-heads",
        type=int,
        default=4,
        help="expert cross-attention heads",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="actions predicted per sample (frames at the dataset fps)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="per rank under torchrun (global batch = batch-size x world size)",
    )
    parser.add_argument("--steps", type=int, default=200, help="total optimizer steps")
    parser.add_argument(
        "--expert-lr",
        type=float,
        default=1e-4,
        help="peak learning rate of the action expert (cosine decay to "
        "10%% after warmup); every component-lr below shares this "
        "schedule shape, scaled to its own peak",
    )
    parser.add_argument(
        "--text-lr",
        type=float,
        default=None,
        help="peak learning rate for the backbone TEXT trunk (decoder "
        "layers up to the deepest exported stream, PLE projections, "
        "multimodal projector); OMIT to keep the trunk frozen (the "
        "historical behavior). Token embeddings and PLE tables always "
        "stay frozen. A live trunk loads fp32 with bf16-autocast "
        "forwards; suggest 1e-5",
    )
    parser.add_argument(
        "--vision-lr",
        type=float,
        default=None,
        help="peak learning rate for the vision tower; OMIT to keep it "
        "frozen. The acuity probe says position is sharpest at the tower "
        "output and dies in the LM layers - expect to leave this unset",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="linear warmup steps to --lr",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
        help="gradient-norm clip over the expert",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="steps between metric logs",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="steps between MAE probes (eval_chunk_mae/train_mae)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="steps between checkpoints",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="dataloader workers per rank under torchrun",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="batches prefetched per dataloader worker",
    )
    parser.add_argument(
        "--video-decoder-cache",
        type=int,
        default=4,
        help="max cached video decoders per dataloader worker (exported as "
        "LEROBOT_VIDEO_DECODER_CACHE_SIZE; lerobot's default of 100 "
        "OOM-kills many-dataset runs)",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="warm start: expert weights from this checkpoint directory, "
        "fresh optimizer and step count (use a new --save-dir)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="full resume: weights + optimizer/scheduler/step from this "
        "checkpoint directory (--steps counts total, including resumed)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device (cuda, cuda:N, cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="training seed: init, data order, τ/ε draws (per-rank streams "
        "derive from it)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="MAE probe size: eval_chunk_mae on held-out episodes (with "
        "--holdout-episodes) plus train_mae on training data; sampled "
        "without replacement per split, sharded across ranks, evaluated in "
        "batches of --batch-size; required when --holdout-episodes > 0, "
        "omit to disable probing",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="probe sampling/noise seed, independent of --seed; matches the "
        "frames bijou.eval --seed picks on the same data and split",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="enable Weights & Biases logging to this project "
        "(WANDB_API_KEY must be set)",
    )
    parser.add_argument("--wandb-run-name", default=None, help="wandb run display name")
    raw = parser.parse_args()
    if raw.init_from is not None and raw.resume is not None:
        parser.error("--init-from and --resume are mutually exclusive")
    if not 0.0 <= raw.holdout_episodes < 1.0:
        parser.error("--holdout-episodes must be in [0, 1)")
    if raw.holdout_episodes > 0 and raw.eval_samples is None:
        parser.error(
            "--eval-samples is required when --holdout-episodes > 0 "
            "(it sizes the held-out eval_chunk_mae probe)",
        )
    if raw.eval_samples is not None and raw.eval_samples < 1:
        parser.error("--eval-samples must be >= 1")
    if raw.expert_lr <= 0:
        parser.error("--expert-lr must be > 0 (the expert always trains)")
    for name, value in (("--text-lr", raw.text_lr), ("--vision-lr", raw.vision_lr)):
        if value is not None and value <= 0:
            parser.error(
                f"{name} {value} is not a usable learning rate — omit the "
                "flag entirely to keep that component frozen",
            )
    if raw.vision_lr is not None and raw.text_lr is None:
        print(
            "NOTE: vision tower unfrozen with the text trunk FROZEN - "
            "gradients still traverse the frozen text stack to reach the "
            "tower (activation cost without text adaptation). Legitimate "
            "but unusual; the standard move is --text-lr alone.",
            file=sys.stderr,
            flush=True,
        )
    return TrainArgs(
        train_data=tuple(raw.train_data),
        exclude=tuple(raw.exclude),
        fps=tuple(raw.fps) if raw.fps else None,
        holdout_episodes=raw.holdout_episodes,
        split_seed=raw.split_seed,
        backbone=raw.backbone,
        save_dir=raw.save_dir,
        init_from=raw.init_from,
        resume=raw.resume,
        instruction=raw.instruction,
        cameras=tuple(raw.cameras) if raw.cameras else None,
        max_cameras=raw.max_cameras,
        max_soft_tokens=raw.max_soft_tokens,
        stream_counts=tuple(raw.stream_counts),
        self_attention_mode=raw.self_attention_mode,
        time_conditioning=raw.time_conditioning,
        expert_hidden=raw.expert_hidden,
        expert_heads=raw.expert_heads,
        expert_intermediate=raw.expert_intermediate,
        expert_cross_heads=raw.expert_cross_heads,
        chunk_size=raw.chunk_size,
        batch_size=raw.batch_size,
        steps=raw.steps,
        expert_lr=raw.expert_lr,
        text_lr=raw.text_lr,
        vision_lr=raw.vision_lr,
        warmup_steps=raw.warmup_steps,
        weight_decay=raw.weight_decay,
        grad_clip=raw.grad_clip,
        log_every=raw.log_every,
        eval_every=raw.eval_every,
        save_every=raw.save_every,
        num_workers=raw.num_workers,
        prefetch_factor=raw.prefetch_factor,
        video_decoder_cache=raw.video_decoder_cache,
        device=raw.device,
        seed=raw.seed,
        eval_samples=raw.eval_samples,
        eval_seed=raw.eval_seed,
        wandb_project=raw.wandb_project,
        wandb_run_name=raw.wandb_run_name,
    )


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Headless rendering for the eval plots regardless of DISPLAY (must be
    # forced before any figure is created; imports stay at the top).
    matplotlib.use("Agg", force=True)
    args = parse_args()
    # Bound each dataloader worker's torchcodec decoder cache. Spawned
    # workers inherit os.environ, and lerobot reads this when it constructs
    # its per-process cache at import time inside the worker. (The main
    # process's own cache was already built with lerobot's default at import
    # time here, but the main process only decodes the small eval set once.)
    if args.video_decoder_cache < 1:
        raise SystemExit("--video-decoder-cache must be >= 1")
    os.environ["LEROBOT_VIDEO_DECODER_CACHE_SIZE"] = str(args.video_decoder_cache)

    # TF32 for fp32 matmuls (torch's default is full-IEEE "highest"): the
    # expert trains in fp32, and true fp32 matmul leaves ~5-7x of H100
    # throughput on the table. TF32's per-op 10-bit mantissa is far above
    # bf16 and standard for training; the bf16 backbone is unaffected.
    torch.set_float32_matmul_precision("high")

    # Data parallelism (torchrun): one full replica + optimizer per rank.
    # Without WORLD_SIZE in the environment this is a plain single-process
    # run — identical behavior to before DDP support existed.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    device = torch.device(args.device)
    rank = 0
    if distributed:
        torch.distributed.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
        )
        rank = torch.distributed.get_rank()
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
    is_main = rank == 0

    # Per-rank RNG stream (τ and ε draws must decorrelate across ranks).
    # Dataloader worker seeds derive from this deterministically: torch
    # draws each worker's base seed from the parent process RNG, so worker
    # seeds are a pure function of (--seed, rank, worker_id). The
    # DistributedSampler below is seeded with the BASE seed on every rank
    # so the shuffled partition stays coordinated.
    torch.manual_seed(args.seed + rank)

    checkpoint_dir = resolve_checkpoint_dir(args.backbone)

    # -- datasets --------------------------------------------------------
    selection = select_datasets(
        args.train_data,
        args.exclude,
        args.chunk_size,
        episode_split=EpisodeSplit.TRAIN,
        holdout_fraction=args.holdout_episodes,
        split_seed=args.split_seed,
        allowed_fps=args.fps,
    )
    action_dim, state_dim = selection.action_dim, selection.state_dim
    per_dataset_stats = selection.per_dataset_stats
    dataset = selection.concat()
    if is_main:
        print(
            f"train data: {len(selection.datasets)} datasets, "
            f"{selection.total_episodes} episodes, {len(dataset)} frames, "
            f"action/state dim {action_dim}/{state_dim}",
            flush=True,
        )
        if args.holdout_episodes > 0:
            print(
                f"episode holdout: {selection.held_out_episodes} episodes "
                f"across {selection.held_out_datasets} datasets excluded "
                f"(fraction {args.holdout_episodes}, split seed "
                f"{args.split_seed})",
                flush=True,
            )
        for camera_set, count in selection.camera_census.most_common():
            print(f"  {count:4d} x cameras {camera_set}", flush=True)
        if selection.dropped:
            print(
                f"dropped {len(selection.dropped)} incompatible datasets:",
                flush=True,
            )
            for reason in selection.dropped:
                print(f"  - {reason}", flush=True)

    # Aggregate stats are NOT used for training math (normalization is per
    # dataset) — they ride along in checkpoints as a fallback for rigs
    # without their own stats.
    normalizers = Normalizers(
        action=Normalizer.from_aggregated_stats(
            list(selection.lerobot_stats.values()),
            "action",
            device,
        ),
        state=Normalizer.from_aggregated_stats(
            list(selection.lerobot_stats.values()),
            "observation.state",
            device,
        ),
    )

    collator = Collator(
        inputs=GemmaInputsCollator(str(checkpoint_dir), args.max_soft_tokens),
        instruction=args.instruction,
        camera_filter=args.cameras,
        max_cameras=args.max_cameras,
    )
    # The explicit generator (both modes) makes the shuffle order and the
    # dataloader worker base-seeds a pure function of (--seed, rank) —
    # otherwise they'd draw from the global RNG and entangle batch order
    # with how much randomness model init happened to consume.
    sampler: torch.utils.data.DistributedSampler[Any] | None = None
    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=torch.Generator().manual_seed(args.seed + rank),
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init if args.num_workers > 0 else None,
        # Spawned (not forked) workers: the parent holds live CUDA state and
        # has decoded video in-process (the eval batch), and torchcodec/ffmpeg
        # deadlock or throw "Could not push packet to decoder" in forked
        # children (verified empirically on the H100 box).
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
        # Pinned batches make DevicePrefetcher's H2D copies truly async; a
        # deeper prefetch queue absorbs the variance of GOP-boundary decodes.
        pin_memory=device.type == "cuda",
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # -- model -----------------------------------------------------------
    expert_config = default_expert_config(
        load_config(checkpoint_dir),
        action_dim=action_dim,
        state_dim=state_dim,
        stream_counts=args.stream_counts,
        hidden_size=args.expert_hidden,
        num_attention_heads=args.expert_heads,
        intermediate_size=args.expert_intermediate,
        cross_attention_heads=args.expert_cross_heads,
        chunk_size=args.chunk_size,
        self_attention_mode=SelfAttentionMode(args.self_attention_mode),
        time_conditioning=TimeConditioning(args.time_conditioning),
    )
    model = from_backbone(
        checkpoint_dir,
        expert_config,
        device=device,
        # A live trunk needs fp32 master weights (bf16 updates at trunk
        # learning rates vanish below bf16 resolution); its forwards run
        # under bf16 autocast in BijouTrainStep. Frozen runs keep the
        # checkpoint dtype (bf16) exactly as before the unfreeze flags.
        dtype=torch.float32 if args.trunk_trained else None,
        expert_dtype=torch.float32,
    )
    trunk_counts = unfreeze_backbone(model, args)
    n_trainable = sum(p.numel() for p in model.expert.parameters())
    if is_main:
        backbone_desc = (
            "frozen backbone"
            if not args.trunk_trained
            else (
                f"LIVE backbone (text {trunk_counts.text / 1e6:.1f}M @ "
                f"lr {args.text_lr if args.text_lr is not None else 0:.1e}, "
                f"vision {trunk_counts.vision / 1e6:.1f}M @ lr "
                f"{args.vision_lr if args.vision_lr is not None else 0:.1e}; "
                "embeddings/PLE tables frozen; fp32 masters, bf16 autocast)"
            )
        )
        print(
            f"model: {backbone_desc} "
            f"({len(model.backbone.language_model.layers)} "
            f"layers, streams {expert_config.streams}) + fp32 expert "
            f"({n_trainable / 1e6:.1f}M params, schedule "
            f"{expert_config.cross_attention_schedule})",
            flush=True,
        )
        if distributed:
            print(
                f"ddp: {world_size} ranks, global batch "
                f"{args.batch_size * world_size} ({args.batch_size}/rank), "
                f"{args.num_workers} dataloader workers/rank",
                flush=True,
            )

    # Fixed-key dicts, deliberately: this is torch's optimizer param-group
    # API format (a third-party boundary), consumed by AdamW below.
    param_groups: list[dict[str, Any]] = [
        {"params": list(model.expert.parameters()), "lr": args.expert_lr},
    ]
    if args.text_lr is not None:
        decayed, undecayed = decay_split(trainable_text_parameters(model))
        param_groups.append({"params": decayed, "lr": args.text_lr})
        param_groups.append(
            {
                "params": undecayed,
                "lr": args.text_lr,
                "weight_decay": 0.0,
            },
        )
    if args.vision_lr is not None:
        assert model.backbone.vision_tower is not None
        decayed, undecayed = decay_split(model.backbone.vision_tower.parameters())
        param_groups.append({"params": decayed, "lr": args.vision_lr})
        param_groups.append(
            {
                "params": undecayed,
                "lr": args.vision_lr,
                "weight_decay": 0.0,
            },
        )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.expert_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        # One kernel launch per param group instead of the foreach chain;
        # CUDA only (CPU runs keep the reference path, which also keeps
        # the CPU loss oracle stable).
        fused=device.type == "cuda",
    )
    # Everything the optimizer updates, for the gradient clip: the frozen
    # path clips exactly the expert (unchanged behavior); a live trunk is
    # clipped jointly with it (one global norm).
    clipped_parameters: list[torch.nn.Parameter] = [
        p for group in param_groups for p in group["params"]
    ]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_lambda(step, args),
    )

    start_step = 0
    checkpoint_to_load = args.init_from or args.resume
    if checkpoint_to_load is not None:
        ensure_matching_expert_config(expert_config, checkpoint_to_load)
        # CPU-load + copy-in: loading straight to the device transiently
        # holds a second copy of the weights next to the built module
        # (see loading.load_adapted_backbone).
        model.expert.load_state_dict(
            load_file(
                str(checkpoint_to_load / "expert.safetensors"),
                device="cpu",
            ),
            strict=True,
        )
        if is_main:
            print(f"loaded expert weights from {checkpoint_to_load}", flush=True)
        # Trunk-trained checkpoints carry the adapted backbone; plain ones
        # don't, and the HF backbone loaded above simply stays (that is the
        # cont45k -> unfreeze continuation path).
        if (checkpoint_to_load / "backbone.safetensors").exists():
            load_adapted_backbone(model, checkpoint_to_load)
            if is_main:
                print(
                    f"loaded ADAPTED backbone weights from "
                    f"{checkpoint_to_load} (bf16 snapshot into "
                    f"{'fp32 masters' if args.trunk_trained else 'bf16'})",
                    flush=True,
                )
    if args.resume is not None:
        optimizer_path = args.resume / "optimizer.pt"
        if not optimizer_path.exists():
            raise SystemExit(
                f"{optimizer_path} missing (checkpoint predates optimizer "
                "saving) — use --init-from for a warm start instead",
            )
        train_state = TrainState.from_payload(
            torch.load(optimizer_path, map_location="cpu", weights_only=True),
        )
        optimizer.load_state_dict(train_state.optimizer)
        scheduler.load_state_dict(train_state.scheduler)
        start_step = train_state.step
        if start_step >= args.steps:
            raise SystemExit(
                f"checkpoint is at step {start_step}, nothing to do with "
                f"--steps {args.steps} (it counts total steps)",
            )
        if is_main:
            print(
                f"resumed optimizer/scheduler at step {start_step} "
                f"(lr {scheduler.get_last_lr()[0]:.2e})",
                flush=True,
            )
        restored = optimizer.param_groups[0]
        base_lr = float(restored.get("initial_lr", restored["lr"]))
        if is_main and (
            base_lr != args.expert_lr
            or float(restored["weight_decay"]) != args.weight_decay
        ):
            print(
                "note: --resume keeps the checkpoint's optimizer "
                f"hyperparameters (base lr {base_lr:.2e}, weight decay "
                f"{restored['weight_decay']}); CLI --expert-lr/--weight-decay "
                "are ignored, --steps/--warmup-steps still shape the schedule",
                flush=True,
            )

    # DDP wiring. Frozen trunk (the default): wrap the expert only — the
    # backbone replica is inference-only and never synced; byte-identical
    # to every run before the unfreeze flags existed. Live trunk: one
    # BijouTrainStep module owns prefix-encode + expert so a single DDP
    # wrapper hooks gradients of both. Wrapping AFTER the weight load means
    # DDP's construction-time broadcast (rank 0 -> all) covers the loaded
    # state. model.expert stays the raw module for eval, clipping and
    # checkpointing in both modes.
    velocity_model: torch.nn.Module = model.expert
    train_step: torch.nn.Module | None = None
    if args.trunk_trained:
        train_step = BijouTrainStep(model)
        if distributed:
            train_step = torch.nn.parallel.DistributedDataParallel(
                train_step,
                device_ids=[device.index] if device.type == "cuda" else None,
                # Backbone/expert buffers are constant RoPE tables etc.;
                # per-step broadcasts would be pure overhead. The
                # trainable-parameter partition guarantees every
                # grad-enabled parameter receives gradients every step
                # (trainable_text_parameters mirrors kv_stop_layer), and
                # the graph never changes.
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
    elif distributed:
        velocity_model = torch.nn.parallel.DistributedDataParallel(
            model.expert,
            device_ids=[device.index] if device.type == "cuda" else None,
            # Expert buffers are constant RoPE tables; per-step broadcasts
            # would be pure overhead.
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    # Fixed MAE probe sets, independent of the training batch size, fetched
    # once and kept as CPU-resident collated batches per rank (collation
    # in-process is safe: dataloader workers are spawned, not forked; GPU
    # memory per eval is bounded by one batch, so probe size costs host RAM
    # only). eval_chunk_mae probes the held-out episodes (needs
    # --holdout-episodes > 0); train_mae probes the training split. Both
    # draws are what bijou.eval --episodes {holdout,train} --seed
    # <eval-seed> would score. Rank 0 keeps raw items strided across its
    # shard for the rich wandb table.
    eval_probe: ProbeSet | None = None
    train_probe: ProbeSet | None = None
    if args.eval_samples is not None:
        if args.holdout_episodes > 0:
            eval_selection = select_datasets(
                args.train_data,
                args.exclude,
                args.chunk_size,
                episode_split=EpisodeSplit.HOLDOUT,
                holdout_fraction=args.holdout_episodes,
                split_seed=args.split_seed,
                allowed_fps=args.fps,
            )
            eval_dataset = eval_selection.concat()
            eval_probe = build_probe_set(
                eval_dataset,
                collator,
                args.eval_samples,
                args.eval_seed,
                rank,
                world_size,
                args.batch_size,
                keep_rich=is_main,
            )
            if is_main:
                print(
                    f"eval probe: {eval_probe.total} of {len(eval_dataset)} "
                    f"held-out-episode frames (seed {args.eval_seed}), "
                    f"sharded over {world_size} rank(s) in batches of "
                    f"{args.batch_size}",
                    flush=True,
                )
            # Only the fetched probe items survive; the second dataset
            # selection (arrow tables, metadata for every dataset) does not
            # need to stay resident for the whole run.
            del eval_selection, eval_dataset
        train_probe = build_probe_set(
            dataset,
            collator,
            args.eval_samples,
            args.eval_seed,
            rank,
            world_size,
            args.batch_size,
            keep_rich=is_main and eval_probe is None,
        )
        if is_main:
            print(
                f"train probe: {train_probe.total} of {len(dataset)} "
                f"training frames (seed {args.eval_seed})",
                flush=True,
            )
    action_names = selection.action_names

    log_file: TextIO | None = None
    if is_main:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        log_file = (args.save_dir / "train_log.jsonl").open("a")

    wandb_run: Any = None
    if args.wandb_project is not None and is_main:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            dir=str(args.save_dir),
            config={
                "train_args": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in dataclasses.asdict(args).items()
                },
                "expert_config": dataclasses.asdict(expert_config),
                "dataset": {
                    "repo_ids": sorted(per_dataset_stats),
                    "episodes": selection.total_episodes,
                    "frames": len(dataset),
                    "camera_sets": {
                        "/".join(k): v for k, v in selection.camera_census.items()
                    },
                },
                "trainable_params": n_trainable,
                "trainable_trunk_params": dataclasses.asdict(trunk_counts),
                "world_size": world_size,
                "global_batch_size": args.batch_size * world_size,
            },
        )

    step = start_step
    # Loss/grad-norm live on-device between log points: a single .item()
    # sync per log_every steps instead of one per step. (grad_norm is
    # identical on all ranks after DDP's gradient sync — no reduce needed.)
    window: list[Tensor] = []
    grad_norm = torch.zeros((), device=device)
    prefetcher = DevicePrefetcher(loader, device)
    epoch = 0
    t_last = time.perf_counter()
    while step < args.steps:
        if sampler is not None:
            # Fresh coordinated shuffle each pass over the data.
            sampler.set_epoch(epoch)
        for batch in prefetcher:
            if step >= args.steps:
                break
            if train_step is not None:
                loss = train_step(batch)
            else:
                prefix = encode_prefix(model, batch)
                loss = flow_matching_loss(velocity_model, prefix, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                clipped_parameters,
                args.grad_clip,
            )
            optimizer.step()
            scheduler.step()
            step += 1
            window.append(loss.detach())

            if step % args.log_every == 0:
                # All ranks participate in the reduce (they hit the same
                # step in lockstep); only rank 0 syncs to host and reports.
                window_mean = torch.stack(window).mean()
                window.clear()
                if distributed:
                    torch.distributed.all_reduce(window_mean)
                    window_mean /= world_size
                dt = (time.perf_counter() - t_last) / args.log_every
                t_last = time.perf_counter()
                if is_main:
                    record = {
                        "step": step,
                        "loss": round(window_mean.item(), 4),
                        "grad_norm": round(grad_norm.item(), 3),
                        "lr": scheduler.get_last_lr()[0],
                        "samples": step * args.batch_size * world_size,
                        "s_per_step": round(dt, 3),
                    }
                    if args.trunk_trained:
                        # Group 1 is the first trunk group (same cosine
                        # shape as the expert's, scaled to its base lr).
                        record["lr_trunk"] = scheduler.get_last_lr()[1]
                    assert log_file is not None
                    print(json.dumps(record), flush=True)
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": record["loss"],
                                "train/grad_norm": record["grad_norm"],
                                "train/lr": record["lr"],
                                "train/samples": record["samples"],
                                "train/s_per_step": record["s_per_step"],
                            },
                            step=step,
                        )

            if train_probe is not None and step % args.eval_every == 0:
                # Collective: every rank scores its shards, the MAE sums
                # all-reduce inside validate (same call order everywhere).
                # Noise decorrelates across ranks via the +rank offset but
                # is identical across eval points, keeping both series
                # comparable step to step.
                probe_record: dict[str, Any] = {"step": step}
                probe_metrics: dict[str, float] = {}
                if eval_probe is not None:
                    eval_mae = validate(
                        model,
                        eval_probe,
                        device,
                        args.eval_seed + rank,
                        distributed=distributed,
                        wandb_run=wandb_run,
                        collator=collator,
                        action_names=action_names,
                        step=step,
                        table_key="eval/samples",
                    )
                    probe_record["eval_chunk_mae"] = round(eval_mae, 4)
                    probe_metrics["eval/chunk_mae"] = eval_mae
                train_mae = validate(
                    model,
                    train_probe,
                    device,
                    args.eval_seed + rank,
                    distributed=distributed,
                    wandb_run=wandb_run,
                    collator=collator,
                    action_names=action_names,
                    step=step,
                    table_key="train/samples",
                )
                probe_record["train_mae"] = round(train_mae, 4)
                probe_metrics["train/mae"] = train_mae
                if is_main:
                    assert log_file is not None
                    print(json.dumps(probe_record), flush=True)
                    log_file.write(json.dumps(probe_record) + "\n")
                    log_file.flush()
                    if wandb_run is not None:
                        wandb_run.log(probe_metrics, step=step)

            if (step % args.save_every == 0 or step == args.steps) and is_main:
                path = save_checkpoint(
                    model,
                    args,
                    normalizers,
                    per_dataset_stats,
                    optimizer,
                    scheduler,
                    step,
                )
                print(f"saved {path}", flush=True)
        epoch += 1

    if log_file is not None:
        log_file.close()
    if wandb_run is not None:
        wandb_run.finish()
    if distributed:
        # Let rank 0 finish its final save/eval before the group dissolves.
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
