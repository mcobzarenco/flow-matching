"""Train Bijou (flow-matching action expert) on a LeRobot v3 dataset.

The frozen truncated backbone encodes the multimodal prefix per batch
(no grad); the expert is optimized with the π0/SmolVLA flow-matching recipe:
τ ~ Beta(1.5, 1) (scaled into (0, 1)), x_τ = τ·ε + (1−τ)·actions, MSE against
the velocity target ε − actions, with episode-boundary action padding masked
out. Actions and state are MEAN_STD-normalized from the dataset stats; the
stats are saved into every checkpoint (inference must unnormalize with the
same stats).

The prompt is the instruction sandwich discussed in the design:
``[instruction][cam_1]...[cam_N][instruction]`` inside a user chat turn,
giving instruction-conditioned image KV and image-conditioned instruction KV
under causal attention.

Training data is selected with ``--train-data``: any mix of dataset
directories and collection roots (scanned for ``*/meta/info.json`` and
``*/*/meta/info.json``, i.e. flat and ``<user>/<dataset>`` layouts). Datasets
whose action/state dims differ from the first selected dataset are dropped
loudly. Camera keys are discovered per sample (sorted, so prompt slots are
positional — the community collections' generic image/image2 keys carry no
reliable wrist-vs-scene semantics); normalization uses count-weighted
aggregated stats over the selection.

Usage::

    uv run python -m bijou.train \
        --train-data ~/datasets/mcobzarenco/community_dataset_v2_v3 \
        --device cuda --steps 5000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fnmatch import fnmatch
from itertools import islice
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import torch
import transformers
import wandb
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from safetensors.torch import save_file
from torch import Tensor

from .expert import PrefixKV, SelfAttentionMode
from .gemma4.loading import load_config, resolve_checkpoint_dir
from .loading import default_expert_config, from_backbone
from .model import BijouModel

DEFAULT_BACKBONE = "google/gemma-4-e2b-it"


@dataclass(frozen=True, slots=True)
class TrainArgs:
    train_data: tuple[Path, ...]
    exclude: tuple[str, ...]
    backbone: str
    save_dir: Path
    instruction: str | None
    cameras: tuple[str, ...] | None
    max_cameras: int | None
    max_soft_tokens: int
    stream_counts: tuple[int, ...]
    self_attention_mode: str
    expert_hidden: int
    expert_heads: int
    expert_intermediate: int
    expert_cross_heads: int
    chunk_size: int
    batch_size: int
    steps: int
    lr: float
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
    eval_samples: int
    wandb_project: str | None
    wandb_run_name: str | None


@dataclass(frozen=True, slots=True)
class CollatedBatch:
    """One collated batch: prefix inputs plus action-chunk targets. All
    fields are always present (unlike the raw per-dataset LeRobot items,
    whose camera keys vary — those stay dicts).

    ``has_padding`` is computed CPU-side in the dataloader workers so the
    training loop never needs a device->host sync to decide whether to build
    padding masks.
    """

    input_ids: Tensor
    attention_mask: Tensor
    pixel_values: Tensor
    image_position_ids: Tensor
    state: Tensor
    actions: Tensor
    action_is_pad: Tensor
    has_padding: bool

    def tensors(self) -> dict[str, Tensor]:
        return {
            f.name: value
            for f in dataclasses.fields(self)
            if isinstance(value := getattr(self, f.name), Tensor)
        }

    def pin_memory(self) -> "CollatedBatch":
        """Called by the DataLoader when ``pin_memory=True`` (torch supports
        custom batch types via this hook); pinned memory makes the H2D
        copies in :class:`DevicePrefetcher` truly asynchronous."""
        return dataclasses.replace(
            self, **{name: t.pin_memory() for name, t in self.tensors().items()}
        )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "CollatedBatch":
        return dataclasses.replace(
            self,
            **{
                name: t.to(device, non_blocking=non_blocking)
                for name, t in self.tensors().items()
            },
        )


class DevicePrefetcher:
    """One-batch-lookahead host->device transfer on a side CUDA stream.

    Centralizes every H2D copy of the training pipeline: the loop receives
    batches already device-resident, and the copy of batch N+1 overlaps the
    compute of batch N (dataloader workers cannot produce CUDA tensors —
    each would need its own CUDA context — so this is the closest torch gets
    to "the loader hands you device batches"). Degrades to plain synchronous
    transfers on non-CUDA devices.
    """

    def __init__(self, loader: Iterable[CollatedBatch], device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def __iter__(self) -> Iterator[CollatedBatch]:
        if self.device.type != "cuda":
            for batch in self.loader:
                yield batch.to(self.device)
            return

        stream = torch.cuda.Stream(self.device)
        compute_stream = torch.cuda.current_stream(self.device)
        batches = iter(self.loader)

        def preload() -> CollatedBatch | None:
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
            for tensor in batch.tensors().values():
                tensor.record_stream(compute_stream)
            yield batch  # consumer enqueues the step's compute, then returns
            batch = preload()  # blocks on workers while the GPU crunches


class Normalizer:
    """MEAN_STD normalization from LeRobot dataset stats."""

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean
        self.std = std

    @classmethod
    def from_stats(
        cls, stats: dict[str, dict[str, Any]], key: str, device: torch.device
    ) -> "Normalizer":
        mean = torch.as_tensor(stats[key]["mean"], dtype=torch.float32, device=device)
        std = torch.as_tensor(stats[key]["std"], dtype=torch.float32, device=device)
        return cls(mean, std)

    @classmethod
    def from_aggregated_stats(
        cls,
        stats_list: list[dict[str, dict[str, Any]]],
        key: str,
        device: torch.device,
    ) -> "Normalizer":
        """Count-weighted aggregation across datasets: the exact combined
        mean, and std via E[x²] composition (all in float64 before rounding
        to float32)."""
        counts = torch.tensor(
            [float(s[key]["count"][0]) for s in stats_list], dtype=torch.float64
        )
        means = torch.stack(
            [torch.as_tensor(s[key]["mean"], dtype=torch.float64) for s in stats_list]
        )
        stds = torch.stack(
            [torch.as_tensor(s[key]["std"], dtype=torch.float64) for s in stats_list]
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


def repo_id_of(dataset_dir: Path) -> str:
    return f"{dataset_dir.parent.name}/{dataset_dir.name}"


def discover_datasets(paths: tuple[Path, ...], exclude: tuple[str, ...]) -> list[Path]:
    """Resolve ``--train-data`` entries to dataset directories.

    A path containing ``meta/info.json`` is a dataset; anything else is
    treated as a collection root and scanned one and two levels deep (flat
    and ``<user>/<dataset>`` layouts). ``exclude`` patterns are fnmatch'd
    against the derived ``<user>/<dataset>`` repo id.
    """
    found: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if (path / "meta" / "info.json").exists():
            found.append(path)
            continue
        nested = sorted(
            info.parent.parent
            for pattern in ("*/meta/info.json", "*/*/meta/info.json")
            for info in path.glob(pattern)
        )
        if not nested:
            raise FileNotFoundError(f"no LeRobot datasets under {path}")
        found.extend(nested)

    selected: list[Path] = []
    seen: set[Path] = set()
    for dataset_dir in found:
        if dataset_dir in seen:
            continue
        seen.add(dataset_dir)
        if any(fnmatch(repo_id_of(dataset_dir), pattern) for pattern in exclude):
            continue
        selected.append(dataset_dir)
    if not selected:
        raise FileNotFoundError("no datasets left after --exclude filtering")
    return selected


def _worker_init(_worker_id: int) -> None:
    # Keep dataloader workers single-threaded: N workers x M torch threads
    # oversubscribes the host.
    torch.set_num_threads(1)


class PrefixCollator:
    """Builds batched multimodal prompts from LeRobot items.

    Renders ``[instruction][cameras...][instruction]`` per sample through the
    Gemma4 processor (chat template, right padding). The processor is built
    lazily so the collator can be pickled into dataloader workers.
    """

    def __init__(
        self,
        checkpoint: str,
        instruction: str | None,
        max_soft_tokens: int,
        camera_filter: tuple[str, ...] | None = None,
        max_cameras: int | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.instruction = instruction
        self.max_soft_tokens = max_soft_tokens
        self.camera_filter = camera_filter
        self.max_cameras = max_cameras
        self._processor: Any = None

    def cameras_of(self, item: dict[str, Any]) -> list[str]:
        """Sorted camera keys of one sample; prompt slots are positional (the
        community collections' image/image2 keys carry no reliable
        wrist-vs-scene semantics — SmolVLA precedent)."""
        cameras = sorted(k for k in item if k.startswith("observation.images."))
        if self.camera_filter is not None:
            allowed = set(self.camera_filter)
            cameras = [
                k
                for k in cameras
                if k in allowed or k.removeprefix("observation.images.") in allowed
            ]
        if not cameras:
            raise ValueError(
                f"sample has no cameras after filtering ({self.camera_filter=})"
            )
        if self.max_cameras is not None:
            cameras = cameras[: self.max_cameras]
        return cameras

    def __getstate__(self) -> dict[str, Any]:
        # Never ship a constructed processor across process boundaries; spawn
        # workers rebuild it lazily.
        return {**self.__dict__, "_processor": None}

    def _to_pil(self, image: Tensor) -> Image.Image:
        array = (image.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(array)

    def __call__(self, items: list[dict[str, Any]]) -> CollatedBatch:
        if self._processor is None:
            # Lazy construction (not import): the collator is pickled into
            # spawned dataloader workers, each of which rebuilds it.
            self._processor = transformers.AutoProcessor.from_pretrained(
                self.checkpoint
            )
            self._processor.tokenizer.padding_side = "right"

        conversations = []
        for item in items:
            instruction = self.instruction or str(item["task"])
            content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
            for camera in self.cameras_of(item):
                content.append({"type": "image", "image": self._to_pil(item[camera])})
            content.append({"type": "text", "text": instruction})
            conversations.append([{"role": "user", "content": content}])

        batch = self._processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # transformers 5.14: per-call processor kwargs must be nested, and
            # a flat `padding=True` alongside `processor_kwargs` silently
            # drops the latter -- both go inside (verified empirically).
            processor_kwargs={
                "max_soft_tokens": self.max_soft_tokens,
                "padding": True,
            },
        )
        return CollatedBatch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_position_ids=batch["image_position_ids"],
            state=torch.stack([item["observation.state"] for item in items]),
            actions=torch.stack([item["action"] for item in items]),
            action_is_pad=torch.stack([item["action_is_pad"] for item in items]),
            # Decided here (CPU, in the worker) so the train loop never syncs.
            has_padding=bool((batch["attention_mask"] == 0).any()),
        )


@dataclass(frozen=True, slots=True)
class Normalizers:
    action: Normalizer
    state: Normalizer


def encode_prefix(model: BijouModel, batch: CollatedBatch) -> PrefixKV:
    """``batch`` must already be device-resident (see DevicePrefetcher)."""
    padding_mask = batch.attention_mask if batch.has_padding else None
    with torch.no_grad():
        return model.encode_prefix(
            batch.input_ids,
            pixel_values=batch.pixel_values,
            image_position_ids=batch.image_position_ids,
            padding_mask=padding_mask,
        )


def flow_matching_loss(
    model: BijouModel,
    prefix: PrefixKV,
    batch: CollatedBatch,
    normalizers: Normalizers,
) -> Tensor:
    """``batch`` must already be device-resident; no transfers happen here."""
    actions = normalizers.action.normalize(batch.actions)
    state = normalizers.state.normalize(batch.state)
    valid = ~batch.action_is_pad

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

    velocity = model(prefix, state, noisy_actions, tau)
    mse = (velocity.float() - target.float()).pow(2)
    # valid [B, chunk] indexes the first two dims of mse [B, chunk, dim].
    return mse[valid].mean()


def _chunk_plot(
    predicted: Tensor, truth: Tensor, valid: Tensor, action_names: list[str]
) -> Any:
    """Per-joint predicted-vs-ground-truth curves over the action chunk.
    Returns a matplotlib figure (caller logs and closes it)."""
    dims = predicted.shape[-1]
    ncols = 3
    nrows = (dims + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False
    )
    steps = range(predicted.shape[0])
    n_valid = int(valid.sum())
    for dim in range(dims):
        ax = axes[dim // ncols][dim % ncols]
        ax.plot(steps[:n_valid], truth[:n_valid, dim].tolist(), label="truth")
        ax.plot(
            steps[:n_valid],
            predicted[:n_valid, dim].tolist(),
            label="predicted",
            linestyle="--",
        )
        name = action_names[dim] if dim < len(action_names) else f"dim {dim}"
        ax.set_title(name, fontsize=9)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    return fig


@torch.no_grad()
def validate(
    model: BijouModel,
    prefix: PrefixKV,
    batch: CollatedBatch,
    normalizers: Normalizers,
    seed: int,
    *,
    wandb_run: Any = None,
    eval_items: list[dict[str, Any]] | None = None,
    collator: PrefixCollator | None = None,
    action_names: list[str] | None = None,
    step: int = 0,
) -> float:
    """Deterministic sampled-chunk MAE in raw action units (the eval-harness
    metric from the SmolVLA work), always computed and returned; wandb is
    additive only — with a run, also logs a table over the eval samples:
    camera images, task, state, and per-joint predicted-vs-truth plots.
    ``batch`` must already be device-resident."""
    state = normalizers.state.normalize(batch.state)
    generator = torch.Generator(device=state.device).manual_seed(seed)
    sampled = model.sample_actions(prefix, state, num_steps=10, generator=generator)
    sampled = normalizers.action.unnormalize(sampled.float())
    truth = batch.actions.float()
    valid = ~batch.action_is_pad
    error = (sampled - truth).abs()
    mae = float(error[valid].mean())

    if wandb_run is not None and eval_items and collator is not None:
        # Cameras vary per sample across mixed datasets: generic positional
        # columns, padded with None where a sample has fewer cameras.
        per_item_cameras = [collator.cameras_of(item) for item in eval_items]
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
        for i, item in enumerate(eval_items):
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
                sampled[i].cpu(), truth[i].cpu(), valid[i].cpu(), action_names or []
            )
            state_str = ", ".join(
                f"{x:.1f}" for x in item["observation.state"].tolist()
            )
            table.add_data(
                i,
                *images,
                str(item["task"]),
                state_str,
                float(error[i][valid[i]].mean()),
                wandb.Image(figure),
            )
            plt.close(figure)
        wandb_run.log({"eval/samples": table}, step=step)
    return mae


def save_checkpoint(
    model: BijouModel,
    args: TrainArgs,
    normalizers: Normalizers,
    step: int,
) -> Path:
    checkpoint_dir = args.save_dir / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.expert.state_dict(), str(checkpoint_dir / "expert.safetensors"))
    metadata = {
        "backbone": args.backbone,
        "expert_config": dataclasses.asdict(model.expert.config),
        # Keys match the dataset feature names (stable checkpoint format).
        "normalization": {
            "action": normalizers.action.state_dict(),
            "observation.state": normalizers.state.state_dict(),
        },
        "train_args": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in dataclasses.asdict(args).items()
        },
        "step": step,
    }
    (checkpoint_dir / "bijou_config.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    return checkpoint_dir


def lr_lambda(step: int, args: TrainArgs) -> float:
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument(
        "--save-dir", type=Path, default=Path("outputs/train/bijou_dev")
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
        help="only use these camera keys when present (full keys or suffixes; "
        "default: all cameras of each sample, sorted)",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=None,
        help="cap cameras per sample (applied after --cameras filtering)",
    )
    parser.add_argument("--max-soft-tokens", type=int, default=140)
    parser.add_argument("--stream-counts", type=int, nargs="*", default=[4, 4, 7])
    parser.add_argument(
        "--self-attention-mode",
        choices=["causal_actions", "bidirectional"],
        default="causal_actions",
    )
    parser.add_argument("--expert-hidden", type=int, default=768)
    parser.add_argument("--expert-heads", type=int, default=6)
    parser.add_argument("--expert-intermediate", type=int, default=3072)
    parser.add_argument("--expert-cross-heads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
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
        help="max open torchcodec decoders cached per dataloader worker "
        "(exported as LEROBOT_VIDEO_DECODER_CACHE_SIZE). lerobot's default "
        "of 100 pins ~50-100MB of ffmpeg buffers per entry; with shuffled "
        "sampling over hundreds of video files that fills every worker's "
        "cache and OOM-killed a 20-worker run at ~190GB host RAM",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=8,
        help="validation set size: samples spread evenly across the dataset; "
        "chunk MAE is reported on these (and, with wandb, per-sample rich "
        "logs: camera frames, task, state, predicted-vs-truth action plots)",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="enable Weights & Biases logging to this project "
        "(WANDB_API_KEY must be set)",
    )
    parser.add_argument("--wandb-run-name", default=None)
    raw = parser.parse_args()
    return TrainArgs(
        train_data=tuple(raw.train_data),
        exclude=tuple(raw.exclude),
        backbone=raw.backbone,
        save_dir=raw.save_dir,
        instruction=raw.instruction,
        cameras=tuple(raw.cameras) if raw.cameras else None,
        max_cameras=raw.max_cameras,
        max_soft_tokens=raw.max_soft_tokens,
        stream_counts=tuple(raw.stream_counts),
        self_attention_mode=raw.self_attention_mode,
        expert_hidden=raw.expert_hidden,
        expert_heads=raw.expert_heads,
        expert_intermediate=raw.expert_intermediate,
        expert_cross_heads=raw.expert_cross_heads,
        chunk_size=raw.chunk_size,
        batch_size=raw.batch_size,
        steps=raw.steps,
        lr=raw.lr,
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
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    checkpoint_dir = resolve_checkpoint_dir(args.backbone)

    # -- datasets --------------------------------------------------------
    dataset_dirs = discover_datasets(args.train_data, args.exclude)
    dataset_infos = [
        json.loads((d / "meta" / "info.json").read_text()) for d in dataset_dirs
    ]
    # Dims are dictated by the first selected dataset; mismatches are
    # dropped loudly (the community collections mix in a few 7/12/14-dof
    # datasets — cross-embodiment padding is out of scope for now).
    action_dim = int(dataset_infos[0]["features"]["action"]["shape"][0])
    state_dim = int(dataset_infos[0]["features"]["observation.state"]["shape"][0])
    datasets: list[LeRobotDataset] = []
    stats_list: list[dict[str, Any]] = []
    camera_census: Counter[tuple[str, ...]] = Counter()
    dropped: list[str] = []
    total_episodes = 0
    for dataset_dir, info in zip(dataset_dirs, dataset_infos, strict=True):
        repo_id = repo_id_of(dataset_dir)
        dims = (
            int(info["features"]["action"]["shape"][0]),
            int(info["features"]["observation.state"]["shape"][0]),
        )
        if dims != (action_dim, state_dim):
            dropped.append(f"{repo_id} (action/state dims {dims[0]}/{dims[1]})")
            continue

        sub_dataset = LeRobotDataset(
            repo_id,
            root=str(dataset_dir),
            delta_timestamps={
                "action": [i / info["fps"] for i in range(args.chunk_size)]
            },
        )
        if sub_dataset.meta.stats is None:
            dropped.append(f"{repo_id} (no stats)")
            continue

        # Some community datasets ship metadata claiming more frames than
        # their parquet actually holds; ConcatDataset sizes by len()

        actual_rows = len(sub_dataset.hf_dataset)
        if len(sub_dataset) != actual_rows:
            dropped.append(
                f"{repo_id} (metadata claims {len(sub_dataset)} frames, "
                f"parquet holds {actual_rows})"
            )
            continue
        datasets.append(sub_dataset)
        stats_list.append(sub_dataset.meta.stats)
        camera_census[
            tuple(
                sorted(
                    k.removeprefix("observation.images.")
                    for k, f in info["features"].items()
                    if f["dtype"] == "video"
                )
            )
        ] += 1
        total_episodes += sub_dataset.num_episodes
    if not datasets:
        raise ValueError("no compatible datasets selected")
    dataset: torch.utils.data.ConcatDataset[dict[str, Any]] = (
        torch.utils.data.ConcatDataset(datasets)
    )
    print(
        f"train data: {len(datasets)} datasets, {total_episodes} episodes, "
        f"{len(dataset)} frames, action/state dim {action_dim}/{state_dim}",
        flush=True,
    )
    for camera_set, count in camera_census.most_common():
        print(f"  {count:4d} x cameras {camera_set}", flush=True)
    if dropped:
        print(f"dropped {len(dropped)} incompatible datasets:", flush=True)
        for reason in dropped:
            print(f"  - {reason}", flush=True)

    normalizers = Normalizers(
        action=Normalizer.from_aggregated_stats(stats_list, "action", device),
        state=Normalizer.from_aggregated_stats(stats_list, "observation.state", device),
    )

    collator = PrefixCollator(
        str(checkpoint_dir),
        args.instruction,
        args.max_soft_tokens,
        camera_filter=args.cameras,
        max_cameras=args.max_cameras,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_init if args.num_workers > 0 else None,
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
    )
    model = from_backbone(
        checkpoint_dir,
        expert_config,
        device=device,
        expert_dtype=torch.float32,
    )
    n_trainable = sum(p.numel() for p in model.expert.parameters())
    print(
        f"model: frozen backbone ({len(model.backbone.language_model.layers)} "
        f"layers, streams {expert_config.streams}) + fp32 expert "
        f"({n_trainable / 1e6:.1f}M params, schedule "
        f"{expert_config.cross_attention_schedule})",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.expert.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, args)
    )

    # Fixed validation set, independent of the training batch size:
    # --eval-samples items spread evenly across the dataset (deterministic),
    # collated in-process (safe: dataloader workers are spawned, not forked)
    # and prefix-encoded once. The raw items keep the original camera frames
    # for rich logging.
    stride = max(len(dataset) // args.eval_samples, 1)
    eval_indices = list(islice(range(0, len(dataset), stride), args.eval_samples))
    eval_items = [dataset[i] for i in eval_indices]
    eval_batch = collator(eval_items).to(device)
    eval_prefix = encode_prefix(model, eval_batch)
    print(
        f"eval set: {len(eval_indices)} samples at dataset indices "
        f"{eval_indices}; prefix {eval_batch.input_ids.shape[1]} tokens "
        f"(soft-token budget {args.max_soft_tokens}/camera)",
        flush=True,
    )
    action_names = list(dataset_infos[0]["features"]["action"].get("names") or [])

    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.save_dir / "train_log.jsonl"
    log_file = log_path.open("a")

    wandb_run: Any = None
    if args.wandb_project is not None:
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
                    "repo_ids": [d.repo_id for d in datasets],
                    "episodes": total_episodes,
                    "frames": len(dataset),
                    "camera_sets": {"/".join(k): v for k, v in camera_census.items()},
                },
                "trainable_params": n_trainable,
            },
        )

    step = 0
    # Loss/grad-norm live on-device between log points: a single .item()
    # sync per log_every steps instead of one per step.
    window: list[Tensor] = []
    grad_norm = torch.zeros((), device=device)
    prefetcher = DevicePrefetcher(loader, device)
    t_last = time.perf_counter()
    while step < args.steps:
        for batch in prefetcher:
            if step >= args.steps:
                break
            prefix = encode_prefix(model, batch)
            loss = flow_matching_loss(model, prefix, batch, normalizers)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.expert.parameters(), args.grad_clip
            )
            optimizer.step()
            scheduler.step()
            step += 1
            window.append(loss.detach())

            if step % args.log_every == 0:
                dt = (time.perf_counter() - t_last) / args.log_every
                record = {
                    "step": step,
                    "loss": round(torch.stack(window).mean().item(), 4),
                    "grad_norm": round(grad_norm.item(), 3),
                    "lr": scheduler.get_last_lr()[0],
                    "s_per_step": round(dt, 3),
                }
                t_last = time.perf_counter()
                window.clear()
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": record["loss"],
                            "train/grad_norm": record["grad_norm"],
                            "train/lr": record["lr"],
                            "train/s_per_step": record["s_per_step"],
                        },
                        step=step,
                    )

            if step % args.eval_every == 0:
                mae = validate(
                    model,
                    eval_prefix,
                    eval_batch,
                    normalizers,
                    args.seed,
                    wandb_run=wandb_run,
                    eval_items=eval_items,
                    collator=collator,
                    action_names=action_names,
                    step=step,
                )
                record = {"step": step, "eval_chunk_mae": round(mae, 4)}
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                if wandb_run is not None:
                    wandb_run.log({"eval/chunk_mae": mae}, step=step)

            if step % args.save_every == 0 or step == args.steps:
                path = save_checkpoint(model, args, normalizers, step)
                print(f"saved {path}", flush=True)

    log_file.close()
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
