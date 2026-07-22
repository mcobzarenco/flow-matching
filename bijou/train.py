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

Usage (dev sample)::

    uv run python -m bijou.train \
        --dataset-root ~/community_dataset_v1_v3/ZGGZZG/so100_drop0 \
        --repo-id ZGGZZG/so100_drop0 --device cuda --steps 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .expert import PrefixKV
from .loading import default_expert_config, from_backbone
from .model import BijouModel

DEFAULT_BACKBONE = "google/gemma-4-e2b-it"


@dataclass(frozen=True, slots=True)
class TrainArgs:
    dataset_root: Path
    repo_id: str
    backbone: str
    save_dir: Path
    instruction: str | None
    cameras: tuple[str, ...] | None
    max_soft_tokens: int
    stream_counts: tuple[int, ...]
    self_attention_mode: str
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
    device: str
    seed: int


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

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / (self.std + 1e-8)

    def unnormalize(self, x: Tensor) -> Tensor:
        return x * (self.std + 1e-8) + self.mean

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class PrefixCollator:
    """Builds batched multimodal prompts from LeRobot items.

    Renders ``[instruction][cameras...][instruction]`` per sample through the
    Gemma4 processor (chat template, right padding). The processor is built
    lazily so the collator can be pickled into dataloader workers.
    """

    def __init__(
        self,
        checkpoint: str,
        cameras: tuple[str, ...],
        instruction: str | None,
        max_soft_tokens: int,
    ) -> None:
        self.checkpoint = checkpoint
        self.cameras = cameras
        self.instruction = instruction
        self.max_soft_tokens = max_soft_tokens
        self._processor: Any = None

    def _to_pil(self, image: Tensor) -> Any:
        from PIL import Image

        array = (image.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(array)

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if self._processor is None:
            import transformers

            self._processor = transformers.AutoProcessor.from_pretrained(
                self.checkpoint
            )
            self._processor.tokenizer.padding_side = "right"

        conversations = []
        for item in items:
            instruction = self.instruction or str(item["task"])
            content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
            for camera in self.cameras:
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
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"],
            "image_position_ids": batch["image_position_ids"],
            "state": torch.stack([item["observation.state"] for item in items]),
            "actions": torch.stack([item["action"] for item in items]),
            "action_is_pad": torch.stack([item["action_is_pad"] for item in items]),
        }


def encode_prefix(
    model: BijouModel, batch: dict[str, Any], device: torch.device
) -> PrefixKV:
    attention_mask = batch["attention_mask"].to(device)
    padding_mask = None if bool(attention_mask.all()) else attention_mask
    with torch.no_grad():
        return model.encode_prefix(
            batch["input_ids"].to(device),
            pixel_values=batch["pixel_values"].to(device),
            image_position_ids=batch["image_position_ids"].to(device),
            padding_mask=padding_mask,
        )


def flow_matching_loss(
    model: BijouModel,
    prefix: PrefixKV,
    batch: dict[str, Any],
    action_normalizer: Normalizer,
    state_normalizer: Normalizer,
    device: torch.device,
) -> Tensor:
    actions = action_normalizer.normalize(batch["actions"].to(device))
    state = state_normalizer.normalize(batch["state"].to(device))
    valid = ~batch["action_is_pad"].to(device)

    noise = torch.randn_like(actions)
    # π0's time distribution: Beta(1.5, 1) squeezed into (0, 1).
    tau = torch.distributions.Beta(1.5, 1.0).sample((actions.shape[0],)).to(device)
    tau = tau * 0.999 + 0.001
    tau_ = tau[:, None, None]
    noisy_actions = tau_ * noise + (1 - tau_) * actions
    target = noise - actions

    velocity = model(prefix, state, noisy_actions, tau)
    mse = (velocity.float() - target.float()).pow(2)
    # valid [B, chunk] indexes the first two dims of mse [B, chunk, dim].
    return mse[valid].mean()


@torch.no_grad()
def evaluate_chunk_mae(
    model: BijouModel,
    prefix: PrefixKV,
    batch: dict[str, Any],
    action_normalizer: Normalizer,
    state_normalizer: Normalizer,
    device: torch.device,
    seed: int,
) -> float:
    """Deterministic sampled-chunk MAE against ground truth, in raw action
    units (the eval-harness metric from the SmolVLA work)."""
    state = state_normalizer.normalize(batch["state"].to(device))
    generator = torch.Generator(device=device).manual_seed(seed)
    sampled = model.sample_actions(prefix, state, num_steps=10, generator=generator)
    sampled = action_normalizer.unnormalize(sampled.float())
    truth = batch["actions"].to(device).float()
    valid = ~batch["action_is_pad"].to(device)
    error = (sampled - truth).abs()
    return float(error[valid].mean())


def save_checkpoint(
    model: BijouModel,
    args: TrainArgs,
    normalizers: dict[str, Normalizer],
    step: int,
) -> Path:
    from safetensors.torch import save_file

    checkpoint_dir = args.save_dir / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.expert.state_dict(), str(checkpoint_dir / "expert.safetensors"))
    metadata = {
        "backbone": args.backbone,
        "expert_config": dataclasses.asdict(model.expert.config),
        "normalization": {k: n.state_dict() for k, n in normalizers.items()},
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
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
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
        help="image feature keys in prompt order (default: all, sorted)",
    )
    parser.add_argument("--max-soft-tokens", type=int, default=140)
    parser.add_argument("--stream-counts", type=int, nargs="*", default=[4, 4, 7])
    parser.add_argument(
        "--self-attention-mode",
        choices=["causal_actions", "bidirectional"],
        default="causal_actions",
    )
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    raw = parser.parse_args()
    return TrainArgs(
        dataset_root=raw.dataset_root.expanduser(),
        repo_id=raw.repo_id,
        backbone=raw.backbone,
        save_dir=raw.save_dir,
        instruction=raw.instruction,
        cameras=tuple(raw.cameras) if raw.cameras else None,
        max_soft_tokens=raw.max_soft_tokens,
        stream_counts=tuple(raw.stream_counts),
        self_attention_mode=raw.self_attention_mode,
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
        device=raw.device,
        seed=raw.seed,
    )


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .expert import SelfAttentionMode
    from .gemma4.loading import load_config, resolve_checkpoint_dir

    checkpoint_dir = resolve_checkpoint_dir(args.backbone)

    # -- dataset ---------------------------------------------------------
    probe = LeRobotDataset(args.repo_id, root=str(args.dataset_root))
    fps = probe.fps
    features = probe.meta.info["features"]
    cameras = args.cameras or tuple(
        sorted(k for k, f in features.items() if f["dtype"] == "video")
    )
    action_dim = int(features["action"]["shape"][0])
    state_dim = int(features["observation.state"]["shape"][0])
    dataset = LeRobotDataset(
        args.repo_id,
        root=str(args.dataset_root),
        delta_timestamps={"action": [i / fps for i in range(args.chunk_size)]},
    )
    print(
        f"dataset: {args.repo_id}: {dataset.num_episodes} episodes, "
        f"{dataset.num_frames} frames, fps {fps}, cameras {cameras}, "
        f"action/state dim {action_dim}/{state_dim}",
        flush=True,
    )
    stats = dataset.meta.stats
    if stats is None:
        raise ValueError(f"dataset {args.repo_id} has no stats (meta/stats.json)")
    action_normalizer = Normalizer.from_stats(stats, "action", device)
    state_normalizer = Normalizer.from_stats(stats, "observation.state", device)

    collator = PrefixCollator(
        str(checkpoint_dir), cameras, args.instruction, args.max_soft_tokens
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        # Keep dataloader workers single-threaded: they fork from a parent
        # with a multi-GB model resident, and N workers x M torch threads
        # oversubscribes the host.
        worker_init_fn=(lambda _worker_id: torch.set_num_threads(1))
        if args.num_workers > 0
        else None,
    )

    # -- model -----------------------------------------------------------
    expert_config = default_expert_config(
        load_config(checkpoint_dir),
        action_dim=action_dim,
        state_dim=state_dim,
        stream_counts=args.stream_counts,
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

    # A fixed batch for the deterministic sampled-chunk eval. Built with an
    # in-process loader: iterating the main (persistent-workers) loader once
    # and abandoning the iterator would leak a full worker pool.
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        drop_last=True,
    )
    eval_batch = next(iter(eval_loader))
    del eval_loader
    eval_prefix = encode_prefix(model, eval_batch, device)
    print(
        f"prefix: {eval_batch['input_ids'].shape[1]} tokens "
        f"(soft-token budget {args.max_soft_tokens}/camera)",
        flush=True,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.save_dir / "train_log.jsonl"
    log_file = log_path.open("a")

    step = 0
    window: list[float] = []
    t_last = time.perf_counter()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            prefix = encode_prefix(model, batch, device)
            loss = flow_matching_loss(
                model, prefix, batch, action_normalizer, state_normalizer, device
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.expert.parameters(), args.grad_clip
            )
            optimizer.step()
            scheduler.step()
            step += 1
            window.append(loss.detach().item())

            if step % args.log_every == 0:
                dt = (time.perf_counter() - t_last) / args.log_every
                t_last = time.perf_counter()
                record = {
                    "step": step,
                    "loss": round(statistics.mean(window), 4),
                    "grad_norm": round(float(grad_norm), 3),
                    "lr": scheduler.get_last_lr()[0],
                    "s_per_step": round(dt, 3),
                }
                window.clear()
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

            if step % args.eval_every == 0:
                mae = evaluate_chunk_mae(
                    model,
                    eval_prefix,
                    eval_batch,
                    action_normalizer,
                    state_normalizer,
                    device,
                    args.seed,
                )
                record = {"step": step, "eval_chunk_mae": round(mae, 4)}
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

            if step % args.save_every == 0 or step == args.steps:
                path = save_checkpoint(
                    model,
                    args,
                    {
                        "action": action_normalizer,
                        "observation.state": state_normalizer,
                    },
                    step,
                )
                print(f"saved {path}", flush=True)

    log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
