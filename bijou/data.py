"""Dataset selection, per-dataset stats, and prompt collation for Bijou.

Shared by training (``bijou.train``) and evaluation (``bijou.eval``) so both
provably select and prepare data the same way:

- ``discover_datasets``: resolve paths/collection roots (flat and
  ``<user>/<dataset>`` layouts), fnmatch excludes, path dedup.
- ``select_datasets``: the guard pipeline — dims anchored by the first
  dataset with standard features; loud drops for bespoke features, dim
  mismatches, missing/non-finite stats, metadata-vs-parquet frame count
  disagreements and cross-root duplicate repo ids.
- ``DatasetStats``/``StatsAttachedDataset``: per-dataset MEAN_STD stats
  attached to every item (per-dataset normalization) with loud, bounded
  substitution of unfetchable samples.
- ``CollatedBatch``/``PrefixCollator``: chat-templated multimodal prompt
  batches ([instruction][cameras...][instruction]) ready for the backbone.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import torch
import transformers
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from torch import Tensor

from .expert import PrefixKV
from .model import BijouModel


@dataclass(frozen=True, slots=True)
class CollatedBatch:
    """One collated batch: prefix inputs plus action-chunk targets. All
    fields are always present (unlike the raw per-dataset LeRobot items,
    whose camera keys vary — those stay dicts).

    ``state``/``actions`` are raw (unnormalized); the per-sample MEAN_STD
    stats of each sample's own dataset ride along ([B, dim] each) so the
    loss and eval normalize per dataset.

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
    action_mean: Tensor
    action_std: Tensor
    state_mean: Tensor
    state_std: Tensor
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
        copies in DevicePrefetcher truly asynchronous."""
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


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """One dataset's MEAN_STD stats for action and state.

    Plain float tuples, deliberately not tensors: the dataset objects are
    pickled into every spawned dataloader worker, and torch shares pickled
    CPU tensors through shared-memory file descriptors — 4 tensors x 300+
    datasets exhausts the default ulimit (observed: EMFILE on worker spawn).
    """

    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    state_mean: tuple[float, ...]
    state_std: tuple[float, ...]

    @classmethod
    def from_lerobot_stats(cls, stats: dict[str, dict[str, Any]]) -> "DatasetStats":
        def as_vector(
            key: str, field: str, floor: float | None = None
        ) -> tuple[float, ...]:
            values = torch.as_tensor(stats[key][field], dtype=torch.float32)
            if floor is not None:
                # Floor the stds: a (near-)constant joint would otherwise
                # amplify float rounding jitter ~1e4x into the normalized
                # targets. At the floor, deviations from the dataset mean
                # pass through ~unscaled.
                values = values.clamp(min=floor)
            return tuple(values.reshape(-1).tolist())

        return cls(
            action_mean=as_vector("action", "mean"),
            action_std=as_vector("action", "std", floor=1e-2),
            state_mean=as_vector("observation.state", "mean"),
            state_std=as_vector("observation.state", "std", floor=1e-2),
        )

    def is_finite(self) -> bool:
        return all(
            math.isfinite(x)
            for vector in (
                self.action_mean,
                self.action_std,
                self.state_mean,
                self.state_std,
            )
            for x in vector
        )

    def state_dict(self) -> dict[str, dict[str, list[float]]]:
        return {
            "action": {
                "mean": list(self.action_mean),
                "std": list(self.action_std),
            },
            "observation.state": {
                "mean": list(self.state_mean),
                "std": list(self.state_std),
            },
        }


class StatsAttachedDataset(torch.utils.data.Dataset[dict[str, Any]]):
    """Wraps one LeRobot dataset so every item carries its dataset's stats
    (per-dataset normalization: between-rig calibration offsets must not
    survive into the training targets) and its ``repo_id``. Tensors are
    materialized per item, in the worker — see the DatasetStats docstring.

    Unfetchable items (e.g. a corrupt video packet — killed two multi-hour
    runs) are substituted with a far-away index from the SAME dataset,
    loudly: the jump escapes the corrupt GOP/file, per-dataset stats stay
    correct, and batch shapes are unaffected. Bounded retries keep systemic
    breakage (a wholly unreadable dataset) fatal rather than silent."""

    # Large prime jump: far enough to land in a different episode/video
    # file; attempts bound the walk if corruption spans multiple regions.
    _RETRY_STRIDE = 9973
    _MAX_ATTEMPTS = 5

    def __init__(self, dataset: LeRobotDataset, stats: DatasetStats) -> None:
        self.dataset = dataset
        self.stats = stats
        self.failed_fetches = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._fetch_with_substitution(index, self._MAX_ATTEMPTS)
        item["repo_id"] = self.dataset.repo_id
        item["action_mean"] = torch.tensor(self.stats.action_mean)
        item["action_std"] = torch.tensor(self.stats.action_std)
        item["state_mean"] = torch.tensor(self.stats.state_mean)
        item["state_std"] = torch.tensor(self.stats.state_std)
        return item

    def _fetch_with_substitution(self, index: int, attempts: int) -> dict[str, Any]:
        try:
            return self.dataset[index]
        except Exception as error:  # noqa: BLE001 - any corrupt-sample failure
            if attempts <= 1:
                raise
            self.failed_fetches += 1
            substitute = (index + self._RETRY_STRIDE) % len(self.dataset)
            print(
                f"[data] {self.dataset.repo_id}[{index}] unfetchable "
                f"({type(error).__name__}: {error}); substituting index "
                f"{substitute} (failure #{self.failed_fetches} in this "
                "process)",
                file=sys.stderr,
                flush=True,
            )
            return self._fetch_with_substitution(substitute, attempts - 1)


def repo_id_of(dataset_dir: Path) -> str:
    return f"{dataset_dir.parent.name}/{dataset_dir.name}"


def discover_datasets(paths: tuple[Path, ...], exclude: tuple[str, ...]) -> list[Path]:
    """Resolve data-path entries to dataset directories.

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


def action_state_dims(info: dict[str, Any]) -> tuple[int, int] | None:
    """Action/state dims from a dataset's info.json, or None when either
    feature is absent (a few community datasets use bespoke feature names,
    e.g. arm_action/hand_action/observation.arm_state — not trainable here)."""
    features = info.get("features") or {}
    if "action" not in features or "observation.state" not in features:
        return None
    return (
        int(features["action"]["shape"][0]),
        int(features["observation.state"]["shape"][0]),
    )


@dataclass(frozen=True, slots=True)
class DataSelection:
    """Outcome of the selection guard pipeline over discovered datasets."""

    datasets: list[StatsAttachedDataset]
    per_dataset_stats: dict[str, DatasetStats]
    lerobot_stats: dict[str, dict[str, Any]]
    camera_census: Counter[tuple[str, ...]]
    dropped: list[str]
    action_dim: int
    state_dim: int
    action_names: list[str]
    total_episodes: int

    def concat(self) -> torch.utils.data.ConcatDataset[dict[str, Any]]:
        return torch.utils.data.ConcatDataset(self.datasets)


def select_datasets(
    paths: tuple[Path, ...], exclude: tuple[str, ...], chunk_size: int
) -> DataSelection:
    """Discover, validate and wrap datasets; drop the incompatible loudly.

    Dims are dictated by the first discovered dataset that declares the
    standard action/observation.state features (the community collections
    mix in a few 7/12/14-dof and bespoke-feature datasets — cross-embodiment
    padding is out of scope for now).
    """
    dataset_dirs = discover_datasets(paths, exclude)
    dataset_infos = [
        json.loads((d / "meta" / "info.json").read_text()) for d in dataset_dirs
    ]
    anchor_info = next(
        (info for info in dataset_infos if action_state_dims(info) is not None),
        None,
    )
    if anchor_info is None:
        raise ValueError(
            "no selected dataset declares action/observation.state features"
        )
    anchor_dims = action_state_dims(anchor_info)
    assert anchor_dims is not None
    action_dim, state_dim = anchor_dims

    datasets: list[StatsAttachedDataset] = []
    per_dataset_stats: dict[str, DatasetStats] = {}
    lerobot_stats: dict[str, dict[str, Any]] = {}
    camera_census: Counter[tuple[str, ...]] = Counter()
    dropped: list[str] = []
    total_episodes = 0
    selected_dirs: dict[str, Path] = {}
    for dataset_dir, info in zip(dataset_dirs, dataset_infos, strict=True):
        repo_id = repo_id_of(dataset_dir)
        # The same repo id can appear under multiple collection roots (e.g.
        # v1 and v2 of the community collections share datasets): training
        # it twice would double-weight it and clash in the stats table.
        # First root in --train-data order wins, dropped loudly.
        if repo_id in selected_dirs:
            dropped.append(
                f"{repo_id} (duplicate at {dataset_dir}; "
                f"keeping {selected_dirs[repo_id]})"
            )
            continue
        dims = action_state_dims(info)
        if dims is None:
            dropped.append(f"{repo_id} (no action/observation.state features)")
            continue
        if dims != (action_dim, state_dim):
            dropped.append(f"{repo_id} (action/state dims {dims[0]}/{dims[1]})")
            continue

        sub_dataset = LeRobotDataset(
            repo_id,
            root=str(dataset_dir),
            delta_timestamps={"action": [i / info["fps"] for i in range(chunk_size)]},
            # Nearest-frame decode tolerance. lerobot's 1e-4 default is
            # unrepresentable deep into v3-format concatenated video files:
            # torchcodec returns fp32 pts, whose resolution at e.g. 1140s
            # (~1.4e-4) exceeds it, so a CORRECT nearest frame gets rejected
            # (observed: kaiserbuffle/hanoi_dc, 19-minute file). Half a
            # frame period is the exact nearest-frame criterion and still
            # catches genuine desync (off by >= a full frame).
            tolerance_s=0.5 / info["fps"],
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

        stats = DatasetStats.from_lerobot_stats(sub_dataset.meta.stats)
        if not stats.is_finite():
            dropped.append(f"{repo_id} (non-finite action/state stats)")
            continue
        datasets.append(StatsAttachedDataset(sub_dataset, stats))
        selected_dirs[repo_id] = dataset_dir
        per_dataset_stats[repo_id] = stats
        lerobot_stats[repo_id] = sub_dataset.meta.stats
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

    return DataSelection(
        datasets=datasets,
        per_dataset_stats=per_dataset_stats,
        lerobot_stats=lerobot_stats,
        camera_census=camera_census,
        dropped=dropped,
        action_dim=action_dim,
        state_dim=state_dim,
        action_names=list(anchor_info["features"]["action"].get("names") or []),
        total_episodes=total_episodes,
    )


def worker_init(_worker_id: int) -> None:
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
        community collections' generic image/image2 keys carry no reliable
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
            action_mean=torch.stack([item["action_mean"] for item in items]),
            action_std=torch.stack([item["action_std"] for item in items]),
            state_mean=torch.stack([item["state_mean"] for item in items]),
            state_std=torch.stack([item["state_std"] for item in items]),
            # Decided here (CPU, in the worker) so the train loop never syncs.
            has_padding=bool((batch["attention_mask"] == 0).any()),
        )


def encode_prefix(model: BijouModel, batch: CollatedBatch) -> PrefixKV:
    """``batch`` must already be device-resident."""
    padding_mask = batch.attention_mask if batch.has_padding else None
    with torch.no_grad():
        return model.encode_prefix(
            batch.input_ids,
            pixel_values=batch.pixel_values,
            image_position_ids=batch.image_position_ids,
            padding_mask=padding_mask,
        )
