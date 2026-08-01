"""Dataset selection, per-dataset stats, and prompt collation for Bijou.

Shared by training (``bijou.train``) and evaluation (``bijou.eval``) so both
provably select and prepare data the same way:

- ``discover_datasets``: resolve paths/collection roots (flat and
  ``<user>/<dataset>`` layouts), fnmatch excludes, path dedup.
- ``select_datasets``: the guard pipeline — dims anchored by the first
  dataset with standard features; loud drops for bespoke features, dim
  mismatches, missing/non-finite stats, metadata-vs-parquet frame count
  disagreements and cross-root duplicate repo ids. Optional deterministic
  per-dataset episode holdout (``EpisodeSplit``/``holdout_episodes``):
  training loads the TRAIN side, eval reproduces the exact HOLDOUT side
  from (fraction, split seed) alone — no persisted split files.
- ``DatasetStats``/``StatsAttachedDataset``: per-dataset stats (MEAN_STD +
  exact q01/q99) attached to every item (per-dataset normalization) with
  loud, bounded substitution of unfetchable samples.

Batch types and the shared Collator live in ``bijou.interface``; the
Gemma prompt strategy in ``bijou.encoders.gemma4``.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, override

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch import Tensor


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """One dataset's normalization stats for action and state: MEAN_STD
    plus the exact corpus q01/q99 quantiles.

    Plain float tuples, deliberately not tensors: the dataset objects are
    pickled into every spawned dataloader worker, and torch shares pickled
    CPU tensors through shared-memory file descriptors — 4 tensors x 300+
    datasets exhausts the default ulimit (observed: EMFILE on worker spawn).

    Quantiles are required on the data path (``from_lerobot_stats``) and
    None only when parsed from a checkpoint whose stats tables predate
    them (``from_state_dict``); consumers that need quantiles check for
    None and fail fast.
    """

    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    state_mean: tuple[float, ...]
    state_std: tuple[float, ...]
    action_q01: tuple[float, ...] | None
    action_q99: tuple[float, ...] | None
    state_q01: tuple[float, ...] | None
    state_q99: tuple[float, ...] | None

    def __post_init__(self) -> None:
        for modality in ("action", "state"):
            low = getattr(self, f"{modality}_q01")
            high = getattr(self, f"{modality}_q99")
            if (low is None) != (high is None):
                raise ValueError(
                    f"{modality} quantiles must be both present or both "
                    f"absent (q01 {'set' if low is not None else 'None'}, "
                    f"q99 {'set' if high is not None else 'None'})",
                )

    @classmethod
    def from_lerobot_stats(cls, stats: dict[str, dict[str, Any]]) -> DatasetStats:
        def as_vector(
            key: str,
            field: str,
            floor: float | None = None,
        ) -> tuple[float, ...]:
            values = torch.as_tensor(stats[key][field], dtype=torch.float32)
            if floor is not None:
                # Floor the stds: a (near-)constant joint would otherwise
                # amplify float rounding jitter ~1e4x into the normalized
                # targets. At the floor, deviations from the dataset mean
                # pass through ~unscaled.
                values = values.clamp(min=floor)
            return tuple(values.reshape(-1).tolist())

        def quantile(key: str, field: str) -> tuple[float, ...]:
            if field not in stats.get(key, {}):
                raise SystemExit(
                    f"dataset stats lack {key}.{field} (exact corpus "
                    "quantiles) — backfill them with `python -m "
                    "ldtools.backfill_quantile_stats --force <dataset_dir>` "
                    "(lerobot-dataset-tools) and retry",
                )
            return as_vector(key, field)

        return cls(
            action_mean=as_vector("action", "mean"),
            action_std=as_vector("action", "std", floor=1e-2),
            state_mean=as_vector("observation.state", "mean"),
            state_std=as_vector("observation.state", "std", floor=1e-2),
            action_q01=quantile("action", "q01"),
            action_q99=quantile("action", "q99"),
            state_q01=quantile("observation.state", "q01"),
            state_q99=quantile("observation.state", "q99"),
        )

    def is_finite(self) -> bool:
        vectors = (
            self.action_mean,
            self.action_std,
            self.state_mean,
            self.state_std,
            self.action_q01 or (),
            self.action_q99 or (),
            self.state_q01 or (),
            self.state_q99 or (),
        )
        return all(math.isfinite(x) for vector in vectors for x in vector)

    def state_dict(self) -> dict[str, dict[str, list[float]]]:
        payload: dict[str, dict[str, list[float]]] = {
            "action": {
                "mean": list(self.action_mean),
                "std": list(self.action_std),
            },
            "observation.state": {
                "mean": list(self.state_mean),
                "std": list(self.state_std),
            },
        }
        if self.action_q01 is not None and self.action_q99 is not None:
            payload["action"]["q01"] = list(self.action_q01)
            payload["action"]["q99"] = list(self.action_q99)
        if self.state_q01 is not None and self.state_q99 is not None:
            payload["observation.state"]["q01"] = list(self.state_q01)
            payload["observation.state"]["q99"] = list(self.state_q99)
        return payload

    @classmethod
    def from_state_dict(cls, data: dict[str, dict[str, list[float]]]) -> DatasetStats:
        """Exact inverse of :meth:`state_dict` — no flooring: serialized
        stats were floored at construction (``from_lerobot_stats``), and a
        checkpoint round-trip must not alter values. Tables written before
        quantiles existed parse with q01/q99 = None."""

        def optional(key: str, field: str) -> tuple[float, ...] | None:
            values = data[key].get(field)
            return tuple(values) if values is not None else None

        return cls(
            action_mean=tuple(data["action"]["mean"]),
            action_std=tuple(data["action"]["std"]),
            state_mean=tuple(data["observation.state"]["mean"]),
            state_std=tuple(data["observation.state"]["std"]),
            action_q01=optional("action", "q01"),
            action_q99=optional("action", "q99"),
            state_q01=optional("observation.state", "q01"),
            state_q99=optional("observation.state", "q99"),
        )

    def item_tensors(self) -> dict[str, Tensor]:
        """The per-item stats tensors exactly as training items carry them
        (materialized fresh per call — see the class docstring for why they
        are not stored as tensors). Quantile keys are present iff the stats
        carry them; the Collator turns their absence into NormStats
        q01/q99 = None.

        Shapes: action_* [action_dim]; state_* [state_dim] (per item — the
        collator stacks in the batch axis)."""
        tensors = {
            "action_mean": torch.tensor(self.action_mean),
            "action_std": torch.tensor(self.action_std),
            "state_mean": torch.tensor(self.state_mean),
            "state_std": torch.tensor(self.state_std),
        }
        if self.action_q01 is not None and self.action_q99 is not None:
            tensors["action_q01"] = torch.tensor(self.action_q01)
            tensors["action_q99"] = torch.tensor(self.action_q99)
        if self.state_q01 is not None and self.state_q99 is not None:
            tensors["state_q01"] = torch.tensor(self.state_q01)
            tensors["state_q99"] = torch.tensor(self.state_q99)
        return tensors


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

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._fetch_with_substitution(index, self._MAX_ATTEMPTS)
        item["repo_id"] = self.dataset.repo_id
        item.update(self.stats.item_tensors())
        return item

    def _fetch_with_substitution(self, index: int, attempts: int) -> dict[str, Any]:
        try:
            return self.dataset[index]
        except Exception as error:
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


class EpisodeSplit(Enum):
    """Which side of the per-dataset episode holdout a selection loads."""

    ALL = "all"
    TRAIN = "train"
    HOLDOUT = "holdout"


def holdout_episodes(
    repo_id: str,
    num_episodes: int,
    fraction: float,
    split_seed: int,
) -> tuple[int, ...]:
    """Deterministic per-dataset episode holdout.

    A pure function of (repo_id, num_episodes, fraction, split_seed):
    reproducible anywhere (training and eval agree without shared state),
    independent of dataset ordering, and deliberately independent of the
    training --seed (which changes across restarts — the holdout must not).
    Every dataset with >= 2 episodes contributes at least one held-out
    episode and always keeps at least one for training.
    """
    if fraction <= 0 or num_episodes < 2:
        return ()
    count = min(num_episodes - 1, max(1, round(fraction * num_episodes)))
    rng = random.Random(f"{split_seed}:{repo_id}")
    return tuple(sorted(rng.sample(range(num_episodes), count)))


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """The slice of a dataset's ``meta/info.json`` the selection pipeline
    consumes, parsed once per dataset instead of dug out of the raw dict at
    every use.

    ``action_state_dims`` is None when the standard action/state features
    are absent (a few community datasets use bespoke feature names, e.g.
    arm_action/hand_action/observation.arm_state — not trainable here).
    """

    fps: float
    total_episodes: int
    action_state_dims: tuple[int, int] | None
    action_names: tuple[str, ...]
    cameras: tuple[str, ...]

    @classmethod
    def from_json(cls, path: Path) -> DatasetInfo:
        data = json.loads(path.read_text())
        features: dict[str, Any] = data.get("features") or {}
        dims = None
        if "action" in features and "observation.state" in features:
            dims = (
                int(features["action"]["shape"][0]),
                int(features["observation.state"]["shape"][0]),
            )
        return cls(
            fps=float(data["fps"]),
            total_episodes=int(data["total_episodes"]),
            action_state_dims=dims,
            action_names=tuple(features.get("action", {}).get("names") or ()),
            cameras=tuple(
                sorted(
                    key.removeprefix("observation.images.")
                    for key, feature in features.items()
                    if feature.get("dtype") == "video"
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class DataSelection:
    """Outcome of the selection guard pipeline over discovered datasets.

    ``total_episodes`` counts the episodes actually loaded (i.e. after any
    episode-split filtering); ``held_out_episodes``/``held_out_datasets``
    count the holdout side across the selected datasets regardless of which
    side this selection loaded.
    """

    datasets: list[StatsAttachedDataset]
    per_dataset_stats: dict[str, DatasetStats]
    lerobot_stats: dict[str, dict[str, Any]]
    camera_census: Counter[tuple[str, ...]]
    dropped: list[str]
    action_dim: int
    state_dim: int
    action_names: list[str]
    total_episodes: int
    episode_split: EpisodeSplit
    held_out_episodes: int
    held_out_datasets: int

    def concat(self) -> torch.utils.data.ConcatDataset[dict[str, Any]]:
        return torch.utils.data.ConcatDataset(self.datasets)


def select_datasets(
    paths: tuple[Path, ...],
    exclude: tuple[str, ...],
    chunk_size: int,
    episode_split: EpisodeSplit = EpisodeSplit.ALL,
    holdout_fraction: float = 0.0,
    split_seed: int = 0,
    allowed_fps: tuple[float, ...] | None = None,
) -> DataSelection:
    """Discover, validate and wrap datasets; drop the incompatible loudly.

    Dims are dictated by the first discovered dataset that declares the
    standard action/observation.state features (the community collections
    mix in a few 7/12/14-dof and bespoke-feature datasets — cross-embodiment
    padding is out of scope for now).

    ``episode_split``/``holdout_fraction``/``split_seed`` select which side
    of the deterministic per-dataset episode holdout to load (see
    ``holdout_episodes``). TRAIN with fraction 0 loads everything —
    identical to ALL. Training and eval reproduce the same split by passing
    the same fraction and split seed; nothing is persisted.

    ``allowed_fps`` drops datasets recorded at any other frame rate (the
    chunk spans ``chunk_size`` NATIVE frames, so mixed-fps corpora mix
    wall-clock horizons — 11.9% of community frames are non-30fps). None =
    keep all (the historical behavior). NOTE: any filter changes the
    concatenated frame indexing, so eval scores are only comparable
    between runs using the same filter.
    """
    if episode_split is EpisodeSplit.HOLDOUT and holdout_fraction <= 0:
        raise ValueError("episode_split=HOLDOUT requires holdout_fraction > 0")
    dataset_dirs = discover_datasets(paths, exclude)
    dataset_infos = [
        DatasetInfo.from_json(d / "meta" / "info.json") for d in dataset_dirs
    ]
    anchor_info = next(
        (info for info in dataset_infos if info.action_state_dims is not None),
        None,
    )
    if anchor_info is None:
        raise ValueError(
            "no selected dataset declares action/observation.state features",
        )
    assert anchor_info.action_state_dims is not None
    action_dim, state_dim = anchor_info.action_state_dims

    datasets: list[StatsAttachedDataset] = []
    per_dataset_stats: dict[str, DatasetStats] = {}
    lerobot_stats: dict[str, dict[str, Any]] = {}
    camera_census: Counter[tuple[str, ...]] = Counter()
    dropped: list[str] = []
    total_episodes = 0
    held_out_total = 0
    held_out_datasets = 0
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
                f"keeping {selected_dirs[repo_id]})",
            )
            continue
        dims = info.action_state_dims
        if dims is None:
            dropped.append(f"{repo_id} (no action/observation.state features)")
            continue
        if dims != (action_dim, state_dim):
            dropped.append(f"{repo_id} (action/state dims {dims[0]}/{dims[1]})")
            continue
        if allowed_fps is not None and info.fps not in allowed_fps:
            allowed = ", ".join(f"{fps:g}" for fps in allowed_fps)
            dropped.append(f"{repo_id} (fps {info.fps:g} not in {{{allowed}}})")
            continue

        held_out = (
            holdout_episodes(
                repo_id,
                info.total_episodes,
                holdout_fraction,
                split_seed,
            )
            if episode_split is not EpisodeSplit.ALL
            else ()
        )
        episodes: list[int] | None
        if episode_split is EpisodeSplit.HOLDOUT:
            if not held_out:
                dropped.append(
                    f"{repo_id} (no held-out episodes: "
                    f"{info.total_episodes} episode(s) total)",
                )
                continue
            episodes = list(held_out)
        elif episode_split is EpisodeSplit.TRAIN and held_out:
            keep = set(range(info.total_episodes)) - set(held_out)
            episodes = sorted(keep)
        else:
            episodes = None

        sub_dataset = LeRobotDataset(
            repo_id,
            root=str(dataset_dir),
            episodes=episodes,
            delta_timestamps={"action": [i / info.fps for i in range(chunk_size)]},
            # Nearest-frame decode tolerance. lerobot's 1e-4 default is
            # unrepresentable deep into v3-format concatenated video files:
            # torchcodec returns fp32 pts, whose resolution at e.g. 1140s
            # (~1.4e-4) exceeds it, so a CORRECT nearest frame gets rejected
            # (observed: kaiserbuffle/hanoi_dc, 19-minute file). Half a
            # frame period is the exact nearest-frame criterion and still
            # catches genuine desync (off by >= a full frame).
            tolerance_s=0.5 / info.fps,
        )
        if sub_dataset.meta.stats is None:
            dropped.append(f"{repo_id} (no stats)")
            continue

        # Some community datasets ship metadata claiming more frames than
        # their parquet actually holds; ConcatDataset sizes by len(). With
        # an episode filter len(dataset) IS len(hf_dataset) (tautology), so
        # the claim must come from per-episode lengths in the metadata.
        actual_rows = len(sub_dataset.hf_dataset)
        if episodes is None:
            claimed_rows = len(sub_dataset)
        else:
            lengths = {
                int(ep): int(n)
                for ep, n in zip(
                    sub_dataset.meta.episodes["episode_index"],
                    sub_dataset.meta.episodes["length"],
                    strict=True,
                )
            }
            claimed_rows = sum(lengths.get(ep, 0) for ep in episodes)
        if claimed_rows != actual_rows:
            dropped.append(
                f"{repo_id} (metadata claims {claimed_rows} frames, "
                f"parquet holds {actual_rows})",
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
        camera_census[info.cameras] += 1
        total_episodes += sub_dataset.num_episodes
        held_out_total += len(held_out)
        held_out_datasets += 1 if held_out else 0
    if not datasets:
        # The caller normally prints the drop list; when NOTHING survives
        # (e.g. an --fps filter excluding every dataset) it never gets the
        # chance, so the reasons must ride in the error itself.
        reasons = "\n".join(f"  - {reason}" for reason in dropped)
        raise ValueError(f"no compatible datasets selected; dropped:\n{reasons}")

    return DataSelection(
        datasets=datasets,
        per_dataset_stats=per_dataset_stats,
        lerobot_stats=lerobot_stats,
        camera_census=camera_census,
        dropped=dropped,
        action_dim=action_dim,
        state_dim=state_dim,
        action_names=list(anchor_info.action_names),
        total_episodes=total_episodes,
        episode_split=episode_split,
        held_out_episodes=held_out_total,
        held_out_datasets=held_out_datasets,
    )


def worker_init(_worker_id: int) -> None:
    # Keep dataloader workers single-threaded: N workers x M torch threads
    # oversubscribes the host.
    torch.set_num_threads(1)
