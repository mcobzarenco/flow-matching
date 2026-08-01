"""Episode evidence shown to a judge: sampled frames + trajectory statistics.

Shared by the API and local judges so both see identical evidence for the
same (episode, num_timesteps, max_image_dim): frames are decoded once into
PIL (the Anthropic side encodes to JPEG at request-build time, the local
side feeds PIL straight to the processor), and the statistics block covers
the FULL trajectory from parquet — no video decoding beyond the sampled
timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image


def short_camera(key: str) -> str:
    """Dataset camera names without the feature-key boilerplate.

    "observation.images.image2" -> "image2". These short names are what the
    judge sees and what ``camera_kinds`` is keyed by — they match the names
    other tooling derives from the feature keys.
    """
    return key.removeprefix("observation.images.")


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    """Everything extracted from the dataset for one episode."""

    repo_id: str
    episode: int
    task: str
    fps: float
    num_frames: int
    duration_s: float
    motor_names: list[str]
    camera_names: list[str]  # dataset short names, e.g. "front", "image2"
    # Anonymous labels ("A", "B", ...) aligned with camera_names — what the
    # judge sees and answers with. Recorded names bias the verdict (measured:
    # a fixed overhead cam named "front" was tagged front over top until
    # anonymized), so judges never see them; callers translate back via
    # EpisodeJudgment.rename_cameras after validation.
    camera_labels: list[str]
    sampled_frames: list[int]  # 1-based frame numbers shown to the judge
    stats_text: str
    # (timestep label, camera LABEL, image) in chronological order
    frames: list[tuple[str, str, Image.Image]]


def tensor_to_image(chw: torch.Tensor, max_dim: int) -> Image.Image:
    """float32 CHW in [0,1] -> PIL RGB, bounded to max_dim on the long side."""
    array = (chw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(array)
    image.thumbnail((max_dim, max_dim))
    return image


def format_stats(
    action: np.ndarray,
    state: np.ndarray,
    motor_names: list[str],
    fps: float,
) -> str:
    """Compact per-motor and whole-trajectory statistics for the prompt."""
    delta = np.abs(np.diff(action, axis=0))
    tracking_error = np.abs(action - state).mean(axis=0)

    header = (
        f"{'motor':<22}{'min':>9}{'max':>9}{'mean':>9}{'std':>8}"
        f"{'path':>9}{'max|d|':>8}{'|a-s|':>8}"
    )
    lines = [header]
    for i, name in enumerate(motor_names):
        lines.append(
            f"{name:<22}"
            f"{action[:, i].min():>9.1f}"
            f"{action[:, i].max():>9.1f}"
            f"{action[:, i].mean():>9.1f}"
            f"{action[:, i].std():>8.1f}"
            f"{delta[:, i].sum():>9.1f}"
            f"{delta[:, i].max():>8.1f}"
            f"{tracking_error[i]:>8.1f}",
        )

    # Fraction of steps where no motor target moves more than 1% of its
    # episode range: a proxy for idle time.
    ranges = action.max(axis=0) - action.min(axis=0)
    idle_threshold = np.maximum(ranges * 0.01, 1e-6)
    idle_fraction = float((delta < idle_threshold).all(axis=1).mean())

    lines += [
        "",
        (
            "columns: action min/max/mean/std over the episode; path = total "
            "travelled distance sum(|delta|); max|d| = largest single-step jump "
            f"(jerkiness proxy, steps are {1000 / fps:.0f} ms apart); |a-s| = "
            "mean |action - state| (commanded target vs. achieved position "
            "tracking error)."
        ),
        f"idle steps (all motors move < 1% of their range): {idle_fraction:.0%}",
        (
            "Units are dataset-dependent (raw joint values, often degrees or a "
            "normalized [-100, 100] range)."
        ),
    ]
    return "\n".join(lines)


def load_episode_summary(
    root: Path,
    repo_id: str,
    episode: int,
    *,
    num_timesteps: int,
    max_image_dim: int,
    cameras: list[str] | None = None,
) -> EpisodeSummary:
    """Decode evidence for one episode (sampled frames only, full-episode
    trajectory statistics from parquet)."""
    dataset = LeRobotDataset(repo_id, root=str(root))
    if not 0 <= episode < dataset.num_episodes:
        raise SystemExit(
            f"episode {episode} out of range (dataset has {dataset.num_episodes})",
        )

    row = dataset.meta.episodes[episode]
    start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
    num_frames = stop - start
    fps = float(dataset.fps)

    tasks = row.get("tasks") or ["<no task recorded>"]
    task = "; ".join(tasks) if isinstance(tasks, list) else str(tasks)

    # Trajectory columns straight from parquet (no video decoding).
    table = dataset.hf_dataset[start:stop]
    action = np.asarray(table["action"], dtype=np.float32)
    state = np.asarray(table["observation.state"], dtype=np.float32)

    feature = dataset.meta.features["action"]
    motor_names = list(
        feature.get("names") or [f"motor_{i}" for i in range(action.shape[1])],
    )

    camera_keys = list(dataset.meta.camera_keys)
    if cameras:
        # Accept either full feature keys or short names in the filter.
        wanted = {short_camera(c) for c in cameras}
        known = {short_camera(k) for k in camera_keys}
        unknown = wanted - known
        if unknown:
            raise SystemExit(
                f"unknown cameras {sorted(unknown)}; dataset has {sorted(known)}",
            )
        camera_keys = [k for k in camera_keys if short_camera(k) in wanted]

    # Deterministic anonymous labels: sorted by dataset name, then A, B, …
    camera_keys = sorted(camera_keys, key=short_camera)
    camera_labels = [chr(ord("A") + i) for i in range(len(camera_keys))]

    # Evenly spaced timesteps, always including the first and last frame.
    picks = np.unique(np.linspace(0, num_frames - 1, num_timesteps).round().astype(int))
    frames: list[tuple[str, str, Image.Image]] = []
    for local_idx in picks:
        item = dataset[start + int(local_idx)]  # decodes video for this frame only
        label = f"frame {local_idx + 1}/{num_frames} (t={local_idx / fps:.1f}s)"
        frames.extend(
            (label, camera_label, tensor_to_image(item[camera], max_image_dim))
            for camera, camera_label in zip(camera_keys, camera_labels, strict=True)
        )

    return EpisodeSummary(
        repo_id=repo_id,
        episode=episode,
        task=task,
        fps=fps,
        num_frames=num_frames,
        duration_s=num_frames / fps,
        motor_names=motor_names,
        camera_names=[short_camera(k) for k in camera_keys],
        camera_labels=camera_labels,
        sampled_frames=[int(local_idx) + 1 for local_idx in picks],
        stats_text=format_stats(action, state, motor_names, fps),
        frames=frames,
    )
