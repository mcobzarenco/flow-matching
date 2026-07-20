"""Open-loop evaluation of SmolVLA on a local LeRobot v3.0 dataset.

For every evaluated frame, the policy predicts a full action chunk from the
frame's observations (cameras + robot state + task instruction). The chunk is
compared against the ground-truth actions recorded in the dataset, i.e. this
is an offline / open-loop evaluation: no robot or simulator is involved.

Reported metrics (in raw action units, degrees for SO-100 joints):
  - chunk_mse / chunk_mae: error over the whole predicted chunk (padding-masked)
  - first_mae:             error of the first action of the chunk only
  - per-motor MAE:         chunk MAE split per action dimension

CLI usage:
    uv run python eval_smolvla.py \
        --root /home/marius/w/community_dataset_v1_v3/ZGGZZG/so100_drop0 \
        --repo-id ZGGZZG/so100_drop0 \
        --episodes 0 1 --stride 30

Interactive usage (e.g. IPython):
    from eval_smolvla import SmolVLAEval

    ev = SmolVLAEval.load()               # defaults: so100_drop0 + smolvla_base
    item = ev.dataset[0]                  # raw dataset item
    obs = extract_observation(item)       # what the preprocessor consumes
    pred = ev.predict_chunk(item)         # (chunk_size, action_dim) tensor

    frame = ev.evaluate_frame(0)          # prediction + ground truth + metrics
    frame.chunk_mae, frame.per_motor_mae

    ep = ev.evaluate_episode(0, stride=30)
    ep.chunk_mae, summarize([ep], ev.motor_names)

Determinism: the only stochastic part of SmolVLA inference is the initial
flow-matching noise. We sample it from a seeded generator, deriving the seed
per frame as `seed + dataset_index`, so a frame's prediction is identical no
matter the stride or evaluation order. Set `seed=None` (API) for stochastic
sampling.

The first run downloads the policy weights and tokenizer from the HF Hub
(~1 GB; public repos, so no authentication is required).
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction

DEFAULT_ROOT = Path("/home/marius/w/community_dataset_v1_v3/ZGGZZG/so100_drop0")
DEFAULT_REPO_ID = "ZGGZZG/so100_drop0"
DEFAULT_POLICY = "lerobot/smolvla_base"

PreprocessorPipeline = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
PostprocessorPipeline = PolicyProcessorPipeline[PolicyAction, PolicyAction]


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def load_policy(
    policy_path: str = DEFAULT_POLICY, device: str | torch.device | None = None
) -> SmolVLAPolicy:
    """Load a SmolVLA checkpoint and prepare it for inference."""
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy.to(resolve_device(device))
    policy.eval()
    return policy


def load_chunked_dataset(
    root: Path | str, repo_id: str, chunk_size: int
) -> LeRobotDataset:
    """Load a local LeRobot v3.0 dataset that yields aligned action chunks.

    `delta_timestamps` makes every item carry `action` of shape
    (chunk_size, action_dim) — the ground truth for a predicted chunk — plus
    an `action_is_pad` mask marking entries past the episode end.
    """
    root = Path(root)
    fps = json.loads((root / "meta" / "info.json").read_text())["fps"]
    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
    return LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)


def policy_camera_keys(policy: SmolVLAPolicy) -> list[str]:
    return [k for k in (policy.config.input_features or {}) if "image" in k]


def build_rename_map(
    dataset_cameras: list[str], policy_cameras: list[str]
) -> dict[str, str]:
    """Map dataset camera keys onto the camera keys the policy was trained with.

    Both sides are taken in sorted order, which matches the convention used
    when the SmolVLA community datasets were standardized (camera1..cameraN).
    Extra policy cameras are simply left absent; SmolVLA masks missing views.
    """
    rename = dict(zip(sorted(dataset_cameras), sorted(policy_cameras)))
    if len(dataset_cameras) > len(policy_cameras):
        dropped = sorted(dataset_cameras)[len(policy_cameras) :]
        print(
            f"warning: dataset has more cameras than the policy accepts, dropping {dropped}"
        )
    return rename


def remap_stats(stats: dict, rename_map: dict[str, str]) -> dict:
    """Re-key dataset stats so they match the policy's (renamed) feature names."""
    return {rename_map.get(k, k): v for k, v in stats.items()}


def build_processors(
    policy: SmolVLAPolicy,
    pretrained_path: str,
    rename_map: dict[str, str],
    device: torch.device,
    dataset_stats: dict | None = None,
) -> tuple[PreprocessorPipeline, PostprocessorPipeline]:
    """Load the policy's pre/post-processor pipelines from its checkpoint.

    When `dataset_stats` is given, the normalizer/unnormalizer steps are
    re-calibrated to the target dataset, exactly like lerobot_train.py does
    when loading a pretrained policy onto a new dataset. Without it the
    pipelines keep the stats saved with the checkpoint.
    """
    preprocessor_overrides: dict[str, dict] = {
        "rename_observations_processor": {"rename_map": rename_map},
        "device_processor": {"device": str(device)},
    }
    postprocessor_overrides: dict[str, dict] = {}
    if dataset_stats is not None:
        stats = remap_stats(dataset_stats, rename_map)
        preprocessor_overrides["normalizer_processor"] = {
            "stats": stats,
            "features": {
                **(policy.config.input_features or {}),
                **(policy.config.output_features or {}),
            },
            "norm_map": policy.config.normalization_mapping,
        }
        postprocessor_overrides["unnormalizer_processor"] = {
            "stats": stats,
            "features": policy.config.output_features or {},
            "norm_map": policy.config.normalization_mapping,
        }

    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )


def extract_observation(item: dict) -> dict:
    """Turn a dataset item into the input dict the preprocessor consumes.

    Keeps the `observation.*` entries (images as (C, H, W) float tensors,
    state as (state_dim,)) plus the natural-language `task`. Ground-truth
    action and bookkeeping columns are dropped; the preprocessor adds the
    batch dimension itself.
    """
    observation = {k: v for k, v in item.items() if k.startswith("observation.")}
    observation["task"] = item["task"]
    return observation


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FrameResult:
    """Prediction vs. ground truth for a single dataset frame."""

    index: int  # global dataset frame index
    pred: torch.Tensor  # (chunk_size, action_dim), raw action units
    gt: torch.Tensor  # (chunk_size, action_dim), from the dataset
    valid: torch.Tensor  # (chunk_size,) bool, False where gt is padding
    inference_seconds: float

    @property
    def diff(self) -> torch.Tensor:
        """(n_valid, action_dim) error over the valid part of the chunk."""
        return (self.pred - self.gt)[self.valid]

    @property
    def chunk_mse(self) -> float:
        return float((self.diff**2).mean())

    @property
    def chunk_mae(self) -> float:
        return float(self.diff.abs().mean())

    @property
    def first_mae(self) -> float:
        return float((self.pred[0] - self.gt[0]).abs().mean())

    @property
    def per_motor_mae(self) -> torch.Tensor:
        return self.diff.abs().mean(dim=0)


@dataclass(frozen=True, eq=False)
class EpisodeResult:
    """Aggregated frame results for one episode."""

    episode: int
    frames: list[FrameResult]

    @property
    def diff(self) -> torch.Tensor:
        return torch.cat([f.diff for f in self.frames])

    @property
    def chunk_mse(self) -> float:
        return float((self.diff**2).mean())

    @property
    def chunk_mae(self) -> float:
        return float(self.diff.abs().mean())

    @property
    def first_mae(self) -> float:
        return sum(f.first_mae for f in self.frames) / len(self.frames)

    @property
    def per_motor_mae(self) -> torch.Tensor:
        return self.diff.abs().mean(dim=0)

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "frames_evaluated": len(self.frames),
            "chunk_mse": self.chunk_mse,
            "chunk_mae": self.chunk_mae,
            "first_mae": self.first_mae,
        }


def summarize(results: list[EpisodeResult], motor_names: list[str]) -> dict:
    """Aggregate episode results into the overall metrics dict."""
    frames = [f for r in results for f in r.frames]
    diff = torch.cat([f.diff for f in frames])
    per_motor = diff.abs().mean(dim=0)
    return {
        "chunk_mse": float((diff**2).mean()),
        "chunk_mae": float(diff.abs().mean()),
        "first_mae": sum(f.first_mae for f in frames) / len(frames),
        "per_motor_mae": {
            str(name): float(v) for name, v in zip(motor_names, per_motor)
        },
        "frames_evaluated": len(frames),
        "avg_inference_seconds": sum(f.inference_seconds for f in frames) / len(frames),
    }


# --------------------------------------------------------------------------
# Evaluation context
# --------------------------------------------------------------------------


@dataclass(eq=False)
class SmolVLAEval:
    """Everything needed to run SmolVLA on dataset frames.

    Build one with `SmolVLAEval.load(...)`, then poke at it interactively:
    `ev.predict_chunk(ev.dataset[0])`, `ev.evaluate_frame(123)`, ...
    """

    policy: SmolVLAPolicy
    preprocessor: PreprocessorPipeline
    postprocessor: PostprocessorPipeline
    dataset: LeRobotDataset
    rename_map: dict[str, str]
    device: torch.device
    amp: bool = False
    seed: int | None = 42

    @classmethod
    def load(
        cls,
        root: Path | str = DEFAULT_ROOT,
        repo_id: str = DEFAULT_REPO_ID,
        policy_path: str = DEFAULT_POLICY,
        device: str | torch.device | None = None,
        use_dataset_stats: bool = True,
        amp: bool = False,
        seed: int | None = 42,
    ) -> "SmolVLAEval":
        device = resolve_device(device)
        policy = load_policy(policy_path, device)
        dataset = load_chunked_dataset(root, repo_id, policy.config.chunk_size)
        rename_map = build_rename_map(
            dataset.meta.camera_keys, policy_camera_keys(policy)
        )
        preprocessor, postprocessor = build_processors(
            policy,
            policy_path,
            rename_map,
            device,
            dataset_stats=(dataset.meta.stats or {}) if use_dataset_stats else None,
        )
        return cls(
            policy, preprocessor, postprocessor, dataset, rename_map, device, amp, seed
        )

    @property
    def motor_names(self) -> list[str]:
        feature = self.dataset.meta.features["action"]
        names = feature.get("names") or [
            f"motor_{i}" for i in range(feature["shape"][0])
        ]
        return list(names)

    def sample_noise(self, seed: int) -> torch.Tensor:
        """Seeded flow-matching noise, matching SmolVLA's internal shape.

        Drawn on CPU so the values are identical regardless of device, then
        moved to the policy's device.
        """
        shape = (1, self.policy.config.chunk_size, self.policy.config.max_action_dim)
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            self.device
        )

    def predict_chunk(self, item: dict, seed: int | None = None) -> torch.Tensor:
        """Predict an action chunk for one (unbatched) dataset item.

        Returns a (chunk_size, action_dim) float32 CPU tensor in raw action
        units (the postprocessor un-normalizes the policy output). `seed`
        falls back to `self.seed`; if both are None the noise is unseeded.
        """
        if seed is None:
            seed = self.seed
        noise = self.sample_noise(seed) if seed is not None else None
        batch = self.preprocessor(extract_observation(item))
        with torch.inference_mode():
            if self.amp and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = self.policy.predict_action_chunk(batch, noise=noise)
            else:
                pred = self.policy.predict_action_chunk(batch, noise=noise)
        pred = self.postprocessor(pred)
        return pred[0].to("cpu", torch.float32)

    def evaluate_frame(self, index: int) -> FrameResult:
        """Predict at one dataset index and compare against the ground truth.

        The noise seed is derived as `self.seed + index`, making the result
        reproducible and independent of evaluation order.
        """
        item = self.dataset[index]
        seed = None if self.seed is None else self.seed + index
        start = time.perf_counter()
        pred = self.predict_chunk(item, seed=seed)
        elapsed = time.perf_counter() - start
        gt = item["action"]
        return FrameResult(
            index=index,
            pred=pred[: gt.shape[0]],
            gt=gt,
            valid=~item["action_is_pad"],
            inference_seconds=elapsed,
        )

    def episode_frame_indices(self, episode: int, stride: int = 1) -> range:
        row = self.dataset.meta.episodes[episode]
        return range(
            int(row["dataset_from_index"]), int(row["dataset_to_index"]), stride
        )

    def evaluate_episode(self, episode: int, stride: int = 1) -> EpisodeResult:
        self.policy.reset()
        frames = [
            self.evaluate_frame(i) for i in self.episode_frame_indices(episode, stride)
        ]
        return EpisodeResult(episode=episode, frames=frames)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Local dataset directory containing meta/, data/, videos/ (v3.0 format).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Dataset repo id (only used for bookkeeping when --root is local).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=DEFAULT_POLICY,
        help="Policy repo id or local checkpoint directory.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to evaluate. Defaults to the first --num-episodes.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=2,
        help="Number of episodes to evaluate when --episodes is not given.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Evaluate every Nth frame of an episode (1 = every frame).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run inference under bfloat16 autocast (faster, slightly less exact).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for the flow-matching noise. Each frame uses seed + its "
        "dataset index, so results are reproducible for any stride/episode subset.",
    )
    parser.add_argument(
        "--pretrained-stats",
        action="store_true",
        help="Keep the policy's saved normalization stats instead of the dataset's. "
        "By default the normalizer/unnormalizer are re-calibrated with the target "
        "dataset stats, mirroring what lerobot's train/fine-tune script does when "
        "loading a pretrained policy onto a dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading policy {args.policy} and dataset {args.repo_id} ...")
    ev = SmolVLAEval.load(
        root=args.root,
        repo_id=args.repo_id,
        policy_path=args.policy,
        device=args.device,
        use_dataset_stats=not args.pretrained_stats,
        amp=args.amp,
        seed=args.seed,
    )
    print(f"camera mapping: {ev.rename_map}")

    episodes = args.episodes
    if not episodes:
        episodes = list(range(min(args.num_episodes, ev.dataset.num_episodes)))

    results = []
    for episode in episodes:
        result = ev.evaluate_episode(episode, args.stride)
        results.append(result)
        print(
            f"episode {episode:4d} | frames {len(result.frames):4d} | "
            f"chunk MSE {result.chunk_mse:10.3f} | "
            f"chunk MAE {result.chunk_mae:8.3f} | "
            f"first-action MAE {result.first_mae:8.3f}"
        )

    overall = summarize(results, ev.motor_names)
    print("\n=== overall (action units, e.g. degrees for SO-100) ===")
    print(f"frames evaluated : {overall['frames_evaluated']}")
    print(f"chunk MSE        : {overall['chunk_mse']:.3f}")
    print(f"chunk MAE        : {overall['chunk_mae']:.3f}")
    print(f"first-action MAE : {overall['first_mae']:.3f}")
    print(f"avg inference    : {overall['avg_inference_seconds'] * 1000:.0f} ms/frame")
    print("per-motor MAE    :")
    for name, value in overall["per_motor_mae"].items():
        print(f"  {name:24s} {value:8.3f}")

    if args.output:
        payload = {
            "policy": args.policy,
            "dataset": {"repo_id": args.repo_id, "root": str(args.root)},
            "config": {
                "episodes": episodes,
                "stride": args.stride,
                "chunk_size": ev.policy.config.chunk_size,
                "device": str(ev.device),
                "amp": ev.amp,
                "seed": ev.seed,
                "camera_rename_map": ev.rename_map,
            },
            "overall": overall,
            "per_episode": [r.to_dict() for r in results],
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
