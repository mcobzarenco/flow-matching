"""SmolVLA as an evaluation policy (ported from the retired eval_smolvla.py).

lerobot policies normalize with dataset stats through their processor
pipelines and expect the camera keys they were trained with. Our eval spans
many datasets with heterogeneous cameras and stats, so the pre/post
processors (with per-dataset stats and a positional camera rename map) are
built lazily and cached per repo_id.

Items are predicted one by one (the lerobot pipeline is unbatched here);
noise is seeded per item as ``seed + global_index``, matching the
determinism convention of the other policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as LerobotSmolVLA
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction
from torch import Tensor

from .policies import sample_noise

PreprocessorPipeline = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
PostprocessorPipeline = PolicyProcessorPipeline[PolicyAction, PolicyAction]


def build_rename_map(
    dataset_cameras: list[str],
    policy_cameras: list[str],
) -> dict[str, str]:
    """Map dataset camera keys onto the policy's camera keys, both in sorted
    order (the convention used when the SmolVLA community datasets were
    standardized). Extra dataset cameras are dropped; SmolVLA masks missing
    policy views."""
    # strict=False: unequal camera counts are the documented case above.
    return dict(zip(sorted(dataset_cameras), sorted(policy_cameras), strict=False))


def stats_to_tensors(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Tensor]]:
    return {
        key: {field: torch.as_tensor(value) for field, value in entry.items()}
        for key, entry in stats.items()
    }


@dataclass(frozen=True, slots=True)
class RepoProcessors:
    """One repo's lerobot processor pipelines: the camera rename map plus
    pre/post processors with that repo's stats baked in."""

    rename_map: dict[str, str]
    preprocessor: PreprocessorPipeline
    postprocessor: PostprocessorPipeline


class SmolVLAEvalPolicy:
    """SmolVLA checkpoint scored on the same items as the other policies."""

    def __init__(
        self,
        policy_path: str,
        *,
        device: torch.device,
        seed: int,
        lerobot_stats: dict[str, dict[str, Any]],
    ) -> None:
        self.name = f"smolvla:{policy_path.rsplit('/', 1)[-1]}"
        self.policy_path = policy_path
        self.device = device
        self.seed = seed
        self.lerobot_stats = lerobot_stats
        self.policy = LerobotSmolVLA.from_pretrained(policy_path)
        self.policy.to(device)
        self.policy.eval()
        self._processors: dict[str, RepoProcessors] = {}

    @property
    def chunk_size(self) -> int:
        return int(self.policy.config.chunk_size)

    def _camera_keys(self) -> list[str]:
        return [k for k in (self.policy.config.input_features or {}) if "image" in k]

    def _processors_for(
        self,
        repo_id: str,
        item: dict[str, Any],
    ) -> RepoProcessors:
        cached = self._processors.get(repo_id)
        if cached is not None:
            return cached
        dataset_cameras = sorted(k for k in item if k.startswith("observation.images."))
        rename_map = build_rename_map(dataset_cameras, self._camera_keys())
        stats = stats_to_tensors(self.lerobot_stats[repo_id])
        stats = {rename_map.get(k, k): v for k, v in stats.items()}
        features = {
            **(self.policy.config.input_features or {}),
            **(self.policy.config.output_features or {}),
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=self.policy_path,
            preprocessor_overrides={
                "rename_observations_processor": {"rename_map": rename_map},
                "device_processor": {"device": str(self.device)},
                "normalizer_processor": {
                    "stats": stats,
                    "features": features,
                    "norm_map": self.policy.config.normalization_mapping,
                },
            },
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "stats": stats,
                    "features": self.policy.config.output_features or {},
                    "norm_map": self.policy.config.normalization_mapping,
                },
            },
        )
        self._processors[repo_id] = RepoProcessors(
            rename_map=rename_map,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        return self._processors[repo_id]

    @torch.no_grad()
    def predict(self, items: list[dict[str, Any]], indices: list[int]) -> list[Tensor]:
        predictions: list[Tensor] = []
        for item, index in zip(items, indices, strict=True):
            repo_id = str(item["repo_id"])
            processors = self._processors_for(repo_id, item)
            observation = {
                k: v for k, v in item.items() if k.startswith("observation.")
            }
            observation["task"] = item["task"]
            noise = sample_noise(
                self.seed + index,
                (1, self.chunk_size, int(self.policy.config.max_action_dim)),
            ).to(self.device)
            batch = processors.preprocessor(observation)
            self.policy.reset()
            predicted = self.policy.predict_action_chunk(batch, noise=noise)
            predicted = processors.postprocessor(predicted)
            predictions.append(predicted[0].to("cpu", torch.float32))
        return predictions
