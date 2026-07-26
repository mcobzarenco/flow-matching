"""Building a Bijou VLA from a Gemma 4 checkpoint.

``from_backbone("google/gemma-4-e2b-it", action_dim=6, state_dim=6)`` gives a
ready-to-train model:

- the backbone is loaded truncated to its non-KV-shared prefix (layers 0–14
  for E2B) with only the needed weights: dropped decoder layers, per-layer
  embedding (PLE) tensors sliced to the kept layers, no audio tower, and the
  LM head is a tie (zero extra memory). ~2.1B params instead of 5.2B for E2B.
  It is frozen (eval mode, ``requires_grad=False``).
- the action expert is freshly initialized on the same device/dtype.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .expert import ActionExpert, ExpertConfig, SelfAttentionMode
from .gemma4.config import Gemma4Config, LayerType
from .gemma4.layers import DEFAULT_ATTENTION_BACKEND, AttentionBackend, DeviceLike
from .gemma4.loading import load_config, load_model, resolve_checkpoint_dir
from .model import BijouModel


def prefix_global_layers(config: Gemma4Config) -> tuple[int, ...]:
    """Global-attention layers within the non-KV-shared prefix — the K/V
    streams available to the expert (E2B: (4, 9, 14); E4B: (5, 11, 17, 23))."""
    text = config.text
    return tuple(
        idx
        for idx in range(text.first_kv_shared_layer_idx)
        if text.layer_types[idx] is LayerType.FULL
    )


def default_expert_config(
    backbone: Gemma4Config,
    *,
    action_dim: int,
    state_dim: int,
    stream_counts: tuple[int, ...] = (4, 4, 7),
    hidden_size: int = 768,
    num_attention_heads: int = 6,
    intermediate_size: int = 3072,
    cross_attention_heads: int = 4,
    chunk_size: int = 50,
    time_embed_dim: int = 256,
    self_attention_mode: SelfAttentionMode = SelfAttentionMode.CAUSAL_ACTIONS,
    self_attention_rope_theta: float = 10_000.0,
) -> ExpertConfig:
    """Expert config with a blocks cross-attention schedule.

    ``stream_counts[i]`` expert layers attend the i-th global prefix layer,
    shallow to deep; the default (4, 4, 7) weights the deepest stream (L14
    for E2B — the one the backbone's own KV-shared half runs on) and gives a
    15-layer expert. For E4B (four streams) pass four counts.
    """
    streams = prefix_global_layers(backbone)
    if len(stream_counts) != len(streams):
        raise ValueError(
            f"stream_counts has {len(stream_counts)} entries but the backbone "
            f"prefix has {len(streams)} global layers ({streams})",
        )
    schedule = tuple(
        chain.from_iterable(
            (stream,) * count
            for stream, count in zip(streams, stream_counts, strict=True)
        ),
    )
    return ExpertConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_activation=backbone.text.hidden_activation,
        rms_norm_eps=backbone.text.rms_norm_eps,
        self_attention_mode=self_attention_mode,
        self_attention_rope_theta=self_attention_rope_theta,
        cross_attention_heads=cross_attention_heads,
        cross_attention_head_dim=backbone.text.global_head_dim,
        cross_attention_rope=backbone.text.rope_parameters[LayerType.FULL],
        cross_attention_schedule=schedule,
        action_dim=action_dim,
        state_dim=state_dim,
        chunk_size=chunk_size,
        time_embed_dim=time_embed_dim,
    )


def from_backbone(
    model_id_or_path: str | Path,
    expert_config: ExpertConfig | None = None,
    *,
    action_dim: int | None = None,
    state_dim: int | None = None,
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    expert_dtype: torch.dtype | None = None,
    attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
) -> BijouModel:
    """Build a Bijou model from a Gemma 4 checkpoint.

    Pass either a full ``expert_config`` or just ``action_dim``/``state_dim``
    to use :func:`default_expert_config`. The backbone is truncated to its
    non-KV-shared prefix, frozen; the expert is freshly initialized.
    ``expert_dtype`` may differ from the backbone dtype (e.g. fp32 expert on
    a bf16 backbone for training — the expert casts its inputs and the
    exported KV streams to its own dtype).
    """
    checkpoint_dir = resolve_checkpoint_dir(model_id_or_path)
    config = load_config(checkpoint_dir)

    if expert_config is None:
        if action_dim is None or state_dim is None:
            raise ValueError(
                "pass either expert_config or both action_dim and state_dim",
            )
        expert_config = default_expert_config(
            config,
            action_dim=action_dim,
            state_dim=state_dim,
        )

    available = prefix_global_layers(config)
    for stream in expert_config.streams:
        if stream not in available:
            raise ValueError(
                f"cross-attention stream {stream} is not a global layer of the "
                f"backbone prefix (available: {available})",
            )

    backbone = load_model(
        checkpoint_dir,
        device="cpu" if device is None else device,
        dtype=dtype,
        attn_backend=attn_backend,
        truncate_layers=config.text.first_kv_shared_layer_idx,
    )
    if expert_dtype is None:
        expert_dtype = dtype if dtype is not None else config.dtype
    expert = ActionExpert(
        expert_config,
        attn_backend=attn_backend,
        device=device,
        dtype=expert_dtype,
    )
    return BijouModel(backbone=backbone, expert=expert)


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    """Metadata of a bijou training checkpoint (bijou_config.json)."""

    backbone: str
    train_args: dict[str, Any]
    step: int
    normalization: dict[str, dict[str, list[float]]]
    per_dataset_normalization: dict[str, dict[str, dict[str, list[float]]]]

    @property
    def chunk_size(self) -> int:
        return int(self.train_args["chunk_size"])

    @property
    def max_soft_tokens(self) -> int:
        return int(self.train_args["max_soft_tokens"])


def expert_config_from_train_args(
    backbone_config: Gemma4Config,
    train_args: dict[str, Any],
    *,
    action_dim: int,
    state_dim: int,
) -> ExpertConfig:
    """Rebuild the expert config a training run used from its recorded args
    (the serialized expert_config in bijou_config.json stringifies enums and
    nested dataclasses; the train args are the clean source)."""
    return default_expert_config(
        backbone_config,
        action_dim=action_dim,
        state_dim=state_dim,
        stream_counts=tuple(train_args["stream_counts"]),
        hidden_size=int(train_args["expert_hidden"]),
        num_attention_heads=int(train_args["expert_heads"]),
        intermediate_size=int(train_args["expert_intermediate"]),
        cross_attention_heads=int(train_args["expert_cross_heads"]),
        chunk_size=int(train_args["chunk_size"]),
        self_attention_mode=SelfAttentionMode(train_args["self_attention_mode"]),
    )


def from_checkpoint(
    checkpoint: str | Path,
    *,
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    expert_dtype: torch.dtype = torch.float32,
    attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
) -> tuple[BijouModel, CheckpointInfo]:
    """Load a bijou training checkpoint directory (as written by
    bijou.train.save_checkpoint): backbone resolved from the recorded id,
    expert config rebuilt from the recorded train args, expert weights
    loaded strictly. Returns the eval-mode model plus checkpoint metadata
    (normalization stats table etc.)."""
    checkpoint = Path(checkpoint)
    meta = json.loads((checkpoint / "bijou_config.json").read_text())
    info = CheckpointInfo(
        backbone=meta["backbone"],
        train_args=meta["train_args"],
        step=int(meta["step"]),
        normalization=meta["normalization"],
        per_dataset_normalization=meta.get("per_dataset_normalization", {}),
    )
    checkpoint_dir = resolve_checkpoint_dir(info.backbone)
    expert_config = expert_config_from_train_args(
        load_config(checkpoint_dir),
        info.train_args,
        action_dim=len(info.normalization["action"]["mean"]),
        state_dim=len(info.normalization["observation.state"]["mean"]),
    )
    model = from_backbone(
        checkpoint_dir,
        expert_config,
        device=device,
        dtype=dtype,
        expert_dtype=expert_dtype,
        attn_backend=attn_backend,
    )
    model.expert.load_state_dict(
        load_file(str(checkpoint / "expert.safetensors"), device=str(device)),
        strict=True,
    )
    model.eval()
    return model, info
