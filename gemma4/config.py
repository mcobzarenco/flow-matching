"""Typed configuration for the pure-torch Gemma 4 implementation.

Mirrors the fields of the HF `Gemma4Config` family that are relevant for
inference, parsed from a checkpoint's ``config.json``. Only fields that this
implementation actually reads are kept — if you add a feature, add its config
field here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import torch


class LayerType(StrEnum):
    """Attention pattern of a decoder layer."""

    SLIDING = "sliding_attention"
    FULL = "full_attention"


class RopeType(StrEnum):
    DEFAULT = "default"
    # p-RoPE: only `partial_rotary_factor * head_dim` dimensions are rotated;
    # the remaining frequencies are zero (cos=1, sin=0 -> identity).
    PROPORTIONAL = "proportional"


class AttentionBackend(StrEnum):
    """How attention is computed.

    EAGER mirrors HF's reference implementation op-for-op (fp32 softmax) and
    is the parity baseline. SDPA uses ``F.scaled_dot_product_attention``
    (fused kernels; numerics differ at bf16-ULP scale, greedy tokens verified
    identical — see gemma4/verify_parity.py) and is the default: measured on
    H100 it is ~1.9x faster / 4.7x leaner on 8k-token text prefill and ~1.6x
    on image prefill, with decode dispatched to eager at q_len==1 (faster
    there). This is a runtime choice, not a checkpoint property: it is never
    read from config.json.
    """

    EAGER = "eager"
    SDPA = "sdpa"


@dataclass(frozen=True, slots=True)
class RopeParameters:
    rope_type: RopeType
    rope_theta: float
    partial_rotary_factor: float = 1.0
    # Linear position scaling factor (divides the inverse frequencies).
    factor: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            rope_type=RopeType(data.get("rope_type", "default")),
            rope_theta=float(data["rope_theta"]),
            partial_rotary_factor=float(data.get("partial_rotary_factor", 1.0)),
            factor=float(data.get("factor", 1.0)),
        )


def _default_layer_types(num_layers: int) -> tuple[LayerType, ...]:
    """Default 5:1 sliding/full pattern (every 6th layer is full attention)."""
    return tuple(
        LayerType.SLIDING if (i + 1) % 6 else LayerType.FULL for i in range(num_layers)
    )


@dataclass(frozen=True, slots=True)
class Gemma4TextConfig:
    """Decoder config. Defaults correspond to Gemma 4 E2B."""

    vocab_size: int = 262_144
    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 35
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    eos_token_ids: tuple[int, ...] = (1,)
    bos_token_id: int = 2
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    sliding_window: int = 512
    layer_types: tuple[LayerType, ...] = field(
        default_factory=lambda: _default_layer_types(35)
    )
    final_logit_softcapping: float | None = 30.0
    # None (E2B) => plain causal; "vision" (12B) => bidirectional attention
    # within each image block; "all" => fully bidirectional.
    use_bidirectional_attention: str | None = None
    rope_parameters: dict[LayerType, RopeParameters] = field(
        default_factory=lambda: {
            LayerType.SLIDING: RopeParameters(
                rope_type=RopeType.DEFAULT, rope_theta=10_000.0
            ),
            LayerType.FULL: RopeParameters(
                rope_type=RopeType.PROPORTIONAL,
                rope_theta=1_000_000.0,
                partial_rotary_factor=0.25,
            ),
        }
    )

    # Per-Layer Embeddings (PLE).
    vocab_size_per_layer_input: int = 262_144
    hidden_size_per_layer_input: int = 256

    # Global (full) attention layers use a wider head dim with p-RoPE.
    global_head_dim: int = 512
    num_global_key_value_heads: int | None = None
    attention_k_eq_v: bool = False

    # The last `num_kv_shared_layers` layers reuse the K/V states of the last
    # non-shared layer of the same layer type.
    num_kv_shared_layers: int = 20
    # KV-shared layers use MLPs with twice the intermediate size.
    use_double_wide_mlp: bool = True

    # MoE (26B-A4B only, not implemented).
    enable_moe_block: bool = False

    attn_backend: AttentionBackend = AttentionBackend.SDPA

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for "
                f"{self.num_hidden_layers} layers"
            )
        if self.layer_types[-1] is not LayerType.FULL:
            raise ValueError("last layer must be full_attention")
        if self.enable_moe_block:
            raise NotImplementedError("MoE blocks (26B-A4B) are not implemented")
        if self.use_bidirectional_attention is not None:
            raise NotImplementedError(
                "use_bidirectional_attention is not implemented (E2B uses None)"
            )

    # -- derived structure ---------------------------------------------------

    @property
    def first_kv_shared_layer_idx(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    def is_kv_shared_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_kv_shared_layer_idx > 0

    def is_kv_source_layer(self, layer_idx: int) -> bool:
        """Whether this layer's K/V states are reused by the KV-shared layers.

        True for the last layer of each layer type before the shared region.
        """
        if self.num_kv_shared_layers <= 0 or self.is_kv_shared_layer(layer_idx):
            return False
        prefix = self.layer_types[: self.first_kv_shared_layer_idx]
        layer_type = self.layer_types[layer_idx]
        last_idx = len(prefix) - 1 - prefix[::-1].index(layer_type)
        return layer_idx == last_idx

    def head_dim_for_layer(self, layer_idx: int) -> int:
        if self.layer_types[layer_idx] is LayerType.FULL and self.global_head_dim:
            return self.global_head_dim
        return self.head_dim

    def head_dim_for_type(self, layer_type: LayerType) -> int:
        if layer_type is LayerType.FULL and self.global_head_dim:
            return self.global_head_dim
        return self.head_dim

    def intermediate_size_for_layer(self, layer_idx: int) -> int:
        double = self.use_double_wide_mlp and self.is_kv_shared_layer(layer_idx)
        return self.intermediate_size * (2 if double else 1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        num_layers = int(data.get("num_hidden_layers", 35))
        if "layer_types" in data:
            layer_types = tuple(LayerType(t) for t in data["layer_types"])
        else:
            layer_types = _default_layer_types(num_layers)
        rope_data = data.get("rope_parameters")
        if rope_data is not None:
            rope_parameters = {
                LayerType(t): RopeParameters.from_dict(params)
                for t, params in rope_data.items()
            }
        else:
            rope_parameters = cls().rope_parameters
        eos = data.get("eos_token_id", 1)
        eos_token_ids = tuple(eos) if isinstance(eos, list) else (int(eos),)
        return cls(
            vocab_size=int(data.get("vocab_size", 262_144)),
            hidden_size=int(data.get("hidden_size", 1536)),
            intermediate_size=int(data.get("intermediate_size", 6144)),
            num_hidden_layers=num_layers,
            num_attention_heads=int(data.get("num_attention_heads", 8)),
            num_key_value_heads=int(data.get("num_key_value_heads", 1)),
            head_dim=int(data.get("head_dim", 256)),
            hidden_activation=str(data.get("hidden_activation", "gelu_pytorch_tanh")),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            pad_token_id=int(data.get("pad_token_id", 0)),
            eos_token_ids=eos_token_ids,
            bos_token_id=int(data.get("bos_token_id", 2)),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
            attention_bias=bool(data.get("attention_bias", False)),
            sliding_window=int(data.get("sliding_window", 512)),
            layer_types=layer_types,
            final_logit_softcapping=(
                float(cap)
                if (cap := data.get("final_logit_softcapping")) is not None
                else None
            ),
            use_bidirectional_attention=data.get("use_bidirectional_attention"),
            rope_parameters=rope_parameters,
            vocab_size_per_layer_input=int(
                data.get("vocab_size_per_layer_input", 262_144)
            ),
            hidden_size_per_layer_input=int(
                data.get("hidden_size_per_layer_input", 256)
            ),
            global_head_dim=int(data.get("global_head_dim", 512)),
            num_global_key_value_heads=data.get("num_global_key_value_heads"),
            attention_k_eq_v=bool(data.get("attention_k_eq_v", False)),
            num_kv_shared_layers=int(data.get("num_kv_shared_layers", 0)),
            use_double_wide_mlp=bool(data.get("use_double_wide_mlp", False)),
            enable_moe_block=bool(data.get("enable_moe_block", False)),
        )


@dataclass(frozen=True, slots=True)
class Gemma4VisionConfig:
    """Encoder-free vision tower config. Defaults correspond to Gemma 4 E2B."""

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100.0
    pooling_kernel_size: int = 3
    patch_size: int = 16
    position_embedding_size: int = 10_240
    use_clipped_linears: bool = True
    standardize: bool = False

    attn_backend: AttentionBackend = AttentionBackend.SDPA

    def __post_init__(self) -> None:
        if self.standardize:
            raise NotImplementedError("standardize=True is not implemented")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        rope_data = data.get("rope_parameters") or {}
        return cls(
            hidden_size=int(data.get("hidden_size", 768)),
            intermediate_size=int(data.get("intermediate_size", 3072)),
            num_hidden_layers=int(data.get("num_hidden_layers", 16)),
            num_attention_heads=int(data.get("num_attention_heads", 12)),
            num_key_value_heads=int(data.get("num_key_value_heads", 12)),
            head_dim=int(data.get("head_dim", 64)),
            hidden_activation=str(data.get("hidden_activation", "gelu_pytorch_tanh")),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope_data.get("rope_theta", 100.0)),
            pooling_kernel_size=int(data.get("pooling_kernel_size", 3)),
            patch_size=int(data.get("patch_size", 16)),
            position_embedding_size=int(data.get("position_embedding_size", 10_240)),
            use_clipped_linears=bool(data.get("use_clipped_linears", False)),
            standardize=bool(data.get("standardize", False)),
        )


@dataclass(frozen=True, slots=True)
class Gemma4Config:
    """Top-level multimodal config (text + vision; the audio tower is not
    implemented and its checkpoint weights are ignored)."""

    text: Gemma4TextConfig = field(default_factory=Gemma4TextConfig)
    vision: Gemma4VisionConfig | None = field(default_factory=Gemma4VisionConfig)
    image_token_id: int = 258_880
    video_token_id: int = 258_884
    audio_token_id: int = 258_881
    boi_token_id: int = 255_999
    eoi_token_id: int = 258_882
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if data.get("model_type") not in (None, "gemma4"):
            raise ValueError(
                f"not a gemma4 config: model_type={data.get('model_type')}"
            )
        vision_data = data.get("vision_config")
        dtype_str = data.get("dtype", "bfloat16")
        dtype = getattr(torch, dtype_str)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"invalid dtype {dtype_str!r}")
        return cls(
            text=Gemma4TextConfig.from_dict(data["text_config"]),
            vision=(
                Gemma4VisionConfig.from_dict(vision_data)
                if vision_data is not None
                else None
            ),
            image_token_id=int(data.get("image_token_id", 258_880)),
            video_token_id=int(data.get("video_token_id", 258_884)),
            audio_token_id=int(data.get("audio_token_id", 258_881)),
            boi_token_id=int(data.get("boi_token_id", 255_999)),
            eoi_token_id=int(data.get("eoi_token_id", 258_882)),
            dtype=dtype,
        )

    @classmethod
    def from_json(cls, path: Path | str) -> Self:
        with open(path) as f:
            return cls.from_dict(json.load(f))
