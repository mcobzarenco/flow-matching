"""Typed configuration for the pure-torch Gemma 4 implementation.

Mirrors the fields of the HF ``Gemma4Config`` family that are relevant for
inference, parsed from a checkpoint's ``config.json``. Only fields that this
implementation actually reads are kept — if you add a feature, add its config
field here.

Config dataclasses carry no defaults: they describe a specific checkpoint
architecture and every field must be explicit. In practice configs are read
from disk (:meth:`Gemma4Config.from_json` via ``load_model``); for
from-scratch construction and tests, :meth:`Gemma4Config.e2b` and
:meth:`Gemma4Config.e4b` build the released E-series architectures in code
(released shapes live as staticmethods on the config, per the styleguide).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import torch

# Re-exports: rope geometry types were lifted to bijou.modelling.nn (they parameterize
# the shared rope primitives); gemma4 configs keep referencing them here.
from ..nn import RopeParameters, RopeType

__all__ = [
    "Gemma4Config",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
    "LayerType",
    "RopeParameters",
    "RopeType",
]


class LayerType(StrEnum):
    """Attention pattern of a decoder layer."""

    SLIDING = "sliding_attention"
    FULL = "full_attention"


@dataclass(frozen=True, slots=True)
class Gemma4TextConfig:
    """Decoder architecture. See :meth:`Gemma4Config.e2b` /
    :meth:`Gemma4Config.e4b` for the released E-series values."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_activation: str
    rms_norm_eps: float
    pad_token_id: int
    eos_token_ids: tuple[int, ...]
    bos_token_id: int
    tie_word_embeddings: bool
    attention_bias: bool
    sliding_window: int
    layer_types: tuple[LayerType, ...]
    final_logit_softcapping: float | None
    # None => plain causal; "vision" (12B) => bidirectional attention within
    # each image block; "all" => fully bidirectional.
    use_bidirectional_attention: str | None
    rope_parameters: dict[LayerType, RopeParameters]

    # Per-Layer Embeddings (PLE).
    vocab_size_per_layer_input: int
    hidden_size_per_layer_input: int

    # Global (full) attention layers use a wider head dim with p-RoPE.
    global_head_dim: int
    num_global_key_value_heads: int | None
    attention_k_eq_v: bool

    # The last `num_kv_shared_layers` layers reuse the K/V states of the last
    # non-shared layer of the same layer type.
    num_kv_shared_layers: int
    # KV-shared layers use MLPs with twice the intermediate size (E2B only).
    use_double_wide_mlp: bool

    # MoE (26B-A4B only, not implemented).
    enable_moe_block: bool

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for "
                f"{self.num_hidden_layers} layers",
            )
        if self.layer_types[-1] is not LayerType.FULL:
            raise ValueError("last layer must be full_attention")
        if self.enable_moe_block:
            raise NotImplementedError("MoE blocks (26B-A4B) are not implemented")
        if self.use_bidirectional_attention is not None:
            raise NotImplementedError(
                "use_bidirectional_attention is not implemented "
                "(the E-series models use None)",
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
        eos = data["eos_token_id"]
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=int(data["head_dim"]),
            hidden_activation=str(data["hidden_activation"]),
            rms_norm_eps=float(data["rms_norm_eps"]),
            pad_token_id=int(data["pad_token_id"]),
            eos_token_ids=tuple(eos) if isinstance(eos, list) else (int(eos),),
            bos_token_id=int(data["bos_token_id"]),
            tie_word_embeddings=bool(data["tie_word_embeddings"]),
            attention_bias=bool(data["attention_bias"]),
            sliding_window=int(data["sliding_window"]),
            layer_types=tuple(LayerType(t) for t in data["layer_types"]),
            final_logit_softcapping=(
                float(cap)
                if (cap := data.get("final_logit_softcapping")) is not None
                else None
            ),
            use_bidirectional_attention=data.get("use_bidirectional_attention"),
            rope_parameters={
                LayerType(t): RopeParameters.from_dict(params)
                for t, params in data["rope_parameters"].items()
            },
            vocab_size_per_layer_input=int(data["vocab_size_per_layer_input"]),
            hidden_size_per_layer_input=int(data["hidden_size_per_layer_input"]),
            global_head_dim=int(data["global_head_dim"]),
            num_global_key_value_heads=(
                int(heads)
                if (heads := data.get("num_global_key_value_heads")) is not None
                else None
            ),
            attention_k_eq_v=bool(data["attention_k_eq_v"]),
            num_kv_shared_layers=int(data["num_kv_shared_layers"]),
            use_double_wide_mlp=bool(data["use_double_wide_mlp"]),
            enable_moe_block=bool(data["enable_moe_block"]),
        )


@dataclass(frozen=True, slots=True)
class Gemma4VisionConfig:
    """Encoder-free vision tower architecture (identical across E2B/E4B)."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_activation: str
    rms_norm_eps: float
    rope_theta: float
    pooling_kernel_size: int
    patch_size: int
    position_embedding_size: int
    use_clipped_linears: bool
    standardize: bool

    def __post_init__(self) -> None:
        if self.standardize:
            raise NotImplementedError("standardize=True is not implemented")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=int(data["head_dim"]),
            hidden_activation=str(data["hidden_activation"]),
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data["rope_parameters"]["rope_theta"]),
            pooling_kernel_size=int(data["pooling_kernel_size"]),
            patch_size=int(data["patch_size"]),
            position_embedding_size=int(data["position_embedding_size"]),
            use_clipped_linears=bool(data["use_clipped_linears"]),
            standardize=bool(data["standardize"]),
        )


@dataclass(frozen=True, slots=True)
class Gemma4Config:
    """Top-level multimodal config (text + vision; the audio tower is not
    implemented and its checkpoint weights are ignored)."""

    text: Gemma4TextConfig
    vision: Gemma4VisionConfig | None
    image_token_id: int
    video_token_id: int
    audio_token_id: int
    boi_token_id: int
    eoi_token_id: int
    dtype: torch.dtype

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if data.get("model_type") not in (None, "gemma4"):
            raise ValueError(
                f"not a gemma4 config: model_type={data.get('model_type')}",
            )
        vision_data = data.get("vision_config")
        dtype = getattr(torch, data["dtype"])
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"invalid dtype {data['dtype']!r}")
        return cls(
            text=Gemma4TextConfig.from_dict(data["text_config"]),
            vision=(
                Gemma4VisionConfig.from_dict(vision_data)
                if vision_data is not None
                else None
            ),
            image_token_id=int(data["image_token_id"]),
            video_token_id=int(data["video_token_id"]),
            audio_token_id=int(data["audio_token_id"]),
            boi_token_id=int(data["boi_token_id"]),
            eoi_token_id=int(data["eoi_token_id"]),
            dtype=dtype,
        )

    @classmethod
    def from_json(cls, path: Path | str) -> Self:
        with Path(path).open() as f:
            return cls.from_dict(json.load(f))

    @staticmethod
    def e2b() -> Gemma4Config:
        """The ``google/gemma-4-e2b-it`` architecture (matches its
        config.json)."""
        return _e_series_config(
            Gemma4TextConfig(
                vocab_size=262_144,
                hidden_size=1536,
                intermediate_size=6144,
                num_hidden_layers=35,
                num_attention_heads=8,
                num_key_value_heads=1,
                head_dim=256,
                hidden_activation="gelu_pytorch_tanh",
                rms_norm_eps=1e-6,
                pad_token_id=0,
                eos_token_ids=(1,),
                bos_token_id=2,
                tie_word_embeddings=True,
                attention_bias=False,
                sliding_window=512,
                layer_types=_hybrid_layer_types(35, period=5),
                final_logit_softcapping=30.0,
                use_bidirectional_attention=None,
                rope_parameters=_e_series_rope_parameters(),
                vocab_size_per_layer_input=262_144,
                hidden_size_per_layer_input=256,
                global_head_dim=512,
                num_global_key_value_heads=None,
                attention_k_eq_v=False,
                num_kv_shared_layers=20,
                use_double_wide_mlp=True,
                enable_moe_block=False,
            ),
        )

    @staticmethod
    def e4b() -> Gemma4Config:
        """The ``google/gemma-4-e4b-it`` architecture (matches its
        config.json)."""
        return _e_series_config(
            Gemma4TextConfig(
                vocab_size=262_144,
                hidden_size=2560,
                intermediate_size=10_240,
                num_hidden_layers=42,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=256,
                hidden_activation="gelu_pytorch_tanh",
                rms_norm_eps=1e-6,
                pad_token_id=0,
                eos_token_ids=(1,),
                bos_token_id=2,
                tie_word_embeddings=True,
                attention_bias=False,
                sliding_window=512,
                layer_types=_hybrid_layer_types(42, period=6),
                final_logit_softcapping=30.0,
                use_bidirectional_attention=None,
                rope_parameters=_e_series_rope_parameters(),
                vocab_size_per_layer_input=262_144,
                hidden_size_per_layer_input=256,
                global_head_dim=512,
                num_global_key_value_heads=None,
                attention_k_eq_v=False,
                num_kv_shared_layers=18,
                use_double_wide_mlp=False,
                enable_moe_block=False,
            ),
        )


# -- released architectures (private assembly helpers) ------------------------


def _hybrid_layer_types(num_layers: int, period: int) -> tuple[LayerType, ...]:
    """Every ``period``-th layer is full attention, the rest sliding.

    Note the released E-series differ here: E2B uses period 5 (4:1) while E4B
    uses period 6 (5:1, the transformers default).
    """
    return tuple(
        LayerType.SLIDING if (i + 1) % period else LayerType.FULL
        for i in range(num_layers)
    )


def _e_series_rope_parameters() -> dict[LayerType, RopeParameters]:
    return {
        LayerType.SLIDING: RopeParameters(
            rope_type=RopeType.DEFAULT,
            rope_theta=10_000.0,
            factor=1.0,
            partial_rotary_factor=1.0,
        ),
        LayerType.FULL: RopeParameters(
            rope_type=RopeType.PROPORTIONAL,
            rope_theta=1_000_000.0,
            factor=1.0,
            partial_rotary_factor=0.25,
        ),
    }


def _e_series_vision_config() -> Gemma4VisionConfig:
    return Gemma4VisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=16,
        num_attention_heads=12,
        num_key_value_heads=12,
        head_dim=64,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        rope_theta=100.0,
        pooling_kernel_size=3,
        patch_size=16,
        position_embedding_size=10_240,
        use_clipped_linears=True,
        standardize=False,
    )


def _e_series_config(text: Gemma4TextConfig) -> Gemma4Config:
    return Gemma4Config(
        text=text,
        vision=_e_series_vision_config(),
        image_token_id=258_880,
        video_token_id=258_884,
        audio_token_id=258_881,
        boi_token_id=255_999,
        eoi_token_id=258_882,
        dtype=torch.bfloat16,
    )
