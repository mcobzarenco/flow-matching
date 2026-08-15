"""Typed configuration for the pure-torch Molmo2 implementation.

Mirrors the fields of the HF ``Molmo2Config`` family that this
implementation actually reads, parsed from a checkpoint's ``config.json``
(``text_config`` for the decoder). Facts pinned from the fetched primary
sources in ``docs/molmo2.md`` — the 4B SKU is stock Qwen3-4B geometry:
36 uniform full-attention layers, GQA 32:8 over head_dim 128, qwen3-style
qk-norm, plain RoPE theta 5e6, untied embeddings with a separate 128-slot
extension matrix.

Config dataclasses carry no defaults: they describe a specific checkpoint
architecture and every field must be explicit (the gemma4 convention).
Features the 4B SKU does not use (per-layer RoPE scaling, post-norm layers,
QKV biases, non-qwen3 qk-norm, dropout) are refused loudly at parse time
rather than half-implemented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch

__all__ = [
    "Molmo2Config",
    "Molmo2TextConfig",
]


@dataclass(frozen=True, slots=True)
class Molmo2TextConfig:
    """Decoder architecture. See :meth:`molmo2_4b` for the released
    Molmo2-4B values."""

    vocab_size: int
    # Separate extension embedding matrix (``wte.new_embedding``); ids in
    # [vocab_size, vocab_size + additional_vocab_size) index into it.
    additional_vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    layer_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads {self.num_attention_heads} not divisible "
                f"by num_key_value_heads {self.num_key_value_heads}",
            )
        if self.tie_word_embeddings:
            raise NotImplementedError(
                "tied embeddings are not implemented (Molmo2 unties them)",
            )

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def total_vocab_size(self) -> int:
        return self.vocab_size + self.additional_vocab_size

    @property
    def fast_block_base(self) -> int:
        """First backbone id of the FAST action block: the 1,026 FAST ids (1,024 BPE + BOA + PAD) do NOT fit
        Qwen3's ~271-id unused tail, so they anchor as a SECOND extension
        block directly after the image specials — ids
        [vocab + additional, vocab + additional + vocab_total), i.e.
        [152,064, 153,090) for the 4B SKU. The trunk carries no rows for
        them: the embedding rows and the fresh untied head rows are
        decoder-owned trainable parameters (Molmo2's own new_embedding
        pattern), while ``wte`` and the shipped ``lm_head`` stay frozen."""
        return self.total_vocab_size

    @staticmethod
    def molmo2_4b() -> Molmo2TextConfig:
        """The ``allenai/Molmo2-4B`` decoder architecture (matches its
        config.json — stock Qwen3-4B geometry)."""
        return Molmo2TextConfig(
            vocab_size=151_936,
            additional_vocab_size=128,
            hidden_size=2560,
            intermediate_size=9728,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            hidden_act="silu",
            layer_norm_eps=1e-6,
            rope_theta=5_000_000.0,
            tie_word_embeddings=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, tie_word_embeddings: bool) -> Self:
        """Parse an HF ``text_config`` dict (``model_type: molmo2_text``).

        ``tie_word_embeddings`` lives on the TOP-LEVEL Molmo2 config, not in
        ``text_config`` — the caller passes it down.
        """
        # molmoact2_text is the same decoder config nested in MolmoAct2
        # checkpoints (see the top-level from_dict note).
        if data.get("model_type") not in (None, "molmo2_text", "molmoact2_text"):
            raise ValueError(
                f"not a molmo2 text config: model_type={data.get('model_type')}",
            )
        if data.get("rope_scaling") is not None:
            raise NotImplementedError(
                "rope_scaling is not implemented (null in the 4B SKU)",
            )
        if data.get("rope_scaling_layers") is not None:
            raise NotImplementedError(
                "rope_scaling_layers (per-layer dynamic RoPE) is not "
                "implemented (null in the 4B SKU)",
            )
        if data.get("norm_after", False):
            raise NotImplementedError(
                "norm_after (post-norm layers) is not implemented "
                "(the 4B SKU is pre-norm)",
            )
        if not data.get("use_qk_norm", False) or data.get("qk_norm_type") != "qwen3":
            raise NotImplementedError(
                "only qwen3-style qk-norm (per-head RMSNorm before RoPE) is "
                f"implemented, got use_qk_norm={data.get('use_qk_norm')} "
                f"qk_norm_type={data.get('qk_norm_type')!r}",
            )
        if data.get("qkv_bias", False):
            raise NotImplementedError("qkv_bias is not implemented")
        for dropout_key in (
            "attention_dropout",
            "embedding_dropout",
            "residual_dropout",
        ):
            if float(data.get(dropout_key, 0.0)) != 0.0:
                raise NotImplementedError(f"{dropout_key} != 0 is not implemented")
        return cls(
            vocab_size=int(data["vocab_size"]),
            additional_vocab_size=int(data["additional_vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=int(data["head_dim"]),
            hidden_act=str(data["hidden_act"]),
            layer_norm_eps=float(data["layer_norm_eps"]),
            rope_theta=float(data["rope_theta"]),
            tie_word_embeddings=tie_word_embeddings,
        )


def _refuse_dropout(data: dict[str, Any], *keys: str) -> None:
    for key in keys:
        if float(data.get(key, 0.0)) != 0.0:
            raise NotImplementedError(f"{key} != 0 is not implemented")


@dataclass(frozen=True, slots=True)
class Molmo2VitConfig:
    """SigLIP-so400m-class vision tower (``vit_config``). The released
    checkpoint ships the tower already truncated to the deepest adapter tap
    (25 of 27 blocks); ``num_hidden_layers`` here is the ARCHITECTURAL
    depth — the backbone truncates at build time exactly like the
    reference."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    layer_norm_eps: float
    image_patch_size: int
    image_num_pos: int
    float32_attention: bool

    @property
    def patch_dim(self) -> int:
        """Flattened pixel count per patch (the patch-embedding input)."""
        return self.image_patch_size * self.image_patch_size * 3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        _refuse_dropout(data, "attention_dropout", "residual_dropout")
        return cls(
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=int(data["head_dim"]),
            hidden_act=str(data["hidden_act"]),
            layer_norm_eps=float(data["layer_norm_eps"]),
            image_patch_size=int(data["image_patch_size"]),
            image_num_pos=int(data["image_num_pos"]),
            float32_attention=bool(data["float32_attention"]),
        )


@dataclass(frozen=True, slots=True)
class Molmo2AdapterConfig:
    """Vision->text connector (``adapter_config``): 2x2 attention pooling
    over concatenated tower taps, then a gated MLP into text hidden."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    text_hidden_size: int
    vit_layers: tuple[int, ...]
    float32_attention: bool
    pooling_attention_mask: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        _refuse_dropout(data, "attention_dropout", "residual_dropout")
        if float(data.get("image_feature_dropout", 0.0)) != 0.0:
            raise NotImplementedError("image_feature_dropout != 0 is not implemented")
        return cls(
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=int(data["head_dim"]),
            hidden_act=str(data["hidden_act"]),
            text_hidden_size=int(data["text_hidden_size"]),
            vit_layers=tuple(int(i) for i in data["vit_layers"]),
            float32_attention=bool(data["float32_attention"]),
            pooling_attention_mask=bool(data["pooling_attention_mask"]),
        )


@dataclass(frozen=True, slots=True)
class Molmo2Config:
    """Top-level Molmo2 checkpoint config. WP1 consumes the text decoder;
    WP2 adds the vision tower + connector and the image-patch id (the
    scatter target). Prompt assembly ids land with WP3."""

    text: Molmo2TextConfig
    vit: Molmo2VitConfig | None
    adapter: Molmo2AdapterConfig | None
    image_patch_id: int
    dtype: torch.dtype

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        # "molmoact2" checkpoints ARE Molmo2 backbones (same nested
        # text/vit/adapter sections) plus an action expert the molmoact2
        # package loads separately — the trunk config parses here as-is.
        if data.get("model_type") not in (None, "molmo2", "molmoact2"):
            raise ValueError(
                f"not a molmo2 config: model_type={data.get('model_type')}",
            )
        dtype = getattr(torch, data["dtype"])
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"invalid dtype {data['dtype']!r}")
        vit_data = data.get("vit_config")
        adapter_data = data.get("adapter_config")
        # Treat a config as vision-less unless BOTH sections are
        # substantive (a placeholder vit_config with no fields is allowed).
        vit = None
        adapter = None
        if vit_data and "hidden_size" in vit_data and adapter_data:
            vit = Molmo2VitConfig.from_dict(vit_data)
            adapter = Molmo2AdapterConfig.from_dict(adapter_data)
        return cls(
            text=Molmo2TextConfig.from_dict(
                data["text_config"],
                tie_word_embeddings=bool(data["tie_word_embeddings"]),
            ),
            vit=vit,
            adapter=adapter,
            image_patch_id=int(data.get("image_patch_id", -1)),
            dtype=dtype,
        )

    @classmethod
    def from_json(cls, path: Path | str) -> Self:
        with Path(path).open() as f:
            return cls.from_dict(json.load(f))
