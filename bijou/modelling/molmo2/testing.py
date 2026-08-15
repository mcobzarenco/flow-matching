"""Tiny random Molmo2-shaped checkpoints for CPU-only tests.

``python -m bijou.modelling.molmo2.testing --output /tmp/tiny-molmo2`` writes a
checkpoint with a miniature random text decoder that is structurally
faithful to Molmo2-4B (qwen3 qk-norm, fused QKV/gate projections, untied
embeddings with an extension matrix) and the real checkpoint's key layout
(``model.transformer.*`` + top-level ``lm_head.weight`` + a dummy
``model.vision_backbone.*`` tensor so the loader's skip path is exercised).
Outputs are meaningless; shapes and code paths are real.

No processor/tokenizer files yet — WP3 (prompt assembly) extends this
fixture with the real tokenizer the way the gemma4 fixture carries the
Gemma processor files.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .config import Molmo2Config
from .text import Molmo2TextModel
from .vision import Molmo2VisionBackbone


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One processor golden fixture: deterministic synthetic input images
    (``sizes`` are (width, height) for ``synthetic_test_image``) with
    camera kinds, at one ``max_crops`` setting. Shared by the banking
    script (reference side) and ``tests/test_molmo2_processor.py``
    (native side) so both always process identical inputs."""

    name: str
    sizes: tuple[tuple[int, int], ...]
    kinds: tuple[str, ...]
    max_crops: int


def golden_cases() -> tuple[GoldenCase, ...]:
    return (
        # The operating point: 480p rig frame, crops off (1x1 tiling).
        GoldenCase("single_640x480_mc1", ((640, 480),), ("wrist",), 1),
        # Two cameras (the rig layout): multi-image hoist + labels + the
        # per-image pooling-index offsets.
        GoldenCase(
            "two_images_mc1",
            ((640, 480), (320, 240)),
            ("wrist", "overhead"),
            1,
        ),
        # Full multi-crop tiling (2x2 at 640x480): margin masking, the
        # crop-window transpose, partial edge pooling groups.
        GoldenCase("single_640x480_mc8", ((640, 480),), ("wrist",), 8),
    )


def tiny_config_json() -> dict[str, Any]:
    """A miniature Molmo2 architecture: 6 uniform layers (deep enough to
    exercise a strictly-interior truncation), GQA 4:2 so grouped-query
    repeat is real, an extension vocab, everything else as small as it can
    sensibly be. ``vit_config`` is present but empty — WP1 parses only
    ``text_config``; WP2 fills it in."""
    return {
        "model_type": "molmo2",
        "architectures": ["Molmo2ForConditionalGeneration"],
        "dtype": "bfloat16",
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "molmo2_text",
            "vocab_size": 512,
            "additional_vocab_size": 8,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_act": "silu",
            "layer_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "use_qk_norm": True,
            "qk_norm_type": "qwen3",
            "qkv_bias": False,
            "norm_after": False,
            "rope_scaling": None,
            "rope_scaling_layers": None,
            "attention_dropout": 0.0,
            "embedding_dropout": 0.0,
            "residual_dropout": 0.0,
        },
        "vit_config": {
            "model_type": "molmo2",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "image_patch_size": 2,
            "image_num_pos": 9,
            "float32_attention": True,
            "attention_dropout": 0.0,
            "residual_dropout": 0.0,
        },
        # vit_layers (-2, -4) leaves the last block untapped, so the
        # build-time tower truncation (25-of-27 on the real checkpoint) is
        # exercised: only 3 of 4 blocks exist.
        "adapter_config": {
            "model_type": "molmo2",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_act": "silu",
            "text_hidden_size": 64,
            "vit_layers": [-2, -4],
            "float32_attention": True,
            "pooling_attention_mask": True,
            "attention_dropout": 0.0,
            "residual_dropout": 0.0,
            "image_feature_dropout": 0.0,
        },
        "image_patch_id": 514,
    }


def write_tiny_text_checkpoint(
    output_dir: Path | str,
    *,
    seed: int = 0,
    config_json: dict[str, Any] | None = None,
) -> Path:
    """Write a loadable tiny random Molmo2 checkpoint to ``output_dir``:
    tiny config.json plus random bf16 model.safetensors with the real key
    layout (keys/dtypes as shipped, including a vision key the text loader
    must skip). ``config_json`` overrides the architecture (the MolmoAct2
    AR tests widen the vocabulary to host an in-base action block);
    None = :func:`tiny_config_json`."""
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    if config_json is None:
        config_json = tiny_config_json()
    (output / "config.json").write_text(json.dumps(config_json, indent=2))

    torch.manual_seed(seed)
    config = Molmo2Config.from_dict(config_json)
    model = Molmo2TextModel(config.text, lm_head=True, device="cpu")
    model = model.to(config.dtype)
    state_dict = {
        (name if name == "lm_head.weight" else f"model.{name}"): tensor.contiguous()
        for name, tensor in model.state_dict().items()
    }
    assert config.vit is not None and config.adapter is not None
    vision = Molmo2VisionBackbone(config.vit, config.adapter, device="cpu")
    vision = vision.to(config.dtype)
    for name, tensor in vision.state_dict().items():
        state_dict[f"model.vision_backbone.{name}"] = tensor.contiguous()
    save_file(state_dict, str(output / "model.safetensors"))
    return output


def _main() -> int:
    parser = argparse.ArgumentParser(description=write_tiny_text_checkpoint.__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(write_tiny_text_checkpoint(args.output, seed=args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
