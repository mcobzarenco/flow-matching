"""Shared test utilities: synthetic images and tiny random checkpoints.

``python -m bijou.modelling.gemma4.testing --output /tmp/tiny-gemma4`` writes a
Gemma4-shaped checkpoint with a miniature random model (~100 MB, dominated by
the vocab-sized embeddings that must match the real tokenizer) plus the real
processor/tokenizer files. Point any CLI's ``--backbone`` at it for fast
plumbing tests that exercise the full load/truncate/process path without the
real weights — outputs are meaningless, shapes and code paths are real.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file

from .config import Gemma4Config
from .loading import resolve_checkpoint_dir
from .model import Gemma4Model


def synthetic_test_image(width: int = 640, height: int = 480) -> Image.Image:
    """Deterministic in-memory RGB test image (no RNG, no files).

    A per-pixel color gradient with a red ellipse and a green square --
    enough structure for the vision tower to produce a meaningful
    description. Shape placement is proportional to the requested size; at
    the default 640x480 it matches the geometry of the image used for the
    recorded H100 parity and benchmark results.
    """
    grad = np.zeros((height, width, 3), dtype=np.uint8)
    grad[..., 0] = (np.arange(width, dtype=np.uint32) * 255 // width)[None, :]
    grad[..., 1] = (np.arange(height, dtype=np.uint32) * 255 // height)[:, None]
    grad[..., 2] = 128
    image = Image.fromarray(grad)

    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (width * 5 // 16, height * 5 // 16, width * 11 // 16, height * 11 // 16),
        fill=(220, 40, 40),
    )
    draw.rectangle(
        (width * 3 // 32, height // 8, width * 9 // 32, height * 3 // 8),
        fill=(40, 200, 80),
    )
    return image


def load_test_image(path: str | None) -> tuple[Image.Image, str]:
    """(image, label): opens ``path`` if given, else the synthetic default."""
    if path is not None:
        return Image.open(path).convert("RGB"), path
    image = synthetic_test_image()
    return image, f"synthetic {image.width}x{image.height}"


# Files the Gemma4 processor/tokenizer needs, copied verbatim from the real
# checkpoint (they carry no model weights).
_PROCESSOR_FILES = (
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def tiny_config_json() -> dict[str, Any]:
    """A miniature Gemma4 architecture, structurally faithful to the E-series:
    hybrid 1:1 sliding/full layers, KV sharing over the last 2 of 8 layers
    (so checkpoint truncation and PLE slicing are exercised), p-RoPE global
    layers, tied head. The vocab and patch size must match the real
    tokenizer/processor; everything else is as small as it can sensibly be.
    The non-KV-shared prefix has three global layers (1, 3, 5), so the
    training default ``stream_counts`` of three entries works unchanged."""
    return {
        "model_type": "gemma4",
        "dtype": "bfloat16",
        "image_token_id": 258_880,
        "video_token_id": 258_884,
        "audio_token_id": 258_881,
        "boi_token_id": 255_999,
        "eoi_token_id": 258_882,
        "text_config": {
            "vocab_size": 262_144,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
            "tie_word_embeddings": True,
            "attention_bias": False,
            "sliding_window": 32,
            "layer_types": ["sliding_attention", "full_attention"] * 4,
            "final_logit_softcapping": 30.0,
            "use_bidirectional_attention": None,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
            "vocab_size_per_layer_input": 262_144,
            "hidden_size_per_layer_input": 8,
            "global_head_dim": 32,
            "num_global_key_value_heads": None,
            "attention_k_eq_v": False,
            "num_kv_shared_layers": 2,
            "use_double_wide_mlp": False,
            "enable_moe_block": False,
        },
        "vision_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_type": "default", "rope_theta": 100.0},
            "pooling_kernel_size": 3,
            "patch_size": 16,
            "position_embedding_size": 128,
            "use_clipped_linears": False,
            "standardize": False,
        },
    }


def write_tiny_checkpoint(
    output_dir: Path | str,
    *,
    processor_source: str | Path = "google/gemma-4-e2b-it",
    seed: int = 0,
) -> Path:
    """Write a loadable tiny random Gemma4 checkpoint to ``output_dir``.

    The directory works everywhere a real checkpoint does (``load_model``,
    ``from_backbone``, ``AutoProcessor.from_pretrained``): tiny config.json,
    random bf16 model.safetensors (keys/dtypes/layout as shipped by the real
    checkpoints, including the packed PLE tensors and no lm_head — tied), and
    the processor files copied from ``processor_source``.
    """
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    config_json = tiny_config_json()
    (output / "config.json").write_text(json.dumps(config_json, indent=2))

    torch.manual_seed(seed)
    config: Gemma4Config = Gemma4Config.from_dict(config_json)
    model = Gemma4Model(config, device="cpu")
    state_dict = {
        f"model.{name}": tensor.contiguous()
        for name, tensor in model.state_dict().items()
        # lm_head is tied to the embedding at load time, like the released
        # checkpoints (a fresh Gemma4Model has an independent random head).
        if name != "lm_head.weight"
    }
    save_file(state_dict, str(output / "model.safetensors"))

    source = resolve_checkpoint_dir(processor_source)
    for filename in _PROCESSOR_FILES:
        shutil.copyfile(source / filename, output / filename)
    return output


def _main() -> int:
    parser = argparse.ArgumentParser(description=write_tiny_checkpoint.__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor-source", default="google/gemma-4-e2b-it")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = write_tiny_checkpoint(
        args.output,
        processor_source=args.processor_source,
        seed=args.seed,
    )
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
