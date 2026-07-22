"""Hackable pure-torch implementation of Gemma 4 (E2B).

Loads HF checkpoints (``google/gemma-4-e2b-it``) and reproduces the reference
transformers implementation bit-exactly in bf16 (eager attention). Text and
vision towers are implemented; the audio tower is not.

Quick start::

    from gemma4 import load_model, generate

    model = load_model("google/gemma-4-e2b-it")
    result = generate(model, input_ids, max_new_tokens=32)
"""

from .cache import KVCache
from .config import (
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    LayerType,
    RopeParameters,
    RopeType,
)
from .generation import GenerationResult, SamplingParams, generate
from .loading import load_config, load_model, resolve_checkpoint_dir
from .model import Gemma4Model, Gemma4Output, build_model
from .text import TextModel
from .vision import VisionModel

__all__ = [
    "Gemma4Config",
    "Gemma4Model",
    "Gemma4Output",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
    "GenerationResult",
    "KVCache",
    "LayerType",
    "RopeParameters",
    "RopeType",
    "SamplingParams",
    "TextModel",
    "VisionModel",
    "build_model",
    "generate",
    "load_config",
    "load_model",
    "resolve_checkpoint_dir",
]
