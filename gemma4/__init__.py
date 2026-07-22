"""Hackable pure-torch implementation of Gemma 4 (E2B).

Loads HF checkpoints (``google/gemma-4-e2b-it``) and reproduces the reference
transformers implementation bit-exactly in bf16 (eager attention). Text and
vision towers are implemented; the audio tower is not.

Verified with ``python -m gemma4.verify_parity`` on H100 (cuda): prefill
logits, cached decode across the 512-token sliding window, greedy generate()
tokens and the image path are all bitwise-identical to HF and run-to-run
stable. Caveat: on CPU, *cached decode* comparisons can differ run-to-run
because oneDNN bf16 gemms in the HF reference are themselves nondeterministic
there (this implementation is self-deterministic on both backends; prefill
matches bitwise everywhere).

Quick start::

    from gemma4 import load_model, generate

    model = load_model("google/gemma-4-e2b-it", device="cuda")
    result = generate(model, input_ids, max_new_tokens=32)

From-scratch construction (e.g. future VLA components) on a target
device/dtype: ``build_model(config, device="cuda", dtype=torch.bfloat16)``.
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
