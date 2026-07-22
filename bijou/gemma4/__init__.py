"""Hackable pure-torch implementation of the Gemma 4 E-series (E2B, E4B).

Loads HF checkpoints (``google/gemma-4-e2b-it``, ``google/gemma-4-e4b-it``)
and reproduces the reference transformers implementation bit-exactly in bf16
(eager attention). Text and vision towers are implemented; the audio tower is
not. Architectures are fully config-driven (:func:`e2b_config` /
:func:`e4b_config` build them in code for from-scratch use).

Correctness is checked with ``python -m bijou.gemma4.verify_parity``: greedy
tokens must match HF exactly and logits must agree within a small tolerance
(bf16-ULP-scale differences are expected and acceptable — we do not promise
bitwise equality, leaving room to optimize kernels; on H100 with eager
attention the port happens to be bitwise-identical today).

Quick start::

    from gemma4 import load_model, generate

    model = load_model("google/gemma-4-e2b-it", device="cuda")
    result = generate(model, input_ids, max_new_tokens=32)

Every module takes explicit ``device``/``dtype`` factory arguments, so
from-scratch components (e.g. a VLA action expert) can be built directly on
the target device, and different submodules may use different dtypes:
``Gemma4Model(config, device="cuda", dtype=torch.bfloat16)``.
"""

from .cache import KVCache
from .config import (
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    LayerType,
    RopeParameters,
    RopeType,
    e2b_config,
    e4b_config,
)
from .generation import GenerationResult, SamplingParams, generate
from .layers import AttentionBackend
from .loading import load_config, load_model, resolve_checkpoint_dir
from .model import Gemma4Model, Gemma4Output, set_attention_backend
from .text import TextModel
from .vision import VisionModel

__all__ = [
    "AttentionBackend",
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
    "e2b_config",
    "e4b_config",
    "generate",
    "load_config",
    "load_model",
    "resolve_checkpoint_dir",
    "set_attention_backend",
]
