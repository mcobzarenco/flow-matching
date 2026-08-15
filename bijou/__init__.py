"""Bijou: vision-language-action model families on Gemma-4 and Molmo2
trunks.

*Bijou* (French: jewel; idiomatically "small but elegant") — a nod to the
Gemma lineage and to the model's on-device ambitions.

Package layout:

- ``bijou.modelling`` — the building blocks: hackable pure-torch trunks
  (``gemma4``, ``molmo2`` — verified against the HF references),
  per-trunk prompt-side encoders, action/text decoders, and the
  collation/memory seam (``modelling.interface``).
- ``bijou.vla`` — the trait lattice (``VLA`` and the ``ARVLA`` /
  ``FlowVLA`` / ``NarratingVLA`` capabilities) that train/eval/rollout
  program against.
- ``bijou.models`` — one concrete family class per real model,
  composing the building blocks with a typed objective payload.
- ``bijou.loading`` — the family registry; ``bijou.checkpoint`` — the
  self-contained checkpoint format it reads.

Quick start::

    from pathlib import Path

    import torch

    from bijou.loading import load_vla

    model = load_vla(Path("outputs/train/run/step_010000"),
                     device="cuda", dtype=torch.bfloat16)
    batch = model.collator()(items).to("cuda")
    actions = model.predict(batch)  # the recorded serving operating point
"""

from .loading import load_vla
from .modelling.decoders.flow import FlowDecoder, FlowDecoderConfig, SelfAttentionMode
from .modelling.encoders.gemma4 import GemmaMemory
from .modelling.interface import MemoryStream
from .vla import VLA, VLAFamily

__all__ = [
    "VLA",
    "FlowDecoder",
    "FlowDecoderConfig",
    "GemmaMemory",
    "MemoryStream",
    "SelfAttentionMode",
    "VLAFamily",
    "load_vla",
]
