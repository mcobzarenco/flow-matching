"""Bijou: a vision-language-action model built on a Gemma 4 backbone.

*Bijou* (French: jewel; idiomatically "small but elegant") — a nod to the
Gemma lineage and to the model's on-device ambitions.

Package layout:

- ``bijou.modelling.gemma4`` — hackable pure-torch Gemma 4 E-series (E2B, E4B)
  implementation, verified against the HF reference
  (``python -m bijou.modelling.gemma4.verify_parity``).
- ``bijou.modelling.encoders`` / ``bijou.modelling.decoders`` / ``bijou.model`` / ``bijou.loading``
  — the VLA: a frozen, truncated backbone encodes the multimodal prefix once
  per observation and exports its global-attention K/V streams as an
  ObservationMemory; action decoders (the flow-matching expert, the AR FAST
  baseline) cross-attend it.

Quick start::

    from bijou import from_backbone

    model = from_backbone("google/gemma-4-e2b-it", action_dim=6, state_dim=6,
                          device="cuda")
    prefix = model.encode_observation(input_ids, pixel_values=..., image_position_ids=...)
    actions = model.sample_actions(prefix, state)
"""

from .loading import default_expert_config, from_backbone, prefix_global_layers
from .model import BijouModel
from .modelling.decoders.flow import FlowDecoder, FlowDecoderConfig, SelfAttentionMode
from .modelling.interface import MemoryStream, ObservationMemory

__all__ = [
    "BijouModel",
    "FlowDecoder",
    "FlowDecoderConfig",
    "MemoryStream",
    "ObservationMemory",
    "SelfAttentionMode",
    "default_expert_config",
    "from_backbone",
    "prefix_global_layers",
]
