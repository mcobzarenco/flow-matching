"""Bijou: a vision-language-action model built on a Gemma 4 backbone.

*Bijou* (French: jewel; idiomatically "small but elegant") — a nod to the
Gemma lineage and to the model's on-device ambitions.

Package layout:

- ``bijou.gemma4`` — hackable pure-torch Gemma 4 E-series (E2B, E4B)
  implementation, verified against the HF reference
  (``python -m bijou.gemma4.verify_parity``).
- ``bijou.expert`` / ``bijou.model`` / ``bijou.loading`` — the VLA: a frozen,
  truncated backbone encodes the multimodal prefix once per observation and
  exports its global-attention K/V streams; a narrow flow-matching action
  expert cross-attends them to denoise action chunks.

Quick start::

    from bijou import from_backbone

    model = from_backbone("google/gemma-4-e2b-it", action_dim=6, state_dim=6,
                          device="cuda")
    prefix = model.encode_prefix(input_ids, pixel_values=..., image_position_ids=...)
    actions = model.sample_actions(prefix, state)
"""

from .decoders.flow import ExpertConfig, FlowDecoder, SelfAttentionMode
from .interface import EncodedPrefix, MemoryStream
from .loading import default_expert_config, from_backbone, prefix_global_layers
from .model import BijouModel

__all__ = [
    "BijouModel",
    "EncodedPrefix",
    "ExpertConfig",
    "FlowDecoder",
    "MemoryStream",
    "SelfAttentionMode",
    "default_expert_config",
    "from_backbone",
    "prefix_global_layers",
]
