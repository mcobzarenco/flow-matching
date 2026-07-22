"""Bijou: a vision-language-action model built on a Gemma 4 backbone.

*Bijou* (French: jewel; idiomatically "small but elegant") — a nod to the
Gemma lineage and to the model's on-device ambitions.

Package layout:

- ``bijou.gemma4`` — hackable pure-torch Gemma 4 (E2B) implementation,
  verified against the HF reference (``python -m bijou.gemma4.verify_parity``).
- VLA-specific components (action expert, policy wrapper, lerobot
  integration) will live here as they are built.
"""
