"""Jimmy: a vision-language-action model built on a Gemma 4 backbone.

Package layout:

- ``jimmy.gemma4`` — hackable pure-torch Gemma 4 (E2B) implementation,
  verified against the HF reference (``python -m jimmy.gemma4.verify_parity``).
- VLA-specific components (action expert, policy wrapper, lerobot
  integration) will live here as they are built.
"""
