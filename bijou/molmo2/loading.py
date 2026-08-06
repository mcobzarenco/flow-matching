"""Checkpoint loading: HF safetensors -> pure-torch Molmo2 text model.

The module tree deliberately mirrors HF's names, so the mapping is mostly
"strip the ``model.`` prefix" (``lm_head.weight`` sits outside ``model.`` in
the checkpoint and outside ``transformer`` here). Checkpoint quirks handled:

- vision tower / connector weights (``model.vision_backbone.*``) are skipped
  until WP2 implements them,
- the released checkpoint ships fp32; pass ``dtype=torch.bfloat16`` to cast
  at load (matching how the mount will run),
- with ``truncate_layers=N`` only decoder blocks ``0..N-1`` are instantiated
  and loaded — the 15-layer mount of the port plan (D2). Unlike gemma4 there
  are no KV-sharing or layer-type constraints: every layer is a legal
  truncation point. The truncated model is built WITHOUT the LM head (the
  mount only exports residual taps); a full-depth load includes it for the
  parity harness.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import torch
from safetensors import safe_open

from ..gemma4.loading import resolve_checkpoint_dir
from ..nn import DEFAULT_ATTENTION_BACKEND, AttentionBackend
from .config import Molmo2Config, Molmo2TextConfig
from .text import Molmo2TextModel

_VISION_PREFIX = "model.vision_backbone."
_LM_HEAD_KEY = "lm_head.weight"
_BLOCK_RE = re.compile(r"^transformer\.blocks\.(\d+)\.")


def load_config(checkpoint_dir: Path) -> Molmo2Config:
    return Molmo2Config.from_json(checkpoint_dir / "config.json")


def truncated_config(config: Molmo2TextConfig, num_layers: int) -> Molmo2TextConfig:
    """Config for the first ``num_layers`` decoder blocks of a checkpoint.

    All Molmo2 layers are identical full-attention blocks, so any depth in
    ``(0, num_hidden_layers]`` is valid (contrast the gemma4 constraints)."""
    if not 0 < num_layers <= config.num_hidden_layers:
        raise ValueError(
            f"num_layers must be in (0, {config.num_hidden_layers}], got {num_layers}",
        )
    return dataclasses.replace(config, num_hidden_layers=num_layers)


def _is_truncated_block_key(key: str, num_layers: int) -> bool:
    match = _BLOCK_RE.match(key)
    return match is not None and int(match.group(1)) >= num_layers


def load_text_model(
    model_id_or_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
    truncate_layers: int | None = None,
) -> Molmo2TextModel:
    """Load the text decoder of a Molmo2 checkpoint, materializing weights
    directly on ``device``.

    ``dtype`` overrides the checkpoint's dtype (the release ships fp32 —
    pass ``torch.bfloat16`` for the mount). The model is returned in eval
    mode with gradients disabled.
    """
    checkpoint_dir = resolve_checkpoint_dir(model_id_or_path)
    config = load_config(checkpoint_dir)
    if dtype is None:
        dtype = config.dtype
    text_config = config.text
    if truncate_layers is not None:
        text_config = truncated_config(text_config, truncate_layers)
    with_lm_head = truncate_layers is None

    model = Molmo2TextModel(
        text_config,
        lm_head=with_lm_head,
        attn_backend=attn_backend,
        device="meta",
    )

    state_dict: dict[str, torch.Tensor] = {}
    weight_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no *.safetensors files in {checkpoint_dir}")
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device=str(device)) as f:
            # it exposes .keys() but not iteration/__contains__.
            for key in f.keys():  # noqa: SIM118
                if key.startswith(_VISION_PREFIX):
                    continue
                if key == _LM_HEAD_KEY:
                    if not with_lm_head:
                        continue
                    name = key
                else:
                    name = key.removeprefix("model.")
                    if truncate_layers is not None and _is_truncated_block_key(
                        name,
                        truncate_layers,
                    ):
                        continue
                tensor = f.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                state_dict[name] = tensor

    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    model.requires_grad_(False)
    # Parameters are already on the target device; this sweeps over the
    # small computed buffers (rope inv_freq), which meta construction
    # materializes on CPU (see bijou.nn.buffer_device).
    return model.to(device)
