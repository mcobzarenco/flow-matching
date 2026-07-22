"""Checkpoint loading: HF safetensors -> pure-torch Gemma4Model.

The module tree deliberately mirrors HF's names, so the mapping is mostly
"strip the ``model.`` prefix". Checkpoint quirks handled here:

- audio tower weights (``model.audio_tower.*``, ``model.embed_audio.*``) are
  skipped — the audio tower is not implemented,
- KV-shared layers ship redundant ``k_proj``/``v_proj``/``k_norm`` weights
  that HF also drops at load time,
- ``lm_head.weight`` is absent and tied to the token embedding,
- all floating-point tensors are cast to the model dtype (bf16), matching
  HF's ``from_pretrained`` behavior.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .config import AttentionBackend, Gemma4Config
from .model import Gemma4Model

_AUDIO_PREFIXES = ("model.audio_tower.", "model.embed_audio.")
_SHARED_KV_SUFFIXES = (
    "k_proj.weight",
    "v_proj.weight",
    "k_norm.weight",
    "v_norm.weight",
)
_LAYER_RE = re.compile(r"^language_model\.layers\.(\d+)\.self_attn\.(.+)$")


def resolve_checkpoint_dir(model_id_or_path: str | Path) -> Path:
    """Accepts a local directory or an HF repo id (resolved via the cache,
    downloading if necessary)."""
    path = Path(model_id_or_path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model_id_or_path)))


def load_config(checkpoint_dir: Path) -> Gemma4Config:
    return Gemma4Config.from_json(checkpoint_dir / "config.json")


def _is_dropped_shared_kv_key(config: Gemma4Config, key: str) -> bool:
    match = _LAYER_RE.match(key)
    if match is None:
        return False
    layer_idx, suffix = int(match.group(1)), match.group(2)
    return config.text.is_kv_shared_layer(layer_idx) and suffix in _SHARED_KV_SUFFIXES


def load_model(
    model_id_or_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    attn_backend: AttentionBackend | None = None,
    config: Gemma4Config | None = None,
) -> Gemma4Model:
    """Load a checkpoint, materializing weights directly on ``device``.

    ``dtype`` overrides the checkpoint's dtype (default bf16); ``attn_backend``
    overrides the attention implementation for both towers. The model is
    returned in eval mode with gradients disabled; for training, re-enable
    with ``model.requires_grad_(True)`` / ``model.train()``.
    """
    checkpoint_dir = resolve_checkpoint_dir(model_id_or_path)
    if config is None:
        config = load_config(checkpoint_dir)
    if dtype is not None:
        config = dataclasses.replace(config, dtype=dtype)
    if attn_backend is not None:
        config = dataclasses.replace(
            config,
            text=dataclasses.replace(config.text, attn_backend=attn_backend),
            vision=(
                dataclasses.replace(config.vision, attn_backend=attn_backend)
                if config.vision is not None
                else None
            ),
        )

    model = Gemma4Model(config, device="meta")

    state_dict: dict[str, torch.Tensor] = {}
    weight_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no *.safetensors files in {checkpoint_dir}")
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device=str(device)) as f:
            for key in f.keys():
                if key.startswith(_AUDIO_PREFIXES):
                    continue
                name = key.removeprefix("model.")
                if _is_dropped_shared_kv_key(config, name):
                    continue
                tensor = f.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(config.dtype)
                state_dict[name] = tensor

    if config.text.tie_word_embeddings and "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["language_model.embed_tokens.weight"]

    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    model.requires_grad_(False)
    # Parameters are already on the target device; this sweeps over the small
    # computed buffers (rope inv_freq, embed scales), which meta construction
    # materializes on CPU (see bijou.gemma4.layers.buffer_device).
    return model.to(device)


def load_generation_defaults(checkpoint_dir: Path) -> dict[str, object]:
    """The checkpoint's ``generation_config.json`` as a plain dict (eos ids,
    sampling defaults); purely informational for callers."""
    path = checkpoint_dir / "generation_config.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
