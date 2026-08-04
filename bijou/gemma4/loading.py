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
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from ..nn import DEFAULT_ATTENTION_BACKEND, AttentionBackend
from .config import Gemma4Config, LayerType
from .model import Gemma4Model

_AUDIO_PREFIXES = ("model.audio_tower.", "model.embed_audio.")
_SHARED_KV_SUFFIXES = (
    "k_proj.weight",
    "v_proj.weight",
    "k_norm.weight",
    "v_norm.weight",
)
_LAYER_RE = re.compile(r"^language_model\.layers\.(\d+)\.self_attn\.(.+)$")
_ANY_LAYER_RE = re.compile(r"^language_model\.layers\.(\d+)\.")
# Keys whose tensors pack one slice per decoder layer along a dimension;
# truncation keeps the first `num_layers` slices (dim, slice axis).
_PLE_PACKED_KEYS = {
    "language_model.embed_tokens_per_layer.weight": 1,
    "language_model.per_layer_model_projection.weight": 0,
}


def resolve_checkpoint_dir(model_id_or_path: str | Path) -> Path:
    """Accepts a local directory or an HF repo id (resolved via the cache,
    downloading if necessary)."""
    path = Path(model_id_or_path)
    if path.is_dir():
        return path
    return Path(snapshot_download(str(model_id_or_path)))


def load_config(checkpoint_dir: Path) -> Gemma4Config:
    return Gemma4Config.from_json(checkpoint_dir / "config.json")


def _is_dropped_shared_kv_key(config: Gemma4Config, key: str) -> bool:
    match = _LAYER_RE.match(key)
    if match is None:
        return False
    layer_idx, suffix = int(match.group(1)), match.group(2)
    return config.text.is_kv_shared_layer(layer_idx) and suffix in _SHARED_KV_SUFFIXES


def _is_truncated_layer_key(key: str, num_layers: int) -> bool:
    match = _ANY_LAYER_RE.match(key)
    return match is not None and int(match.group(1)) >= num_layers


def truncated_config(config: Gemma4Config, num_layers: int) -> Gemma4Config:
    """Config for the first ``num_layers`` decoder layers of a checkpoint.

    Truncation may not cross into the KV-shared region (those layers have no
    K/V weights to keep) and must end on a full-attention layer. The
    truncated model has no KV sharing; run with a :class:`KVCache` to export
    per-layer K/V (full-attention layers cache the entire sequence).
    """
    text = config.text
    if not 0 < num_layers <= text.first_kv_shared_layer_idx:
        raise ValueError(
            f"num_layers must be in (0, {text.first_kv_shared_layer_idx}] "
            f"(the non-KV-shared prefix), got {num_layers}",
        )
    layer_types = text.layer_types[:num_layers]
    if layer_types[-1] is not LayerType.FULL:
        raise ValueError(
            f"truncation point must end on a full_attention layer; layer "
            f"{num_layers - 1} is {layer_types[-1]}",
        )
    text = dataclasses.replace(
        text,
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        num_kv_shared_layers=0,
    )
    return dataclasses.replace(config, text=text)


def truncate_backbone_state(
    state: dict[str, torch.Tensor],
    config: Gemma4Config,
) -> dict[str, torch.Tensor]:
    """Adapt a backbone state dict saved at FULL depth to a truncated
    build: drop layer weights at indices >= the build's depth and slice
    the packed per-layer-embedding tensors (one slice per layer, fused
    along one dim — `_PLE_PACKED_KEYS`) to the kept layers. A no-op
    when depths already match. Lossless by construction for the kept
    layers: slices are layer-major, so prefix layers keep exactly their
    own rows/columns — this is the in-memory twin of load_model's
    on-disk truncated read."""
    num_layers = config.text.num_hidden_layers
    ple_width = num_layers * config.text.hidden_size_per_layer_input
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if _is_truncated_layer_key(key, num_layers):
            continue
        axis = _PLE_PACKED_KEYS.get(key)
        if axis is not None and value.shape[axis] > ple_width:
            value = value.narrow(axis, 0, ple_width)
        result[key] = value
    return result


def load_model(
    model_id_or_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
    config: Gemma4Config | None = None,
    truncate_layers: int | None = None,
) -> Gemma4Model:
    """Load a checkpoint, materializing weights directly on ``device``.

    ``dtype`` overrides the checkpoint's dtype (the released E-series ship
    bf16); ``attn_backend`` selects the attention implementation for both
    towers. With ``truncate_layers=N`` only the first N decoder layers are
    instantiated and loaded — including only the first N slices of the packed
    per-layer-embedding (PLE) tensors — e.g. the Bijou prefix encoder keeps
    just the non-KV-shared prefix (see :func:`truncated_config`). The model
    is returned in eval mode with gradients disabled; for training, re-enable
    with ``model.requires_grad_(True)`` / ``model.train()``.
    """
    checkpoint_dir = resolve_checkpoint_dir(model_id_or_path)
    if config is None:
        config = load_config(checkpoint_dir)
    if dtype is not None:
        config = dataclasses.replace(config, dtype=dtype)
    if truncate_layers is not None:
        config = truncated_config(config, truncate_layers)

    model = Gemma4Model(config, attn_backend=attn_backend, device="meta")

    ple_slice_dim = truncate_layers and (
        truncate_layers * config.text.hidden_size_per_layer_input
    )
    state_dict: dict[str, torch.Tensor] = {}
    weight_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no *.safetensors files in {checkpoint_dir}")
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device=str(device)) as f:
            # it exposes .keys() but not iteration/__contains__.
            for key in f.keys():  # noqa: SIM118
                if key.startswith(_AUDIO_PREFIXES):
                    continue
                name = key.removeprefix("model.")
                if _is_dropped_shared_kv_key(config, name):
                    continue
                if truncate_layers is not None and _is_truncated_layer_key(
                    name,
                    truncate_layers,
                ):
                    continue
                if ple_slice_dim and (axis := _PLE_PACKED_KEYS.get(name)) is not None:
                    # Read only the kept slices from disk.
                    tensor_slice = f.get_slice(key)
                    if axis == 0:
                        tensor = tensor_slice[:ple_slice_dim, :]
                    else:
                        tensor = tensor_slice[:, :ple_slice_dim]
                else:
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
    # materializes on CPU (see bijou.nn.buffer_device).
    return model.to(device)


@dataclass(frozen=True, slots=True)
class GenerationDefaults:
    """The consumed slice of a checkpoint's ``generation_config.json``:
    the eos ids HF's generate() stops on (int or list in the JSON,
    normalized to a tuple; None when absent or no config file exists)."""

    eos_token_ids: tuple[int, ...] | None


def load_generation_defaults(checkpoint_dir: Path) -> GenerationDefaults:
    path = checkpoint_dir / "generation_config.json"
    if not path.exists():
        return GenerationDefaults(eos_token_ids=None)
    with path.open() as f:
        data = json.load(f)
    eos = data.get("eos_token_id")
    if eos is None:
        return GenerationDefaults(eos_token_ids=None)
    if isinstance(eos, int):
        return GenerationDefaults(eos_token_ids=(eos,))
    return GenerationDefaults(eos_token_ids=tuple(int(t) for t in eos))
