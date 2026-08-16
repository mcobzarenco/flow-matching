"""Checkpoint loading: HF safetensors -> pure-torch Molmo2 text model.

Two load forms share the mount machinery:

- :func:`load_text_model`/:func:`load_vision_backbone` read an
  HF-LAYOUT artifact directory (dir-glob over ``*.safetensors``) and
  translate keys — the importers', parity harnesses' and fresh-run
  mount path. The module tree deliberately mirrors HF's names, so the
  mapping is mostly "strip the ``model.`` prefix" (``lm_head.weight``
  sits outside ``model.`` in the checkpoint and outside ``transformer``
  here). Checkpoint quirks handled: MolmoAct2 checkpoints carry the
  flow action expert (``model.action_expert.*``) and a persisted
  derived RoPE table (``transformer.rotary_emb.inv_freq``) the trunk
  loader skips; the released checkpoint ships fp32 — pass
  ``dtype=torch.bfloat16`` to cast at load (matching how the mount will
  run); with ``truncate_layers=N`` only decoder blocks ``0..N-1`` are
  instantiated and loaded — the truncated prefix mount (unlike gemma4
  there are no KV-sharing or layer-type constraints; the truncated
  model is built WITHOUT the LM head).
- :func:`load_text_model_from_file`/:func:`load_vision_backbone_from_file`
  read a VLA checkpoint's per-part trunk files
  (``backbone_text``/``backbone_vision``), which already carry OUR key
  names — translation happened ONCE at import
  (:func:`import_backbone_state`), so the load is a plain strict
  ``load_state_dict`` with no skip-prefixes. Always full-depth with the
  head: the family compositions mount whole trunks.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from ..gemma4.loading import resolve_checkpoint_dir
from ..nn import DEFAULT_ATTENTION_BACKEND, AttentionBackend
from .config import Molmo2Config, Molmo2TextConfig
from .text import Molmo2TextModel
from .vision import Molmo2VisionBackbone

_VISION_PREFIX = "model.vision_backbone."
# MolmoAct2 checkpoints carry the flow action expert alongside the trunk;
# the converter extracts those tensors (flow_decoder.safetensors) and the
# molmo_flow decoder loads them — the trunk loader skips the prefix.
_ACTION_EXPERT_PREFIX = "model.action_expert."
_LM_HEAD_KEY = "lm_head.weight"
# MolmoAct2 exports persist the derived RoPE table; ours is a computed
# non-persistent buffer.
_ROTARY_INV_FREQ_KEY = "model.transformer.rotary_emb.inv_freq"
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
                if key.startswith((_VISION_PREFIX, _ACTION_EXPERT_PREFIX)):
                    continue
                if key == _LM_HEAD_KEY:
                    if not with_lm_head:
                        continue
                    name = key
                else:
                    name = key.removeprefix("model.")
                    # MolmoAct2 exports persist the derived RoPE table;
                    # ours is a computed non-persistent buffer.
                    if name == "transformer.rotary_emb.inv_freq":
                        continue
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
    # materializes on CPU (see bijou.modelling.nn.buffer_device).
    return model.to(device)


def load_text_model_from_file(
    config: Molmo2TextConfig,
    text_file: Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
) -> Molmo2TextModel:
    """Load the full text decoder (head included) from a per-part weight
    file carrying OUR key names (a VLA checkpoint's ``backbone_text``) —
    a plain strict load, no key translation, no skip-prefixes. ``dtype``
    casts floating tensors at load (None = the file's dtypes verbatim).
    The model is returned in eval mode with gradients disabled."""
    if not text_file.is_file():
        raise SystemExit(f"backbone part file {text_file} is missing")
    model = Molmo2TextModel(
        config,
        lm_head=True,
        attn_backend=attn_backend,
        device="meta",
    )
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(text_file, framework="pt", device=str(device)) as f:
        # it exposes .keys() but not iteration/__contains__.
        for key in f.keys():  # noqa: SIM118
            tensor = f.get_tensor(key)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            state_dict[key] = tensor
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    model.requires_grad_(False)
    # Sweep the small computed buffers (rope inv_freq) onto the device.
    return model.to(device)


def load_vision_backbone_from_file(
    config: Molmo2Config,
    vision_file: Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> Molmo2VisionBackbone:
    """Load the vision tower + connector from a per-part weight file
    carrying OUR key names (a VLA checkpoint's ``backbone_vision``) —
    the strict from-files twin of :func:`load_vision_backbone`."""
    if config.vit is None or config.adapter is None:
        raise SystemExit("backbone config has no vision tower")
    if not vision_file.is_file():
        raise SystemExit(f"backbone part file {vision_file} is missing")
    model = Molmo2VisionBackbone(config.vit, config.adapter, device="meta")
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(vision_file, framework="pt", device=str(device)) as f:
        # it exposes .keys() but not iteration/__contains__.
        for key in f.keys():  # noqa: SIM118
            tensor = f.get_tensor(key)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            state_dict[key] = tensor
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


@dataclass(frozen=True, slots=True)
class ImportedBackboneState:
    """An HF Molmo2/MolmoAct2 artifact's tensors translated to OUR key
    names and partitioned by role: ``text`` = the decoder stack + the
    untied head (:class:`Molmo2TextModel` keys); ``vision`` = tower +
    connector (:class:`Molmo2VisionBackbone` keys — exactly the
    ``backbone_vision`` LR group's members); ``expert`` = the MolmoAct2
    flow action expert (prefix-stripped; empty on plain Molmo2).
    ``skipped`` names every source key deliberately not imported.
    Tensor bytes and dtypes are verbatim — extraction is key-filtering,
    never value change."""

    text: dict[str, torch.Tensor]
    vision: dict[str, torch.Tensor]
    expert: dict[str, torch.Tensor]
    skipped: tuple[str, ...]


def import_backbone_state(checkpoint_dir: Path) -> ImportedBackboneState:
    """Translate an HF-layout Molmo2/MolmoAct2 artifact into per-part
    states (the ONE place translation happens; loads of the produced
    files are plain strict ``load_state_dict``).

    The key partition is AUDITED: every source shard key must classify
    as text, vision, expert, or known-skipped (the persisted derived
    RoPE table) — an unclassified tensor refuses the import with the
    keys named, so a layout drift in a new release can never silently
    drop weights."""
    text: dict[str, torch.Tensor] = {}
    vision: dict[str, torch.Tensor] = {}
    expert: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    unclassified: list[str] = []
    weight_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not weight_files:
        raise SystemExit(f"no *.safetensors files in {checkpoint_dir}")
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            # it exposes .keys() but not iteration/__contains__.
            for key in f.keys():  # noqa: SIM118
                if key == _LM_HEAD_KEY:
                    text[key] = f.get_tensor(key)
                elif key == _ROTARY_INV_FREQ_KEY:
                    skipped.append(key)
                elif key.startswith(_VISION_PREFIX):
                    vision[key.removeprefix(_VISION_PREFIX)] = f.get_tensor(key)
                elif key.startswith(_ACTION_EXPERT_PREFIX):
                    expert[key.removeprefix(_ACTION_EXPERT_PREFIX)] = f.get_tensor(key)
                elif key.startswith("model.transformer."):
                    text[key.removeprefix("model.")] = f.get_tensor(key)
                else:
                    unclassified.append(key)
    if unclassified:
        raise SystemExit(
            f"{checkpoint_dir}: import audit FAILED — "
            f"{len(unclassified)} source key(s) classify as neither text, "
            f"vision, expert, nor known-skipped: {sorted(unclassified)} — "
            "refusing to import a layout this translation does not fully "
            "cover",
        )
    if not text:
        raise SystemExit(f"{checkpoint_dir}: no text-stack tensors found")
    return ImportedBackboneState(
        text=text,
        vision=vision,
        expert=expert,
        skipped=tuple(sorted(skipped)),
    )


def load_vision_backbone(
    model_id_or_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> Molmo2VisionBackbone:
    """Load the vision tower + connector of a Molmo2 checkpoint.

    Standalone (WP2): the WP4 encoder assembles it with the text mount; the
    parity harness gates it at this boundary against the reference module.
    """
    checkpoint_dir = resolve_checkpoint_dir(model_id_or_path)
    config = load_config(checkpoint_dir)
    if config.vit is None or config.adapter is None:
        raise ValueError(f"{checkpoint_dir} has no vision tower")
    if dtype is None:
        dtype = config.dtype

    model = Molmo2VisionBackbone(config.vit, config.adapter, device="meta")

    state_dict: dict[str, torch.Tensor] = {}
    weight_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no *.safetensors files in {checkpoint_dir}")
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device=str(device)) as f:
            # it exposes .keys() but not iteration/__contains__.
            for key in f.keys():  # noqa: SIM118
                if not key.startswith(_VISION_PREFIX):
                    continue
                tensor = f.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                state_dict[key.removeprefix(_VISION_PREFIX)] = tensor

    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)
