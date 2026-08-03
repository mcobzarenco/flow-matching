"""Building a Bijou VLA from a Gemma 4 checkpoint.

``from_backbone("google/gemma-4-e2b-it", action_dim=6, state_dim=6)`` gives a
ready-to-train model:

- the backbone is loaded truncated to its non-KV-shared prefix (layers 0–14
  for E2B) with only the needed weights: dropped decoder layers, per-layer
  embedding (PLE) tensors sliced to the kept layers, no audio tower, and the
  LM head is a tie (zero extra memory). ~2.1B params instead of 5.2B for E2B.
  It is frozen (eval mode, ``requires_grad=False``).
- the action expert is freshly initialized on the same device/dtype.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import Tensor

from .aux_text import AuxDecodeConfig, build_aux_runtime
from .data import DatasetStats
from .decoders.ar_backbone import ARBackboneConfig, ARBackboneDecoder
from .decoders.ar_fast import ARFastConfig, ARFastDecoder
from .decoders.flow import (
    ExpertConfig,
    FlowDecoder,
    SelfAttentionMode,
    TimeConditioning,
)
from .encoders.gemma4 import PROMPT_FORMAT, GemmaEncoder
from .fast.codec import ActionCodec
from .gemma4.config import Gemma4Config, LayerType
from .gemma4.loading import load_config, load_model, resolve_checkpoint_dir
from .gemma4.model import Gemma4Model
from .interface import kv_stream_name
from .model import BijouModel
from .nn import DEFAULT_ATTENTION_BACKEND, AttentionBackend, DeviceLike

# bijou_config.json schema version. Format 3 sections the metadata by
# role — backbone (the shared network), prompt (the prompt-side
# strategy), decoder (the tagged head config) — replacing format 2's
# encoder/decoder pair, which had replaced the original layout (backbone
# + expert_config keys). The read side synthesizes current semantics
# from BOTH older formats, so every existing checkpoint keeps loading
# without conversion.
CHECKPOINT_FORMAT = 3


class BackboneDepth(StrEnum):
    """How much of the backbone stack a checkpoint's model runs."""

    # Truncated to the non-KV-shared prefix (layers 0..14 for E2B) — the
    # cross-attention decoders' backbone; formats 1/2 are always this.
    PREFIX = "prefix"
    # The whole stack — the decoder-only path (prompt encode still stops
    # at the prefix; the suffix runs the KV-shared deep half).
    FULL = "full"


@dataclass(frozen=True, slots=True)
class BackboneConfig:
    """The backbone section of bijou_config.json: which pretrained
    checkpoint (``id``: HF id or local path), how deep it is mounted.
    Adaptedness is deliberately NOT recorded here — ``backbone.safetensors``
    presence is the (test-gated) invariant for a backbone that differs
    from pristine HF."""

    id: str
    depth: BackboneDepth

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "depth": self.depth.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackboneConfig:
        return cls(
            id=str(data["id"]),
            depth=BackboneDepth(data["depth"]),
        )


class PromptKind(StrEnum):
    """Tag of a prompt-side encoder config in bijou_config.json.
    (Format 2 called this section "encoder" and kept the backbone id
    inside it; format 3 moves the id to the backbone section.)"""

    GEMMA4 = "gemma4"


class DecoderKind(StrEnum):
    """Tag of an action-decoder config in bijou_config.json."""

    FLOW = "flow"
    AR_FAST = "ar_fast"
    AR_BACKBONE = "ar_backbone"


@dataclass(frozen=True, slots=True)
class GemmaPromptConfig:
    """The Gemma prompt-side strategy as recorded in a checkpoint.

    ``exports`` are backbone layer indices whose K/V become the memory
    streams (named ``kv{layer}`` — backbone internals live HERE, never in
    decoder configs). ``format`` is the prompt layout version
    (encoders.gemma4.PROMPT_FORMAT when written; 1 = tag-less legacy,
    2 = bracket camera tags, 3 = pipe-unified extended sandwich with
    the [generate|…] request and the soft state token — whose
    projection width is ``state_dim``). Loading REFUSES formats < 3:
    the prompt-side parameter set (state_proj) does not exist there
    and no checkpoint worth preserving does either (owner call,
    2026-08-03)."""

    exports: tuple[int, ...]
    max_soft_tokens: int
    format: int
    state_dim: int
    # Value-conditioning fields trained into the user turn's bracket
    # block (empty = unconditioned). Inference must render matching
    # conditioning — BijouPolicy reads this to configure its collator.
    condition_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": PromptKind.GEMMA4.value,
            "exports": list(self.exports),
            "max_soft_tokens": self.max_soft_tokens,
            "format": self.format,
            "state_dim": self.state_dim,
            "condition_fields": list(self.condition_fields),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GemmaPromptConfig:
        # Pre-format-3 checkpoints recorded camera_tags: bool (or
        # nothing): parse to their format int, refuse at load below.
        if "format" in data:
            format_version = int(data["format"])
        else:
            format_version = 2 if bool(data.get("camera_tags", False)) else 1
        if format_version < PROMPT_FORMAT:
            raise SystemExit(
                f"checkpoint prompt format {format_version} < "
                f"{PROMPT_FORMAT}: pre-3 prompts (no [generate|…], no soft "
                "state token, colon conditioning) are refused — retrain "
                "(no back-compat, 2026-08-03)",
            )
        return cls(
            exports=tuple(int(layer) for layer in data["exports"]),
            max_soft_tokens=int(data["max_soft_tokens"]),
            format=format_version,
            state_dim=int(data["state_dim"]),
            condition_fields=tuple(
                str(field) for field in data.get("condition_fields", [])
            ),
        )

    @property
    def stream_names(self) -> tuple[str, ...]:
        return tuple(kv_stream_name(layer) for layer in self.exports)


@dataclass(frozen=True, slots=True)
class FlowDecoderConfig:
    """The flow-matching action decoder as recorded in a checkpoint.

    ``schedule`` references encoder stream NAMES, one per decoder layer
    (its length is the decoder depth). Cross-attention head_dim and rope
    are per-stream geometry declared by the encoder, deliberately absent
    here."""

    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    hidden_activation: str
    rms_norm_eps: float
    self_attention_mode: SelfAttentionMode
    self_attention_rope_theta: float
    cross_attention_heads: int
    schedule: tuple[str, ...]
    action_dim: int
    state_dim: int
    chunk_size: int
    time_embed_dim: int
    time_conditioning: TimeConditioning

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": DecoderKind.FLOW.value,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_activation": self.hidden_activation,
            "rms_norm_eps": self.rms_norm_eps,
            "self_attention_mode": self.self_attention_mode.value,
            "self_attention_rope_theta": self.self_attention_rope_theta,
            "cross_attention_heads": self.cross_attention_heads,
            "schedule": list(self.schedule),
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "chunk_size": self.chunk_size,
            "time_embed_dim": self.time_embed_dim,
            "time_conditioning": self.time_conditioning.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowDecoderConfig:
        return cls(
            hidden_size=int(data["hidden_size"]),
            num_attention_heads=int(data["num_attention_heads"]),
            intermediate_size=int(data["intermediate_size"]),
            hidden_activation=str(data["hidden_activation"]),
            rms_norm_eps=float(data["rms_norm_eps"]),
            self_attention_mode=SelfAttentionMode(data["self_attention_mode"]),
            self_attention_rope_theta=float(data["self_attention_rope_theta"]),
            cross_attention_heads=int(data["cross_attention_heads"]),
            schedule=tuple(str(name) for name in data["schedule"]),
            action_dim=int(data["action_dim"]),
            state_dim=int(data["state_dim"]),
            chunk_size=int(data["chunk_size"]),
            time_embed_dim=int(data["time_embed_dim"]),
            time_conditioning=TimeConditioning(data["time_conditioning"]),
        )


def parse_prompt_config(data: dict[str, Any]) -> GemmaPromptConfig:
    kind = PromptKind(data["kind"])
    match kind:
        case PromptKind.GEMMA4:
            return GemmaPromptConfig.from_dict(data)


def ar_fast_config_to_dict(config: ARFastConfig) -> dict[str, Any]:
    return {
        "kind": DecoderKind.AR_FAST.value,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "intermediate_size": config.intermediate_size,
        "hidden_activation": config.hidden_activation,
        "rms_norm_eps": config.rms_norm_eps,
        "self_attention_rope_theta": config.self_attention_rope_theta,
        "cross_attention_heads": config.cross_attention_heads,
        "schedule": list(config.schedule),
        "tokenizer": config.tokenizer,
        "vocab_total": config.vocab_total,
        "state_dim": config.state_dim,
        "chunk_size": config.chunk_size,
        "action_dim": config.action_dim,
    }


def ar_fast_config_from_dict(data: dict[str, Any]) -> ARFastConfig:
    return ARFastConfig(
        hidden_size=int(data["hidden_size"]),
        num_attention_heads=int(data["num_attention_heads"]),
        intermediate_size=int(data["intermediate_size"]),
        hidden_activation=str(data["hidden_activation"]),
        rms_norm_eps=float(data["rms_norm_eps"]),
        self_attention_rope_theta=float(data["self_attention_rope_theta"]),
        cross_attention_heads=int(data["cross_attention_heads"]),
        schedule=tuple(str(name) for name in data["schedule"]),
        tokenizer=str(data["tokenizer"]),
        vocab_total=int(data["vocab_total"]),
        state_dim=int(data["state_dim"]),
        chunk_size=int(data["chunk_size"]),
        action_dim=int(data["action_dim"]),
    )


def ar_backbone_config_to_dict(config: ARBackboneConfig) -> dict[str, Any]:
    return {
        "kind": DecoderKind.AR_BACKBONE.value,
        "tokenizer": config.tokenizer,
        "vocab_total": config.vocab_total,
        "block_base": config.block_base,
        "chunk_size": config.chunk_size,
        "action_dim": config.action_dim,
        "suffix_format": config.suffix_format,
        "aux": None if config.aux is None else config.aux.to_dict(),
    }


def ar_backbone_config_from_dict(data: dict[str, Any]) -> ARBackboneConfig:
    aux = data.get("aux")  # absent in pre-aux checkpoints == null
    return ARBackboneConfig(
        tokenizer=str(data["tokenizer"]),
        vocab_total=int(data["vocab_total"]),
        block_base=int(data["block_base"]),
        chunk_size=int(data["chunk_size"]),
        action_dim=int(data["action_dim"]),
        # Absent = legacy pre-opener checkpoint (format 1) — refused by
        # ARBackboneConfig (formats < 5 have incompatible parameters).
        suffix_format=int(data.get("suffix_format", 1)),
        aux=None if aux is None else AuxDecodeConfig.from_dict(aux),
    )


def parse_decoder_config(
    data: dict[str, Any],
) -> FlowDecoderConfig | ARFastConfig | ARBackboneConfig:
    kind = DecoderKind(data["kind"])
    match kind:
        case DecoderKind.FLOW:
            return FlowDecoderConfig.from_dict(data)
        case DecoderKind.AR_FAST:
            return ar_fast_config_from_dict(data)
        case DecoderKind.AR_BACKBONE:
            return ar_backbone_config_from_dict(data)


def decoder_schema_dict(
    decoder: FlowDecoder | ARFastDecoder | ARBackboneDecoder,
) -> dict[str, Any]:
    """The checkpoint-schema dict of a built decoder (write side + the
    --init-from config guard)."""
    match decoder:
        case FlowDecoder():
            return flow_decoder_config_from_expert(decoder.config).to_dict()
        case ARFastDecoder():
            return ar_fast_config_to_dict(decoder.config)
        case ARBackboneDecoder():
            return ar_backbone_config_to_dict(decoder.config)


def resolve_action_codec(ref: str) -> ActionCodec:
    """Load a FAST tokenizer artifact from a local directory or from the
    hub (``<user>/<repo>/<subfolder>``, e.g.
    mcobzarenco/bijou-checkpoints/fast_tokenizer_v1)."""
    local = Path(ref).expanduser()
    if (local / "fast_config.json").exists():
        return ActionCodec.load(local)
    parts = ref.split("/")
    if len(parts) < 3:
        raise SystemExit(
            f"--fast-tokenizer {ref!r} is neither a local artifact directory "
            "nor a hub reference of the form <user>/<repo>/<subfolder>",
        )
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:])
    downloaded = snapshot_download(
        repo_id,
        repo_type="model",
        allow_patterns=[f"{subfolder}/*"],
    )
    return ActionCodec.load(Path(downloaded) / subfolder)


def prefix_global_layers(config: Gemma4Config) -> tuple[int, ...]:
    """Global-attention layers within the non-KV-shared prefix — the K/V
    streams available to the expert (E2B: (4, 9, 14); E4B: (5, 11, 17, 23))."""
    text = config.text
    return tuple(
        idx
        for idx in range(text.first_kv_shared_layer_idx)
        if text.layer_types[idx] is LayerType.FULL
    )


def default_expert_config(
    backbone: Gemma4Config,
    *,
    action_dim: int,
    state_dim: int,
    stream_counts: tuple[int, ...] = (4, 4, 7),
    hidden_size: int = 768,
    num_attention_heads: int = 6,
    intermediate_size: int = 3072,
    cross_attention_heads: int = 4,
    chunk_size: int = 50,
    time_embed_dim: int = 256,
    self_attention_mode: SelfAttentionMode = SelfAttentionMode.CAUSAL_ACTIONS,
    self_attention_rope_theta: float = 10_000.0,
    time_conditioning: TimeConditioning = TimeConditioning.ADDITIVE,
) -> ExpertConfig:
    """Expert config with a blocks cross-attention schedule.

    ``stream_counts[i]`` expert layers attend the i-th global prefix layer,
    shallow to deep; the default (4, 4, 7) weights the deepest stream (L14
    for E2B — the one the backbone's own KV-shared half runs on) and gives a
    15-layer expert. For E4B (four streams) pass four counts.
    """
    streams = prefix_global_layers(backbone)
    if len(stream_counts) != len(streams):
        raise ValueError(
            f"stream_counts has {len(stream_counts)} entries but the backbone "
            f"prefix has {len(streams)} global layers ({streams})",
        )
    schedule = tuple(
        chain.from_iterable(
            (stream,) * count
            for stream, count in zip(streams, stream_counts, strict=True)
        ),
    )
    return ExpertConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_activation=backbone.text.hidden_activation,
        rms_norm_eps=backbone.text.rms_norm_eps,
        self_attention_mode=self_attention_mode,
        self_attention_rope_theta=self_attention_rope_theta,
        cross_attention_heads=cross_attention_heads,
        cross_attention_head_dim=backbone.text.global_head_dim,
        cross_attention_rope=backbone.text.rope_parameters[LayerType.FULL],
        cross_attention_schedule=schedule,
        action_dim=action_dim,
        state_dim=state_dim,
        chunk_size=chunk_size,
        time_embed_dim=time_embed_dim,
        time_conditioning=time_conditioning,
    )


def from_backbone(
    model_id_or_path: str | Path,
    expert_config: ExpertConfig | None = None,
    *,
    action_dim: int | None = None,
    state_dim: int | None = None,
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    expert_dtype: torch.dtype | None = None,
    attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
    max_soft_tokens: int = 140,
) -> BijouModel:
    """Build a Bijou model (GemmaEncoder + FlowDecoder) from a Gemma 4
    checkpoint (prompt format 3 — the only implemented layout).

    Pass either a full ``expert_config`` or just ``action_dim``/``state_dim``
    to use :func:`default_expert_config`. The backbone is truncated to its
    non-KV-shared prefix, frozen; the decoder is freshly initialized.
    ``expert_dtype`` may differ from the backbone dtype (e.g. fp32 expert on
    a bf16 backbone for training — the decoder casts its inputs and the
    exported KV streams to its own dtype). ``max_soft_tokens`` parameterizes
    the encoder's inputs collator (the CLI default when not loading a
    checkpoint that recorded it).
    """
    checkpoint_dir = resolve_checkpoint_dir(model_id_or_path)
    config = load_config(checkpoint_dir)

    if expert_config is None:
        if action_dim is None or state_dim is None:
            raise ValueError(
                "pass either expert_config or both action_dim and state_dim",
            )
        expert_config = default_expert_config(
            config,
            action_dim=action_dim,
            state_dim=state_dim,
        )

    available = prefix_global_layers(config)
    for stream in expert_config.streams:
        if stream not in available:
            raise ValueError(
                f"cross-attention stream {stream} is not a global layer of the "
                f"backbone prefix (available: {available})",
            )

    if expert_dtype is None:
        expert_dtype = dtype if dtype is not None else config.dtype
    backbone, encoder = build_gemma_encoder(
        checkpoint_dir,
        config,
        exports=expert_config.streams,
        max_soft_tokens=max_soft_tokens,
        state_dim=expert_config.state_dim,
        device=device,
        dtype=dtype,
        attn_backend=attn_backend,
    )
    decoder = FlowDecoder(
        expert_config,
        attn_backend=attn_backend,
        device=device,
        dtype=expert_dtype,
    )
    return BijouModel(backbone=backbone, encoder=encoder, decoder=decoder)


def build_gemma_encoder(
    checkpoint_dir: Path,
    config: Gemma4Config,
    *,
    exports: tuple[int, ...],
    max_soft_tokens: int,
    state_dim: int,
    device: DeviceLike,
    dtype: torch.dtype | None,
    attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
    depth: BackboneDepth = BackboneDepth.PREFIX,
) -> tuple[Gemma4Model, GemmaEncoder]:
    """The Gemma backbone (frozen; truncated to its non-KV-shared layer
    prefix at the default PREFIX depth, whole stack at FULL) plus its
    prompt-side encoder strategy — the pair BijouModel composes. The
    encoder's state_proj is freshly zero-initialized; checkpoint loads
    overwrite it from prompt.safetensors."""
    backbone = load_model(
        checkpoint_dir,
        device="cpu" if device is None else device,
        dtype=dtype,
        attn_backend=attn_backend,
        truncate_layers=(
            config.text.first_kv_shared_layer_idx
            if depth is BackboneDepth.PREFIX
            else None
        ),
    )
    encoder = GemmaEncoder(
        backbone.config,
        exports=exports,
        processor_dir=str(checkpoint_dir),
        max_soft_tokens=max_soft_tokens,
        state_dim=state_dim,
        device=device,
        # The prompt-side params are "new parameters": fp32 like the
        # decoder's, whatever the backbone dtype (autocast covers the
        # forward).
        dtype=torch.float32,
    )
    return backbone, encoder


@dataclass(frozen=True, slots=True)
class CheckpointTrainArgs:
    """The architecture-determining subset of a checkpoint's recorded train
    args — every field a loader needs to rebuild the model. Present in all
    checkpoints since the format's introduction; newer recorded args (lr,
    seeds, data paths, ...) stay in the raw JSON but are not needed here."""

    decoder_hidden: int
    decoder_heads: int
    decoder_intermediate: int
    decoder_cross_heads: int
    stream_counts: tuple[int, ...]
    self_attention_mode: SelfAttentionMode
    chunk_size: int
    max_soft_tokens: int
    time_conditioning: TimeConditioning

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointTrainArgs:
        # Old checkpoints recorded the decoder shape under the historical
        # "expert_*" arg names; both spellings load forever.
        def either(new: str, old: str) -> Any:
            return data[new] if new in data else data[old]

        return cls(
            decoder_hidden=int(either("decoder_hidden", "expert_hidden")),
            decoder_heads=int(either("decoder_heads", "expert_heads")),
            decoder_intermediate=int(
                either("decoder_intermediate", "expert_intermediate"),
            ),
            decoder_cross_heads=int(
                either("decoder_cross_heads", "expert_cross_heads"),
            ),
            stream_counts=tuple(int(n) for n in data["stream_counts"]),
            self_attention_mode=SelfAttentionMode(data["self_attention_mode"]),
            chunk_size=int(data["chunk_size"]),
            max_soft_tokens=int(data["max_soft_tokens"]),
            # Pre-adaRMS checkpoints (no such key) are additive.
            time_conditioning=TimeConditioning(
                data.get("time_conditioning", TimeConditioning.ADDITIVE.value),
            ),
        )


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    """Read-side view of a checkpoint's ``bijou_config.json``: the parsed
    subset consumers need (the write side is :class:`CheckpointMetadata`).

    ``normalization`` is the count-weighted aggregate over the training
    datasets — a fallback for rigs without stats; ``per_dataset_normalization``
    is the per-dataset stats table (keyed by repo id — genuinely dynamic).
    """

    backbone: str
    train_args: CheckpointTrainArgs
    step: int
    normalization: DatasetStats
    per_dataset_normalization: dict[str, DatasetStats]
    # The prompt-side conditioning the checkpoint trained with (empty =
    # none/pre-conditioning checkpoint) — inference collators must
    # render matching fields.
    condition_fields: tuple[str, ...]

    @property
    def chunk_size(self) -> int:
        return self.train_args.chunk_size

    @property
    def max_soft_tokens(self) -> int:
        return self.train_args.max_soft_tokens


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Write-side schema of ``bijou_config.json`` (bijou.train fills it,
    :func:`from_checkpoint` reads the result back as CheckpointInfo).
    Writes format 3: role-sectioned metadata — ``backbone`` (which
    pretrained checkpoint, how deep), ``prompt`` (the tagged prompt-side
    config), ``decoder`` (the tagged config). Format-1/2 files remain readable via
    :func:`checkpoint_sections`.

    ``train_args`` is the full CLI record as a JSON-ready dict — prepared by
    the caller because this module must not import bijou.train's TrainArgs
    (the import DAG points the other way).
    """

    backbone: BackboneConfig
    prompt: GemmaPromptConfig
    decoder: dict[str, Any]
    normalization: DatasetStats
    per_dataset_normalization: dict[str, DatasetStats]
    train_args: dict[str, Any]
    step: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "backbone": self.backbone.to_dict(),
            "prompt": self.prompt.to_dict(),
            "decoder": self.decoder,
            "step": self.step,
            "train_args": self.train_args,
            # Training normalized per dataset; inference must normalize
            # with the deployment rig's stats. "normalization" keeps the
            # count-weighted aggregate as a fallback for rigs without stats
            # (keys match the dataset feature names; stable format).
            "normalization": self.normalization.state_dict(),
            "per_dataset_normalization": {
                repo_id: stats.state_dict()
                for repo_id, stats in sorted(
                    self.per_dataset_normalization.items(),
                )
            },
        }


@dataclass(frozen=True, slots=True)
class CheckpointSections:
    """A checkpoint's role sections with format differences erased — the
    ONE read-side entry for anything that needs a checkpoint's
    architecture. ``prompt``/``decoder`` are None only for format-1
    checkpoints (flow-only era: the decoder is synthesized from the
    recorded train args, and the prompt side had no recorded config
    beyond max_soft_tokens, which lives in train_args)."""

    backbone: BackboneConfig
    prompt: GemmaPromptConfig | None
    decoder: FlowDecoderConfig | ARFastConfig | ARBackboneConfig | None


def checkpoint_sections(meta: dict[str, Any]) -> CheckpointSections:
    """Parse any ``bijou_config.json`` payload (format 1, 2 or 3) into
    sections. Pure — no file or hub access. Dispatch is on the recorded
    format number (format-1 files predate the field): the "backbone" key
    is a section in format 3 but a plain id string in format 1."""
    fmt = int(meta.get("format", 1))
    if fmt >= 3:
        return CheckpointSections(
            backbone=BackboneConfig.from_dict(meta["backbone"]),
            prompt=parse_prompt_config(meta["prompt"]),
            decoder=parse_decoder_config(meta["decoder"]),
        )
    if fmt == 2:  # backbone id inside the encoder section
        return CheckpointSections(
            backbone=BackboneConfig(
                id=str(meta["encoder"]["backbone"]),
                depth=BackboneDepth.PREFIX,
            ),
            prompt=parse_prompt_config(meta["encoder"]),
            decoder=parse_decoder_config(meta["decoder"]),
        )
    # Format 1: backbone id at the top level, flow-only, no tagged configs.
    return CheckpointSections(
        backbone=BackboneConfig(id=str(meta["backbone"]), depth=BackboneDepth.PREFIX),
        prompt=None,
        decoder=None,
    )


def expert_config_from_train_args(
    backbone_config: Gemma4Config,
    train_args: CheckpointTrainArgs,
    *,
    action_dim: int,
    state_dim: int,
) -> ExpertConfig:
    """Rebuild the expert config a format-1 checkpoint's training run used
    from its recorded args — the legacy synthesizer's core (the serialized
    expert_config in old bijou_config.json stringifies enums and nested
    dataclasses; the train args are the clean source)."""
    return default_expert_config(
        backbone_config,
        action_dim=action_dim,
        state_dim=state_dim,
        stream_counts=train_args.stream_counts,
        hidden_size=train_args.decoder_hidden,
        num_attention_heads=train_args.decoder_heads,
        intermediate_size=train_args.decoder_intermediate,
        cross_attention_heads=train_args.decoder_cross_heads,
        chunk_size=train_args.chunk_size,
        self_attention_mode=train_args.self_attention_mode,
        time_conditioning=train_args.time_conditioning,
    )


def flow_decoder_config_from_expert(expert_config: ExpertConfig) -> FlowDecoderConfig:
    """ExpertConfig → the checkpoint-schema decoder config: schedule ints
    become stream names; cross-attention head_dim/rope (encoder-declared
    geometry) drop out. Pure and total — the write-side half of the
    format-2 bridge."""
    return FlowDecoderConfig(
        hidden_size=expert_config.hidden_size,
        num_attention_heads=expert_config.num_attention_heads,
        intermediate_size=expert_config.intermediate_size,
        hidden_activation=expert_config.hidden_activation,
        rms_norm_eps=expert_config.rms_norm_eps,
        self_attention_mode=expert_config.self_attention_mode,
        self_attention_rope_theta=expert_config.self_attention_rope_theta,
        cross_attention_heads=expert_config.cross_attention_heads,
        schedule=tuple(
            kv_stream_name(layer) for layer in expert_config.cross_attention_schedule
        ),
        action_dim=expert_config.action_dim,
        state_dim=expert_config.state_dim,
        chunk_size=expert_config.chunk_size,
        time_embed_dim=expert_config.time_embed_dim,
        time_conditioning=expert_config.time_conditioning,
    )


def expert_config_from_architecture(
    prompt: GemmaPromptConfig,
    decoder: FlowDecoderConfig,
    backbone_config: Gemma4Config,
) -> ExpertConfig:
    """Compose the prompt + decoder configs back into the expert's
    construction config: schedule names resolve to backbone layer
    indices against the prompt section's exports; cross-attention
    geometry comes from the backbone's global layers. Validates the
    references — unknown stream name or unconsumed export is a config
    error."""
    by_name = dict(zip(prompt.stream_names, prompt.exports, strict=True))
    unknown = [name for name in decoder.schedule if name not in by_name]
    if unknown:
        raise SystemExit(
            f"decoder schedule references unknown stream(s) {sorted(set(unknown))}; "
            f"the encoder exports {list(prompt.stream_names)}",
        )
    unused = [name for name in prompt.stream_names if name not in decoder.schedule]
    if unused:
        raise SystemExit(
            f"encoder export(s) {unused} are not consumed by the decoder "
            "schedule — remove them from the encoder config or schedule them",
        )
    return ExpertConfig(
        hidden_size=decoder.hidden_size,
        num_attention_heads=decoder.num_attention_heads,
        intermediate_size=decoder.intermediate_size,
        hidden_activation=decoder.hidden_activation,
        rms_norm_eps=decoder.rms_norm_eps,
        self_attention_mode=decoder.self_attention_mode,
        self_attention_rope_theta=decoder.self_attention_rope_theta,
        cross_attention_heads=decoder.cross_attention_heads,
        cross_attention_head_dim=backbone_config.text.global_head_dim,
        cross_attention_rope=backbone_config.text.rope_parameters[LayerType.FULL],
        cross_attention_schedule=tuple(by_name[name] for name in decoder.schedule),
        action_dim=decoder.action_dim,
        state_dim=decoder.state_dim,
        chunk_size=decoder.chunk_size,
        time_embed_dim=decoder.time_embed_dim,
        time_conditioning=decoder.time_conditioning,
    )


# The LM head is tied to the token embeddings (one storage, two state-dict
# keys): safetensors refuses aliased tensors, and Bijou never runs the head
# anyway — excluded on save, allowed to be exactly the missing key on load.
BACKBONE_UNSAVED_KEYS = frozenset({"lm_head.weight"})


def backbone_snapshot(model: BijouModel) -> dict[str, Tensor]:
    """The backbone state for ``backbone.safetensors`` (written by bijou.train
    when any unfreeze flag is on): parameters cast bf16 (the fp32 masters'
    precision beyond bf16 lives only in optimizer.pt), buffers at native
    dtype (RoPE inv_freq tables are fp32 by design — bf16 would corrupt
    them). The copy+cast happens HOST-side: a device-side cast would
    transiently allocate ~4.3 GB of VRAM, an OOM at the ~79 GB/80 GB
    occupancy measured for the live-backbone DDP config (A100, batch 32)."""
    parameter_names = {name for name, _ in model.backbone.named_parameters()}
    return {
        name: (
            tensor.detach().cpu().to(torch.bfloat16)
            if name in parameter_names
            else tensor.detach().cpu()
        ).contiguous()
        for name, tensor in model.backbone.state_dict().items()
        if name not in BACKBONE_UNSAVED_KEYS
    }


def load_adapted_backbone(
    model: BijouModel,
    checkpoint: Path,
) -> None:
    """Load ``backbone.safetensors`` over the (already-built) truncated
    backbone. The file is materialized on CPU and streamed into the live
    parameters by ``load_state_dict``'s copy semantics — loading it straight
    to the target device would transiently hold a second full backbone
    (~4.3 GB) next to the built one, which OOMed the 8 GiB laptop GPU at
    rollout. The same copy semantics cast the bf16 snapshot into whatever
    dtype the backbone was built with (bf16 for eval/rollout, fp32 masters
    for a live-backbone continuation)."""
    state = load_file(str(checkpoint / "backbone.safetensors"), device="cpu")
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    problems = [name for name in missing if name not in BACKBONE_UNSAVED_KEYS]
    if problems or unexpected:
        raise SystemExit(
            f"backbone.safetensors mismatch at {checkpoint}: "
            f"missing {problems}, unexpected {list(unexpected)}",
        )


def from_checkpoint(
    checkpoint: str | Path,
    *,
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    expert_dtype: torch.dtype = torch.float32,
    attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
) -> tuple[BijouModel, CheckpointInfo]:
    """Load a bijou training checkpoint directory (as written by
    bijou.train.save_checkpoint): backbone resolved from the recorded id,
    expert config rebuilt from the recorded train args, expert weights
    loaded strictly. Returns the eval-mode model plus checkpoint metadata
    (normalization stats table etc.)."""
    checkpoint = Path(checkpoint)
    meta = json.loads((checkpoint / "bijou_config.json").read_text())
    sections = checkpoint_sections(meta)
    info = CheckpointInfo(
        backbone=sections.backbone.id,
        condition_fields=(
            sections.prompt.condition_fields if sections.prompt is not None else ()
        ),
        train_args=CheckpointTrainArgs.from_dict(meta["train_args"]),
        step=int(meta["step"]),
        normalization=DatasetStats.from_state_dict(meta["normalization"]),
        per_dataset_normalization={
            repo_id: DatasetStats.from_state_dict(entry)
            for repo_id, entry in meta.get("per_dataset_normalization", {}).items()
        },
    )
    checkpoint_dir = resolve_checkpoint_dir(info.backbone)
    if isinstance(sections.decoder, ARFastConfig | ARBackboneConfig):
        decoder_config = sections.decoder
        assert sections.prompt is not None  # tagged formats carry it
        backbone_config = load_config(checkpoint_dir)
        if (
            isinstance(decoder_config, ARBackboneConfig)
            and sections.backbone.depth is not BackboneDepth.FULL
        ):
            raise SystemExit(
                f"{checkpoint} records an ar_backbone decoder with a "
                f"'{sections.backbone.depth}' backbone — its suffix runs "
                "the KV-shared deep half, which only the full stack has",
            )
        backbone, encoder = build_gemma_encoder(
            checkpoint_dir,
            backbone_config,
            exports=sections.prompt.exports,
            max_soft_tokens=sections.prompt.max_soft_tokens,
            state_dim=sections.prompt.state_dim,
            device=device,
            dtype=dtype,
            attn_backend=attn_backend,
            depth=sections.backbone.depth,
        )
        decoder: ARFastDecoder | ARBackboneDecoder
        if isinstance(decoder_config, ARFastConfig):
            decoder = ARFastDecoder(
                decoder_config,
                encoder.stream_geometries(),
                resolve_action_codec(decoder_config.tokenizer),
                attn_backend=attn_backend,
                device=device,
                dtype=expert_dtype,
            )
        else:
            # The backbone checkpoint's own tokenizer — the artifact the
            # collator rendered training text with (opener + value
            # lines); format 5 always needs it at construction.
            text_tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(checkpoint_dir),
            )
            aux_runtime = (
                build_aux_runtime(decoder_config.aux, text_tokenizer)
                if decoder_config.aux is not None
                else None
            )
            decoder = ARBackboneDecoder(
                decoder_config,
                backbone_config.text,
                resolve_action_codec(decoder_config.tokenizer),
                tokenizer=text_tokenizer,
                aux_runtime=aux_runtime,
                device=device,
                dtype=expert_dtype,
            )
        model = BijouModel(backbone=backbone, encoder=encoder, decoder=decoder)
    else:
        if sections.decoder is None:
            raise SystemExit(
                f"{checkpoint} is a format-1 (pre-prompt-section) "
                "checkpoint — refused with the pre-format-3 prompts "
                "(no back-compat, 2026-08-03)",
            )
        assert sections.prompt is not None  # tagged formats carry it
        expert_config = expert_config_from_architecture(
            sections.prompt,
            sections.decoder,
            load_config(checkpoint_dir),
        )
        model = from_backbone(
            checkpoint_dir,
            expert_config,
            device=device,
            dtype=dtype,
            expert_dtype=expert_dtype,
            attn_backend=attn_backend,
            max_soft_tokens=info.max_soft_tokens,
        )
    # CPU-load + copy-in for the same transient-memory reason as
    # load_adapted_backbone (the expert file is 1.6 GB fp32).
    model.decoder.load_state_dict(
        load_file(str(checkpoint / "expert.safetensors"), device="cpu"),
        strict=True,
    )
    # Prompt-side parameters (state_proj) — format-3 checkpoints always
    # carry them.
    model.encoder.load_state_dict(
        load_file(str(checkpoint / "prompt.safetensors"), device="cpu"),
        strict=True,
    )
    # Trunk-trained checkpoints (any --unfreeze-*-lr > 0) carry the adapted
    # backbone; checkpoints without the file load the HF backbone exactly
    # as before the file existed.
    if (checkpoint / "backbone.safetensors").exists():
        load_adapted_backbone(model, checkpoint)
        print(f"loaded adapted backbone from {checkpoint}", flush=True)
    model.eval()
    return model, info
