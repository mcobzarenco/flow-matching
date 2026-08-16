"""Checkpoint section schemas and their section → module builders.

A checkpoint records its architecture as tagged SECTION dicts — one per
role: ``backbone`` (which pretrained trunk, how deep), ``prompt`` (the
prompt-side strategy) and one per action decoder. This module owns those
records (parse/serialize) and the builders that turn a parsed section
into the corresponding module, so every reader of a checkpoint — the
family ``from_checkpoint`` constructors in ``bijou.models``, train's
write side, and ``bijou.convert_legacy`` — reconstructs architecture
through the SAME machinery and cannot drift.

Import DAG: ``loading`` → ``models/*`` → this module → ``modelling/*``.
Nothing here imports ``loading`` or ``models``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn

from .data import DatasetStats
from .fast.molmoact2 import MolmoAct2FastTokenizer
from .modelling.aux_text import AuxDecodeConfig, build_aux_runtime
from .modelling.codecs import FastActionCodec, MolmoAct2ActionCodec
from .modelling.decoders.ar_gemma import GemmaARDecoder
from .modelling.decoders.ar_molmo2 import Molmo2ARDecoder
from .modelling.decoders.ar_molmoact2 import MolmoAct2ARDecoder
from .modelling.decoders.ar_suffix import MOLMOACT2_SUFFIX_FORMAT, ARDecoderConfig
from .modelling.decoders.flow import (
    FlowDecoder,
    FlowDecoderConfig,
    SelfAttentionMode,
    TimeConditioning,
)
from .modelling.decoders.molmo_flow import (
    MolmoFlowConfig,
    MolmoFlowDecoder,
    MolmoFlowRuntime,
    TimeLaw,
)
from .modelling.encoders.gemma4 import PROMPT_FORMAT, GemmaEncoder
from .modelling.gemma4.config import Gemma4Config, Gemma4TextConfig, LayerType
from .modelling.gemma4.loading import (
    load_model,
    load_model_from_files,
    resolve_checkpoint_dir,
    truncate_backbone_state,
)
from .modelling.gemma4.model import Gemma4Model
from .modelling.interface import kv_stream_name
from .modelling.molmo2.config import Molmo2TextConfig
from .modelling.molmo2.model import Molmo2Model
from .modelling.molmo2.tokenizer import Molmo2TextTokenizer, newline_carrier_ids
from .modelling.nn import DEFAULT_ATTENTION_BACKEND, AttentionBackend, DeviceLike


class BackboneDepth(StrEnum):
    """How much of the backbone stack a checkpoint's model runs."""

    # Truncated to the non-KV-shared prefix (layers 0..14 for E2B) — the
    # cross-attention decoders' backbone; formats 1/2 are always this.
    PREFIX = "prefix"
    # The whole stack — the decoder-only path (prompt encode still stops
    # at the prefix; the suffix runs the KV-shared deep half).
    FULL = "full"


@dataclass(frozen=True, slots=True)
class BackboneFiles:
    """A VLA checkpoint's per-part trunk weight files: ``text`` = the
    text stack (+ untied lm_head on Molmo trunks), ``vision`` = tower +
    connector — exactly the ``backbone_vision`` LR group's members.
    Both carry OUR key names (translation happened once at import), so
    every load is a plain strict ``load_state_dict``."""

    text: Path
    vision: Path


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
    # The Molmo2-4B trunk port (Molmo2 port plan, 2026-08-06). The tag
    # is reserved so checkpoints written by port work packages have a
    # stable identity from day one; loading dispatches once the Molmo2
    # prompt config lands (WP4).
    MOLMO2 = "molmo2"
    # The MolmoAct2 prompt format (architecture.md §8.13):
    # their verbatim QA template, discrete state tokens, uint8
    # single-view images — written by the converter from day one; the
    # encoder mode lands with step 4.
    MOLMOACT2 = "molmoact2"


class DecoderKind(StrEnum):
    """Tag of an action-decoder config in bijou_config.json."""

    FLOW = "flow"
    AR_FAST = "ar_fast"
    AR_BACKBONE = "ar_backbone"
    # The MolmoAct2 action expert as a first-class decoder
    # (architecture.md §8.13): per-layer-KV-conditioned DiT, ascending-t
    # flow. Written by the converter (step 2); the decoder module lands
    # with step 3, model assembly with step 5.
    MOLMO_FLOW = "molmo_flow"


def _refuse_residual_exports(data: dict[str, Any], trunk: str) -> None:
    """Residual conditioning no longer exists (superseded by
    molmo_flow as the flow-on-Molmo2 story): a checkpoint recording
    non-empty ``residual_exports`` cannot rebuild at HEAD — refused by
    name rather than silently dropping its adapters (git tag
    'pre-decoder-simplify' still loads them). Absent/empty keys (every
    surviving checkpoint) parse unchanged."""
    taps = data.get("residual_exports", [])
    if taps:
        raise SystemExit(
            f"this {trunk} checkpoint records residual-tap exports "
            f"{list(taps)} — residual conditioning was removed after "
            "molmo_flow superseded it; load the checkpoint from git "
            "history at tag 'pre-decoder-simplify'",
        )


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
    and no checkpoint worth preserving does either."""

    exports: tuple[int, ...]
    max_soft_tokens: int
    format: int
    state_dim: int
    # Value-conditioning fields trained into the user turn's bracket
    # block (empty = unconditioned). Inference must render matching
    # conditioning — BijouPolicy reads this to configure its collator.
    condition_fields: tuple[str, ...]
    # Whether prompts carried the [generate|…] bracket. Implied for
    # ar_backbone (the request IS its interface); opt-in for other
    # decoders (--prompt-generate-bracket — stage-2 trunk consistency:
    # an AR-pretrained trunk shaped its conditioning/state positions
    # WITH the bracket present). Inference must render what training
    # rendered.
    generate_bracket: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": PromptKind.GEMMA4.value,
            "exports": list(self.exports),
            "max_soft_tokens": self.max_soft_tokens,
            "format": self.format,
            "state_dim": self.state_dim,
            "condition_fields": list(self.condition_fields),
            "generate_bracket": self.generate_bracket,
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
        _refuse_residual_exports(data, "gemma4")
        return cls(
            exports=tuple(int(layer) for layer in data["exports"]),
            max_soft_tokens=int(data["max_soft_tokens"]),
            format=format_version,
            state_dim=int(data["state_dim"]),
            condition_fields=tuple(
                str(field) for field in data.get("condition_fields", [])
            ),
            # Absent on checkpoints predating the flag (2026-08-04):
            # those flow prompts carried no bracket; ar_backbone
            # consumers OR this with their decoder kind.
            generate_bracket=bool(data.get("generate_bracket", False)),
        )

    @property
    def stream_names(self) -> tuple[str, ...]:
        """Index-aligned with ``self.exports``."""
        return tuple(kv_stream_name(layer) for layer in self.exports)


@dataclass(frozen=True, slots=True)
class Molmo2PromptConfig:
    """The Molmo2 prompt-side strategy as recorded in a checkpoint
    ``max_crops`` is the trunk's one image-budget
    knob (its dial is crops, not Gemma's ``max_soft_tokens`` — the two
    are deliberately not conflated); ``format`` is
    ``encoders.molmo2.MOLMO2_PROMPT_FORMAT`` (namespaced per trunk, not
    a Gemma bump). ``state_dim``/``condition_fields``/
    ``generate_bracket`` mirror the Gemma fields — BijouPolicy reads
    them to configure inference collation."""

    max_crops: int
    format: int
    state_dim: int
    condition_fields: tuple[str, ...]
    generate_bracket: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": PromptKind.MOLMO2.value,
            "max_crops": self.max_crops,
            "format": self.format,
            "state_dim": self.state_dim,
            "condition_fields": list(self.condition_fields),
            "generate_bracket": self.generate_bracket,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Molmo2PromptConfig:
        _refuse_residual_exports(data, "molmo2")
        return cls(
            max_crops=int(data["max_crops"]),
            format=int(data["format"]),
            state_dim=int(data["state_dim"]),
            condition_fields=tuple(
                str(field) for field in data.get("condition_fields", [])
            ),
            generate_bracket=bool(data.get("generate_bracket", False)),
        )


@dataclass(frozen=True, slots=True)
class MolmoAct2PromptConfig:
    """The MolmoAct2 prompt-side strategy as recorded in a checkpoint
    (§8.13): their verbatim template with the tag's
    ``setup_type``/``control_mode`` rendered in special-token wrappers,
    the q01/q99-normalized state as ``num_state_tokens``-bin discrete
    prompt tokens, and their uint8 single-view 378x378 image path.

    ``action_mode`` is the SOURCE checkpoint's mask flavor and is
    load-bearing for the expert weights: under ``'both'`` the encoder
    mask strips EOS positions (including the leading BOS, which IS
    ``<|im_end|>``) and discrete action spans from the expert's
    cross-attention context. ``n_obs_steps`` is asserted 1 at convert
    time and recorded for provenance."""

    format: int
    norm_tag: str
    setup_type: str
    control_mode: str
    num_state_tokens: int
    state_dim: int
    action_mode: str
    n_obs_steps: int
    camera_keys: tuple[str, ...]
    # The prefill split point (the narration switch): False = their serving
    # prompt verbatim (``<action_output>`` in the prefill, the parity
    # surface); True = the prefill stops at the ChatML opener and aux
    # text decodes as suffix. Converted checkpoints are always False
    # (their models never narrated); a narration-on RUN records True.
    narration: bool

    # The bracket/conditioning surfaces are bijou-prompt concepts; this
    # format has neither. Properties (not fields) so generic consumers
    # (train's checkpoint resolution, BijouPolicy) read them uniformly
    # without the schema pretending they are configurable.
    @property
    def condition_fields(self) -> tuple[str, ...]:
        return ()

    @property
    def generate_bracket(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": PromptKind.MOLMOACT2.value,
            "format": self.format,
            "norm_tag": self.norm_tag,
            "setup_type": self.setup_type,
            "control_mode": self.control_mode,
            "num_state_tokens": self.num_state_tokens,
            "state_dim": self.state_dim,
            "action_mode": self.action_mode,
            "n_obs_steps": self.n_obs_steps,
            "camera_keys": list(self.camera_keys),
            "narration": self.narration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MolmoAct2PromptConfig:
        return cls(
            format=int(data["format"]),
            norm_tag=str(data["norm_tag"]),
            setup_type=str(data["setup_type"]),
            control_mode=str(data["control_mode"]),
            num_state_tokens=int(data["num_state_tokens"]),
            state_dim=int(data["state_dim"]),
            action_mode=str(data["action_mode"]),
            n_obs_steps=int(data["n_obs_steps"]),
            camera_keys=tuple(str(key) for key in data["camera_keys"]),
            # Absent on step-2-era conversions == their serving layout.
            narration=bool(data.get("narration", False)),
        )


@dataclass(frozen=True, slots=True)
class MolmoFlowDecoderConfig:
    """The MolmoAct2 action expert as recorded in a checkpoint (§8.13):
    the expert geometry (mirrors the port's ``ActionExpertConfig``
    fields), the trunk KV width it conditions on (``llm_kv_dim`` —
    recorded so the decoder builds without re-deriving from the trunk
    config), their inference flow parameters, and the REAL action
    geometry of the tag (``action_dim`` of ``max_action_dim`` padded
    dims, ``action_horizon`` of ``max_horizon``).

    ``normalization`` is the decoder-owned scheme tag:
    ``"q01q99"`` = clamp-normalized targets against the checkpoint's
    ``normalization`` DatasetStats table (whose q01/q99 fields ARE the
    tag's merged table — stored once, not duplicated here)."""

    max_horizon: int
    max_action_dim: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    ffn_multiple_of: int
    timestep_embed_dim: int
    context_layer_norm: bool
    qk_norm: bool
    qk_norm_eps: float
    rope: bool
    causal_attn: bool
    llm_kv_dim: int
    num_flow_steps: int
    mask_action_dim_padding: bool
    action_dim: int
    action_horizon: int
    n_action_steps: int
    normalization: str
    # The training t-law, ``t = offset + scale·Beta(α, β)`` — recorded
    # from the source config (released: 0.001 + 0.999·Beta(1, 1.5)),
    # never assumed; the step-5 trainer samples exactly this law.
    time_offset: float
    time_scale: float
    beta_alpha: float
    beta_beta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": DecoderKind.MOLMO_FLOW.value,
            "max_horizon": self.max_horizon,
            "max_action_dim": self.max_action_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "ffn_multiple_of": self.ffn_multiple_of,
            "timestep_embed_dim": self.timestep_embed_dim,
            "context_layer_norm": self.context_layer_norm,
            "qk_norm": self.qk_norm,
            "qk_norm_eps": self.qk_norm_eps,
            "rope": self.rope,
            "causal_attn": self.causal_attn,
            "llm_kv_dim": self.llm_kv_dim,
            "num_flow_steps": self.num_flow_steps,
            "mask_action_dim_padding": self.mask_action_dim_padding,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "n_action_steps": self.n_action_steps,
            "normalization": self.normalization,
            "time_offset": self.time_offset,
            "time_scale": self.time_scale,
            "beta_alpha": self.beta_alpha,
            "beta_beta": self.beta_beta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MolmoFlowDecoderConfig:
        return cls(
            max_horizon=int(data["max_horizon"]),
            max_action_dim=int(data["max_action_dim"]),
            hidden_size=int(data["hidden_size"]),
            num_layers=int(data["num_layers"]),
            num_heads=int(data["num_heads"]),
            mlp_ratio=float(data["mlp_ratio"]),
            ffn_multiple_of=int(data["ffn_multiple_of"]),
            timestep_embed_dim=int(data["timestep_embed_dim"]),
            context_layer_norm=bool(data["context_layer_norm"]),
            qk_norm=bool(data["qk_norm"]),
            qk_norm_eps=float(data["qk_norm_eps"]),
            rope=bool(data["rope"]),
            causal_attn=bool(data["causal_attn"]),
            llm_kv_dim=int(data["llm_kv_dim"]),
            num_flow_steps=int(data["num_flow_steps"]),
            mask_action_dim_padding=bool(data["mask_action_dim_padding"]),
            action_dim=int(data["action_dim"]),
            action_horizon=int(data["action_horizon"]),
            n_action_steps=int(data["n_action_steps"]),
            normalization=str(data["normalization"]),
            time_offset=float(data["time_offset"]),
            time_scale=float(data["time_scale"]),
            beta_alpha=float(data["beta_alpha"]),
            beta_beta=float(data["beta_beta"]),
        )


@dataclass(frozen=True, slots=True)
class FlowDecoderSection:
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
    # SnapFlow φ_s target-time embedding; absent in checkpoints predating
    # the field (from_dict defaults False — they load unchanged).
    target_time_embed: bool = False

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
            "target_time_embed": self.target_time_embed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowDecoderSection:
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
            target_time_embed=bool(data.get("target_time_embed", False)),
        )


def parse_prompt_config(
    data: dict[str, Any],
) -> GemmaPromptConfig | Molmo2PromptConfig | MolmoAct2PromptConfig:
    kind = PromptKind(data["kind"])
    match kind:
        case PromptKind.GEMMA4:
            return GemmaPromptConfig.from_dict(data)
        case PromptKind.MOLMO2:
            return Molmo2PromptConfig.from_dict(data)
        case PromptKind.MOLMOACT2:
            return MolmoAct2PromptConfig.from_dict(data)


def ar_backbone_config_to_dict(config: ARDecoderConfig) -> dict[str, Any]:
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


def ar_backbone_config_from_dict(data: dict[str, Any]) -> ARDecoderConfig:
    aux = data.get("aux")  # absent in pre-aux checkpoints == null
    return ARDecoderConfig(
        tokenizer=str(data["tokenizer"]),
        vocab_total=int(data["vocab_total"]),
        block_base=int(data["block_base"]),
        chunk_size=int(data["chunk_size"]),
        action_dim=int(data["action_dim"]),
        # Absent = legacy pre-opener checkpoint (format 1) — refused by
        # ARDecoderConfig (formats < 5 have incompatible parameters).
        suffix_format=int(data.get("suffix_format", 1)),
        aux=None if aux is None else AuxDecodeConfig.from_dict(aux),
    )


def parse_decoder_config(
    data: dict[str, Any],
) -> FlowDecoderSection | ARDecoderConfig | MolmoFlowDecoderConfig:
    kind = DecoderKind(data["kind"])
    match kind:
        case DecoderKind.FLOW:
            return FlowDecoderSection.from_dict(data)
        case DecoderKind.AR_FAST:
            # Retired 2026-08-13: superseded by ar_backbone on quality,
            # parameter count and deployment (architecture.md §8.3).
            raise SystemExit(
                "this checkpoint records an ar_fast decoder — the ar_fast "
                "kind was removed after being superseded by ar_backbone; "
                "load it from git history at tag 'pre-decoder-simplify'",
            )
        case DecoderKind.AR_BACKBONE:
            return ar_backbone_config_from_dict(data)
        case DecoderKind.MOLMO_FLOW:
            return MolmoFlowDecoderConfig.from_dict(data)


def decoder_schema_dict(
    decoder: (
        FlowDecoder
        | GemmaARDecoder
        | Molmo2ARDecoder
        | MolmoAct2ARDecoder
        | MolmoFlowDecoder
    ),
) -> dict[str, Any]:
    """The checkpoint-schema dict of a built decoder (write side + the
    --init-from config guard). The Molmo2 suffix decoder records the SAME
    ar_backbone section (identical config shape) — the trunk axis lives
    in the PROMPT section's kind. molmo_flow returns the schema the
    loader stashed at build (this module owns the schema; the decoder
    cannot import it — geometry, t-law and action facts never change
    during an AE fine-tune, and the run's table lives in the metadata
    normalization row)."""
    match decoder:
        case MolmoFlowDecoder():
            if decoder.checkpoint_schema is None:
                raise ValueError(
                    "molmo_flow decoder has no stashed checkpoint schema — "
                    "was it built outside build_molmo_flow_decoder?",
                )
            return dict(decoder.checkpoint_schema)
        case FlowDecoder():
            return flow_decoder_config_from_expert(decoder.config).to_dict()
        case GemmaARDecoder() | Molmo2ARDecoder() | MolmoAct2ARDecoder():
            # The MolmoAct2 concrete records the SAME section shape —
            # suffix_format 6 + the released-tokenizer ref are the
            # discriminating fields; the trunk axis stays the PROMPT kind.
            return ar_backbone_config_to_dict(decoder.config)


def resolve_action_codec(ref: str) -> FastActionCodec:
    """Load OUR fitted FAST tokenizer artifact from a local directory or
    from the hub (``<user>/<repo>/<subfolder>``, e.g.
    mcobzarenco/bijou-checkpoints/fast_tokenizer_v1)."""
    local = Path(ref).expanduser()
    if (local / "fast_config.json").exists():
        return FastActionCodec.load(local)
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
    return FastActionCodec.load(Path(downloaded) / subfolder)


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
    target_time_embed: bool = False,
) -> FlowDecoderConfig:
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
    return FlowDecoderConfig(
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
        target_time_embed=target_time_embed,
    )


def build_gemma_encoder(
    weights: Path | BackboneFiles,
    config: Gemma4Config,
    *,
    tokenizer_dir: Path,
    exports: tuple[int, ...],
    max_soft_tokens: int,
    state_dim: int,
    device: DeviceLike,
    dtype: torch.dtype | None,
    attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
    depth: BackboneDepth = BackboneDepth.PREFIX,
    offload_ple: bool = False,
) -> tuple[Gemma4Model, GemmaEncoder]:
    """The Gemma backbone (frozen; truncated to its non-KV-shared layer
    prefix at the default PREFIX depth, whole stack at FULL) plus its
    prompt-side encoder strategy — the pair the composition root mounts.
    ``weights`` is an HF-layout artifact directory (fresh runs and
    importers — the translating dir-glob mount) or a VLA checkpoint's
    per-part files (plain strict loads); ``tokenizer_dir`` feeds the
    encoder's processor. The encoder's state_proj is freshly
    zero-initialized; checkpoint loads overwrite it from
    prompt.safetensors."""
    truncate_layers = (
        config.text.first_kv_shared_layer_idx if depth is BackboneDepth.PREFIX else None
    )
    if isinstance(weights, BackboneFiles):
        if offload_ple:
            raise SystemExit(
                "offload_ple is a serving mount knob of the HF-artifact "
                "path; checkpoint loads park the PLE table post-mount "
                "(to_device_with_ple_parked)",
            )
        backbone = load_model_from_files(
            config,
            text_file=weights.text,
            vision_file=weights.vision,
            device="cpu" if device is None else device,
            dtype=dtype,
            attn_backend=attn_backend,
            truncate_layers=truncate_layers,
        )
    else:
        backbone = load_model(
            weights,
            device="cpu" if device is None else device,
            dtype=dtype,
            attn_backend=attn_backend,
            truncate_layers=truncate_layers,
            offload_ple=offload_ple,
        )
    encoder = GemmaEncoder(
        backbone.config,
        exports=exports,
        processor_dir=str(tokenizer_dir),
        max_soft_tokens=max_soft_tokens,
        state_dim=state_dim,
        device=device,
        # The prompt-side params are "new parameters": fp32 like the
        # decoder's, whatever the backbone dtype (autocast covers the
        # forward).
        dtype=torch.float32,
    )
    return backbone, encoder


def build_gemma_flow_parts(
    weights: Path | BackboneFiles,
    config: Gemma4Config,
    expert_config: FlowDecoderConfig,
    *,
    tokenizer_dir: Path,
    max_soft_tokens: int,
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    expert_dtype: torch.dtype,
    attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
) -> tuple[Gemma4Model, GemmaEncoder, FlowDecoder]:
    """The gemma-flow assembly: validate the cross-attention streams
    against the backbone's global prefix layers, mount the truncated
    backbone + encoder pair (``weights``/``tokenizer_dir`` as in
    :func:`build_gemma_encoder`), and build a freshly-initialized flow
    decoder (checkpoint loads overwrite its weights)."""
    available = prefix_global_layers(config)
    for stream in expert_config.streams:
        if stream not in available:
            raise ValueError(
                f"cross-attention stream {stream} is not a global layer of "
                f"the backbone prefix (available: {available})",
            )
    backbone, encoder = build_gemma_encoder(
        weights,
        config,
        tokenizer_dir=tokenizer_dir,
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
    return backbone, encoder, decoder


def flow_decoder_config_from_expert(
    expert_config: FlowDecoderConfig,
) -> FlowDecoderSection:
    """FlowDecoderConfig → the checkpoint-schema decoder config: schedule ints
    become stream names; cross-attention head_dim/rope (encoder-declared
    geometry) drop out. Pure and total — the write-side half of the
    format-2 bridge."""
    return FlowDecoderSection(
        hidden_size=expert_config.hidden_size,
        num_attention_heads=expert_config.num_attention_heads,
        intermediate_size=expert_config.intermediate_size,
        hidden_activation=expert_config.hidden_activation,
        rms_norm_eps=expert_config.rms_norm_eps,
        self_attention_mode=expert_config.self_attention_mode,
        self_attention_rope_theta=expert_config.self_attention_rope_theta,
        cross_attention_heads=expert_config.cross_attention_heads,
        schedule=tuple(
            expert_config.stream_name(layer)
            for layer in expert_config.cross_attention_schedule
        ),
        action_dim=expert_config.action_dim,
        state_dim=expert_config.state_dim,
        chunk_size=expert_config.chunk_size,
        time_embed_dim=expert_config.time_embed_dim,
        time_conditioning=expert_config.time_conditioning,
        target_time_embed=expert_config.target_time_embed,
    )


def expert_config_from_architecture(
    prompt: GemmaPromptConfig,
    decoder: FlowDecoderSection,
    backbone_config: Gemma4Config,
) -> FlowDecoderConfig:
    """Compose the prompt + decoder configs back into the expert's
    construction config: schedule names resolve to backbone layer
    indices against the prompt section's K/V exports (``kv{i}``);
    cross-attention geometry comes from the backbone's global layers.
    Validates the references — an unknown stream name or an unconsumed
    export is a config error."""
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
    return FlowDecoderConfig(
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
        target_time_embed=decoder.target_time_embed,
    )


# The LM head is tied to the token embeddings (one storage, two state-dict
# keys): safetensors refuses aliased tensors, and Bijou never runs the head
# anyway — excluded on save, allowed to be exactly the missing key on load.
# Gemma-only: the Molmo2 head is untied and rides the text part for real.
BACKBONE_UNSAVED_KEYS = frozenset({"lm_head.weight"})


def split_gemma_backbone_state[T](
    state: dict[str, T],
) -> tuple[dict[str, T], dict[str, T]]:
    """Partition a mounted :class:`Gemma4Model` state dict into the
    checkpoint's (text, vision) part states: vision = ``vision_tower.*``
    (exactly the ``backbone_vision`` LR group — ``embed_vision`` trains
    under the TEXT lr and rides the text part), text = everything else
    minus the tied lm_head alias (:data:`BACKBONE_UNSAVED_KEYS`).
    Value-generic: callers split raw tensors or annotated entries."""
    text: dict[str, T] = {}
    vision: dict[str, T] = {}
    for key, value in state.items():
        if key.startswith("vision_tower."):
            vision[key] = value
        elif key not in BACKBONE_UNSAVED_KEYS:
            text[key] = value
    return text, vision


def split_molmo2_backbone_state[T](
    state: dict[str, T],
) -> tuple[dict[str, T], dict[str, T]]:
    """Partition a mounted :class:`Molmo2Model` state dict into the
    checkpoint's (text, vision) part states — the wrapper's ``text.``/
    ``vision.`` prefixes strip to the part files' submodule-level keys
    (``transformer.*`` + the untied ``lm_head.weight``; tower +
    connector). An unrecognized key is a wiring bug, refused by name.
    Value-generic: callers split raw tensors or annotated entries."""
    text: dict[str, T] = {}
    vision: dict[str, T] = {}
    for key, value in state.items():
        if key.startswith("text."):
            text[key.removeprefix("text.")] = value
        elif key.startswith("vision."):
            vision[key.removeprefix("vision.")] = value
        else:
            raise ValueError(
                f"Molmo2Model state key {key!r} is neither text.* nor "
                "vision.* — not a partitionable trunk state",
            )
    return text, vision


def load_backbone_part_states(backbone: nn.Module, files: BackboneFiles) -> None:
    """Load a checkpoint's per-part trunk files over an (already-built)
    backbone — the stage-2 inheritance path (the family loaders mount
    from files directly instead). The files are materialized on CPU and
    streamed into the live parameters by ``load_state_dict``'s copy
    semantics — loading straight to the target device would transiently
    hold a second full backbone (~4.3 GB) next to the built one, which
    OOMed the 8 GiB laptop GPU at rollout. The same copy semantics cast
    bf16 part files into whatever dtype the backbone was built with
    (bf16 for eval/rollout, fp32 masters for a live-backbone
    continuation). Gemma text parts saved at FULL depth load into
    truncated builds (stage-2: a flow prefix encoder inheriting an
    ar_backbone trunk): deeper layers are dropped and the packed
    per-layer-embedding tensors sliced to the kept layers — a no-op at
    matching depth."""
    text = load_file(str(files.text), device="cpu")
    vision = load_file(str(files.vision), device="cpu")
    if isinstance(backbone, Molmo2Model):
        # Molmo2 parts are full-model (untied head included — its key
        # is not the Gemma alias) and always full-depth; no truncation
        # arm exists.
        state = {f"text.{key}": value for key, value in text.items()}
        state.update({f"vision.{key}": value for key, value in vision.items()})
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(
                f"backbone part files mismatch at {files.text.parent}: "
                f"missing {list(missing)}, unexpected {list(unexpected)}",
            )
        return
    assert isinstance(backbone, Gemma4Model)  # the two trunk lineages
    state = truncate_backbone_state({**text, **vision}, backbone.config)
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    problems = [name for name in missing if name not in BACKBONE_UNSAVED_KEYS]
    if problems or unexpected:
        raise SystemExit(
            f"backbone part files mismatch at {files.text.parent}: "
            f"missing {problems}, unexpected {list(unexpected)}",
        )


def build_gemma_ar_decoder(
    tokenizer_dir: Path,
    config: ARDecoderConfig,
    text_config: Gemma4TextConfig,
    *,
    narration_weight: float = 1.0,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> GemmaARDecoder:
    """Format-5 AR section → the Gemma suffix decoder: the trunk's own
    text tokenizer (the artifact the collator rendered training text
    with — opener + value lines; format 5 always needs it at
    construction — read from the checkpoint's tokenizer/ or an artifact
    directory), the aux decode runtime when the section trained value
    lines, and the hub-routable codec resolution."""
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
    )
    aux_runtime = (
        build_aux_runtime(config.aux, text_tokenizer)
        if config.aux is not None
        else None
    )
    return GemmaARDecoder(
        config,
        text_config,
        resolve_action_codec(config.tokenizer),
        tokenizer=text_tokenizer,
        aux_runtime=aux_runtime,
        narration_weight=narration_weight,
        device=device,
        dtype=dtype,
    )


def build_molmo2_ar_decoder(
    tokenizer_ref: str,
    config: ARDecoderConfig,
    text_config: Molmo2TextConfig,
    *,
    narration_weight: float = 1.0,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
) -> Molmo2ARDecoder:
    """Format-5 AR section → the Molmo2 suffix decoder: the trunk's own
    ChatML tokenizer (``tokenizer.json`` in the checkpoint's tokenizer/
    or an artifact directory), the newline-carrier ban (Qwen merges
    ``…\\n`` pieces), the aux decode runtime when the section trained
    value lines, and the hub-routable codec resolution."""
    tokenizer = Molmo2TextTokenizer(tokenizer_ref)
    carriers = newline_carrier_ids(
        tokenizer,
        text_vocab_size=text_config.vocab_size,
        terminator_id=tokenizer.encode(
            "\n",
            add_special_tokens=False,
        )[0],
    )
    aux_runtime = (
        build_aux_runtime(
            config.aux,
            tokenizer,
            newline_carrier_ban=True,
        )
        if config.aux is not None
        else None
    )
    return Molmo2ARDecoder(
        config,
        text_config,
        resolve_action_codec(config.tokenizer),
        tokenizer=tokenizer,
        aux_runtime=aux_runtime,
        narration_weight=narration_weight,
        newline_carrier_ids=carriers,
        device=device,
        dtype=dtype,
    )


def build_molmo_flow_decoder(
    section: MolmoFlowDecoderConfig,
    normalization: DatasetStats,
    *,
    device: DeviceLike = None,
    dtype: torch.dtype | None = None,
) -> MolmoFlowDecoder:
    """The section → module bridge: build the expert from the recorded
    geometry, configure it with the tag's action geometry, recorded
    t-law and the q01/q99 clamp table (the checkpoint normalization
    row's quantile fields — stored once, §8.13), and stash
    the write-side schema dict (decoders cannot import this module).
    Weights are NOT loaded here — the caller owns that (converted
    checkpoints inject compat tensors; fresh sections would init)."""
    if normalization.action_q01 is None or normalization.action_q99 is None:
        raise SystemExit(
            "molmo_flow needs the q01/q99 quantile rows in the checkpoint "
            "normalization table (the decoder-owned clamp "
            "table) — this table predates them",
        )
    if len(normalization.action_q01) != section.action_dim:
        raise SystemExit(
            f"normalization action rows are {len(normalization.action_q01)}-wide "
            f"but the decoder section records action_dim={section.action_dim}",
        )
    config = MolmoFlowConfig(
        max_horizon=section.max_horizon,
        max_action_dim=section.max_action_dim,
        hidden_size=section.hidden_size,
        num_layers=section.num_layers,
        num_heads=section.num_heads,
        mlp_ratio=section.mlp_ratio,
        ffn_multiple_of=section.ffn_multiple_of,
        timestep_embed_dim=section.timestep_embed_dim,
        dropout=0.0,
        attn_dropout=0.0,
        context_layer_norm=section.context_layer_norm,
        qk_norm=section.qk_norm,
        qk_norm_eps=section.qk_norm_eps,
        rope=section.rope,
        causal_attn=section.causal_attn,
        llm_kv_dim=section.llm_kv_dim,
    )
    decoder = config.build()
    decoder.configure(
        MolmoFlowRuntime(
            action_dim=section.action_dim,
            action_horizon=section.action_horizon,
            n_action_steps=section.n_action_steps,
            num_flow_steps=section.num_flow_steps,
            mask_action_dim_padding=section.mask_action_dim_padding,
            time_law=TimeLaw(
                offset=section.time_offset,
                scale=section.time_scale,
                beta_alpha=section.beta_alpha,
                beta_beta=section.beta_beta,
            ),
        ),
        action_q01=torch.tensor(normalization.action_q01, dtype=torch.float32),
        action_q99=torch.tensor(normalization.action_q99, dtype=torch.float32),
        checkpoint_schema=section.to_dict(),
    )
    if dtype is not None:
        decoder = decoder.to(dtype)
    if device is not None:
        decoder = decoder.to(device)
    return decoder


# The released FAST action-tokenizer artifact — the discrete head's
# codec. AR/joint checkpoints record their own ref
# (ARDecoderConfig.tokenizer); RELEASE-class checkpoints predate the
# discrete read, so the AR read of one defaults to the canonical
# artifact (the box snapshot pins d45593b4c8…).
MOLMOACT2_FAST_TOKENIZER_REF = "allenai/MolmoAct2-FAST-Tokenizer"


def build_molmoact2_ar_decoder(
    config: ARDecoderConfig,
    prompt: MolmoAct2PromptConfig,
    text_config: Molmo2TextConfig,
    tokenizer_ref: str,
) -> MolmoAct2ARDecoder:
    """Format-6 AR section → the discrete-head decoder: the
    action-mode refusal (BY NAME — the rig-ft exports are
    'continuous', their fine-tune never trained the discrete head),
    hub-routable codec resolution, then the constructor's own id-space
    guards against the trunk tokenizer. Shared by the legacy and the
    family ``from_checkpoint`` paths (format-6 checkpoints) and the
    release-read path (probes, ``--objective ar/joint`` init)."""
    if prompt.action_mode != "both":
        raise SystemExit(
            f"the discrete pathway exists on action_mode='both' "
            f"checkpoints; this one records {prompt.action_mode!r} — the "
            "rig-ft exports are 'continuous' (their fine-tune never "
            "trained the discrete head; the reference decode refuses "
            "them too)",
        )
    fast = MolmoAct2FastTokenizer.load(resolve_checkpoint_dir(config.tokenizer))
    codec = MolmoAct2ActionCodec(
        fast,
        time_horizon=config.chunk_size,
        action_dim=config.action_dim,
    )
    return MolmoAct2ARDecoder(
        config,
        text_config,
        codec,
        tokenizer=Molmo2TextTokenizer(tokenizer_ref),
    )


def molmoact2_ar_config_from_flow_section(
    flow: MolmoFlowDecoderConfig,
    prompt: MolmoAct2PromptConfig,
    tokenizer_ref: str,
    *,
    fast_tokenizer: str = MOLMOACT2_FAST_TOKENIZER_REF,
) -> ARDecoderConfig:
    """The discrete (AR) read of a RELEASE-class checkpoint, whose
    decoder section is molmo_flow and which records no format-6
    section: geometry from the flow section — refusing non-identity
    output tails loudly (their tail slices ``[n_obs_steps-1 :
    n_obs_steps-1+n_action_steps]`` of the decoded horizon; the
    first-class decode budget IS the executed chunk, and the release
    is identity: n_obs_steps 1, n_action_steps == horizon 30) — block
    width from the released artifact, ``block_base`` from the trunk
    tokenizer's own ``<action_0>`` row (the decoder's constructor
    re-verifies it against ``<action_start>``/``<action_end>``)."""
    if prompt.n_obs_steps != 1 or flow.n_action_steps != flow.action_horizon:
        raise SystemExit(
            f"non-identity discrete output tail (n_obs_steps "
            f"{prompt.n_obs_steps}, n_action_steps {flow.n_action_steps} "
            f"of horizon {flow.action_horizon}) — the first-class decode "
            "budget IS the executed chunk; this checkpoint's tail has no "
            "first-class consumer",
        )
    fast = MolmoAct2FastTokenizer.load(resolve_checkpoint_dir(fast_tokenizer))
    backend = Molmo2TextTokenizer(tokenizer_ref).tokenizer
    block_base = backend.token_to_id("<action_0>")
    if block_base is None:
        raise SystemExit(
            f"{tokenizer_ref} tokenizer has no <action_0> — not a "
            "MolmoAct2 'both'-mode vocabulary",
        )
    return ARDecoderConfig(
        tokenizer=fast_tokenizer,
        vocab_total=fast.block_vocab,
        block_base=int(block_base),
        chunk_size=flow.action_horizon,
        action_dim=flow.action_dim,
        suffix_format=MOLMOACT2_SUFFIX_FORMAT,
        aux=None,
    )


def molmoact2_fresh_flow_section(
    ar_config: ARDecoderConfig,
) -> MolmoFlowDecoderConfig:
    """The molmo_flow section for ``--expert-init fresh`` from an
    ar-only source (which carries no expert to inherit): the released
    expert ARCHITECTURE (:meth:`MolmoFlowConfig.released_so100_101` —
    the literals' home) plus the released serving/t-law constants
    (num_flow_steps 10, mask_action_dim_padding, t = 0.001 +
    0.999·Beta(1, 1.5) — the convert-time reads of their config,
    oracle-pinned in test_convert_molmoact2), geometry from the
    format-6 section (identity output tail by construction: the decode
    budget IS the executed chunk). The q01/q99 clamp table binds at
    build time from the run's normalization row, as for every
    molmo_flow section."""
    shape = MolmoFlowConfig.released_so100_101()
    if ar_config.chunk_size > shape.max_horizon:
        raise SystemExit(
            f"chunk {ar_config.chunk_size} exceeds the released expert's "
            f"max_horizon {shape.max_horizon} — a fresh released-shape "
            "expert cannot host this geometry",
        )
    return MolmoFlowDecoderConfig(
        max_horizon=shape.max_horizon,
        max_action_dim=shape.max_action_dim,
        hidden_size=shape.hidden_size,
        num_layers=shape.num_layers,
        num_heads=shape.num_heads,
        mlp_ratio=shape.mlp_ratio,
        ffn_multiple_of=shape.ffn_multiple_of,
        timestep_embed_dim=shape.timestep_embed_dim,
        context_layer_norm=shape.context_layer_norm,
        qk_norm=shape.qk_norm,
        qk_norm_eps=shape.qk_norm_eps,
        rope=shape.rope,
        causal_attn=shape.causal_attn,
        llm_kv_dim=shape.llm_kv_dim,
        num_flow_steps=10,
        mask_action_dim_padding=True,
        action_dim=ar_config.action_dim,
        action_horizon=ar_config.chunk_size,
        n_action_steps=ar_config.chunk_size,
        normalization="q01q99",
        time_offset=0.001,
        time_scale=0.999,
        beta_alpha=1.0,
        beta_beta=1.5,
    )
