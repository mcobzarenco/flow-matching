"""Loading VLA checkpoints — the family registry plus the read-side
glue every loader shares.

A checkpoint directory (``metadata.json``, ``bijou/checkpoint.py``)
loads through the :data:`FAMILIES` registry via :func:`load_vla`,
which returns the family class recorded in the metadata. Legacy
``bijou_config.json`` directories do not load — ``bijou.convert_legacy``
converts them (and owns the frozen legacy reader).

The checkpoint SECTION schemas and section → module builders live in
``bijou.sections`` (re-exported here so existing import sites keep
working): the VLA metadata carries component configs verbatim as the
same tagged section dicts, so families, train's write side and the
converter all parse architecture through the SAME machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .checkpoint import read_metadata
from .data import DatasetStats
from .modelling.decoders.ar_suffix import ARDecoderConfig
from .modelling.decoders.flow import (
    SelfAttentionMode,
    TimeConditioning,
)
from .modelling.encoders.gemma4 import PROMPT_FORMAT
from .modelling.gemma4.loading import resolve_checkpoint_dir
from .models.gemma_ar import GemmaARVLA
from .models.gemma_flow import GemmaFlowVLA
from .models.molmo2_ar import Molmo2ARVLA
from .models.molmoact2_ar import MolmoAct2ARVLA
from .models.molmoact2_flow import MolmoAct2FlowVLA
from .models.molmoact2_joint import MolmoAct2JointVLA
from .sections import (
    BACKBONE_UNSAVED_KEYS,
    MOLMOACT2_FAST_TOKENIZER_REF,
    BackboneConfig,
    BackboneDepth,
    BackboneFiles,
    DecoderKind,
    FlowDecoderSection,
    GemmaPromptConfig,
    Molmo2PromptConfig,
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
    PromptKind,
    ar_backbone_config_from_dict,
    ar_backbone_config_to_dict,
    build_gemma_ar_decoder,
    build_gemma_encoder,
    build_gemma_flow_parts,
    build_molmo2_ar_decoder,
    build_molmo_flow_decoder,
    build_molmoact2_ar_decoder,
    decoder_schema_dict,
    default_expert_config,
    expert_config_from_architecture,
    flow_decoder_config_from_expert,
    load_backbone_part_states,
    molmoact2_ar_config_from_flow_section,
    molmoact2_fresh_flow_section,
    parse_decoder_config,
    parse_prompt_config,
    prefix_global_layers,
    resolve_action_codec,
    split_gemma_backbone_state,
    split_molmo2_backbone_state,
)
from .vla import VLA, VLAFamily

__all__ = [
    "BACKBONE_UNSAVED_KEYS",
    "FAMILIES",
    "MOLMOACT2_FAST_TOKENIZER_REF",
    "PROMPT_FORMAT",
    "ARDecoderConfig",
    "BackboneConfig",
    "BackboneDepth",
    "BackboneFiles",
    "CheckpointTrainArgs",
    "DecoderKind",
    "FlowDecoderSection",
    "GemmaPromptConfig",
    "Molmo2PromptConfig",
    "MolmoAct2PromptConfig",
    "MolmoFlowDecoderConfig",
    "PromptKind",
    "ar_backbone_config_from_dict",
    "ar_backbone_config_to_dict",
    "build_gemma_ar_decoder",
    "build_gemma_encoder",
    "build_gemma_flow_parts",
    "build_molmo2_ar_decoder",
    "build_molmo_flow_decoder",
    "build_molmoact2_ar_decoder",
    "decoder_schema_dict",
    "default_expert_config",
    "expert_config_from_architecture",
    "flow_decoder_config_from_expert",
    "load_backbone_part_states",
    "load_vla",
    "molmo_flow_state_table",
    "molmoact2_ar_config_from_flow_section",
    "molmoact2_fresh_flow_section",
    "parse_decoder_config",
    "parse_prompt_config",
    "prefix_global_layers",
    "resolve_action_codec",
    "resolve_checkpoint_dir",
    "split_gemma_backbone_state",
    "split_molmo2_backbone_state",
]


@dataclass(frozen=True, slots=True)
class CheckpointTrainArgs:
    """The architecture-determining subset of a checkpoint's recorded train
    args — every field a resumed/warm-started run needs to rebuild the
    model. The VLA metadata records ``train_args`` verbatim (converted
    legacy checkpoints keep their historical key spellings), and this
    parse normalizes both.

    This is the READ-side encoding of "what rebuilds the model"; the
    write side is bijou.train's architecture-flag partition (its
    checkpoint-inferred flags resolve from these fields under
    --resume/--init-from). tests/test_train_args.py pins the two
    encodings to each other so they cannot drift."""

    decoder: str
    decoder_hidden: int
    decoder_heads: int
    decoder_intermediate: int
    decoder_cross_heads: int
    stream_counts: tuple[int, ...]
    self_attention_mode: SelfAttentionMode
    chunk_size: int
    max_soft_tokens: int
    max_crops: int
    time_conditioning: TimeConditioning
    target_time_embed: bool
    fast_tokenizer: str | None
    # The molmoact2 objective matrix: which of the family's trained
    # pathways this run optimized — 'flow' (the molmo_flow expert, the
    # historical implicit value), 'ar' (the discrete head), 'joint'
    # (both, L_flow + λ·L_CE). 'flow' for every checkpoint predating
    # the field and for every non-molmoact2 run (the field is inert
    # off-family).
    objective: str = "flow"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointTrainArgs:
        # Old checkpoints recorded the decoder shape under the historical
        # "expert_*" arg names; both spellings load forever. Keys a
        # checkpoint predates read as the value its era implied (the
        # pre-AR era was flow-on-KV-exports with no riders).
        def either(new: str, old: str) -> Any:
            return data[new] if new in data else data[old]

        fast_tokenizer = data.get("fast_tokenizer")
        if data.get("conditioning_streams", "kv") == "residual" or bool(
            data.get("joint_ce", False),
        ):
            # Defense in depth: the prompt section's residual_exports
            # refusal fires first on every real artifact (residual and
            # joint-CE runs both recorded taps); this catches a
            # hand-edited or truncated config the same loud way.
            raise SystemExit(
                "this checkpoint trained with residual conditioning / the "
                "joint-CE arm — removed after molmo_flow superseded the "
                "flow-on-Molmo2 attachment; load it from git history at "
                "tag 'pre-decoder-simplify'",
            )
        return cls(
            decoder=str(data.get("decoder", "flow")),
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
            max_crops=int(data.get("max_crops", 1)),
            # Pre-adaRMS checkpoints (no such key) are additive.
            time_conditioning=TimeConditioning(
                data.get("time_conditioning", TimeConditioning.ADDITIVE.value),
            ),
            target_time_embed=bool(data.get("target_time_embed", False)),
            fast_tokenizer=None if fast_tokenizer is None else str(fast_tokenizer),
            objective=str(data.get("objective", "flow")),
        )


def molmoact2_action_table(normalization: DatasetStats) -> tuple[Tensor, Tensor]:
    """The ONE merged q01/q99 ACTION table (the molmoact2 ar/joint
    shared-table convention) as fp32 CPU tensors — the row CE targets
    tokenize under at training AND the row the AR block decode must
    detokenize under at serving. Eval builds its collator through this
    helper so the decode side can never fall back to per-item dataset
    quantiles (another rig's ranges — the sim100 token-leg seam,
    2026-08-17: a v2-table decode of merged-table tokens sign-inverted
    the lift channel)."""
    if normalization.action_q01 is None or normalization.action_q99 is None:
        raise SystemExit(
            "molmoact2 ar/joint serving needs action q01/q99 in the "
            "checkpoint's normalization table (the merged action scheme) "
            "— this table carries none",
        )
    return (
        torch.tensor(normalization.action_q01, dtype=torch.float32),
        torch.tensor(normalization.action_q99, dtype=torch.float32),
    )


def molmo_flow_state_table(normalization: DatasetStats) -> tuple[Tensor, Tensor]:
    """The merged q01/q99 STATE clamp table (§8.13) as the
    Collator's ``state_q01``/``state_q99`` pair — [state_dim] fp32 CPU
    tensors. molmo_flow state tokens are BINNED from this normalization
    (the MolmoAct2 encoder consumes clamp-normalized state), so a table
    without quantiles is a hard stop: falling back to per-sample
    mean/std would silently shift every state bin off its trained
    meaning. Train and eval both build their collators through this
    helper so the two sides cannot drift."""
    if normalization.state_q01 is None or normalization.state_q99 is None:
        raise SystemExit(
            "molmo_flow needs state q01/q99 in the checkpoint's "
            "normalization table (the merged state scheme, §8.13) — "
            "this table carries none",
        )
    return (
        torch.tensor(normalization.state_q01, dtype=torch.float32),
        torch.tensor(normalization.state_q99, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# The family registry — loading the VLA checkpoint format.

FAMILIES: dict[VLAFamily, type[VLA[Any]]] = {
    VLAFamily.GEMMA_FLOW: GemmaFlowVLA,
    VLAFamily.GEMMA_AR: GemmaARVLA,
    VLAFamily.MOLMO2_AR: Molmo2ARVLA,
    VLAFamily.MOLMOACT2_FLOW: MolmoAct2FlowVLA,
    VLAFamily.MOLMOACT2_AR: MolmoAct2ARVLA,
    VLAFamily.MOLMOACT2_JOINT: MolmoAct2JointVLA,
}


def load_vla(
    checkpoint: Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> VLA[Any]:
    """Load a VLA checkpoint directory (``metadata.json`` format) as the
    family class its metadata records — the ONE narrowing boundary where
    the batch-inputs pairing erases to ``VLA[Any]`` (consumers restore
    precision through generic functions and capability isinstance
    checks). ``dtype`` mounts the backbone; decoders and prompt-side
    parameters stay fp32 (the "new parameters" convention)."""
    metadata = read_metadata(checkpoint)
    return FAMILIES[metadata.family].from_checkpoint(
        checkpoint,
        device=device,
        dtype=dtype,
    )
