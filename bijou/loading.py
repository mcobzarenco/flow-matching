"""Building a Bijou VLA from a Gemma 4 checkpoint.

``from_backbone("google/gemma-4-e2b-it", action_dim=6, state_dim=6)`` gives a
ready-to-train model:

- the backbone is loaded truncated to its non-KV-shared prefix (layers 0–14
  for E2B) with only the needed weights: dropped decoder layers, per-layer
  embedding (PLE) tensors sliced to the kept layers, no audio tower, and the
  LM head is a tie (zero extra memory). ~2.1B params instead of 5.2B for E2B.
  It is frozen (eval mode, ``requires_grad=False``).
- the action expert is freshly initialized on the same device/dtype.

Two checkpoint worlds load through this module:

- the LEGACY ``bijou_config.json`` layout via :func:`from_checkpoint`
  (returns the ``BijouModel`` composition root);
- the VLA format (``metadata.json``, ``bijou/checkpoint.py``) via the
  :data:`FAMILIES` registry and :func:`load_vla` (returns the family
  class recorded in the metadata).

The checkpoint SECTION schemas and section → module builders both
worlds share live in ``bijou.sections`` (re-exported here so existing
import sites keep working).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor

from .checkpoint import read_metadata
from .data import DatasetStats
from .model import BijouModel
from .modelling.decoders.ar_suffix import MOLMOACT2_SUFFIX_FORMAT, ARDecoderConfig
from .modelling.decoders.flow import (
    FlowDecoderConfig,
    SelfAttentionMode,
    TimeConditioning,
)
from .modelling.decoders.molmo_flow import load_expert_state
from .modelling.encoders.gemma4 import PROMPT_FORMAT
from .modelling.encoders.molmo2 import Molmo2Encoder
from .modelling.encoders.molmoact2 import MolmoAct2Encoder
from .modelling.gemma4.config import Gemma4Config
from .modelling.gemma4.loading import load_config, resolve_checkpoint_dir
from .modelling.molmo2.loading import load_config as load_molmo2_config
from .modelling.molmo2.model import load_model as load_molmo2_model
from .modelling.nn import DEFAULT_ATTENTION_BACKEND, AttentionBackend, DeviceLike
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
    load_backbone_state,
    molmoact2_ar_config_from_flow_section,
    molmoact2_fresh_flow_section,
    parse_decoder_config,
    parse_prompt_config,
    prefix_global_layers,
    resolve_action_codec,
)
from .vla import VLA, VLAFamily

__all__ = [
    "BACKBONE_UNSAVED_KEYS",
    "CHECKPOINT_FORMAT",
    "FAMILIES",
    "MOLMOACT2_FAST_TOKENIZER_REF",
    "PROMPT_FORMAT",
    "ARDecoderConfig",
    "BackboneConfig",
    "BackboneDepth",
    "CheckpointInfo",
    "CheckpointMetadata",
    "CheckpointSections",
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
    "backbone_snapshot",
    "build_gemma_ar_decoder",
    "build_gemma_encoder",
    "build_gemma_flow_parts",
    "build_molmo2_ar_decoder",
    "build_molmo_flow_decoder",
    "build_molmoact2_ar_decoder",
    "checkpoint_sections",
    "decoder_schema_dict",
    "default_expert_config",
    "expert_config_from_architecture",
    "expert_config_from_train_args",
    "flow_decoder_config_from_expert",
    "from_backbone",
    "from_checkpoint",
    "load_adapted_backbone",
    "load_backbone_init",
    "load_backbone_state",
    "load_vla",
    "molmo_flow_state_table",
    "molmoact2_ar_config_from_flow_section",
    "molmoact2_fresh_flow_section",
    "parse_decoder_config",
    "parse_prompt_config",
    "prefix_global_layers",
    "read_checkpoint_info",
    "resolve_action_codec",
    "resolve_checkpoint_dir",
]

# bijou_config.json schema version. Format 3 sections the metadata by
# role — backbone (the shared network), prompt (the prompt-side
# strategy), decoder (the tagged head config) — replacing format 2's
# encoder/decoder pair, which had replaced the original layout (backbone
# + expert_config keys). The read side synthesizes current semantics
# from BOTH older formats, so every existing checkpoint keeps loading
# without conversion.
CHECKPOINT_FORMAT = 3


def from_backbone(
    model_id_or_path: str | Path,
    expert_config: FlowDecoderConfig | None = None,
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

    if expert_dtype is None:
        expert_dtype = dtype if dtype is not None else config.dtype
    backbone, encoder, decoder = build_gemma_flow_parts(
        checkpoint_dir,
        config,
        expert_config,
        max_soft_tokens=max_soft_tokens,
        device=device,
        dtype=dtype,
        expert_dtype=expert_dtype,
        attn_backend=attn_backend,
    )
    return BijouModel(backbone=backbone, encoder=encoder, decoder=decoder)


@dataclass(frozen=True, slots=True)
class CheckpointTrainArgs:
    """The architecture-determining subset of a checkpoint's recorded train
    args — every field a loader (or a resumed/warm-started run) needs to
    rebuild the model. Present in all checkpoints since the format's
    introduction; newer recorded args (lr, seeds, data paths, ...) stay in
    the raw JSON but are not needed here.

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
    # The molmoact2 objective matrix (retirement phase 3): which of the
    # family's trained pathways this run optimized — 'flow' (the
    # molmo_flow expert, the historical implicit value), 'ar' (the
    # discrete head), 'joint' (both, L_flow + λ·L_CE). 'flow' for every
    # checkpoint predating the field and for every non-molmoact2 run
    # (the field is inert off-family).
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
    # Whether training prompts carried [generate|…] (False on
    # checkpoints predating the flag — ar_backbone consumers OR with
    # their decoder kind, where the bracket is implied).
    generate_bracket: bool

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
    prompt: GemmaPromptConfig | Molmo2PromptConfig | MolmoAct2PromptConfig
    decoder: dict[str, Any]
    normalization: DatasetStats
    per_dataset_normalization: dict[str, DatasetStats]
    train_args: dict[str, Any]
    step: int
    # The joint flow+CE rider's decoder schema; its
    # weights ride as ``joint_ce.safetensors``. None on
    # every non-joint checkpoint — the key is then absent, so files
    # round-trip byte-identically to the pre-field format.
    joint_ce: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "backbone": self.backbone.to_dict(),
            "prompt": self.prompt.to_dict(),
            "decoder": self.decoder,
            **({"joint_ce": self.joint_ce} if self.joint_ce is not None else {}),
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
    prompt: GemmaPromptConfig | Molmo2PromptConfig | MolmoAct2PromptConfig | None
    decoder: FlowDecoderSection | ARDecoderConfig | MolmoFlowDecoderConfig | None


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
) -> FlowDecoderConfig:
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
    backbone (see :func:`bijou.sections.load_backbone_state` for the
    CPU-materialization and depth-slicing contract)."""
    load_backbone_state(model.backbone, checkpoint)


def load_backbone_init(model: BijouModel, checkpoint: Path) -> None:
    """Stage-2 warm start: inherit ONLY the backbone and the prompt-side
    parameters (state_proj) from a checkpoint, leaving the decoder as
    built — the point is mounting a DIFFERENT decoder family (e.g. a
    fresh flow expert) on an adapted trunk, so there is deliberately no
    decoder-config match check here. Loud when the checkpoint carries no
    adapted backbone: inheriting a pristine trunk would silently run the
    stock-backbone arm of an ablation twice."""
    snapshot = checkpoint / "backbone.safetensors"
    if not snapshot.exists():
        raise SystemExit(
            f"--backbone-init-from {checkpoint}: no backbone.safetensors — "
            "that checkpoint's backbone is pristine HF, so there is nothing "
            "to inherit (for a stock backbone simply omit the flag)",
        )
    load_adapted_backbone(model, checkpoint)
    # Prompt-side parameters (state_proj) travel with the trunk: the
    # backbone was adapted WITH this projection feeding its prompt —
    # loading one without the other would shift the trunk's input
    # distribution. Format-3 checkpoints always write the file.
    model.encoder.load_state_dict(
        load_file(str(checkpoint / "prompt.safetensors"), device="cpu"),
        strict=True,
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


def read_checkpoint_info(checkpoint: str | Path) -> CheckpointInfo:
    """Parse a checkpoint directory's ``bijou_config.json`` into
    :class:`CheckpointInfo` — metadata only, no weights touched. The
    train CLI uses this to resolve checkpoint-inferred architecture
    flags under --resume/--init-from before any model is built;
    :func:`from_checkpoint` composes it with the weight load."""
    checkpoint = Path(checkpoint)
    meta = json.loads((checkpoint / "bijou_config.json").read_text())
    sections = checkpoint_sections(meta)
    return CheckpointInfo(
        backbone=sections.backbone.id,
        condition_fields=(
            sections.prompt.condition_fields if sections.prompt is not None else ()
        ),
        generate_bracket=(
            sections.prompt.generate_bracket if sections.prompt is not None else False
        ),
        train_args=CheckpointTrainArgs.from_dict(meta["train_args"]),
        step=int(meta["step"]),
        normalization=DatasetStats.from_state_dict(meta["normalization"]),
        per_dataset_normalization={
            repo_id: DatasetStats.from_state_dict(entry)
            for repo_id, entry in meta.get("per_dataset_normalization", {}).items()
        },
    )


def from_checkpoint(
    checkpoint: str | Path,
    *,
    device: DeviceLike = "cpu",
    dtype: torch.dtype | None = None,
    expert_dtype: torch.dtype = torch.float32,
    attn_backend: AttentionBackend = DEFAULT_ATTENTION_BACKEND,
    offload_ple: bool = False,
) -> tuple[BijouModel, CheckpointInfo]:
    """Load a bijou training checkpoint directory (as written by
    bijou.train.save_checkpoint): backbone resolved from the recorded id,
    expert config rebuilt from the recorded train args, expert weights
    loaded strictly. Returns the eval-mode model plus checkpoint metadata
    (normalization stats table etc.). ``offload_ple`` parks the PLE
    token table in host RAM (full-depth ar_backbone checkpoints on
    small GPUs — see gemma4.loading.load_model); prefix-depth
    checkpoints refuse it loudly rather than ignore it."""
    checkpoint = Path(checkpoint)
    meta = json.loads((checkpoint / "bijou_config.json").read_text())
    sections = checkpoint_sections(meta)
    info = read_checkpoint_info(checkpoint)
    if isinstance(sections.decoder, MolmoFlowDecoderConfig) or isinstance(
        sections.prompt,
        MolmoAct2PromptConfig,
    ):
        if not isinstance(sections.prompt, MolmoAct2PromptConfig):
            raise SystemExit(
                f"{checkpoint} records the molmo_flow decoder under a "
                f"{type(sections.prompt).__name__} — molmo_flow rides the "
                "molmoact2 prompt format only (§8.13)",
            )
        if offload_ple:
            raise SystemExit(
                "--offload-ple parks Gemma's PLE token table; molmo2-family "
                "trunks have no PLE — drop the flag",
            )
        if isinstance(sections.decoder, ARDecoderConfig):
            # The discrete-head arm (retirement phase 2): format-6
            # ar_backbone section + molmoact2 prompt. Cheap refusals
            # BEFORE any trunk download/mount.
            if sections.decoder.suffix_format != MOLMOACT2_SUFFIX_FORMAT:
                raise SystemExit(
                    f"{checkpoint} pairs a format-"
                    f"{sections.decoder.suffix_format} ar_backbone section "
                    "with the molmoact2 prompt — this prompt family's "
                    f"emission is format {MOLMOACT2_SUFFIX_FORMAT} "
                    "(value-line checkpoints ride the Gemma/Molmo2 "
                    "prompts)",
                )
            if (checkpoint / "expert.safetensors").exists():
                raise SystemExit(
                    f"{checkpoint} is an ar-only molmoact2 checkpoint but "
                    "carries expert.safetensors — the discrete head owns "
                    "no parameters; joint checkpoints record the "
                    "molmo_flow decoder section (with the AR rider in "
                    "joint_ce) instead",
                )
            ar_trunk_dir = resolve_checkpoint_dir(info.backbone)
            ar_decoder = build_molmoact2_ar_decoder(
                sections.decoder,
                sections.prompt,
                load_molmo2_config(ar_trunk_dir).text,
                info.backbone,
            )
            backbone = load_molmo2_model(
                ar_trunk_dir,
                device="cpu" if device is None else device,
                dtype=torch.bfloat16 if dtype is None else dtype,
            )
            ar_encoder = MolmoAct2Encoder(
                info.backbone,
                setup_type=sections.prompt.setup_type,
                control_mode=sections.prompt.control_mode,
                num_state_tokens=sections.prompt.num_state_tokens,
                action_mode=sections.prompt.action_mode,
                narration=sections.prompt.narration,
            )
            model = BijouModel(
                backbone=backbone,
                encoder=ar_encoder,
                decoder=ar_decoder,
            )
            # No expert.safetensors (guarded above) and no
            # prompt.safetensors (the encoder owns nothing); trunk
            # deltas ride the standard file.
            if (checkpoint / "backbone.safetensors").exists():
                load_adapted_backbone(model, checkpoint)
                print(f"loaded adapted backbone from {checkpoint}", flush=True)
            model.eval()
            return model, info
        if not isinstance(sections.decoder, MolmoFlowDecoderConfig):
            raise SystemExit(
                f"{checkpoint} records a "
                f"{type(sections.decoder).__name__} under the molmoact2 "
                "prompt — this family carries molmo_flow (the expert "
                "pathway) or a format-6 ar_backbone section (the "
                "discrete head)",
            )
        trunk_dir = resolve_checkpoint_dir(info.backbone)
        backbone = load_molmo2_model(
            trunk_dir,
            device="cpu" if device is None else device,
            dtype=torch.bfloat16 if dtype is None else dtype,
        )
        molmo_flow_encoder = MolmoAct2Encoder(
            info.backbone,
            setup_type=sections.prompt.setup_type,
            control_mode=sections.prompt.control_mode,
            num_state_tokens=sections.prompt.num_state_tokens,
            action_mode=sections.prompt.action_mode,
            narration=sections.prompt.narration,
        )
        molmo_flow = build_molmo_flow_decoder(
            sections.decoder,
            info.normalization,
            device=device,
            dtype=expert_dtype,
        )
        load_expert_state(
            molmo_flow,
            load_file(str(checkpoint / "expert.safetensors"), device="cpu"),
        )
        molmo_flow.to(device=device, dtype=expert_dtype)
        model = BijouModel(
            backbone=backbone,
            encoder=molmo_flow_encoder,
            decoder=molmo_flow,
        )
        # Joint checkpoints (retirement phase 3) carry the discrete
        # head's format-6 section in the joint_ce slot — mount the
        # parameterless rider so --resume rebuilds the composition and
        # the AR read of a joint checkpoint stays one construction away.
        joint_section = meta.get("joint_ce")
        if joint_section is not None:
            rider_config = ar_backbone_config_from_dict(joint_section)
            model.joint_ce = build_molmoact2_ar_decoder(
                rider_config,
                sections.prompt,
                load_molmo2_config(trunk_dir).text,
                info.backbone,
            )
        # No prompt.safetensors (the encoder has zero parameters) and no
        # backbone.safetensors (the trunk IS the recorded artifact —
        # conversion invariant); a future trunk-trained molmo_flow run
        # will write and reload one through the standard path below.
        if (checkpoint / "backbone.safetensors").exists():
            load_adapted_backbone(model, checkpoint)
            print(f"loaded adapted backbone from {checkpoint}", flush=True)
        model.eval()
        return model, info
    checkpoint_dir = resolve_checkpoint_dir(info.backbone)
    if isinstance(sections.prompt, Molmo2PromptConfig):
        if not isinstance(sections.decoder, ARDecoderConfig):
            # Flow-on-molmo2 was the residual-conditioning arm — removed
            # 2026-08-13 (superseded by molmo_flow); its checkpoints are
            # already refused by the prompt section's residual_exports
            # guard, so this arm names the general rule.
            raise SystemExit(
                f"{checkpoint} is a molmo2 checkpoint with a "
                f"{type(sections.decoder).__name__} decoder — this trunk "
                "supports the ar_backbone (suffix) decoder only (flow on "
                "Molmo2 is molmo_flow, §8.13)",
            )
        if offload_ple:
            raise SystemExit(
                "--offload-ple parks Gemma's PLE token table; molmo2 has "
                "no PLE — drop the flag",
            )
        molmo2_config = load_molmo2_config(checkpoint_dir)
        # The release ships fp32; bf16 is the mount convention for
        # eval/rollout (an explicit dtype — fp32 masters for a
        # continuation — wins).
        backbone = load_molmo2_model(
            checkpoint_dir,
            device="cpu" if device is None else device,
            dtype=torch.bfloat16 if dtype is None else dtype,
        )
        encoder = Molmo2Encoder(
            info.backbone,
            max_crops=sections.prompt.max_crops,
            state_dim=sections.prompt.state_dim,
            hidden_size=molmo2_config.text.hidden_size,
            device=device,
            dtype=expert_dtype,
        )
        molmo2_decoder = build_molmo2_ar_decoder(
            info.backbone,
            sections.decoder,
            molmo2_config.text,
            device=device,
            dtype=expert_dtype,
        )
        model = BijouModel(
            backbone=backbone,
            encoder=encoder,
            decoder=molmo2_decoder,
        )
    elif isinstance(sections.decoder, ARDecoderConfig):
        decoder_config = sections.decoder
        assert isinstance(sections.prompt, GemmaPromptConfig)  # molmo2 handled above
        backbone_config = load_config(checkpoint_dir)
        if sections.backbone.depth is not BackboneDepth.FULL:
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
            offload_ple=offload_ple,
        )
        decoder = build_gemma_ar_decoder(
            checkpoint_dir,
            decoder_config,
            backbone_config.text,
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
        if offload_ple:
            raise SystemExit(
                "--offload-ple targets full-depth ar_backbone checkpoints "
                "(9.6 GB bf16); this prefix-depth checkpoint fits small "
                "GPUs without it — drop the flag",
            )
        assert isinstance(sections.prompt, GemmaPromptConfig)  # molmo2 handled above
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
