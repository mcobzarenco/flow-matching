"""Train-run configuration: the CLI flag surface and its resolution.

:class:`TrainArgs` is the resolved, validated configuration of one
training run. Parsing is two-layered: ``from_namespace`` owns every
rule that needs flag EXPLICITNESS or a checkpoint -- the ``--family``
requirement/refusal, the checkpoint-inferred flag refusals
(``ARCH_FLAGS``), sentinel resolution (fresh-run ``ARCH_DEFAULTS`` vs
the checkpoint's recorded architecture, with ``--objective`` selecting
a molmoact2 pathway under ``--init-from``), and the "flag X requires
flag Y" checks -- while value invariants of the RESOLVED config live
once in ``__post_init__`` (``from_namespace`` translates them to
``parser.error``, so the CLI keeps its usage-line UX and direct
construction can never build an invalid config).

:func:`reconcile_lr_offer` is the build-time half of the LR-vs-offer
reconciliation: LR flags against the built model's structural
``param_groups()`` offer, both directions -- an LR flag for an empty
offered group is a contradiction and dies loudly; an offered group
without its flag freezes with a printed note, never silently.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from ..annotations import ConditionField
from ..checkpoint import read_metadata
from ..data import parse_repeat_specs
from ..loading import CheckpointTrainArgs
from ..modelling.aux_text import AuxField
from ..modelling.decoders.flow import TimeConditioning
from ..models.objectives import SnapflowObjective
from ..sections import parse_prompt_config
from ..vla import VLAFamily

DEFAULT_BACKBONE = "google/gemma-4-e2b-it"


#: Families whose action decoder is the trunk-suffix role (they consume
#: --fast-tokenizer and may train aux value lines).
AR_SUFFIX_FAMILIES = (VLAFamily.GEMMA_AR.value, VLAFamily.MOLMO2_AR.value)
#: Families riding the MolmoAct2 prompt format — inherit-only (they
#: train from a converted checkpoint; --objective selects the pathway
#: under --init-from).
MOLMOACT2_FAMILIES = (
    VLAFamily.MOLMOACT2_FLOW.value,
    VLAFamily.MOLMOACT2_AR.value,
    VLAFamily.MOLMOACT2_JOINT.value,
)


class ArchSection(StrEnum):
    """Which checkpoint section an architecture flag belongs to — the
    unit of inheritance under --init-from (see ARCH_FLAGS)."""

    BACKBONE = "backbone"
    PROMPT = "prompt"
    DECODER = "decoder"
    # Zero-init structure additions whose extended model IS the source
    # checkpoint until trained (φ_s), plus the molmoact2 objective
    # selector: legal to declare on --init-from, refused on --resume
    # (the optimizer param groups / recorded objective would not match).
    EXTENSION = "extension"


#: The checkpoint-inferred architecture flags: raw-namespace attribute ->
#: (CLI flag, section). Under --resume ALL of these are refused as flags
#: and resolve from the checkpoint; under --init-from, BACKBONE/PROMPT/
#: DECODER flags are refused (inherited sections — a fresh decoder on an
#: inherited trunk is --backbone-init-from + --family), EXTENSION flags
#: stay legal. ``--family`` itself is handled beside this table: required
#: on fresh runs, refused whenever a checkpoint is given (metadata is the
#: source). Every entry except ``objective`` is a TrainArgs field; the
#: read side is the checkpoint metadata (family, objective) plus
#: loading.CheckpointTrainArgs (the recorded train_args) —
#: tests/test_train_args.py pins the encodings to each other.
ARCH_FLAGS: dict[str, tuple[str, ArchSection]] = {
    "backbone": ("--backbone", ArchSection.BACKBONE),
    "max_soft_tokens": ("--max-soft-tokens", ArchSection.PROMPT),
    "max_crops": ("--max-crops", ArchSection.PROMPT),
    "prompt_generate_bracket": (
        "--prompt-generate-bracket",
        ArchSection.PROMPT,
    ),
    "decoder_hidden": ("--decoder-hidden", ArchSection.DECODER),
    "decoder_heads": ("--decoder-heads", ArchSection.DECODER),
    "decoder_intermediate": ("--decoder-intermediate", ArchSection.DECODER),
    "decoder_cross_heads": ("--decoder-cross-heads", ArchSection.DECODER),
    "chunk_size": ("--chunk-size", ArchSection.DECODER),
    "stream_counts": ("--stream-counts", ArchSection.DECODER),
    "self_attention_mode": ("--self-attention-mode", ArchSection.DECODER),
    "time_conditioning": ("--time-conditioning", ArchSection.DECODER),
    "fast_tokenizer": ("--fast-tokenizer", ArchSection.DECODER),
    "target_time_embed": ("--target-time-embed", ArchSection.EXTENSION),
    # The molmoact2 pathway selector: EXTENSION so --init-from may pick
    # any pathway of a source checkpoint (the transition matrix), while
    # --resume stays locked to the recorded family.
    "objective": ("--objective", ArchSection.EXTENSION),
}


#: Fresh-run values for the ARCH_FLAGS sentinels (argparse defaults
#: moved here so "omitted" is distinguishable from "passed the
#: default" — the refusal rule needs explicitness). ``objective`` has
#: no fresh-run value: the flag is an --init-from pathway selector and
#: is refused on fresh runs.
ARCH_DEFAULTS: dict[str, Any] = {
    "backbone": DEFAULT_BACKBONE,
    "max_soft_tokens": 140,
    "max_crops": 1,
    "prompt_generate_bracket": False,
    "decoder_hidden": 768,
    "decoder_heads": 6,
    "decoder_intermediate": 3072,
    "decoder_cross_heads": 4,
    "chunk_size": 50,
    "stream_counts": (4, 4, 7),
    "self_attention_mode": "causal_actions",
    "time_conditioning": TimeConditioning.ADDITIVE.value,
    "fast_tokenizer": None,
    "target_time_embed": False,
    "objective": None,
}


@dataclass(frozen=True, slots=True)
class CheckpointResolution:
    """The read-side facts TrainArgs resolution needs from a --resume/
    --init-from checkpoint — parsed off the VLA metadata by
    :func:`resolve_checkpoint` (I/O stays at the CLI edge; tests
    fabricate this directly)."""

    family: VLAFamily
    backbone: str
    step: int
    # The recorded objective's tagged dict (kind + payload knobs) —
    # locked under --resume, which reconstructs the payload from it.
    objective: dict[str, Any]
    train_args: CheckpointTrainArgs
    condition_fields: tuple[str, ...]
    generate_bracket: bool

    @property
    def objective_kind(self) -> str:
        return str(self.objective.get("kind", "flow"))


def resolve_checkpoint(checkpoint: Path) -> CheckpointResolution:
    """Read a VLA checkpoint's metadata into the resolution facts —
    metadata only, no weights touched. Converted legacy checkpoints
    carry their historical train_args key spellings;
    :class:`~bijou.loading.CheckpointTrainArgs` normalizes both."""
    metadata = read_metadata(checkpoint)
    prompt = parse_prompt_config(dict(metadata.components["prompt"]["config"]))
    return CheckpointResolution(
        family=metadata.family,
        backbone=metadata.backbone_id,
        step=metadata.step,
        objective=dict(metadata.objective),
        train_args=CheckpointTrainArgs.from_dict(metadata.train_args),
        condition_fields=prompt.condition_fields,
        generate_bracket=prompt.generate_bracket,
    )


@dataclass(frozen=True, slots=True)
class TrainArgs:
    train_data: tuple[Path, ...]
    exclude: tuple[str, ...]
    fps: tuple[float, ...] | None
    camera_counts: tuple[int, ...] | None
    holdout_episodes: float
    split_seed: int
    # The model family this run trains (VLAFamily value). Fresh runs
    # declare it; under --init-from/--resume it is checkpoint-inferred
    # (metadata.family, transformed by --objective under --init-from
    # for the molmoact2 pathway matrix).
    family: str
    backbone: str
    save_dir: Path
    init_from: Path | None
    resume: Path | None
    # --resume restarts the data stream (epoch 0 shuffle, per-rank τ/ε
    # streams) under --seed, so a same-seed resume replays exactly the
    # batches and noise draws the checkpoint already trained on. The
    # fresh-seed convention is enforced at startup; this flag is the
    # explicit reproduction-only escape hatch.
    allow_same_seed_resume: bool
    # Stage-2: inherit ONLY the (frozen or re-trained) backbone + prompt
    # state_proj from a checkpoint; the decoder builds fresh under this
    # run's --family — deliberately unconstrained by the source.
    backbone_init_from: Path | None
    # Render [generate|actions] in prompts for non-AR families (implied
    # and always-on for the AR-suffix families): stage-2 trunk
    # consistency.
    prompt_generate_bracket: bool
    instruction: str | None
    cameras: tuple[str, ...] | None
    max_cameras: int | None
    max_soft_tokens: int
    max_crops: int
    stream_counts: tuple[int, ...]
    # Knowledge insulation on the molmo_flow KV seam: detach the
    # extracted per-layer K/V before the flow decoder. Part of the
    # joint objective payload; on molmoact2_flow it is legal only with
    # a frozen trunk (where it is a numerical no-op, kept for
    # provenance).
    insulate_flow: bool
    self_attention_mode: str
    time_conditioning: str
    # SnapFlow φ_s target-time embedding on the flow decoder (implied by
    # --distill snapflow; loadable over an unextended checkpoint — the
    # sanctioned additive warm start).
    target_time_embed: bool
    # Training objective variant: None = the family's standard
    # objective; "snapflow" = the self-distillation mix (gemma_flow
    # only). Recorded in the checkpoint's objective metadata; locked
    # under --resume.
    distill: str | None
    # The snapflow mix's payload knobs (--snapflow-alpha /
    # --snapflow-shortcut-weight): REQUIRED with --distill snapflow —
    # no silent defaults, a run's mix is always declared (the
    # historical runs used 0.5 / 0.1) — and refused without it; under
    # --resume they reconstruct from the recorded objective payload.
    snapflow_alpha: float | None
    snapflow_shortcut_weight: float | None
    fast_tokenizer: str | None
    aux_fields: tuple[str, ...] | None
    narration_weight: float
    aux_dropout: float
    field_dropout: float
    aux_prompt_hash: str | None
    camera_kind_dropout: float
    instruction_augment: float
    condition_fields: tuple[str, ...] | None
    condition_dropout: float
    subgoal_dropout: float
    # Anti-shortcut regularizer (arXiv:2506.23944): probability a
    # sample's proprioceptive state is masked to its dataset mean at
    # collation (normalized token ≡ 0). Probes score intact state.
    state_dropout: float
    # Sim2real appearance regularizer: probability a camera frame gets
    # the bijou.modelling.image_augment photometric recipe at collation. Probes
    # and evals always see clean frames.
    image_augment: float
    decoder_hidden: int
    decoder_heads: int
    decoder_intermediate: int
    decoder_cross_heads: int
    chunk_size: int
    batch_size: int
    # Length-grouped batches (throughput; changes batch composition —
    # paired arms must share the flag).
    bucket_by_length: bool
    # Forward/backward each loader batch in this many equal sample
    # chunks with gradient accumulation (1 = the byte-identical
    # unchunked path). Memory fallback only: sample composition,
    # effective batch and the LR schedule are invariant, and the
    # per-step gradient equals the unchunked one up to fp reduction
    # order (each slice forwards against the SAME summed counts).
    backward_chunks: int
    # Shard optimizer state across DDP ranks (ZeRO stage 1,
    # ZeroRedundancyOptimizer). Update semantics are exact — each
    # parameter's Adam state lives on exactly one rank and the updated
    # shards are broadcast after each step — only the per-rank memory
    # changes. Requires torchrun (world > 1).
    zero1: bool
    # With --backward-chunks > 1 under torchrun: skip the DDP wrapper
    # entirely — replicate its construction-time rank-0 state broadcast,
    # accumulate every chunk's gradients with plain autograd, and
    # allreduce param.grad in-place once per step. DDP's reducer bucket
    # buffers (a full fp32 gradient copy) are allocated AT CONSTRUCTION
    # even if the reducer never syncs — the measured 13.6 GiB block from
    # the molmo2 smoke-ladder snapshot. Gradient identical to the DDP
    # sync up to fp reduction order (same sum / world).
    chunk_grad_allreduce: bool
    # Activation checkpointing over the molmo2 decoder blocks:
    # recompute each block in backward instead of retaining its
    # interior activations (measured need: ~2.4 GiB/sample
    # saved activations on the live-trunk prefix). Memory only — the
    # gradient is oracle-pinned bitwise to the plain step; engages
    # wherever the trunk runs under grad, no-grad paths untouched.
    activation_checkpointing: bool
    steps: int
    decoder_lr: float
    backbone_text_lr: float | None
    backbone_vision_lr: float | None
    warmup_steps: int
    # Linear LR ramp anchored at the RESUME step (0 = off; requires
    # --resume): the extension-run warm restart.
    rewarmup_steps: int
    weight_decay: float
    grad_clip: float

    log_every: int
    eval_every: int
    save_every: int
    num_workers: int
    prefetch_factor: int
    video_decoder_cache: int
    device: str
    seed: int
    eval_samples: int | None
    eval_seed: int
    wandb_project: str | None
    wandb_run_name: str | None
    # Opt out of async checkpoint saves (--sync-save): block stepping for
    # the full serialize+write — and under --zero1 for the rank-by-rank
    # consolidate broadcast (~15.5 min/save measured on the molmo2 AR
    # 4xDDP run, ~14% of wall time). The default captures a device->CPU
    # snapshot at the boundary (seconds) and gathers/merges/writes on a
    # background thread over a dedicated gloo group; the written bytes
    # are oracle-pinned identical (tests/test_async_save.py). Defaulted
    # (unlike every field above) so checkpoints predating the flag
    # replay their train_args cleanly.
    sync_save: bool = False
    # AdamW moments in host RAM (CPUOffloadAdamW): the update runs
    # torch's CPU kernels on pinned fp32 mirrors — semantics exact
    # (elementwise AdamW; oracle-pinned), only the states' residence
    # changes. The lever that fits a live-trunk molmoact2 run on one
    # 80 GiB card (measured: 33.7 GiB of moments for the joint set).
    # Single-process only: refused under torchrun and with --zero1.
    # Defaulted (like sync_save) so checkpoints predating the flag
    # replay their train_args cleanly.
    offload_optim: bool = False
    # λ of the joint objective — a run hyperparameter like the LRs
    # (re-passable on --resume; recorded for provenance and in the
    # objective payload). 1.0 = the KI no-tuning default.
    joint_ce_weight: float = 1.0
    # The flow decoder's weight source under --init-from (molmoact2
    # flow/joint): 'inherit' = the source checkpoint's flow decoder;
    # 'fresh' = released-shape adaLN-Zero init (the stage-2 recipe —
    # REQUIRED from ar-only sources, which carry no flow decoder); any
    # other value = a VLA checkpoint dir whose flow_decoder.safetensors
    # loads under a config-equality guard (the two-source init).
    flow_decoder_init: str = "inherit"
    # "adamw" (default) or "adamc" — AdamC (arXiv 2506.02285) is AdamW
    # with a time-varying decay coefficient on the hidden ("normalized")
    # layers: λ̂_t = λ·γ_t/γ_max, written into the corrected param
    # groups' weight_decay before every step so the stock (fused) AdamW
    # kernel applies it bit-exactly. Output-head parameters
    # (``VLA.output_head_parameters``) keep standard AdamW decay per the
    # paper's Algorithm 1; 1-D parameters stay undecayed as everywhere
    # else. Defaulted (like sync_save) so checkpoints predating the flag
    # replay their train_args cleanly.
    optimizer: str = "adamw"
    # Raw --dataset-repeat PATTERN=COUNT specs (bijou.data.parse_repeat_specs
    # parses at use): replicate matching datasets in the concatenated train
    # set — the oversample lever for vanishing-share datasets. Training-only
    # (the in-train holdout eval never repeats); changes the concatenated
    # frame indexing, so a resume must pass the identical spec. Defaulted
    # (like sync_save) so checkpoints predating the flag replay their
    # train_args cleanly.
    dataset_repeat: tuple[str, ...] = ()

    @property
    def backbone_trained(self) -> bool:
        return self.backbone_text_lr is not None or self.backbone_vision_lr is not None

    def __post_init__(self) -> None:
        """Value invariants of the RESOLVED config — the single encoding
        (from_namespace translates these to parser.error for the CLI;
        direct construction gets the same text as ValueError). Rules
        needing flag EXPLICITNESS (checkpoint-inferred refusals,
        "drop the flag") live in from_namespace: this class only sees
        resolved values."""
        # Fail fast, before any data/model build; select_datasets
        # re-parses at use (TrainArgs keeps the raw strings for
        # checkpoint replay). parse_repeat_specs raises ValueError with
        # the offending spec — exactly this method's contract.
        parse_repeat_specs(self.dataset_repeat)
        families = {f.value for f in VLAFamily}
        if self.family not in families:
            raise ValueError(
                f"unknown family {self.family!r} — one of {sorted(families)}",
            )
        if self.rewarmup_steps > 0 and self.resume is None:
            raise ValueError(
                "--rewarmup-steps anchors at the resume step — it requires "
                "--resume (fresh runs use --warmup-steps)",
            )
        if self.offload_optim and self.zero1:
            raise ValueError(
                "--offload-optim and --zero1 are both optimizer-memory "
                "levers and compose to nothing sensible — offload is the "
                "single-GPU spelling, zero1 the multi-rank one; pick one",
            )
        if self.allow_same_seed_resume and self.resume is None:
            raise ValueError(
                "--allow-same-seed-resume only applies to --resume "
                "(fresh runs and --init-from have no seed to collide with)",
            )
        if not 0.0 <= self.holdout_episodes < 1.0:
            raise ValueError("--holdout-episodes must be in [0, 1)")
        if self.holdout_episodes > 0 and self.eval_samples is None:
            raise ValueError(
                "--eval-samples is required when --holdout-episodes > 0 "
                "(it sizes the held-out eval_chunk_mae probe)",
            )
        if self.eval_samples is not None and self.eval_samples < 1:
            raise ValueError("--eval-samples must be >= 1")
        if self.decoder_lr <= 0:
            raise ValueError("--decoder-lr must be > 0")
        for name, value in (
            ("--backbone-text-lr", self.backbone_text_lr),
            ("--backbone-vision-lr", self.backbone_vision_lr),
        ):
            if value is not None and value <= 0:
                raise ValueError(
                    f"{name} {value} is not a usable learning rate — omit "
                    "the flag entirely to keep that component frozen",
                )
        if self.family in AR_SUFFIX_FAMILIES and self.fast_tokenizer is None:
            raise ValueError(
                f"--family {self.family} requires --fast-tokenizer (the "
                "suffix decoder's action codec)",
            )
        if self.family not in AR_SUFFIX_FAMILIES and self.fast_tokenizer is not None:
            raise ValueError(
                "--fast-tokenizer is only consumed by the AR-suffix "
                "families (gemma_ar, molmo2_ar) — the molmoact2 discrete "
                "pathway records its released codec in the checkpoint",
            )
        if self.joint_ce_weight <= 0:
            raise ValueError(
                f"--joint-ce-weight {self.joint_ce_weight} must be > 0 "
                "(λ = 0 is the flow pathway: --objective flow)",
            )
        if self.joint_ce_weight != 1.0 and self.family != "molmoact2_joint":
            raise ValueError(
                "--joint-ce-weight scales the joint objective's CE term — "
                "it requires the molmoact2_joint family (--objective joint)",
            )
        if self.flow_decoder_init != "inherit" and self.family not in (
            "molmoact2_flow",
            "molmoact2_joint",
        ):
            raise ValueError(
                "--flow-decoder-init selects the molmoact2 flow decoder's "
                f"weight source; family {self.family} builds no such "
                "decoder — drop the flag",
            )
        if (
            self.family in ("molmoact2_ar", "molmoact2_joint")
            and self.backbone_text_lr is None
        ):
            raise ValueError(
                f"family {self.family} trains the trunk's own action rows "
                "(the discrete decoder has ZERO parameters) — "
                "--backbone-text-lr is required",
            )
        if self.family in MOLMOACT2_FAMILIES:
            # Inherit-only: the molmoact2 compositions rebuild from a
            # converted checkpoint's sections; from-scratch molmoact2 on
            # bijou prompts is a deliberate non-goal.
            if self.init_from is None and self.resume is None:
                raise ValueError(
                    f"family {self.family} trains from a checkpoint only "
                    "(--init-from a converted MolmoAct2 checkpoint, or "
                    "--resume)",
                )
            if self.condition_fields is not None:
                raise ValueError(
                    "--condition-fields cannot ride the molmoact2 prompt "
                    "format — it has no bytes for the bracket surfaces",
                )
            # Deliberately NOT refused: --camera-kind-dropout (inert —
            # this format never renders kinds) and --instruction-augment
            # (task text renders; judge rewrites are format-compatible).
            if self.family == "molmoact2_ar" and self.insulate_flow:
                raise ValueError(
                    "--insulate-flow is the flow KV seam; molmoact2_ar "
                    "builds no flow decoder to insulate",
                )
            if (
                self.family == "molmoact2_flow"
                and self.backbone_trained
                and self.insulate_flow
            ):
                raise ValueError(
                    "--insulate-flow with an unfrozen trunk trains the "
                    "trunk on NOTHING under the flow objective (insulated "
                    "flow is its only term) — --objective joint gives the "
                    "trunk the CE rider, or freeze the trunk / open the "
                    "seam",
                )
        elif self.insulate_flow:
            raise ValueError(
                "--insulate-flow is the molmo_flow KV seam; other families "
                "have no insulable conditioning seam (the gemma flow "
                "decoder trains against a frozen trunk)",
            )
        if self.family != "gemma_flow" and self.time_conditioning != "additive":
            raise ValueError(
                "--time-conditioning is gemma_flow-only (other families "
                "have no τ-conditioned decoder to configure)",
            )
        if self.family != "gemma_flow" and self.distill is not None:
            raise ValueError(
                "--distill is gemma_flow-only (it distills the velocity field)",
            )
        if self.distill == "snapflow":
            if self.snapflow_alpha is None or self.snapflow_shortcut_weight is None:
                raise ValueError(
                    "--distill snapflow declares its mix explicitly: pass "
                    "--snapflow-alpha AND --snapflow-shortcut-weight (the "
                    "historical runs used 0.5 and 0.1) — there is no "
                    "silent default",
                )
            # Value invariants live ONCE, on the payload — construct
            # and discard so a bad mix dies at the parse boundary.
            SnapflowObjective(
                alpha=self.snapflow_alpha,
                shortcut_weight=self.snapflow_shortcut_weight,
            )
        elif (
            self.snapflow_alpha is not None or self.snapflow_shortcut_weight is not None
        ):
            raise ValueError(
                "--snapflow-alpha/--snapflow-shortcut-weight parameterize "
                "the snapflow mix — they require --distill snapflow",
            )
        if self.family != "gemma_flow" and self.target_time_embed:
            raise ValueError(
                "--target-time-embed is gemma_flow-only (φ_s conditions τ)",
            )
        if self.distill == "snapflow" and not self.target_time_embed:
            raise ValueError(
                "--distill snapflow needs the φ_s extension "
                "(target_time_embed) — from_namespace implies it on fresh/"
                "--init-from runs; a resumed checkpoint without φ_s cannot "
                "grow it (extend via --init-from --target-time-embed)",
            )
        if self.aux_fields is not None and self.family not in AR_SUFFIX_FAMILIES:
            raise ValueError(
                "--aux-fields rides the AR-suffix families only (gemma_ar, molmo2_ar)",
            )
        if self.aux_fields is not None and len(self.aux_fields) == 0:
            raise ValueError(
                "--aux-fields given with no fields — omit the flag instead",
            )
        if self.aux_fields is not None:
            # Template order is an invariant, not a preference (AuxSpec
            # re-guards, but that fires only after dataset selection).
            ordered = [f.value for f in AuxField if f.value in self.aux_fields]
            if list(self.aux_fields) != ordered:
                raise ValueError(
                    f"--aux-fields must keep template order {ordered} "
                    f"(got {list(self.aux_fields)})",
                )
            if self.cameras is not None or self.max_cameras is not None:
                # The 'visible' aux indices are positions in the full
                # sorted camera set; camera selection would silently
                # shift them (the Collator re-guards, but that fires
                # only after dataset selection).
                raise ValueError(
                    "--aux-fields cannot combine with --cameras/--max-cameras",
                )
        if self.narration_weight <= 0:
            raise ValueError(
                "--narration-weight must be > 0 (omit --aux-fields to disable)",
            )
        for name, value in (
            ("--aux-dropout", self.aux_dropout),
            ("--field-dropout", self.field_dropout),
            ("--camera-kind-dropout", self.camera_kind_dropout),
            ("--state-dropout", self.state_dropout),
            ("--condition-dropout", self.condition_dropout),
            ("--subgoal-dropout", self.subgoal_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} {value} outside [0, 1)")
        if not 0.0 <= self.instruction_augment <= 1.0:
            raise ValueError(
                f"--instruction-augment {self.instruction_augment} outside [0, 1]",
            )
        if not 0.0 <= self.image_augment <= 1.0:
            raise ValueError(
                f"--image-augment {self.image_augment} outside [0, 1]",
            )
        if self.condition_fields is not None and len(self.condition_fields) == 0:
            raise ValueError("--condition-fields given with no fields — omit the flag")
        if self.condition_fields is not None:
            ordered = [
                f.value for f in ConditionField if f.value in self.condition_fields
            ]
            if list(self.condition_fields) != ordered:
                raise ValueError(
                    f"--condition-fields must keep template order {ordered} "
                    f"(got {list(self.condition_fields)})",
                )

    @classmethod
    def from_namespace(
        cls,
        raw: argparse.Namespace,
        parser: argparse.ArgumentParser,
        *,
        checkpoint: CheckpointResolution | None,
    ) -> TrainArgs:
        """Parse the RAW namespace (arch flags carry None sentinels, so
        explicitness is visible) into a validated TrainArgs.

        Owns everything that needs the namespace or the checkpoint:
        the family requirement/refusal, the checkpoint-inferred-flag
        refusals (ARCH_FLAGS), resolution (sentinel -> ARCH_DEFAULTS on
        fresh runs, -> the checkpoint's recorded architecture under
        --resume/--init-from, with --objective transforming the
        molmoact2 family under --init-from), and the "flag X requires
        flag Y" explicitness checks. Value invariants of the resolved
        config live once, in __post_init__ — raised ValueErrors are
        translated to parser.error here, so the CLI keeps its
        usage-line UX while direct construction gets the same message.
        ``checkpoint`` is the parsed metadata of --resume/--init-from
        (read by the caller: I/O stays at the CLI edge)."""
        if raw.init_from is not None and raw.resume is not None:
            parser.error("--init-from and --resume are mutually exclusive")
        if raw.backbone_init_from is not None and (
            raw.init_from is not None or raw.resume is not None
        ):
            parser.error(
                "--backbone-init-from is mutually exclusive with --init-from/"
                "--resume (those load the decoder too)",
            )
        resume = raw.resume is not None
        assert (checkpoint is None) == (raw.init_from is None and not resume)

        # -- family resolution ---------------------------------------------
        if checkpoint is None:
            if raw.family is None:
                parser.error(
                    "--family is required for fresh runs (one of "
                    f"{', '.join(f.value for f in VLAFamily)}); under "
                    "--init-from/--resume it is checkpoint-inferred — "
                    "drop the flag there",
                )
            if raw.objective is not None:
                parser.error(
                    "--objective selects among a molmoact2 checkpoint's "
                    "trained pathways under --init-from — fresh runs "
                    "declare --family instead",
                )
            family = str(raw.family)
        else:
            if raw.family is not None:
                parser.error(
                    "--family is checkpoint-inferred under --init-from/"
                    f"--resume (checkpoint records "
                    f"{checkpoint.family.value!r}) — drop the flag"
                    + (
                        "; --objective selects among a molmoact2 checkpoint's pathways"
                        if not resume
                        else ""
                    ),
                )
            family = checkpoint.family.value

        # -- checkpoint-inferred flag refusals ----------------------------
        if checkpoint is not None:
            recorded = dataclasses.asdict(checkpoint.train_args)
            recorded["backbone"] = checkpoint.backbone
            recorded["objective"] = checkpoint.objective_kind
            recorded["prompt_generate_bracket"] = (
                checkpoint.generate_bracket
                or checkpoint.family.value in AR_SUFFIX_FAMILIES
            )
            for field, (flag, section) in ARCH_FLAGS.items():
                if getattr(raw, field) is None:
                    continue
                shown = recorded.get(field)
                shown = shown.value if isinstance(shown, StrEnum) else shown
                if resume:
                    parser.error(
                        f"{flag} is checkpoint-inferred under --resume "
                        f"(checkpoint records {field}={shown!r}) — drop the "
                        "flag; --resume rebuilds the recorded architecture",
                    )
                if section is not ArchSection.EXTENSION:
                    parser.error(
                        f"{flag} is inherited from the --init-from checkpoint "
                        f"(it records {field}={shown!r}) — drop the flag; a "
                        "fresh decoder on an inherited trunk is "
                        "--backbone-init-from + --family (the stage-2 path)",
                    )
                # EXTENSION flags (φ_s, --objective) stay legal on
                # --init-from: zero-init additions and the pathway matrix.
            if raw.distill is not None and resume:
                parser.error(
                    "--distill is part of the recorded objective "
                    f"(checkpoint records {checkpoint.objective_kind!r}) — "
                    "locked under --resume; declare a new objective via "
                    "--init-from",
                )
            if resume and (
                raw.snapflow_alpha is not None
                or raw.snapflow_shortcut_weight is not None
            ):
                parser.error(
                    "--snapflow-alpha/--snapflow-shortcut-weight are part "
                    "of the recorded objective "
                    f"(checkpoint records {checkpoint.objective_kind!r}) — "
                    "locked under --resume; the recorded payload "
                    "reconstructs them",
                )

        # -- the molmoact2 pathway matrix (--objective under --init-from) --
        if raw.objective is not None and checkpoint is not None:
            if checkpoint.family.value not in MOLMOACT2_FAMILIES:
                parser.error(
                    "--objective selects among the molmoact2 family's "
                    f"trained pathways; {checkpoint.family.value} has a "
                    "single objective",
                )
            family = f"molmoact2_{raw.objective}"

        # -- resolution ----------------------------------------------------
        arch: dict[str, Any] = {}
        for field in ARCH_FLAGS:
            if field == "objective":
                continue  # folded into the family above
            explicit = getattr(raw, field)
            if explicit is not None:
                arch[field] = explicit
                continue
            if checkpoint is None:
                arch[field] = ARCH_DEFAULTS[field]
                continue
            match field:
                case "backbone":
                    arch[field] = checkpoint.backbone
                case "prompt_generate_bracket":
                    # AR-suffix sources rendered the bracket implicitly
                    # (the flag itself is refused there) — inheriting the
                    # recorded False would silently drop it from stage-2
                    # prompts, the exact drift this rule exists to stop.
                    arch[field] = (
                        checkpoint.generate_bracket
                        or checkpoint.family.value in AR_SUFFIX_FAMILIES
                    )
                case "self_attention_mode" | "time_conditioning":
                    arch[field] = getattr(checkpoint.train_args, field).value
                case _:
                    arch[field] = getattr(checkpoint.train_args, field)
        arch["stream_counts"] = tuple(arch["stream_counts"])

        # -- the recorded objective under --resume --------------------------
        distill = raw.distill
        snapflow_alpha = raw.snapflow_alpha
        snapflow_shortcut_weight = raw.snapflow_shortcut_weight
        if resume:
            assert checkpoint is not None
            if checkpoint.objective_kind == "snapflow":
                # The recorded payload reconstructs the mix (its flags
                # were refused above) — train-written snapflow records
                # always carry both knobs (objective_to_json).
                distill = "snapflow"
                snapflow_alpha = float(checkpoint.objective["alpha"])
                snapflow_shortcut_weight = float(
                    checkpoint.objective["shortcut_weight"],
                )
            else:
                distill = None

        if distill == "snapflow" and not arch["target_time_embed"]:
            # φ_s is implied — but only where the structure is mutable
            # (fresh decoders); a resumed checkpoint cannot grow it.
            arch["target_time_embed"] = True

        # -- explicitness checks needing the resolved family ---------------
        if family in AR_SUFFIX_FAMILIES or family in MOLMOACT2_FAMILIES:
            # The backbone IS the architecture on the suffix families;
            # the molmoact2 families rebuild theirs from the source
            # checkpoint's sections either way: decoder shape flags and
            # the cross-attention schedule describe models these runs
            # don't build. (Under --init-from/--resume the ARCH loop
            # already refused them; this covers fresh AR-suffix runs.)
            for flag, attribute in (
                ("--decoder-hidden", "decoder_hidden"),
                ("--decoder-heads", "decoder_heads"),
                ("--decoder-intermediate", "decoder_intermediate"),
                ("--decoder-cross-heads", "decoder_cross_heads"),
                ("--stream-counts", "stream_counts"),
            ):
                if getattr(raw, attribute) is not None:
                    parser.error(
                        f"{flag} sizes the gemma flow decoder; family "
                        f"{family} has no such decoder — drop the flag",
                    )
        if family in AR_SUFFIX_FAMILIES and raw.prompt_generate_bracket is not None:
            parser.error(
                "--prompt-generate-bracket is implied (always on) for the "
                "AR-suffix families — drop the flag so runs have one "
                "spelling",
            )
        if family == "molmoact2_ar" and raw.decoder_lr is not None:
            # LR-vs-offer reconciliation, the parse-time half: the
            # family's 'decoder' param group is structurally EMPTY
            # (parameterless decoder + prompt side), so an explicit LR
            # for it contradicts the offer. main() re-checks the built
            # model's actual offer for the backbone groups.
            parser.error(
                "--decoder-lr given, but the 'decoder' param group "
                "receives no gradients under molmoact2_ar's objective "
                "(the discrete decoder and prompt side own zero "
                "parameters — the trunk trains via --backbone-text-lr)",
            )

        # -- "flag X requires flag Y" + conditional defaults ---------------
        if raw.aux_dropout is not None and raw.aux_fields is None:
            parser.error("--aux-dropout requires --aux-fields (it drops aux labels)")
        if raw.aux_prompt_hash is not None and raw.aux_fields is None:
            parser.error(
                "--aux-prompt-hash requires --aux-fields (it gates aux labels)",
            )
        if raw.field_dropout is not None and raw.aux_fields is None:
            parser.error("--field-dropout requires --aux-fields (it drops requests)")
        if raw.condition_dropout is not None and raw.condition_fields is None:
            parser.error("--condition-dropout requires --condition-fields")
        subgoal_conditioned = raw.condition_fields is not None and (
            "subgoal" in raw.condition_fields
        )
        if raw.subgoal_dropout is not None and not subgoal_conditioned:
            parser.error("--subgoal-dropout requires subgoal in --condition-fields")
        aux_dropout = (
            raw.aux_dropout
            if raw.aux_dropout is not None
            else (0.1 if raw.aux_fields is not None else 0.0)
        )
        field_dropout = (
            raw.field_dropout
            if raw.field_dropout is not None
            else (0.1 if raw.aux_fields is not None else 0.0)
        )
        condition_dropout = (
            raw.condition_dropout
            if raw.condition_dropout is not None
            else (0.1 if raw.condition_fields is not None else 0.0)
        )
        subgoal_dropout = (
            raw.subgoal_dropout
            if raw.subgoal_dropout is not None
            else (0.5 if subgoal_conditioned else 0.0)
        )

        if raw.joint_ce_weight is not None and family != "molmoact2_joint":
            parser.error(
                "--joint-ce-weight scales the joint objective's CE term — "
                f"this run resolves family {family}",
            )
        if raw.flow_decoder_init is not None:
            if resume:
                parser.error(
                    "--flow-decoder-init is an --init-from concern — a "
                    "resumed run's flow decoder weights come from the "
                    "resume checkpoint",
                )
            if raw.init_from is None:
                parser.error(
                    "--flow-decoder-init selects the flow decoder's weight "
                    "source when warm-starting — it requires --init-from",
                )

        if raw.backbone_vision_lr is not None and raw.backbone_text_lr is None:
            print(
                "NOTE: vision tower unfrozen with the text stack FROZEN - "
                "gradients still traverse the frozen text stack to reach the "
                "tower (activation cost without text adaptation). Legitimate "
                "but unusual; the standard move is --backbone-text-lr alone.",
                file=sys.stderr,
                flush=True,
            )

        try:
            return cls(
                train_data=tuple(raw.train_data),
                exclude=tuple(raw.exclude),
                dataset_repeat=tuple(raw.dataset_repeat),
                fps=tuple(raw.fps) if raw.fps else None,
                camera_counts=(tuple(raw.camera_counts) if raw.camera_counts else None),
                holdout_episodes=raw.holdout_episodes,
                split_seed=raw.split_seed,
                family=family,
                backbone=arch["backbone"],
                save_dir=raw.save_dir,
                init_from=raw.init_from,
                resume=raw.resume,
                allow_same_seed_resume=raw.allow_same_seed_resume,
                backbone_init_from=raw.backbone_init_from,
                prompt_generate_bracket=arch["prompt_generate_bracket"],
                instruction=raw.instruction,
                cameras=tuple(raw.cameras) if raw.cameras else None,
                max_cameras=raw.max_cameras,
                max_soft_tokens=arch["max_soft_tokens"],
                max_crops=arch["max_crops"],
                stream_counts=arch["stream_counts"],
                insulate_flow=raw.insulate_flow,
                joint_ce_weight=(
                    raw.joint_ce_weight if raw.joint_ce_weight is not None else 1.0
                ),
                flow_decoder_init=(
                    raw.flow_decoder_init
                    if raw.flow_decoder_init is not None
                    else "inherit"
                ),
                self_attention_mode=arch["self_attention_mode"],
                time_conditioning=arch["time_conditioning"],
                target_time_embed=arch["target_time_embed"],
                distill=distill,
                snapflow_alpha=snapflow_alpha,
                snapflow_shortcut_weight=snapflow_shortcut_weight,
                fast_tokenizer=arch["fast_tokenizer"],
                aux_fields=(
                    tuple(raw.aux_fields) if raw.aux_fields is not None else None
                ),
                narration_weight=raw.narration_weight,
                aux_dropout=aux_dropout,
                field_dropout=field_dropout,
                aux_prompt_hash=raw.aux_prompt_hash,
                camera_kind_dropout=raw.camera_kind_dropout,
                instruction_augment=raw.instruction_augment,
                condition_fields=(
                    tuple(raw.condition_fields)
                    if raw.condition_fields is not None
                    else None
                ),
                condition_dropout=condition_dropout,
                subgoal_dropout=subgoal_dropout,
                state_dropout=raw.state_dropout,
                image_augment=raw.image_augment,
                decoder_hidden=arch["decoder_hidden"],
                decoder_heads=arch["decoder_heads"],
                decoder_intermediate=arch["decoder_intermediate"],
                decoder_cross_heads=arch["decoder_cross_heads"],
                chunk_size=arch["chunk_size"],
                batch_size=raw.batch_size,
                bucket_by_length=raw.bucket_by_length,
                backward_chunks=raw.backward_chunks,
                zero1=raw.zero1,
                sync_save=raw.sync_save,
                chunk_grad_allreduce=raw.chunk_grad_allreduce,
                activation_checkpointing=raw.activation_checkpointing,
                offload_optim=raw.offload_optim,
                steps=raw.steps,
                decoder_lr=(raw.decoder_lr if raw.decoder_lr is not None else 1e-4),
                backbone_text_lr=raw.backbone_text_lr,
                backbone_vision_lr=raw.backbone_vision_lr,
                warmup_steps=raw.warmup_steps,
                rewarmup_steps=raw.rewarmup_steps,
                weight_decay=raw.weight_decay,
                optimizer=raw.optimizer,
                grad_clip=raw.grad_clip,
                log_every=raw.log_every,
                eval_every=raw.eval_every,
                save_every=raw.save_every,
                num_workers=raw.num_workers,
                prefetch_factor=raw.prefetch_factor,
                video_decoder_cache=raw.video_decoder_cache,
                device=raw.device,
                seed=raw.seed,
                eval_samples=raw.eval_samples,
                eval_seed=raw.eval_seed,
                wandb_project=raw.wandb_project,
                wandb_run_name=raw.wandb_run_name,
            )
        except ValueError as error:
            parser.error(str(error))


def train_args_record(args: TrainArgs) -> dict[str, Any]:
    """The verbatim provenance dict checkpoint metadata records: asdict
    with JSON-ready leaves (Paths stringify — including inside tuples;
    tuples become lists, the JSON round-trip form)."""

    def jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [jsonable(item) for item in value]
        return value

    return {key: jsonable(value) for key, value in dataclasses.asdict(args).items()}


def reconcile_lr_offer(
    offer: dict[str, list[torch.nn.Parameter]],
    *,
    family: str,
    backbone_text_lr: float | None,
    backbone_vision_lr: float | None,
) -> list[str]:
    """LR flags vs the model's structural offer, both directions (D4):
    an LR flag for an EMPTY offered group is a contradiction and dies
    loudly; an offered non-empty group without its LR flag freezes —
    returned as the notes the caller prints (omit-to-freeze is the
    opt-in convention, but the freeze is never silent)."""
    notes: list[str] = []
    for flag, lr, group in (
        ("--backbone-text-lr", backbone_text_lr, "backbone_text"),
        ("--backbone-vision-lr", backbone_vision_lr, "backbone_vision"),
    ):
        params = offer[group]
        if lr is not None and len(params) == 0:
            raise SystemExit(
                f"{flag} given, but {group!r} receives no gradients under "
                f"{family}'s objective — drop the flag",
            )
        if lr is None and len(params) > 0:
            count = sum(p.numel() for p in params)
            notes.append(
                f"param group {group!r} ({count / 1e6:.1f}M params) offered "
                f"by {family} but {flag} not given: FROZEN",
            )
    return notes


def _build_parser() -> argparse.ArgumentParser:
    """The train CLI's flag surface. Checkpoint-inferred architecture
    flags (ARCH_FLAGS, plus --family itself) carry ``None`` sentinel
    defaults so ``TrainArgs.from_namespace`` can tell "omitted" from
    "passed the default" — fresh-run values live in ARCH_DEFAULTS."""
    parser = argparse.ArgumentParser(
        prog="python -m bijou.train",
        description="Train a Bijou VLA family on LeRobot v3 datasets "
        "(dataset directories and/or collection roots). Runs on a single "
        "GPU by default and data-parallel under torchrun; checkpoints "
        "carry everything bijou.eval and bijou.rollout need.",
    )
    parser.add_argument(
        "--family",
        choices=[f.value for f in VLAFamily],
        default=None,
        help="the model family this run trains (required for fresh runs; "
        "checkpoint-inferred under --init-from/--resume — drop the flag "
        "there). The molmoact2_* families are inherit-only: they train "
        "from a converted checkpoint via --init-from, where --objective "
        "selects the pathway",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        nargs="+",
        required=True,
        help="dataset directories and/or collection roots, mixed freely",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="fnmatch patterns against <user>/<dataset> repo ids to skip",
    )
    parser.add_argument(
        "--dataset-repeat",
        nargs="*",
        default=[],
        metavar="PATTERN=COUNT",
        help="oversample matching datasets: fnmatch PATTERN against "
        "<user>/<dataset> repo ids, COUNT (>= 1) replicas in the "
        "concatenated train set (first matching spec wins; a spec matching "
        "no selected dataset is fatal). Training-only — the in-train "
        "holdout eval never repeats. Changes the concatenated frame "
        "indexing: resumes must pass the identical spec",
    )
    parser.add_argument(
        "--fps",
        type=float,
        nargs="+",
        default=None,
        help="keep only datasets recorded at one of these frame rates "
        "(e.g. --fps 30); default keeps every fps, the historical "
        "behavior. Filtering changes the concatenated frame indexing, so "
        "eval numbers are only comparable between runs with the same "
        "filter",
    )
    parser.add_argument(
        "--camera-counts",
        type=int,
        nargs="+",
        default=None,
        help="keep only datasets with one of these camera COUNTS (e.g. "
        "--camera-counts 1 2): prompt length is ~160 + 140/camera, so "
        "mixed counts in a batch pad short prompts to the longest "
        "(wasted prefix compute + rank stragglers). Default keeps all. "
        "Same eval-comparability caveat as --fps",
    )
    parser.add_argument(
        "--holdout-episodes",
        type=float,
        default=0.0,
        help="fraction of each dataset's episodes to exclude from training; "
        "score them with bijou.eval --episodes holdout (same fraction and "
        "--split-seed)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="episode-holdout seed, independent of --seed so restarts "
        "never shift the split",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        help=f"backbone HF model id or local checkpoint path (fresh-run "
        f"default {DEFAULT_BACKBONE}; checkpoint-inferred under "
        "--resume/--init-from — drop the flag there)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/train/bijou_dev"),
        help="output directory: step_NNNNNN/ checkpoints, train_log.jsonl, wandb files",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="override the per-frame task string from the dataset",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="only use these camera keys when present (full keys or "
        "suffixes; default: all cameras of each sample, sorted)",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=None,
        help="cap cameras per sample (applied after --cameras filtering)",
    )
    parser.add_argument(
        "--max-soft-tokens",
        type=int,
        default=None,
        help="vision soft-token budget per camera in the prompt "
        "(Gemma trunks; molmo2 trunks use --max-crops; fresh-run "
        "default 140, checkpoint-inferred under --resume/--init-from)",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=None,
        help="molmo2 trunks: crops per camera image (fresh-run default 1 "
        "= the standard operating point, 410 image tokens/camera — "
        "the smallest layout inside the shipped distribution; ignored "
        "for Gemma trunks; checkpoint-inferred under --resume/--init-from)",
    )
    parser.add_argument(
        "--stream-counts",
        type=int,
        nargs="*",
        default=None,
        help="gemma_flow: decoder cross-attention layers per backbone KV "
        "stream, shallow to deep (0 skips a stream); fresh-run default "
        "4 4 7, checkpoint-inferred under --resume/--init-from",
    )
    parser.add_argument(
        "--insulate-flow",
        action="store_true",
        help="knowledge insulation on the molmo_flow KV seam (their "
        "post-train recipe): the extracted per-layer K/V detach before "
        "the flow decoder, so flow gradients into every trunk parameter "
        "are exactly zero. molmoact2 families only (flow with a frozen "
        "trunk, or joint — the RL-then-refine recipe); irrelevant (and "
        "refused) elsewhere",
    )
    parser.add_argument(
        "--objective",
        choices=["flow", "ar", "joint"],
        default=None,
        help="the molmoact2 pathway selector under --init-from (the "
        "transition matrix): flow = the molmo_flow decoder "
        "(molmoact2_flow), ar = the trunk's discrete head (molmoact2_ar "
        "— zero decoder parameters, requires --backbone-text-lr), "
        "joint = both (molmoact2_joint, L_flow + λ·L_CE). Omitted, the "
        "source checkpoint's family carries over; refused under "
        "--resume (the recorded family is locked) and on fresh runs "
        "(declare --family)",
    )
    parser.add_argument(
        "--joint-ce-weight",
        type=float,
        default=None,
        help="λ of the joint objective (L_flow + λ·L_CE); default 1.0 — "
        "the KI no-tuning value. A run hyperparameter like the LRs "
        "(re-passable on --resume); requires the molmoact2_joint family; "
        "must be > 0 (λ = 0 is spelled --objective flow)",
    )
    parser.add_argument(
        "--flow-decoder-init",
        default=None,
        help="the flow decoder's weight source under --init-from "
        "(molmoact2 flow/joint): 'inherit' (default) = the source "
        "checkpoint's flow decoder; 'fresh' = released-shape adaLN-Zero "
        "init (REQUIRED from ar-only sources — the stage-2 recipe); any "
        "other value = a VLA checkpoint dir whose "
        "flow_decoder.safetensors loads under a config-equality guard "
        "(the two-source init — note the borrowed decoder's outputs "
        "live in ITS training table's normalized space)",
    )
    parser.add_argument(
        "--self-attention-mode",
        choices=["causal_actions", "bidirectional"],
        default=None,
        help="gemma_flow decoder self-attention over the action chunk "
        "(fresh-run default causal_actions, checkpoint-inferred under "
        "--resume/--init-from)",
    )
    parser.add_argument(
        "--time-conditioning",
        choices=[m.value for m in TimeConditioning],
        default=None,
        help="how flow time τ conditions the gemma_flow decoder: 'additive' "
        "(π0-style input add, the default) or 'adarms' (DiT-style per-layer "
        "scale/gate, identity at init). adarms changes the architecture — a "
        "fresh decoder only (cannot --init-from an additive checkpoint)",
    )
    parser.add_argument(
        "--target-time-embed",
        action="store_true",
        default=None,
        help="extend the gemma_flow decoder with the SnapFlow φ_s "
        "target-time embedding (zero-initialized output: inert until "
        "trained; may --init-from an unextended checkpoint — step 0 is "
        "then exactly that checkpoint). Implied by --distill snapflow",
    )
    parser.add_argument(
        "--distill",
        choices=["snapflow"],
        default=None,
        help="training objective variant: 'snapflow' = self-distillation "
        "toward 1-NFE decoding (L = α·L_FM + (1−α)·λ·L_shortcut with "
        "stop-gradient two-step-Euler shortcut targets; α/λ are the "
        "REQUIRED --snapflow-alpha/--snapflow-shortcut-weight flags). "
        "gemma_flow only; enables --target-time-embed; recorded in the "
        "checkpoint's objective and locked under --resume",
    )
    parser.add_argument(
        "--snapflow-alpha",
        type=float,
        default=None,
        help="the snapflow mix's FM share α ∈ (0, 1) — required with "
        "--distill snapflow, refused without it; no silent default (the "
        "historical runs used 0.5). Under --resume the recorded "
        "objective payload reconstructs it — drop the flag",
    )
    parser.add_argument(
        "--snapflow-shortcut-weight",
        type=float,
        default=None,
        help="the snapflow mix's shortcut multiplier λ > 0 — required "
        "with --distill snapflow, refused without it; no silent default "
        "(the historical runs used 0.1). Under --resume the recorded "
        "objective payload reconstructs it — drop the flag",
    )
    parser.add_argument(
        "--aux-fields",
        nargs="*",
        choices=[f.value for f in AuxField],
        default=None,
        help="train aux text generation from judge annotations (AR-suffix "
        "families only): fields rendered before BOA in template order; "
        "datasets whose annotation stamp is absent/stale train as "
        "unjudged, loudly. Omit to train actions only (the historical "
        "objective)",
    )
    parser.add_argument(
        "--narration-weight",
        type=float,
        default=0.5,
        help="weight of the aux-text CE component (total = action + "
        "w*aux); labels are weak supervision (~80%% judge agreement), "
        "hence the modest default",
    )
    parser.add_argument(
        "--aux-dropout",
        type=float,
        default=None,
        help="probability a LABELED sample's request set collapses to "
        "[generate|actions] (its value lines dropped): keeps the "
        "deployment fast path trained under dense annotation. Default "
        "0.1 when --aux-fields is on; requires --aux-fields",
    )
    parser.add_argument(
        "--field-dropout",
        type=float,
        default=None,
        help="per labeled field, independent probability of dropping it "
        "from the sample's request set (request and target move "
        "together), so all SUBSETS of the labeled set appear in "
        "training and inference-time partial requests stay "
        "in-distribution. Default 0.1 when --aux-fields is on; requires "
        "--aux-fields",
    )
    parser.add_argument(
        "--aux-prompt-hash",
        default=None,
        help="optional per-run pin: datasets whose annotation stamp was "
        "materialized under any other judge prompt hash train as "
        "unjudged, loudly (default: every materialized stamp is the "
        "blessed selection and is consumed as-is; the checkpoint records "
        "the distinct stamps). Pin during sweeps that must fail loudly "
        "on a mid-sweep re-materialization",
    )
    parser.add_argument(
        "--camera-kind-dropout",
        type=float,
        default=0.1,
        help="probability a camera's semantic kind tag renders as "
        "'unknown' at train time (per camera per visit): keeps unjudged "
        "rigs in-distribution at inference. Probes always render true "
        "kinds",
    )
    parser.add_argument(
        "--instruction-augment",
        type=float,
        default=0.0,
        help="probability of swapping the recorded task string for a "
        "uniformly drawn judge-suggested rewrite (phrasing diversity; "
        "judged episodes only — unjudged keep the recorded string). "
        "Probes always score the recorded instruction",
    )
    parser.add_argument(
        "--state-dropout",
        type=float,
        default=0.0,
        help="probability a sample's proprioceptive state is masked to "
        "its dataset mean at collation (normalized state token exactly "
        "zero — the eval --mask-state semantics): trains the policy to "
        "act from vision when state is uninformative, against the "
        "proprioception shortcut (arXiv:2506.23944). Probes always "
        "score intact state",
    )
    parser.add_argument(
        "--image-augment",
        type=float,
        default=0.0,
        help="probability a camera frame gets the train-time sim2real "
        "photometric recipe at collation (brightness/contrast/"
        "saturation/hue/gamma jitter, Gaussian sensor noise, slight "
        "defocus blur, JPEG artifacts, small random crop/translate — "
        "the bijou.modelling.image_augment v0 spec). Probes and evals always "
        "see clean frames",
    )
    parser.add_argument(
        "--condition-fields",
        nargs="*",
        choices=[f.value for f in ConditionField],
        default=None,
        help="outcome conditioning: render these hindsight labels as the "
        "user turn's trailing bracket block ([outcome: success]"
        "[smoothness: high]) — failed/partial demos train under their "
        "own label instead of as-if-good, and inference asks for the "
        "behavior it wants. Unlabeled episodes render nothing",
    )
    parser.add_argument(
        "--condition-dropout",
        type=float,
        default=None,
        help="per-field probability an outcome/smoothness label renders "
        "nothing at train time (keeps the unconditioned marginal "
        "trained). Default 0.1 when --condition-fields is on; requires "
        "--condition-fields",
    )
    parser.add_argument(
        "--subgoal-dropout",
        type=float,
        default=None,
        help="probability the subgoal PROMPT hint renders nothing (its "
        "own rate: deployment mostly runs planner-less, so the "
        "unconditioned context must stay well-trained — and on dropped "
        "draws the aux segment predicts the subgoal instead, the exact "
        "complement). Default 0.5 when subgoal is in --condition-fields; "
        "requires it",
    )
    parser.add_argument(
        "--fast-tokenizer",
        default=None,
        help="FAST tokenizer artifact for the AR-suffix families: a local "
        "directory or <user>/<repo>/<subfolder> on the hub (e.g. "
        "mcobzarenco/bijou-checkpoints/fast_tokenizer_v1)",
    )
    parser.add_argument(
        "--decoder-hidden",
        type=int,
        default=None,
        help="gemma_flow decoder hidden size (fresh-run default 768)",
    )
    parser.add_argument(
        "--decoder-heads",
        type=int,
        default=None,
        help="gemma_flow decoder self-attention heads (fresh-run default 6)",
    )
    parser.add_argument(
        "--decoder-intermediate",
        type=int,
        default=None,
        help="gemma_flow decoder MLP intermediate size (fresh-run default 3072)",
    )
    parser.add_argument(
        "--decoder-cross-heads",
        type=int,
        default=None,
        help="gemma_flow decoder cross-attention heads (fresh-run default 4)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="actions predicted per sample (frames at the dataset fps; "
        "fresh-run default 50, checkpoint-inferred under "
        "--resume/--init-from)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="per rank under torchrun (global batch = batch-size x world size)",
    )
    parser.add_argument(
        "--bucket-by-length",
        action="store_true",
        help="group batches by prompt length (effective camera count) so "
        "rows pad ~nothing instead of to the batch max — throughput only; "
        "changes batch composition, so paired arms must share the flag",
    )
    parser.add_argument(
        "--backward-chunks",
        type=int,
        default=1,
        help="split each optimizer step's forward/backward into this many "
        "equal sample chunks with gradient accumulation (memory fallback "
        "when the loader batch doesn't fit; sample composition, effective "
        "batch and every LR constant are invariant — the gradient equals "
        "the unchunked one up to fp reduction order). Must divide "
        "--batch-size; 1 = the byte-identical unchunked path",
    )
    parser.add_argument(
        "--zero1",
        action="store_true",
        help="shard optimizer state across DDP ranks (ZeRO-1, "
        "torch ZeroRedundancyOptimizer): Adam moments live on one rank "
        "per parameter and updated shards broadcast after each step — "
        "update semantics exact, per-rank optimizer memory ~1/world. "
        "Requires torchrun with world size > 1",
    )
    parser.add_argument(
        "--sync-save",
        action="store_true",
        help="write checkpoints synchronously (legacy path): stepping "
        "blocks for the full serialize+write, and under --zero1 for the "
        "rank-by-rank consolidate broadcast first. Default is async — "
        "device->CPU snapshot at the boundary (seconds), then "
        "gather+merge+write on a background thread over a dedicated "
        "gloo group; written bytes identical (tests/test_async_save.py)",
    )
    parser.add_argument(
        "--chunk-grad-allreduce",
        action="store_true",
        help="with --backward-chunks > 1 under torchrun, train WITHOUT "
        "the DDP wrapper: rank-0 state broadcast at init, plain autograd "
        "gradient accumulation across chunks, one in-place allreduce of "
        "param.grad per step. DDP's reducer bucket buffers (a full fp32 "
        "gradient copy, allocated at construction) never exist. Gradient "
        "equals the DDP sync up to fp reduction order. Requires torchrun "
        "with world size > 1 and --backward-chunks > 1",
    )
    parser.add_argument(
        "--offload-optim",
        action="store_true",
        help="keep the AdamW moments in host RAM (pinned fp32 mirrors, "
        "torch CPU kernels — update semantics exact, oracle-pinned): "
        "the states block of a live-trunk run (measured 33.7 GiB on the "
        "4.2B-param molmoact2_joint set) never touches the GPU, at "
        "~2x trainable-bytes PCIe traffic per step. Single-process "
        "only (refused under torchrun; exclusive with --zero1)",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="activation checkpointing over the molmo2 decoder blocks: "
        "recompute each block in backward instead of retaining "
        "its interior activations — memory only, the gradient is "
        "oracle-pinned bitwise to the plain step. Engages wherever the "
        "trunk runs under grad (live-trunk prefix encode + CE suffix); "
        "no-grad paths are untouched. Molmo2 trunks only",
    )
    parser.add_argument("--steps", type=int, default=200, help="total optimizer steps")
    parser.add_argument(
        "--decoder-lr",
        type=float,
        default=None,
        help="peak learning rate of the 'decoder' param group (cosine "
        "decay to 10%% after warmup; default 1e-4); every component-lr "
        "below shares this schedule shape, scaled to its own peak. "
        "Explicitly passing it for a family whose decoder group is "
        "structurally empty (molmoact2_ar) is an error",
    )
    parser.add_argument(
        "--backbone-text-lr",
        type=float,
        default=None,
        help="peak learning rate for the backbone TEXT stack (decoder "
        "layers up to the deepest exported stream, PLE projections, "
        "multimodal projector); OMIT to keep the backbone frozen (the "
        "historical behavior — the freeze is printed, never silent). "
        "Token embeddings and PLE tables always stay frozen. A live "
        "backbone loads fp32 with bf16-autocast forwards; suggest 1e-5",
    )
    parser.add_argument(
        "--backbone-vision-lr",
        type=float,
        default=None,
        help="peak learning rate for the vision tower; OMIT to keep it "
        "frozen. The acuity probe says position is sharpest at the tower "
        "output and dies in the LM layers - expect to leave this unset",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="linear warmup steps to --lr",
    )
    parser.add_argument(
        "--rewarmup-steps",
        type=int,
        default=0,
        help="extension runs (--resume with a larger --steps): ramp the "
        "LR linearly from ~0 over this many steps AT the resume point "
        "before following the new cosine — the schedule's own warmup is "
        "anchored at step 0 and cannot re-warm a resume",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adamc"],
        default="adamw",
        help="adamc = AdamW with corrected weight decay (arXiv "
        "2506.02285): hidden matrices decay at λ·γt/γmax (the group's "
        "weight_decay tracks the LR schedule), output heads "
        "(VLA.output_head_parameters — audited per family) keep "
        "standard decay, 1-D parameters stay undecayed; adamw is the "
        "unchanged default",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
        help="gradient-norm clip over everything trained",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="steps between metric logs",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="steps between MAE probes (eval_chunk_mae/train_mae)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="steps between checkpoints",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="dataloader workers per rank under torchrun",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="batches prefetched per dataloader worker",
    )
    parser.add_argument(
        "--video-decoder-cache",
        type=int,
        default=4,
        help="max cached video decoders per dataloader worker (exported as "
        "LEROBOT_VIDEO_DECODER_CACHE_SIZE; lerobot's default of 100 "
        "OOM-kills many-dataset runs)",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="warm start: model weights from this VLA checkpoint "
        "directory, fresh optimizer and step count (use a new "
        "--save-dir). The family is checkpoint-inferred; --objective "
        "selects a molmoact2 pathway",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="full resume: weights + optimizer/scheduler/step from this "
        "VLA checkpoint directory (--steps counts total, including "
        "resumed); demands a --seed the checkpoint was not trained with "
        "(see --allow-same-seed-resume)",
    )
    parser.add_argument(
        "--allow-same-seed-resume",
        action="store_true",
        help="resume with the checkpoint's own --seed anyway, replaying "
        "the same epoch-0 shuffle and τ/ε draws it already trained on — "
        "reproduction of a historical run only; the default refuses",
    )
    parser.add_argument(
        "--backbone-init-from",
        type=Path,
        default=None,
        help="stage-2 warm start: load ONLY backbone.safetensors + "
        "prompt.safetensors from this checkpoint (any decoder family), "
        "build the decoder fresh under this run's --family — e.g. a new "
        "flow decoder reading an AR-pretrained trunk. Unlike --init-from, "
        "the source's decoder config is ignored",
    )
    parser.add_argument(
        "--prompt-generate-bracket",
        action="store_true",
        default=None,
        help="render [generate|actions] in prompts for non-AR families "
        "(stage-2 trunk consistency: an AR-pretrained trunk shaped its "
        "conditioning/state positions WITH the bracket). The AR-suffix "
        "families always render it — passing this there is an error, not "
        "a no-op. Checkpoint-inferred under --resume/--init-from (an "
        "AR-suffix source infers True — its prompts carried the bracket "
        "implicitly)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device (cuda, cuda:N, cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="training seed: init, data order, τ/ε draws (per-rank streams "
        "derive from it)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="MAE probe size: eval_chunk_mae on held-out episodes (with "
        "--holdout-episodes) plus train_mae on training data; sampled "
        "without replacement per split, sharded across ranks, evaluated in "
        "batches of --batch-size; required when --holdout-episodes > 0, "
        "omit to disable probing",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="probe sampling/noise seed, independent of --seed; matches the "
        "frames bijou.eval --seed picks on the same data and split",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="enable Weights & Biases logging to this project "
        "(WANDB_API_KEY must be set)",
    )
    parser.add_argument("--wandb-run-name", default=None, help="wandb run display name")
    return parser


def parse_args() -> TrainArgs:
    parser = _build_parser()
    raw = parser.parse_args()
    checkpoint_path = raw.resume if raw.resume is not None else raw.init_from
    checkpoint = (
        resolve_checkpoint(checkpoint_path) if checkpoint_path is not None else None
    )
    return TrainArgs.from_namespace(raw, parser, checkpoint=checkpoint)
