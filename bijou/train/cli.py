"""Train a Bijou VLA family on a LeRobot v3 dataset.

A run trains ONE model family (``--family``, §5 of
docs/architecture.md): the family class owns assembly, precision
policy and loss composition; this module owns data, optimizer policy,
the DDP-correct two-phase loop, and checkpoints. Fresh runs declare
the family; under ``--init-from``/``--resume`` it is checkpoint-
inferred (the flag is refused, like every architecture flag).

The two-phase loop (the DDP contract): per step,
``counts = model.loss_counts(batch)`` (data-only) → all-reduce →
``report = model(batch, counts=counts)`` → ``report.objective``
backward. The objective is the per-rank scalar whose DDP mean is the
global objective, exact under uneven per-rank counts; chunked backward
(``--backward-chunks``) forwards micro-slices against the SAME summed
counts, so the accumulated gradient equals the unchunked one up to fp
reduction order.

Actions and state are MEAN_STD-normalized **per dataset** (each sample
uses its own dataset's stats, the π0/SmolVLA convention): 59–95% of the
aggregate action variance across the community collections is
between-dataset rig offsets that images cannot see, and normalizing
them away is what makes the state→action identity learnable (measured:
aggregate normalization left the trained model behind the state-copy
baseline). Checkpoints store the per-dataset stats table plus a
count-weighted aggregate as a fallback for rigs without stats;
inference must unnormalize with the deployment rig's stats.

Training data is selected with ``--train-data`` via the shared selection
pipeline in ``bijou.data`` (also used by ``bijou.eval``): collection roots
and dataset dirs, loud drops for incompatible/corrupt datasets, per-dataset
stats attachment, chat-templated prompt collation.

Checkpoints are written in the VLA format (``bijou/checkpoint.py``):
``metadata.json`` + per-component safetensors + the always-present
backbone (trained snapshot, or a hard-linked pristine mirror).

Usage::

    uv run python -m bijou.train --family gemma_flow \
        --train-data ~/datasets/mcobzarenco/community_dataset_v2_v3 \
        --device cuda --steps 5000

    # Multi-GPU: one full replica + optimizer per GPU (DDP). --batch-size
    # and --num-workers are PER RANK; logged loss and eval MAE are
    # all-reduced across ranks (the eval set is sharded);
    # logging/checkpoints happen on rank 0. Without torchrun the script
    # runs exactly as before.
    MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 \
    uv run torchrun --standalone --nproc-per-node=4 -m bijou.train \
        --family gemma_flow \
        --train-data ~/datasets/mcobzarenco/community_dataset_v2_v3 \
        --device cuda --steps 20000
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import matplotlib
import torch
import transformers
import wandb
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.distributed.optim import ZeroRedundancyOptimizer

from ..annotations import ConditionField
from ..async_save import (
    AsyncCheckpointSaver,
    capture_optimizer_state,
    copy_to_cpu,
)
from ..checkpoint import VLAMetadata, backbone_directory, read_metadata
from ..data import (
    EpisodeSplit,
    LengthBucketedBatchSampler,
    StatsAttachedDataset,
    parse_repeat_specs,
    select_datasets,
    worker_init,
)
from ..fast.molmoact2 import MolmoAct2FastTokenizer
from ..loading import molmo_flow_state_table
from ..modelling.aux_text import (
    AUX_TEMPLATE_VERSION,
    AuxDecodeConfig,
    AuxField,
    AuxSpec,
    build_aux_runtime,
)
from ..modelling.codecs import ActionCodec, MolmoAct2ActionCodec
from ..modelling.decoders.ar_gemma import GemmaARDecoder
from ..modelling.decoders.ar_molmo2 import Molmo2ARDecoder
from ..modelling.decoders.ar_molmoact2 import MolmoAct2ARDecoder
from ..modelling.decoders.ar_suffix import (
    MOLMOACT2_SUFFIX_FORMAT,
    SUFFIX_FORMAT,
    ARDecoderConfig,
)
from ..modelling.decoders.flow import (
    SelfAttentionMode,
    TimeConditioning,
)
from ..modelling.decoders.molmo_flow import load_expert_state
from ..modelling.encoders.gemma4 import GemmaInputsCollator
from ..modelling.encoders.molmo2 import Molmo2Encoder, Molmo2InputsCollator
from ..modelling.encoders.molmoact2 import MolmoAct2Encoder, MolmoAct2InputsCollator
from ..modelling.gemma4.loading import load_config, resolve_checkpoint_dir
from ..modelling.interface import Collator, SamplingMethod
from ..modelling.molmo2.config import Molmo2TextConfig
from ..modelling.molmo2.loading import load_config as load_molmo2_config
from ..modelling.molmo2.model import Molmo2Model
from ..modelling.molmo2.model import load_model as load_molmo2_model
from ..modelling.molmo2.tokenizer import Molmo2TextTokenizer, newline_carrier_ids
from ..models.gemma_ar import GemmaARVLA
from ..models.gemma_flow import GemmaFlowVLA
from ..models.molmo2_ar import Molmo2ARVLA
from ..models.molmoact2_ar import MolmoAct2ARVLA
from ..models.molmoact2_flow import MolmoAct2FlowVLA, molmoact2_prompt_of
from ..models.molmoact2_joint import JointObjective, MolmoAct2JointVLA
from ..models.objectives import ARObjective, FlowObjective, SnapflowObjective
from ..models.serving import ARServing, FlowServing
from ..sections import (
    BackboneDepth,
    FlowDecoderSection,
    GemmaPromptConfig,
    MolmoAct2PromptConfig,
    MolmoFlowDecoderConfig,
    build_gemma_encoder,
    build_gemma_flow_parts,
    build_molmo_flow_decoder,
    build_molmoact2_ar_decoder,
    default_expert_config,
    expert_config_from_architecture,
    load_backbone_state,
    molmoact2_ar_config_from_flow_section,
    molmoact2_fresh_flow_section,
    parse_decoder_config,
    parse_prompt_config,
    resolve_action_codec,
)
from ..vla import VLA
from .args import (
    AR_SUFFIX_FAMILIES,
    MOLMOACT2_FAMILIES,
    TrainArgs,
    parse_args,
    reconcile_lr_offer,
    train_args_record,
)
from .loop import (
    ChunkedBatch,
    ChunkingCollator,
    DevicePrefetcher,
    ProbeSet,
    allreduce_gradients,
    broadcast_module_states,
    build_probe_set,
    summed_loss_counts,
    validate,
)
from .saving import (
    CheckpointTensors,
    Normalizer,
    Normalizers,
    TrainableVLA,
    TrainState,
    build_objective,
    build_vla_metadata,
    capture_checkpoint_tensors,
    objective_to_json,
    save_checkpoint,
    serving_to_json,
    write_checkpoint,
)


@dataclass(frozen=True, slots=True)
class BackboneParameterCounts:
    """Trainable backbone parameters enabled by :func:`unfreeze_backbone`,
    by subsystem (0 = that subsystem stayed frozen)."""

    text: int
    vision: int


def unfreeze_backbone(model: VLA[Any], args: TrainArgs) -> BackboneParameterCounts:
    """Flip ``requires_grad`` on the requested backbone subsets; everything
    else stays frozen (the trunk loaders freeze the whole backbone).
    Empty-group contradictions are the reconciliation's job
    (:func:`reconcile_lr_offer` runs first).

    Freezing by ``requires_grad`` alone is sufficient for efficiency too:
    token embeddings, PLE tables and (when frozen) the vision tower feed
    the decoder grad-free inputs, so autograd never builds their graphs —
    no activation cost, no backward — without any code-path changes
    inside the trunks.
    """
    groups = model.param_groups()
    text = 0
    vision = 0
    if args.backbone_text_lr is not None:
        for parameter in groups["backbone_text"]:
            parameter.requires_grad_(True)
            text += parameter.numel()
    if args.backbone_vision_lr is not None:
        for parameter in groups["backbone_vision"]:
            parameter.requires_grad_(True)
            vision += parameter.numel()
    return BackboneParameterCounts(text=text, vision=vision)


def decay_split(
    parameters: Iterable[torch.nn.Parameter],
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """(decayed, undecayed): weight decay belongs on matmul weights, not on
    norm scales — the standard ndim heuristic (norm weights and biases are
    1-D)."""
    decayed: list[torch.nn.Parameter] = []
    undecayed: list[torch.nn.Parameter] = []
    for parameter in parameters:
        (decayed if parameter.dim() >= 2 else undecayed).append(parameter)
    return decayed, undecayed


def build_optimizer_param_groups(
    model: VLA[Any],
    *,
    optimizer_name: str,
    decoder_lr: float,
    backbone_text_lr: float | None,
    backbone_vision_lr: float | None,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], list[tuple[str, float, float]], list[bool]]:
    """Optimizer param groups + bookkeeping, in construction order.

    Returns ``(param_groups, cli_groups, adamc_corrected)``:
    ``param_groups`` in torch's optimizer API format (fixed-key dicts, a
    third-party boundary); ``cli_groups`` the CLI intent per group as
    (name, lr, weight_decay) — what --resume's restored optimizer state
    is checked against; ``adamc_corrected`` flags the groups that take
    the AdamC corrected decay (parallel to param_groups; all-False under
    adamw, where the training loop never reads it).

    adamw: the historical construction, byte-identical — one decoder
    group (decoder + prompt-side encoder params, all at the default
    decay) plus a decayed/undecayed split per unfrozen backbone subset.

    adamc: the decoder group splits by role — hidden matrices →
    corrected decay, output head (``VLA.output_head_parameters``) →
    standard decay, 1-D (norm scales, biases) → no decay. Families
    whose partition is unaudited refuse adamc loudly from that method.
    Backbone hidden matrices are exactly the paper's "normalized"
    layers (the trunk head/embeddings are frozen out of the groups by
    design), so the existing decayed split takes the corrected flag.

    Both modes end with the shared/tied-parameter guard: a parameter
    object appearing in two groups would be stepped and decayed twice
    (the tied-lm_head failure mode); one appearing in none would take
    gradients but never step. Groups must exactly cover the trainable
    set, disjointly — loud SystemExit otherwise."""
    named_groups = model.param_groups()
    param_groups: list[dict[str, Any]] = []
    cli_groups: list[tuple[str, float, float]] = []
    adamc_corrected: list[bool] = []
    if optimizer_name == "adamc":
        head_ids = {id(p) for p in model.output_head_parameters() if p.dim() >= 2}
        decayed, undecayed = decay_split(named_groups["decoder"])
        hidden = [p for p in decayed if id(p) not in head_ids]
        heads = [p for p in decayed if id(p) in head_ids]
        if len(heads) != len(head_ids):
            raise SystemExit(
                "--optimizer adamc: an output-head parameter is not in "
                "the decoder param group — the corrected/standard "
                "partition would misroute it (audit "
                "output_head_parameters vs model.param_groups)",
            )
        param_groups.append({"params": hidden, "lr": decoder_lr})
        cli_groups.append(("decoder (corrected decay)", decoder_lr, weight_decay))
        adamc_corrected.append(True)
        param_groups.append({"params": heads, "lr": decoder_lr})
        cli_groups.append(("decoder head (standard decay)", decoder_lr, weight_decay))
        adamc_corrected.append(False)
        param_groups.append(
            {"params": undecayed, "lr": decoder_lr, "weight_decay": 0.0},
        )
        cli_groups.append(("decoder (no decay)", decoder_lr, 0.0))
        adamc_corrected.append(False)
    else:
        param_groups.append({"params": named_groups["decoder"], "lr": decoder_lr})
        cli_groups.append(("decoder", decoder_lr, weight_decay))
        adamc_corrected.append(False)
    for group_name, group_lr in (
        ("backbone_text", backbone_text_lr),
        ("backbone_vision", backbone_vision_lr),
    ):
        if group_lr is None:
            continue
        assert named_groups[group_name]  # reconcile_lr_offer validated
        decayed, undecayed = decay_split(named_groups[group_name])
        param_groups.append({"params": decayed, "lr": group_lr})
        cli_groups.append((f"{group_name} (decayed)", group_lr, weight_decay))
        adamc_corrected.append(optimizer_name == "adamc")
        param_groups.append(
            {
                "params": undecayed,
                "lr": group_lr,
                "weight_decay": 0.0,
            },
        )
        cli_groups.append((f"{group_name} (no decay)", group_lr, 0.0))
        adamc_corrected.append(False)
    flat_group_params = [p for group in param_groups for p in group["params"]]
    if len(flat_group_params) != len({id(p) for p in flat_group_params}):
        raise SystemExit(
            "param groups overlap — a shared/tied parameter appears in "
            "two optimizer groups and would be decayed twice",
        )
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if {id(p) for p in flat_group_params} != trainable_ids:
        raise SystemExit(
            "param groups do not cover the trainable parameter set "
            "exactly — a trainable parameter is missing from (or frozen "
            "yet present in) the optimizer groups",
        )
    return param_groups, cli_groups, adamc_corrected


def backbone_group_index(cli_groups: Sequence[tuple[str, float, float]]) -> int | None:
    """Index of the first backbone param group, in construction order.

    The decoder contributes ONE leading group under adamw but THREE
    under adamc (hidden/head/no-decay), so a hardcoded index reads the
    decoder head's lr as ``lr_backbone`` in adamc logs — the group
    table is the only stable source."""
    return next(
        (
            index
            for index, (name, _, _) in enumerate(cli_groups)
            if name.startswith("backbone_")
        ),
        None,
    )


def apply_adamc_weight_decay(
    optimizer: torch.optim.Optimizer,
    corrected_indices: Sequence[int],
    weight_decay: float,
) -> None:
    """One AdamC decay update, called immediately BEFORE optimizer.step().

    λ̂_t = λ·γ_t/γ_max per corrected group, written into the group's
    ``weight_decay`` so the stock (possibly fused) AdamW kernel applies
    it — bit-exact AdamC, no custom kernel, O(#groups) Python per step.
    ``group["lr"]`` holds γ_t at this point (the scheduler steps after
    the optimizer), and γ_max per group is its ``initial_lr`` (LambdaLR
    records it; every lr_lambda branch peaks at exactly 1.0, warmup and
    the re-warmup ramp included — during warmup λ̂_t < λ is the paper's
    intended behavior). ZeRO-1's step() copies group attributes wrapper
    → local optimizer before the sharded step, so this same write covers
    the --zero1 path exactly."""
    for index in corrected_indices:
        group = optimizer.param_groups[index]
        group["weight_decay"] = (
            weight_decay * float(group["lr"]) / float(group["initial_lr"])
        )


def lr_lambda(step: int, args: TrainArgs, resume_step: int = 0) -> float:
    """Cosine with linear warmup, flooring at 10% of peak. With
    ``rewarmup_steps`` and a resume, a second linear ramp anchors at the
    RESUME step: an extension that re-raises the LR off the floor (e.g.
    --steps 40k→80k jumps ~1e-5→5.5e-5) would otherwise shock a model
    whose first moments were accumulated in the floor-LR regime — the
    ramp rebuilds update magnitudes gradually while keeping the (LR-
    independent, well-estimated) second moments. Re-warm + re-decay,
    the continued-pretraining recipe."""
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
    base = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    if args.rewarmup_steps > 0 and step >= resume_step:
        base *= min(1.0, (step - resume_step + 1) / args.rewarmup_steps)
    return base


def check_resume_seed(resume: Path, seed: int, *, allow_same_seed: bool) -> str:
    """The fresh-seed-on-resume convention, enforced. Nothing restores the
    data-stream position on --resume: the loop restarts at epoch 0 with
    the --seed shuffle and per-rank τ/ε streams, so resuming with the
    checkpoint's own seed replays exactly the batches and noise draws it
    already trained on. Returns the line to log; raises SystemExit on a
    same-seed resume unless --allow-same-seed-resume (reproduction of a
    historical run) was passed explicitly."""
    metadata_path = resume / "metadata.json"
    try:
        recorded = json.loads(metadata_path.read_text())
    except FileNotFoundError:
        raise SystemExit(
            f"{metadata_path} missing — not a checkpoint directory "
            "(legacy bijou_config.json directories convert via "
            "bijou.convert_legacy)",
        ) from None
    checkpoint_seed = recorded.get("train_args", {}).get("seed")
    if checkpoint_seed is None:
        return (
            "WARNING: resume seed check skipped — checkpoint records no "
            "train_args seed (converted artifacts predate it); the "
            "fresh-seed-on-resume convention cannot be verified, make "
            "sure --seed differs from the original run's"
        )
    if int(checkpoint_seed) != seed:
        return (
            f"resume seed check: fresh --seed {seed} (checkpoint trained "
            f"with {checkpoint_seed})"
        )
    if allow_same_seed:
        return (
            f"WARNING: same-seed resume (--seed {seed}) allowed by "
            "--allow-same-seed-resume — the epoch-0 shuffle and τ/ε "
            "draws REPLAY batches the checkpoint already trained on"
        )
    raise SystemExit(
        f"--resume with the checkpoint's own --seed {seed}: resume "
        "restarts the data stream at epoch 0, so the same seed replays "
        "exactly the batches and τ/ε draws already trained on. Pass a "
        "fresh --seed (any value the run's segments have not used), or "
        "--allow-same-seed-resume to reproduce a historical run.",
    )


def resume_hyperparameter_notes(
    optimizer: torch.optim.Optimizer,
    cli_groups: Sequence[tuple[str, float, float]],
    adamc_corrected: Sequence[bool] | None = None,
) -> list[str]:
    """After optimizer.load_state_dict on --resume the checkpoint's
    hyperparameters win and CLI values are ignored — surface EVERY
    ignored difference. (The historical note checked param group 0 only:
    a changed --backbone-*-lr on resume was silently ignored.)
    ``cli_groups`` carries (name, lr, weight_decay) per param group, in
    construction order. AdamC corrected groups are the one exception to
    "the checkpoint wins" for weight decay: their group weight_decay is
    recomputed from the CLI λ before every step, so the checkpoint's
    saved λ̂ (λ scaled by the save-time LR ratio) is transient state,
    not a hyperparameter — noted, never compared."""
    assert len(optimizer.param_groups) == len(cli_groups)
    if adamc_corrected is not None:
        assert len(adamc_corrected) == len(cli_groups)
    notes: list[str] = []
    for index, (group, (name, cli_lr, cli_decay)) in enumerate(
        zip(
            optimizer.param_groups,
            cli_groups,
            strict=True,
        ),
    ):
        base_lr = float(group.get("initial_lr", group["lr"]))
        if base_lr != cli_lr:
            notes.append(
                f"{name}: base lr {base_lr:.2e} (CLI {cli_lr:.2e} ignored)",
            )
        if adamc_corrected is not None and adamc_corrected[index]:
            notes.append(
                f"{name}: weight decay is schedule-managed (adamc) — "
                f"recomputed each step as --weight-decay {cli_decay} "
                "x γt/γmax; the CLI value GOVERNS here",
            )
            continue
        if float(group["weight_decay"]) != cli_decay:
            notes.append(
                f"{name}: weight decay {group['weight_decay']} "
                f"(CLI {cli_decay} ignored)",
            )
    return notes


def rehome_fused_step_tensors(
    optimizer: torch.optim.Optimizer,
    *,
    fused: bool,
) -> int:
    """PyTorch's ``Optimizer.load_state_dict`` re-homes floating-point
    state to each param's device but decides where integer ``step``
    tensors live from the SAVED group's fused/capturable flags — and a
    consolidated resume payload (async-save's device→CPU capture +
    ZeRO-1 merge) stores everything CPU-tagged without those flags, so
    the steps stay on CPU and fused AdamW aborts at the first
    ``optimizer.step()`` ('state_steps is on cpu…', measured live on
    a resumed run). Move each param's step
    counter to the param's device. Exact no-op for non-fused (CPU)
    runs, whose reference kernels want CPU scalar steps. Handles the
    ZeRO-1 wrapper by operating on its local inner optimizer."""
    if not fused:
        return 0
    inner = getattr(optimizer, "optim", optimizer)
    moved = 0
    for group in inner.param_groups:
        for param in group["params"]:
            state = inner.state.get(param)
            if not state:
                continue
            step = state.get("step")
            if isinstance(step, torch.Tensor) and step.device != param.device:
                state["step"] = step.to(param.device)
                moved += 1
    return moved


def length_bucket_keys(
    datasets: list[StatsAttachedDataset],
    collator: Collator[Any],
) -> list[int]:
    """Per-index prompt-length key for ``--bucket-by-length``: the
    EFFECTIVE camera count (the collator's own selection policy applied
    to each dataset's camera metadata — filter, truncation, prompt
    order). Camera count dominates prompt length (``--max-soft-tokens``
    per camera vs tens of tokens of text that also vary per draw), and
    a bucketing key only needs to order the lengths. Metadata-only: no
    item is fetched, no video touched.
    """
    keys: list[int] = []
    for sub in datasets:
        # A synthetic item carrying exactly what cameras_of consumes:
        # the image keys and the per-dataset camera-kind map.
        probe_item: dict[str, Any] = dict.fromkeys(
            sub.dataset.meta.camera_keys,
        )
        probe_item["camera_kinds"] = sub.camera_kinds
        keys.extend([len(collator.cameras_of(probe_item))] * len(sub))
    return keys


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Headless rendering for the eval plots regardless of DISPLAY (must be
    # forced before any figure is created; imports stay at the top).
    matplotlib.use("Agg", force=True)
    args = parse_args()
    # Bound each dataloader worker's torchcodec decoder cache. Spawned
    # workers inherit os.environ, and lerobot reads this when it constructs
    # its per-process cache at import time inside the worker. (The main
    # process's own cache was already built with lerobot's default at import
    # time here, but the main process only decodes the small eval set once.)
    if args.video_decoder_cache < 1:
        raise SystemExit("--video-decoder-cache must be >= 1")
    if args.backward_chunks < 1 or args.batch_size % args.backward_chunks != 0:
        raise SystemExit(
            f"--backward-chunks {args.backward_chunks} must be >= 1 and "
            f"divide --batch-size {args.batch_size} (equal chunks)",
        )
    os.environ["LEROBOT_VIDEO_DECODER_CACHE_SIZE"] = str(args.video_decoder_cache)

    # TF32 for fp32 matmuls (torch's default is full-IEEE "highest"): the
    # flow decoders train in fp32, and true fp32 matmul leaves ~5-7x of H100
    # throughput on the table. TF32's per-op 10-bit mantissa is far above
    # bf16 and standard for training; the bf16 backbone is unaffected.
    torch.set_float32_matmul_precision("high")

    # Data parallelism (torchrun): one full replica + optimizer per rank.
    # Without WORLD_SIZE in the environment this is a plain single-process
    # run — identical behavior to before DDP support existed.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if args.zero1 and not distributed:
        raise SystemExit(
            "--zero1 shards optimizer state across ranks and needs "
            "torchrun with world size > 1 (a single-process run has "
            "nothing to shard across)",
        )
    if args.chunk_grad_allreduce and (not distributed or args.backward_chunks < 2):
        raise SystemExit(
            "--chunk-grad-allreduce replaces DDP's final-chunk gradient "
            "sync and needs torchrun with world size > 1 AND "
            "--backward-chunks > 1 (without both there is no reducer "
            "sync to replace)",
        )
    device = torch.device(args.device)
    rank = 0
    if distributed:
        torch.distributed.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
        )
        rank = torch.distributed.get_rank()
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
    is_main = rank == 0

    # Memory forensics (BIJOU_MEM_SNAPSHOT=<path prefix>): record every
    # CUDA allocation with its python stack from process start; the
    # __main__ epilogue dumps a per-rank snapshot pickle if the run
    # dies OOM (analyze with torch.cuda._memory_viz).
    # Measurement instrument for smoke runs — never set on a
    # real launch (host-side overhead per allocation).
    if os.environ.get("BIJOU_MEM_SNAPSHOT") and device.type == "cuda":
        torch.cuda.memory._record_memory_history(max_entries=250_000)
        if is_main:
            print(
                "mem-snapshot: recording allocation history "
                f"(dump prefix {os.environ['BIJOU_MEM_SNAPSHOT']})",
                flush=True,
            )

    # Fail fast, before data/model build: the fresh-seed-on-resume
    # convention (all ranks read the same file — every rank raises or
    # none does).
    if args.resume is not None:
        seed_note = check_resume_seed(
            args.resume,
            args.seed,
            allow_same_seed=args.allow_same_seed_resume,
        )
        if is_main:
            print(seed_note, flush=True)

    # Per-rank RNG stream (τ and ε draws must decorrelate across ranks).
    # Dataloader worker seeds derive from this deterministically: torch
    # draws each worker's base seed from the parent process RNG, so worker
    # seeds are a pure function of (--seed, rank, worker_id). The
    # DistributedSampler below is seeded with the BASE seed on every rank
    # so the shuffled partition stays coordinated.
    torch.manual_seed(args.seed + rank)

    # -- source checkpoint + trunk directory ------------------------------
    # Under --init-from/--resume the trunk mounts from the checkpoint's
    # own materialization (self-containment: the pristine backbone/
    # mirror, or the recorded artifact for trained trunks); fresh runs
    # resolve --backbone.
    source = args.init_from if args.init_from is not None else args.resume
    source_metadata: VLAMetadata | None = None
    if source is not None:
        source_metadata = read_metadata(source)
        checkpoint_dir = backbone_directory(source, source_metadata)
    else:
        checkpoint_dir = resolve_checkpoint_dir(args.backbone)
    # Trunk cross-check: the family DECLARES the trunk lineage; the
    # artifact's own config.json must agree.
    backbone_model_type = json.loads(
        (checkpoint_dir / "config.json").read_text(),
    ).get("model_type", "")
    molmo2_trunk = backbone_model_type in ("molmo2", "molmoact2")
    if args.family.startswith("gemma") and molmo2_trunk:
        raise SystemExit(
            f"--family {args.family} rides a Gemma trunk but {args.backbone} "
            f"is a {backbone_model_type} artifact",
        )
    if args.family == "molmo2_ar" and backbone_model_type != "molmo2":
        raise SystemExit(
            f"--family molmo2_ar rides a Molmo2 trunk but {args.backbone} "
            f"records model_type {backbone_model_type!r}",
        )
    if backbone_model_type == "molmoact2" and args.family not in MOLMOACT2_FAMILIES:
        raise SystemExit(
            f"{args.backbone} is a MolmoAct2 artifact — it hosts the "
            "molmoact2 families only (inherit them: --init-from a "
            "converted checkpoint)",
        )

    # The molmoact2 compositions rebuild from the source checkpoint's
    # component sections (inherit-only) — read them once, early. The
    # pathway matrix derives per-family sections here: the flow section
    # for flow/joint (synthesized under --flow-decoder-init fresh from
    # ar-only sources), the format-6 AR section for ar/joint (derived
    # from a flow-section source's geometry + trunk tokenizer on first
    # transition).
    molmoact2_prompt: MolmoAct2PromptConfig | None = None
    molmo_flow_section: MolmoFlowDecoderConfig | None = None
    molmoact2_ar_section: ARDecoderConfig | None = None
    source_has_flow_weights = False
    if args.family in MOLMOACT2_FAMILIES:
        assert source is not None and source_metadata is not None  # inherit-only
        molmoact2_prompt = molmoact2_prompt_of(source_metadata)
        flow_record = source_metadata.components.get("flow_decoder")
        if flow_record is not None:
            parsed_flow = parse_decoder_config(dict(flow_record["config"]))
            if not isinstance(parsed_flow, MolmoFlowDecoderConfig):
                raise SystemExit(
                    f"{source} records a {type(parsed_flow).__name__} as "
                    "flow_decoder — the molmoact2 flow pathway carries the "
                    "molmo_flow section",
                )
            molmo_flow_section = parsed_flow
            source_has_flow_weights = bool(flow_record["weights"])
        ar_record = source_metadata.components.get("ar_decoder")
        if ar_record is not None:
            parsed_ar = parse_decoder_config(dict(ar_record["config"]))
            if not isinstance(parsed_ar, ARDecoderConfig):
                raise SystemExit(
                    f"{source} records a {type(parsed_ar).__name__} as "
                    "ar_decoder — expected a format-6 ar_backbone section",
                )
            if parsed_ar.suffix_format != MOLMOACT2_SUFFIX_FORMAT:
                raise SystemExit(
                    f"{source} records a format-{parsed_ar.suffix_format} "
                    "ar_decoder under the molmoact2 prompt — this prompt "
                    f"family's emission is format {MOLMOACT2_SUFFIX_FORMAT}",
                )
            molmoact2_ar_section = parsed_ar
        if args.family in ("molmoact2_ar", "molmoact2_joint") and (
            molmoact2_ar_section is None
        ):
            # A flow/release source records no format-6 section: derive
            # it once (geometry from the flow section, block ids from
            # the trunk tokenizer's own <action_0>).
            if molmo_flow_section is None:
                raise SystemExit(
                    f"{source} records neither a flow_decoder nor an "
                    "ar_decoder section — not a molmoact2 checkpoint "
                    "this run can rebuild from",
                )
            molmoact2_ar_section = molmoact2_ar_config_from_flow_section(
                molmo_flow_section,
                molmoact2_prompt,
                str(checkpoint_dir),
            )
        if args.family in ("molmoact2_flow", "molmoact2_joint"):
            if molmo_flow_section is None or not source_has_flow_weights:
                # An ar-only source carries no flow decoder to inherit.
                if args.flow_decoder_init == "inherit":
                    raise SystemExit(
                        f"--objective {'flow' if args.family == 'molmoact2_flow' else 'joint'} "
                        f"from the ar-only checkpoint {source} — it carries "
                        "no flow decoder to inherit; pass "
                        "--flow-decoder-init fresh (released-shape init) or "
                        "--flow-decoder-init <checkpoint> (borrow one)",
                    )
                if args.flow_decoder_init == "fresh":
                    assert molmoact2_ar_section is not None
                    molmo_flow_section = molmoact2_fresh_flow_section(
                        molmoact2_ar_section,
                    )
            if args.flow_decoder_init not in ("inherit", "fresh"):
                # The two-source init: the flow section comes from the
                # REF checkpoint (config-equality is the guard); read it
                # now so the decoder builds the right shape.
                ref_metadata = read_metadata(Path(args.flow_decoder_init))
                ref_record = ref_metadata.components.get("flow_decoder")
                if ref_record is None or not ref_record["weights"]:
                    raise SystemExit(
                        f"--flow-decoder-init {args.flow_decoder_init} is "
                        "not a molmo_flow-carrying checkpoint — nothing to "
                        "borrow",
                    )
                ref_section = parse_decoder_config(dict(ref_record["config"]))
                assert isinstance(ref_section, MolmoFlowDecoderConfig)
                if (
                    molmo_flow_section is not None
                    and source_has_flow_weights
                    and ref_section != molmo_flow_section
                ):
                    raise SystemExit(
                        f"--flow-decoder-init {args.flow_decoder_init}: the "
                        "ref's molmo_flow section differs from the source "
                        "checkpoint's — borrowed weights must match the "
                        "shape being built exactly (config-equality guard)",
                    )
                molmo_flow_section = ref_section

    # -- datasets --------------------------------------------------------
    selection = select_datasets(
        args.train_data,
        args.exclude,
        args.chunk_size,
        episode_split=EpisodeSplit.TRAIN,
        holdout_fraction=args.holdout_episodes,
        split_seed=args.split_seed,
        allowed_fps=args.fps,
        allowed_camera_counts=args.camera_counts,
        required_prompt_hash=args.aux_prompt_hash,
        load_episode_annotations=(
            args.instruction_augment > 0 or args.condition_fields is not None
        ),
        repeats=parse_repeat_specs(args.dataset_repeat),
    )
    action_dim, state_dim = selection.action_dim, selection.state_dim
    per_dataset_stats = selection.per_dataset_stats
    dataset = selection.concat()
    if is_main:
        # per_dataset_stats counts unique datasets (selection.datasets
        # carries --dataset-repeat replicas); frame count is effective.
        print(
            f"train data: {len(selection.per_dataset_stats)} datasets, "
            f"{selection.total_episodes} episodes, {len(dataset)} frames, "
            f"action/state dim {action_dim}/{state_dim}",
            flush=True,
        )
        # len(dataset) above already includes the replicas, so the share
        # printed here is the effective (post-repeat) mixture share.
        unique_frames = {sub.dataset.repo_id: len(sub) for sub in selection.datasets}
        for repo_id, count in selection.repeats.items():
            frames = unique_frames[repo_id]
            print(
                f"dataset repeat: {repo_id} x{count} ({frames} -> "
                f"{count * frames} frames, {count * frames / len(dataset):.2%} "
                "effective share)",
                flush=True,
            )
        if args.holdout_episodes > 0:
            print(
                f"episode holdout: {selection.held_out_episodes} episodes "
                f"across {selection.held_out_datasets} datasets excluded "
                f"(fraction {args.holdout_episodes}, split seed "
                f"{args.split_seed})",
                flush=True,
            )
        for camera_set, count in selection.camera_census.most_common():
            print(f"  {count:4d} x cameras {camera_set}", flush=True)
        if selection.dropped:
            print(
                f"dropped {len(selection.dropped)} incompatible datasets:",
                flush=True,
            )
            for reason in selection.dropped:
                print(f"  - {reason}", flush=True)

    # Aggregate stats are NOT used for training math (normalization is per
    # dataset) — they ride along in checkpoints as a fallback for rigs
    # without their own stats.
    normalizers = Normalizers(
        action=Normalizer.from_aggregated_stats(
            list(selection.lerobot_stats.values()),
            "action",
            device,
        ),
        state=Normalizer.from_aggregated_stats(
            list(selection.lerobot_stats.values()),
            "observation.state",
            device,
        ),
    )

    action_codec: ActionCodec | None = (
        resolve_action_codec(args.fast_tokenizer)
        if args.fast_tokenizer is not None
        else None
    )
    # The molmoact2 ar/joint pathways tokenize CE targets through the
    # released codec under the ONE merged table (their shared-table
    # convention, and molmo_flow's clamp table — same row, one source of
    # truth). The decoder itself resolves the codec at build; the
    # COLLATOR needs the same tables to tokenize targets.
    molmoact2_action_table: tuple[Tensor, Tensor] | None = None
    if args.family in ("molmoact2_ar", "molmoact2_joint"):
        assert molmoact2_ar_section is not None and source_metadata is not None
        assert action_codec is None  # --fast-tokenizer refused for the family
        action_codec = MolmoAct2ActionCodec(
            MolmoAct2FastTokenizer.load(
                resolve_checkpoint_dir(molmoact2_ar_section.tokenizer),
            ),
            time_horizon=molmoact2_ar_section.chunk_size,
            action_dim=molmoact2_ar_section.action_dim,
            # The reference recipe's behavior for real chunks that hit
            # the 7 released-BPE holes: tokenize the short stream as-is
            # (counted + printed by the codec). Refusal is for parity
            # harnesses; a training run cannot die on a data artifact
            # the released tokenizer itself produces.
            allow_quantization_holes=True,
        )
        source_normalization = source_metadata.stats
        if (
            source_normalization.action_q01 is None
            or source_normalization.action_q99 is None
        ):
            raise SystemExit(
                "the source checkpoint's normalization row carries no "
                "action q01/q99 — the discrete head tokenizes under the "
                "merged table; this table predates it",
            )
        molmoact2_action_table = (
            torch.tensor(source_normalization.action_q01, dtype=torch.float32),
            torch.tensor(source_normalization.action_q99, dtype=torch.float32),
        )
    aux_decode_config: AuxDecodeConfig | None = None
    aux_spec: AuxSpec | None = None
    if args.aux_fields is not None:
        assert action_codec is not None  # parse guard (AR-suffix-only)
        if not selection.annotated_repos:
            raise SystemExit(
                "--aux-fields but NO selected dataset carries a materialized "
                "annotation stamp — nothing to supervise (see the [aux] "
                "lines above"
                + (
                    f"; required prompt hash {args.aux_prompt_hash}"
                    if args.aux_prompt_hash is not None
                    else ""
                )
                + ")",
            )
        # The stamps ARE the label provenance (the in-band blessed
        # selection); the checkpoint records the distinct set. Mixed
        # stamps in one corpus are legitimate (each dataset's labels
        # match its own stamp) — pin --aux-prompt-hash to refuse them.
        aux_decode_config = AuxDecodeConfig(
            template_version=AUX_TEMPLATE_VERSION,
            fields=tuple(AuxField(f) for f in args.aux_fields),
            prompt_hash="+".join(
                sorted({s.prompt_hash for s in selection.annotation_stamps}),
            ),
            judge_model="+".join(
                sorted({s.judge_model for s in selection.annotation_stamps}),
            ),
        )
        aux_spec = AuxSpec(
            tokenizer_dir=str(checkpoint_dir),
            fields=aux_decode_config.fields,
            annotated_repos=selection.annotated_repos,
            block_base=(
                load_molmo2_config(checkpoint_dir).text.fast_block_base
                if molmo2_trunk
                else load_config(checkpoint_dir).text.vocab_size
                - action_codec.vocab_total
            ),
            dropout=args.aux_dropout,
            field_dropout=args.field_dropout,
            native_backend=molmo2_trunk,
        )
        if is_main:
            print(
                f"aux: {len(selection.annotated_repos)} annotated dataset(s) "
                f"@ {aux_decode_config.prompt_hash} "
                f"(judges: {aux_decode_config.judge_model}), "
                f"fields {list(args.aux_fields)}, loss weight "
                f"{args.aux_loss_weight}, request dropout {args.aux_dropout}, "
                f"field dropout {args.field_dropout}",
                flush=True,
            )
    inputs_collator: Any
    if args.family in MOLMOACT2_FAMILIES:
        assert molmoact2_prompt is not None
        inputs_collator = MolmoAct2InputsCollator(
            str(checkpoint_dir),
            setup_type=molmoact2_prompt.setup_type,
            control_mode=molmoact2_prompt.control_mode,
            num_state_tokens=molmoact2_prompt.num_state_tokens,
            action_mode=molmoact2_prompt.action_mode,
            narration=molmoact2_prompt.narration,
        )
    elif molmo2_trunk:
        inputs_collator = Molmo2InputsCollator(str(checkpoint_dir), args.max_crops)
    else:
        inputs_collator = GemmaInputsCollator(
            str(checkpoint_dir),
            args.max_soft_tokens,
        )
    state_table: tuple[Tensor, Tensor] | None = None
    if args.family in MOLMOACT2_FAMILIES:
        assert source_metadata is not None
        state_table = molmo_flow_state_table(source_metadata.stats)
    collator = Collator(
        inputs=inputs_collator,
        instruction=args.instruction,
        camera_filter=args.cameras,
        max_cameras=args.max_cameras,
        action_codec=action_codec,
        aux=aux_spec,
        generate_bracket=(
            args.family in AR_SUFFIX_FAMILIES or args.prompt_generate_bracket
        ),
        generate_override=None,
        camera_kind_dropout=args.camera_kind_dropout,
        instruction_augment=args.instruction_augment,
        condition_fields=tuple(
            ConditionField(f) for f in (args.condition_fields or ())
        ),
        condition_dropout=args.condition_dropout,
        subgoal_condition_dropout=args.subgoal_dropout,
        state_dropout=args.state_dropout,
        image_augment=args.image_augment,
        state_q01=state_table[0] if state_table is not None else None,
        state_q99=state_table[1] if state_table is not None else None,
        action_q01=(
            molmoact2_action_table[0] if molmoact2_action_table is not None else None
        ),
        action_q99=(
            molmoact2_action_table[1] if molmoact2_action_table is not None else None
        ),
    )
    if is_main and args.state_dropout > 0:
        print(
            f"state dropout: p={args.state_dropout} — proprioceptive state "
            "masked to the dataset mean per sample (train-time "
            "regularizer; probes score intact state)",
            flush=True,
        )
    if is_main and args.image_augment > 0:
        print(
            f"image augment: p={args.image_augment} — per-frame sim2real "
            "photometric recipe at collation (train-time regularizer; "
            "probes and evals see clean frames)",
            flush=True,
        )
    if is_main and (args.instruction_augment > 0 or args.condition_fields):
        # Deduped by repo id: selection.datasets carries --dataset-repeat
        # replicas of the same objects.
        labeled_episodes = sum(
            len(sub.episode_annotations)
            for sub in {sub.dataset.repo_id: sub for sub in selection.datasets}.values()
        )
        print(
            f"episode annotations: {labeled_episodes} labeled episode(s); "
            f"instruction augment p={args.instruction_augment}, "
            f"conditioning {list(args.condition_fields or [])} "
            f"(dropout {args.condition_dropout}, subgoal "
            f"{args.subgoal_dropout})",
            flush=True,
        )
    if is_main and selection.camera_kinds:
        tagged = sum(len(v) for v in selection.camera_kinds.values())
        print(
            f"camera kinds: {tagged} tagged camera(s) across "
            f"{len(selection.camera_kinds)} dataset(s); untagged render "
            f"'unknown' (kind dropout {args.camera_kind_dropout})",
            flush=True,
        )
    # Probes and eval tables always see the TRUE labels: aux-mode and
    # camera-kind dropout are training-time regularizers, so the
    # probe-side collator runs dropout-0 clones.
    probe_collator = dataclasses.replace(
        collator,
        aux=(
            dataclasses.replace(aux_spec, dropout=0.0, field_dropout=0.0)
            if aux_spec is not None
            else None
        ),
        # Probe prompts run the deployment fast path ([generate|
        # actions]) so the scalar chunk_mae is comparable across
        # aux-on/off arms; the samples table re-collates its rich rows
        # with generate_override = the trained fields (validate).
        generate_override=(() if args.family in AR_SUFFIX_FAMILIES else None),
        camera_kind_dropout=0.0,
        instruction_augment=0.0,
        # dropout-0 conditioning = TRUE-label conditioning for the
        # HINDSIGHT fields (Q1: score against truth ⇒ condition on
        # truth). The SUBGOAL hint is an operator/planner INPUT, not a
        # hindsight label — probes run the deployment-default
        # (planner-less) context, which also lets the samples table
        # exercise subgoal PREDICTION (with it always prompt-fed, the
        # anti-copy coupling correctly suppressed every generated
        # subgoal and the table looked broken — found on the first
        # full-recipe run).
        condition_fields=tuple(
            f for f in collator.condition_fields if f is not ConditionField.SUBGOAL
        ),
        condition_dropout=0.0,
        subgoal_condition_dropout=0.0,
        # Probes score deployment conditions: intact state (the masked
        # readout is the offline --mask-state reliance probe) and
        # clean frames (eval never augments).
        state_dropout=0.0,
        image_augment=0.0,
    )
    # The explicit generator (both modes) makes the shuffle order and the
    # dataloader worker base-seeds a pure function of (--seed, rank) —
    # otherwise they'd draw from the global RNG and entangle batch order
    # with how much randomness model init happened to consume.
    common_loader_kwargs: dict[str, Any] = {
        "generator": torch.Generator().manual_seed(args.seed + rank),
        "num_workers": args.num_workers,
        # Chunked backward wraps the TRAIN collate only (probe batches
        # eval whole); batch composition per step is unchanged — the
        # sampler side never sees the chunking.
        "collate_fn": (
            ChunkingCollator(collator, args.backward_chunks)
            if args.backward_chunks > 1
            else collator
        ),
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": worker_init if args.num_workers > 0 else None,
        # Spawned (not forked) workers: the parent holds live CUDA state and
        # has decoded video in-process (the eval batch), and torchcodec/ffmpeg
        # deadlock or throw "Could not push packet to decoder" in forked
        # children (verified empirically on the H100 box).
        "multiprocessing_context": "spawn" if args.num_workers > 0 else None,
        # Pinned batches make DevicePrefetcher's H2D copies truly async; a
        # deeper prefetch queue absorbs the variance of GOP-boundary decodes.
        "pin_memory": device.type == "cuda",
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
    }
    sampler: torch.utils.data.DistributedSampler[Any] | None = None
    bucket_sampler: LengthBucketedBatchSampler | None = None
    if args.bucket_by_length:
        # Replaces BOTH the plain shuffle and the DistributedSampler:
        # every rank derives the same global batch list from (--seed,
        # epoch) and takes its round-robin slice.
        bucket_sampler = LengthBucketedBatchSampler(
            length_bucket_keys(selection.datasets, collator),
            batch_size=args.batch_size,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
        if is_main:
            census = torch.bincount(bucket_sampler.keys)
            print(
                "length bucketing: frames per camera count "
                f"{ {k: int(n) for k, n in enumerate(census) if n} } "
                f"({len(bucket_sampler)} batches/rank/epoch)",
                flush=True,
            )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=bucket_sampler,
            **common_loader_kwargs,
        )
    else:
        if distributed:
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
            **common_loader_kwargs,
        )

    # -- model -----------------------------------------------------------
    # A live backbone needs fp32 master weights (bf16 updates at backbone
    # learning rates vanish below bf16 resolution); its forwards run
    # under bf16 autocast inside the family's forward. Frozen runs keep
    # the mount convention (checkpoint dtype for Gemma, bf16 for Molmo2)
    # exactly as before the unfreeze flags.
    backbone_dtype = torch.float32 if args.backbone_trained else None
    objective = build_objective(args)
    model: TrainableVLA
    backbone_module: nn.Module
    serving: FlowServing | ARServing
    # The recorded serving operating point: inherited verbatim from the
    # source checkpoint when its kind survives the pathway transition,
    # else the family's derivation rule (written explicitly either way —
    # no silent defaults, D6).
    phi_s_extension = False
    if args.family in MOLMOACT2_FAMILIES:
        assert molmoact2_prompt is not None and source_metadata is not None
        geometry: MolmoFlowDecoderConfig | ARDecoderConfig | None = (
            molmo_flow_section
            if molmo_flow_section is not None
            else molmoact2_ar_section
        )
        assert geometry is not None  # the source block derived one
        geometry_dim = geometry.action_dim
        geometry_horizon = (
            geometry.action_horizon
            if isinstance(geometry, MolmoFlowDecoderConfig)
            else geometry.chunk_size
        )
        if action_dim != geometry_dim:
            raise SystemExit(
                f"data action dim {action_dim} != the checkpoint's "
                f"{geometry_dim} — wrong rig/corpus for this artifact",
            )
        if state_dim != molmoact2_prompt.state_dim:
            raise SystemExit(
                f"data state dim {state_dim} != the checkpoint's "
                f"{molmoact2_prompt.state_dim}",
            )
        if args.chunk_size != geometry_horizon:
            raise SystemExit(
                f"--chunk-size {args.chunk_size} != the checkpoint horizon "
                f"{geometry_horizon} (checkpoint-inferred "
                "under --init-from/--resume — drop any explicit flag)",
            )
        molmoact2_backbone = load_molmo2_model(
            checkpoint_dir,
            device=device,
            dtype=(backbone_dtype if backbone_dtype is not None else torch.bfloat16),
        )
        molmoact2_encoder = MolmoAct2Encoder(
            str(checkpoint_dir),
            setup_type=molmoact2_prompt.setup_type,
            control_mode=molmoact2_prompt.control_mode,
            num_state_tokens=molmoact2_prompt.num_state_tokens,
            action_mode=molmoact2_prompt.action_mode,
            narration=molmoact2_prompt.narration,
        )
        molmoact2_encoder.prompt_schema = molmoact2_prompt.to_dict()
        assert state_table is not None  # built with the collator
        molmoact2_encoder.state_table = (
            tuple(state_table[0].tolist()),
            tuple(state_table[1].tolist()),
        )
        if molmoact2_action_table is not None:
            # The ar-run save side reads the action table off the
            # encoder stash (flow/joint read the decoder's own tables).
            molmoact2_encoder.action_table = (
                tuple(molmoact2_action_table[0].tolist()),
                tuple(molmoact2_action_table[1].tolist()),
            )
        molmo2_text_config = load_molmo2_config(checkpoint_dir).text
        if args.family == "molmoact2_ar":
            assert molmoact2_ar_section is not None
            assert isinstance(objective, ARObjective)
            serving = (
                ARServing.from_dict(source_metadata.serving)
                if source_metadata.serving.get("kind") == "ar"
                else ARServing()
            )
            ar_decoder = build_molmoact2_ar_component_from_section(
                molmoact2_ar_section,
                molmoact2_prompt,
                molmo2_text_config,
                checkpoint_dir,
            )
            model = MolmoAct2ARVLA(
                molmoact2_backbone,
                molmoact2_encoder,
                ar_decoder,
                objective=objective,
                serving=serving,
            )
            schedule_desc = (
                f"molmoact2 discrete head (format-6 suffix; block "
                f"[{molmoact2_ar_section.block_base}, "
                f"{molmoact2_ar_section.block_base + molmoact2_ar_section.vocab_total}"
                ") — zero decoder params, the trunk trains)"
            )
        else:
            assert molmo_flow_section is not None
            serving = (
                FlowServing.from_dict(source_metadata.serving)
                if source_metadata.serving.get("kind") == "flow"
                else FlowServing(
                    num_steps=molmo_flow_section.num_flow_steps,
                    method=SamplingMethod.EULER,
                )
            )
            flow_decoder = build_molmo_flow_decoder(
                molmo_flow_section,
                source_metadata.stats,
                device=device,
                dtype=torch.float32,
            )
            if args.family == "molmoact2_flow":
                assert isinstance(objective, FlowObjective)
                model = MolmoAct2FlowVLA(
                    molmoact2_backbone,
                    molmoact2_encoder,
                    flow_decoder,
                    objective=objective,
                    serving=serving,
                )
            else:
                assert molmoact2_ar_section is not None
                assert isinstance(objective, JointObjective)
                model = MolmoAct2JointVLA(
                    molmoact2_backbone,
                    molmoact2_encoder,
                    flow_decoder,
                    build_molmoact2_ar_component_from_section(
                        molmoact2_ar_section,
                        molmoact2_prompt,
                        molmo2_text_config,
                        checkpoint_dir,
                    ),
                    objective=objective,
                    serving=serving,
                )
            schedule_desc = (
                f"molmo_flow {molmo_flow_section.num_layers}-layer KV "
                f"conditioning (seam "
                f"{'INSULATED (KI)' if args.insulate_flow else 'open'}; "
                f"t-law {molmo_flow_section.time_offset} + "
                f"{molmo_flow_section.time_scale}*Beta("
                f"{molmo_flow_section.beta_alpha}, "
                f"{molmo_flow_section.beta_beta}))"
            )
            if args.family == "molmoact2_joint":
                schedule_desc += (
                    f" + joint CE rider (λ={args.joint_ce_weight:g}, "
                    "flow KV extracted before the CE append)"
                )
        backbone_module = molmoact2_backbone
    elif args.family == "gemma_flow":
        assert isinstance(objective, FlowObjective | SnapflowObjective)
        backbone_config = load_config(checkpoint_dir)
        if source_metadata is not None:
            # The recorded sections are the architecture (the CLI's
            # shape flags were refused above); φ_s may extend the
            # recorded section — the sanctioned zero-init warm start.
            prompt = parse_prompt_config(
                dict(source_metadata.components["prompt"]["config"]),
            )
            if not isinstance(prompt, GemmaPromptConfig):
                raise SystemExit(
                    f"{source} records a {type(prompt).__name__} prompt — "
                    "gemma_flow rides the gemma4 prompt strategy",
                )
            section = parse_decoder_config(
                dict(source_metadata.components["flow_decoder"]["config"]),
            )
            if not isinstance(section, FlowDecoderSection):
                raise SystemExit(
                    f"{source} records a {type(section).__name__} as "
                    "flow_decoder — gemma_flow carries the flow section",
                )
            if section.target_time_embed and not args.target_time_embed:
                raise SystemExit(
                    f"{source} records a φ_s-extended decoder but this run "
                    "resolves target_time_embed=False — dropping trained "
                    "parameters is not a warm start",
                )
            if args.target_time_embed and not section.target_time_embed:
                phi_s_extension = True
                section = dataclasses.replace(section, target_time_embed=True)
                if is_main:
                    print(
                        f"note: φ_s target-time extension over {source} "
                        "(saved decoder has no target-time embedding) — the "
                        "new parameters initialize fresh with "
                        "zero-initialized output (step-0 model ≡ "
                        "checkpoint), sanctioned distill warm start",
                        flush=True,
                    )
            expert_config = expert_config_from_architecture(
                prompt,
                section,
                backbone_config,
            )
            if expert_config.action_dim != action_dim:
                raise SystemExit(
                    f"data action dim {action_dim} != the checkpoint's "
                    f"{expert_config.action_dim} — wrong rig/corpus",
                )
            max_soft_tokens = prompt.max_soft_tokens
        else:
            expert_config = default_expert_config(
                backbone_config,
                action_dim=action_dim,
                state_dim=state_dim,
                stream_counts=args.stream_counts,
                hidden_size=args.decoder_hidden,
                num_attention_heads=args.decoder_heads,
                intermediate_size=args.decoder_intermediate,
                cross_attention_heads=args.decoder_cross_heads,
                chunk_size=args.chunk_size,
                self_attention_mode=SelfAttentionMode(args.self_attention_mode),
                time_conditioning=TimeConditioning(args.time_conditioning),
                target_time_embed=args.target_time_embed,
            )
            max_soft_tokens = args.max_soft_tokens
        gemma_backbone, gemma_encoder, gemma_flow_decoder = build_gemma_flow_parts(
            checkpoint_dir,
            backbone_config,
            expert_config,
            max_soft_tokens=max_soft_tokens,
            device=device,
            dtype=backbone_dtype,
            expert_dtype=torch.float32,
        )
        serving = FlowServing(num_steps=5, method=SamplingMethod.HEUN)
        if source_metadata is not None and source_metadata.serving.get("kind") == (
            "flow"
        ):
            serving = FlowServing.from_dict(source_metadata.serving)
        model = GemmaFlowVLA(
            gemma_backbone,
            gemma_encoder,
            gemma_flow_decoder,
            objective=objective,
            serving=serving,
        )
        backbone_module = gemma_backbone
        schedule_desc = str(expert_config.cross_attention_schedule)
    elif args.family == "molmo2_ar":
        assert args.fast_tokenizer is not None  # parse guard
        assert action_codec is not None
        assert isinstance(objective, ARObjective)
        molmo2_config = load_molmo2_config(checkpoint_dir)
        # Full multimodal model (decoder + untied head + vision tower);
        # live runs get fp32 masters, frozen runs the bf16 mount
        # convention (the release ships fp32 — never mount that raw).
        molmo2_backbone = load_molmo2_model(
            checkpoint_dir,
            device=device,
            dtype=(backbone_dtype if backbone_dtype is not None else torch.bfloat16),
        )
        molmo2_encoder = Molmo2Encoder(
            str(checkpoint_dir),
            max_crops=args.max_crops,
            state_dim=state_dim,
            hidden_size=molmo2_config.text.hidden_size,
            device=device,
            dtype=torch.float32,
        )
        if source_metadata is not None:
            recorded_section = parse_decoder_config(
                dict(source_metadata.components["ar_decoder"]["config"]),
            )
            if not isinstance(recorded_section, ARDecoderConfig):
                raise SystemExit(
                    f"{source} records a {type(recorded_section).__name__} "
                    "as ar_decoder — molmo2_ar carries the ar_backbone "
                    "section",
                )
            ar_backbone_config = ar_section_for_run(
                recorded_section,
                aux=aux_decode_config,
                fast_tokenizer=args.fast_tokenizer,
                is_main=is_main,
            )
        else:
            ar_backbone_config = ARDecoderConfig(
                tokenizer=args.fast_tokenizer,
                vocab_total=action_codec.vocab_total,
                # The SECOND extension block, directly after the 128 image
                # specials — Qwen3's ~271-id unused tail cannot hold the
                # 1,026 FAST ids (the second-extension-block anchoring; the
                # embedding and fresh untied head rows are decoder-owned
                # trainables).
                block_base=molmo2_config.text.fast_block_base,
                chunk_size=args.chunk_size,
                action_dim=action_dim,
                suffix_format=SUFFIX_FORMAT,
                aux=aux_decode_config,
            )
        molmo2_text_tokenizer = Molmo2TextTokenizer(str(checkpoint_dir))
        carriers = newline_carrier_ids(
            molmo2_text_tokenizer,
            text_vocab_size=molmo2_config.text.vocab_size,
            terminator_id=molmo2_text_tokenizer.encode(
                "\n",
                add_special_tokens=False,
            )[0],
        )
        molmo2_aux_runtime = (
            build_aux_runtime(
                aux_decode_config,
                molmo2_text_tokenizer,
                newline_carrier_ban=True,
            )
            if aux_decode_config is not None
            else None
        )
        molmo2_decoder = Molmo2ARDecoder(
            ar_backbone_config,
            molmo2_config.text,
            action_codec,
            tokenizer=molmo2_text_tokenizer,
            aux_runtime=molmo2_aux_runtime,
            aux_loss_weight=args.aux_loss_weight,
            newline_carrier_ids=carriers,
            device=device,
            dtype=torch.float32,
        )
        # Block logits/embeddings start near the frozen tables' row means
        # (full-vocab CE competes against text priors); DDP's
        # construction broadcast makes rank 0's draw authoritative.
        molmo2_decoder.init_tables_from_backbone(molmo2_backbone)
        serving = ARServing()
        model = Molmo2ARVLA(
            molmo2_backbone,
            molmo2_encoder,
            molmo2_decoder,
            objective=objective,
            serving=serving,
        )
        backbone_module = molmo2_backbone
        schedule_desc = (
            f"molmo2 full-depth suffix, FAST extension block @ "
            f"{ar_backbone_config.block_base}"
        )
    elif args.family == "gemma_ar":
        assert args.fast_tokenizer is not None  # parse guard
        assert action_codec is not None
        assert isinstance(objective, ARObjective)
        backbone_config = load_config(checkpoint_dir)
        if source_metadata is not None:
            prompt = parse_prompt_config(
                dict(source_metadata.components["prompt"]["config"]),
            )
            if not isinstance(prompt, GemmaPromptConfig):
                raise SystemExit(
                    f"{source} records a {type(prompt).__name__} prompt — "
                    "gemma_ar rides the gemma4 prompt strategy",
                )
            recorded_section = parse_decoder_config(
                dict(source_metadata.components["ar_decoder"]["config"]),
            )
            if not isinstance(recorded_section, ARDecoderConfig):
                raise SystemExit(
                    f"{source} records a {type(recorded_section).__name__} "
                    "as ar_decoder — gemma_ar carries the ar_backbone "
                    "section",
                )
            ar_backbone_config = ar_section_for_run(
                recorded_section,
                aux=aux_decode_config,
                fast_tokenizer=args.fast_tokenizer,
                is_main=is_main,
            )
            exports = prompt.exports
            max_soft_tokens = prompt.max_soft_tokens
        else:
            # Prefill still stops at the deepest non-KV-shared layer; its
            # stream export rides along unused (the decoder reads the
            # CACHE).
            exports = (backbone_config.text.first_kv_shared_layer_idx - 1,)
            max_soft_tokens = args.max_soft_tokens
            # Tail-anchored block: the last vocab_total ids sit inside
            # the tokenizer's unused tail (E2B: 261118.. ⊂ the 3259-id
            # run at 258885..262143) — no magic constant, adapts to any
            # backbone, recorded in the checkpoint's decoder section.
            ar_backbone_config = ARDecoderConfig(
                tokenizer=args.fast_tokenizer,
                vocab_total=action_codec.vocab_total,
                block_base=backbone_config.text.vocab_size - action_codec.vocab_total,
                chunk_size=args.chunk_size,
                action_dim=action_dim,
                suffix_format=SUFFIX_FORMAT,
                aux=aux_decode_config,
            )
        gemma_ar_backbone, gemma_ar_encoder = build_gemma_encoder(
            checkpoint_dir,
            backbone_config,
            exports=exports,
            max_soft_tokens=max_soft_tokens,
            state_dim=state_dim,
            device=device,
            dtype=backbone_dtype,
            depth=BackboneDepth.FULL,
        )
        text_tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(checkpoint_dir),
        )
        aux_runtime = (
            build_aux_runtime(aux_decode_config, text_tokenizer)
            if aux_decode_config is not None
            else None
        )
        gemma_ar_decoder = GemmaARDecoder(
            ar_backbone_config,
            backbone_config.text,
            action_codec,
            tokenizer=text_tokenizer,
            aux_runtime=aux_runtime,
            aux_loss_weight=args.aux_loss_weight,
            device=device,
            dtype=torch.float32,
        )
        # Block logits start near the average text logit (full-vocab CE
        # competes against text priors); DDP's construction broadcast
        # makes rank 0's draw authoritative.
        gemma_ar_decoder.init_tables_from_backbone(gemma_ar_backbone)
        serving = ARServing()
        model = GemmaARVLA(
            gemma_ar_backbone,
            gemma_ar_encoder,
            gemma_ar_decoder,
            objective=objective,
            serving=serving,
        )
        backbone_module = gemma_ar_backbone
        schedule_desc = (
            f"full-depth suffix, FAST block @ {ar_backbone_config.block_base}"
        )
    else:
        # TrainArgs restricts --family to the kinds above; reaching here
        # is a dispatch bug, not a user error.
        raise AssertionError(f"unhandled family {args.family!r}")
    assert model.spec.family.value == args.family  # construction routed right

    # -- LR flags vs the structural offer (D4, both directions) -----------
    freeze_notes = reconcile_lr_offer(
        model.param_groups(),
        family=args.family,
        backbone_text_lr=args.backbone_text_lr,
        backbone_vision_lr=args.backbone_vision_lr,
    )
    if is_main:
        for note in freeze_notes:
            print(note, flush=True)
    # The parse-time family rule and the built offer must agree (the
    # decoder group is empty iff the family is the parameterless one).
    assert (len(model.param_groups()["decoder"]) == 0) == (
        args.family == "molmoact2_ar"
    )
    backbone_counts = unfreeze_backbone(model, args)
    if args.activation_checkpointing:
        if not isinstance(backbone_module, Molmo2Model):
            raise SystemExit(
                "--activation-checkpointing is wired for the molmo2 decoder stack only",
            )
        backbone_module.text.transformer.gradient_checkpointing = True
        if is_main:
            print(
                "activation checkpointing ON (molmo2 decoder blocks: "
                "recompute in backward, gradient-identical; engages only "
                "where the trunk runs under grad)",
                flush=True,
            )
    state_proj = getattr(model.encoder, "state_proj", None)
    if not args.backbone_trained and state_proj is not None:
        # Frozen runs encode the prefix under no_grad (the family's
        # forward), so the prompt-side state projection CANNOT receive
        # gradients there — freeze it rather than hand DDP a grad-less
        # trainable (static_graph errors on the first backward
        # otherwise). The zero init makes the state token exactly inert:
        # frozen-backbone behavior matches the pre-state-token model.
        # Training it under a frozen backbone (stage 2) needs a
        # grad-transparent prefix — a deliberate future change, not a
        # — a deliberate future change, not a
        # default. (The molmoact2 encoder has NO prompt-side parameters
        # — state is discrete in the ids — so there is nothing to freeze
        # there.)
        state_proj.requires_grad_(False)
        if is_main:
            # Zero-init => exactly inert; a --backbone-init-from load
            # replaces the zeros with the source's TRAINED projection,
            # which then rides frozen (live token, matched to the
            # inherited trunk).
            print(
                "prompt state token FROZEN (no gradient path through a "
                "no-grad prefix encode): inert at zero init, live if "
                "--backbone-init-from supplies a trained projection",
                flush=True,
            )
    if args.distill is not None and is_main:
        print(
            f"distill: {args.distill} objective "
            f"(α={args.snapflow_alpha:g}, λ={args.snapflow_shortcut_weight:g}, "
            "stop-gradient two-step-Euler shortcut targets, no EMA "
            "teacher)",
            flush=True,
        )
    n_trainable = sum(p.numel() for p in model.param_groups()["decoder"])
    if is_main:
        backbone_desc = (
            "frozen backbone"
            if not args.backbone_trained
            else (
                f"LIVE backbone (text {backbone_counts.text / 1e6:.1f}M @ "
                f"lr {args.backbone_text_lr if args.backbone_text_lr is not None else 0:.1e}, "
                f"vision {backbone_counts.vision / 1e6:.1f}M @ lr "
                f"{args.backbone_vision_lr if args.backbone_vision_lr is not None else 0:.1e}; "
                "embeddings/PLE tables frozen; fp32 masters, bf16 autocast)"
            )
        )
        print(
            f"model: --family {args.family} ({backbone_desc}) + "
            f"{n_trainable / 1e6:.1f}M trainable decoder-group params "
            f"(schedule {schedule_desc}; objective "
            f"{objective_to_json(objective)}, serving "
            f"{serving_to_json(serving)})",
            flush=True,
        )
        if distributed:
            print(
                f"ddp: {world_size} ranks, global batch "
                f"{args.batch_size * world_size} ({args.batch_size}/rank), "
                f"{args.num_workers} dataloader workers/rank",
                flush=True,
            )
        if args.backward_chunks > 1:
            print(
                f"chunked backward: {args.backward_chunks} x "
                f"{args.batch_size // args.backward_chunks} per rank per "
                f"step (loader batch {args.batch_size} unchanged; gradient "
                "== unchunked up to fp reduction order"
                + (
                    "; grad sync: explicit in-place allreduce, no DDP wrapper"
                    if args.chunk_grad_allreduce
                    else ""
                )
                + ")",
                flush=True,
            )

    # Fixed-key dicts, deliberately: this is torch's optimizer param-group
    # API format (a third-party boundary), consumed by AdamW below. The
    # model's named groups route to per-component learning rates.
    param_groups, cli_groups, adamc_corrected = build_optimizer_param_groups(
        model,
        optimizer_name=args.optimizer,
        decoder_lr=args.decoder_lr,
        backbone_text_lr=args.backbone_text_lr,
        backbone_vision_lr=args.backbone_vision_lr,
        weight_decay=args.weight_decay,
    )
    adamw_kwargs: dict[str, Any] = {
        "lr": args.decoder_lr,
        "betas": (0.9, 0.95),
        "weight_decay": args.weight_decay,
        # One kernel launch per param group instead of the foreach chain;
        # CUDA only (CPU runs keep the reference path, which also keeps
        # the CPU loss oracle stable).
        "fused": device.type == "cuda",
    }
    optimizer: torch.optim.Optimizer
    if args.zero1:
        # ZeRO-1: each parameter's Adam state lives on exactly one rank
        # (greedy size-balanced partition per group); after each local
        # step the updated shards broadcast so every replica sees the
        # same weights DDP's gradient allreduce already guarantees the
        # same update for. Exact, not approximate — only per-rank
        # optimizer memory changes (~1/world of the moments).
        optimizer = ZeroRedundancyOptimizer(
            param_groups,
            optimizer_class=torch.optim.AdamW,
            **adamw_kwargs,
        )
    else:
        optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)
    # Everything the optimizer updates, for the gradient clip: the frozen
    # path clips exactly the decoder group (unchanged behavior); a live
    # backbone is clipped jointly with it (one global norm).
    clipped_parameters: list[torch.nn.Parameter] = [
        p for group in param_groups for p in group["params"]
    ]
    # The re-warmup ramp needs the resume step BEFORE the scheduler
    # state loads — read it from the checkpoint's metadata.
    resume_step = 0
    if args.resume is not None:
        assert source_metadata is not None
        resume_step = source_metadata.step
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_lambda(step, args, resume_step),
    )
    # AdamC: the corrected groups' indices, read by the pre-step decay
    # update in the loop. None ⇒ adamw, loop takes zero extra work.
    adamc_indices: list[int] | None = None
    if args.optimizer == "adamc":
        adamc_indices = [
            index for index, corrected in enumerate(adamc_corrected) if corrected
        ]
        assert adamc_indices  # the decoder hidden group always exists
        if is_main:
            n_corrected = n_standard = n_undecayed = 0
            for index, group in enumerate(param_groups):
                n_params = sum(p.numel() for p in group["params"])
                if adamc_corrected[index]:
                    n_corrected += n_params
                elif float(group.get("weight_decay", 1.0)) == 0.0:
                    n_undecayed += n_params
                else:
                    n_standard += n_params
            print(
                f"optimizer: AdamC (2506.02285) — corrected decay "
                f"λ·γt/γmax on {n_corrected / 1e6:.1f}M hidden params, "
                f"standard AdamW decay on {n_standard / 1e6:.1f}M "
                f"output-head params, {n_undecayed / 1e6:.1f}M 1-D "
                f"params undecayed; λ={args.weight_decay:g}, γmax per "
                "group = its peak (post-warmup) lr",
                flush=True,
            )

    async_saver: AsyncCheckpointSaver | None = None
    if not args.sync_save:
        # The background shard gather must never share the training NCCL
        # communicator (concurrent collectives on one comm are undefined)
        # or touch the GPU — a dedicated CPU (gloo) group carries it.
        # new_group is itself collective: every rank calls it here.
        save_group = (
            torch.distributed.new_group(backend="gloo")
            if distributed and args.zero1
            else None
        )
        async_saver = AsyncCheckpointSaver(
            group=save_group,
            is_main=is_main,
            world_size=world_size,
            zero1=args.zero1,
        )

    start_step = 0
    adapted_backbone_source: Path | None = None
    if source is not None:
        assert source_metadata is not None
        load_family_weights(
            model,
            source,
            source_metadata,
            args=args,
            device=device,
            phi_s_extension=phi_s_extension,
            is_main=is_main,
        )
        if source_metadata.backbone_trained:
            load_backbone_state(backbone_module, source)
            if not args.backbone_trained:
                # Frozen inherited backbone: every checkpoint this run
                # saves must carry the snapshot too (see save_checkpoint).
                adapted_backbone_source = source / "backbone.safetensors"
            if is_main:
                print(
                    f"loaded TRAINED backbone weights from {source} "
                    f"(bf16 snapshot into "
                    f"{'fp32 masters' if args.backbone_trained else 'bf16'})",
                    flush=True,
                )
            if args.resume is not None and args.backbone_trained and is_main:
                print(
                    "WARNING: live-backbone resume is NOT lossless — "
                    "checkpoints hold a bf16 backbone snapshot, so the "
                    "fp32 masters restart snapped to the bf16 grid and "
                    "sub-bf16-resolution updates accumulated before this "
                    "boundary are discarded (Adam moments are restored; "
                    "masters are never serialized)",
                    flush=True,
                )
    elif args.backbone_init_from is not None:
        # Stage-2: trunk (+ prompt state_proj) inherited, decoder fresh.
        stage2_backbone_init(model, backbone_module, args.backbone_init_from)
        if not args.backbone_trained:
            adapted_backbone_source = args.backbone_init_from / "backbone.safetensors"
        if is_main:
            print(
                f"stage-2 init: ADAPTED backbone + prompt state_proj from "
                f"{args.backbone_init_from} "
                f"({'fp32 masters' if args.backbone_trained else 'frozen bf16'}); "
                "decoder built fresh",
                flush=True,
            )
    if args.resume is not None:
        optimizer_path = args.resume / "optimizer.pt"
        if not optimizer_path.exists():
            raise SystemExit(
                f"{optimizer_path} missing (a converted checkpoint carries "
                "no optimizer state) — use --init-from for a warm start "
                "instead",
            )
        train_state = TrainState.from_payload(
            torch.load(optimizer_path, map_location="cpu", weights_only=True),
        )
        optimizer.load_state_dict(train_state.optimizer)
        rehomed = rehome_fused_step_tensors(
            optimizer,
            fused=bool(adamw_kwargs["fused"]),
        )
        if is_main and rehomed:
            print(
                f"re-homed {rehomed} optimizer step counters to device "
                "(fused AdamW resume from a CPU-tagged payload)",
                flush=True,
            )
        scheduler.load_state_dict(train_state.scheduler)
        start_step = train_state.step
        if start_step >= args.steps:
            raise SystemExit(
                f"checkpoint is at step {start_step}, nothing to do with "
                f"--steps {args.steps} (it counts total steps)",
            )
        if is_main:
            print(
                f"resumed optimizer/scheduler at step {start_step} "
                f"(lr {scheduler.get_last_lr()[0]:.2e})",
                flush=True,
            )
        hyper_notes = resume_hyperparameter_notes(
            optimizer,
            cli_groups,
            adamc_corrected if args.optimizer == "adamc" else None,
        )
        if is_main and hyper_notes:
            print(
                "note: --resume keeps the checkpoint's optimizer "
                "hyperparameters — CLI lr/weight-decay flags are ignored "
                "for every param group (--steps/--warmup-steps/"
                "--rewarmup-steps still shape the schedule):",
                flush=True,
            )
            for line in hyper_notes:
                print(f"  {line}", flush=True)

    # DDP wiring: the FAMILY is the train-step module (its forward IS the
    # objective — D5's entry point), so one wrapper hooks gradients of
    # everything a run trains, for both the frozen and live-backbone
    # regimes (a frozen backbone contributes no trainable parameters —
    # DDP's reducer ignores it). Wrapping AFTER the weight load means
    # DDP's construction-time broadcast (rank 0 -> all) covers the loaded
    # state. ``model`` stays the raw module for eval, clipping and
    # checkpointing.
    train_step: Callable[..., Any] = model
    if distributed and args.chunk_grad_allreduce:
        # No DDP wrapper AT ALL on this path: the constructor would
        # allocate reducer bucket buffers (a full fp32 gradient copy,
        # measured 13.6 GiB on the molmo2 config) even if the reducer
        # never syncs. Replicate the two things DDP provided — the
        # construction-time rank-0 state broadcast here, and the
        # per-step gradient sync via allreduce_gradients in the loop.
        broadcast_module_states(model)
    elif distributed:
        train_step = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            # Backbone/decoder buffers are constant RoPE tables etc.;
            # per-step broadcasts would be pure overhead. The trainable
            # partition guarantees every grad-enabled parameter receives
            # gradients every step, and the graph never changes.
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            # Chunked backward runs n forwards + a no_sync accumulation
            # per step; the graph is still static per iteration, but the
            # static_graph fast path's interplay with no_sync is not a
            # risk worth carrying for a memory-fallback path — plain DDP
            # is the well-trodden grad-accumulation path. Unchunked runs
            # keep the historical flag (and its recorded-graph perf).
            static_graph=args.backward_chunks == 1,
        )

    # The probe's flow operating point: 10 solver steps at the family's
    # RECORDED serving method (heun for gemma_flow, euler for the
    # molmoact2 flow pathways) — eval is a measurement, integration
    # error well below model error.
    flow_probe_method = serving.method if isinstance(serving, FlowServing) else None
    probe_aux_fields = tuple(AuxField(f) for f in (args.aux_fields or ()))

    # Fixed MAE probe sets, independent of the training batch size, fetched
    # once and kept as CPU-resident collated batches per rank (collation
    # in-process is safe: dataloader workers are spawned, not forked; GPU
    # memory per eval is bounded by one batch, so probe size costs host RAM
    # only). eval_chunk_mae probes the held-out episodes (needs
    # --holdout-episodes > 0); train_mae probes the training split. Both
    # draws are what bijou.eval --episodes {holdout,train} --seed
    # <eval-seed> would score. Rank 0 keeps raw items strided across its
    # shard for the rich wandb table.
    eval_probe: ProbeSet[Any] | None = None
    train_probe: ProbeSet[Any] | None = None
    if args.eval_samples is not None:
        if args.holdout_episodes > 0:
            eval_selection = select_datasets(
                args.train_data,
                args.exclude,
                args.chunk_size,
                episode_split=EpisodeSplit.HOLDOUT,
                holdout_fraction=args.holdout_episodes,
                split_seed=args.split_seed,
                allowed_fps=args.fps,
                allowed_camera_counts=args.camera_counts,
                # The probe must see the SAME judged/unjudged partition
                # as training: without the pin, a stamp-mismatched
                # dataset trains as unjudged but probes with full tags —
                # train and instrument silently disagree on the prompt
                # distribution.
                required_prompt_hash=args.aux_prompt_hash,
                # The eval probe needs the SAME per-episode labels the
                # train side loads: Q1 conditions probe frames on their
                # true labels, Q2 slices by outcome, Q3 flips it — all
                # three are silently dead without them (caught in the
                # pre-launch audit of the first conditioned run).
                load_episode_annotations=(
                    args.instruction_augment > 0 or args.condition_fields is not None
                ),
            )
            eval_dataset = eval_selection.concat()
            eval_probe = build_probe_set(
                eval_dataset,
                probe_collator,
                args.eval_samples,
                args.eval_seed,
                rank,
                world_size,
                args.batch_size,
                keep_rich=is_main,
            )
            if is_main:
                print(
                    f"eval probe: {eval_probe.total} of {len(eval_dataset)} "
                    f"held-out-episode frames (seed {args.eval_seed}), "
                    f"sharded over {world_size} rank(s) in batches of "
                    f"{args.batch_size}",
                    flush=True,
                )
            # Only the fetched probe items survive; the second dataset
            # selection (arrow tables, metadata for every dataset) does not
            # need to stay resident for the whole run.
            del eval_selection, eval_dataset
        train_probe = build_probe_set(
            dataset,
            probe_collator,
            args.eval_samples,
            args.eval_seed,
            rank,
            world_size,
            args.batch_size,
            keep_rich=is_main and eval_probe is None,
        )
        if is_main:
            print(
                f"train probe: {train_probe.total} of {len(dataset)} "
                f"training frames (seed {args.eval_seed})",
                flush=True,
            )
    action_names = selection.action_names

    log_file: TextIO | None = None
    if is_main:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        log_file = (args.save_dir / "train_log.jsonl").open("a")

    wandb_run: Any = None
    if args.wandb_project is not None and is_main:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            dir=str(args.save_dir),
            config={
                "train_args": train_args_record(args),
                "family": args.family,
                "objective": objective_to_json(objective),
                "serving": serving_to_json(serving),
                "dataset": {
                    "repo_ids": sorted(per_dataset_stats),
                    "episodes": selection.total_episodes,
                    "frames": len(dataset),
                    "camera_sets": {
                        "/".join(k): v for k, v in selection.camera_census.items()
                    },
                },
                "trainable_params": n_trainable,
                "trainable_backbone_params": dataclasses.asdict(backbone_counts),
                "world_size": world_size,
                "global_batch_size": args.batch_size * world_size,
            },
        )

    step = start_step
    # Loss + per-component (sum, count) windows live on-device between
    # log points: a single .item() sync per log_every steps instead of
    # one per step. (grad_norm is identical on all ranks after DDP's
    # gradient sync — no reduce needed.) Component sums/counts aggregate
    # count-weighted across the window AND across ranks (all-reduced as
    # totals), so sparsely-populated batches weigh by their actual
    # elements instead of diluting a batch-mean — the aux convention,
    # now uniform over every component.
    window: list[Tensor] = []
    window_component_sums: dict[str, list[Tensor]] = {}
    window_component_counts: dict[str, list[Tensor]] = {}
    multi_component = False
    grad_norm = torch.zeros((), device=device)
    prefetcher = DevicePrefetcher(loader, device)
    epoch = 0
    # Lifetime vram peak carried in Python: the CUDA counter is reset
    # per log window (windowed peak field), so the monotone lifetime
    # read existing consumers parse must survive the resets here.
    vram_lifetime_peak_gib = 0.0
    t_last = time.perf_counter()
    while step < args.steps:
        if sampler is not None:
            # Fresh coordinated shuffle each pass over the data.
            sampler.set_epoch(epoch)
        elif bucket_sampler is not None:
            # Same convention: reshuffle + regroup each pass.
            bucket_sampler.set_epoch(epoch)
        for batch in prefetcher:
            if step >= args.steps:
                break
            if isinstance(batch, ChunkedBatch):
                # Chunked backward (--backward-chunks > 1): every chunk
                # forwards against the SAME summed-then-all-reduced
                # counts and accumulates gradients, syncing DDP only on
                # the last chunk — the accumulated gradient and every
                # logged quantity match the unchunked step up to fp
                # reduction order, at one chunk's activation footprint.
                counts = summed_loss_counts(model, batch.chunks)
                if distributed:
                    for key in sorted(counts):
                        torch.distributed.all_reduce(counts[key])
                optimizer.zero_grad(set_to_none=True)
                loss = torch.zeros((), device=device)
                step_sums: dict[str, Tensor] = {}
                step_counts: dict[str, Tensor] = {}
                for i, chunk in enumerate(batch.chunks):
                    if (
                        distributed
                        and not args.chunk_grad_allreduce
                        and i < len(batch.chunks) - 1
                    ):
                        # DDP-reducer path: accumulate under no_sync,
                        # sync on the last chunk. Under
                        # --chunk-grad-allreduce there is NO DDP wrapper
                        # (see construction) — plain autograd
                        # accumulation every chunk, then the explicit
                        # allreduce below is the whole gradient sync.
                        assert isinstance(
                            train_step,
                            torch.nn.parallel.DistributedDataParallel,
                        )
                        sync_ctx = train_step.no_sync()
                    else:
                        sync_ctx = nullcontext()
                    with sync_ctx:
                        report = train_step(chunk, counts=counts)
                        report.objective.backward()
                    assert set(report.components) == set(counts)
                    loss = loss + report.objective.detach()
                    for key, component in report.components.items():
                        component_sum = component.sum.detach()
                        step_sums[key] = (
                            component_sum
                            if key not in step_sums
                            else step_sums[key] + component_sum
                        )
                        step_counts[key] = (
                            component.count
                            if key not in step_counts
                            else step_counts[key] + component.count
                        )
                if distributed and args.chunk_grad_allreduce:
                    allreduce_gradients(clipped_parameters)
            else:
                counts = model.loss_counts(batch)
                if distributed:
                    for key in sorted(counts):
                        torch.distributed.all_reduce(counts[key])
                report = train_step(batch, counts=counts)
                assert set(report.components) == set(counts)
                optimizer.zero_grad(set_to_none=True)
                report.objective.backward()
                loss = report.objective.detach()
                step_sums = {
                    key: component.sum.detach()
                    for key, component in report.components.items()
                }
                step_counts = {
                    key: component.count for key, component in report.components.items()
                }
            grad_norm = torch.nn.utils.clip_grad_norm_(
                clipped_parameters,
                args.grad_clip,
            )
            if adamc_indices is not None:
                apply_adamc_weight_decay(optimizer, adamc_indices, args.weight_decay)
            optimizer.step()
            scheduler.step()
            step += 1
            window.append(loss)
            multi_component = len(step_sums) > 1
            for key, value in step_sums.items():
                window_component_sums.setdefault(key, []).append(value)
                window_component_counts.setdefault(key, []).append(
                    step_counts[key].float(),
                )

            if step % args.log_every == 0:
                # All ranks participate in the reduce (they hit the same
                # step in lockstep); only rank 0 syncs to host and
                # reports. Components ride as (sum, count) totals — the
                # reduced ratio is the element-weighted mean over the
                # window across all ranks. Every step appends every
                # component (key set is run-constant), so the collective
                # stays aligned.
                window_mean = torch.stack(window).mean()
                component_totals = {
                    key: torch.stack(
                        [
                            torch.stack(window_component_sums[key]).sum(),
                            torch.stack(window_component_counts[key]).sum(),
                        ],
                    )
                    for key in sorted(window_component_sums)
                }
                window.clear()
                window_component_sums.clear()
                window_component_counts.clear()
                if distributed:
                    torch.distributed.all_reduce(window_mean)
                    window_mean /= world_size
                    for totals in component_totals.values():
                        torch.distributed.all_reduce(totals)
                dt = (time.perf_counter() - t_last) / args.log_every
                t_last = time.perf_counter()
                if is_main:
                    record = {
                        "step": step,
                        "loss": round(window_mean.item(), 4),
                        "grad_norm": round(grad_norm.item(), 3),
                        "lr": scheduler.get_last_lr()[0],
                        "samples": step * args.batch_size * world_size,
                        "s_per_step": round(dt, 3),
                    }
                    if device.type == "cuda":
                        # True torch-side peaks — nvidia-smi only sees
                        # the reserved pool, which under
                        # expandable_segments never shrinks (the molmo2
                        # smoke-ladder lesson: reserved shadows hid the
                        # live/steady-state gap entirely). The CUDA
                        # counter is reset each window: window peak is
                        # the new direct read (one long batch no longer
                        # ratchets it forever), lifetime peak keeps its
                        # monotone semantics via the Python-side max.
                        window_peak_gib = (
                            torch.cuda.max_memory_allocated(device) / 2**30
                        )
                        vram_lifetime_peak_gib = max(
                            vram_lifetime_peak_gib,
                            window_peak_gib,
                        )
                        torch.cuda.reset_peak_memory_stats(device)
                        record["vram_alloc_peak_gib"] = round(
                            vram_lifetime_peak_gib,
                            2,
                        )
                        record["vram_window_peak_gib"] = round(
                            window_peak_gib,
                            2,
                        )
                        record["vram_reserved_gib"] = round(
                            torch.cuda.memory_reserved(device) / 2**30,
                            2,
                        )
                    if multi_component:
                        # Component chart series keep the historical
                        # names (loss_action, loss_aux); single-
                        # component runs log "loss" alone, exactly as
                        # before.
                        for key, totals in component_totals.items():
                            if float(totals[1]) > 0:
                                record[f"loss_{key}"] = round(
                                    float(totals[0] / totals[1]),
                                    4,
                                )
                    if args.backbone_trained:
                        # Same cosine shape as the decoder group's,
                        # scaled to the backbone's base lr; the index
                        # comes from the group table (adamc's decoder
                        # split shifts it — a hardcoded 1 logged the
                        # decoder head's lr).
                        index = backbone_group_index(cli_groups)
                        assert index is not None  # backbone_trained
                        record["lr_backbone"] = scheduler.get_last_lr()[index]
                    assert log_file is not None
                    print(json.dumps(record), flush=True)
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                    if wandb_run is not None:
                        wandb_metrics = {
                            "train/loss": record["loss"],
                            "train/grad_norm": record["grad_norm"],
                            "train/lr": record["lr"],
                            "train/samples": record["samples"],
                            "train/s_per_step": record["s_per_step"],
                        }
                        for key in component_totals:
                            if f"loss_{key}" in record:
                                wandb_metrics[f"train/loss_{key}"] = record[
                                    f"loss_{key}"
                                ]
                        wandb_run.log(wandb_metrics, step=step)

            if (
                os.environ.get("BIJOU_MEM_SNAPSHOT")
                and device.type == "cuda"
                and step == int(os.environ.get("BIJOU_MEM_SNAPSHOT_STEP", "0"))
            ):
                # Forensics without an OOM: every rank dumps its
                # allocation history at this step — the history covers
                # everything since process start (incl. the step-1
                # transient peak), not just live blocks.
                path = (
                    f"{os.environ['BIJOU_MEM_SNAPSHOT']}_step{step}"
                    f"_rank{os.environ.get('RANK', '0')}.pickle"
                )
                torch.cuda.memory._dump_snapshot(path)
                print(f"mem-snapshot: step {step} — dumped {path}", flush=True)

            if train_probe is not None and step % args.eval_every == 0:
                # Collective: every rank scores its shards, the MAE sums
                # all-reduce inside validate (same call order everywhere).
                # Noise decorrelates across ranks via the +rank offset but
                # is identical across eval points, keeping both series
                # comparable step to step.
                probe_record: dict[str, Any] = {"step": step}
                probe_metrics: dict[str, float] = {}
                if eval_probe is not None:
                    eval_mae = validate(
                        model,
                        eval_probe,
                        device,
                        args.eval_seed + rank,
                        distributed=distributed,
                        wandb_run=wandb_run,
                        collator=probe_collator,
                        action_names=action_names,
                        step=step,
                        table_key="eval/samples",
                        aux_fields=probe_aux_fields,
                        flow_probe_method=flow_probe_method,
                    )
                    probe_record["eval_chunk_mae"] = round(eval_mae, 4)
                    probe_metrics["eval/chunk_mae"] = eval_mae
                train_mae = validate(
                    model,
                    train_probe,
                    device,
                    args.eval_seed + rank,
                    distributed=distributed,
                    wandb_run=wandb_run,
                    collator=probe_collator,
                    action_names=action_names,
                    step=step,
                    table_key="train/samples",
                    aux_fields=probe_aux_fields,
                    flow_probe_method=flow_probe_method,
                )
                probe_record["train_mae"] = round(train_mae, 4)
                probe_metrics["train/mae"] = train_mae
                if is_main:
                    assert log_file is not None
                    print(json.dumps(probe_record), flush=True)
                    log_file.write(json.dumps(probe_record) + "\n")
                    log_file.flush()
                    if wandb_run is not None:
                        wandb_run.log(probe_metrics, step=step)

            save_boundary = step % args.save_every == 0 or step == args.steps
            if save_boundary and async_saver is not None:
                # Async save: main-thread device->CPU capture (seconds),
                # then gather+merge+write in the background while
                # stepping resumes. Under zero1 every rank participates
                # (the shard gather is collective on the side gloo
                # group); otherwise only rank 0 has anything to do.
                # submit() first joins the previous save — a boundary
                # can never overtake the write before it.
                if args.zero1 or is_main:
                    capture_start = time.monotonic()
                    optimizer_capture = capture_optimizer_state(
                        optimizer,
                        zero1=args.zero1,
                        is_main=is_main,
                    )
                    write: Callable[[dict[str, Any]], Path] | None = None
                    if is_main:
                        target_dir = args.save_dir / f"step_{step:06d}"
                        tensors = capture_checkpoint_tensors(
                            model,
                            backbone_module,
                            args=args,
                            adapted_backbone_source=adapted_backbone_source,
                            pristine_trunk_dir=checkpoint_dir,
                        )
                        metadata = build_vla_metadata(
                            model,
                            args=args,
                            normalizers=normalizers,
                            per_dataset_stats=per_dataset_stats,
                            step=step,
                            adapted_backbone_source=adapted_backbone_source,
                        )
                        # Captured now (copy_to_cpu doubles as a deep
                        # copy): a later scheduler.step() must not leak
                        # the next lr into this boundary's file.
                        scheduler_state = copy_to_cpu(scheduler.state_dict())

                        def write_async(
                            optimizer_state: dict[str, Any],
                            *,
                            _dir: Path = target_dir,
                            _tensors: CheckpointTensors = tensors,
                            _metadata: VLAMetadata = metadata,
                            _scheduler: dict[str, Any] = scheduler_state,
                            _step: int = step,
                            _started: float = capture_start,
                        ) -> Path:
                            path = write_checkpoint(
                                _dir,
                                metadata=_metadata,
                                tensors=_tensors,
                                train_state_payload=TrainState(
                                    optimizer=optimizer_state,
                                    scheduler=_scheduler,
                                    step=_step,
                                ).to_payload(),
                            )
                            print(
                                f"saved {path} (async, "
                                f"{time.monotonic() - _started:.1f}s "
                                "behind the boundary)",
                                flush=True,
                            )
                            return path

                        write = write_async
                        print(
                            f"checkpoint step {step}: captured in "
                            f"{time.monotonic() - capture_start:.1f}s; "
                            "gather+write continue in background",
                            flush=True,
                        )
                    async_saver.submit(capture=optimizer_capture, write=write)
            elif save_boundary:
                if args.zero1:
                    # Collective — every rank streams its optimizer shard
                    # to rank 0 BEFORE the is_main-gated save below reads
                    # state_dict() (which raises unconsolidated).
                    # Transient: one remote shard on-device at a time.
                    assert isinstance(optimizer, ZeroRedundancyOptimizer)
                    optimizer.consolidate_state_dict(to=0)
                if is_main:
                    path = save_checkpoint(
                        model,
                        backbone_module,
                        args=args,
                        normalizers=normalizers,
                        per_dataset_stats=per_dataset_stats,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        adapted_backbone_source=adapted_backbone_source,
                        pristine_trunk_dir=checkpoint_dir,
                    )
                    print(f"saved {path}", flush=True)
        epoch += 1

    if async_saver is not None:
        # The final boundary's write must land before the process (and
        # under DDP the group) goes away — anything chained on this run
        # reads the endpoint checkpoint the moment train exits.
        async_saver.join()
    if log_file is not None:
        log_file.close()
    if wandb_run is not None:
        wandb_run.finish()
    if distributed:
        # Let rank 0 finish its final save/eval before the group dissolves.
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


def ar_section_for_run(
    recorded: ARDecoderConfig,
    *,
    aux: AuxDecodeConfig | None,
    fast_tokenizer: str,
    is_main: bool,
) -> ARDecoderConfig:
    """The AR-suffix decoder config an --init-from/--resume run builds:
    the RECORDED section with this run's aux request substituted — aux
    is a data-side format dial (same parameter set), so a difference is
    the sanctioned warm-start pattern (enable aux on an aux-less base),
    noted loudly, never an error. The codec ref must match: the
    collator already tokenized with the run's artifact."""
    if recorded.tokenizer != fast_tokenizer:
        raise SystemExit(
            f"checkpoint records fast tokenizer {recorded.tokenizer!r} but "
            f"this run resolves {fast_tokenizer!r} — the codec is "
            "checkpoint-inferred; drop the flag",
        )
    if recorded.aux != aux and is_main:
        print(
            "note: data-side decoder config differs from the checkpoint "
            f"(aux: {recorded.aux} -> {aux}) — sanctioned warm-start "
            "pattern, proceeding with the run's format",
            flush=True,
        )
    return dataclasses.replace(recorded, aux=aux)


def build_molmoact2_ar_component_from_section(
    section: ARDecoderConfig,
    prompt: MolmoAct2PromptConfig,
    text_config: Molmo2TextConfig,
    trunk_dir: Path,
) -> MolmoAct2ARDecoder:
    """The parameterless discrete decoder from a format-6 section —
    the train-side twin of
    :func:`bijou.models.molmoact2_ar.build_molmoact2_ar_component`
    (which parses a metadata record; this one takes the already-parsed
    section the pathway matrix derived)."""
    return build_molmoact2_ar_decoder(
        section,
        prompt,
        text_config,
        str(trunk_dir),
    )


def load_family_weights(
    model: TrainableVLA,
    source: Path,
    metadata: VLAMetadata,
    *,
    args: TrainArgs,
    device: torch.device,
    phi_s_extension: bool,
    is_main: bool,
) -> None:
    """Load the source checkpoint's component weights into the built
    family (--init-from/--resume). Strict always — checkpoints carrying
    parameters this code deleted are refused by the key mismatch, no
    migration path. One sanctioned exception: the φ_s target-time
    extension may miss EXACTLY the fresh φ_s keys, which keep their
    built init (zero-init output ⇒ step-0 model ≡ checkpoint)."""
    if isinstance(model, MolmoAct2FlowVLA | MolmoAct2JointVLA):
        # The flow decoder's weight source (--flow-decoder-init):
        # inherit = the source checkpoint; fresh = the built adaLN-Zero
        # init (stage-2); <ref> = another checkpoint's decoder under the
        # config-equality guard the source block enforced.
        if args.flow_decoder_init == "fresh":
            if is_main:
                print(
                    "flow-decoder-init: FRESH released-shape decoder "
                    "(adaLN-Zero init) — no flow decoder weights inherited",
                    flush=True,
                )
        else:
            weights_source = (
                Path(args.flow_decoder_init)
                if args.flow_decoder_init != "inherit"
                else source
            )
            if args.flow_decoder_init != "inherit" and is_main:
                print(
                    f"flow-decoder-init: borrowing the flow decoder from "
                    f"{weights_source} (two-source init) — its weights "
                    "live in ITS training table's normalized space; this "
                    "run clamps with the SOURCE checkpoint's table "
                    "(provenance: both recorded in train_args)",
                    flush=True,
                )
            load_expert_state(
                model.flow_decoder,
                load_file(
                    str(weights_source / "flow_decoder.safetensors"),
                    device="cpu",
                ),
            )
            model.flow_decoder.to(device=device, dtype=torch.float32)
            if is_main:
                print(f"loaded flow decoder weights from {weights_source}", flush=True)
        return  # molmoact2 prompt side is parameterless; trunk loads at call site
    if isinstance(model, MolmoAct2ARVLA):
        # The discrete decoder owns ZERO parameters — there is nothing
        # to load (trunk deltas ride backbone.safetensors at the call
        # site; a flow-section source's flow_decoder.safetensors is
        # deliberately left behind: the ar pathway drops the decoder).
        return
    if isinstance(model, GemmaFlowVLA):
        decoder_state = load_file(
            str(source / "flow_decoder.safetensors"),
            device="cpu",
        )
        if phi_s_extension:
            phi_s_keys = {
                key
                for key in model.flow_decoder.state_dict()
                if key.startswith(("target_time_in_proj.", "target_time_out_proj."))
            }
            missing, unexpected = model.flow_decoder.load_state_dict(
                decoder_state,
                strict=False,
            )
            if set(missing) != phi_s_keys or unexpected:
                raise SystemExit(
                    f"flow_decoder.safetensors mismatch at {source} beyond "
                    f"the φ_s extension: missing {sorted(missing)} "
                    f"(expected exactly {sorted(phi_s_keys)}), unexpected "
                    f"{sorted(unexpected)}",
                )
        else:
            model.flow_decoder.load_state_dict(decoder_state, strict=True)
    else:
        model.ar_decoder.load_state_dict(
            load_file(str(source / "ar_decoder.safetensors"), device="cpu"),
            strict=True,
        )
    # Prompt-side parameters (state_proj) — the gemma/molmo2 encoders
    # always carry them; the metadata's prompt component records the
    # fact explicitly.
    if metadata.components["prompt"]["weights"]:
        model.encoder.load_state_dict(
            load_file(str(source / "prompt.safetensors"), device="cpu"),
            strict=True,
        )
    elif len(list(model.encoder.parameters())) > 0:
        raise SystemExit(
            f"{source} records no prompt weights but the encoder has "
            "prompt-side parameters — not a valid checkpoint for this "
            "composition",
        )
    if is_main:
        print(f"loaded decoder + prompt weights from {source}", flush=True)


def stage2_backbone_init(
    model: TrainableVLA,
    backbone: nn.Module,
    checkpoint: Path,
) -> None:
    """Stage-2 warm start: inherit ONLY the backbone and the prompt-side
    parameters (state_proj) from a checkpoint, leaving the decoder as
    built — the point is mounting a DIFFERENT decoder family (a fresh
    flow decoder on an adapted trunk), so there is deliberately no
    decoder-config match check here. Loud when the checkpoint carries no
    trained backbone: inheriting a pristine trunk would silently run the
    stock-backbone arm of an ablation twice."""
    snapshot = checkpoint / "backbone.safetensors"
    if not snapshot.exists():
        raise SystemExit(
            f"--backbone-init-from {checkpoint}: no backbone.safetensors — "
            "that checkpoint's backbone is pristine, so there is nothing "
            "to inherit (for a stock backbone simply omit the flag)",
        )
    load_backbone_state(backbone, checkpoint)
    # Prompt-side parameters (state_proj) travel with the trunk: the
    # backbone was adapted WITH this projection feeding its prompt —
    # loading one without the other would shift the trunk's input
    # distribution.
    prompt_path = checkpoint / "prompt.safetensors"
    if prompt_path.exists():
        model.encoder.load_state_dict(
            load_file(str(prompt_path), device="cpu"),
            strict=True,
        )
    elif len(list(model.encoder.parameters())) > 0:
        raise SystemExit(
            f"--backbone-init-from {checkpoint}: no prompt.safetensors but "
            "this composition's encoder has prompt-side parameters — the "
            "trunk and its state projection travel together",
        )
