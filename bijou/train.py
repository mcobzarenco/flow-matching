"""Train Bijou (flow-matching action expert) on a LeRobot v3 dataset.

The frozen truncated backbone encodes the observation per batch
(no grad); the expert is optimized with the π0/SmolVLA flow-matching recipe:
τ ~ Beta(1.5, 1) (scaled into (0, 1)), x_τ = τ·ε + (1−τ)·actions, MSE against
the velocity target ε − actions, over the full chunk — episode-boundary
chunks carry repeat-last-action targets (hold still after task completion)
rather than masked padding. Actions and state are MEAN_STD-normalized
**per dataset** (each sample uses its own dataset's stats, the π0/SmolVLA
convention): 59–95% of the
aggregate action variance across the community collections is between-dataset
rig offsets that images cannot see, and normalizing them away is what makes
the state→action identity learnable (measured: aggregate normalization left
the trained model behind the state-copy baseline). Checkpoints store the
per-dataset stats table plus a count-weighted aggregate as a fallback for
rigs without stats; inference must unnormalize with the deployment rig's
stats.

The prompt is the instruction sandwich discussed in the design:
``[instruction][cam_1]...[cam_N][instruction]`` inside a user chat turn,
giving instruction-conditioned image KV and image-conditioned instruction KV
under causal attention.

Training data is selected with ``--train-data`` via the shared selection
pipeline in ``bijou.data`` (also used by ``bijou.eval``): collection roots
and dataset dirs, loud drops for incompatible/corrupt datasets, per-dataset
stats attachment, chat-templated prompt collation.

Usage::

    uv run python -m bijou.train \
        --train-data ~/datasets/mcobzarenco/community_dataset_v2_v3 \
        --device cuda --steps 5000

    # Multi-GPU: one full replica + optimizer per GPU (DDP over the expert
    # only; the frozen backbone is never synced). --batch-size and
    # --num-workers are PER RANK; logged loss and eval MAE are all-reduced
    # across ranks (the eval set is sharded); logging/checkpoints happen on
    # rank 0. Without torchrun the script runs exactly as before.
    MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 \
    uv run torchrun --standalone --nproc-per-node=4 -m bijou.train \
        --train-data ~/datasets/mcobzarenco/community_dataset_v2_v3 \
        --device cuda --steps 20000
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, override

import matplotlib
import matplotlib.pyplot as plt
import torch
import transformers
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.distributed.optim import ZeroRedundancyOptimizer

import wandb

from .annotations import ConditionField
from .async_save import (
    AsyncCheckpointSaver,
    capture_optimizer_state,
    copy_to_cpu,
)
from .aux_text import (
    AUX_TEMPLATE_VERSION,
    SUFFIX_FORMAT,
    AuxDecodeConfig,
    AuxField,
    AuxGeneration,
    AuxSpec,
    aux_label_text,
    build_aux_runtime,
)
from .data import (
    DatasetStats,
    EpisodeSplit,
    LengthBucketedBatchSampler,
    StatsAttachedDataset,
    select_datasets,
    worker_init,
)
from .decoders.ar_backbone import (
    ARBackboneConfig,
    ARBackboneDecoder,
    ARSuffixDecoder,
)
from .decoders.ar_fast import ARFastConfig, ARFastDecoder
from .decoders.ar_molmo2 import Molmo2ARDecoder
from .decoders.flow import (
    SNAPFLOW_ALPHA,
    SNAPFLOW_LAMBDA,
    FlowDecoder,
    SelfAttentionMode,
    TimeConditioning,
    flow_matching_loss_sums,
)
from .encoders.gemma4 import PROMPT_FORMAT, GemmaEncoder, GemmaInputsCollator
from .encoders.molmo2 import (
    MOLMO2_PROMPT_FORMAT,
    Molmo2Encoder,
    Molmo2InputsCollator,
)
from .gemma4.config import Gemma4Config
from .gemma4.loading import load_config, resolve_checkpoint_dir
from .gemma4.model import Gemma4Model
from .interface import (
    BatchInputs,
    CollatedBatch,
    Collator,
    ObservationMemory,
    kv_stream_name,
)
from .loading import (
    BackboneConfig,
    BackboneDepth,
    CheckpointMetadata,
    GemmaPromptConfig,
    Molmo2PromptConfig,
    backbone_snapshot,
    build_gemma_encoder,
    decoder_schema_dict,
    default_expert_config,
    from_backbone,
    load_adapted_backbone,
    load_backbone_init,
    molmo2_residual_expert_config,
    prefix_global_layers,
    residual_expert_config,
    resolve_action_codec,
)
from .model import BijouModel
from .molmo2.loading import load_config as load_molmo2_config
from .molmo2.model import Molmo2Model
from .molmo2.model import load_model as load_molmo2_model
from .molmo2.tokenizer import Molmo2TextTokenizer, newline_carrier_ids

DEFAULT_BACKBONE = "google/gemma-4-e2b-it"
# Rows in the wandb probe tables (each costs camera images + a
# matplotlib figure per eval — TWICE for aux runs, the fast-path table
# and the all-fields table): a spot check, deliberately small — 32 rows
# of figures were a measured ~34s/eval rank-0 straggler (2026-08-03).
EVAL_TABLE_ROWS = 12


@dataclass(frozen=True, slots=True)
class TrainArgs:
    train_data: tuple[Path, ...]
    exclude: tuple[str, ...]
    fps: tuple[float, ...] | None
    camera_counts: tuple[int, ...] | None
    holdout_episodes: float
    split_seed: int
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
    # state_proj from a checkpoint; the decoder builds fresh — decoder
    # family/config deliberately unconstrained by the source checkpoint.
    backbone_init_from: Path | None
    # Render [generate|actions] in prompts for non-AR decoders (implied
    # and always-on for ar_backbone): stage-2 trunk consistency.
    prompt_generate_bracket: bool
    instruction: str | None
    cameras: tuple[str, ...] | None
    max_cameras: int | None
    max_soft_tokens: int
    max_crops: int
    stream_counts: tuple[int, ...]
    # Conditioning surface of the flow expert: "kv" = exported K/V of the
    # global prefix layers (the shipped default, scheduled by
    # --stream-counts); "residual" = FULL residual streams — hidden state
    # after every prefix layer through learned decoder-side adapters,
    # expert layer i reading trunk layer i (arch-batch-1 arm B).
    conditioning_streams: str
    # The attach-screen seam flags (pre-reg 2026-08-07, molmo2 flow
    # phase). --seam-stop-grad: detach the raw residual taps before
    # adapter projection — flow-loss gradients into the trunk exactly
    # zero (the π0.5/KI seam). --joint-ce: ride the phase-1 CE objective
    # (Molmo2ARDecoder suffix, aux fields and all) beside the flow loss
    # at fixed weight 1.0 — the K arm; requires a live trunk and the
    # stop-grad seam. --joint-unfrozen-seam: the narrowly-scoped escape
    # (F-then-joint pre-reg 2026-08-09) admitting --joint-ce with the
    # seam OPEN — flow gradients into the trunk — for --init-from warm
    # starts only; random-init naive joint stays refused (the published
    # KI collapse the guard exists for).
    seam_stop_grad: bool
    joint_ce: bool
    joint_unfrozen_seam: bool
    self_attention_mode: str
    time_conditioning: str
    # SnapFlow φ_s target-time embedding on the flow decoder (implied by
    # --distill snapflow; loadable over an unextended checkpoint — the
    # sanctioned additive warm start).
    target_time_embed: bool
    # Training objective variant: None = the decoder's standard
    # objective; "snapflow" = the self-distillation loss mix (flow only,
    # α/λ frozen in bijou.decoders.flow).
    distill: str | None
    decoder: str
    fast_tokenizer: str | None
    aux_fields: tuple[str, ...] | None
    aux_loss_weight: float
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
    # order (sum-form losses over full-batch normalizers).
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
    # Activation checkpointing over the molmo2 decoder blocks (#20):
    # recompute each block in backward instead of retaining its
    # interior activations (measured need 2026-08-06: ~2.4 GiB/sample
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
    # "adamw" (default) or "adamc" — AdamC (arXiv 2506.02285) is AdamW
    # with a time-varying decay coefficient on the hidden ("normalized")
    # layers: λ̂_t = λ·γ_t/γ_max, written into the corrected param
    # groups' weight_decay before every step so the stock (fused) AdamW
    # kernel applies it bit-exactly. Output-head parameters keep
    # standard AdamW decay per the paper's Algorithm 1; 1-D parameters
    # stay undecayed as everywhere else. Defaulted (like sync_save) so
    # checkpoints predating the flag replay their train_args cleanly.
    optimizer: str = "adamw"

    @property
    def backbone_trained(self) -> bool:
        return self.backbone_text_lr is not None or self.backbone_vision_lr is not None


class DevicePrefetcher[I: BatchInputs]:
    """One-batch-lookahead host->device transfer on a side CUDA stream.

    Centralizes every H2D copy of the training pipeline: the loop receives
    batches already device-resident, and the copy of batch N+1 overlaps the
    compute of batch N (dataloader workers cannot produce CUDA tensors —
    each would need its own CUDA context — so this is the closest torch gets
    to "the loader hands you device batches"). Degrades to plain synchronous
    transfers on non-CUDA devices.
    """

    def __init__(
        self,
        loader: Iterable[CollatedBatch[I] | ChunkedBatch[I]],
        device: torch.device,
    ) -> None:
        self.loader = loader
        self.device = device

    def __iter__(self) -> Iterator[CollatedBatch[I] | ChunkedBatch[I]]:
        if self.device.type != "cuda":
            for batch in self.loader:
                yield batch.to(self.device)
            return

        stream = torch.cuda.Stream(self.device)
        compute_stream = torch.cuda.current_stream(self.device)
        batches = iter(self.loader)

        def preload() -> CollatedBatch[I] | ChunkedBatch[I] | None:
            cpu_batch = next(batches, None)
            if cpu_batch is None:
                return None
            with torch.cuda.stream(stream):
                return cpu_batch.to(self.device, non_blocking=True)

        batch = preload()
        while batch is not None:
            compute_stream.wait_stream(stream)
            # The tensors were allocated on the side stream; tell the caching
            # allocator they are consumed on the compute stream.
            for tensor in batch.all_tensors():
                tensor.record_stream(compute_stream)
            yield batch  # consumer enqueues the step's compute, then returns
            batch = preload()  # blocks on workers while the GPU crunches


@dataclass(frozen=True, slots=True)
class ChunkedBatch[I: BatchInputs]:
    """One optimizer step's samples, collated as ``--backward-chunks``
    equal sub-batches for chunked backward. Splitting happens at COLLATE
    time (each chunk pads to its own max length — position ids are
    padding-mask cumsums, so padding width is inert to the math), which
    also shrinks the per-forward activation footprint below a sliced
    full-width batch. Duck-types CollatedBatch's transfer surface for
    the DataLoader pin hook and DevicePrefetcher."""

    chunks: tuple[CollatedBatch[I], ...]

    def all_tensors(self) -> list[Tensor]:
        return [t for chunk in self.chunks for t in chunk.all_tensors()]

    def pin_memory(self) -> ChunkedBatch[I]:
        return ChunkedBatch(tuple(c.pin_memory() for c in self.chunks))

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> ChunkedBatch[I]:
        return ChunkedBatch(
            tuple(c.to(device, non_blocking=non_blocking) for c in self.chunks),
        )


@dataclass
class ChunkingCollator[I: BatchInputs]:
    """Train-loader collate for chunked backward: split the step's item
    list into ``chunks`` contiguous, size-balanced sub-lists (equal by
    construction — --backward-chunks divides --batch-size and the train
    loaders drop last; a short straggler batch would still split
    near-evenly and stay exact, since normalization is by global
    counts) and collate each separately. Sample composition per step is
    identical to the unchunked loader's; the probe collator is never
    wrapped."""

    collator: Collator[I]
    chunks: int

    def __call__(self, samples: list[Any]) -> ChunkedBatch[I]:
        bounds = [(len(samples) * i) // self.chunks for i in range(self.chunks + 1)]
        return ChunkedBatch(
            tuple(
                self.collator(samples[start:stop])
                for start, stop in itertools.pairwise(bounds)
                if stop > start
            ),
        )


def broadcast_module_states(module: torch.nn.Module) -> None:
    """The one-time rank-0 state broadcast DDP's constructor performs,
    without the reducer DDP would also build — its bucket buffers are a
    full fp32 gradient copy allocated AT CONSTRUCTION, not at first
    sync (the measured 13.6 GiB block from the molmo2 smoke-ladder
    snapshot, 2026-08-06). Params and buffers both; state_dict values
    are live views, so in-place broadcast lands in the module."""
    for tensor in module.state_dict().values():
        if isinstance(tensor, torch.Tensor):
            torch.distributed.broadcast(tensor, src=0)


def allreduce_gradients(parameters: list[torch.nn.Parameter]) -> None:
    """The ``--chunk-grad-allreduce`` gradient sync: one explicit
    in-place allreduce of every accumulated ``param.grad`` (sum, then
    divide by world — DDP's averaging semantics, differing only in fp
    reduction order). Runs after the full chunk loop, so no reducer
    bucket buffers ever coexist with the accumulated gradients. Every
    parameter handed in must carry a gradient (the same trainable-
    partition contract DDP's static bucketing relies on); a missing one
    dies loudly rather than letting replicas desynchronize."""
    grads: list[torch.Tensor] = []
    for p in parameters:
        if p.grad is None:
            raise RuntimeError(
                "chunk-grad-allreduce: a trainable parameter has no "
                f"gradient after the chunk loop (shape {tuple(p.shape)}) "
                "— the every-parameter-gets-gradients contract is broken",
            )
        grads.append(p.grad)
    handles = [torch.distributed.all_reduce(g, async_op=True) for g in grads]
    for handle in handles:
        assert handle is not None  # async_op=True always returns a Work
        handle.wait()
    world = float(torch.distributed.get_world_size())
    for g in grads:
        g.div_(world)


class Normalizer:
    """MEAN_STD normalization from LeRobot dataset stats.

    ``mean``/``std``: [dim] (action_dim or state_dim); normalize/unnormalize
    broadcast over any leading axes ([*, dim] -> [*, dim])."""

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean
        self.std = std

    @classmethod
    def from_stats(
        cls,
        stats: dict[str, dict[str, Any]],
        key: str,
        device: torch.device,
    ) -> Normalizer:
        mean = torch.as_tensor(stats[key]["mean"], dtype=torch.float32, device=device)
        std = torch.as_tensor(stats[key]["std"], dtype=torch.float32, device=device)
        return cls(mean, std)

    @classmethod
    def from_aggregated_stats(
        cls,
        stats_list: list[dict[str, dict[str, Any]]],
        key: str,
        device: torch.device,
    ) -> Normalizer:
        """Count-weighted aggregation across datasets: the exact combined
        mean, and std via E[x²] composition (all in float64 before rounding
        to float32)."""
        counts = torch.tensor(
            [float(s[key]["count"][0]) for s in stats_list],
            dtype=torch.float64,
        )
        means = torch.stack(
            [torch.as_tensor(s[key]["mean"], dtype=torch.float64) for s in stats_list],
        )
        stds = torch.stack(
            [torch.as_tensor(s[key]["std"], dtype=torch.float64) for s in stats_list],
        )
        total = counts.sum()
        weights = (counts / total)[:, None]
        mean = (weights * means).sum(dim=0)
        second_moment = (weights * (stds.pow(2) + means.pow(2))).sum(dim=0)
        std = (second_moment - mean.pow(2)).clamp(min=0).sqrt()
        return cls(
            mean.to(dtype=torch.float32, device=device),
            std.to(dtype=torch.float32, device=device),
        )

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / (self.std + 1e-8)

    def unnormalize(self, x: Tensor) -> Tensor:
        return x * (self.std + 1e-8) + self.mean

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


@dataclass(frozen=True, slots=True)
class Normalizers:
    action: Normalizer
    state: Normalizer


@dataclass(frozen=True, slots=True)
class BackboneParameterCounts:
    """Trainable backbone parameters enabled by :func:`unfreeze_backbone`,
    by subsystem (0 = that subsystem stayed frozen)."""

    text: int
    vision: int


def unfreeze_backbone(model: BijouModel, args: TrainArgs) -> BackboneParameterCounts:
    """Flip ``requires_grad`` on the requested backbone subsets; everything
    else stays frozen (``load_model`` freezes the whole backbone).

    Freezing by ``requires_grad`` alone is sufficient for efficiency too:
    token embeddings, PLE tables and (when frozen) the vision tower feed
    the decoder grad-free inputs, so autograd never builds their graphs —
    no activation cost, no backward — without any code-path changes
    inside gemma4.
    """
    groups = model.param_groups()
    text = 0
    vision = 0
    if args.backbone_text_lr is not None:
        for parameter in groups["backbone_text"]:
            parameter.requires_grad_(True)
            text += parameter.numel()
    if args.backbone_vision_lr is not None:
        if not groups["backbone_vision"]:
            raise SystemExit(
                f"--backbone-vision-lr {args.backbone_vision_lr} but the "
                f"backbone ({args.backbone}) has no vision tower — drop the "
                "flag or use a multimodal backbone",
            )
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


def adamc_output_head_parameters(
    model: BijouModel[Any, Any],
) -> list[torch.nn.Parameter]:
    """The trainable output-layer parameters for --optimizer adamc.

    AdamC (2506.02285, Algorithm 1) applies the corrected decay to
    "normalized" layers only — in a transformer, every hidden matrix —
    while the OUTPUT layer keeps standard AdamW decay. The audited
    decoders:

    - ``Molmo2ARDecoder``: ``fast_head`` (fresh untied logit rows; the
      shipped trunk ``lm_head`` and ``wte`` are frozen and never reach
      the optimizer). ``fast_embed`` is an untied input table and stays
      on the corrected side with the other hidden matrices.
    - ``ARBackboneDecoder`` (Gemma trunk): ``fast_embed.weight`` — the
      table doubles as the block-logits head (``hidden @ fast_embed.Tᵀ``),
      a TIED embedding/head pair. One parameter object, one group,
      standard decay; the group-disjointness assert at construction
      keeps any future tied pair from being decayed twice.

    Any other decoder (flow, ar_fast) aborts loudly: its output layer
    has not been audited for the corrected/standard split. Likewise, if
    a trunk's tied ``lm_head``/embedding is ever unfrozen it must be
    added here (today every trunk head is frozen by design)."""
    decoder = model.decoder
    if isinstance(decoder, Molmo2ARDecoder):
        return list(decoder.fast_head.parameters())
    if isinstance(decoder, ARBackboneDecoder):
        return list(decoder.fast_embed.parameters())
    raise SystemExit(
        f"--optimizer adamc: the output-head partition is not audited "
        f"for {type(decoder).__name__} — extend "
        "adamc_output_head_parameters with its logits/output parameters "
        "before training this decoder with adamc",
    )


def build_optimizer_param_groups(
    model: BijouModel[Any, Any],
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
    corrected decay, output head (adamc_output_head_parameters) →
    standard decay, 1-D (norm scales, biases) → no decay. Backbone
    hidden matrices are exactly the paper's "normalized" layers (the
    trunk head/embeddings are frozen out of the groups by design), so
    the existing decayed split takes the corrected flag.

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
        head_ids = {id(p) for p in adamc_output_head_parameters(model) if p.dim() >= 2}
        decayed, undecayed = decay_split(named_groups["decoder"])
        hidden = [p for p in decayed if id(p) not in head_ids]
        heads = [p for p in decayed if id(p) in head_ids]
        if len(heads) != len(head_ids):
            raise SystemExit(
                "--optimizer adamc: an output-head parameter is not in "
                "the decoder param group — the corrected/standard "
                "partition would misroute it (audit "
                "adamc_output_head_parameters vs model.param_groups)",
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
        assert named_groups[group_name]  # unfreeze_backbone validated
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


class BijouTrainStep[I: BatchInputs](torch.nn.Module):
    """One training forward — prefix encode + decoder objective — as a
    single module, so ONE DDP wrapper hooks gradients of everything a run
    trains (frozen-backbone runs simply carry no trainable backbone parameters).

    ``backbone_trained`` selects the prefix-encode regime:
    - False (frozen): no-grad encode at the backbone's native dtype —
      byte-identical math to every frozen run and to the CPU loss oracle
      (the autocast context is constructed disabled: a no-op).
    - True (live): grad-transparent encode under bf16 autocast on CUDA,
      over fp32 master weights (direct bf16 updates vanish below bf16
      resolution at backbone-scale learning rates).

    The decoder runs OUTSIDE the autocast region either way, fp32 with
    TF32 matmuls; it already casts the (possibly autocast-bf16) K/V
    streams to its own dtype.
    """

    def __init__(
        self,
        model: BijouModel[I, Any],
        *,
        backbone_trained: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.backbone_trained = backbone_trained

    @override
    def forward(
        self,
        batch: CollatedBatch[I],
        normalizers: (
            tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor, Tensor | None] | None
        ) = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Batch (shapes in CollatedBatch's docstring) -> (total loss with
        graph, detached action component, detached aux CE sum | None,
        aux position count | None). Single-component objectives return
        (loss, loss.detach(), None, None).

        ``normalizers`` switches to chunked-backward sum form: ``batch``
        is one chunk, the tuple is the FULL step's (action count, aux
        count | None) summed over all chunks, and the return is (this
        chunk's normalized loss share with graph, detached action loss
        SUM, detached aux CE sum | None, aux position count | None) —
        the shares sum over chunks to exactly the unchunked total (up
        to fp reduction order), so backwarding each share accumulates
        the unchunked gradient.

        The joint arm (``model.joint_ce`` set) uses THREE normalizers
        (flow count, CE action count, CE aux count | None) and returns
        the CE branch's action CE in the aux slots (the phase-1
        ``loss_action`` analog — the pre-registered CE-health read);
        the action slot carries the flow component."""
        inputs = batch.encoder_inputs
        # Any batch tensor names the device; the inputs protocol has no
        # per-field surface (trunk-generic).
        device_type = next(iter(inputs.tensors().values())).device.type
        with torch.autocast(
            device_type,
            torch.bfloat16,
            enabled=device_type == "cuda" and self.backbone_trained,
        ):
            memory = self.model.encode(inputs, with_grad=self.backbone_trained)
            if isinstance(self.model.decoder, ARSuffixDecoder):
                # ar_backbone's "decoder" IS the backbone: its suffix
                # forward belongs in the same regime as the prefix —
                # live runs (fp32 masters) run it under bf16 autocast,
                # matching frozen-run numerics and HALVING every
                # loss-side tensor incl. the [B, S, 262k] logits (fp32
                # suffix forwards OOM'd the first full-recipe run at
                # B11); the CE itself upcasts to fp32 inside the loss.
                # Frozen runs construct this context disabled — a no-op,
                # byte-identical to the historical path (oracle-exact).
                if normalizers is None:
                    return self.model.loss_components(memory, batch)
                assert len(normalizers) == 2  # AR runs: (action, aux)
                return self._chunk_share(memory, batch, normalizers)
            # The joint-CE rider is a suffix forward too — same autocast
            # regime as ar_backbone above, for the same [B, S, 153k]
            # logits reason; its sums leave the region as tensors and
            # compose with the fp32 flow loss below.
            joint_ce_sums = (
                self.model.joint_ce_loss_sums(memory, batch)
                if self.model.joint_ce is not None
                else None
            )
        # Cross-attention decoders are fp32-by-design OUTSIDE autocast.
        if joint_ce_sums is not None:
            if normalizers is not None:
                assert len(normalizers) == 3  # joint runs: 3 normalizers
                return self._joint_share(memory, batch, joint_ce_sums, normalizers)
            return self._joint_share(memory, batch, joint_ce_sums, None)
        if normalizers is None:
            return self.model.loss_components(memory, batch)
        assert len(normalizers) == 2  # non-joint runs: (action, aux)
        return self._chunk_share(memory, batch, normalizers)

    def _joint_share(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[I],
        ce_sums: tuple[Tensor, Tensor, Tensor | None, Tensor | None],
        normalizers: tuple[Tensor, Tensor, Tensor | None] | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """The joint arm's composition: fp32 flow sums (outside autocast)
        + the CE branch's sums (computed by the caller inside it), over
        this batch's own counts (unchunked) or the full-step normalizers
        (chunked) — ONE composition for both modes, so chunked and
        unchunked joint numerics agree up to fp reduction order. CE
        weight 1.0 fixed (KI's no-tuning result), aux weight the
        rider's own — the phase-1 objective verbatim."""
        joint_ce = self.model.joint_ce
        assert joint_ce is not None
        decoder = self.model.decoder
        assert isinstance(decoder, FlowDecoder)
        ce_action_sum, ce_action_count, ce_aux_sum, ce_aux_count = ce_sums
        flow_sum, flow_count = flow_matching_loss_sums(decoder, memory, batch)
        if normalizers is None:
            flow_norm, ce_action_norm, ce_aux_norm = (
                flow_count,
                ce_action_count,
                ce_aux_count,
            )
        else:
            flow_norm, ce_action_norm, ce_aux_norm = normalizers
        loss = flow_sum / flow_norm + ce_action_sum / ce_action_norm
        if ce_aux_sum is not None:
            assert ce_aux_norm is not None
            loss = loss + joint_ce.aux_loss_weight * (
                ce_aux_sum / ce_aux_norm.clamp(min=1)
            )
        return (
            loss,
            (
                (flow_sum / flow_norm).detach()
                if normalizers is None
                else flow_sum.detach()
            ),
            ce_action_sum.detach(),
            ce_action_count,
        )

    def _chunk_share(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[I],
        normalizers: tuple[Tensor, Tensor | None],
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        action_norm, aux_norm = normalizers
        action_sum, _, aux_sum, aux_count = self.model.loss_component_sums(
            memory,
            batch,
        )
        loss = action_sum / action_norm
        if aux_sum is not None:
            decoder = self.model.decoder
            assert isinstance(decoder, ARSuffixDecoder)
            assert aux_norm is not None
            loss = loss + decoder.aux_loss_weight * (aux_sum / aux_norm.clamp(min=1))
        return (
            loss,
            action_sum.detach(),
            None if aux_sum is None else aux_sum.detach(),
            aux_count,
        )


def _chunk_plot(
    predicted: Tensor,
    truth: Tensor,
    valid: Tensor,
    state: Tensor,
    action_names: list[str],
) -> Any:
    """Per-joint curves over the action chunk: ground truth, the model's
    prediction, and the trivial state-copy baseline (hold current joint
    positions — the minimum bar a learned policy must clear). Returns a
    matplotlib figure (caller logs and closes it).

    Shapes: predicted/truth [chunk, action_dim]; valid [chunk] (bool);
    state [state_dim] — one sample, CPU-resident."""
    dims = predicted.shape[-1]
    ncols = 3
    nrows = (dims + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.5 * nrows),
        squeeze=False,
    )
    steps = range(predicted.shape[0])
    n_valid = int(valid.sum())
    for dim in range(dims):
        ax = axes[dim // ncols][dim % ncols]
        ax.plot(
            steps[:n_valid],
            truth[:n_valid, dim].tolist(),
            label="truth",
            color="black",
            linewidth=1.8,
        )
        ax.plot(
            steps[:n_valid],
            [float(state[dim])] * n_valid,
            label="state-copy",
            color="tab:blue",
            linestyle="--",
            linewidth=1.2,
        )
        ax.plot(
            steps[:n_valid],
            predicted[:n_valid, dim].tolist(),
            label="predicted",
            color="tab:orange",
            linestyle="--",
            linewidth=1.2,
        )
        name = action_names[dim] if dim < len(action_names) else f"dim {dim}"
        ax.set_title(name, fontsize=9)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    return fig


@dataclass(frozen=True, slots=True)
class RichRow:
    """One probe sample's prediction, kept (CPU-side) for the wandb table.
    ``noise`` is the flow draw the prediction integrated (None for AR) —
    the Q3 tripwire re-decodes these rows and must reuse it."""

    sampled: Tensor
    truth: Tensor
    valid: Tensor
    state: Tensor
    noise: Tensor | None = None


@dataclass(frozen=True, slots=True)
class ProbeSet[I: BatchInputs]:
    """This rank's shard of a seeded MAE probe, CPU-resident between evals.

    ``total`` is the global probe size across ranks. ``rich_items`` are raw
    items kept for wandb rich logging at ``rich_positions`` (positions in
    this shard's streaming order, strided across the shard so the table
    spans the concatenated datasets instead of the earliest ones).
    """

    total: int
    batches: list[CollatedBatch[I]]
    rich_items: list[dict[str, Any]]
    rich_positions: tuple[int, ...]
    # Per shard item (streaming order): the episode's hindsight outcome
    # label (None = unlabeled) — the Q2 slicing key: per-outcome MAE
    # buckets, with the success slice as the open-loop deployment proxy
    # and the unlabeled slice as the continuity anchor.
    outcomes: tuple[str | None, ...]


OUTCOME_BUCKETS = ("success", "partial", "failure", "unlabeled")


def build_probe_set[I: BatchInputs](
    dataset: torch.utils.data.ConcatDataset[dict[str, Any]],
    collator: Collator[I],
    num_samples: int,
    seed: int,
    rank: int,
    world_size: int,
    batch_size: int,
    *,
    keep_rich: bool,
) -> ProbeSet[I]:
    """Draw, fetch and collate one probe set's shard for this rank.

    The frame draw is exactly bijou.eval's: seeded sampling without
    replacement over the same selection scores the same frames. The sorted
    draw is striped round-robin across ranks, so every shard spreads evenly
    over the concatenated datasets.
    """
    num = min(num_samples, len(dataset))
    indices = sorted(random.Random(seed).sample(range(len(dataset)), num))
    shard = indices[rank::world_size]
    items = [dataset[i] for i in shard]
    batches = [
        collator(items[i : i + batch_size]) for i in range(0, len(items), batch_size)
    ]
    rich_positions: tuple[int, ...] = ()
    rich_items: list[dict[str, Any]] = []
    if keep_rich and items:
        stride = max(len(items) // EVAL_TABLE_ROWS, 1)
        rich_positions = tuple(range(0, len(items), stride))[:EVAL_TABLE_ROWS]
        rich_items = [items[p] for p in rich_positions]
    return ProbeSet(
        total=num,
        batches=batches,
        rich_items=rich_items,
        rich_positions=rich_positions,
        outcomes=tuple(item.get("condition_outcome") for item in items),
    )


def holding_accuracy(
    generations: list[AuxGeneration],
    items: list[dict[str, Any]],
) -> float | None:
    """Generated-vs-label holding accuracy over the labeled table rows —
    read straight off the table decode (the request-conditioned format
    elicits every requested field, so a separate likelihood probe is no
    longer needed: the constrained value in the MAIN decode is the
    measurement, in exactly the training context). None when no row is
    labeled."""
    correct = 0
    labeled = 0
    for generation, item in zip(generations, items, strict=True):
        value = item.get("annotation.holding")
        if value is None or not bool(torch.isfinite(value)):
            continue
        if generation.holding is None:
            continue
        correct += int(generation.holding == bool(int(value)))
        labeled += 1
    if labeled == 0:
        return None
    return correct / labeled


@torch.no_grad()
def validate[I: BatchInputs](
    model: BijouModel[I, Any],
    probe: ProbeSet[I],
    device: torch.device,
    seed: int,
    *,
    distributed: bool = False,
    wandb_run: Any = None,
    collator: Collator[I] | None = None,
    action_names: list[str] | None = None,
    step: int = 0,
    table_key: str = "eval/samples",
) -> float:
    """Sampled-chunk MAE in raw action units over this rank's shard of the
    probe set; with ``distributed`` the sums all-reduce to the global value
    (collective — every rank must call this at the same step). Batches
    arrive CPU-resident and visit the device one at a time, and the
    observation memory is re-encoded per eval, so probe size costs host RAM, not GPU memory.
    The valid-element-weighted aggregation is exactly bijou.eval's
    chunk_mae. Normalization is per dataset (each sample's own stats,
    matching training). With a wandb run and a probe carrying rich items,
    also logs a table under ``table_key``: camera images, task, state,
    per-joint predicted-vs-truth plots."""
    totals = torch.zeros(2, device=device)  # [abs-error sum, valid elements]
    rich_rows: list[RichRow] = []
    slice_totals = torch.zeros(len(OUTCOME_BUCKETS), 2, device=device)
    wanted = iter(probe.rich_positions)
    next_rich = next(wanted, None)
    base = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for cpu_batch in probe.batches:
        batch = cpu_batch.to(device)
        # Decoder-agnostic: flow integrates Heun-10 (eval is a measurement —
        # integration error well below model error; 0.018 vs 0.05 mean
        # deviation at the Heun-5 deployment default), AR decodes greedily
        # and ignores the solver knobs. Raw units either way.
        # ar_backbone scores the ACT fast path here (comparable across
        # aux-on / aux-off arms); the rich table below is the FREE-mode
        # surface for aux-capable checkpoints.
        prediction = model.predict_chunk(
            batch,
            generator=generator,
            num_steps=10,
            generate=(),
        )
        sampled = prediction.actions
        truth = batch.actions.float()
        valid = ~batch.action_is_pad
        error = (sampled - truth).abs()
        totals[0] += error[valid].sum()
        totals[1] += valid.sum() * error.shape[-1]
        # Q2 slices: per-frame error bucketed by the episode's hindsight
        # outcome (probes condition on TRUE labels, so the success slice
        # is the open-loop deployment proxy and the unlabeled slice the
        # continuity anchor). Fixed [4, 2] tensor — collective-aligned.
        frame_error = (error * valid[..., None]).sum(dim=(1, 2))
        frame_count = valid.sum(dim=1) * error.shape[-1]
        for i in range(sampled.shape[0]):
            outcome = probe.outcomes[base + i]
            bucket = (
                OUTCOME_BUCKETS.index(outcome)
                if outcome in OUTCOME_BUCKETS
                else OUTCOME_BUCKETS.index("unlabeled")
            )
            slice_totals[bucket, 0] += frame_error[i]
            slice_totals[bucket, 1] += frame_count[i]
        # Batches stream in shard order: pick off the rich positions
        # (matching probe.rich_items one-to-one) as they pass.
        while next_rich is not None and next_rich < base + sampled.shape[0]:
            i = next_rich - base
            rich_rows.append(
                RichRow(
                    sampled=sampled[i].cpu(),
                    truth=truth[i].cpu(),
                    valid=valid[i].cpu(),
                    state=batch.state[i].cpu(),
                    noise=(
                        prediction.noise[i].cpu()
                        if prediction.noise is not None
                        else None
                    ),
                ),
            )
            next_rich = next(wanted, None)
        base += sampled.shape[0]
    if distributed:
        torch.distributed.all_reduce(totals)
        torch.distributed.all_reduce(slice_totals)
    mae = float(totals[0] / totals[1].clamp(min=1))
    metric_prefix = table_key.split("/")[0]
    if wandb_run is not None:
        # Labeled buckets only (all-unlabeled probes log nothing extra);
        # skip the redundant all-in-one-bucket case.
        labeled = {
            bucket: float(slice_totals[i, 0] / slice_totals[i, 1])
            for i, bucket in enumerate(OUTCOME_BUCKETS)
            if float(slice_totals[i, 1]) > 0
        }
        if len(labeled) > 1:
            wandb_run.log(
                {
                    f"{metric_prefix}/chunk_mae_{bucket}": value
                    for bucket, value in labeled.items()
                },
                step=step,
            )

    if wandb_run is not None and probe.rich_items and collator is not None:
        # Two tables over the same rich rows (rank-0-only, no
        # collectives, bounded to EVAL_TABLE_ROWS):
        #   {table_key} — chunk columns straight off the scalar pass
        #     (fast path: prompt says [generate|actions], the suffix
        #     carries NO aux text — actions condition on the user
        #     message only; chunk_mae matches the logged scalar's
        #     measurement condition). The aux_generated/aux_label
        #     columns are a SIDE-CHANNEL from the all-fields decode of
        #     the same items — what the model says for this observation
        #     next to the fast-path chunk, deliberately mixed
        #     conditions, labeled here so nobody rediscovers it as a
        #     bug (owner-requested pairing, 2026-08-03).
        #   {table_key}_all_fields — the all-fields decode's OWN rows:
        #     generations vs labels plus the chunk that followed the
        #     model's self-generated context (never compared to the
        #     scalar).
        decoder = model.decoder
        generations: list[AuxGeneration] | None = None
        rich_actions: Tensor | None = None
        aux_fields: tuple[AuxField, ...] = ()
        if isinstance(decoder, ARSuffixDecoder) and decoder.config.aux is not None:
            aux_fields = decoder.config.aux.fields
            table_collator = dataclasses.replace(
                collator,
                generate_override=aux_fields,
            )
            rich_batch = table_collator(probe.rich_items).to(device)
            rich_memory = model.encode(rich_batch.encoder_inputs, with_grad=False)
            rich_prediction = decoder.predict_chunk(
                model.backbone,
                rich_memory,
                rich_batch,
                generate=aux_fields,
            )
            generations = rich_prediction.generations
            assert generations is not None  # ar_backbone always generates
            rich_actions = rich_prediction.actions.cpu()

        # Cameras vary per sample across mixed datasets: generic positional
        # columns, padded with None where a sample has fewer cameras.
        per_item_cameras = [collator.cameras_of(item) for item in probe.rich_items]
        n_slots = max(len(cams) for cams in per_item_cameras)

        def row_images(item: dict[str, Any], cams: list[str]) -> list[Any]:
            images: list[Any] = [
                wandb.Image(
                    (item[camera].clamp(0, 1) * 255)
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .numpy(),
                    caption=camera.removeprefix("observation.images."),
                )
                for camera in cams
            ]
            return images + [None] * (n_slots - len(cams))

        def state_str(item: dict[str, Any]) -> str:
            return ", ".join(f"{x:.1f}" for x in item["observation.state"].tolist())

        table = wandb.Table(
            columns=[
                "sample",
                *(f"camera_{i}" for i in range(n_slots)),
                "task",
                "state",
                "chunk_mae",
                "pred_vs_truth",
                *(["aux_generated", "aux_label"] if generations is not None else []),
            ],
        )
        for i, (item, row) in enumerate(zip(probe.rich_items, rich_rows, strict=True)):
            figure = _chunk_plot(
                row.sampled,
                row.truth,
                row.valid,
                row.state,
                action_names or [],
            )
            aux_columns: tuple[str, ...] = ()
            if generations is not None:
                aux_columns = (
                    generations[i].text,
                    aux_label_text(item, aux_fields),
                )
            table.add_data(
                i,
                *row_images(item, per_item_cameras[i]),
                str(item["task"]),
                state_str(item),
                float((row.sampled - row.truth).abs()[row.valid].mean()),
                wandb.Image(figure),
                *aux_columns,
            )
            plt.close(figure)
        wandb_run.log({table_key: table}, step=step)

        if generations is not None and rich_actions is not None:
            # Paired signal, free (the decode already ran for the aux
            # columns): masked MAE of the chunks that followed the
            # model's SELF-GENERATED field lines, same 12 rows as the
            # fast-path table — does narration help or hurt the
            # actions? Small-n, directional only; never compared to
            # the full-probe scalar. (The dedicated all-fields table
            # was dropped 2026-08-03 — visually a subset of the main
            # table once it regained the aux columns.)
            all_fields_mae = [
                float((rich_actions[i] - row.truth).abs()[row.valid].mean())
                for i, row in enumerate(rich_rows)
            ]
            wandb_run.log(
                {
                    f"{table_key}_all_fields_mae": sum(all_fields_mae)
                    / len(all_fields_mae),
                },
                step=step,
            )

            if AuxField.HOLDING in aux_fields:
                accuracy = holding_accuracy(generations, probe.rich_items)
                if accuracy is not None:
                    wandb_run.log(
                        {f"{table_key}_holding_acc": accuracy},
                        step=step,
                    )

        # Q3 — conditioning sensitivity, THE tripwire for silent
        # conditioning collapse: on labeled non-success rich rows, decode
        # once more with outcome overridden to "success" and log the mean
        # |Δ| against the true-conditioned scalar-pass predictions.
        # Pre-registered: > 0 and growing; ≈ 0 means the model ignores
        # the label and the failed-demo mass trained as-if-good. The
        # override decode reuses each row's scalar-pass noise: with a
        # fresh draw, a flow decoder's |Δ| has a floor at the sampling
        # variance even when the model is conditioning-blind — the exact
        # state this alarm exists to catch (deep-dive finding 3).
        if ConditionField.OUTCOME in collator.condition_fields:
            flipped = [
                (i, item)
                for i, item in enumerate(probe.rich_items)
                if item.get("condition_outcome") not in (None, "success")
            ]
            if flipped:
                override_items = [
                    {**item, "condition_outcome": "success"} for _, item in flipped
                ]
                override_batch = collator(override_items).to(device)
                flipped_noise = [
                    row
                    for row in (rich_rows[i].noise for i, _ in flipped)
                    if row is not None
                ]
                override_prediction = model.predict_chunk(
                    override_batch,
                    generator=generator,
                    noise=(
                        torch.stack(flipped_noise).to(device)
                        if len(flipped_noise) == len(flipped)
                        else None
                    ),
                    num_steps=10,
                    generate=(),
                )
                deltas = [
                    float(
                        (override_prediction.actions[j].cpu() - rich_rows[i].sampled)
                        .abs()[rich_rows[i].valid]
                        .mean(),
                    )
                    for j, (i, _) in enumerate(flipped)
                ]
                wandb_run.log(
                    {
                        f"{metric_prefix}/condition_sensitivity": sum(deltas)
                        / len(deltas),
                    },
                    step=step,
                )
    return mae


@dataclass(frozen=True, slots=True)
class TrainState:
    """optimizer.pt payload — everything --resume needs beyond the weights.
    Stored as a plain dict on disk (torch.load(weights_only=True) rejects
    custom classes), so the payload methods are the (de)serialization edge.
    """

    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    step: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "step": self.step,
        }

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> TrainState:
        return cls(
            optimizer=data["optimizer"],
            scheduler=data["scheduler"],
            step=int(data["step"]),
        )


def aggregate_stats(normalizers: Normalizers) -> DatasetStats:
    """The count-weighted aggregate stats as a DatasetStats (the checkpoint
    fallback entry for rigs without their own stats)."""
    return DatasetStats(
        action_mean=tuple(normalizers.action.mean.tolist()),
        action_std=tuple(normalizers.action.std.tolist()),
        state_mean=tuple(normalizers.state.mean.tolist()),
        state_std=tuple(normalizers.state.std.tolist()),
        # Quantiles do not compose across datasets (a mean of per-dataset
        # quantiles regresses extremes — the exact bug the corpus backfill
        # fixed), so the aggregate fallback honestly carries none.
        action_q01=None,
        action_q99=None,
        state_q01=None,
        state_q99=None,
    )


def link_or_copy(source: Path, destination: Path) -> None:
    """Hardlink ``source`` at ``destination``, falling back to a plain copy
    across filesystems. Used for inherited (frozen) backbone snapshots:
    byte-identical content, so ten checkpoints of a frozen-backbone run cost
    one file's disk."""
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


@dataclass(frozen=True, slots=True)
class CheckpointTensors:
    """The model-side CPU snapshot of one checkpoint — everything the
    writer needs that came off the device. Captured on the main thread at
    the save boundary (the copies are the boundary's values); consumed by
    ``write_checkpoint`` on either the main thread (sync path) or the
    async saver's background thread."""

    expert: dict[str, Tensor]
    prompt: dict[str, Tensor]
    joint_ce: dict[str, Tensor] | None
    # Trained backbones: the bf16 snapshot dict. Inherited-frozen ones:
    # the source file to link/copy instead (see save_checkpoint's
    # invariant note).
    backbone: dict[str, Tensor] | None
    backbone_source: Path | None


def capture_checkpoint_tensors(
    model: BijouModel,
    *,
    args: TrainArgs,
    adapted_backbone_source: Path | None,
) -> CheckpointTensors:
    """Device->CPU copies of every tensor the checkpoint serializes.
    ``copy=True`` even for CPU runs: the snapshot must not alias live
    parameters the next optimizer step mutates."""

    def snapshot(module: torch.nn.Module) -> dict[str, Tensor]:
        return {
            name: tensor.detach().to("cpu", copy=True).contiguous()
            for name, tensor in module.state_dict().items()
        }

    return CheckpointTensors(
        expert=snapshot(model.decoder),
        prompt=snapshot(model.encoder),
        joint_ce=snapshot(model.joint_ce) if model.joint_ce is not None else None,
        # backbone_snapshot already lands host-side (the device-side cast
        # would transiently cost ~4.3 GB VRAM — see its docstring).
        backbone=backbone_snapshot(model) if args.backbone_trained else None,
        backbone_source=(
            adapted_backbone_source if not args.backbone_trained else None
        ),
    )


def write_checkpoint(
    checkpoint_dir: Path,
    *,
    tensors: CheckpointTensors,
    metadata_json: str,
    train_state_payload: dict[str, Any],
) -> Path:
    """Serialize one checkpoint ATOMICALLY: everything lands in a
    sibling ``.tmp`` directory first and a single ``os.rename`` publishes
    it — a crash mid-write leaves every earlier checkpoint intact and no
    half-written ``step_*`` directory a resume/eval could mistake for a
    real one (the ``.tmp`` debris is evidence, clobbered by the next
    attempt). Pure CPU+disk — safe on the async saver's background
    thread."""
    staging_dir = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    save_file(tensors.expert, str(staging_dir / "expert.safetensors"))
    # Prompt-side parameters (state_proj) — always present at prompt
    # format 3.
    save_file(tensors.prompt, str(staging_dir / "prompt.safetensors"))
    if tensors.joint_ce is not None:
        # The K arm's CE rider (its continued FAST tables) — the
        # trunk-drift read's AR view loads from here.
        save_file(tensors.joint_ce, str(staging_dir / "joint_ce.safetensors"))
    if tensors.backbone is not None:
        # Adapted backbones ride along; from_checkpoint/--init-from detect the
        # file by presence. Frozen-pristine runs write exactly the
        # historical layout.
        save_file(tensors.backbone, str(staging_dir / "backbone.safetensors"))
    elif tensors.backbone_source is not None:
        link_or_copy(
            tensors.backbone_source,
            staging_dir / "backbone.safetensors",
        )
    # Adam moments etc. (~2x expert params); --init-from ignores this
    # file. NB --resume is a lossless continuation only in the
    # frozen-backbone regime: a live backbone's fp32 masters round-trip
    # through the bf16 snapshot above, discarding sub-bf16-resolution
    # updates at every resume boundary (loud warning at resume load).
    torch.save(train_state_payload, staging_dir / "optimizer.pt")
    (staging_dir / "bijou_config.json").write_text(metadata_json)
    if checkpoint_dir.exists():
        # Re-saving the same step (same-boundary resume): replace the old
        # directory wholesale rather than overwriting into it.
        shutil.rmtree(checkpoint_dir)
    staging_dir.rename(checkpoint_dir)
    return checkpoint_dir


def save_checkpoint(
    model: BijouModel,
    *,
    args: TrainArgs,
    normalizers: Normalizers,
    per_dataset_stats: dict[str, DatasetStats],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    adapted_backbone_source: Path | None,
) -> Path:
    """Write one self-contained checkpoint directory (the synchronous
    entry: capture + write inline; the async path drives the same capture
    and writer from bijou.async_save).

    Invariant: ``backbone.safetensors`` is present iff the model's backbone
    differs from pristine ``HF(args.backbone)`` — either because this run
    trains it (snapshot the live fp32 masters) or because it was INHERITED
    from an adapted checkpoint via --init-from/--resume with the unfreeze
    flags off (``adapted_backbone_source``; the backbone is then frozen and
    byte-identical to that file, so it is linked/copied rather than
    re-serialized). Conditioning only on ``args.backbone_trained`` paired a
    decoder fine-tuned against adapted features with the pristine backbone on
    load — silently (found 2026-07-31, ft-rig arm F)."""
    train_state = TrainState(
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        step=step,
    )
    return write_checkpoint(
        args.save_dir / f"step_{step:06d}",
        tensors=capture_checkpoint_tensors(
            model,
            args=args,
            adapted_backbone_source=adapted_backbone_source,
        ),
        metadata_json=build_checkpoint_metadata(
            model,
            args=args,
            normalizers=normalizers,
            per_dataset_stats=per_dataset_stats,
            step=step,
        ),
        train_state_payload=train_state.to_payload(),
    )


def build_checkpoint_metadata(
    model: BijouModel,
    *,
    args: TrainArgs,
    normalizers: Normalizers,
    per_dataset_stats: dict[str, DatasetStats],
    step: int,
) -> str:
    """The ``bijou_config.json`` text. Cheap and pure — runs at capture
    time so the async writer holds no model references."""
    encoder = model.encoder
    prompt_config: GemmaPromptConfig | Molmo2PromptConfig
    if isinstance(encoder, GemmaEncoder):
        prompt_config = GemmaPromptConfig(
            exports=encoder.exports,
            residual_exports=encoder.residual_exports,
            max_soft_tokens=args.max_soft_tokens,
            format=PROMPT_FORMAT,
            state_dim=encoder.state_dim,
            condition_fields=tuple(args.condition_fields or ()),
            generate_bracket=(
                args.decoder == "ar_backbone" or args.prompt_generate_bracket
            ),
        )
        # Structural fact of the built model: a truncated backbone has
        # its KV-shared region cut away (truncated_config), a full one
        # keeps it — no plumbing to drift.
        backbone_config = model.backbone.config
        assert isinstance(backbone_config, Gemma4Config)
        depth = (
            BackboneDepth.FULL
            if backbone_config.text.num_kv_shared_layers > 0
            else BackboneDepth.PREFIX
        )
    elif isinstance(encoder, Molmo2Encoder):
        prompt_config = Molmo2PromptConfig(
            max_crops=encoder.max_crops,
            format=MOLMO2_PROMPT_FORMAT,
            state_dim=encoder.state_dim,
            condition_fields=tuple(args.condition_fields or ()),
            generate_bracket=(
                args.decoder == "ar_backbone" or args.prompt_generate_bracket
            ),
            residual_exports=encoder.residual_exports,
        )
        # The Molmo2 AR trunk is always mounted full-depth (36 layers —
        # the suffix reads the shipped head).
        depth = BackboneDepth.FULL
    else:
        raise TypeError(
            f"save_checkpoint has no prompt-config writer for {type(encoder).__name__}",
        )
    metadata = CheckpointMetadata(
        backbone=BackboneConfig(
            id=args.backbone,
            depth=depth,
        ),
        prompt=prompt_config,
        decoder=decoder_schema_dict(model.decoder),
        joint_ce=(
            decoder_schema_dict(model.joint_ce) if model.joint_ce is not None else None
        ),
        normalization=aggregate_stats(normalizers),
        per_dataset_normalization=per_dataset_stats,
        train_args={
            k: str(v) if isinstance(v, Path) else v
            for k, v in dataclasses.asdict(args).items()
        },
        step=step,
    )
    return json.dumps(metadata.to_json_dict(), indent=2, default=str)


def ensure_matching_decoder_config(
    decoder: FlowDecoder | ARFastDecoder | ARBackboneDecoder | Molmo2ARDecoder,
    checkpoint: Path,
) -> dict[str, Any]:
    """Loud, early failure when a checkpoint's decoder differs from the
    CLI's (strict state-dict loading would also fail, but with worse
    diagnostics — and silently NOT fail for same-shape config differences
    like the cross-attention schedule). Handles both checkpoint formats:
    format 2 compares decoder schema dicts; format 1 predates AR decoders
    and compares the historical serialized expert_config. Returns the
    checkpoint's saved decoder config dict (the weight loader keys its
    format-migration tolerance off it)."""
    meta = json.loads((checkpoint / "bijou_config.json").read_text())
    if "decoder" in meta:
        saved = meta["decoder"]
        current = decoder_schema_dict(decoder)
        if isinstance(decoder, FlowDecoder):
            # Back-compat: checkpoints predating the φ_s field carry no
            # key; absent means unextended.
            saved.setdefault("target_time_embed", False)
    elif isinstance(decoder, FlowDecoder):
        saved = meta["expert_config"]
        # Back-compat: fields added to ExpertConfig after a checkpoint was
        # written are absent from its serialized config; fill their defaults
        # so an unchanged run still matches. A pre-adaRMS checkpoint is
        # additive.
        saved.setdefault("time_conditioning", TimeConditioning.ADDITIVE.value)
        saved.setdefault("target_time_embed", False)
        # Pre-residual-conditioning checkpoints are K/V-conditioned.
        saved.setdefault("residual_streams", False)
        saved.setdefault("residual_stream_dim", None)
        saved.setdefault("cross_attention_kv_heads", None)
        current = json.loads(
            json.dumps(dataclasses.asdict(decoder.config), default=str),
        )
    else:
        raise SystemExit(
            f"{checkpoint} is a format-1 checkpoint (flow-only era); it "
            "cannot initialize a non-flow decoder",
        )
    if current != saved:
        # Aux is a data-side format dial: a difference confined to it
        # is the sanctioned warm-start pattern (enable aux on an
        # aux-less format-5 base — same parameter set) — note, not an
        # error. suffix_format differences are hard errors: pre-5
        # parameter sets no longer exist.
        # φ_s target-time extension (SnapFlow distill warm start): a CLI
        # True over a saved False is sanctioned — the embedding is purely
        # additive and zero-initialized, so step 0 IS the checkpoint; the
        # weight loader tolerates exactly the fresh φ_s keys (keyed off
        # the returned saved config). The reverse direction would DROP
        # trained parameters and stays a hard error.
        extension = ("target_time_embed",)
        current_ext = {k: v for k, v in current.items() if k not in extension}
        saved_ext = {k: v for k, v in saved.items() if k not in extension}
        if (
            current_ext == saved_ext
            and current.get("target_time_embed")
            and not saved.get("target_time_embed")
        ):
            print(
                f"note: φ_s target-time extension over {checkpoint} "
                "(saved decoder has no target-time embedding) — the new "
                "parameters initialize fresh with zero-initialized output "
                "(step-0 model ≡ checkpoint), sanctioned distill warm "
                "start, proceeding",
                flush=True,
            )
            return saved
        data_side = ("aux",)
        current_core = {k: v for k, v in current.items() if k not in data_side}
        saved_core = {k: v for k, v in saved.items() if k not in data_side}
        if current_core == saved_core:
            differing = [key for key in data_side if current.get(key) != saved.get(key)]
            print(
                f"note: data-side decoder config differs from {checkpoint} "
                f"({', '.join(f'{k}: {saved.get(k)} -> {current.get(k)}' for k in differing)}) "
                "— sanctioned warm-start pattern, proceeding with the "
                "CLI's format",
                flush=True,
            )
            return saved
        raise SystemExit(
            f"decoder config mismatch vs {checkpoint}:\n"
            f"  checkpoint: {json.dumps(saved, sort_keys=True)}\n"
            f"  cli:        {json.dumps(current, sort_keys=True)}",
        )
    return saved


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
    config_path = resume / "bijou_config.json"
    try:
        recorded = json.loads(config_path.read_text())
    except FileNotFoundError:
        raise SystemExit(
            f"{config_path} missing — not a checkpoint directory",
        ) from None
    checkpoint_seed = recorded.get("train_args", {}).get("seed")
    if checkpoint_seed is None:
        return (
            "WARNING: resume seed check skipped — checkpoint predates "
            "train_args seed recording; the fresh-seed-on-resume "
            "convention cannot be verified, make sure --seed differs "
            "from the original run's"
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
    ``optimizer.step()`` ('state_steps is on cpu…', measured live at
    the molmo2 60k resume, 2026-08-08 10:15Z). Move each param's step
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


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(
        prog="python -m bijou.train",
        description="Train the Bijou action expert on LeRobot v3 datasets "
        "(dataset directories and/or collection roots). Runs on a single "
        "GPU by default and data-parallel under torchrun; checkpoints "
        "carry everything bijou.eval and bijou.rollout need.",
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
        default=DEFAULT_BACKBONE,
        help="backbone HF model id or local checkpoint path",
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
        default=140,
        help="vision soft-token budget per camera in the prompt "
        "(Gemma trunks; molmo2 trunks use --max-crops)",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=1,
        help="molmo2 trunks: crops per camera image (1 = the port plan's "
        "operating point, 410 image tokens/camera — the smallest layout "
        "inside the shipped distribution); ignored for Gemma trunks",
    )
    parser.add_argument(
        "--stream-counts",
        type=int,
        nargs="*",
        default=[4, 4, 7],
        help="decoder cross-attention layers per backbone KV stream, "
        "shallow to deep (0 skips a stream)",
    )
    parser.add_argument(
        "--conditioning-streams",
        choices=["kv", "residual"],
        default="kv",
        help="flow-expert conditioning surface: 'kv' = exported K/V of the "
        "global prefix layers (default; scheduled by --stream-counts), "
        "'residual' = FULL residual streams — the hidden state after every "
        "prefix layer via learned decoder-side adapters, expert layer i "
        "reading trunk layer i (flow decoder only; --stream-counts must "
        "stay at its default — the schedule is structural)",
    )
    parser.add_argument(
        "--seam-stop-grad",
        action="store_true",
        help="stop-gradient on the expert→trunk seam (molmo2 flow + "
        "residual conditioning): raw taps are detached before adapter "
        "projection, so flow-loss gradients into every trunk parameter "
        "are exactly zero while a live trunk still trains through the "
        "--joint-ce branch (the π0.5/KI recipe; attach-screen pre-reg "
        "2026-08-07)",
    )
    parser.add_argument(
        "--joint-ce",
        action="store_true",
        help="K arm of the attach-screen: ride the phase-1 CE objective "
        "(Molmo2ARDecoder suffix — --fast-tokenizer/--aux-fields flags "
        "apply to it verbatim) beside the flow loss at fixed weight 1.0 "
        "(KI's no-tuning result; deliberately not a knob). Requires "
        "molmo2 + --decoder flow + --conditioning-streams residual + "
        "--backbone-text-lr + --seam-stop-grad (random-init naive joint "
        "is an oracle-only negative control, never a run; "
        "--joint-unfrozen-seam is the warm-start-only escape)",
    )
    parser.add_argument(
        "--joint-unfrozen-seam",
        action="store_true",
        help="opt-in escape from the naive-joint guard (F-then-joint "
        "pre-reg 2026-08-09): admits --joint-ce WITHOUT --seam-stop-grad, "
        "so flow-loss gradients enter the trunk through the taps — the "
        "APT regime. Sane only when --init-from supplies an already-"
        "converged expert (required); the guard's collapse reasoning "
        "(KI: an uninformed head's early gradients wreck the trunk) is "
        "about random init, so fresh runs stay refused",
    )
    parser.add_argument(
        "--self-attention-mode",
        choices=["causal_actions", "bidirectional"],
        default="causal_actions",
        help="decoder self-attention over the action chunk",
    )
    parser.add_argument(
        "--time-conditioning",
        choices=[m.value for m in TimeConditioning],
        default=TimeConditioning.ADDITIVE.value,
        help="how flow time τ conditions the flow decoder: 'additive' (π0-style "
        "input add, the default) or 'adarms' (DiT-style per-layer scale/"
        "gate, identity at init). adarms changes the architecture — a fresh "
        "decoder only (cannot --init-from an additive checkpoint)",
    )
    parser.add_argument(
        "--target-time-embed",
        action="store_true",
        help="extend the flow decoder with the SnapFlow φ_s target-time "
        "embedding (zero-initialized output: inert until trained; may "
        "--init-from an unextended checkpoint — step 0 is then exactly "
        "that checkpoint). Implied by --distill snapflow",
    )
    parser.add_argument(
        "--distill",
        choices=["snapflow"],
        default=None,
        help="training objective variant: 'snapflow' = self-distillation "
        "toward 1-NFE decoding (L = α·L_FM + (1−α)·λ·L_shortcut with "
        "stop-gradient two-step-Euler shortcut targets; α=0.5, λ=0.1 "
        "frozen in code per the 2026-08-06 pre-registration). Flow "
        "decoder only; enables --target-time-embed",
    )
    parser.add_argument(
        "--decoder",
        choices=["flow", "ar_fast", "ar_backbone"],
        default="flow",
        help="action decoder: 'flow' (velocity field, the default), "
        "'ar_fast' (autoregressive FAST tokens through a fresh "
        "cross-attention decoder) or 'ar_backbone' (FAST tokens decoded "
        "by the FULL backbone itself — the decoder-only path; trains a "
        "~11M vocabulary patch, usually with --backbone-text-lr). AR "
        "decoders require --fast-tokenizer; the --decoder-* shape flags "
        "size flow/ar_fast only",
    )
    parser.add_argument(
        "--aux-fields",
        nargs="*",
        choices=[f.value for f in AuxField],
        default=None,
        help="train aux text generation from judge annotations (ar_backbone "
        "only): fields rendered before BOA in template order; datasets "
        "whose annotation stamp is absent/stale train as unjudged, loudly. "
        "Omit to train actions only (the historical objective)",
    )
    parser.add_argument(
        "--aux-loss-weight",
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
        help="FAST tokenizer artifact: a local directory or "
        "<user>/<repo>/<subfolder> on the hub (e.g. "
        "mcobzarenco/bijou-checkpoints/fast_tokenizer_v1)",
    )
    parser.add_argument(
        "--decoder-hidden",
        type=int,
        default=768,
        help="decoder hidden size",
    )
    parser.add_argument(
        "--decoder-heads",
        type=int,
        default=6,
        help="decoder self-attention heads",
    )
    parser.add_argument(
        "--decoder-intermediate",
        type=int,
        default=3072,
        help="decoder MLP intermediate size",
    )
    parser.add_argument(
        "--decoder-cross-heads",
        type=int,
        default=4,
        help="decoder cross-attention heads",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="actions predicted per sample (frames at the dataset fps)",
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
        "--activation-checkpointing",
        action="store_true",
        help="activation checkpointing over the molmo2 decoder blocks "
        "(#20): recompute each block in backward instead of retaining "
        "its interior activations — memory only, the gradient is "
        "oracle-pinned bitwise to the plain step. Engages wherever the "
        "trunk runs under grad (live-trunk prefix encode + CE suffix); "
        "no-grad paths are untouched. Molmo2 trunks only",
    )
    parser.add_argument("--steps", type=int, default=200, help="total optimizer steps")
    parser.add_argument(
        "--decoder-lr",
        type=float,
        default=1e-4,
        help="peak learning rate of the action decoder (cosine decay to "
        "10%% after warmup); every component-lr below shares this "
        "schedule shape, scaled to its own peak",
    )
    parser.add_argument(
        "--backbone-text-lr",
        type=float,
        default=None,
        help="peak learning rate for the backbone TEXT stack (decoder "
        "layers up to the deepest exported stream, PLE projections, "
        "multimodal projector); OMIT to keep the backbone frozen (the "
        "historical behavior). Token embeddings and PLE tables always "
        "stay frozen. A live backbone loads fp32 with bf16-autocast "
        "forwards; suggest 1e-5",
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
        "weight_decay tracks the LR schedule), output heads keep "
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
        help="warm start: expert weights from this checkpoint directory, "
        "fresh optimizer and step count (use a new --save-dir)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="full resume: weights + optimizer/scheduler/step from this "
        "checkpoint directory (--steps counts total, including resumed); "
        "demands a --seed the checkpoint was not trained with (see "
        "--allow-same-seed-resume)",
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
        "build the decoder fresh — e.g. a new flow expert reading an "
        "AR-pretrained trunk. Unlike --init-from, the source's decoder "
        "config is ignored",
    )
    parser.add_argument(
        "--prompt-generate-bracket",
        action="store_true",
        help="render [generate|actions] in prompts for non-AR decoders "
        "(stage-2 trunk consistency: an AR-pretrained trunk shaped its "
        "conditioning/state positions WITH the bracket). ar_backbone "
        "always renders it — passing this there is an error, not a "
        "no-op",
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
    raw = parser.parse_args()
    if raw.init_from is not None and raw.resume is not None:
        parser.error("--init-from and --resume are mutually exclusive")
    if raw.backbone_init_from is not None and (
        raw.init_from is not None or raw.resume is not None
    ):
        parser.error(
            "--backbone-init-from is mutually exclusive with --init-from/"
            "--resume (those load the decoder too)",
        )
    if raw.prompt_generate_bracket and raw.decoder == "ar_backbone":
        parser.error(
            "--prompt-generate-bracket is implied (always on) for "
            "ar_backbone — drop the flag so runs have one spelling",
        )
    if raw.rewarmup_steps > 0 and raw.resume is None:
        parser.error(
            "--rewarmup-steps anchors at the resume step — it requires "
            "--resume (fresh runs use --warmup-steps)",
        )
    if raw.allow_same_seed_resume and raw.resume is None:
        parser.error(
            "--allow-same-seed-resume only applies to --resume "
            "(fresh runs and --init-from have no seed to collide with)",
        )
    if not 0.0 <= raw.holdout_episodes < 1.0:
        parser.error("--holdout-episodes must be in [0, 1)")
    if raw.holdout_episodes > 0 and raw.eval_samples is None:
        parser.error(
            "--eval-samples is required when --holdout-episodes > 0 "
            "(it sizes the held-out eval_chunk_mae probe)",
        )
    if raw.eval_samples is not None and raw.eval_samples < 1:
        parser.error("--eval-samples must be >= 1")
    if raw.decoder_lr <= 0:
        parser.error("--decoder-lr must be > 0 (the decoder always trains)")
    if raw.decoder in ("ar_fast", "ar_backbone") and raw.fast_tokenizer is None:
        parser.error(f"--decoder {raw.decoder} requires --fast-tokenizer")
    if raw.decoder == "flow" and raw.fast_tokenizer is not None and not raw.joint_ce:
        parser.error(
            "--fast-tokenizer is only consumed by the AR decoders "
            "(or the --joint-ce CE rider)",
        )
    if raw.seam_stop_grad and (
        raw.decoder != "flow" or raw.conditioning_streams != "residual"
    ):
        parser.error(
            "--seam-stop-grad detaches residual taps at the expert seam — "
            "it requires --decoder flow --conditioning-streams residual",
        )
    if raw.joint_ce:
        # The K arm's frozen preconditions (attach-screen pre-reg
        # 2026-08-07): the CE branch is the phase-1 objective continuing
        # VERBATIM (live trunk, FAST suffix, bracketed prompts), and the
        # seam must be stop-grad — random-init naive joint (flow
        # gradients into the trunk) exists only as an oracle negative
        # control. --joint-unfrozen-seam (guarded below) is the
        # warm-start-only escape.
        if raw.decoder != "flow" or raw.conditioning_streams != "residual":
            parser.error(
                "--joint-ce rides the flow expert — it requires --decoder "
                "flow --conditioning-streams residual",
            )
        if not raw.seam_stop_grad and not raw.joint_unfrozen_seam:
            parser.error(
                "--joint-ce without --seam-stop-grad is the naive-joint "
                "arm — a published collapse (KI), refused as a run",
            )
        if raw.fast_tokenizer is None:
            parser.error("--joint-ce requires --fast-tokenizer (its CE suffix)")
        if raw.backbone_text_lr is None:
            parser.error(
                "--joint-ce without --backbone-text-lr trains no trunk — "
                "the CE branch would ride frozen; use the F arm instead",
            )
        if not raw.prompt_generate_bracket:
            parser.error(
                "--joint-ce requires --prompt-generate-bracket: phase-1 "
                "prompts carried the [generate|…] request block, and the "
                "CE branch continuing verbatim means rendering it",
            )
        if raw.distill is not None:
            parser.error("--joint-ce and --distill are mutually exclusive")
    if raw.joint_unfrozen_seam:
        # The escape's own scope (F-then-joint pre-reg 2026-08-09): it
        # modifies --joint-ce only, contradicts an explicit stop-grad,
        # and is warm-start-only — the collapse the guard refuses is a
        # random-init expert's early gradients, so a fresh run gets no
        # escape.
        if not raw.joint_ce:
            parser.error(
                "--joint-unfrozen-seam is the naive-joint guard escape — "
                "it modifies --joint-ce and does nothing without it",
            )
        if raw.seam_stop_grad:
            parser.error(
                "--joint-unfrozen-seam contradicts --seam-stop-grad — the "
                "escape exists to run the seam open; pick one",
            )
        if raw.init_from is None:
            parser.error(
                "--joint-unfrozen-seam requires --init-from: the naive-"
                "joint collapse (KI) is a random-init pathology, so the "
                "open seam is admitted only for warm starts whose expert "
                "is already informed (F-then-joint pre-reg 2026-08-09)",
            )
    if raw.decoder != "flow" and raw.time_conditioning != "additive":
        parser.error(
            "--time-conditioning is flow-only (AR decoders have no \u03c4)",
        )
    if raw.decoder != "flow" and raw.distill is not None:
        parser.error("--distill is flow-only (it distills the velocity field)")
    if raw.decoder != "flow" and raw.target_time_embed:
        parser.error("--target-time-embed is flow-only (\u03c6_s conditions \u03c4)")
    if raw.distill == "snapflow":
        # The shortcut term forwards at s=0 \u2014 \u03c6_s is required, not optional.
        raw.target_time_embed = True
    if raw.aux_fields is not None and raw.decoder != "ar_backbone" and not raw.joint_ce:
        parser.error(
            "--aux-fields is ar_backbone-only (aux rides its suffix — or "
            "the --joint-ce rider's)",
        )
    if raw.aux_fields is not None and not raw.aux_fields:
        parser.error("--aux-fields given with no fields — omit the flag instead")
    if raw.aux_fields is not None:
        # Template order is an invariant, not a preference (AuxSpec
        # re-guards, but that fires only after dataset selection).
        ordered = [f.value for f in AuxField if f.value in raw.aux_fields]
        if list(raw.aux_fields) != ordered:
            parser.error(
                f"--aux-fields must keep template order {ordered} "
                f"(got {list(raw.aux_fields)})",
            )
    if raw.aux_loss_weight <= 0:
        parser.error("--aux-loss-weight must be > 0 (omit --aux-fields to disable)")
    if raw.aux_fields is not None and (raw.cameras or raw.max_cameras is not None):
        # The 'visible' aux indices are positions in the full sorted
        # camera set; camera selection would silently shift them (the
        # Collator re-guards, but that fires only after dataset
        # selection).
        parser.error("--aux-fields cannot combine with --cameras/--max-cameras")
    if raw.aux_dropout is not None and raw.aux_fields is None:
        parser.error("--aux-dropout requires --aux-fields (it drops aux labels)")
    if raw.aux_prompt_hash is not None and raw.aux_fields is None:
        parser.error("--aux-prompt-hash requires --aux-fields (it gates aux labels)")
    if raw.aux_dropout is not None and not 0.0 <= raw.aux_dropout < 1.0:
        parser.error(f"--aux-dropout {raw.aux_dropout} outside [0, 1)")
    if not 0.0 <= raw.camera_kind_dropout < 1.0:
        parser.error(
            f"--camera-kind-dropout {raw.camera_kind_dropout} outside [0, 1)",
        )
    if not 0.0 <= raw.instruction_augment <= 1.0:
        parser.error(
            f"--instruction-augment {raw.instruction_augment} outside [0, 1]",
        )
    if not 0.0 <= raw.state_dropout < 1.0:
        parser.error(f"--state-dropout {raw.state_dropout} outside [0, 1)")
    if raw.condition_fields is not None and not raw.condition_fields:
        parser.error("--condition-fields given with no fields — omit the flag")
    if raw.condition_fields is not None:
        ordered = [f.value for f in ConditionField if f.value in raw.condition_fields]
        if list(raw.condition_fields) != ordered:
            parser.error(
                f"--condition-fields must keep template order {ordered} "
                f"(got {list(raw.condition_fields)})",
            )
    if raw.condition_dropout is not None and raw.condition_fields is None:
        parser.error("--condition-dropout requires --condition-fields")
    if raw.condition_dropout is not None and not 0.0 <= raw.condition_dropout < 1.0:
        parser.error(
            f"--condition-dropout {raw.condition_dropout} outside [0, 1)",
        )
    condition_dropout = (
        raw.condition_dropout
        if raw.condition_dropout is not None
        else (0.1 if raw.condition_fields is not None else 0.0)
    )
    subgoal_conditioned = raw.condition_fields is not None and (
        "subgoal" in raw.condition_fields
    )
    if raw.subgoal_dropout is not None and not subgoal_conditioned:
        parser.error("--subgoal-dropout requires subgoal in --condition-fields")
    if raw.subgoal_dropout is not None and not 0.0 <= raw.subgoal_dropout < 1.0:
        parser.error(f"--subgoal-dropout {raw.subgoal_dropout} outside [0, 1)")
    subgoal_dropout = (
        raw.subgoal_dropout
        if raw.subgoal_dropout is not None
        else (0.5 if subgoal_conditioned else 0.0)
    )
    aux_dropout = (
        raw.aux_dropout
        if raw.aux_dropout is not None
        else (0.1 if raw.aux_fields is not None else 0.0)
    )
    if raw.field_dropout is not None and raw.aux_fields is None:
        parser.error("--field-dropout requires --aux-fields (it drops requests)")
    if raw.field_dropout is not None and not 0.0 <= raw.field_dropout < 1.0:
        parser.error(f"--field-dropout {raw.field_dropout} outside [0, 1)")
    field_dropout = (
        raw.field_dropout
        if raw.field_dropout is not None
        else (0.1 if raw.aux_fields is not None else 0.0)
    )
    if raw.decoder == "ar_backbone":
        # The backbone IS the architecture: decoder shape flags and the
        # cross-attention schedule describe models this run doesn't build.
        for flag, attribute in (
            ("--decoder-hidden", "decoder_hidden"),
            ("--decoder-heads", "decoder_heads"),
            ("--decoder-intermediate", "decoder_intermediate"),
            ("--decoder-cross-heads", "decoder_cross_heads"),
            ("--stream-counts", "stream_counts"),
        ):
            if getattr(raw, attribute) != parser.get_default(attribute):
                parser.error(
                    f"{flag} sizes the flow/ar_fast decoders; ar_backbone "
                    "IS the backbone — drop the flag",
                )
    if raw.conditioning_streams == "residual":
        # Residual conditioning is a flow-expert architecture: the schedule
        # is structural (layer i reads trunk layer i), so the K/V schedule
        # knob has nothing to size.
        if raw.decoder != "flow":
            parser.error(
                "--conditioning-streams residual is a flow-expert "
                f"architecture; --decoder {raw.decoder} conditions on "
                "K/V exports only",
            )
        if raw.stream_counts != parser.get_default("stream_counts"):
            parser.error(
                "--stream-counts schedules K/V conditioning; under "
                "--conditioning-streams residual the schedule is structural "
                "(one stream per prefix layer, 1:1 ascending) — drop the "
                "flag",
            )
    for name, value in (
        ("--backbone-text-lr", raw.backbone_text_lr),
        ("--backbone-vision-lr", raw.backbone_vision_lr),
    ):
        if value is not None and value <= 0:
            parser.error(
                f"{name} {value} is not a usable learning rate — omit the "
                "flag entirely to keep that component frozen",
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
    return TrainArgs(
        train_data=tuple(raw.train_data),
        exclude=tuple(raw.exclude),
        fps=tuple(raw.fps) if raw.fps else None,
        camera_counts=tuple(raw.camera_counts) if raw.camera_counts else None,
        holdout_episodes=raw.holdout_episodes,
        split_seed=raw.split_seed,
        backbone=raw.backbone,
        save_dir=raw.save_dir,
        init_from=raw.init_from,
        resume=raw.resume,
        allow_same_seed_resume=raw.allow_same_seed_resume,
        backbone_init_from=raw.backbone_init_from,
        prompt_generate_bracket=raw.prompt_generate_bracket,
        instruction=raw.instruction,
        cameras=tuple(raw.cameras) if raw.cameras else None,
        max_cameras=raw.max_cameras,
        max_soft_tokens=raw.max_soft_tokens,
        max_crops=raw.max_crops,
        stream_counts=tuple(raw.stream_counts),
        conditioning_streams=raw.conditioning_streams,
        seam_stop_grad=raw.seam_stop_grad,
        joint_ce=raw.joint_ce,
        joint_unfrozen_seam=raw.joint_unfrozen_seam,
        self_attention_mode=raw.self_attention_mode,
        time_conditioning=raw.time_conditioning,
        target_time_embed=raw.target_time_embed,
        distill=raw.distill,
        decoder=raw.decoder,
        fast_tokenizer=raw.fast_tokenizer,
        aux_fields=tuple(raw.aux_fields) if raw.aux_fields is not None else None,
        aux_loss_weight=raw.aux_loss_weight,
        aux_dropout=aux_dropout,
        field_dropout=field_dropout,
        aux_prompt_hash=raw.aux_prompt_hash,
        camera_kind_dropout=raw.camera_kind_dropout,
        instruction_augment=raw.instruction_augment,
        condition_fields=(
            tuple(raw.condition_fields) if raw.condition_fields is not None else None
        ),
        condition_dropout=condition_dropout,
        subgoal_dropout=subgoal_dropout,
        state_dropout=raw.state_dropout,
        decoder_hidden=raw.decoder_hidden,
        decoder_heads=raw.decoder_heads,
        decoder_intermediate=raw.decoder_intermediate,
        decoder_cross_heads=raw.decoder_cross_heads,
        chunk_size=raw.chunk_size,
        batch_size=raw.batch_size,
        bucket_by_length=raw.bucket_by_length,
        backward_chunks=raw.backward_chunks,
        zero1=raw.zero1,
        sync_save=raw.sync_save,
        chunk_grad_allreduce=raw.chunk_grad_allreduce,
        activation_checkpointing=raw.activation_checkpointing,
        steps=raw.steps,
        decoder_lr=raw.decoder_lr,
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
            f"divide --batch-size {args.batch_size} (equal chunks — the "
            "pre-registered ladder rungs)",
        )
    os.environ["LEROBOT_VIDEO_DECODER_CACHE_SIZE"] = str(args.video_decoder_cache)

    # TF32 for fp32 matmuls (torch's default is full-IEEE "highest"): the
    # expert trains in fp32, and true fp32 matmul leaves ~5-7x of H100
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
    # dies OOM (analyze with torch.cuda._memory_viz or fontaine's
    # reader). Measurement instrument for smoke rungs — never set on a
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

    checkpoint_dir = resolve_checkpoint_dir(args.backbone)
    # Trunk dispatch: the backbone family is a structural fact of the
    # checkpoint (its own config.json), not a CLI axis.
    molmo2_trunk = (
        json.loads((checkpoint_dir / "config.json").read_text()).get(
            "model_type",
            "",
        )
        == "molmo2"
    )
    if molmo2_trunk and args.decoder == "ar_fast":
        raise SystemExit(
            "molmo2 backbones support --decoder ar_backbone (phase 1) "
            "and --decoder flow with residual conditioning (the "
            "attach-screen pre-reg, 2026-08-07) — ar_fast conditions on "
            "K/V exports this trunk does not produce",
        )
    if (
        molmo2_trunk
        and args.decoder == "flow"
        and args.conditioning_streams != "residual"
    ):
        raise SystemExit(
            "molmo2 flow experts condition on residual taps only "
            "(--conditioning-streams residual): the trunk exports no K/V "
            "streams (attach-screen pre-reg, 2026-08-07)",
        )
    if (args.seam_stop_grad or args.joint_ce) and not molmo2_trunk:
        # joint_ce checked too: --joint-unfrozen-seam admits joint_ce
        # without seam_stop_grad, and the rider is built only on the
        # molmo2 branch — a gemma joint run would silently drop it.
        raise SystemExit(
            "--seam-stop-grad / --joint-ce are the molmo2 attach-screen "
            "seam flags; the gemma residual arm trains under a frozen "
            "trunk (no seam to cut)",
        )

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
    )
    action_dim, state_dim = selection.action_dim, selection.state_dim
    per_dataset_stats = selection.per_dataset_stats
    dataset = selection.concat()
    if is_main:
        print(
            f"train data: {len(selection.datasets)} datasets, "
            f"{selection.total_episodes} episodes, {len(dataset)} frames, "
            f"action/state dim {action_dim}/{state_dim}",
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

    action_codec = (
        resolve_action_codec(args.fast_tokenizer)
        if args.fast_tokenizer is not None
        else None
    )
    aux_decode_config: AuxDecodeConfig | None = None
    aux_spec: AuxSpec | None = None
    if args.aux_fields is not None:
        assert action_codec is not None  # parse_args guard (ar_backbone-only)
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
    collator = Collator(
        inputs=(
            Molmo2InputsCollator(str(checkpoint_dir), args.max_crops)
            if molmo2_trunk
            else GemmaInputsCollator(
                str(checkpoint_dir),
                args.max_soft_tokens,
            )
        ),
        instruction=args.instruction,
        camera_filter=args.cameras,
        max_cameras=args.max_cameras,
        action_codec=action_codec,
        aux=aux_spec,
        generate_bracket=args.decoder == "ar_backbone" or args.prompt_generate_bracket,
        generate_override=None,
        camera_kind_dropout=args.camera_kind_dropout,
        instruction_augment=args.instruction_augment,
        condition_fields=tuple(
            ConditionField(f) for f in (args.condition_fields or ())
        ),
        condition_dropout=args.condition_dropout,
        subgoal_condition_dropout=args.subgoal_dropout,
        state_dropout=args.state_dropout,
    )
    if is_main and args.state_dropout > 0:
        print(
            f"state dropout: p={args.state_dropout} — proprioceptive state "
            "masked to the dataset mean per sample (train-time "
            "regularizer; probes score intact state)",
            flush=True,
        )
    if is_main and (args.instruction_augment > 0 or args.condition_fields):
        labeled_episodes = sum(
            len(dataset.episode_annotations) for dataset in selection.datasets
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
        generate_override=(() if args.decoder == "ar_backbone" else None),
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
        # readout is the offline --mask-state reliance probe).
        state_dropout=0.0,
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
    # under bf16 autocast in BijouTrainStep. Frozen runs keep the
    # checkpoint dtype (bf16) exactly as before the unfreeze flags.
    backbone_dtype = torch.float32 if args.backbone_trained else None
    model: BijouModel[Any, Any]
    if args.decoder == "flow" and molmo2_trunk:
        # The attach-screen composition (pre-reg 2026-08-07): full
        # multimodal Molmo2 trunk + residual-tap encoder (the pinned
        # stride-3 rule) + flow expert reading the taps 1:1 ascending.
        molmo2_config = load_molmo2_config(checkpoint_dir)
        expert_config = molmo2_residual_expert_config(
            molmo2_config,
            action_dim=action_dim,
            state_dim=state_dim,
            hidden_size=args.decoder_hidden,
            num_attention_heads=args.decoder_heads,
            intermediate_size=args.decoder_intermediate,
            cross_attention_heads=args.decoder_cross_heads,
            chunk_size=args.chunk_size,
            self_attention_mode=SelfAttentionMode(args.self_attention_mode),
            time_conditioning=TimeConditioning(args.time_conditioning),
            target_time_embed=args.target_time_embed,
        )
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
            residual_exports=expert_config.streams,
            device=device,
            dtype=torch.float32,
        )
        model = BijouModel(
            backbone=molmo2_backbone,
            encoder=molmo2_encoder,
            decoder=FlowDecoder(expert_config, device=device, dtype=torch.float32),
        )
        model.seam_stop_grad = args.seam_stop_grad
        if args.joint_ce:
            # The K arm's CE rider: the phase-1 decoder build VERBATIM
            # (same config surface, same table init) — its objective,
            # tables and aux runtime continue exactly as in phase 1.
            assert args.fast_tokenizer is not None  # parse_args guard
            assert action_codec is not None
            joint_ar_config = ARBackboneConfig(
                tokenizer=args.fast_tokenizer,
                vocab_total=action_codec.vocab_total,
                block_base=molmo2_config.text.fast_block_base,
                chunk_size=args.chunk_size,
                action_dim=action_dim,
                suffix_format=SUFFIX_FORMAT,
                aux=aux_decode_config,
            )
            joint_text_tokenizer = Molmo2TextTokenizer(str(checkpoint_dir))
            joint_carriers = newline_carrier_ids(
                joint_text_tokenizer,
                text_vocab_size=molmo2_config.text.vocab_size,
                terminator_id=joint_text_tokenizer.encode(
                    "\n",
                    add_special_tokens=False,
                )[0],
            )
            joint_rider = Molmo2ARDecoder(
                joint_ar_config,
                molmo2_config.text,
                action_codec,
                tokenizer=joint_text_tokenizer,
                aux_runtime=(
                    build_aux_runtime(
                        aux_decode_config,
                        joint_text_tokenizer,
                        newline_carrier_ban=True,
                    )
                    if aux_decode_config is not None
                    else None
                ),
                aux_loss_weight=args.aux_loss_weight,
                newline_carrier_ids=joint_carriers,
                device=device,
                dtype=torch.float32,
            )
            joint_rider.init_tables_from_backbone(molmo2_backbone)
            model.joint_ce = joint_rider
        if args.seam_stop_grad:
            seam_desc = "stop-grad"
        elif args.joint_ce:
            # Only reachable via --joint-unfrozen-seam (parse guard) —
            # the launch banner must say the trunk takes flow gradients.
            seam_desc = "UNFROZEN (flow grads enter the trunk)"
        else:
            seam_desc = "transparent"
        schedule_desc = (
            f"molmo2 residual taps {expert_config.streams} (1:1 ascending, "
            f"learned adapters; seam {seam_desc}"
            f"{', joint CE rider' if args.joint_ce else ''})"
        )
    elif args.decoder == "flow":
        if args.conditioning_streams == "residual":
            expert_config = residual_expert_config(
                load_config(checkpoint_dir),
                action_dim=action_dim,
                state_dim=state_dim,
                hidden_size=args.decoder_hidden,
                num_attention_heads=args.decoder_heads,
                intermediate_size=args.decoder_intermediate,
                cross_attention_heads=args.decoder_cross_heads,
                chunk_size=args.chunk_size,
                self_attention_mode=SelfAttentionMode(args.self_attention_mode),
                time_conditioning=TimeConditioning(args.time_conditioning),
                target_time_embed=args.target_time_embed,
            )
        else:
            expert_config = default_expert_config(
                load_config(checkpoint_dir),
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
        model = from_backbone(
            checkpoint_dir,
            expert_config,
            device=device,
            dtype=backbone_dtype,
            expert_dtype=torch.float32,
        )
        schedule_desc = (
            f"residual res0..res{max(expert_config.streams)} (1:1 ascending, "
            "learned adapters)"
            if expert_config.residual_streams
            else str(expert_config.cross_attention_schedule)
        )
    elif args.decoder == "ar_backbone" and molmo2_trunk:
        assert args.fast_tokenizer is not None  # parse_args guard
        assert action_codec is not None
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
        ar_backbone_config = ARBackboneConfig(
            tokenizer=args.fast_tokenizer,
            vocab_total=action_codec.vocab_total,
            # The SECOND extension block, directly after the 128 image
            # specials — Qwen3's ~271-id unused tail cannot hold the
            # 1,026 FAST ids (port plan §6 amendment; the embedding and
            # fresh untied head rows are decoder-owned trainables).
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
        model = BijouModel(
            backbone=molmo2_backbone,
            encoder=molmo2_encoder,
            decoder=molmo2_decoder,
        )
        schedule_desc = (
            f"molmo2 full-depth suffix, FAST extension block @ "
            f"{ar_backbone_config.block_base}"
        )
    elif args.decoder == "ar_backbone":
        assert args.fast_tokenizer is not None  # parse_args guard
        assert action_codec is not None
        backbone_config = load_config(checkpoint_dir)
        # Prefill still stops at the deepest non-KV-shared layer; its
        # stream export rides along unused (the decoder reads the CACHE).
        stop = backbone_config.text.first_kv_shared_layer_idx - 1
        backbone, encoder = build_gemma_encoder(
            checkpoint_dir,
            backbone_config,
            exports=(stop,),
            max_soft_tokens=args.max_soft_tokens,
            state_dim=state_dim,
            device=device,
            dtype=backbone_dtype,
            depth=BackboneDepth.FULL,
        )
        # Tail-anchored block: the last vocab_total ids sit inside the
        # tokenizer's unused tail (E2B: 261118.. ⊂ the 3259-id run at
        # 258885..262143) — no magic constant, adapts to any backbone,
        # recorded in the checkpoint's decoder section.
        ar_backbone_config = ARBackboneConfig(
            tokenizer=args.fast_tokenizer,
            vocab_total=action_codec.vocab_total,
            block_base=backbone_config.text.vocab_size - action_codec.vocab_total,
            chunk_size=args.chunk_size,
            action_dim=action_dim,
            suffix_format=SUFFIX_FORMAT,
            aux=aux_decode_config,
        )
        text_tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(checkpoint_dir),
        )
        aux_runtime = (
            build_aux_runtime(aux_decode_config, text_tokenizer)
            if aux_decode_config is not None
            else None
        )
        ar_backbone_decoder = ARBackboneDecoder(
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
        ar_backbone_decoder.init_tables_from_backbone(backbone)
        model = BijouModel(
            backbone=backbone,
            encoder=encoder,
            decoder=ar_backbone_decoder,
        )
        schedule_desc = (
            f"full-depth suffix, FAST block @ {ar_backbone_config.block_base}"
        )
    else:
        assert args.fast_tokenizer is not None  # parse_args guard
        assert action_codec is not None
        backbone_config = load_config(checkpoint_dir)
        streams = prefix_global_layers(backbone_config)
        if len(args.stream_counts) != len(streams):
            raise SystemExit(
                f"--stream-counts has {len(args.stream_counts)} entries but "
                f"the backbone prefix has {len(streams)} global layers "
                f"({streams})",
            )
        ar_schedule = tuple(
            kv_stream_name(stream)
            for stream, count in zip(streams, args.stream_counts, strict=True)
            for _ in range(count)
        )
        exports = tuple(
            stream
            for stream, count in zip(streams, args.stream_counts, strict=True)
            if count > 0
        )
        backbone, encoder = build_gemma_encoder(
            checkpoint_dir,
            backbone_config,
            exports=exports,
            max_soft_tokens=args.max_soft_tokens,
            state_dim=state_dim,
            device=device,
            dtype=backbone_dtype,
        )
        ar_config = ARFastConfig(
            hidden_size=args.decoder_hidden,
            num_attention_heads=args.decoder_heads,
            intermediate_size=args.decoder_intermediate,
            hidden_activation=backbone_config.text.hidden_activation,
            rms_norm_eps=backbone_config.text.rms_norm_eps,
            self_attention_rope_theta=10_000.0,
            cross_attention_heads=args.decoder_cross_heads,
            schedule=ar_schedule,
            tokenizer=args.fast_tokenizer,
            vocab_total=action_codec.vocab_total,
            state_dim=state_dim,
            chunk_size=args.chunk_size,
            action_dim=action_dim,
        )
        model = BijouModel(
            backbone=backbone,
            encoder=encoder,
            decoder=ARFastDecoder(
                ar_config,
                encoder.stream_geometries(),
                action_codec,
                device=device,
                dtype=torch.float32,
            ),
        )
        schedule_desc = str(ar_schedule)
    backbone_counts = unfreeze_backbone(model, args)
    if args.activation_checkpointing:
        if not isinstance(model.backbone, Molmo2Model):
            raise SystemExit(
                "--activation-checkpointing is wired for the molmo2 decoder stack only",
            )
        model.backbone.text.transformer.gradient_checkpointing = True
        if is_main:
            print(
                "activation checkpointing ON (molmo2 decoder blocks: "
                "recompute in backward, gradient-identical; engages only "
                "where the trunk runs under grad)",
                flush=True,
            )
    if not args.backbone_trained:
        # Frozen runs encode the prefix under no_grad (BijouTrainStep),
        # so the prompt-side state projection CANNOT receive gradients
        # there — freeze it rather than hand DDP a grad-less trainable
        # (static_graph errors on the first backward otherwise). The
        # zero init makes the state token exactly inert: frozen-backbone
        # behavior matches the pre-state-token model. Training it under
        # a frozen backbone (stage 2) needs a grad-transparent prefix —
        # a deliberate future change, not a default.
        model.encoder.state_proj.requires_grad_(False)
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
    model.distill = args.distill
    if args.distill is not None and is_main:
        print(
            f"distill: {args.distill} objective "
            f"(α={SNAPFLOW_ALPHA}, λ={SNAPFLOW_LAMBDA}, stop-gradient "
            "two-step-Euler shortcut targets, no EMA teacher)",
            flush=True,
        )
    n_trainable = sum(p.numel() for p in model.decoder.parameters()) + (
        sum(p.numel() for p in model.joint_ce.parameters())
        if model.joint_ce is not None
        else 0
    )
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
        banner_encoder = model.encoder
        banner_backbone = model.backbone
        if isinstance(banner_encoder, GemmaEncoder):
            streams_desc = (
                f"residual taps {banner_encoder.residual_exports}"
                if banner_encoder.residual_exports
                else str(banner_encoder.exports)
            )
            assert isinstance(banner_backbone, Gemma4Model)
            n_backbone_layers = len(banner_backbone.language_model.layers)
        else:
            assert isinstance(banner_backbone, Molmo2Model)
            assert isinstance(banner_encoder, Molmo2Encoder)
            streams_desc = (
                f"residual taps {banner_encoder.residual_exports}"
                if banner_encoder.residual_exports
                else "prefix cache (AR suffix)"
            )
            n_backbone_layers = len(banner_backbone.text.transformer.blocks)
        print(
            f"model: {backbone_desc} "
            f"({n_backbone_layers} "
            f"layers, streams {streams_desc}) + fp32 "
            f"{args.decoder} decoder ({n_trainable / 1e6:.1f}M params, "
            f"schedule {schedule_desc})",
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
    # model's named groups route to per-component learning rates; the
    # head group always trains at --decoder-lr.
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
    # path clips exactly the expert (unchanged behavior); a live backbone is
    # clipped jointly with it (one global norm).
    clipped_parameters: list[torch.nn.Parameter] = [
        p for group in param_groups for p in group["params"]
    ]
    # The re-warmup ramp needs the resume step BEFORE the scheduler
    # state loads — read it from the checkpoint's metadata.
    resume_step = (
        int(json.loads((args.resume / "bijou_config.json").read_text())["step"])
        if args.resume is not None
        else 0
    )
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
    checkpoint_to_load = args.init_from or args.resume
    if checkpoint_to_load is not None:
        saved_decoder_config = ensure_matching_decoder_config(
            model.decoder,
            checkpoint_to_load,
        )
        # CPU-load + copy-in: loading straight to the device transiently
        # holds a second copy of the weights next to the built module
        # (see loading.load_adapted_backbone).
        expert_state = load_file(
            str(checkpoint_to_load / "expert.safetensors"),
            device="cpu",
        )
        # Strict always: pre-format-5 checkpoints carry parameters this
        # code deleted (mode tables, suffix state_proj) and are refused
        # by the key mismatch — no migration path (owner call,
        # 2026-08-03). One sanctioned exception: the φ_s target-time
        # extension (config guard above) may miss EXACTLY the fresh φ_s
        # keys, which keep their built init (zero-init output => step-0
        # model ≡ checkpoint).
        phi_s_extension = (
            isinstance(model.decoder, FlowDecoder)
            and model.decoder.config.target_time_embed
            and not saved_decoder_config.get("target_time_embed", False)
        )
        if phi_s_extension:
            phi_s_keys = {
                key
                for key in model.decoder.state_dict()
                if key.startswith(("target_time_in_proj.", "target_time_out_proj."))
            }
            missing, unexpected = model.decoder.load_state_dict(
                expert_state,
                strict=False,
            )
            if set(missing) != phi_s_keys or unexpected:
                raise SystemExit(
                    f"expert.safetensors mismatch at {checkpoint_to_load} "
                    f"beyond the φ_s extension: missing {sorted(missing)} "
                    f"(expected exactly {sorted(phi_s_keys)}), unexpected "
                    f"{sorted(unexpected)}",
                )
        else:
            model.decoder.load_state_dict(expert_state, strict=True)
        # Prompt-side parameters (state_proj) — format-3-prompt
        # checkpoints always write the file.
        model.encoder.load_state_dict(
            load_file(
                str(checkpoint_to_load / "prompt.safetensors"),
                device="cpu",
            ),
            strict=True,
        )
        if model.joint_ce is not None:
            # A joint run's continuation must continue the rider too —
            # a checkpoint without the file is not a joint checkpoint.
            rider_path = checkpoint_to_load / "joint_ce.safetensors"
            if not rider_path.exists():
                raise SystemExit(
                    f"--joint-ce continuation from {checkpoint_to_load}, "
                    "but it carries no joint_ce.safetensors — that "
                    "checkpoint was not written by a joint run (use "
                    "--backbone-init-from for the stage-2 warm start)",
                )
            model.joint_ce.load_state_dict(
                load_file(str(rider_path), device="cpu"),
                strict=True,
            )
        if is_main:
            print(f"loaded expert weights from {checkpoint_to_load}", flush=True)
        # Backbone-trained checkpoints carry the adapted file; plain ones
        # don't, and the HF backbone loaded above simply stays (that is the
        # cont45k -> unfreeze continuation path).
        if (checkpoint_to_load / "backbone.safetensors").exists():
            load_adapted_backbone(model, checkpoint_to_load)
            if not args.backbone_trained:
                # Frozen inherited backbone: every checkpoint this run saves
                # must carry the snapshot too (see save_checkpoint).
                adapted_backbone_source = checkpoint_to_load / "backbone.safetensors"
            if is_main:
                print(
                    f"loaded ADAPTED backbone weights from "
                    f"{checkpoint_to_load} (bf16 snapshot into "
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
        load_backbone_init(model, args.backbone_init_from)
        if not args.backbone_trained:
            adapted_backbone_source = args.backbone_init_from / "backbone.safetensors"
        if model.joint_ce is not None:
            # "The phase-1 CE objective continuing VERBATIM" includes its
            # decoder-owned FAST tables: the source checkpoint's
            # expert.safetensors IS the phase-1 Molmo2ARDecoder state
            # (same config surface), loaded strictly — a fresh-table
            # rider would restart the action head, not continue it.
            model.joint_ce.load_state_dict(
                load_file(
                    str(args.backbone_init_from / "expert.safetensors"),
                    device="cpu",
                ),
                strict=True,
            )
            if is_main:
                print(
                    "joint-CE rider: phase-1 FAST tables loaded from "
                    f"{args.backbone_init_from}/expert.safetensors "
                    "(the CE branch continues, not restarts)",
                    flush=True,
                )
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
                f"{optimizer_path} missing (checkpoint predates optimizer "
                "saving) — use --init-from for a warm start instead",
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

    # DDP wiring: ONE train-step module owns prefix encode + objective, so
    # a single wrapper hooks gradients of everything trained, for both the
    # frozen and live-backbone regimes (the frozen backbone contributes no
    # trainable parameters — DDP's reducer ignores it). Wrapping AFTER the
    # weight load means DDP's construction-time broadcast (rank 0 -> all)
    # covers the loaded state. model.decoder stays the raw module for
    # eval, clipping and checkpointing.
    train_step: torch.nn.Module = BijouTrainStep(
        model,
        backbone_trained=args.backbone_trained,
    )
    if distributed and args.chunk_grad_allreduce:
        # No DDP wrapper AT ALL on this path: the constructor would
        # allocate reducer bucket buffers (a full fp32 gradient copy,
        # measured 13.6 GiB on the molmo2 rung) even if the reducer
        # never syncs. Replicate the two things DDP provided — the
        # construction-time rank-0 state broadcast here, and the
        # per-step gradient sync via allreduce_gradients in the loop.
        broadcast_module_states(train_step)
    elif distributed:
        train_step = torch.nn.parallel.DistributedDataParallel(
            train_step,
            device_ids=[device.index] if device.type == "cuda" else None,
            # Backbone/decoder buffers are constant RoPE tables etc.;
            # per-step broadcasts would be pure overhead. The trainable
            # partition guarantees every grad-enabled parameter receives
            # gradients every step (frozen: the whole decoder; live backbone:
            # trainable_text_parameters mirrors kv_stop_layer), and the
            # graph never changes.
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            # Chunked backward runs n forwards + a no_sync accumulation
            # per step; the graph is still static per iteration, but the
            # static_graph fast path's interplay with no_sync is not a
            # risk worth carrying for a memory-fallback rung — plain DDP
            # is the well-trodden grad-accumulation path. Unchunked runs
            # keep the historical flag (and its recorded-graph perf).
            static_graph=args.backward_chunks == 1,
        )

    # Fixed MAE probe sets, independent of the training batch size, fetched
    # once and kept as CPU-resident collated batches per rank (collation
    # in-process is safe: dataloader workers are spawned, not forked; GPU
    # memory per eval is bounded by one batch, so probe size costs host RAM
    # only). eval_chunk_mae probes the held-out episodes (needs
    # --holdout-episodes > 0); train_mae probes the training split. Both
    # draws are what bijou.eval --episodes {holdout,train} --seed
    # <eval-seed> would score. Rank 0 keeps raw items strided across its
    # shard for the rich wandb table.
    eval_probe: ProbeSet | None = None
    train_probe: ProbeSet | None = None
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
                "train_args": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in dataclasses.asdict(args).items()
                },
                "decoder_config": decoder_schema_dict(model.decoder),
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
    # Loss/grad-norm live on-device between log points: a single .item()
    # sync per log_every steps instead of one per step. (grad_norm is
    # identical on all ranks after DDP's gradient sync — no reduce needed.)
    window: list[Tensor] = []
    window_action: list[Tensor] = []
    window_aux_sum: list[Tensor] = []
    window_aux_count: list[Tensor] = []
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
                # Chunked backward (--backward-chunks > 1): normalize
                # each chunk's sum-form loss by the FULL step's counts
                # (data-only, computed before any forward) and
                # accumulate gradients, syncing DDP only on the last
                # chunk — the accumulated gradient and every logged
                # quantity match the unchunked step up to fp reduction
                # order, at one chunk's activation footprint.
                step_normalizers: (
                    tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor, Tensor | None]
                )
                if model.joint_ce is not None:
                    # Joint arm: three full-step normalizers (flow
                    # elements, CE action tokens, CE aux positions);
                    # the logged action component is the FLOW loss, so
                    # its window division uses the flow count.
                    joint_counts = [
                        model.joint_loss_count_normalizers(c) for c in batch.chunks
                    ]
                    action_norm = torch.stack([c[0] for c in joint_counts]).sum()
                    ce_action_norm = torch.stack(
                        [c[1] for c in joint_counts],
                    ).sum()
                    ce_aux_norm = (
                        torch.stack(
                            [n for c in joint_counts if (n := c[2]) is not None],
                        ).sum()
                        if joint_counts[0][2] is not None
                        else None
                    )
                    step_normalizers = (action_norm, ce_action_norm, ce_aux_norm)
                else:
                    counts = [model.loss_count_normalizers(c) for c in batch.chunks]
                    action_norm = torch.stack([c[0] for c in counts]).sum()
                    aux_norm = (
                        torch.stack(
                            [n for c in counts if (n := c[1]) is not None],
                        ).sum()
                        if counts[0][1] is not None
                        else None
                    )
                    step_normalizers = (action_norm, aux_norm)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.zeros((), device=device)
                action_sum_total = torch.zeros((), device=device)
                aux_sum = None
                aux_count = None
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
                        share, share_action, share_aux, share_aux_count = train_step(
                            chunk,
                            normalizers=step_normalizers,
                        )
                        share.backward()
                    loss = loss + share.detach()
                    action_sum_total = action_sum_total + share_action
                    if share_aux is not None and share_aux_count is not None:
                        aux_sum = share_aux if aux_sum is None else aux_sum + share_aux
                        aux_count = (
                            share_aux_count
                            if aux_count is None
                            else aux_count + share_aux_count
                        )
                if distributed and args.chunk_grad_allreduce:
                    allreduce_gradients(clipped_parameters)
                action_component = action_sum_total / action_norm
            else:
                loss, action_component, aux_sum, aux_count = train_step(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                clipped_parameters,
                args.grad_clip,
            )
            if adamc_indices is not None:
                apply_adamc_weight_decay(optimizer, adamc_indices, args.weight_decay)
            optimizer.step()
            scheduler.step()
            step += 1
            window.append(loss.detach())
            window_action.append(action_component)
            if aux_sum is not None and aux_count is not None:
                window_aux_sum.append(aux_sum)
                window_aux_count.append(aux_count)

            if step % args.log_every == 0:
                # All ranks participate in the reduce (they hit the same
                # step in lockstep); only rank 0 syncs to host and reports.
                # Aux rides as (CE sum, position count) — the reduced
                # ratio is the position-weighted mean over the window
                # across all ranks, so sparsely-labeled batches weigh by
                # their actual aux tokens instead of diluting a
                # batch-mean toward 0. Aux runs append every step (0-sum
                # batches included), so the collective stays aligned.
                window_mean = torch.stack(window).mean()
                action_mean = torch.stack(window_action).mean()
                aux_totals = (
                    torch.stack(
                        [
                            torch.stack(window_aux_sum).sum(),
                            torch.stack(window_aux_count).sum().float(),
                        ],
                    )
                    if window_aux_sum
                    else None
                )
                window.clear()
                window_action.clear()
                window_aux_sum.clear()
                window_aux_count.clear()
                if distributed:
                    torch.distributed.all_reduce(window_mean)
                    window_mean /= world_size
                    torch.distributed.all_reduce(action_mean)
                    action_mean /= world_size
                    if aux_totals is not None:
                        torch.distributed.all_reduce(aux_totals)
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
                    if aux_totals is not None:
                        record["loss_action"] = round(action_mean.item(), 4)
                        if float(aux_totals[1]) > 0:
                            record["loss_aux"] = round(
                                float(aux_totals[0] / aux_totals[1]),
                                4,
                            )
                    if args.backbone_trained:
                        # Group 1 is the first backbone group (same cosine
                        # shape as the expert's, scaled to its base lr).
                        record["lr_backbone"] = scheduler.get_last_lr()[1]
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
                        if "loss_action" in record:
                            wandb_metrics["train/loss_action"] = record["loss_action"]
                        if "loss_aux" in record:
                            wandb_metrics["train/loss_aux"] = record["loss_aux"]
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
                        checkpoint_dir = args.save_dir / f"step_{step:06d}"
                        tensors = capture_checkpoint_tensors(
                            model,
                            args=args,
                            adapted_backbone_source=adapted_backbone_source,
                        )
                        metadata_json = build_checkpoint_metadata(
                            model,
                            args=args,
                            normalizers=normalizers,
                            per_dataset_stats=per_dataset_stats,
                            step=step,
                        )
                        # Captured now (copy_to_cpu doubles as a deep
                        # copy): a later scheduler.step() must not leak
                        # the next lr into this boundary's file.
                        scheduler_state = copy_to_cpu(scheduler.state_dict())

                        def write_async(
                            optimizer_state: dict[str, Any],
                            *,
                            _dir: Path = checkpoint_dir,
                            _tensors: CheckpointTensors = tensors,
                            _metadata: str = metadata_json,
                            _scheduler: dict[str, Any] = scheduler_state,
                            _step: int = step,
                            _started: float = capture_start,
                        ) -> Path:
                            path = write_checkpoint(
                                _dir,
                                tensors=_tensors,
                                metadata_json=_metadata,
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
                        args=args,
                        normalizers=normalizers,
                        per_dataset_stats=per_dataset_stats,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        adapted_backbone_source=adapted_backbone_source,
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


if __name__ == "__main__":
    try:
        sys.exit(main())
    except torch.OutOfMemoryError:
        # The whole point of the recording: bank the allocation history
        # BEFORE the process dies, one pickle per rank.
        prefix = os.environ.get("BIJOU_MEM_SNAPSHOT")
        if prefix:
            path = f"{prefix}_rank{os.environ.get('RANK', '0')}.pickle"
            torch.cuda.memory._dump_snapshot(path)
            print(f"mem-snapshot: OOM — dumped {path}", flush=True)
        raise
