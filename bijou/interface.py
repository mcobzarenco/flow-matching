"""The observation-encoder ↔ action-decoder seam.

An encoder strategy turns one observation (instruction + camera frames
[+ state, eventually]) into an :class:`ObservationMemory`: named memory
streams a decoder cross-attends. The streams' static geometry (:class:`
StreamGeometry`) is declared by the encoder at construction time so a
decoder can size its query projections and RoPE behavior without knowing
what kind of trunk produced the memory (see ``docs/plan.md``).

Stream names are defined by the encoder (the Gemma trunk exports its
global layers' K/V as ``"kv{layer}"``); decoder schedules reference those
names, and composition validates the references (unknown name or unused
export = loud error).

The trunk network itself is owned by the composition root
(:class:`bijou.model.BijouModel`), not by the encoder: the encoder is
the prompt-side strategy (collation, prefix encode, unfreeze partition)
and receives the trunk as an argument — one network can serve several
roles (prefix encoder for cross-attention decoders; prefix + suffix
runner for the decoder-only path).
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, Self

import torch
from torch import Tensor, nn

from .fast.codec import ActionCodec
from .nn import RopeParameters


def kv_stream_name(layer_idx: int) -> str:
    """The Gemma trunk's stream-naming convention: K/V of backbone layer
    ``layer_idx`` is exported as ``"kv{layer_idx}"``. Shared by the
    producer (observation encode) and the consumer (the flow decoder's
    int-schedule config) until the encoder abstraction owns it."""
    return f"kv{layer_idx}"


@dataclass(frozen=True, slots=True)
class StreamGeometry:
    """Static per-stream contract, known at construction time.

    ``rope`` set: keys arrive position-encoded and the decoder must RoPE
    its queries at positions ≥ the per-sample real memory width (the
    Gemma streams' contract). ``rope`` None: positions are baked into the
    memory (e.g. an adapter with learned positions); the decoder applies
    no query RoPE for this stream.
    """

    kv_heads: int
    head_dim: int
    rope: RopeParameters | None


@dataclass(frozen=True, slots=True)
class MemoryStream:
    """One exported stream's K/V.

    Shapes: key/value [B, kv_heads, P, head_dim]."""

    key: Tensor
    value: Tensor


@dataclass(frozen=True, slots=True)
class ObservationMemory:
    """The value crossing the encoder → decoder seam: named memory streams
    (insertion-ordered as the encoder's exports) plus the (padded) memory
    width P and, for padded batches, the True-means-real padding mask
    [B, P]. Per-sample real lengths (decoder query position bases) derive
    from the mask; ``length`` is the KV width and the position base only
    for unpadded batches."""

    streams: dict[str, MemoryStream]
    length: int
    padding_mask: Tensor | None

    @property
    def batch_size(self) -> int:
        first = next(iter(self.streams.values()))
        return first.key.shape[0]


class BatchInputs(Protocol):
    """What encoder-specific batch inputs must support: the transfer hooks
    CollatedBatch delegates to (DataLoader pinning and H2D moves) plus a
    tensor walk (stream-sync bookkeeping in DevicePrefetcher)."""

    def pin_memory(self) -> Self: ...

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Self: ...

    def tensors(self) -> dict[str, Tensor]: ...


def _replace_tensors[T](value: T, transform: Any) -> T:
    """dataclasses.replace with ``transform`` applied to every Tensor field
    (shared by the batch dataclasses' pin_memory/to hooks)."""
    return dataclasses.replace(
        value,  # type: ignore[type-var]  # dataclass-typed by every caller
        **{
            field.name: transform(attr)
            for field in dataclasses.fields(value)  # type: ignore[arg-type]
            if isinstance(attr := getattr(value, field.name), Tensor)
        },
    )


@dataclass(frozen=True, slots=True)
class NormStats:
    """One modality's normalization stats, per sample: each tensor [B, dim]
    (dim = action_dim or state_dim). Every sample carries its OWN dataset's
    stats — per-dataset normalization; nothing here is aggregated across
    the batch.

    ``q01``/``q99`` are None only when the stats were resolved from a
    checkpoint whose tables predate quantiles (old-checkpoint rollout);
    batches built from datasets always carry them (selection requires
    backfilled stats). Consumers that need quantiles check for None and
    fail fast; the flow path never reads them."""

    mean: Tensor
    std: Tensor
    q01: Tensor | None
    q99: Tensor | None

    def tensors(self) -> dict[str, Tensor]:
        return {
            field.name: attr
            for field in dataclasses.fields(self)
            if isinstance(attr := getattr(self, field.name), Tensor)
        }

    def pin_memory(self) -> NormStats:
        return _replace_tensors(self, lambda t: t.pin_memory())

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> NormStats:
        return _replace_tensors(self, lambda t: t.to(device, non_blocking=non_blocking))


@dataclass(frozen=True, slots=True)
class CollatedBatch[I: BatchInputs]:
    """One collated batch: encoder-specific inputs plus the
    trunk-agnostic action-chunk targets and per-sample stats.

    ``state``/``actions`` are raw (unnormalized); ``action_is_pad`` marks
    positions past the episode end, where the value is the last real
    action repeated (see flow_matching_loss)."""

    encoder_inputs: I
    state: Tensor  # [B, state_dim]
    actions: Tensor  # [B, chunk, action_dim]
    action_is_pad: Tensor  # [B, chunk]  (bool)
    action_stats: NormStats  # each [B, action_dim]
    state_stats: NormStats  # each [B, state_dim]
    # FAST token ids [B, T_tok] ([BOA, t_1..t_k] per ActionCodec.encode,
    # PAD-padded to the batch max — no EOA: sequence length is fixed by
    # the FAST grammar), present iff the Collator was built with an
    # ActionCodec (AR decoders). No separate mask: PAD is a reserved id,
    # exclusions derive from it, and right padding + causal attention
    # hides PAD from real positions.
    action_tokens: Tensor | None

    def all_tensors(self) -> list[Tensor]:
        """Every tensor in the batch, nested fields included (stream-sync
        bookkeeping walks these after async H2D copies)."""
        return [
            self.state,
            self.actions,
            self.action_is_pad,
            *([self.action_tokens] if self.action_tokens is not None else []),
            *self.action_stats.tensors().values(),
            *self.state_stats.tensors().values(),
            *self.encoder_inputs.tensors().values(),
        ]

    def pin_memory(self) -> CollatedBatch[I]:
        """Called by the DataLoader when ``pin_memory=True`` (torch supports
        custom batch types via this hook); pinned memory makes the H2D
        copies in DevicePrefetcher truly asynchronous. ``_replace_tensors``
        covers action_tokens (a direct Tensor field) in both hooks."""
        moved = _replace_tensors(self, lambda t: t.pin_memory())
        return dataclasses.replace(
            moved,
            encoder_inputs=self.encoder_inputs.pin_memory(),
            action_stats=self.action_stats.pin_memory(),
            state_stats=self.state_stats.pin_memory(),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> CollatedBatch[I]:
        moved = _replace_tensors(
            self,
            lambda t: t.to(device, non_blocking=non_blocking),
        )
        return dataclasses.replace(
            moved,
            encoder_inputs=self.encoder_inputs.to(device, non_blocking=non_blocking),
            action_stats=self.action_stats.to(device, non_blocking=non_blocking),
            state_stats=self.state_stats.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One camera's frame with its slot name (post-filter, sorted — e.g.
    "front", "wrist"; community datasets carry generic image/image2 names
    with no reliable semantics, rig datasets carry real ones). Encoder
    strategies MAY render the name into the prompt or ignore it (the
    Gemma collator's positional behavior)."""

    name: str
    image: Tensor  # [3, height, width], float, [0, 1]


@dataclass(frozen=True, slots=True)
class PromptInputs:
    """One sample's prompt-side payload, assembled by the shared Collator
    (instruction override + camera policy applied). Extension point for
    future prompt-side signals (e.g. discretized state as text)."""

    instruction: str
    cameras: tuple[CameraFrame, ...]


class InputsCollator[I: BatchInputs](Protocol):
    """Encoder-specific half of collation: a batch of PromptInputs → the
    encoder's input tensors. Implementations are pickled into spawned
    dataloader workers — build heavy processors lazily and drop them in
    ``__getstate__``."""

    def __call__(self, samples: list[PromptInputs]) -> I: ...


class ActionDecoder(nn.Module, ABC):
    """An action decoder: chunk-space inference over an observation memory.
    Training objectives are module-level functions beside each decoder
    (``flow_matching_loss``, ``ar_fast_loss``), dispatched by
    ``BijouModel.loss`` — the decoder classes own parameters and forwards,
    not objectives. ``forward`` stays decoder-specific (the flow decoder's
    is the velocity field)."""

    @abstractmethod
    def predict_chunk(
        self,
        memory: ObservationMemory,
        batch: CollatedBatch[Any],
        *,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """RAW-unit action chunk [B, chunk, action_dim]; normalization via
        the batch's per-sample stats happens inside."""


_IMAGE_KEY_PREFIX = "observation.images."


@dataclass
class Collator[I: BatchInputs]:
    """The ONE trunk-agnostic collator: stacks state/actions/targets,
    attaches per-sample NormStats, applies the camera-selection policy and
    the instruction override, and delegates encoder-input production to the
    encoder's strategy. Never subclassed per encoder."""

    inputs: InputsCollator[I]
    instruction: str | None
    camera_filter: tuple[str, ...] | None
    max_cameras: int | None
    action_codec: ActionCodec | None

    def _action_tokens(self, items: list[dict[str, Any]]) -> Tensor | None:
        """Tokenize each item's action chunk (worker-side CPU), PAD-pad to
        the batch max. Quantiles are the tokenizer-fit normalization and
        are required — items resolved from an old checkpoint's stats table
        cannot feed an AR decoder."""
        codec = self.action_codec
        if codec is None:
            return None
        sequences: list[list[int]] = []
        for item in items:
            if "action_q01" not in item:
                raise SystemExit(
                    f"item from {item.get('repo_id', '<unknown>')} carries no "
                    "action quantiles — AR tokenization needs the exact "
                    "q01/q99 the tokenizer was fitted under (backfilled "
                    "dataset stats; old checkpoint stats tables cannot "
                    "drive AR training)",
                )
            sequences.append(
                codec.encode(
                    item["action"].numpy(),
                    item["action_q01"].numpy(),
                    item["action_q99"].numpy(),
                ),
            )
        width = max(len(s) for s in sequences)
        return torch.tensor(
            [s + [codec.pad] * (width - len(s)) for s in sequences],
            dtype=torch.long,
        )

    def cameras_of(self, item: dict[str, Any]) -> list[str]:
        """Sorted camera keys of one sample; prompt slots are positional (the
        community collections' generic image/image2 keys carry no reliable
        wrist-vs-scene semantics — SmolVLA precedent)."""
        cameras = sorted(k for k in item if k.startswith(_IMAGE_KEY_PREFIX))
        if self.camera_filter is not None:
            allowed = set(self.camera_filter)
            cameras = [
                k
                for k in cameras
                if k in allowed or k.removeprefix(_IMAGE_KEY_PREFIX) in allowed
            ]
        if not cameras:
            raise ValueError(
                f"sample has no cameras after filtering ({self.camera_filter=})",
            )
        if self.max_cameras is not None:
            cameras = cameras[: self.max_cameras]
        return cameras

    def _stats(self, items: list[dict[str, Any]], modality: str) -> NormStats:
        """Stack one modality's per-item stats tensors; quantiles are all
        present (dataset items) or all absent (items built from an
        old checkpoint's stats table) — a mixed batch is a wiring bug."""
        with_quantiles = [f"{modality}_q01" in item for item in items]
        if any(with_quantiles) and not all(with_quantiles):
            raise ValueError(
                f"batch mixes items with and without {modality} quantile "
                "stats — items from datasets always carry them, items from "
                "an old checkpoint's stats table never do; do not mix",
            )
        return NormStats(
            mean=torch.stack([item[f"{modality}_mean"] for item in items]),
            std=torch.stack([item[f"{modality}_std"] for item in items]),
            q01=(
                torch.stack([item[f"{modality}_q01"] for item in items])
                if all(with_quantiles)
                else None
            ),
            q99=(
                torch.stack([item[f"{modality}_q99"] for item in items])
                if all(with_quantiles)
                else None
            ),
        )

    def __call__(self, items: list[dict[str, Any]]) -> CollatedBatch[I]:
        """Collate ``B = len(items)`` LeRobot items. Per-item inputs consumed
        here:
          - observation.images.*: [3, height, width] each  (float, [0, 1])
          - observation.state: [state_dim]
          - action: [chunk, action_dim]
          - action_is_pad: [chunk]  (bool)
          - action_/state_{mean,std[,q01,q99]}: [action_dim]/[state_dim]
            (attached by StatsAttachedDataset)
          - task: str  (overridden by ``instruction`` when set)
        """
        samples = [
            PromptInputs(
                instruction=self.instruction or str(item["task"]),
                cameras=tuple(
                    CameraFrame(
                        name=key.removeprefix(_IMAGE_KEY_PREFIX),
                        image=item[key],
                    )
                    for key in self.cameras_of(item)
                ),
            )
            for item in items
        ]
        return CollatedBatch(
            encoder_inputs=self.inputs(samples),
            state=torch.stack([item["observation.state"] for item in items]),
            actions=torch.stack([item["action"] for item in items]),
            action_is_pad=torch.stack([item["action_is_pad"] for item in items]),
            action_stats=self._stats(items, "action"),
            state_stats=self._stats(items, "state"),
            action_tokens=self._action_tokens(items),
        )
