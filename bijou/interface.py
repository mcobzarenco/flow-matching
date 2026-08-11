"""The observation-encoder ↔ action-decoder seam.

An encoder strategy turns one observation (instruction + camera frames
[+ state, eventually]) into an :class:`ObservationMemory`: named memory
streams a decoder cross-attends. The streams' static geometry (:class:`
StreamGeometry`) is declared by the encoder at construction time so a
decoder can size its query projections and RoPE behavior without knowing
what kind of backbone produced the memory (see ``docs/plan.md``).

Stream names are defined by the encoder (the Gemma backbone exports its
global layers' K/V as ``"kv{layer}"``); decoder schedules reference those
names, and composition validates the references (unknown name or unused
export = loud error).

The backbone network itself is owned by the composition root
(:class:`bijou.model.BijouModel`), not by the encoder: the encoder is
the prompt-side strategy (collation, prefix encode, unfreeze partition)
and receives the backbone as an argument — one network can serve several
roles (prefix encoder for cross-attention decoders; prefix + suffix
runner for the decoder-only path).
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Self, override

import torch
from torch import Tensor, nn

from .annotations import ConditionField
from .aux_text import (
    IMAGE_KEY_PREFIX,
    AuxField,
    AuxGeneration,
    AuxSpec,
    assemble_suffix,
    camera_prompt_order,
    generate_text,
    subgoal_text,
)
from .fast.codec import ActionCodec
from .nn import RopeParameters


class SamplingMethod(Enum):
    """ODE solver for integrating a flow decoder's velocity field — part
    of the decoder-facing API surface (it rides ``predict_chunk``), and
    convention-NEUTRAL: "Euler"/"Heun" mean the same thing under either
    time parametrization, so both flow decoders (``decoders/flow.py``'s
    π 0 descent, ``decoders/molmo_flow.py``'s ascending integration)
    share this ONE name. The loops themselves stay per-module — they
    carry the mirrored conventions and independent byte-parity
    obligations (§8.13 decision 2).

    EULER: 1 model evaluation per step, first-order (global error
    O(1/n)); molmo_flow's serving default (their reference loop).
    HEUN: explicit trapezoidal predictor-corrector, 2 evaluations per
    step, second-order (O(1/n²)); the better quality-per-evaluation
    trade for all but the very smallest step counts (Karras et al.,
    EDM); flow.py's default.
    """

    EULER = "euler"
    HEUN = "heun"


def kv_stream_name(layer_idx: int) -> str:
    """The Gemma backbone's stream-naming convention: K/V of backbone layer
    ``layer_idx`` is exported as ``"kv{layer_idx}"``. Lives at module
    level because producer (observation encode) and consumer (the flow
    decoder's int-schedule config) both spell it."""
    return f"kv{layer_idx}"


def residual_stream_name(layer_idx: int) -> str:
    """Residual-stream conditioning naming: the hidden state AFTER backbone
    layer ``layer_idx`` travels as the raw tap ``"res{layer_idx}"``; the
    flow decoder's learned adapter projects it into a MemoryStream of the
    same name (arch-batch-1 arm B, 2026-08-06)."""
    return f"res{layer_idx}"


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
class BijouPrediction:
    """Everything the model predicts for one observation batch, crossing
    the seam back to the caller: one action chunk per sample (RAW units
    — the field mirrors ``CollatedBatch.actions``, the ground truth it
    is scored against) plus, for decoders with a text surface
    (ar_backbone), one :class:`AuxGeneration` per row. ``None``
    generations = this decoder kind produces no text (flow, ar_fast);
    ar_backbone always returns the list — rows are empty-text under ACT
    decode.

    ``noise`` is the initial noise the flow solver actually integrated
    (supplied or drawn), kept so a paired re-decode can reuse it — the
    Q3 conditioning tripwire needs |Δ| against the SAME draw, or the
    sampling variance floors the signal for a conditioning-blind model.
    None for decoders that draw none (ar_fast, ar_backbone).

    Shapes: actions [B, chunk, action_dim] (raw action units);
    noise [B, chunk, action_dim] (normalized units)."""

    actions: Tensor
    generations: list[AuxGeneration] | None
    noise: Tensor | None = None


@dataclass(frozen=True, slots=True)
class ObservationMemory:
    """The value crossing the encoder → decoder seam: named memory streams
    (insertion-ordered as the encoder's exports) plus the (padded) memory
    width P and, for padded batches, the True-means-real padding mask
    [B, P]. Per-sample real lengths (decoder query position bases) derive
    from the mask; ``length`` is the KV width and the position base only
    for unpadded batches.

    ``cache`` is the full prefix KV cache the encode produced — every
    non-KV-shared layer's K/V, of which the named streams are zero-copy
    views — retained only when the decoder consumes the whole prefix
    state (the decoder-only backbone path continues the suffix through it);
    None for stream-consuming decoders, freeing the non-exported layers.
    Its concrete type is a TRUNK-private contract between the producing
    encoder and the decoder that continues the trunk (the Gemma path's
    ``gemma4.cache.KVCache``) — opaque at this seam so the seam depends
    on no trunk. Consumers check for None and isinstance-narrow to their
    trunk's cache type, failing fast on either mismatch.

    ``residuals`` are RAW residual-stream taps (hidden state after backbone
    layer i, [B, P, backbone_hidden], keyed ``res{i}``) — exported by the
    encoder but NOT yet conditioning streams: the flow decoder's learned
    adapters project them into ``streams`` entries of the same name
    (FlowDecoder.attach_residual_streams), OUTSIDE the possibly-no-grad
    prefix encode so the adapters train under a frozen backbone. None once
    attached (or when the model has no residual conditioning)."""

    streams: dict[str, MemoryStream]
    length: int
    padding_mask: Tensor | None
    cache: object | None = None
    residuals: dict[str, Tensor] | None = None

    @property
    def batch_size(self) -> int:
        if self.streams:
            return next(iter(self.streams.values())).key.shape[0]
        assert self.residuals is not None  # one of the two always exists
        return next(iter(self.residuals.values())).shape[0]


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
    backbone-agnostic action-chunk targets and per-sample stats.

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
    # exclusions derive from it, and causal attention plus PAD-position
    # loss masking hide it from real positions.
    action_tokens: Tensor | None
    # Aux-augmented suffix in BACKBONE id space ([aux text ids]
    # [block_base+BOA..actions], block-PAD-padded) + the aux-position
    # mask, present iff the Collator was built with an AuxSpec
    # (ar_backbone aux training). None keeps every non-aux path
    # byte-identical.
    suffix_tokens: Tensor | None
    suffix_is_aux: Tensor | None

    def all_tensors(self) -> list[Tensor]:
        """Every tensor in the batch, nested fields included (stream-sync
        bookkeeping walks these after async H2D copies)."""
        return [
            self.state,
            self.actions,
            self.action_is_pad,
            *([self.action_tokens] if self.action_tokens is not None else []),
            *([self.suffix_tokens] if self.suffix_tokens is not None else []),
            *([self.suffix_is_aux] if self.suffix_is_aux is not None else []),
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
    """One camera's frame with its slot name (post-filter — e.g.
    "front", "wrist"; community datasets carry generic image/image2 names
    with no reliable semantics, rig datasets carry real ones) and its
    semantic KIND (judge camera-kind vocabulary; "unknown" when the
    dataset is unjudged, the camera is untagged, or kind dropout fired).
    Slot ORDER is (raw kind, short name) — camera_prompt_order; dropout
    retags but never reorders. Encoder strategies render the kind
    (camera tags) or ignore both."""

    name: str
    kind: str
    image: Tensor  # [3, height, width], float, [0, 1]


@dataclass(frozen=True, slots=True)
class PromptInputs:
    """One sample's prompt-side payload, assembled by the shared Collator
    (instruction override + camera policy + conditioning applied).
    ``condition_text`` is the bracket block between the cameras and the
    closing instruction copy — value conditioning plus the always-
    present ``[generate|…]`` request (prompt format 3), e.g.
    ``"[outcome|success][generate|subgoal actions]"`` — already
    rendered, so encoder strategies just place it. ``state`` is the
    NORMALIZED proprioceptive vector for the prompt's trailing soft
    state token (per-sample stats applied at collation)."""

    instruction: str
    cameras: tuple[CameraFrame, ...]
    condition_text: str
    state: Tensor  # [state_dim], normalized


class InputsCollator[I: BatchInputs](Protocol):
    """Encoder-specific half of collation: a batch of PromptInputs → the
    encoder's input tensors. Implementations are pickled into spawned
    dataloader workers — build heavy processors lazily and drop them in
    ``__getstate__``."""

    def __call__(self, samples: list[PromptInputs]) -> I: ...


class ObservationEncoder[I: BatchInputs, B: nn.Module](nn.Module, abc.ABC):
    """ABC of a trunk's prompt-side strategy (``docs/plan.md``): the
    inputs-collation strategy, the prefix encode, and the trunk's
    unfreeze surface. Generic over its collated-inputs type ``I`` and
    its trunk type ``B`` — the composition root (BijouModel) owns the
    trunk network once and passes it into the compute methods, pairing
    trunk and encoder consistently by construction.

    The module carries exactly the PROMPT-side parameters (e.g. the
    Gemma strategy's soft-state projection), never trunk ones — trunk
    subsets are exposed through :meth:`param_groups` instead, so the
    root can route component learning rates without owning the split."""

    @abc.abstractmethod
    def stream_geometries(self) -> dict[str, StreamGeometry]:
        """Static geometry per stream name; keys and order match every
        ObservationMemory this encoder produces. Names are the encoder's
        vocabulary (``kv{i}``/``res{i}`` today); decoder schedules
        reference them and composition validates the references."""

    @abc.abstractmethod
    def inputs_collator(self) -> InputsCollator[I]:
        """The encoder-specific half of collation (pickleable into
        spawned dataloader workers)."""

    @abc.abstractmethod
    def encode(
        self,
        backbone: B,
        inputs: I,
        *,
        with_grad: bool,
        retain_cache: bool = False,
    ) -> ObservationMemory:
        """Run the trunk over one collated batch's multimodal prefix and
        export the memory streams. ``with_grad=False`` runs under
        no_grad (eval/rollout/frozen training); True leaves autograd on
        for live-trunk training. ``retain_cache`` keeps the trunk's full
        prefix cache on the returned memory (ObservationMemory.cache)
        for decoders that continue the trunk through it."""

    @abc.abstractmethod
    def param_groups(self, backbone: B) -> dict[str, list[nn.Parameter]]:
        """Named unfreezable trunk subsets (e.g. ``"text"``/``"vision"``)
        — the component-lr flags route here. Groups must be EXACT (only
        parameters that participate in a forward): DDP requires every
        grad-enabled parameter to receive gradients each step."""


# Action decoders are plain nn.Modules; the composition contract lives in
# BijouModel's concrete union (exhaustive match dispatch — pyright-gated),
# not an ABC: the decoders' signatures legitimately differ (the flow
# decoder's forward is a velocity field with solver knobs; the ar_backbone
# decoder runs against the backbone itself), and every consumer goes
# through the root. Shared conventions: training objectives are
# module-level functions beside each decoder (``flow_matching_loss``,
# ``ar_fast_loss``, ``ar_backbone_loss``), and ``predict_chunk`` returns
# RAW-unit chunks [B, chunk, action_dim] (per-sample stats applied inside).

_IMAGE_KEY_PREFIX = IMAGE_KEY_PREFIX


def mask_state_item(item: dict[str, Any]) -> dict[str, Any]:
    """Replace the item's proprioceptive state with its dataset's state
    mean, so the normalized soft state token collates to EXACTLY zero
    (x − x ≡ 0 bitwise) — the prompt stays well-formed but carries zero
    state information. The ONE masking primitive, shared by the
    eval-side reliance probe (``--mask-state``) and the train-time
    ``--state-dropout`` regularizer so their semantics can never drift.
    Items are rebuilt, never mutated: baselines (state-copy is the
    intact-state reference) and the truth actions see the originals."""
    return {**item, "observation.state": item["state_mean"].clone()}


@dataclass
class Collator[I: BatchInputs]:
    """The ONE backbone-agnostic collator: stacks state/actions/targets,
    attaches per-sample NormStats, applies the camera-selection policy and
    the instruction override, and delegates encoder-input production to the
    encoder's strategy. Never subclassed per encoder.

    Camera kinds travel WITH the items (``item["camera_kinds"]``: short
    name → semantic kind, attached per dataset by StatsAttachedDataset —
    the same convention as the stats — or handed to
    ``rollout.observation_to_item`` as a plain per-camera dict); a
    missing map or camera ⇒ "unknown". ``camera_kind_dropout`` replaces
    a resolved kind with "unknown" per camera per visit (train-time
    regularizer — inference on unjudged rigs stays in-distribution; the
    probe-side collator runs a dropout-0 clone). Draws come from a
    per-process generator seeded from ``torch.initial_seed()`` (the
    AuxSpec convention)."""

    inputs: InputsCollator[I]
    instruction: str | None
    camera_filter: tuple[str, ...] | None
    max_cameras: int | None
    action_codec: ActionCodec | None
    # Aux text rendering (ar_backbone only); requires action_codec.
    aux: AuxSpec | None
    # Render the [generate|…] request bracket (ar_backbone prompts —
    # aux or not: [generate|actions] is the fast path and is ALWAYS
    # present for that family; other decoders never see it).
    generate_bracket: bool
    # Inference/probe override: force THIS request on every row's
    # prompt (template order, 'actions' implied) instead of the per-row
    # training draw; suffix targets are then not assembled (decode-only
    # batches). () = the deployment fast path.
    generate_override: tuple[AuxField, ...] | None
    camera_kind_dropout: float
    # With probability p, swap the recorded task string for a uniformly
    # drawn judge-suggested rewrite (item["suggested_instructions"],
    # attached per episode by StatsAttachedDataset) — phrasing-diversity
    # augmentation. The CLI --instruction override always wins; probes
    # run an augment-0 clone (evals score the recorded instruction).
    instruction_augment: float
    # Prompt conditioning (§C1/C2): fields rendered as the user turn's
    # trailing bracket block — subgoal from the frame's segment label
    # (or an explicit item["condition_subgoal"] override: planner/CLI),
    # outcome/smoothness from the item's hindsight labels; None renders
    # nothing. Per-field dropout keeps the unconditioned marginal
    # trained; the subgoal hint has its OWN rate (deployment mostly
    # runs without a planner, so unconditioned must stay the
    # well-trained context). Probes run dropout-0 clones = TRUE-label
    # conditioning (score against truth ⇒ condition on truth;
    # deployment asks for what it wants).
    condition_fields: tuple[ConditionField, ...]
    condition_dropout: float
    subgoal_condition_dropout: float
    # With probability p per sample, mask proprioceptive state to the
    # dataset mean (``mask_state_item`` — the reliance probe's exact
    # semantics: normalized token ≡ 0): the anti-shortcut regularizer
    # from the causal-confusion line (arXiv:2506.23944). Train-time
    # only; probes and inference score intact state (dropout-0 clones).
    # Default 0.0 so every non-train construction site stays inert, and
    # p=0 draws nothing from the RNG (existing streams byte-identical).
    state_dropout: float = 0.0
    _generator: torch.Generator | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.camera_kind_dropout < 1.0:
            raise ValueError(
                f"camera kind dropout {self.camera_kind_dropout} outside [0, 1)",
            )
        if not 0.0 <= self.instruction_augment <= 1.0:
            raise ValueError(
                f"instruction augment {self.instruction_augment} outside [0, 1]",
            )
        ordered = tuple(f for f in ConditionField if f in self.condition_fields)
        if ordered != self.condition_fields:
            raise ValueError(
                f"condition fields must keep template order "
                f"{[f.value for f in ConditionField]}; got "
                f"{[f.value for f in self.condition_fields]}",
            )
        if not 0.0 <= self.condition_dropout < 1.0:
            raise ValueError(
                f"condition dropout {self.condition_dropout} outside [0, 1)",
            )
        if not 0.0 <= self.subgoal_condition_dropout < 1.0:
            raise ValueError(
                f"subgoal condition dropout {self.subgoal_condition_dropout} "
                "outside [0, 1)",
            )
        if not 0.0 <= self.state_dropout < 1.0:
            raise ValueError(
                f"state dropout {self.state_dropout} outside [0, 1)",
            )
        if self.aux is not None and (
            self.camera_filter is not None or self.max_cameras is not None
        ):
            raise ValueError(
                "aux rendering with camera selection: the 'visible' "
                "field's indices are positions in the FULL sorted camera "
                "set — camera_filter/max_cameras would silently shift "
                "them; train aux without camera selection",
            )
        if self.generate_override is not None:
            if not self.generate_bracket:
                raise ValueError(
                    "generate_override requires generate_bracket — the "
                    "override is rendered INTO the prompt's [generate|…]",
                )
            ordered = tuple(f for f in AuxField if f in self.generate_override)
            if ordered != self.generate_override:
                raise ValueError(
                    f"generate_override must keep template order; got "
                    f"{[f.value for f in self.generate_override]}",
                )
        if self.aux is not None and not self.generate_bracket:
            raise ValueError(
                "aux rendering without the generate bracket: the request "
                "set IS prompt conditioning — they ship together",
            )

    @override
    def __getstate__(self) -> dict[str, Any]:
        # Generators don't pickle; spawned workers re-seed lazily.
        return {**self.__dict__, "_generator": None}

    def _rng(self) -> torch.Generator:
        if self._generator is None:
            self._generator = torch.Generator().manual_seed(torch.initial_seed())
        return self._generator

    def _camera_kind(self, item: dict[str, Any], camera: str) -> str:
        kind = (item.get("camera_kinds") or {}).get(camera, "unknown")
        if (
            kind != "unknown"
            and self.camera_kind_dropout > 0.0
            and float(torch.rand((), generator=self._rng())) < self.camera_kind_dropout
        ):
            return "unknown"
        return kind

    def _condition_text(self, item: dict[str, Any]) -> tuple[str, bool]:
        """The user turn's value-conditioning block, ``[key|value]`` per
        configured field whose value exists and survives its dropout
        draw (bracket-delimited — the chat template trims text-part
        edge whitespace; the [generate|…] request is appended by
        __call__ after the aux draw), plus whether the SUBGOAL rendered
        — the aux draw suppresses the subgoal field then (anti-copy
        coupling)."""
        parts: list[str] = []
        subgoal_rendered = False
        for field_name in self.condition_fields:
            if field_name is ConditionField.SUBGOAL:
                # Explicit None handling, no truthiness: an operator's
                # EMPTY-string override means "no hint" and must not
                # fall through to the frame label it suppresses (the
                # `or` form did exactly that, and KeyError'd on
                # label-less rollout items; 2026-08-03).
                override = item.get("condition_subgoal")
                if override is None:
                    value = subgoal_text(item)
                elif override.strip() == "":
                    value = None
                else:
                    value = override
                dropout = self.subgoal_condition_dropout
            else:
                value = item.get(f"condition_{field_name.value}")
                dropout = self.condition_dropout
            if value is None:
                continue
            if dropout > 0.0 and (
                float(torch.rand((), generator=self._rng())) < dropout
            ):
                continue
            if field_name is ConditionField.SUBGOAL:
                subgoal_rendered = True
            parts.append(f"[{field_name.value}|{value}]")
        return "".join(parts), subgoal_rendered

    def _instruction(self, item: dict[str, Any]) -> str:
        """The prompt instruction for one item: CLI override > sampled
        judge rewrite (probability ``instruction_augment``, uniform over
        the episode's suggestions) > the recorded task string. Both
        sandwich copies receive the same string by construction."""
        if self.instruction is not None:
            return self.instruction
        recorded = str(item["task"])
        if self.instruction_augment > 0.0:
            suggestions = tuple(item.get("suggested_instructions") or ())
            if suggestions and (
                float(torch.rand((), generator=self._rng())) < self.instruction_augment
            ):
                pick = int(
                    torch.randint(len(suggestions), (), generator=self._rng()),
                )
                return suggestions[pick]
        return recorded

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
        """Camera keys of one sample in PROMPT order: sorted by
        (semantic kind, short name) via :func:`camera_prompt_order` —
        RAW kinds from the item's map, so same-kind rigs order
        identically regardless of their private short names, and a
        kind-dropout draw (applied to the TAG TEXT downstream) can
        never reorder images. Untagged datasets degenerate to plain
        short-name order (the community collections' generic
        image/image2 keys carry no reliable wrist-vs-scene semantics —
        SmolVLA precedent)."""
        kinds = item.get("camera_kinds") or {}
        short_names = [
            k.removeprefix(_IMAGE_KEY_PREFIX)
            for k in item
            if k.startswith(_IMAGE_KEY_PREFIX)
        ]
        cameras = [
            _IMAGE_KEY_PREFIX + name
            for name in camera_prompt_order(kinds, sorted(short_names))
        ]
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
        if self.state_dropout > 0.0:
            # Item-level rewrite so EVERY downstream state consumer —
            # the normalized prompt token below AND the raw
            # CollatedBatch.state the decoder conditions on — sees the
            # masked value; actions/targets are untouched.
            items = [
                mask_state_item(item)
                if float(torch.rand((), generator=self._rng())) < self.state_dropout
                else item
                for item in items
            ]
        samples: list[PromptInputs] = []
        aux_rows: list[list[int]] = []
        for item in items:
            condition_text, subgoal_rendered = self._condition_text(item)
            # The request draw feeds BOTH surfaces: the prompt's
            # [generate|…] bracket and the suffix targets — one draw,
            # always consistent (request ⊆ labeled by construction).
            if self.generate_override is not None:
                request: tuple[AuxField, ...] = self.generate_override
                aux_rows.append([])
            elif self.aux is not None:
                request, ids = self.aux.draw(
                    item,
                    suppress_subgoal=subgoal_rendered,
                )
                aux_rows.append(ids)
            else:
                request = ()
                aux_rows.append([])
            if self.generate_bracket:
                condition_text += generate_text(request)
            state = item["observation.state"]
            samples.append(
                PromptInputs(
                    instruction=self._instruction(item),
                    condition_text=condition_text,
                    cameras=tuple(
                        CameraFrame(
                            name=(name := key.removeprefix(_IMAGE_KEY_PREFIX)),
                            kind=self._camera_kind(item, name),
                            image=item[key],
                        )
                        for key in self.cameras_of(item)
                    ),
                    state=(state - item["state_mean"]) / item["state_std"],
                ),
            )
        action_tokens = self._action_tokens(items)
        suffix_tokens: Tensor | None = None
        suffix_is_aux: Tensor | None = None
        if self.aux is not None and self.generate_override is None:
            if action_tokens is None or self.action_codec is None:
                raise ValueError(
                    "aux rendering requires an ActionCodec (aux rides the "
                    "AR suffix) — build the Collator with both or neither",
                )
            suffix_tokens, suffix_is_aux = assemble_suffix(
                aux_rows,
                action_tokens,
                block_base=self.aux.block_base,
                codec_pad=self.action_codec.pad,
            )
        return CollatedBatch(
            encoder_inputs=self.inputs(samples),
            state=torch.stack([item["observation.state"] for item in items]),
            actions=torch.stack([item["action"] for item in items]),
            action_is_pad=torch.stack([item["action_is_pad"] for item in items]),
            action_stats=self._stats(items, "action"),
            state_stats=self._stats(items, "state"),
            action_tokens=action_tokens,
            suffix_tokens=suffix_tokens,
            suffix_is_aux=suffix_is_aux,
        )
