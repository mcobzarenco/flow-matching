"""The observation-encoder ↔ action-decoder seam.

An encoder strategy turns one observation (instruction + camera frames
[+ state, eventually]) into its trunk's PER-TRUNK memory value — the
dataclass defined beside the encoder (``encoders/gemma4.GemmaMemory``:
named memory streams a decoder cross-attends, plus the prefix cache
where a suffix decoder continues it; ``encoders/molmo2.Molmo2Memory``:
the typed prefix cache, the whole product). Cache types are static on
those values — a decoder states its trunk's memory in its signature and
the wrong pairing is a type error, not a runtime narrow.

Encoders are plain nn.Modules — a CONVENTION, not a base class (the
composition rule): each trunk's prompt-side strategy exposes
``inputs_collator()`` (the pickleable collation half, an
:class:`InputsCollator`), ``encode(backbone, inputs, *, with_grad, …)``
producing its memory value, and ``param_groups(backbone)``
(named unfreezable trunk subsets — EXACT sets: DDP requires every
grad-enabled parameter to receive gradients each step). The module
carries exactly the PROMPT-side parameters (e.g. the Gemma strategy's
soft-state projection), never trunk ones; every consumer reaches the
encoder through its family's trait surface, so conformance is the
family's construction, not a subtype check.

Stream names are defined by the encoder (the Gemma backbone exports its
global layers' K/V as ``"kv{layer}"``); decoder schedules reference those
names, and composition validates the references (unknown name or unused
export = loud error).

The backbone network itself is owned by the model family (the concrete
:class:`bijou.vla.VLA` classes in ``bijou.models``), not by the encoder:
the encoder receives the backbone as an argument — one network can
serve several roles (prefix encoder for cross-attention decoders;
prefix + suffix runner for the decoder-only path).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Self, override

import numpy as np
import torch
from torch import Tensor

from ..annotations import ConditionField
from .aux_text import (
    IMAGE_KEY_PREFIX,
    AuxField,
    AuxSpec,
    assemble_suffix,
    camera_prompt_order,
    generate_text,
    subgoal_text,
)
from .codecs import ActionCodec
from .image_augment import augment_image


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


@dataclass(frozen=True, slots=True)
class MemoryStream:
    """One exported stream's K/V.

    Shapes: key/value [B, kv_heads, P, head_dim]."""

    key: Tensor
    value: Tensor


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


# Action decoders are plain nn.Modules; each family class composes the
# concrete decoder(s) it was built with, not an ABC: the decoders'
# signatures legitimately differ (the flow decoder's forward is a
# velocity field with solver knobs; the ar_backbone decoder runs against
# the backbone itself), and every consumer goes through the family's
# trait surface. Shared conventions: training objectives are
# module-level functions beside each decoder (``flow_matching_loss``,
# ``ar_backbone_loss``) consuming NORMALIZED targets, and decoders
# operate in normalized space end to end — they never read batch stats
# or choose a table. The FAMILY owns the raw↔normalized boundary (its
# per-sample stats policy, or its checkpoint-recorded quantile table —
# the merged-table families' ``action_quantiles``) and wraps the
# decode into its typed prediction struct (bijou.vla). The one nuance:
# the discrete suffix decode emits raw chunks because detokenization is
# one fused map — its q01/q99 pair is an explicit caller argument, so
# the table CHOICE still lives with the family.

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
    # With probability p per camera frame, apply the sim2real
    # photometric recipe (bijou.modelling.image_augment: brightness/contrast/
    # saturation/hue/gamma jitter, sensor noise, slight defocus, JPEG
    # artifacts, small random crop/translate) at collation — the
    # pi0/OpenVLA-class appearance-shift regularizer. Train-time only:
    # the probe clone runs augment-0 and the eval-side construction
    # sites leave the default, so scored/served frames are NEVER
    # augmented. Default 0.0 keeps every existing path byte-identical —
    # p=0 draws nothing from the RNG and passes tensors through by
    # identity (no clone/clamp/dtype round-trip; the molmoact2 uint8
    # truncation downstream would amplify any float epsilon).
    image_augment: float = 0.0
    # The molmoact2-format state scheme (§8.13 decision 6): when set
    # ([state_dim] fp32 each), PromptInputs.state is q01/q99-CLAMP
    # normalized with this ONE merged table — their semantics — instead
    # of the per-sample mean/std stats. None keeps every existing path
    # byte-identical. Both-or-neither, checked at construction.
    state_q01: Tensor | None = None
    state_q99: Tensor | None = None
    # The molmoact2-format ACTION table: when set ([action_dim] fp32
    # each), CE action targets tokenize under this ONE merged table —
    # the shared-table training convention, sourced from the family's
    # quantile table so encode can never drift from the decode/clamp
    # side (which the family applies itself; batch stats never drive a
    # decode). Collate-time only, hence codec-required; None keeps
    # every existing path byte-identical (Gemma/Molmo2 AR tokenize with
    # per-sample dataset quantiles). Both-or-neither.
    action_q01: Tensor | None = None
    action_q99: Tensor | None = None
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
        if not 0.0 <= self.image_augment <= 1.0:
            raise ValueError(
                f"image augment {self.image_augment} outside [0, 1]",
            )
        if (self.state_q01 is None) != (self.state_q99 is None):
            raise ValueError(
                "state_q01/state_q99 travel together (the merged clamp "
                "table) — got one without the other",
            )
        if (self.action_q01 is None) != (self.action_q99 is None):
            raise ValueError(
                "action_q01/action_q99 travel together (the merged action "
                "table) — got one without the other",
            )
        if self.action_q01 is not None and self.action_codec is None:
            raise ValueError(
                "a merged action table without an action codec has "
                "nothing to tokenize — decode-side tables are the "
                "family's quantile table, never collator state",
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

    def _augment_image(self, image: Tensor) -> Tensor:
        """One camera frame through the aug gate: identity (zero RNG)
        when off, per-frame Bernoulli(p) then the bijou.modelling.image_augment
        recipe when on. Rebuilds, never mutates — dataloader items are
        shared with the raw batch surfaces."""
        if self.image_augment <= 0.0:
            return image
        if float(torch.rand((), generator=self._rng())) >= self.image_augment:
            return image
        return augment_image(image, self._rng())

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
            if self.action_q01 is not None and self.action_q99 is not None:
                # The merged-table override (molmoact2 ar/joint): every
                # sample tokenizes under the ONE table the run trains
                # with — per-item quantiles deliberately unused.
                sequences.append(
                    codec.encode(
                        item["action"].numpy(),
                        self.action_q01.numpy(),
                        self.action_q99.numpy(),
                    ),
                )
                continue
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
            if self.state_q01 is not None and self.state_q99 is not None:
                # The molmoact2-format scheme: ONE merged table, clamp
                # to [-1, 1] (their normalizer; the encoder mode only
                # bins the result into discrete state tokens).
                denom = self.state_q99 - self.state_q01
                denom = torch.where(denom == 0, torch.full_like(denom, 1e-8), denom)
                normalized_state = (
                    2.0 * (state.to(torch.float32) - self.state_q01) / denom - 1.0
                ).clamp(-1.0, 1.0)
            else:
                normalized_state = (state - item["state_mean"]) / item["state_std"]
            samples.append(
                PromptInputs(
                    instruction=self._instruction(item),
                    condition_text=condition_text,
                    cameras=tuple(
                        CameraFrame(
                            name=(name := key.removeprefix(_IMAGE_KEY_PREFIX)),
                            kind=self._camera_kind(item, name),
                            image=self._augment_image(item[key]),
                        )
                        for key in self.cameras_of(item)
                    ),
                    state=normalized_state,
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


@dataclass(frozen=True)
class ARSampling:
    """Action-block temperature sampling for ONE decode call: one CPU
    RNG per batch row, each consuming one Gumbel vector per decode
    step. Only the action block samples — value lines stay greedy (a
    value read is a classification, not a distribution read). CPU-side
    noise keeps sampled ids identical regardless of device, mirroring
    eval's CPU-seeded flow noise convention; per-row streams make each
    row's ids independent of batch composition (the logits themselves
    carry sharding.py's bf16 batch-shape caveat). Callers key the RNGs
    per (frame identity, draw) — eval's ``stable_sample_rng``."""

    temperature: float
    rngs: tuple[np.random.Generator, ...]

    def __post_init__(self) -> None:
        if not self.temperature > 0:
            raise ValueError(
                f"sampling temperature must be > 0, got {self.temperature} "
                "(greedy is sampling=None, not a temperature limit)",
            )


@dataclass(frozen=True, slots=True)
class ValueCandidate:
    """One text-only decode of a single free-text aux value line (a
    subgoal-draws candidate): the parsed value (the ``_parse_aux`` stripped
    convention, so conditioning on it is byte-compatible with the
    self-subgoal probe's
    ``generated_subgoal``) plus the per-step distribution stats that
    make the frozen scorers exactly recomputable offline
    (``bijou.eval.subgoal_scoring``). Per decode step — every step the
    row was active, including the natural terminator step, EXCLUDING a
    budget-forced terminator (that step's distribution chose something
    else; ``truncated`` records the force): ``chosen_logprob`` = the
    emitted id's log-prob and ``mean_logprob`` = the mean log-prob over
    all ``allowed_vocab`` legal text ids, both under the float32
    log-softmax of the text-masked value logits the decode itself
    chose/sampled from. An empty candidate records exactly its
    terminator step, so stats are never empty."""

    text: str
    truncated: bool
    chosen_logprob: tuple[float, ...]
    mean_logprob: tuple[float, ...]
    allowed_vocab: int


@dataclass(frozen=True, slots=True)
class ActionCaptureStep:
    """One ACTION-phase decode step's scoring surface (mcselect —
    the masked-contrast scorer's conditional side, collected during the
    decode rather than re-forwarded): the pre-mask BLOCK logits the
    step chose from, the grammar mask it applied, which rows were still
    decoding (remaining symbol budget > 0 at step start — exactly the
    rows whose emitted id this step is real), and the emitted backbone
    ids. Rows with ``active`` False emit PAD by construction and their
    columns are meaningless."""

    block_logits: Tensor  # [B, vocab_total] float32, PRE-mask
    allowed: Tensor  # [B, vocab_total] bool — the applied grammar mask
    active: Tensor  # [B] bool — remaining > 0 at step start
    chosen: Tensor  # [B] long — emitted backbone ids
