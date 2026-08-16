"""The VLA trait lattice — the model contract train/eval/rollout
program against (docs/vla-architecture.md is the plan that introduces
it; docs/architecture.md subsumes it as phases land).

A :class:`VLA` is a complete vision-language-action model: it owns its
trunk, prompt side, and action/text decoders, and how those are
assembled is entirely the implementation's business. The base trait is
what EVERY consumer needs (collate, train, predict, route parameters,
persist); the sub-traits — :class:`ARVLA`, :class:`FlowVLA`,
:class:`NarratingVLA` — are capabilities a family declares by
inheritance, so "this model has a discrete action decoder" is a
type-level fact: consumers state requirements in signatures
(``def replay[I](model: ARVLA[I])``) and the wrong family is a type
error, not a runtime raise.

All traits are STATELESS (no fields, no ``__init__``) — pure interface
plus contracts. Families live in ``bijou/models/``, one class per
(trunk, trained-surface set); shared machinery is composed modules and
free functions, never intermediate base classes.

Pairing contract: batches passed to :meth:`VLA.forward` and the
predict methods come from THIS model's :meth:`VLA.collator` — the type
parameter ``I`` names that coupling, and loading is the single
boundary where it erases to ``VLA[Any]`` (generic consumer functions
restore precision).

Import DAG: ``models/*`` → ``vla`` → ``modelling/*`` → ``fast``.
Nothing in ``modelling`` imports this module.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self, override

import torch
from torch import Tensor, nn

from .modelling.aux_text import AuxField, AuxGeneration
from .modelling.interface import (
    ActionCaptureStep,
    ARSampling,
    BatchInputs,
    CollatedBatch,
    InputsCollator,
    SamplingMethod,
    ValueCandidate,
)

# ---------------------------------------------------------------------------
# Identity


class VLAFamily(Enum):
    """The closed set of model families — loading's registry keys and the
    value recorded/printed for provenance (rollout banner, eval reports,
    checkpoint metadata). One member per (trunk, trained-surface set);
    the surface set is also visible in the family class's trait bases,
    and the two must agree."""

    GEMMA_FLOW = "gemma_flow"
    GEMMA_AR = "gemma_ar"
    MOLMO2_AR = "molmo2_ar"
    MOLMOACT2_FLOW = "molmoact2_flow"
    MOLMOACT2_AR = "molmoact2_ar"
    MOLMOACT2_JOINT = "molmoact2_joint"


@dataclass(frozen=True, slots=True)
class VLASpec:
    """The model's identity card — what rollout prints and eval records.
    Derived by each family from its configs (a VIEW, never independent
    state)."""

    family: VLAFamily
    chunk_size: int
    action_dim: int


# ---------------------------------------------------------------------------
# Training currency


@dataclass(frozen=True, slots=True)
class Loss:
    """A loss in transportable form: the unnormalized sum over
    contributing elements plus the count that normalizes it — the
    DDP-correct currency: ranks all-reduce ``sum`` and ``count``
    separately and divide AFTER, so uneven per-rank counts (aux text,
    holdout rows) cannot skew the mean. The scalar loss is always the
    quotient — there is deliberately no ``.value`` shortcut.

    Shapes:
      - ``sum``: [] (0-d) — graph-connected on the training path
      - ``count``: [] (0-d) — detached; elements contributing to sum
    """

    sum: Tensor
    count: Tensor


@dataclass(frozen=True, slots=True)
class LossReport:
    """One batch's training objective, decomposed for logging.

    ``objective`` is the per-rank graph scalar built with the
    ALL-REDUCED counts the loop passed to :meth:`VLA.forward` — its DDP
    gradient average equals the gradient of the global objective even
    under uneven per-rank counts. ``components`` are the chart series,
    keyed by the family's own component names ("action_flow", "action_ar", "narration") — a
    dict by design: the key set is family-dynamic but run-constant, and
    equals :meth:`VLA.loss_counts`' key set (the loop enforces this).
    Component sums are graph-connected (they are the objective's
    addends); logging detaches.

    Shapes:
      - ``objective``: [] (0-d) — graph-connected
    """

    objective: Tensor
    components: dict[str, Loss]


# ---------------------------------------------------------------------------
# Prediction currency (one struct per trait method — a flow prediction
# with generations, or an AR one with a noise draw, cannot be built)


@dataclass(frozen=True, slots=True)
class ARPrediction:
    """A discrete-decoder decode's product.

    Shapes:
      - ``actions``: [B, chunk, action_dim] — RAW action units (mirrors
        ``CollatedBatch.actions``, the ground truth it is scored
        against)
    """

    actions: Tensor


@dataclass(frozen=True, slots=True)
class FlowPrediction:
    """A flow integration's product. ``noise`` is ALWAYS the initial
    draw the solver actually integrated (supplied or drawn) — paired
    re-decodes must reuse it, or sampling variance floors any
    conditioning-sensitivity signal.

    Shapes:
      - ``actions``: [B, chunk, action_dim] — RAW action units
      - ``noise``: [B, chunk, action_dim] — normalized units
    """

    actions: Tensor
    noise: Tensor


@dataclass(frozen=True, slots=True)
class NarratedPrediction:
    """A narrated pass's product: actions plus one
    :class:`~bijou.modelling.aux_text.AuxGeneration` per batch row (raw
    text is the report ground truth; a lenient-parse failure is a None
    field inside the generation, never a missing row).

    Shapes:
      - ``actions``: [B, chunk, action_dim] — RAW action units
    """

    actions: Tensor
    generations: list[AuxGeneration]


# ---------------------------------------------------------------------------
# The traits


class VLA[I: BatchInputs](nn.Module, abc.ABC):
    """A complete vision-language-action model (module docstring).

    ``nn.Module`` because implementations must BE modules (DDP wrap,
    optimizer, device moves, ``state_dict``); abstract because the
    trait owns no state and no algorithm — families implement every
    method.

    Contracts every implementation honors:

    - batches come from this model's :meth:`collator` (the ``I``
      pairing); behavior on foreign batches is undefined by type
      design, not defensively checked;
    - :meth:`forward` owns its OWN precision policy (autocast regions,
      fp32 seams, loss-term ordering) — callers apply no ambient
      autocast;
    - the predict surfaces run under ``torch.no_grad``
      (implementations decorate) and return RAW action units via the
      batch's per-dataset quantile stats;
    - models are ALWAYS constructed with an objective — eval/rollout
      construction passes the checkpoint's recorded one, so a loaded
      model can always compute its own training loss.
    """

    @property
    @abc.abstractmethod
    def spec(self) -> VLASpec:
        """Identity card, derived from the family's configs."""

    @abc.abstractmethod
    def collator(self) -> InputsCollator[I]:
        """The sole producer of this model's batches. Pickleable — it
        crosses into spawned dataloader workers."""

    @abc.abstractmethod
    def loss_counts(self, batch: CollatedBatch[I]) -> dict[str, Tensor]:
        """Per-component element counts for this batch (detached, 0-d).
        The loop all-reduces these BEFORE :meth:`forward` — global
        normalizers are what make the DDP gradient average exact under
        uneven per-rank counts. Key set is run-constant and equals the
        report's ``components`` keys."""

    @override
    @abc.abstractmethod
    def forward(
        self,
        batch: CollatedBatch[I],
        *,
        counts: dict[str, Tensor],
    ) -> LossReport:
        """One batch's training objective (the DDP entry point).
        ``counts`` are the all-reduced returns of :meth:`loss_counts`
        (passed through un-reduced on a single process). Chunked
        backward is supported by construction: calling forward on
        micro-batch slices with the SAME global counts yields objective
        addends whose sum equals the full-batch objective."""

    @abc.abstractmethod
    def predict(self, batch: CollatedBatch[I]) -> Tensor:
        """Actions at the model's RECORDED serving operating point
        (checkpoint metadata — solver and step count for flow, greedy
        decode for AR), taking no knobs so every family can answer and
        cross-family paired evals compare like with like. Knobbed
        inference lives on the capability traits.

        Shapes:
          - returns: [B, chunk, action_dim] — RAW action units
        """

    @abc.abstractmethod
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Named parameter groups — the LR-routing vocabulary
        ("decoder", "backbone_text", "backbone_vision") — as the
        STRUCTURAL offer under this model's construction: a parameter
        appears iff the objective's graph can deliver it a gradient
        (insulation empties a flow-only run's backbone groups;
        construction-frozen tensors never appear). Which offered groups
        actually train, and at what LR, is optimizer policy — train.py
        cross-checks the flags against this offer and errors on
        contradictions. Groups are disjoint; DDP's exactness contract
        rides on their union minus policy freezes."""

    @abc.abstractmethod
    def output_head_parameters(self) -> list[nn.Parameter]:
        """The trainable OUTPUT-projection parameters — the subset that
        keeps standard AdamW decay under ``--optimizer adamc`` while
        every hidden matrix gets the corrected decay. May be empty
        (adamc then degenerates to uniform corrected decay); a family
        with a frozen output head answers with the empty list,
        explicitly."""

    @abc.abstractmethod
    def checkpoint_components(self) -> dict[str, nn.Module]:
        """Component name → module subtree, the family's declaration of
        its checkpoint sections. The loading toolkit maps each entry to
        ``<name>.safetensors``; the backbone is NOT listed here (the
        toolkit handles it via the hard-link rule)."""

    @classmethod
    @abc.abstractmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Self:
        """Reconstruct the trained model from a checkpoint directory
        with no reference to any other directory (self-containment is
        the format's invariant)."""


class ARVLA[I: BatchInputs](VLA[I], abc.ABC):
    """The discrete-action-decoder capability: this model emits its
    action chunk as tokens, so the block can be teacher-forced,
    sampled, and captured — the GRPO and chunk-NLL instrument surface.
    Nothing about text: narration is :class:`NarratingVLA`."""

    @abc.abstractmethod
    def predict_ar(
        self,
        batch: CollatedBatch[I],
        *,
        sampling: ARSampling | None = None,
        capture: list[ActionCaptureStep] | None = None,
    ) -> ARPrediction:
        """The action-block decode, never any text: prompt encode, the
        family's opener, BOA forced (its identity is scaffold, not a
        decision), then the grammar-masked block decode — each step
        masks ids whose symbol expansion exceeds the remaining budget,
        PAD legal only at budget zero.

        ``sampling=None`` decodes greedily (deterministic per frame —
        the deployment and paired-eval path); an :class:`ARSampling`
        switches the ACTION block to per-row temperature sampling.
        ``capture``, when given, receives one
        :class:`ActionCaptureStep` per decode step, taken from the very
        logits the decode chose from — no re-forward, no numeric drift
        vs the executed decode."""

    @abc.abstractmethod
    def teacher_forced_block_logits(
        self,
        batch: CollatedBatch[I],
        action_ids: Tensor,
    ) -> Tensor:
        """Block logits for GIVEN action ids under teacher forcing —
        one prefill plus one suffix forward, no decode loop (the
        chunk-NLL metric and the GRPO replay-logprob surface).
        Deterministic per frame; batch composition moves bf16 reduction
        order, so cross-run comparisons pin batch shape.

        Shapes:
          - ``action_ids``: [B, S] long — block-relative ids
          - returns: [B, S, vocab_total] float32 — position j scores
            ``action_ids[:, j]``
        """


class FlowVLA[I: BatchInputs](VLA[I], abc.ABC):
    """The flow-matching capability: actions come from integrating a
    learned velocity field from Gaussian noise."""

    @abc.abstractmethod
    def predict_flow(
        self,
        batch: CollatedBatch[I],
        *,
        num_steps: int,
        method: SamplingMethod,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> FlowPrediction:
        """Integrate the velocity field with an EXPLICIT operating
        point — the knobs are required here; defaults live only in
        :meth:`VLA.predict`. ``noise`` supplies the initial draw
        (paired re-decodes reuse a prior prediction's); otherwise it is
        drawn from ``generator`` on CPU (device-independent draws,
        eval's seeding convention).

        Shapes:
          - ``noise``: [B, chunk, action_dim] — normalized units
        """


class NarratingVLA[I: BatchInputs](VLA[I], abc.ABC):
    """The text-surface capability: the model was trained to emit aux
    value lines (subgoal, holding, progress, event, visible) and can
    generate them at inference — beside whichever action decoder the
    family has. Orthogonal to :class:`ARVLA`: a format-5 AR family
    narrates inside its one AR pass; a narrated-flow family decodes a
    text suffix beside flow-sampled actions."""

    @abc.abstractmethod
    def predict_narrated(
        self,
        batch: CollatedBatch[I],
        *,
        generate: tuple[AuxField, ...],
    ) -> NarratedPrediction:
        """The narrated pass: requested fields decoded as text, then
        actions. ``generate`` must be non-empty (action-only inference
        is :meth:`VLA.predict` / :meth:`ARVLA.predict_ar`), a subset of
        the checkpoint's TRAINED fields in template order (AuxField
        declaration order), and equal to the request the batch was
        collated with — a mismatch sits off the conditioning and is a
        loud error.

        Per field, in order: greedy text decode under the field's value
        budget until the ``\\n`` terminator — budget exhaustion forces
        the terminator and counts a loud fallback; newline-carrier ids
        are banned mid-value; HOLDING's first token is constrained to
        its candidate set."""

    @abc.abstractmethod
    def predict_with_value_candidates(
        self,
        batch: CollatedBatch[I],
        *,
        field: AuxField,
        generate: tuple[AuxField, ...],
        draws: int,
        sampling_for_draw: Callable[[int], ARSampling],
    ) -> tuple[NarratedPrediction, list[list[ValueCandidate]]]:
        """The subgoal-draws instrument: ONE prefill, then (a) the full
        narrated pass, op-identical to :meth:`predict_narrated` so the
        draws=0 limit stays bit-exact against it, and (b) ``draws + 1``
        text-only decodes of ``field`` against the restored prefix
        cache — candidate 0 greedy (its per-step stats make the greedy
        candidate scorable), candidates 1..draws temperature-sampled
        under ``sampling_for_draw(draw)``. Free-text fields only (typed
        check); the greedy candidate's text must equal the full pass's
        parsed value — a mismatch is a broken instrument and exits
        loudly. Returns (full-pass prediction, per-row candidates)."""
